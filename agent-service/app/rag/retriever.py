import asyncio
from collections.abc import Callable
import math
from typing import Any, Literal
from uuid import uuid5, NAMESPACE_DNS

from langchain_milvus import Milvus
from pymilvus import Function, FunctionType
from sqlalchemy import select

from app.core.logging import get_logger
from app.infra.database import AsyncSessionFactory
from app.modules.product.catalog_data import load_builtin_catalog
from app.modules.product.models import Product, ProductFeature
from app.rag.models import ParentChunk
from app.rag.query_rewriter import InsuranceQueryRewriter
from app.rag.repository import ParentChunkRepository

logger = get_logger(__name__)

RetrievalStrategy = Literal[
    "dense_only",
    "hybrid_no_rerank",
    "hybrid_rerank_single",
    "hyde_hybrid_rerank",
    "full_enhanced",
]


def create_rrf_ranker(k: int = 60) -> Function:
    """Milvus 内置 RRF 融合重排（不调用语义模型重排，用于 Hybrid 基线对照）"""
    return Function(
        name="rrf_ranker",
        input_field_names=[],
        function_type=FunctionType.RERANK,
        params={"reranker": "rrf", "k": k},
    )


def create_reranker(query: str) -> Function:
    """Milvus 的 reranker function，采用阿里百炼 gte-rerank-v2 重排模型"""
    return Function(
        name="dashscope_semantic_ranker",
        input_field_names=["text"],
        function_type=FunctionType.RERANK,
        params={
            "reranker": "model",
            "provider": "ali",
            "model_name": "gte-rerank-v2",
            "queries": [query],
            "max_client_batch_size": 5,
        },
    )


def deduplicate_parent_chunks(chunks: list[ParentChunk]) -> list[ParentChunk]:
    """按 (clause_name, section_path) 保持召回顺序去重，解决共享 PDF 条款重复召回问题。"""
    seen: set[tuple[str, tuple[str, ...]]] = set()
    unique: list[ParentChunk] = []
    for chunk in chunks:
        key = (chunk.clause_name, tuple(chunk.section_path or ()))
        if key in seen:
            continue
        seen.add(key)
        unique.append(chunk)
    return unique


def merge_round_robin(
    grouped_chunks: list[list[ParentChunk]],
    k: int,
    *,
    deduplicate: bool = True,
) -> list[ParentChunk]:
    """跨产品并行检索后按轮询交叉合并，防止单一产品霸榜并可选去重。"""
    merged: list[ParentChunk] = []
    seen_keys: set[tuple[Any, ...]] = set()
    max_len = max((len(group) for group in grouped_chunks), default=0)

    for idx in range(max_len):
        for group in grouped_chunks:
            if idx >= len(group):
                continue
            chunk = group[idx]
            dedup_key = (
                (chunk.clause_name, tuple(chunk.section_path or ()))
                if deduplicate
                else (chunk.product_id, str(chunk.id))
            )
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            merged.append(chunk)
            if len(merged) >= k:
                return merged
    return merged


def _build_product_expr(product_ids: list[int] | None) -> str | None:
    if not product_ids:
        return None
    if len(product_ids) == 1:
        return f"product_id == {product_ids[0]}"
    joined = ", ".join(str(pid) for pid in product_ids)
    return f"product_id in [{joined}]"


class ParentChunkRetriever:
    """增强版父子块检索器：支持术语改写、跨产品并行定额对比、共享条款去重及无条款结构化兜底。"""

    def __init__(
        self,
        vector_store: Milvus,
        rewriter: InsuranceQueryRewriter | None = None,
        session_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.rewriter = rewriter or InsuranceQueryRewriter()
        self.session_factory = session_factory or AsyncSessionFactory
        logger.info("RAG Retriever 初始化完成✅️")

    async def _fetch_parents_in_order(self, parent_ids: list[str]) -> list[ParentChunk]:
        unique_ids = list(dict.fromkeys(str(pid) for pid in parent_ids if pid))
        if not unique_ids:
            return []

        async with self.session_factory() as session:
            repository = ParentChunkRepository(session)
            parent_chunks = await repository.list_by_ids(unique_ids)

        parent_chunk_map = {str(parent.id): parent for parent in parent_chunks}
        return [parent_chunk_map[pid] for pid in unique_ids if pid in parent_chunk_map]

    async def _build_structured_fallback_chunk(self, product_id: int) -> ParentChunk | None:
        """无 PDF 条款的扩展产品（如 id 13-50）优雅降级返回结构化保障摘要块。"""
        product_name: str | None = None
        summary_lines: list[str] = []

        try:
            async with self.session_factory() as session:
                row = (
                    await session.execute(
                        select(Product, ProductFeature)
                        .outerjoin(ProductFeature, ProductFeature.product_id == Product.id)
                        .where(Product.id == product_id)
                    )
                ).first()
                if row is not None:
                    product, feature = row[0], row[1]
                    product_name = product.name
                    summary_lines.extend(
                        [
                            f"产品名称：{product.name}（承保公司：{product.insurer}，险种：{product.category}）",
                            f"适用人群：{product.target_group or '详见产品说明'}",
                            f"参考年保费：¥{product.min_premium or '-'} 起",
                            f"产品简介：{product.description or '无'}",
                            f"核心亮点：{'、'.join(product.highlights or []) or '无'}",
                        ]
                    )
                    if feature is not None:
                        renewal = (
                            f"保证续保{feature.guarantee_renewal_years}年"
                            if feature.guarantee_renewal_years
                            else "非多年保证续保"
                        )
                        summary_lines.extend(
                            [
                                f"承保年龄：{feature.min_age}-{feature.max_age}周岁，职业上限：1-{feature.max_occupation_class}类",
                                f"等待期：{feature.waiting_period_days}天，默认免赔额：¥{feature.deductible_default}",
                                f"续保与标签：{renewal} | {'、'.join(feature.coverage_tags or [])}",
                            ]
                        )
        except Exception as exc:
            logger.debug("数据库兜底查询失败，尝试读取内置产品目录", product_id=product_id, error=str(exc))

        if not summary_lines:
            builtin_map = {item.id: item for item in load_builtin_catalog()}
            cand = builtin_map.get(product_id)
            if cand is None:
                return None
            product_name = cand.name
            summary_lines.append(cand.to_semantic_text())

        summary_lines.append("注：该产品为结构化目录扩展产品，暂未录入完整 PDF 合同条款原文。")
        return ParentChunk(
            id=uuid5(NAMESPACE_DNS, f"structured-fallback-{product_id}"),
            product_id=product_id,
            clause_name=f"结构化产品说明（{product_name}）",
            section_path=[product_name or f"产品{product_id}", "结构化保障摘要（暂无完整PDF条款原文）"],
            content="\n".join(summary_lines),
        )

    async def _retrieve_single(
        self,
        query: str,
        *,
        k: int,
        product_id: int | None,
        use_rewrite: bool,
        deduplicate: bool,
        fetch_k: int,
    ) -> list[ParentChunk]:
        search_query = self.rewriter.rewrite_by_dict(query) if use_rewrite else query
        expr = f"product_id == {product_id}" if product_id is not None else None
        # 全局检索或去重模式适度放大子块召回量，防止共享条款去重后不足 k 条
        child_k = max(k * 3, 12) if (deduplicate and product_id is None) else k
        effective_fetch_k = max(fetch_k, child_k * 2)

        child_chunks = await self.vector_store.asimilarity_search(
            query=search_query,
            k=child_k,
            fetch_k=effective_fetch_k,
            expr=expr,
            reranker=create_reranker(search_query),
        )
        parent_ids = [
            str(child.metadata["parent_id"])
            for child in child_chunks
            if child.metadata and "parent_id" in child.metadata
        ]
        parents = await self._fetch_parents_in_order(parent_ids)

        if deduplicate:
            parents = deduplicate_parent_chunks(parents)

        if not parents and product_id is not None:
            fallback = await self._build_structured_fallback_chunk(product_id)
            if fallback is not None:
                return [fallback]

        return parents[:k]

    async def retrieve(
        self,
        query: str,
        k: int = 3,
        product_id: int | None = None,
        *,
        product_ids: list[int] | None = None,
        use_rewrite: bool = True,
        deduplicate: bool = True,
        fetch_k: int = 15,
    ) -> list[ParentChunk]:
        """检索父知识块：向后兼容单 product_id，并支持多产品并行定额对比与全局去重检索。"""
        if product_ids is not None:
            resolved_ids = list(dict.fromkeys(int(pid) for pid in product_ids))
        elif product_id is not None:
            resolved_ids = [int(product_id)]
        else:
            resolved_ids = None

        if resolved_ids and len(resolved_ids) > 1:
            per_k = max(2, math.ceil(k / len(resolved_ids)))
            grouped = await asyncio.gather(
                *(
                    self._retrieve_single(
                        query,
                        k=per_k,
                        product_id=pid,
                        use_rewrite=use_rewrite,
                        deduplicate=deduplicate,
                        fetch_k=fetch_k,
                    )
                    for pid in resolved_ids
                )
            )
            return merge_round_robin(list(grouped), k=k, deduplicate=deduplicate)

        single_pid = resolved_ids[0] if resolved_ids else None
        return await self._retrieve_single(
            query,
            k=k,
            product_id=single_pid,
            use_rewrite=use_rewrite,
            deduplicate=deduplicate,
            fetch_k=fetch_k,
        )

    async def retrieve_by_strategy(
        self,
        query: str,
        strategy: RetrievalStrategy,
        *,
        k: int = 5,
        product_ids: list[int] | None = None,
        fetch_k: int = 15,
    ) -> list[ParentChunk]:
        """策略可控的检索入口，专用于 eval_rag 消融实验公平对比 5 组方案。"""
        resolved_ids = list(dict.fromkeys(int(pid) for pid in product_ids)) if product_ids else None
        expr = _build_product_expr(resolved_ids)

        if strategy == "full_enhanced":
            return await self.retrieve(
                query,
                k=k,
                product_ids=resolved_ids,
                use_rewrite=True,
                deduplicate=True,
                fetch_k=fetch_k,
            )

        if strategy == "dense_only":
            embedder = getattr(self.vector_store, "embedding_func", None) or getattr(
                self.vector_store, "embedding_function", None
            )
            if embedder is None:
                raise RuntimeError("vector_store 未配置 embedding_func")
            if hasattr(embedder, "aembed_query"):
                query_vec = await embedder.aembed_query(query)
            else:
                query_vec = await asyncio.to_thread(embedder.embed_query, query)
            raw_res = await asyncio.to_thread(
                self.vector_store.client.search,
                collection_name=self.vector_store.collection_name,
                data=[query_vec],
                anns_field="dense",
                limit=max(k * 2, 10),
                filter=expr or "",
                output_fields=["parent_id", "product_id", "text"],
            )
            hits = raw_res[0] if raw_res else []
            parent_ids = [
                str((hit.get("entity") or hit).get("parent_id", ""))
                for hit in hits
                if (hit.get("entity") or hit).get("parent_id")
            ]
            parents = await self._fetch_parents_in_order(parent_ids)
            return parents[:k]

        if strategy == "hybrid_no_rerank":
            child_chunks = await self.vector_store.asimilarity_search(
                query=query,
                k=k,
                fetch_k=fetch_k,
                expr=expr,
                reranker=create_rrf_ranker(),
            )
        elif strategy == "hybrid_rerank_single":
            child_chunks = await self.vector_store.asimilarity_search(
                query=query,
                k=k,
                fetch_k=fetch_k,
                expr=expr,
                reranker=create_reranker(query),
            )
        elif strategy == "hyde_hybrid_rerank":
            hyde_text = await self.rewriter.generate_hyde_passage(query)
            child_chunks = await self.vector_store.asimilarity_search(
                query=hyde_text,
                k=k,
                fetch_k=fetch_k,
                expr=expr,
                reranker=create_reranker(query),
            )
        else:
            raise ValueError(f"未知检索策略: {strategy}")

        parent_ids = [
            str(child.metadata["parent_id"])
            for child in child_chunks
            if child.metadata and "parent_id" in child.metadata
        ]
        parents = await self._fetch_parents_in_order(parent_ids)
        return parents[:k]
