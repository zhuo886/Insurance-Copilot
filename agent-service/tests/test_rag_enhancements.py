from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.rag.models import ParentChunk
from app.rag.query_rewriter import TERM_SYNONYM_MAP, InsuranceQueryRewriter
from app.rag.retriever import (
    ParentChunkRetriever,
    _build_product_expr,
    deduplicate_parent_chunks,
    merge_round_robin,
)


def _make_parent(pid: int, clause: str, path: list[str], content: str) -> ParentChunk:
    return ParentChunk(
        id=uuid4(),
        product_id=pid,
        clause_name=clause,
        section_path=path,
        content=content,
    )


def test_query_rewriter_expands_colloquial_terms_and_avoids_duplicates():
    rewriter = InsuranceQueryRewriter()

    q1 = rewriter.rewrite_by_dict("买完多久能赔，自费药能报吗")
    assert "等待期" in q1
    assert "外购药品" in q1

    q2 = rewriter.rewrite_by_dict("哪些不赔，怎么理赔需要什么资料")
    assert "责任免除" in q2
    assert "保险金申请" in q2

    q3 = rewriter.rewrite_by_dict("这款产品等待期是多少天")
    assert q3 == "这款产品等待期是多少天"
    assert rewriter.rewrite_by_dict("   ") == ""


@pytest.mark.parametrize("trigger", list(TERM_SYNONYM_MAP.keys()))
def test_query_rewriter_covers_every_synonym_trigger(trigger: str):
    rewriter = InsuranceQueryRewriter()
    out = rewriter.rewrite_by_dict(f"请问{trigger}怎么算")
    for expected in TERM_SYNONYM_MAP[trigger]:
        assert expected in out


async def test_query_rewriter_llm_and_hyde_fallback_on_failure():
    rewriter = InsuranceQueryRewriter()
    assert await rewriter.rewrite("后悔退保", use_llm=False) == "后悔退保 犹豫期 合同解除 现金价值"

    class GoodLLM:
        async def ainvoke(self, _msgs):
            return SimpleNamespace(content="  犹豫期 合同解除 现金价值  ")

    rewriter._llm = GoodLLM()
    assert await rewriter.rewrite("后悔退保", use_llm=True) == "犹豫期 合同解除 现金价值"
    assert "犹豫期" in await rewriter.generate_hyde_passage("犹豫期几天")

    class BrokenLLM:
        async def ainvoke(self, _msgs):
            raise RuntimeError("network error")

    rewriter._llm = BrokenLLM()
    assert await rewriter.rewrite("后悔退保", use_llm=True) == "后悔退保 犹豫期 合同解除 现金价值"
    assert await rewriter.generate_hyde_passage("原始条款问题") == "原始条款问题"


def test_deduplicate_parent_chunks_preserves_order_for_shared_pdf():
    shared_clause = "复星联合优选二号长期住院医疗险(星相守2号)条款.pdf"
    c4_a = _make_parent(4, shared_clause, ["2 提供的保障", "2.1 保险计划"], "个人版保障A")
    c5_a = _make_parent(5, shared_clause, ["2 提供的保障", "2.1 保险计划"], "转保版共享同一条款A")
    c6_a = _make_parent(6, "众安尊享e生2026条款.pdf", ["第六条", "外购药"], "众安外购药")
    c5_b = _make_parent(5, shared_clause, ["2.10 责任免除"], "转保版条款B")

    deduped = deduplicate_parent_chunks([c4_a, c5_a, c6_a, c5_b])
    assert [c.product_id for c in deduped] == [4, 6, 5]
    assert len(deduped) == 3


async def test_parallel_compare_retrieval_prevents_single_product_dominance():
    p4_1 = _make_parent(4, "星相守2号.pdf", ["2.1 保险计划"], "星相守免赔额1万")
    p4_2 = _make_parent(4, "星相守2号.pdf", ["2.8 保险责任"], "星相守外购药不限清单")
    p6_1 = _make_parent(6, "尊享e生2026.pdf", ["第七条 免赔额"], "尊享e生免赔额1万")
    p6_2 = _make_parent(6, "尊享e生2026.pdf", ["第六条 外购药"], "尊享e生外购药清单")
    parent_map = {str(c.id): c for c in (p4_1, p4_2, p6_1, p6_2)}

    class FakeVectorStore:
        async def asimilarity_search(self, query, k, fetch_k=10, expr=None, reranker=None):
            if expr == "product_id == 4":
                return [
                    SimpleNamespace(metadata={"parent_id": str(p4_1.id)}),
                    SimpleNamespace(metadata={"parent_id": str(p4_2.id)}),
                ][:k]
            if expr == "product_id == 6":
                return [
                    SimpleNamespace(metadata={"parent_id": str(p6_1.id)}),
                    SimpleNamespace(metadata={"parent_id": str(p6_2.id)}),
                ][:k]
            return []

    retriever = ParentChunkRetriever(vector_store=FakeVectorStore())  # type: ignore[arg-type]

    async def fake_fetch(parent_ids):
        return [parent_map[pid] for pid in parent_ids if pid in parent_map]

    retriever._fetch_parents_in_order = fake_fetch  # type: ignore[method-assign]

    results = await retriever.retrieve("免赔额和自费药区别", k=4, product_ids=[4, 6])
    assert len(results) == 4
    assert [c.product_id for c in results] == [4, 6, 4, 6]


async def test_missing_clause_returns_empty_without_fabrication():
    class EmptyVectorStore:
        async def asimilarity_search(self, query, k, fetch_k=10, expr=None, reranker=None):
            return []

    @asynccontextmanager
    async def broken_session():
        raise RuntimeError("offline mode")
        yield  # pragma: no cover

    retriever = ParentChunkRetriever(
        vector_store=EmptyVectorStore(),  # type: ignore[arg-type]
        session_factory=broken_session,
    )
    chunks = await retriever.retrieve("这款产品保什么", k=3, product_id=999)
    assert chunks == []


def test_merge_round_robin_caps_and_deduplicate_switch():
    a = [_make_parent(1, "shared.pdf", [f"s{i}"], f"a{i}") for i in range(3)]
    b = [_make_parent(2, "shared.pdf", [f"s{i}"], f"b{i}") for i in range(3)]
    merged_unique = merge_round_robin([a, b], k=5, deduplicate=True)
    assert len(merged_unique) == 3

    merged_all = merge_round_robin([a, b], k=5, deduplicate=False)
    assert [x.product_id for x in merged_all] == [1, 2, 1, 2, 1]


def test_build_product_expr_formats():
    assert _build_product_expr(None) is None
    assert _build_product_expr([]) is None
    assert _build_product_expr([6]) == "product_id == 6"
    assert _build_product_expr([4, 6, 10]) == "product_id in [4, 6, 10]"


async def test_retrieve_by_strategy_routes_dense_hybrid_and_unknown_strategies():
    p4 = _make_parent(4, "星相守2号.pdf", ["2.1 保险计划"], "免赔额1万元")
    parent_map = {str(p4.id): p4}
    calls: list[dict] = []

    class FakeEmbedder:
        async def aembed_query(self, q: str):
            return [0.1, 0.2]

    class FakeClient:
        def search(self, **kwargs):
            calls.append({"dense_search": kwargs})
            return [[{"entity": {"parent_id": str(p4.id)}}]]

    class FakeStore:
        collection_name = "insurance_collection"
        embedding_func = FakeEmbedder()
        client = FakeClient()

        async def asimilarity_search(self, query, k, fetch_k=10, expr=None, reranker=None):
            calls.append({"query": query, "reranker": reranker.name if reranker else None})
            return [SimpleNamespace(metadata={"parent_id": str(p4.id)})]

    retriever = ParentChunkRetriever(vector_store=FakeStore())  # type: ignore[arg-type]

    async def fake_fetch(parent_ids):
        return [parent_map[pid] for pid in parent_ids if pid in parent_map]

    retriever._fetch_parents_in_order = fake_fetch  # type: ignore[method-assign]

    res_dense = await retriever.retrieve_by_strategy("等待期", "dense_only", k=2)
    assert [c.product_id for c in res_dense] == [4]

    res_rrf = await retriever.retrieve_by_strategy("买完多久能赔", "hybrid_no_rerank", k=2)
    assert len(res_rrf) == 1 and calls[-1]["reranker"] == "rrf_ranker"

    res_full = await retriever.retrieve_by_strategy("买完多久能赔", "full_enhanced", k=2)
    assert len(res_full) == 1 and calls[-1]["reranker"] == "dashscope_semantic_ranker"
    assert "等待期" in calls[-1]["query"]

    with pytest.raises(ValueError, match="未知检索策略"):
        await retriever.retrieve_by_strategy("等待期", "nonexistent_strategy")  # type: ignore[arg-type]
