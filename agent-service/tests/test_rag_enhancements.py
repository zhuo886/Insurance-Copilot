from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4

from app.rag.models import ParentChunk
from app.rag.query_rewriter import InsuranceQueryRewriter
from app.rag.retriever import (
    ParentChunkRetriever,
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

    # 已有规范词时不应重复拼接
    q3 = rewriter.rewrite_by_dict("这款产品等待期是多少天")
    assert q3 == "这款产品等待期是多少天"


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
    # 轮询交叉合并：4 与 6 各占 2 条，杜绝单产品霸榜
    assert [c.product_id for c in results] == [4, 6, 4, 6]


async def test_virtual_product_without_clauses_triggers_structured_fallback():
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
    chunks = await retriever.retrieve("这款产品保什么", k=3, product_id=13)
    assert len(chunks) == 1
    assert chunks[0].product_id == 13
    assert "平安长相安2号" in chunks[0].clause_name
    assert "结构化" in chunks[0].content


def test_merge_round_robin_caps_at_k():
    a = [_make_parent(1, "c1.pdf", [f"s{i}"], f"a{i}") for i in range(4)]
    b = [_make_parent(2, "c2.pdf", [f"s{i}"], f"b{i}") for i in range(4)]
    merged = merge_round_robin([a, b], k=5)
    assert len(merged) == 5
    assert [x.product_id for x in merged] == [1, 2, 1, 2, 1]
