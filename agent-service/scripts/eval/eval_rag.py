"""RAG 检索 5 组方案实测消融评测脚本（基于真实 Milvus + Postgres 45 条黄金问答集）。

用法：在 agent-service 目录执行
    uv run python scripts/eval/eval_rag.py
"""

import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.logging import get_logger
from app.infra.database import close_database
from app.rag import close_rag, get_retriever
from app.rag.models import ParentChunk
from app.rag.retriever import RetrievalStrategy

logger = get_logger(__name__)

CASES_PATH = Path(__file__).resolve().parent / "rag_golden_cases.json"
REPORTS_DIR = PROJECT_ROOT / "reports"

STRATEGIES: list[tuple[str, RetrievalStrategy]] = [
    ("1. Dense Only (向量单路)", "dense_only"),
    ("2. Hybrid (Dense+BM25 RRF)", "hybrid_no_rerank"),
    ("3. Hybrid + gte-rerank-v2 (单次查询)", "hybrid_rerank_single"),
    ("4. HyDE + Hybrid + Rerank (竞品)", "hyde_hybrid_rerank"),
    ("5. 术语改写 + 并行定额 + 去重 + Rerank (最终方案)", "full_enhanced"),
]


def _chunk_matches_target(chunk: ParentChunk, target: dict[str, Any]) -> bool:
    clause_kw = target["clause_keyword"]
    if clause_kw not in chunk.clause_name:
        return False
    section_str = " > ".join(chunk.section_path or [])
    return any(sec_kw in section_str for sec_kw in target["section_keywords"])


def _compute_duplicate_rate(chunks: list[ParentChunk]) -> float:
    if not chunks:
        return 0.0
    unique_keys = {(c.clause_name, tuple(c.section_path or ())) for c in chunks}
    return (len(chunks) - len(unique_keys)) / len(chunks)


def _compute_case_metrics(
    case: dict[str, Any],
    chunks: list[ParentChunk],
) -> dict[str, float | None]:
    targets = case["expected_targets"]
    top3 = chunks[:3]
    top5 = chunks[:5]

    hit3 = sum(1 for t in targets if any(_chunk_matches_target(c, t) for c in top3))
    hit5 = sum(1 for t in targets if any(_chunk_matches_target(c, t) for c in top5))
    recall3 = hit3 / len(targets)
    recall5 = hit5 / len(targets)

    mrr = 0.0
    for rank, chunk in enumerate(top5, start=1):
        if any(_chunk_matches_target(chunk, t) for t in targets):
            mrr = 1.0 / rank
            break

    compare_complete: float | None = None
    if case.get("case_type") == "compare":
        compare_complete = 1.0 if hit5 == len(targets) else 0.0

    dup_rate = _compute_duplicate_rate(top5)
    return {
        "recall3": recall3,
        "recall5": recall5,
        "mrr": mrr,
        "compare_complete": compare_complete,
        "dup_rate": dup_rate,
    }


async def evaluate_strategy(
    display_name: str,
    strategy: RetrievalStrategy,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    rag_retriever = get_retriever()
    sem = asyncio.Semaphore(3)
    recall3_list: list[float] = []
    recall5_list: list[float] = []
    mrr_list: list[float] = []
    compare_list: list[float] = []
    dup_list: list[float] = []
    latencies: list[float] = []

    async def _run_one(case: dict[str, Any]) -> tuple[dict[str, Any], list[ParentChunk], float]:
        async with sem:
            t0 = time.perf_counter()
            chunks = await rag_retriever.retrieve_by_strategy(
                query=case["query"],
                strategy=strategy,
                k=5,
                product_ids=case.get("product_ids"),
                fetch_k=15,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return case, chunks, elapsed_ms

    outputs = await asyncio.gather(*(_run_one(case) for case in cases))
    for case, chunks, elapsed_ms in outputs:
        metrics = _compute_case_metrics(case, chunks)
        recall3_list.append(float(metrics["recall3"] or 0.0))
        recall5_list.append(float(metrics["recall5"] or 0.0))
        mrr_list.append(float(metrics["mrr"] or 0.0))
        dup_list.append(float(metrics["dup_rate"] or 0.0))
        if metrics["compare_complete"] is not None:
            compare_list.append(float(metrics["compare_complete"]))
        latencies.append(elapsed_ms)

    sorted_lat = sorted(latencies)
    p95_idx = min(len(sorted_lat) - 1, int(0.95 * len(sorted_lat)))

    summary = {
        "strategy": display_name,
        "cases": len(cases),
        "recall_at_3": round(statistics.mean(recall3_list) * 100, 2),
        "recall_at_5": round(statistics.mean(recall5_list) * 100, 2),
        "mrr": round(statistics.mean(mrr_list), 4),
        "compare_completeness": round(statistics.mean(compare_list) * 100, 2) if compare_list else 0.0,
        "duplicate_rate": round(statistics.mean(dup_list) * 100, 2),
        "p95_latency_ms": round(sorted_lat[p95_idx], 2),
    }
    logger.info("RAG策略评测完成", **summary)
    return summary


def render_markdown_report(results: list[dict[str, Any]]) -> str:
    lines = [
        "# RAG 检索增强 5 方案实测消融报告",
        "",
        f"- 黄金评测集：`{results[0]['cases']}` 条真实条款问答（25 条单品口语 + 10 条跨产品对比 + 10 条全局无 ID 检索）",
        "- 知识库规模：12 款保险产品真实合同条款，PG `1381` 个父块 / Milvus `1779` 个子块",
        "",
        "| 对比方案 | Recall@3 (%) | Recall@5 (%) | MRR | 跨产品对比完整率 (%) | Top-5 重复块率 (%) | P95 耗时 (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for row in results:
        lines.append(
            f"| {row['strategy']} | {row['recall_at_3']:.2f} | {row['recall_at_5']:.2f} | "
            f"{row['mrr']:.4f} | {row['compare_completeness']:.2f} | "
            f"{row['duplicate_rate']:.2f} | {row['p95_latency_ms']:.2f} |"
        )
    return "\n".join(lines) + "\n"


async def main() -> None:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []
    try:
        for display_name, strategy in STRATEGIES:
            res = await evaluate_strategy(display_name, strategy, cases)
            results.append(res)
    finally:
        close_rag()
        await close_database()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    md_report = render_markdown_report(results)
    (REPORTS_DIR / "rag_eval.md").write_text(md_report, encoding="utf-8")
    (REPORTS_DIR / "rag_eval.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n" + md_report)


if __name__ == "__main__":
    asyncio.run(main())
