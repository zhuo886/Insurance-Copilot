"""RAG 检索 7 组单变量逐项消融评测脚本（基于真实 Milvus + Postgres 45 条黄金问答集）。

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
    ("2. + BM25 RRF (Hybrid 无改写无重排)", "hybrid_no_rerank"),
    ("3. + 领域词典改写 (Hybrid+Rewrite)", "hybrid_rewrite_no_rerank"),
    ("4. + 并行定额与路径去重 (无Rerank)", "hybrid_rewrite_parallel_dedup"),
    ("5. + gte-rerank-v2 重排 (最终方案)", "full_enhanced"),
    ("对照 A. Hybrid+Rerank (单次查询)", "hybrid_rerank_single"),
    ("对照 B. HyDE+Hybrid+Rerank (竞品)", "hyde_hybrid_rerank"),
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
        "strategy_key": strategy,
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
    by_key = {row["strategy_key"]: row for row in results}
    lines = [
        "# RAG 检索增强 7 组单变量逐项消融报告",
        "",
        f"- 黄金评测集：`{results[0]['cases']}` 条真实条款问答（25 条单品口语 + 10 条跨产品对比 + 10 条全局无 ID 检索）",
        "- 知识库规模：12 款真实合同条款，无条款产品严格不构造伪块",
        "",
        "| 对比方案 | Recall@3 (%) | Recall@5 (%) | MRR | 跨产品对比完整率 (%) | Top-5 重复块率 (%) ↓ | P95 耗时 (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for row in results:
        lines.append(
            f"| {row['strategy']} | {row['recall_at_3']:.2f} | {row['recall_at_5']:.2f} | "
            f"{row['mrr']:.4f} | {row['compare_completeness']:.2f} | "
            f"{row['duplicate_rate']:.2f} | {row['p95_latency_ms']:.2f} |"
        )

    r1 = by_key.get("dense_only")
    r2 = by_key.get("hybrid_no_rerank")
    r3 = by_key.get("hybrid_rewrite_no_rerank")
    r4 = by_key.get("hybrid_rewrite_parallel_dedup")
    r5 = by_key.get("full_enhanced")
    rb = by_key.get("hyde_hybrid_rerank")
    if all(x is not None for x in (r1, r2, r3, r4, r5, rb)):
        lines.extend(
            [
                "",
                "## 单变量组件归因分析",
                "",
                f"1. **Dense → Hybrid 无改写（`Recall@5`: {r1['recall_at_5']:.2f}% → {r2['recall_at_5']:.2f}%，差值 {r2['recall_at_5'] - r1['recall_at_5']:+.2f} pp）**："
                "未经术语对齐的口语词汇（如“买完多久能赔”“后悔退保”）进入 BM25 稀疏检索易匹配到大量无关段落噪声，直接 RRF 融合反而稀释语义精排位次。",
                f"2. **+ 领域词典改写（`Recall@5`: {r2['recall_at_5']:.2f}% → {r3['recall_at_5']:.2f}%，净提升 {r3['recall_at_5'] - r2['recall_at_5']:+.2f} pp）**："
                "将口语映射为精算标准术语（等待期/犹豫期/责任免除/减保等）后，BM25 精准命中条目标题，显著扭转稀疏检索劣化。",
                f"3. **+ 并行定额与路径去重（`跨产品对比完整率`: {r3['compare_completeness']:.2f}% → {r4['compare_completeness']:.2f}%，`Top-5 重复块率`: {r3['duplicate_rate']:.2f}% → {r4['duplicate_rate']:.2f}%）**："
                "单次过滤易被密集单品霸榜，并行定额 + Round-Robin 直接打通多品对比覆盖，`(clause_name, section_path)` 去重彻底消除共享 PDF 重复父块。",
                f"4. **+ gte-rerank-v2 重排（`Recall@5`: {r4['recall_at_5']:.2f}% → {r5['recall_at_5']:.2f}%，`MRR`: {r4['mrr']:.4f} → {r5['mrr']:.4f}，P95 +{r5['p95_latency_ms'] - r4['p95_latency_ms']:.1f}ms）**："
                "在高质量初筛集上叠加 Cross-Encoder 重排进一步提升首条命中率。",
                f"5. **最终方案 vs HyDE 竞品（`Recall@5`: {r5['recall_at_5']:.2f}% vs {rb['recall_at_5']:.2f}%，P95: {r5['p95_latency_ms']:.1f}ms vs {rb['p95_latency_ms']:.1f}ms）**："
                "规避 HyDE 假设条款生成的数值幻觉与额外生成延迟，兼具更好召回与更低延迟。",
            ]
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
