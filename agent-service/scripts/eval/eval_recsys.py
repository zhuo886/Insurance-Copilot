"""搜推漏斗 4 组方案离线消融评测脚本（同步预留 --with-llm-listwise 竞品对比）。

用法：在 agent-service 目录执行
    uv run python scripts/eval/eval_recsys.py --offline
    uv run python scripts/eval/eval_recsys.py --with-llm-listwise
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.modules.product.catalog_data import load_builtin_catalog
from app.recsys.filter import filter_eligible_candidates
from app.recsys.pipeline import RecSysPipeline
from app.recsys.pricing import PremiumEstimator
from app.recsys.recall import OfflineSemanticRecall
from app.recsys.schemas import ProductCandidate, RecUserProfile

CASES_PATH = Path(__file__).resolve().parent / "recsys_cases.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_CAT_ORDER = ["medical", "accident", "critical_illness", "life"]


def compute_ground_truth_relevance(
    case: dict[str, Any],
    catalog: list[ProductCandidate],
    estimator: PremiumEstimator,
) -> dict[int, int]:
    """基于独立承保规则、目标险种与专家精选列表计算 [0,1,2,3] 分级相关性真值。"""
    profile = RecUserProfile(**case["profile"])
    preferred_ids = set(case.get("preferred_ids") or [])
    target_cats = set(profile.target_categories)
    pref_tags = [t.lower() for t in profile.preferences]

    relevances: dict[int, int] = {}
    for product in catalog:
        if target_cats and product.category not in target_cats:
            relevances[product.id] = 0
            continue
        if profile.age < product.min_age or profile.age > product.max_age:
            relevances[product.id] = 0
            continue
        if profile.occupation_class > product.max_occupation_class:
            relevances[product.id] = 0
            continue

        est = estimator.estimate(product, profile)
        if profile.annual_budget > 0 and est.annual_premium > profile.annual_budget * 1.15:
            relevances[product.id] = 0
            continue

        if product.id in preferred_ids:
            relevances[product.id] = 3
        else:
            tag_blob = " ".join(
                [product.name, product.description, *product.highlights, *product.coverage_tags]
            ).lower()
            if pref_tags and any(tag in tag_blob for tag in pref_tags):
                relevances[product.id] = 2
            else:
                relevances[product.id] = 1
    return relevances


def compute_ndcg_at_k(ranked_ids: list[int], rel_map: dict[int, int], k: int = 5) -> float:
    dcg = 0.0
    for idx, pid in enumerate(ranked_ids[:k], start=1):
        rel = rel_map.get(pid, 0)
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(idx + 1)

    ideal_rels = sorted((r for r in rel_map.values() if r > 0), reverse=True)[:k]
    idcg = sum((2**rel - 1) / math.log2(idx + 1) for idx, rel in enumerate(ideal_rels, start=1))
    return round(dcg / idcg, 4) if idcg > 0 else 0.0


def compute_recall_at_k(ranked_ids: list[int], rel_map: dict[int, int], k: int = 10) -> float:
    relevant = {pid for pid, rel in rel_map.items() if rel >= 2}
    if not relevant:
        relevant = {pid for pid, rel in rel_map.items() if rel >= 1}
    if not relevant:
        return 1.0
    hits = len(set(ranked_ids[:k]) & relevant)
    return round(hits / len(relevant), 4)


def _is_eligible(product: ProductCandidate, profile: RecUserProfile) -> bool:
    if profile.target_categories and product.category not in profile.target_categories:
        return False
    if profile.age < product.min_age or profile.age > product.max_age:
        return False
    if profile.occupation_class > product.max_occupation_class:
        return False
    return True


def run_sql_baseline(
    profile: RecUserProfile,
    catalog: list[ProductCandidate],
    estimator: PremiumEstimator,
) -> tuple[list[int], list[int], float, int]:
    """Baseline 1：复刻原版 SQL 按 min_premium 升序、每类取前 1 款。"""
    t0 = time.perf_counter()
    cats = profile.target_categories or DEFAULT_CAT_ORDER
    ordered_pool = sorted(
        [p for p in catalog if p.category in cats],
        key=lambda x: (x.min_premium, x.id),
    )
    ranked_ids = [p.id for p in ordered_pool]

    selected_ids: list[int] = []
    seen_cats: set[str] = set()
    for p in ordered_pool:
        if p.category not in seen_cats:
            selected_ids.append(p.id)
            seen_cats.add(p.category)
        if len(selected_ids) >= 4:
            break
    latency_ms = (time.perf_counter() - t0) * 1000
    return ranked_ids, selected_ids, latency_ms, 0


def run_semantic_plus_filter(
    profile: RecUserProfile,
    catalog: list[ProductCandidate],
    estimator: PremiumEstimator,
) -> tuple[list[int], list[int], float, int]:
    """Baseline 2：单路语义向量召回(Top-15) + 硬过滤，每类贪心取首款。"""
    t0 = time.perf_counter()
    cat_map = {p.id: p for p in catalog}
    recalled_ids = OfflineSemanticRecall().recall(profile, catalog, limit=15)
    recalled = [cat_map[pid] for pid in recalled_ids if pid in cat_map]
    eligible, _ = filter_eligible_candidates(recalled, profile, estimator)
    ranked_ids = [p.id for p, _ in eligible]

    selected_ids: list[int] = []
    seen_cats: set[str] = set()
    for p, _ in eligible:
        if p.category not in seen_cats:
            selected_ids.append(p.id)
            seen_cats.add(p.category)
        if len(selected_ids) >= 4:
            break
    latency_ms = (time.perf_counter() - t0) * 1000
    return ranked_ids, selected_ids, latency_ms, 0


def run_full_funnel_no_portfolio(
    profile: RecUserProfile,
    catalog: list[ProductCandidate],
    pipeline: RecSysPipeline,
) -> tuple[list[int], list[int], float, int]:
    """实验组 3：多路 RRF + 硬过滤 + 5 维精排，但不做 MMR 与预算组合优化（每类直接取最高分）。"""
    t0 = time.perf_counter()
    result = pipeline.recommend(profile, catalog, max_items=4)
    ranked_ids = [x.product.id for x in result.ranked_candidates]

    selected_ids: list[int] = []
    seen_cats: set[str] = set()
    for x in result.ranked_candidates:
        if x.product.category not in seen_cats:
            selected_ids.append(x.product.id)
            seen_cats.add(x.product.category)
        if len(selected_ids) >= 4:
            break
    latency_ms = (time.perf_counter() - t0) * 1000
    return ranked_ids, selected_ids, latency_ms, 0


def run_full_funnel_with_portfolio(
    profile: RecUserProfile,
    catalog: list[ProductCandidate],
    pipeline: RecSysPipeline,
) -> tuple[list[int], list[int], float, int]:
    """完整方案 4：四阶段完整漏斗（+ MMR 多样性与预算约束分组背包优化）。"""
    t0 = time.perf_counter()
    result = pipeline.recommend(profile, catalog, max_items=4)
    ranked_ids = [x.product.id for x in result.ranked_candidates]
    selected_ids = [x.product.id for x in result.selected]
    latency_ms = (time.perf_counter() - t0) * 1000
    return ranked_ids, selected_ids, latency_ms, 0


def create_llm_listwise_runner(catalog: list[ProductCandidate]):
    """构建带异常兜底与连接复用的纯 LLM Listwise 直排竞品调用器。"""
    from langchain.chat_models import init_chat_model
    from langchain_core.messages import HumanMessage, SystemMessage
    from pydantic import BaseModel, Field

    from app.core.config import settings
    from app.core.logging import get_logger

    logger = get_logger(__name__)

    class LLMListwiseOutput(BaseModel):
        ranked_ids: list[int] = Field(description="按匹配度从高到低排名前10的产品ID")
        selected_ids: list[int] = Field(description="最终在预算内推荐给用户的1-4个产品ID组合")

    model = init_chat_model(
        model=settings.llm.chat_model,
        model_provider="openai",
        api_key=settings.llm.api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        extra_body={"thinking": {"type": "disabled"}},
    ).with_structured_output(LLMListwiseOutput, include_raw=True)

    catalog_text = "\n".join(
        f"ID={p.id} | {p.name} | 险种={p.category} | 适用={p.target_group} | 职业上限=1-{p.max_occupation_class}类 | 参考保费={p.min_premium}~{p.max_premium or p.min_premium} | 亮点={','.join(p.highlights)}"
        for p in catalog
    )

    def _run(
        profile: RecUserProfile,
        _catalog: list[ProductCandidate],
    ) -> tuple[list[int], list[int], float, int]:
        user_text = (
            f"用户画像：年龄={profile.age}岁，性别={profile.gender}，职业={profile.occupation_class}类，"
            f"医保={'有' if profile.has_social_security else '无'}，年预算={profile.annual_budget}元，"
            f"目标险种={profile.target_categories or '不限'}，偏好={profile.preferences}。\n"
            f"需求：{profile.query_text}\n\n候选产品：\n{catalog_text}"
        )

        t0 = time.perf_counter()
        try:
            resp = model.invoke(
                [
                    SystemMessage(content="你是保险产品排序引擎，请严格按预算、年龄和职业条件筛选并输出产品ID。"),
                    HumanMessage(content=user_text),
                ]
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            parsed = resp.get("parsed") if isinstance(resp, dict) else resp
            raw_msg = resp.get("raw") if isinstance(resp, dict) else None
            usage = getattr(raw_msg, "usage_metadata", None) or {}
            tokens = int(usage.get("total_tokens", 0))

            ranked = list(getattr(parsed, "ranked_ids", []) or [])
            selected = list(getattr(parsed, "selected_ids", []) or [])
            return ranked, selected, latency_ms, tokens
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.warning("LLM Listwise 单条评测异常，已兜底空结果", error=str(exc))
            return [], [], latency_ms, 0

    return _run


def evaluate_strategy(
    name: str,
    cases: list[dict[str, Any]],
    catalog: list[ProductCandidate],
    estimator: PremiumEstimator,
    runner,
    max_workers: int = 1,
) -> dict[str, Any]:
    catalog_map = {p.id: p for p in catalog}
    recall10_scores: list[float] = []
    ndcg5_scores: list[float] = []
    eligibility_passes: list[float] = []
    budget_satisfactions: list[float] = []
    category_coverages: list[float] = []
    latencies: list[float] = []
    tokens_list: list[int] = []

    def _run_case(case: dict[str, Any]):
        profile = RecUserProfile(**case["profile"])
        return case, profile, runner(profile, catalog)

    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            outputs = list(pool.map(_run_case, cases))
    else:
        outputs = [_run_case(case) for case in cases]

    for case, profile, (ranked_ids, selected_ids, latency_ms, tokens) in outputs:
        rel_map = compute_ground_truth_relevance(case, catalog, estimator)

        recall10_scores.append(compute_recall_at_k(ranked_ids, rel_map, k=10))
        ndcg5_scores.append(compute_ndcg_at_k(ranked_ids, rel_map, k=5))
        latencies.append(latency_ms)
        tokens_list.append(tokens)

        selected_products = [catalog_map[pid] for pid in selected_ids if pid in catalog_map]
        all_eligible = bool(selected_products) and all(_is_eligible(p, profile) for p in selected_products)
        eligibility_passes.append(1.0 if all_eligible else 0.0)

        total_prem = sum(estimator.estimate(p, profile).annual_premium for p in selected_products)
        within_budget = bool(selected_products) and (
            profile.annual_budget <= 0 or total_prem <= profile.annual_budget * 1.001
        )
        budget_satisfactions.append(1.0 if within_budget else 0.0)

        target_cats = set(profile.target_categories) if profile.target_categories else set(
            DEFAULT_CAT_ORDER[:3]
        )
        covered_cats = {p.category for p in selected_products if p.category in target_cats}
        category_coverages.append(len(covered_cats) / max(len(target_cats), 1))

    sorted_lat = sorted(latencies)
    p95_idx = min(len(sorted_lat) - 1, int(0.95 * len(sorted_lat)))

    return {
        "strategy": name,
        "cases": len(cases),
        "recall_at_10": round(statistics.mean(recall10_scores) * 100, 2),
        "ndcg_at_5": round(statistics.mean(ndcg5_scores), 4),
        "eligibility_pass_rate": round(statistics.mean(eligibility_passes) * 100, 2),
        "budget_satisfaction_rate": round(statistics.mean(budget_satisfactions) * 100, 2),
        "category_coverage_rate": round(statistics.mean(category_coverages) * 100, 2),
        "p95_latency_ms": round(sorted_lat[p95_idx], 2),
        "avg_tokens": round(statistics.mean(tokens_list), 1),
    }


def render_markdown_report(results: list[dict[str, Any]]) -> str:
    lines = [
        "# 搜推漏斗离线消融评测报告",
        "",
        f"- 测试用例数：`{results[0]['cases']}` 条典型场景画像（涵盖少儿/青年/中老年/5-6类高危/紧预算多险种）",
        "- 候选产品池：`50` 款结构化保险产品（12 款含真条款 + 38 款结构化扩展）",
        "",
        "| 对比方案 | Recall@10 (%) | NDCG@5 | 承保合规率 (%) | 组合预算满足率 (%) | 目标险种覆盖率 (%) | P95 耗时 (ms) | 平均排序 Token |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for row in results:
        lines.append(
            f"| {row['strategy']} | {row['recall_at_10']:.2f} | {row['ndcg_at_5']:.4f} | "
            f"{row['eligibility_pass_rate']:.2f} | {row['budget_satisfaction_rate']:.2f} | "
            f"{row['category_coverage_rate']:.2f} | {row['p95_latency_ms']:.2f} | {row['avg_tokens']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="搜推四阶段漏斗消融评测")
    parser.add_argument("--offline", action="store_true", help="仅运行离线确定性+语义消融（默认）")
    parser.add_argument("--with-llm-listwise", action="store_true", help="额外评测纯 LLM Listwise 直排竞品")
    args = parser.parse_args()

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    catalog = load_builtin_catalog()
    estimator = PremiumEstimator()
    pipeline = RecSysPipeline(estimator=estimator)

    strategies = [
        ("1. SQL规则排序 (Baseline)", lambda p, c: run_sql_baseline(p, c, estimator), 1),
        ("2. 单路语义召回 + 硬过滤", lambda p, c: run_semantic_plus_filter(p, c, estimator), 1),
        ("3. 多路RRF + 5维精排 (无组合优化)", lambda p, c: run_full_funnel_no_portfolio(p, c, pipeline), 1),
        ("4. 完整四阶段漏斗 (+MMR+预算背包)", lambda p, c: run_full_funnel_with_portfolio(p, c, pipeline), 1),
    ]
    if args.with_llm_listwise:
        strategies.append(("竞品. 纯LLM Listwise直排", create_llm_listwise_runner(catalog), 4))

    results = [
        evaluate_strategy(name, cases, catalog, estimator, fn, max_workers=workers)
        for name, fn, workers in strategies
    ]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    md_report = render_markdown_report(results)
    (REPORTS_DIR / "recsys_eval.md").write_text(md_report, encoding="utf-8")
    (REPORTS_DIR / "recsys_eval.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(md_report)


if __name__ == "__main__":
    main()
