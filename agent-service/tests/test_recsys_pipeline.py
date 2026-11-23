from app.modules.product.catalog_data import load_builtin_catalog
from app.recsys.filter import filter_eligible_candidates
from app.recsys.pipeline import RecSysPipeline
from app.recsys.pricing import PremiumEstimator
from app.recsys.recall import rrf_fuse
from app.recsys.reranker import mmr_diversify, optimize_budget_portfolio
from app.recsys.schemas import (
    PremiumEstimate,
    ProductCandidate,
    RecUserProfile,
    ScoreBreakdown,
    ScoredCandidate,
)
from app.recsys.scorer import score_candidates


def test_rrf_fuse_combines_channels_and_preserves_multi_category_quota():
    ranked_lists = {
        "semantic": [4, 5, 6, 13, 14, 2],
        "bm25": [4, 13, 2, 10, 33],
        "structured": [10, 33, 2, 4, 26],
    }
    category_map = {
        4: "medical",
        5: "medical",
        6: "medical",
        13: "medical",
        14: "medical",
        2: "critical_illness",
        26: "critical_illness",
        10: "accident",
        33: "accident",
    }
    fused = rrf_fuse(
        ranked_lists,
        k=60,
        top_k=8,
        category_map=category_map,
        target_categories=["medical", "critical_illness", "accident"],
        min_per_category=2,
    )
    fused_ids = [pid for pid, _, _ in fused]
    assert len(fused_ids) == 8
    assert sum(1 for pid in fused_ids if category_map[pid] == "critical_illness") >= 2
    assert sum(1 for pid in fused_ids if category_map[pid] == "accident") >= 2


def test_hard_filter_blocks_age_and_occupation_violations():
    catalog = load_builtin_catalog()
    estimator = PremiumEstimator()

    # 62 岁 2 类职业老人：必须拦截达尔文12号(max 35)、达尔文宝贝(max 17)、小蜜蜂6号(max 50)
    senior_profile = RecUserProfile(age=62, occupation_class=2, annual_budget=5000)
    eligible, rejections = filter_eligible_candidates(catalog, senior_profile, estimator)
    eligible_ids = {p.id for p, _ in eligible}
    rejected_ids = {r.product_id for r in rejections}

    assert 1 not in eligible_ids and 1 in rejected_ids
    assert 2 not in eligible_ids and 2 in rejected_ids
    assert 10 not in eligible_ids and 10 in rejected_ids
    # 众安尊享e生(id=6, max_age=70) 与老年意外险(id=34, 50-85) 应通过
    assert 6 in eligible_ids
    assert 34 in eligible_ids

    # 30 岁 6 类高危职业：必须拦截所有 max_occupation_class <= 4 的产品
    high_risk_profile = RecUserProfile(age=30, occupation_class=6, annual_budget=5000)
    eligible_hr, _ = filter_eligible_candidates(catalog, high_risk_profile, estimator)
    assert all(p.max_occupation_class >= 6 for p, _ in eligible_hr)
    assert any(p.id == 36 for p, _ in eligible_hr)


def test_scorer_and_mmr_penalize_duplicate_insurer_and_clause():
    p1 = ProductCandidate(
        id=4,
        name="星相守2号个人版",
        clause_name="shared.pdf",
        category="medical",
        insurer="复星联合",
        has_clause=True,
        guarantee_renewal_years=20,
        coverage_tags=["保证续保20年", "外购药"],
    )
    p2 = ProductCandidate(
        id=5,
        name="星相守2号转保版",
        clause_name="shared.pdf",
        category="medical",
        insurer="复星联合",
        has_clause=True,
        guarantee_renewal_years=20,
        coverage_tags=["保证续保20年"],
    )
    p3 = ProductCandidate(
        id=13,
        name="长相安2号",
        clause_name="other.md",
        category="medical",
        insurer="平安健康",
        has_clause=False,
        guarantee_renewal_years=20,
        coverage_tags=["保证续保20年", "外购药"],
    )

    est = PremiumEstimate(product_id=4, annual_premium=350.0, pricing_mode="estimated_formula")
    scored = [
        ScoredCandidate(
            product=p1,
            estimate=est,
            scores=ScoreBreakdown(
                semantic_rrf=0.9, profile_fit=0.9, coverage_tag_match=0.9,
                budget_efficiency=0.9, clause_quality=0.9, total_score=0.90,
            ),
        ),
        ScoredCandidate(
            product=p2,
            estimate=est.model_copy(update={"product_id": 5}),
            scores=ScoreBreakdown(
                semantic_rrf=0.88, profile_fit=0.88, coverage_tag_match=0.88,
                budget_efficiency=0.88, clause_quality=0.88, total_score=0.88,
            ),
        ),
        ScoredCandidate(
            product=p3,
            estimate=est.model_copy(update={"product_id": 13}),
            scores=ScoreBreakdown(
                semantic_rrf=0.87, profile_fit=0.87, coverage_tag_match=0.87,
                budget_efficiency=0.87, clause_quality=0.87, total_score=0.87,
            ),
        ),
    ]
    diversified = mmr_diversify(scored, lambda_param=0.7, top_k=3)
    # 第二位应优先选不同保险公司且不同条款文件的 p3，而不是共享条款同保司的 p2
    assert diversified[0].product.id == 4
    assert diversified[1].product.id == 13


def test_budget_portfolio_optimization_respects_budget_and_handles_tight_fallback():
    def make_cand(pid: int, cat: str, prem: float, score: float) -> ScoredCandidate:
        return ScoredCandidate(
            product=ProductCandidate(
                id=pid,
                name=f"P{pid}",
                clause_name=f"c{pid}.md",
                category=cat,  # type: ignore[arg-type]
                insurer=f"Insurer{pid}",
            ),
            estimate=PremiumEstimate(product_id=pid, annual_premium=prem, pricing_mode="estimated_formula"),
            scores=ScoreBreakdown(
                semantic_rrf=score, profile_fit=score, coverage_tag_match=score,
                budget_efficiency=score, clause_quality=score, total_score=score,
            ),
        )

    candidates = [
        make_cand(1, "medical", 400, 0.90),
        make_cand(2, "critical_illness", 2200, 0.95),
        make_cand(3, "critical_illness", 1100, 0.85),
        make_cand(4, "accident", 150, 0.88),
    ]
    # 预算 2000：应能覆盖全部 3 个险种（400 + 1100 + 150 = 1650），而不是选 2200 元重疾导致超支或少险种
    profile = RecUserProfile(age=30, annual_budget=2000, target_categories=["medical", "critical_illness", "accident"])
    portfolio = optimize_budget_portfolio(candidates, profile)
    assert portfolio.within_budget is True
    assert portfolio.total_estimated_premium == 1650.0
    assert set(portfolio.covered_categories) == {"medical", "critical_illness", "accident"}

    # 极端预算 100 元时优雅降级并准确给出 budget_gap
    tight_profile = RecUserProfile(age=30, annual_budget=100, target_categories=["medical", "accident"])
    tight_portfolio = optimize_budget_portfolio(candidates, tight_profile)
    assert tight_portfolio.within_budget is False
    assert tight_portfolio.budget_gap > 0
    assert len(tight_portfolio.selected) == 1


def test_full_recsys_pipeline_end_to_end_across_personas():
    catalog = load_builtin_catalog()
    pipeline = RecSysPipeline()

    # 1. 少儿配置（5岁，预算 1500，医疗+重疾+意外）
    kid_profile = RecUserProfile(
        age=5,
        annual_budget=1500,
        target_categories=["medical", "critical_illness", "accident"],
        preferences=["少儿特疾", "0免赔"],
        query_text="5岁宝宝配齐医疗重疾意外，预算1500",
    )
    kid_result = pipeline.recommend(kid_profile, catalog)
    assert kid_result.within_budget is True
    assert all(x.product.min_age <= 5 <= x.product.max_age for x in kid_result.selected)
    assert len(kid_result.trace.stages) == 4
    assert kid_result.trace.stages[0].output_count <= 15

    # 2. 62岁老人父母（医疗+意外，预算 2500）
    senior_profile = RecUserProfile(
        age=62,
        annual_budget=2500,
        target_categories=["medical", "accident"],
        preferences=["核保宽松", "老人意外"],
        query_text="给62岁父母买医疗险和防跌倒意外险，三高友好",
    )
    senior_result = pipeline.recommend(senior_profile, catalog)
    assert senior_result.within_budget is True
    assert set(senior_result.covered_categories) == {"medical", "accident"}
    assert all(x.product.min_age <= 62 <= x.product.max_age for x in senior_result.selected)

    # 3. 32岁 6类高危职业（重疾+医疗+意外，预算 3500）
    hr_profile = RecUserProfile(
        age=32,
        occupation_class=6,
        annual_budget=3500,
        target_categories=["medical", "critical_illness", "accident"],
        query_text="32岁高空作业6类职业配置医疗重疾意外",
    )
    hr_result = pipeline.recommend(hr_profile, catalog)
    assert hr_result.within_budget is True
    assert all(x.product.max_occupation_class >= 6 for x in hr_result.selected)
