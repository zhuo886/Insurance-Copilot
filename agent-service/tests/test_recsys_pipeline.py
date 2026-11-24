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


def test_rrf_fuse_combines_channels_and_preserves_multi_category_quota():
    ranked_lists = {
        "semantic": [4, 5, 6, 2, 3, 10],
        "bm25": [4, 6, 2, 10, 11],
        "structured": [10, 11, 2, 4, 3],
    }
    category_map = {
        4: "medical",
        5: "medical",
        6: "medical",
        2: "critical_illness",
        3: "critical_illness",
        10: "accident",
        11: "accident",
    }
    fused = rrf_fuse(
        ranked_lists,
        k=60,
        top_k=7,
        category_map=category_map,
        target_categories=["medical", "critical_illness", "accident"],
        min_per_category=2,
    )
    fused_ids = [pid for pid, _, _ in fused]
    assert len(fused_ids) == 7
    assert sum(1 for pid in fused_ids if category_map[pid] == "critical_illness") >= 2
    assert sum(1 for pid in fused_ids if category_map[pid] == "accident") >= 2


def test_hard_filter_blocks_age_and_occupation_violations_on_12_products():
    catalog = load_builtin_catalog()
    assert len(catalog) == 12
    estimator = PremiumEstimator()

    # 62 岁 2 类职业老人：必须拦截达尔文12号(max 35)、星相守个人版(max 55)、小蜜蜂6号(max 50)
    senior_profile = RecUserProfile(age=62, occupation_class=2, annual_budget=5000)
    eligible, rejections = filter_eligible_candidates(catalog, senior_profile, estimator)
    eligible_ids = {p.id for p, _ in eligible}
    rejected_ids = {r.product_id for r in rejections}

    for blocked in (1, 2, 3, 4, 10, 11, 12):
        assert blocked not in eligible_ids and blocked in rejected_ids
    # 众安尊享e生(id=6, max_age=70) 与 星相守转保版(id=5, max_age=70) 应通过
    assert 5 in eligible_ids
    assert 6 in eligible_ids

    # 30 岁 6 类高危职业：必须拦截所有 max_occupation_class <= 4 的产品，仅 6/7/8/9 合规
    high_risk_profile = RecUserProfile(age=30, occupation_class=6, annual_budget=15000)
    eligible_hr, _ = filter_eligible_candidates(catalog, high_risk_profile, estimator)
    assert all(p.max_occupation_class >= 6 for p, _ in eligible_hr)
    assert {p.id for p, _ in eligible_hr} == {6, 7, 8, 9}


def test_mmr_penalizes_duplicate_insurer_and_shared_clause():
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
        id=6,
        name="众安尊享e生2026",
        clause_name="other.pdf",
        category="medical",
        insurer="众安保险",
        has_clause=True,
        guarantee_renewal_years=1,
        coverage_tags=["百万医疗", "外购药"],
    )

    est = PremiumEstimate(product_id=4, annual_premium=350.0, pricing_mode="table")
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
            estimate=est.model_copy(update={"product_id": 6}),
            scores=ScoreBreakdown(
                semantic_rrf=0.87, profile_fit=0.87, coverage_tag_match=0.87,
                budget_efficiency=0.87, clause_quality=0.87, total_score=0.87,
            ),
        ),
    ]
    diversified = mmr_diversify(scored, lambda_param=0.7, top_k=3)
    # 第二位优先选不同保司与不同条款的 p3，而非共享条款同保司的 p2
    assert diversified[0].product.id == 4
    assert diversified[1].product.id == 6


def test_budget_portfolio_optimization_respects_budget_and_handles_tight_fallback():
    def make_cand(pid: int, cat: str, prem: float, score: float) -> ScoredCandidate:
        return ScoredCandidate(
            product=ProductCandidate(
                id=pid,
                name=f"P{pid}",
                clause_name=f"c{pid}.pdf",
                category=cat,  # type: ignore[arg-type]
                insurer=f"Insurer{pid}",
            ),
            estimate=PremiumEstimate(product_id=pid, annual_premium=prem, pricing_mode="table"),
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
    profile = RecUserProfile(age=30, annual_budget=2000, target_categories=["medical", "critical_illness", "accident"])
    portfolio = optimize_budget_portfolio(candidates, profile)
    assert portfolio.within_budget is True
    assert portfolio.total_estimated_premium == 1650.0
    assert set(portfolio.covered_categories) == {"medical", "critical_illness", "accident"}

    tight_profile = RecUserProfile(age=30, annual_budget=100, target_categories=["medical", "accident"])
    tight_portfolio = optimize_budget_portfolio(candidates, tight_profile)
    assert tight_portfolio.within_budget is False
    assert tight_portfolio.budget_gap > 0
    assert len(tight_portfolio.selected) == 1


def test_full_recsys_pipeline_end_to_end_across_real_12_products():
    catalog = load_builtin_catalog()
    pipeline = RecSysPipeline()

    # 1. 少儿配置（5岁，预算 1800，医疗+重疾+意外）
    kid_profile = RecUserProfile(
        age=5,
        annual_budget=1800,
        target_categories=["medical", "critical_illness", "accident"],
        preferences=["少儿特疾", "学平险"],
        query_text="5岁宝宝配齐医疗重疾意外，预算1800",
    )
    kid_result = pipeline.recommend(kid_profile, catalog)
    assert kid_result.within_budget is True
    assert set(kid_result.covered_categories) == {"medical", "critical_illness", "accident"}
    assert all(x.product.min_age <= 5 <= x.product.max_age for x in kid_result.selected)

    # 2. 62岁老人（医疗，预算 3000）
    senior_profile = RecUserProfile(
        age=62,
        annual_budget=3000,
        target_categories=["medical"],
        preferences=["中老年医疗", "0免赔"],
        query_text="给62岁父亲买长期医疗险，预算3000",
    )
    senior_result = pipeline.recommend(senior_profile, catalog)
    assert senior_result.within_budget is True
    assert senior_result.covered_categories == ["medical"]
    assert all(x.product.min_age <= 62 <= x.product.max_age for x in senior_result.selected)

    # 3. 32岁 6类高危职业请求医疗+意外：医疗命中 ID 6，真实库无6类意外险则如实标注 missing_categories
    hr_profile = RecUserProfile(
        age=32,
        occupation_class=6,
        annual_budget=2000,
        target_categories=["medical", "accident"],
        query_text="32岁高空作业6类职业配置医疗和意外",
    )
    hr_result = pipeline.recommend(hr_profile, catalog)
    assert hr_result.within_budget is True
    assert [x.product.id for x in hr_result.selected] == [6]
    assert "accident" in hr_result.missing_categories
    assert all(x.product.max_occupation_class >= 6 for x in hr_result.selected)
