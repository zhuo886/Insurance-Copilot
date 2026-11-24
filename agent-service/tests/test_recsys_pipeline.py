import pytest

from app.modules.product.catalog_data import load_builtin_catalog
from app.recsys.filter import filter_eligible_candidates
from app.recsys.pipeline import RecSysPipeline
from app.recsys.pricing import PremiumEstimator
from app.recsys.recall import build_profile_query_text, rrf_fuse, tokenize_insurance_text
from app.recsys.reranker import mmr_diversify, optimize_budget_portfolio
from app.recsys.schemas import (
    PremiumEstimate,
    ProductCandidate,
    RecUserProfile,
    ScoreBreakdown,
    ScoredCandidate,
)
from app.recsys.scorer import extract_query_preference_tags, score_candidates


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

    senior_profile = RecUserProfile(age=62, occupation_class=2, annual_budget=5000)
    eligible, rejections = filter_eligible_candidates(catalog, senior_profile, estimator)
    eligible_ids = {p.id for p, _ in eligible}
    rejected_ids = {r.product_id for r in rejections}

    for blocked in (1, 2, 3, 4, 10, 11, 12):
        assert blocked not in eligible_ids and blocked in rejected_ids
    assert 5 in eligible_ids and 6 in eligible_ids

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


def test_tokenize_and_build_profile_query_text():
    assert tokenize_insurance_text("") == []
    tokens = tokenize_insurance_text("6类职业保证续保20年医疗险")
    assert "保证续保" in tokens and "20" in tokens and "医疗" in tokens

    kid_q = build_profile_query_text(RecUserProfile(age=5, target_categories=["medical"]))
    assert "少儿" in kid_q and "医疗险" in kid_q

    hr_q = build_profile_query_text(RecUserProfile(age=55, occupation_class=6, has_social_security=False))
    assert "中老年" in hr_q and "高危职业" in hr_q and "无社保" in hr_q


def test_rrf_fuse_empty_and_single_channel_and_category_filter():
    assert rrf_fuse({}) == []
    single = rrf_fuse({"semantic": [6, 4, 2]}, top_k=2)
    assert [pid for pid, _, _ in single] == [6, 4]

    catalog = load_builtin_catalog()
    estimator = PremiumEstimator()
    eligible, rejections = filter_eligible_candidates(
        catalog, RecUserProfile(age=30, target_categories=["accident"]), estimator
    )
    assert all(p.category == "accident" for p, _ in eligible)
    assert any(any("不在目标范围" in msg for msg in r.reasons) for r in rejections)


def test_scorer_factors_bounds_and_loose_underwriting_bonus():
    assert score_candidates([], RecUserProfile(age=30), {}) == []

    p_loose = ProductCandidate(
        id=1, name="众民保百万医疗", clause_name="a.md", category="medical",
        insurer="众安", health_disclosure_level="loose", coverage_tags=["无社保友好", "核保宽松"],
    )
    p_strict = ProductCandidate(
        id=2, name="严格医疗", clause_name="b.md", category="medical",
        insurer="友邦", health_disclosure_level="strict", coverage_tags=["百万医疗"],
    )
    profile = RecUserProfile(age=30, has_social_security=False, annual_budget=1000, query_text="三高结节无社保怎么买")
    est1 = PremiumEstimate(product_id=1, annual_premium=300.0, pricing_mode="table")
    est2 = PremiumEstimate(product_id=2, annual_premium=300.0, pricing_mode="table")

    scored = score_candidates([(p_strict, est2), (p_loose, est1)], profile, {1: 0.05, 2: 0.05})
    assert scored[0].product.id == 1
    for item in scored:
        assert 0.0 <= item.scores.total_score <= 1.0
        assert 0.0 <= item.scores.profile_fit <= 1.0

    tags = extract_query_preference_tags(RecUserProfile(age=30, preferences=["外购药"], query_text="想要外购药和0免赔"))
    assert tags == ["外购药", "0免赔"]


def test_mmr_and_portfolio_empty_and_unlimited_budget():
    assert mmr_diversify([]) == []
    empty_port = optimize_budget_portfolio([], RecUserProfile(age=30, target_categories=["medical", "accident"]))
    assert empty_port.selected == []
    assert empty_port.missing_categories == ["medical", "accident"]

    cand = ScoredCandidate(
        product=ProductCandidate(id=6, name="尊享e生", clause_name="6.md", category="medical", insurer="众安"),
        estimate=PremiumEstimate(product_id=6, annual_premium=318.0, pricing_mode="table"),
        scores=ScoreBreakdown(semantic_rrf=0.9, profile_fit=0.9, coverage_tag_match=0.9, budget_efficiency=0.9, clause_quality=0.9, total_score=0.9),
    )
    assert len(mmr_diversify([cand], lambda_param=1.0, top_k=1)) == 1
    unbounded = optimize_budget_portfolio([cand], RecUserProfile(age=30, annual_budget=0, target_categories=["medical"]))
    assert unbounded.within_budget is True
    assert unbounded.selected[0].product.id == 6


def test_builtin_12_products_catalog_invariants():
    catalog = load_builtin_catalog()
    assert len(catalog) == 12
    assert [p.id for p in catalog] == list(range(1, 13))
    assert all(p.min_age <= p.max_age and 1 <= p.max_occupation_class <= 6 for p in catalog)
    assert all(p.has_clause for p in catalog)
