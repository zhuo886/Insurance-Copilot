from app.modules.product.catalog_data import load_builtin_catalog
from app.recsys.pricing import PremiumEstimator
from app.recsys.schemas import RecUserProfile


def test_tier1_rate_table_hit_for_seeded_products():
    catalog = {item.id: item for item in load_builtin_catalog()}
    estimator = PremiumEstimator()

    profile_f30 = RecUserProfile(age=30, gender="female", occupation_class=1, has_social_security=True)
    est_darwen = estimator.estimate(catalog[2], profile_f30, coverage_amount=500000.0)
    assert est_darwen.pricing_mode == "table"
    assert est_darwen.annual_premium == 3900.0

    est_med = estimator.estimate(catalog[6], profile_f30, deductible=10000.0)
    assert est_med.pricing_mode == "table"
    assert est_med.annual_premium == 350.0

    est_acc = estimator.estimate(catalog[10], profile_f30, coverage_amount=500000.0)
    assert est_acc.pricing_mode == "table"
    assert est_acc.annual_premium == 136.0


def test_tier2_mock_adapter_fallback_when_table_misses():
    catalog = {item.id: item for item in load_builtin_catalog()}
    estimator = PremiumEstimator()

    # 众安尊享e生在无社保条件时跳过 Tier 1 精确表，命中 Tier 2 mock_adapter
    profile_no_ss = RecUserProfile(age=35, gender="male", occupation_class=2, has_social_security=False)
    est = estimator.estimate(catalog[6], profile_no_ss, deductible=10000.0)
    assert est.pricing_mode == "mock_adapter"
    # 300 * 1.6 (age 31-50) * 1.3 (no ss) = 624.0
    assert est.annual_premium == 624.0


def test_tier3_formula_covers_all_50_products_and_is_age_monotonic():
    catalog = load_builtin_catalog()
    estimator = PremiumEstimator()
    assert len(catalog) == 50

    # 强制清空前两级表以测试全部 50 款的 Tier 3 近似公式
    formula_estimator = PremiumEstimator(rate_table_items={}, mock_bindings={})
    for product in catalog:
        young_age = product.min_age
        mid_age = (product.min_age + product.max_age) // 2
        old_age = product.max_age

        est_young = formula_estimator.estimate(product, RecUserProfile(age=young_age))
        est_mid = formula_estimator.estimate(product, RecUserProfile(age=mid_age))
        est_old = formula_estimator.estimate(product, RecUserProfile(age=old_age))

        assert est_young.pricing_mode == "estimated_formula"
        assert est_young.annual_premium > 0
        assert est_young.annual_premium <= est_mid.annual_premium <= est_old.annual_premium

        # 同时验证默认分级估算器对全量 50 款均可返回正保费
        est_default = estimator.estimate(product, RecUserProfile(age=mid_age))
        assert est_default.annual_premium > 0
