from app.modules.product.catalog_data import load_builtin_catalog
from app.recsys.pricing import PremiumEstimator
from app.recsys.schemas import RecUserProfile


def test_tier1_official_rate_tables_hit_11_of_12_products():
    catalog = {item.id: item for item in load_builtin_catalog()}
    assert len(catalog) == 12
    estimator = PremiumEstimator()

    profile_f30 = RecUserProfile(age=30, gender="female", occupation_class=1, has_social_security=True)
    profile_m30 = RecUserProfile(age=30, gender="male", occupation_class=1, has_social_security=True)

    # ID 2: 达尔文12号（每千元保额费率，30岁女20年交=17.11，30岁男=18.24）
    est_darwen_500k = estimator.estimate(catalog[2], profile_f30, coverage_amount=500000.0)
    assert est_darwen_500k.pricing_mode == "table"
    assert est_darwen_500k.is_estimated is False
    assert est_darwen_500k.annual_premium == 8555.0

    est_darwen_default = estimator.estimate(catalog[2], profile_m30)
    assert est_darwen_default.pricing_mode == "table"
    assert est_darwen_default.annual_premium == 1824.0

    # ID 3: 医联有盟（35岁有医保20年交 10万基础表）
    est_ylym = estimator.estimate(catalog[3], RecUserProfile(age=35, gender="male", has_social_security=True))
    assert est_ylym.pricing_mode == "table"
    assert est_ylym.is_estimated is False
    assert est_ylym.annual_premium == 3175.0

    # ID 4 & 5: 星相守2号个人版(60岁有医保计划一=1594) 与 转保版(62岁无等待期=2600)
    est_xxs4 = estimator.estimate(catalog[4], RecUserProfile(age=60, has_social_security=True), deductible=10000.0)
    assert est_xxs4.pricing_mode == "table"
    assert est_xxs4.annual_premium == 1594.0

    est_xxs5 = estimator.estimate(catalog[5], RecUserProfile(age=62, has_social_security=True))
    assert est_xxs5.pricing_mode == "table"
    assert est_xxs5.annual_premium == 2600.0

    # ID 6: 众安尊享e生2026（30岁有社保=318.0，无社保=679.0）
    est_zx_yes = estimator.estimate(catalog[6], profile_f30, deductible=10000.0)
    assert est_zx_yes.pricing_mode == "table"
    assert est_zx_yes.annual_premium == 318.0

    est_zx_no = estimator.estimate(
        catalog[6],
        RecUserProfile(age=30, gender="male", occupation_class=5, has_social_security=False),
        deductible=10000.0,
    )
    assert est_zx_no.pricing_mode == "table"
    assert est_zx_no.annual_premium == 679.0

    # ID 7 / 8 / 9: 终身寿险每千元保费对应基本保额官方表
    est_life7 = estimator.estimate(catalog[7], profile_f30)
    est_life8 = estimator.estimate(catalog[8], profile_m30)
    est_life9 = estimator.estimate(catalog[9], profile_f30)
    assert est_life7.pricing_mode == "table" and est_life7.annual_premium == 10000.0
    assert est_life7.breakdown["basic_sum_assured"] == 43610.0
    assert est_life8.pricing_mode == "table" and est_life8.annual_premium == 5000.0
    assert est_life8.breakdown["basic_sum_assured"] == 21830.0
    assert est_life9.pricing_mode == "table" and est_life9.annual_premium == 10000.0
    assert est_life9.breakdown["basic_sum_assured"] == 46430.0

    # ID 10 / 11 / 12: 意外险固定计划官方费率表（2类职业范围匹配）
    est_xmf = estimator.estimate(catalog[10], RecUserProfile(age=30, occupation_class=2))
    assert est_xmf.pricing_mode == "table"
    assert est_xmf.annual_premium == 136.0

    est_rose = estimator.estimate(catalog[11], RecUserProfile(age=30, occupation_class=2))
    assert est_rose.pricing_mode == "table"
    assert est_rose.annual_premium == 169.0

    est_xxt = estimator.estimate(catalog[12], RecUserProfile(age=10, occupation_class=1))
    assert est_xxt.pricing_mode == "table"
    assert est_xxt.annual_premium == 240.0

    est_xxt_basic = estimator.estimate(
        catalog[12], RecUserProfile(age=10, occupation_class=1), plan_code="basic"
    )
    assert est_xxt_basic.pricing_mode == "table"
    assert est_xxt_basic.annual_premium == 180.0


def test_missing_pdf_product_id1_falls_back_with_explicit_disclaimer():
    catalog = {item.id: item for item in load_builtin_catalog()}
    estimator = PremiumEstimator()

    kid_young = estimator.estimate(catalog[1], RecUserProfile(age=0, gender="female"))
    kid_mid = estimator.estimate(catalog[1], RecUserProfile(age=8, gender="female"))
    kid_old = estimator.estimate(catalog[1], RecUserProfile(age=17, gender="female"))

    assert kid_young.pricing_mode == "estimated_formula"
    assert kid_young.is_estimated is True
    assert "暂缺官方精算费率表" in kid_young.pricing_disclaimer
    assert kid_young.annual_premium > 0
    assert kid_young.annual_premium <= kid_mid.annual_premium <= kid_old.annual_premium
