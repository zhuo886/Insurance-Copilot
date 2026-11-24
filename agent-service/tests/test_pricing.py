import pytest

from app.modules.premium.service import (
    _extract_answer,
    _parse_gender,
    _parse_occupation_class,
    _parse_social_security,
)
from app.modules.product.catalog_data import load_builtin_catalog
from app.recsys.pricing import PremiumEstimator, _occupation_matches
from app.recsys.schemas import ProductCandidate, RecUserProfile


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


@pytest.mark.parametrize(
    ("pid", "age", "expected_mode"),
    [
        (1, 5, "estimated_formula"),
        (2, 30, "table"),
        (3, 35, "table"),
        (4, 30, "table"),
        (5, 62, "table"),
        (6, 30, "table"),
        (7, 30, "table"),
        (8, 30, "table"),
        (9, 30, "table"),
        (10, 30, "table"),
        (11, 30, "table"),
        (12, 10, "table"),
    ],
)
def test_every_builtin_product_estimates_cleanly(pid: int, age: int, expected_mode: str):
    catalog = {item.id: item for item in load_builtin_catalog()}
    estimator = PremiumEstimator()
    est = estimator.estimate(catalog[pid], RecUserProfile(age=age, occupation_class=1, has_social_security=True))
    assert est.pricing_mode == expected_mode
    assert est.annual_premium > 0


@pytest.mark.parametrize(
    ("rule", "user_occ", "expected"),
    [
        ("all", 6, True),
        ("*", 3, True),
        ("", 1, True),
        ("1-4", 4, True),
        ("1-4", 5, False),
        ("3", 3, True),
        ("3", 2, False),
        ("invalid", 1, False),
    ],
)
def test_occupation_matches_handles_all_formats(rule: str, user_occ: int, expected: bool):
    assert _occupation_matches(rule, user_occ) is expected


def test_coverage_amount_scaling_and_social_security_differential():
    catalog = {item.id: item for item in load_builtin_catalog()}
    estimator = PremiumEstimator()
    profile = RecUserProfile(age=30, gender="female", has_social_security=True)

    p10 = estimator.estimate(catalog[2], profile, coverage_amount=100000.0).annual_premium
    p30 = estimator.estimate(catalog[2], profile, coverage_amount=300000.0).annual_premium
    p50 = estimator.estimate(catalog[2], profile, coverage_amount=500000.0).annual_premium
    assert p30 == round(p10 * 3, 2)
    assert p50 == round(p10 * 5, 2)

    # 医疗险无社保费率应严格高于有社保费率
    with_ss = estimator.estimate(catalog[6], RecUserProfile(age=30, has_social_security=True)).annual_premium
    no_ss = estimator.estimate(catalog[6], RecUserProfile(age=30, has_social_security=False)).annual_premium
    assert no_ss > with_ss


def test_custom_rate_table_injection_per_10000_sum_assured():
    prod = ProductCandidate(id=99, name="测试定寿", clause_name="t.md", category="life", insurer="测试")
    est = PremiumEstimator(
        rate_table_items={
            99: [
                {
                    "plan_code": "standard",
                    "age_min": 18,
                    "age_max": 50,
                    "gender": "male",
                    "social_security": "all",
                    "occupation_class": "1-4",
                    "payment_period": "20y",
                    "rate_type": "per_10000_sum_assured",
                    "rate_value": 12.5,
                }
            ]
        }
    )
    res = est.estimate(prod, RecUserProfile(age=30, gender="male", occupation_class=2), coverage_amount=1000000.0)
    assert res.pricing_mode == "table"
    assert res.annual_premium == 1250.0
    assert res.breakdown["rate_per_10000"] == 12.5


def test_premium_service_answer_parsers():
    nested = {"insured": {"age": 28, "gender": "female"}, "flat_key": 10000}
    assert _extract_answer(nested, "insured.age", "age") == 28
    assert _extract_answer(nested, "flat_key") == 10000
    assert _extract_answer(nested, "insured.missing", "alt", default=1) == 1

    assert _parse_social_security(False) is False
    assert _parse_social_security("无社保") is False
    assert _parse_social_security("yes") is True

    assert _parse_occupation_class("1-4类") == 4
    assert _parse_occupation_class("99") == 6
    assert _parse_occupation_class("abc") == 1

    assert _parse_gender("F") == "female"
    assert _parse_gender("女") == "female"
    assert _parse_gender("male") == "male"
