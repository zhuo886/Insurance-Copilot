from typing import Any

from app.recsys.schemas import PremiumEstimate, ProductCandidate, RecUserProfile

# 对应 01-init-biz-schema.sql 中 V5 预置的结构化费率样例
DEFAULT_RATE_TABLE_ITEMS: dict[int, list[dict[str, Any]]] = {
    2: [
        {
            "plan_code": "standard",
            "rate_type": "per_10000_sum_assured",
            "age_min": 30,
            "age_max": 30,
            "gender": "female",
            "occupation_class": "all",
            "social_security": "all",
            "payment_period": "20y",
            "coverage_period": "lifetime",
            "rate_value": 78.0,
            "premium": None,
        },
        {
            "plan_code": "standard",
            "rate_type": "per_10000_sum_assured",
            "age_min": 30,
            "age_max": 30,
            "gender": "male",
            "occupation_class": "all",
            "social_security": "all",
            "payment_period": "20y",
            "coverage_period": "lifetime",
            "rate_value": 86.0,
            "premium": None,
        },
    ],
    6: [
        {
            "plan_code": "standard",
            "rate_type": "age_band_plan_premium",
            "age_min": 0,
            "age_max": 30,
            "gender": "all",
            "occupation_class": "all",
            "social_security": "yes",
            "deductible": 10000.0,
            "rate_value": None,
            "premium": 350.0,
        },
        {
            "plan_code": "standard",
            "rate_type": "age_band_plan_premium",
            "age_min": 31,
            "age_max": 50,
            "gender": "all",
            "occupation_class": "all",
            "social_security": "yes",
            "deductible": 10000.0,
            "rate_value": None,
            "premium": 600.0,
        },
        {
            "plan_code": "standard",
            "rate_type": "age_band_plan_premium",
            "age_min": 51,
            "age_max": 70,
            "gender": "all",
            "occupation_class": "all",
            "social_security": "yes",
            "deductible": 10000.0,
            "rate_value": None,
            "premium": 1200.0,
        },
    ],
    10: [
        {
            "plan_code": "standard",
            "rate_type": "fixed_plan_premium",
            "age_min": None,
            "age_max": None,
            "gender": "all",
            "occupation_class": "1",
            "social_security": "all",
            "coverage_amount": 500000.0,
            "rate_value": None,
            "premium": 136.0,
        },
    ],
}

# 对应 01-init-biz-schema.sql 中 V6 预置的 mock_config 适配配置
DEFAULT_MOCK_BINDINGS: dict[int, dict[str, Any]] = {
    2: {
        "base_premium_by_plan": {"basic": 500, "classic": 700, "premium": 950},
        "age_factor": [{"min": 0, "max": 17, "factor": 0.8}, {"min": 18, "max": 35, "factor": 1.0}],
        "gender_factor": {"male": 1.1, "female": 1.0},
        "coverage_unit": 500000,
        "coverage_factor_enabled": True,
    },
    6: {
        "base_premium_by_plan": {"standard": 300, "plus": 460},
        "age_factor": [
            {"min": 0, "max": 30, "factor": 1.0},
            {"min": 31, "max": 50, "factor": 1.6},
            {"min": 51, "max": 70, "factor": 3.0},
        ],
        "social_security_factor": {"yes": 1.0, "no": 1.3},
        "deductible_factor": {"0": 1.5, "10000": 1.0},
        "coverage_unit": 1000000,
        "coverage_factor_enabled": True,
    },
    10: {
        "base_premium_by_plan": {"basic": 120, "classic": 180, "premium": 260},
        "age_factor": [{"min": 18, "max": 45, "factor": 1.0}, {"min": 46, "max": 65, "factor": 1.4}],
        "occupation_factor": {"1-4": 1.0, "5-6": 1.8},
        "coverage_unit": 500000,
        "coverage_factor_enabled": True,
    },
}


def _match_range_key(value: int, mapping: dict[str, float]) -> float:
    str_val = str(value)
    if str_val in mapping:
        return float(mapping[str_val])
    for key, factor in mapping.items():
        if "-" in key:
            parts = key.split("-", maxsplit=1)
            if parts[0].isdigit() and parts[1].isdigit():
                lo, hi = int(parts[0]), int(parts[1])
                if lo <= value <= hi:
                    return float(factor)
    return 1.0


def _match_age_factor(age: int, bands: list[dict[str, Any]]) -> float:
    for band in bands:
        lo = band.get("min", 0)
        hi = band.get("max", 120)
        if lo <= age <= hi:
            return float(band.get("factor", 1.0))
    return 1.0


class PremiumEstimator:
    """三级降级保费估算引擎：rate_table_items -> mock_adapter -> product_features 系数公式"""

    def __init__(
        self,
        rate_table_items: dict[int, list[dict[str, Any]]] | None = None,
        mock_bindings: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        self.rate_table_items = (
            DEFAULT_RATE_TABLE_ITEMS if rate_table_items is None else rate_table_items
        )
        self.mock_bindings = (
            DEFAULT_MOCK_BINDINGS if mock_bindings is None else mock_bindings
        )

    def estimate(
        self,
        product: ProductCandidate,
        profile: RecUserProfile,
        *,
        plan_code: str = "standard",
        coverage_amount: float | None = None,
        deductible: float | None = None,
    ) -> PremiumEstimate:
        effective_coverage = coverage_amount or profile.coverage_amount
        effective_deductible = (
            product.deductible_default if deductible is None else deductible
        )

        # Tier 1: 精确结构化费率表匹配
        tier1 = self._try_rate_table(
            product=product,
            profile=profile,
            plan_code=plan_code,
            coverage_amount=effective_coverage,
            deductible=effective_deductible,
        )
        if tier1 is not None:
            return tier1

        # Tier 2: 适配器 Mock 因子试算
        tier2 = self._try_mock_adapter(
            product=product,
            profile=profile,
            plan_code=plan_code,
            coverage_amount=effective_coverage,
            deductible=effective_deductible,
        )
        if tier2 is not None:
            return tier2

        # Tier 3: 基于 product_features 的精算近似估算（覆盖全部 50 款产品）
        return self._estimate_by_formula(
            product=product,
            profile=profile,
            plan_code=plan_code,
            coverage_amount=effective_coverage,
            deductible=effective_deductible,
        )

    def _try_rate_table(
        self,
        product: ProductCandidate,
        profile: RecUserProfile,
        plan_code: str,
        coverage_amount: float | None,
        deductible: float,
    ) -> PremiumEstimate | None:
        items = self.rate_table_items.get(product.id) or []
        ss_flag = "yes" if profile.has_social_security else "no"
        occ_str = str(profile.occupation_class)

        for item in items:
            item_plan = item.get("plan_code", "standard")
            if plan_code != "standard" and item_plan != plan_code:
                continue
            age_min = item.get("age_min")
            age_max = item.get("age_max")
            if age_min is not None and profile.age < age_min:
                continue
            if age_max is not None and profile.age > age_max:
                continue
            gender = item.get("gender", "all")
            if gender not in ("all", "unisex", profile.gender):
                continue
            ss = item.get("social_security", "all")
            if ss not in ("all", ss_flag):
                continue
            occ = item.get("occupation_class", "all")
            if occ not in ("all", occ_str):
                continue
            item_deductible = item.get("deductible")
            if item_deductible is not None and float(item_deductible) != float(deductible):
                continue

            rate_type = item.get("rate_type")
            if item.get("premium") is not None:
                return PremiumEstimate(
                    product_id=product.id,
                    annual_premium=round(float(item["premium"]), 2),
                    pricing_mode="table",
                    plan_code=item_plan,
                    breakdown={"rate_type": rate_type, "source": "rate_table_items"},
                )
            if item.get("rate_value") is not None and rate_type == "per_10000_sum_assured":
                sum_assured = coverage_amount or 500000.0
                premium = float(item["rate_value"]) * (sum_assured / 10000.0)
                return PremiumEstimate(
                    product_id=product.id,
                    annual_premium=round(premium, 2),
                    pricing_mode="table",
                    plan_code=item_plan,
                    breakdown={
                        "rate_type": rate_type,
                        "rate_per_10000": float(item["rate_value"]),
                        "coverage_amount": sum_assured,
                    },
                )
        return None

    def _try_mock_adapter(
        self,
        product: ProductCandidate,
        profile: RecUserProfile,
        plan_code: str,
        coverage_amount: float | None,
        deductible: float,
    ) -> PremiumEstimate | None:
        config = self.mock_bindings.get(product.id)
        if not config:
            return None

        base_map: dict[str, float] = config.get("base_premium_by_plan") or {}
        if not base_map:
            return None

        chosen_plan = plan_code if plan_code in base_map else (
            "classic" if "classic" in base_map else next(iter(base_map))
        )
        base = float(base_map[chosen_plan])

        age_factor = _match_age_factor(profile.age, config.get("age_factor") or [])
        gender_map = config.get("gender_factor") or {}
        gender_factor = float(gender_map.get(profile.gender, 1.0))

        ss_map = config.get("social_security_factor") or {}
        ss_key = "yes" if profile.has_social_security else "no"
        ss_factor = float(ss_map.get(ss_key, 1.0))

        deductible_map = config.get("deductible_factor") or {}
        deductible_key = str(int(deductible))
        deductible_factor = float(deductible_map.get(deductible_key, 1.0))

        occ_map = config.get("occupation_factor") or {}
        occ_factor = _match_range_key(profile.occupation_class, occ_map) if occ_map else 1.0

        cov_factor = 1.0
        if config.get("coverage_factor_enabled") and coverage_amount:
            cov_unit = float(config.get("coverage_unit") or 500000.0)
            if cov_unit > 0:
                cov_factor = max(0.5, float(coverage_amount) / cov_unit)

        premium = base * age_factor * gender_factor * ss_factor * deductible_factor * occ_factor * cov_factor
        return PremiumEstimate(
            product_id=product.id,
            annual_premium=round(premium, 2),
            pricing_mode="mock_adapter",
            plan_code=chosen_plan,
            breakdown={
                "base_premium": base,
                "age_factor": age_factor,
                "gender_factor": gender_factor,
                "social_security_factor": ss_factor,
                "deductible_factor": deductible_factor,
                "occupation_factor": occ_factor,
                "coverage_factor": round(cov_factor, 3),
            },
        )

    def _estimate_by_formula(
        self,
        product: ProductCandidate,
        profile: RecUserProfile,
        plan_code: str,
        coverage_amount: float | None,
        deductible: float,
    ) -> PremiumEstimate:
        min_p = max(float(product.min_premium or 100.0), 1.0)
        span_age = max(product.max_age - product.min_age, 1)
        clamped_age = min(max(profile.age, product.min_age), product.max_age)
        age_ratio = (clamped_age - product.min_age) / span_age

        if product.max_premium is not None and product.max_premium > min_p:
            max_p = float(product.max_premium)
            # 医疗/重疾随年龄呈指数攀升，意外相对平缓
            exponent = 1.65 if product.category in ("medical", "critical_illness") else 1.15
            base = min_p + (max_p - min_p) * (age_ratio ** exponent)
        else:
            # 寿险等未封顶产品：基于起投门槛与年龄风险因子估算
            base = min_p * (1.0 + 0.35 * age_ratio) * product.base_rate_factor

        ss_factor = 1.25 if (product.category == "medical" and not profile.has_social_security) else 1.0
        gender_factor = 1.06 if (
            product.category in ("critical_illness", "life") and profile.gender == "male"
        ) else 1.0
        occ_factor = 1.18 if (
            product.category == "accident" and profile.occupation_class >= 5
        ) else 1.0
        deductible_factor = 1.15 if (
            product.category == "medical" and deductible == 0 and product.deductible_default > 0
        ) else 1.0

        cov_factor = 1.0
        if coverage_amount and product.category in ("critical_illness", "life", "accident"):
            default_unit = 500000.0
            cov_factor = max(0.6, min(2.5, coverage_amount / default_unit))

        premium = max(min_p, base * ss_factor * gender_factor * occ_factor * deductible_factor * cov_factor)
        return PremiumEstimate(
            product_id=product.id,
            annual_premium=round(premium, 2),
            pricing_mode="estimated_formula",
            plan_code=plan_code,
            breakdown={
                "age_ratio": round(age_ratio, 3),
                "ss_factor": ss_factor,
                "gender_factor": gender_factor,
                "occ_factor": occ_factor,
                "deductible_factor": deductible_factor,
                "coverage_factor": round(cov_factor, 3),
            },
        )
