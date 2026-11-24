from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from app.recsys.schemas import PremiumEstimate, ProductCandidate, RecUserProfile

_PARSED_RATE_TABLES_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "parsed" / "rate_tables.json"
)
_ESTIMATED_DISCLAIMER = (
    "该产品暂缺官方精算费率表，当前金额为基于保费区间的参考估算，实际价格以投保页核保为准"
)
_PAYMENT_PERIOD_PRIORITY = {
    "20y": 0,
    "15y": 1,
    "10y": 2,
    "5y": 3,
    "3y": 4,
    "annual": 5,
    "single": 6,
}


@lru_cache(maxsize=1)
def load_parsed_rate_tables() -> dict[int, list[dict[str, Any]]]:
    """加载从官方精算 PDF 解析得到的 11/12 款真实产品结构化费率表。"""
    if not _PARSED_RATE_TABLES_PATH.exists():
        return {}
    raw = json.loads(_PARSED_RATE_TABLES_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}
    return {int(pid): list(rows) for pid, rows in raw.items() if isinstance(rows, list)}


def _occupation_matches(rule_occ: str, user_occ: int) -> bool:
    """判断费率条目职业条件（支持 'all'、'1-4' 区间或单值 '1'）是否覆盖当前职业类别。"""
    cleaned = str(rule_occ or "all").strip().lower()
    if cleaned in ("all", "*", ""):
        return True
    if "-" in cleaned:
        parts = cleaned.split("-", maxsplit=1)
        if parts[0].isdigit() and parts[1].isdigit():
            return int(parts[0]) <= user_occ <= int(parts[1])
    if cleaned.isdigit():
        return int(cleaned) == user_occ
    return False


class PremiumEstimator:
    """诚实二级保费引擎：官方精算费率表直查（11/12 款） -> 缺失费率表降级估算（明示免责）。"""

    def __init__(
        self,
        rate_table_items: dict[int, list[dict[str, Any]]] | None = None,
    ) -> None:
        self.rate_table_items = (
            load_parsed_rate_tables() if rate_table_items is None else rate_table_items
        )

    def estimate(
        self,
        product: ProductCandidate,
        profile: RecUserProfile,
        *,
        plan_code: str = "standard",
        coverage_amount: float | None = None,
        deductible: float | None = None,
        payment_period: str | None = None,
    ) -> PremiumEstimate:
        effective_coverage = coverage_amount if coverage_amount is not None else profile.coverage_amount
        effective_deductible = (
            product.deductible_default if deductible is None else deductible
        )

        # Tier 1: 官方结构化费率表精确匹配
        tier1 = self._try_rate_table(
            product=product,
            profile=profile,
            plan_code=plan_code,
            coverage_amount=effective_coverage,
            deductible=effective_deductible,
            payment_period=payment_period,
        )
        if tier1 is not None:
            return tier1

        # Tier 2: 仅对缺失官方费率表（如 ID=1 达尔文宝贝计划12号）做区间近似估算并明确标示
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
        payment_period: str | None,
    ) -> PremiumEstimate | None:
        items = self.rate_table_items.get(product.id) or []
        if not items:
            return None

        ss_flag = "yes" if profile.has_social_security else "no"
        requested_plan = (plan_code or "standard").strip().lower()
        requested_pay = payment_period.strip().lower() if payment_period else None

        candidates: list[tuple[tuple[int, int, int], dict[str, Any]]] = []
        for item in items:
            item_plan = str(item.get("plan_code") or "standard").strip().lower()
            if item_plan != requested_plan:
                continue

            age_min = item.get("age_min")
            age_max = item.get("age_max")
            if age_min is not None and profile.age < int(age_min):
                continue
            if age_max is not None and profile.age > int(age_max):
                continue

            gender = str(item.get("gender") or "all").strip().lower()
            if gender not in ("all", "unisex", profile.gender):
                continue

            ss = str(item.get("social_security") or "all").strip().lower()
            if ss not in ("all", ss_flag):
                continue

            occ = str(item.get("occupation_class") or "all").strip()
            if not _occupation_matches(occ, profile.occupation_class):
                continue

            item_pay = str(item.get("payment_period") or "").strip().lower()
            if requested_pay is not None and item_pay and item_pay != requested_pay:
                continue

            item_deductible = item.get("deductible")
            if item_deductible is not None:
                if float(item_deductible) != float(deductible):
                    deductible_rank = 2
                    continue
                deductible_rank = 0
            else:
                deductible_rank = 1

            pay_rank = _PAYMENT_PERIOD_PRIORITY.get(item_pay, 99)
            ss_rank = 0 if ss == ss_flag else 1
            candidates.append(((deductible_rank, pay_rank, ss_rank), item))

        if not candidates:
            return None

        candidates.sort(key=lambda pair: pair[0])
        matched = candidates[0][1]
        rate_type = str(matched.get("rate_type") or "")
        item_plan = str(matched.get("plan_code") or requested_plan)
        source_meta = {
            "rate_type": rate_type,
            "source_table": matched.get("source_table") or "official_rate_table",
            "source_page": matched.get("source_page"),
            "payment_period": matched.get("payment_period"),
            "coverage_period": matched.get("coverage_period"),
        }

        if rate_type == "per_1000_sum_assured" and matched.get("rate_value") is not None:
            sum_assured = float(coverage_amount or matched.get("coverage_amount") or 100000.0)
            rate_val = float(matched["rate_value"])
            premium = rate_val * (sum_assured / 1000.0)
            return PremiumEstimate(
                product_id=product.id,
                annual_premium=round(premium, 2),
                pricing_mode="table",
                plan_code=item_plan,
                is_estimated=False,
                pricing_disclaimer="",
                breakdown={
                    **source_meta,
                    "rate_per_1000": rate_val,
                    "coverage_amount": sum_assured,
                },
            )

        if rate_type == "per_10000_sum_assured" and matched.get("rate_value") is not None:
            sum_assured = float(coverage_amount or matched.get("coverage_amount") or 100000.0)
            rate_val = float(matched["rate_value"])
            premium = rate_val * (sum_assured / 10000.0)
            return PremiumEstimate(
                product_id=product.id,
                annual_premium=round(premium, 2),
                pricing_mode="table",
                plan_code=item_plan,
                is_estimated=False,
                pricing_disclaimer="",
                breakdown={
                    **source_meta,
                    "rate_per_10000": rate_val,
                    "coverage_amount": sum_assured,
                },
            )

        if rate_type == "per_1000_premium_sum_assured" and matched.get("rate_value") is not None:
            min_prem = float(matched.get("premium") or product.min_premium or 5000.0)
            sum_per_1000 = float(matched["rate_value"])
            if coverage_amount is not None and float(coverage_amount) > 0 and sum_per_1000 > 0:
                premium = max(min_prem, float(coverage_amount) / sum_per_1000 * 1000.0)
            else:
                premium = min_prem
            basic_sum_assured = round((premium / 1000.0) * sum_per_1000, 2)
            return PremiumEstimate(
                product_id=product.id,
                annual_premium=round(premium, 2),
                pricing_mode="table",
                plan_code=item_plan,
                is_estimated=False,
                pricing_disclaimer="",
                breakdown={
                    **source_meta,
                    "min_annual_premium": min_prem,
                    "sum_assured_per_1000_premium": sum_per_1000,
                    "basic_sum_assured": basic_sum_assured,
                },
            )

        if matched.get("premium") is not None:
            return PremiumEstimate(
                product_id=product.id,
                annual_premium=round(float(matched["premium"]), 2),
                pricing_mode="table",
                plan_code=item_plan,
                is_estimated=False,
                pricing_disclaimer="",
                breakdown={
                    **source_meta,
                    "coverage_amount": matched.get("coverage_amount"),
                    "deductible": matched.get("deductible"),
                },
            )

        return None

    def _estimate_by_formula(
        self,
        product: ProductCandidate,
        profile: RecUserProfile,
        plan_code: str,
        coverage_amount: float | None,
        deductible: float,
    ) -> PremiumEstimate:
        """降级估算公式：仅用于官方目录未提供费率表 PDF 的产品（并明确标示估算免责）。"""
        min_p = max(float(product.min_premium or 100.0), 1.0)
        span_age = max(product.max_age - product.min_age, 1)
        clamped_age = min(max(profile.age, product.min_age), product.max_age)
        age_ratio = (clamped_age - product.min_age) / span_age

        if product.max_premium is not None and product.max_premium > min_p:
            max_p = float(product.max_premium)
            exponent = 1.65 if product.category in ("medical", "critical_illness") else 1.15
            base = min_p + (max_p - min_p) * (age_ratio ** exponent)
        else:
            base = min_p * (1.0 + 0.35 * age_ratio) * product.base_rate_factor

        ss_factor = 1.25 if (product.category == "medical" and not profile.has_social_security) else 1.0
        gender_factor = (
            1.06 if (product.category in ("critical_illness", "life") and profile.gender == "male") else 1.0
        )
        occ_factor = 1.18 if (product.category == "accident" and profile.occupation_class >= 5) else 1.0
        deductible_factor = (
            1.15 if (product.category == "medical" and deductible == 0 and product.deductible_default > 0) else 1.0
        )

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
            is_estimated=True,
            pricing_disclaimer=_ESTIMATED_DISCLAIMER,
            breakdown={
                "age_ratio": round(age_ratio, 3),
                "ss_factor": ss_factor,
                "gender_factor": gender_factor,
                "occ_factor": occ_factor,
                "deductible_factor": deductible_factor,
                "coverage_factor": round(cov_factor, 3),
            },
        )
