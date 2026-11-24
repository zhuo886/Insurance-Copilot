from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.insurance_plan.models import PremiumQuote
from app.modules.insurance_plan.repository import InsurancePlanRepository
from app.modules.insurance_plan.service import _DB_PRICING_MODE_MAP
from app.modules.premium.exceptions import (
    InvalidPricingParamsError,
    MissingPricingParamsError,
    PricingBindingNotFoundError,
)
from app.modules.premium.schemas import (
    PremiumBreakdownItem,
    PremiumCalculateRequest,
    PremiumCalculateResponse,
)
from app.modules.product.catalog_data import load_builtin_catalog
from app.modules.product.repository import ProductRepository
from app.recsys.pricing import PremiumEstimator
from app.recsys.schemas import ProductCandidate, RecUserProfile


def _extract_answer(answers: dict[str, Any], *paths: str, default: Any = None) -> Any:
    """兼容扁平点号 key（'insured.age'）与嵌套对象（{'insured': {'age': 30}}）。"""
    for path in paths:
        if path in answers and answers[path] is not None:
            return answers[path]
        if "." in path:
            cur: Any = answers
            ok = True
            for part in path.split("."):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    ok = False
                    break
            if ok and cur is not None:
                return cur
    return default


def _parse_social_security(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ("no", "false", "0", "无", "无社保", "无医保"):
        return False
    return True


def _parse_occupation_class(val: Any) -> int:
    if isinstance(val, int):
        return max(1, min(6, val))
    s = str(val or "1").strip()
    if "-" in s:
        tail = s.split("-")[-1]
        tail_digits = "".join(ch for ch in tail if ch.isdigit())
        if tail_digits:
            return max(1, min(6, int(tail_digits)))
    digits = "".join(ch for ch in s if ch.isdigit())
    return max(1, min(6, int(digits))) if digits else 1


def _parse_gender(val: Any) -> str:
    s = str(val or "male").strip().lower()
    if s in ("female", "f", "女"):
        return "female"
    return "male"


class PremiumService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.product_repo = ProductRepository(session)
        self.plan_repo = InsurancePlanRepository(session)

    async def calculate(
        self,
        req: PremiumCalculateRequest,
        *,
        user_id: int | None = None,
    ) -> PremiumCalculateResponse:
        if req.product_id is None or int(req.product_id) <= 0:
            raise InvalidPricingParamsError("product_id is required")

        product_id = int(req.product_id)
        row = await self.product_repo.find_with_feature_by_id(product_id)
        if row is None:
            raise PricingBindingNotFoundError("Pricing binding not found")
        product, feature = row

        answers = req.answers or {}
        raw_age = _extract_answer(answers, "insured.age", "age")
        if raw_age is None or str(raw_age).strip() == "":
            raise MissingPricingParamsError("Missing required parameter: insured.age")
        try:
            age = int(raw_age)
        except (TypeError, ValueError) as exc:
            raise InvalidPricingParamsError("insured.age 必须为整数") from exc

        gender = _parse_gender(_extract_answer(answers, "insured.gender", "gender", default="male"))
        occ_class = _parse_occupation_class(
            _extract_answer(answers, "insured.occupation_class", "occupation_class", default=1)
        )
        has_ss = _parse_social_security(
            _extract_answer(answers, "insured.social_security", "has_social_security", default="yes")
        )
        raw_plan_code = str(
            _extract_answer(answers, "plan_code", "coverage.plan_code", default="standard")
        ).strip()
        raw_cov = _extract_answer(answers, "coverage.coverage_amount", "coverage_amount")
        coverage_amount = float(raw_cov) if raw_cov not in (None, "") else None
        raw_ded = _extract_answer(answers, "coverage.deductible", "deductible")
        deductible = float(raw_ded) if raw_ded not in (None, "") else None
        raw_pay = _extract_answer(answers, "coverage.payment_period", "payment_period")
        payment_period = str(raw_pay).strip() if raw_pay not in (None, "") else None

        builtin_map = {p.id: p for p in load_builtin_catalog()}
        fb = builtin_map.get(product.id)
        candidate = ProductCandidate(
            id=product.id,
            name=product.name,
            clause_name=product.clause_name,
            category=product.category,  # type: ignore[arg-type]
            insurer=product.insurer,
            image_url=product.image_url,
            description=product.description or "",
            min_premium=float(product.min_premium or 0.0),
            max_premium=float(product.max_premium) if product.max_premium is not None else None,
            target_group=product.target_group or "",
            highlights=list(product.highlights or []),
            status=product.status,
            min_age=feature.min_age if feature else (fb.min_age if fb else 0),
            max_age=feature.max_age if feature else (fb.max_age if fb else 70),
            max_occupation_class=(
                feature.max_occupation_class
                if feature
                else (fb.max_occupation_class if fb else 4)
            ),
            waiting_period_days=(
                feature.waiting_period_days if feature else (fb.waiting_period_days if fb else 0)
            ),
            guarantee_renewal_years=(
                feature.guarantee_renewal_years
                if feature
                else (fb.guarantee_renewal_years if fb else None)
            ),
            deductible_default=(
                float(feature.deductible_default)
                if feature
                else (fb.deductible_default if fb else 0.0)
            ),
            has_clause=feature.has_clause if feature else True,
            is_featured=feature.is_featured if feature else True,
            coverage_tags=(
                list(feature.coverage_tags or []) if feature else (fb.coverage_tags if fb else [])
            ),
            health_disclosure_level=(
                feature.health_disclosure_level  # type: ignore[arg-type]
                if feature
                else (fb.health_disclosure_level if fb else "standard")
            ),
            insurer_tier=(
                feature.insurer_tier  # type: ignore[arg-type]
                if feature
                else (fb.insurer_tier if fb else "top")
            ),
            base_rate_factor=(
                float(feature.base_rate_factor) if feature else (fb.base_rate_factor if fb else 1.0)
            ),
        )

        if age < candidate.min_age or age > candidate.max_age:
            raise InvalidPricingParamsError(
                f"被保人年龄 {age} 周岁超出《{candidate.name}》承保范围（{candidate.min_age}-{candidate.max_age}周岁）"
            )
        if occ_class > candidate.max_occupation_class:
            raise InvalidPricingParamsError(
                f"职业类别 {occ_class} 类超出《{candidate.name}》承保上限（1-{candidate.max_occupation_class}类）"
            )

        try:
            db_tables = await self.plan_repo.find_rate_table_items([product_id])
        except Exception:
            db_tables = {}
        estimator = PremiumEstimator(rate_table_items=db_tables if db_tables else None)
        profile = RecUserProfile(
            age=age,
            gender=gender,  # type: ignore[arg-type]
            occupation_class=occ_class,
            has_social_security=has_ss,
            coverage_amount=coverage_amount,
        )

        # 若兼容旧模板传入 basic/classic/plus 且表内未单独定义，则映射至标准费率 plan_code
        effective_plan = raw_plan_code
        table_rows = estimator.rate_table_items.get(product_id) or []
        available_plans = {str(r.get("plan_code") or "standard").lower() for r in table_rows}
        if available_plans and effective_plan.lower() not in available_plans:
            if "standard" in available_plans:
                effective_plan = "standard"

        estimate = estimator.estimate(
            candidate,
            profile,
            plan_code=effective_plan,
            coverage_amount=coverage_amount,
            deductible=deductible,
            payment_period=payment_period,
        )

        annual_str = f"{estimate.annual_premium:.2f}"
        expires_at = datetime.now(timezone.utc) + timedelta(hours=24)
        warnings = [
            (
                estimate.pricing_disclaimer
                if estimate.is_estimated and estimate.pricing_disclaimer
                else "本试算结果依据内置官方精算费率表计算，最终保费以承保公司核保为准"
            ),
            "本页面仅作投保前保费试算参考，不构成正式保险合同要约",
        ]
        breakdown_items = [
            PremiumBreakdownItem(
                item="main",
                name=estimate.plan_code,
                premium=annual_str,
                source={
                    "mode": estimate.pricing_mode,
                    "is_estimated": estimate.is_estimated,
                    **estimate.breakdown,
                },
            )
        ]

        quote_id = uuid4()
        result_payload = {
            "quote_id": str(quote_id),
            "product_id": product_id,
            "plan_code": estimate.plan_code,
            "pricing_mode": estimate.pricing_mode,
            "annual_premium": annual_str,
            "payment_frequency": "annual",
            "breakdown": [b.model_dump() for b in breakdown_items],
            "warnings": warnings,
            "expires_at": expires_at.isoformat(),
        }

        async with self.session.begin():
            quote = PremiumQuote(
                id=quote_id,
                user_id=user_id,
                product_id=product_id,
                request_params={
                    "scenario": req.scenario,
                    "age": age,
                    "gender": gender,
                    "occupation_class": occ_class,
                    "has_social_security": has_ss,
                    "plan_code": estimate.plan_code,
                    "coverage_amount": coverage_amount,
                    "deductible": deductible,
                    "payment_period": payment_period,
                },
                result=result_payload,
                total_premium=Decimal(annual_str),
                pricing_mode=_DB_PRICING_MODE_MAP.get(estimate.pricing_mode, "local_fallback"),
                expires_at=expires_at,
            )
            await self.plan_repo.add_quote(quote)

        return PremiumCalculateResponse(
            quote_id=str(quote_id),
            product_id=product_id,
            plan_code=estimate.plan_code,
            pricing_mode=estimate.pricing_mode,
            annual_premium=annual_str,
            payment_frequency="annual",
            breakdown=breakdown_items,
            warnings=warnings,
            expires_at=expires_at.isoformat(),
        )
