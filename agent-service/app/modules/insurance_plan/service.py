from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.insurance_plan.exceptions import (
    InsurancePlanItemNotFoundError,
    InsurancePlanNotFoundError,
)
from app.modules.insurance_plan.models import (
    InsurancePlan,
    InsurancePlanItem,
    Policy,
    PremiumQuote,
)
from app.modules.insurance_plan.repository import InsurancePlanRepository
from app.modules.insurance_plan.schemas import (
    InsurancePlanItemInput,
    InsurancePlanItemResponse,
    InsurancePlanListResponse,
    InsurancePlanResponse,
    PlanProductSummary,
    SaveCurrentPlanRequest,
)
from app.modules.product.models import Product

_DB_PRICING_MODE_MAP = {
    "table": "table",
    "mock_adapter": "mock_adapter",
    "demo_formula": "demo_formula",
    "insurer_api": "insurer_api",
    "local_fallback": "local_fallback",
    "estimated_formula": "local_fallback",
}

_DEFAULT_COVERAGE_BY_CATEGORY = {
    "medical": Decimal("1000000.00"),
    "critical_illness": Decimal("300000.00"),
    "accident": Decimal("500000.00"),
    "life": Decimal("500000.00"),
}


class InsurancePlanService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repository = InsurancePlanRepository(session)

    @staticmethod
    def _dedup_items(items: list[InsurancePlanItemInput]) -> list[InsurancePlanItemInput]:
        deduped: list[InsurancePlanItemInput] = []
        seen_pids: set[int] = set()
        for item in items:
            if item.product_id in seen_pids:
                continue
            seen_pids.add(item.product_id)
            deduped.append(item)
        return deduped

    @staticmethod
    def _build_plan_response(
        plan: InsurancePlan,
        rows: list[tuple[InsurancePlanItem, Product]],
    ) -> InsurancePlanResponse:
        item_responses = [
            InsurancePlanItemResponse(
                id=item.id,
                plan_id=item.plan_id,
                product_id=item.product_id,
                name=product.name,
                category=product.category,
                image_url=product.image_url,
                description=product.description,
                highlights=list(product.highlights or []),
                priority=item.priority,
                recommendation_reason=item.recommendation_reason,
                annual_premium_budget=item.annual_premium_budget,
                status=item.status,
                product=PlanProductSummary(
                    id=product.id,
                    name=product.name,
                    category=product.category,
                    insurer=product.insurer,
                    image_url=product.image_url,
                ),
            )
            for item, product in rows
        ]
        return InsurancePlanResponse(
            id=plan.id,
            user_id=plan.user_id,
            plan_name=plan.plan_name,
            summary=plan.summary,
            budget_note=getattr(plan, "budget_note", None),
            insured_profile=dict(plan.insured_profile or {}),
            annual_premium_budget=plan.annual_premium_budget,
            version=int(getattr(plan, "version", 1) or 1),
            status=plan.status,
            created_at=plan.created_at,
            updated_at=plan.updated_at,
            items=item_responses,
        )

    async def list_plans(self, *, user_id: int) -> InsurancePlanListResponse:
        plans = await self.repository.list_owned_plans(user_id)
        responses: list[InsurancePlanResponse] = []
        for plan in plans:
            rows = await self.repository.list_items_with_products(plan.id)
            responses.append(self._build_plan_response(plan, rows))
        return InsurancePlanListResponse(items=responses)

    async def get_current_plan(self, *, user_id: int) -> InsurancePlanResponse:
        plan = await self.repository.find_current_plan(user_id)
        if plan is None:
            raise InsurancePlanNotFoundError
        rows = await self.repository.list_items_with_products(plan.id)
        return self._build_plan_response(plan, rows)

    async def save_current_plan(
        self,
        *,
        user_id: int,
        request: SaveCurrentPlanRequest,
    ) -> InsurancePlanResponse:
        deduped_items = self._dedup_items(request.items)
        inferred_budget = request.annual_premium_budget
        if inferred_budget is None:
            raw_profile_budget = (request.insured_profile or {}).get("budget") or (
                request.insured_profile or {}
            ).get("annual_budget")
            if raw_profile_budget is not None:
                inferred_budget = Decimal(str(raw_profile_budget))
            else:
                item_sum = sum(
                    (it.annual_premium_budget for it in deduped_items if it.annual_premium_budget is not None),
                    Decimal("0"),
                )
                inferred_budget = item_sum if item_sum > 0 else None

        async with self.session.begin():
            plan = await self.repository.find_current_plan(user_id)
            if plan is None:
                plan = InsurancePlan(
                    id=uuid4(),
                    user_id=user_id,
                    plan_name=request.plan_name[:120],
                    summary=request.summary,
                    budget_note=request.budget_note,
                    insured_profile=dict(request.insured_profile or {}),
                    annual_premium_budget=inferred_budget,
                    version=1,
                    status="uninsured",
                )
                await self.repository.add_plan(plan)
            else:
                plan.plan_name = request.plan_name[:120]
                plan.summary = request.summary
                plan.budget_note = request.budget_note
                plan.insured_profile = dict(request.insured_profile or {})
                plan.annual_premium_budget = inferred_budget
                plan.version = int(getattr(plan, "version", 1) or 1) + 1
                plan.status = "uninsured"
                await self.repository.delete_plan_items(plan.id)

            db_items = [
                InsurancePlanItem(
                    id=uuid4(),
                    plan_id=plan.id,
                    product_id=item.product_id,
                    priority=max(1, item.priority or idx),
                    recommendation_reason=item.recommendation_reason,
                    annual_premium_budget=item.annual_premium_budget,
                    status="uninsured",
                )
                for idx, item in enumerate(deduped_items, start=1)
            ]
            await self.repository.add_items(db_items)

        return await self.get_plan(user_id=user_id, plan_id=plan.id)

    async def create_plan(
        self,
        *,
        user_id: int,
        plan_name: str,
        summary: str | None,
        insured_profile: dict[str, Any],
        annual_premium_budget: Decimal | None,
        items: list[InsurancePlanItemInput],
        budget_note: str | None = None,
    ) -> InsurancePlanResponse:
        deduped_items = self._dedup_items(items)
        plan_id = uuid4()
        async with self.session.begin():
            plan = InsurancePlan(
                id=plan_id,
                user_id=user_id,
                plan_name=plan_name[:120],
                summary=summary,
                budget_note=budget_note,
                insured_profile=insured_profile,
                annual_premium_budget=annual_premium_budget,
                version=1,
                status="uninsured",
            )
            await self.repository.add_plan(plan)

            db_items = [
                InsurancePlanItem(
                    id=uuid4(),
                    plan_id=plan_id,
                    product_id=item.product_id,
                    priority=max(1, item.priority or idx),
                    recommendation_reason=item.recommendation_reason,
                    annual_premium_budget=item.annual_premium_budget,
                    status="uninsured",
                )
                for idx, item in enumerate(deduped_items, start=1)
            ]
            await self.repository.add_items(db_items)

        return await self.get_plan(user_id=user_id, plan_id=plan_id)

    async def get_plan(self, *, user_id: int, plan_id: UUID) -> InsurancePlanResponse:
        plan = await self.repository.find_owned_plan(plan_id, user_id)
        if plan is None:
            raise InsurancePlanNotFoundError
        rows = await self.repository.list_items_with_products(plan_id)
        return self._build_plan_response(plan, rows)

    async def mark_item_insured(
        self,
        *,
        user_id: int,
        plan_id: UUID,
        item_id: UUID,
    ) -> InsurancePlanResponse:
        async with self.session.begin():
            plan = await self.repository.find_owned_plan(plan_id, user_id)
            if plan is None:
                raise InsurancePlanNotFoundError

            item = await self.repository.find_item(plan_id, item_id)
            if item is None:
                raise InsurancePlanItemNotFoundError

            if item.status != "insured":
                item.status = "insured"
                now = datetime.now(timezone.utc)
                product = await self.repository.find_product(item.product_id)
                display_name = await self.repository.get_user_display_name(user_id)
                profile = dict(plan.insured_profile or {})
                person_name = str(profile.get("name") or display_name)
                cat = product.category if product else "medical"
                cov_amt = _DEFAULT_COVERAGE_BY_CATEGORY.get(cat, Decimal("500000.00"))
                prem_amt = item.annual_premium_budget or (product.min_premium if product else None)

                policy = Policy(
                    id=uuid4(),
                    user_id=user_id,
                    product_id=item.product_id,
                    policy_number=f"POL-{now:%Y%m%d}-{item.id.hex[:6].upper()}",
                    application_no=f"APP-{now:%Y%m%d}-{item.id.hex[:6].upper()}",
                    status="active",
                    effective_at=now,
                    expires_at=now + timedelta(days=365),
                    holder_name=person_name,
                    insured_name=person_name,
                    insured_id_no_masked="110101********1234",
                    insured_phone="138****8888",
                    coverage_amount=cov_amt,
                    premium_amount=prem_amt,
                    payment_frequency="annual",
                    beneficiary={"type": "法定受益人"},
                    policy_snapshot={
                        "plan_id": str(plan.id),
                        "plan_name": plan.plan_name,
                        "product_id": item.product_id,
                        "product_name": product.name if product else None,
                        "recommendation_reason": item.recommendation_reason,
                    },
                    coverage_snapshot={
                        "category": cat,
                        "insurer": product.insurer if product else None,
                        "highlights": list(product.highlights or []) if product else [],
                        "target_group": product.target_group if product else None,
                    },
                    paid_at=now,
                    issued_at=now,
                )
                await self.repository.add_policy(policy)

            all_items = await self.repository.list_plan_items(plan_id)
            if all_items and all(x.status == "insured" for x in all_items):
                plan.status = "insured"
            else:
                plan.status = "applying"

        return await self.get_plan(user_id=user_id, plan_id=plan_id)

    async def list_user_policies_and_claims(self, user_id: int) -> dict[str, Any]:
        insured_name = await self.repository.get_user_display_name(user_id)
        policy_rows = await self.repository.list_policies_with_products(user_id)
        claim_rows = await self.repository.list_claims_with_products(user_id)

        policies = [
            {
                "id": str(policy.id),
                "policy_number": policy.policy_number,
                "status": policy.status,
                "effective_at": policy.effective_at.isoformat(),
                "expires_at": policy.expires_at.isoformat() if policy.expires_at else None,
                "insured_name": policy.insured_name or insured_name,
                "coverage_amount": float(policy.coverage_amount) if policy.coverage_amount is not None else None,
                "premium_amount": float(policy.premium_amount) if policy.premium_amount is not None else None,
                "product_id": policy.product_id,
                "product": {
                    "id": product.id,
                    "name": product.name,
                    "category": product.category,
                    "image_url": product.image_url,
                },
            }
            for policy, product in policy_rows
        ]

        claims = [
            {
                "id": str(claim.id),
                "claim_number": claim.claim_number,
                "policy_id": str(claim.policy_id) if claim.policy_id else None,
                "product_id": claim.product_id,
                "product_name": product.name if product else None,
                "claim_type": claim.claim_type,
                "amount": float(claim.amount) if claim.amount is not None else None,
                "description": claim.description,
                "status": claim.status,
                "submitted_at": claim.submitted_at.isoformat(),
                "updated_at": claim.updated_at.isoformat(),
            }
            for claim, product in claim_rows
        ]
        return {"policies": policies, "claims": claims}

    async def save_premium_quote(
        self,
        *,
        user_id: int | None,
        product_id: int,
        request_params: dict[str, Any],
        result: dict[str, Any],
        total_premium: float,
        pricing_mode: str,
    ) -> UUID:
        mapped_mode = _DB_PRICING_MODE_MAP.get(pricing_mode, "local_fallback")
        quote_id = uuid4()
        async with self.session.begin():
            quote = PremiumQuote(
                id=quote_id,
                user_id=user_id,
                product_id=product_id,
                request_params=request_params,
                result=result,
                total_premium=Decimal(str(round(total_premium, 2))),
                pricing_mode=mapped_mode,
                expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
            )
            await self.repository.add_quote(quote)
        return quote_id
