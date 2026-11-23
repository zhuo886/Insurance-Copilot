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
    InsurancePlanResponse,
)

_DB_PRICING_MODE_MAP = {
    "table": "table",
    "mock_adapter": "mock_adapter",
    "demo_formula": "demo_formula",
    "insurer_api": "insurer_api",
    "local_fallback": "local_fallback",
    "estimated_formula": "local_fallback",
}


class InsurancePlanService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repository = InsurancePlanRepository(session)

    async def create_plan(
        self,
        *,
        user_id: int,
        plan_name: str,
        summary: str | None,
        insured_profile: dict[str, Any],
        annual_premium_budget: Decimal | None,
        items: list[InsurancePlanItemInput],
    ) -> InsurancePlanResponse:
        # 遵循 uk_insurance_plan_items_plan_product 唯一约束，按 product_id 去重
        deduped_items: list[InsurancePlanItemInput] = []
        seen_pids: set[int] = set()
        for item in items:
            if item.product_id in seen_pids:
                continue
            seen_pids.add(item.product_id)
            deduped_items.append(item)

        plan_id = uuid4()
        async with self.session.begin():
            plan = InsurancePlan(
                id=plan_id,
                user_id=user_id,
                plan_name=plan_name[:120],
                summary=summary,
                insured_profile=insured_profile,
                annual_premium_budget=annual_premium_budget,
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
            )
            for item, product in rows
        ]
        return InsurancePlanResponse(
            id=plan.id,
            user_id=plan.user_id,
            plan_name=plan.plan_name,
            summary=plan.summary,
            insured_profile=dict(plan.insured_profile or {}),
            annual_premium_budget=plan.annual_premium_budget,
            status=plan.status,
            created_at=plan.created_at,
            updated_at=plan.updated_at,
            items=item_responses,
        )

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
                policy = Policy(
                    id=uuid4(),
                    user_id=user_id,
                    product_id=item.product_id,
                    policy_number=f"POL-{now:%Y%m%d}-{uuid4().hex[:6].upper()}",
                    status="active",
                    effective_at=now,
                    expires_at=now + timedelta(days=365),
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
                "insured_name": insured_name,
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
