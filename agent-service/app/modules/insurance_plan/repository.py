from typing import Any
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.insurance_plan.models import (
    Claim,
    InsurancePlan,
    InsurancePlanItem,
    Policy,
    PremiumQuote,
)
from app.modules.product.models import Product


class InsurancePlanRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def find_owned_plan(self, plan_id: UUID, user_id: int) -> InsurancePlan | None:
        return await self.session.scalar(
            select(InsurancePlan).where(
                InsurancePlan.id == plan_id,
                InsurancePlan.user_id == user_id,
            )
        )

    async def list_items_with_products(
        self,
        plan_id: UUID,
    ) -> list[tuple[InsurancePlanItem, Product]]:
        rows = await self.session.execute(
            select(InsurancePlanItem, Product)
            .join(Product, Product.id == InsurancePlanItem.product_id)
            .where(InsurancePlanItem.plan_id == plan_id)
            .order_by(InsurancePlanItem.priority.asc(), InsurancePlanItem.created_at.asc())
        )
        return [(row[0], row[1]) for row in rows.all()]

    async def find_item(self, plan_id: UUID, item_id: UUID) -> InsurancePlanItem | None:
        return await self.session.scalar(
            select(InsurancePlanItem).where(
                InsurancePlanItem.id == item_id,
                InsurancePlanItem.plan_id == plan_id,
            )
        )

    async def list_plan_items(self, plan_id: UUID) -> list[InsurancePlanItem]:
        rows = await self.session.scalars(
            select(InsurancePlanItem).where(InsurancePlanItem.plan_id == plan_id)
        )
        return list(rows.all())

    async def add_plan(self, plan: InsurancePlan) -> None:
        self.session.add(plan)

    async def add_items(self, items: list[InsurancePlanItem]) -> None:
        self.session.add_all(items)

    async def add_policy(self, policy: Policy) -> None:
        self.session.add(policy)

    async def add_quote(self, quote: PremiumQuote) -> None:
        self.session.add(quote)

    async def get_user_display_name(self, user_id: int) -> str:
        name = await self.session.scalar(
            text("SELECT COALESCE(display_name, username) FROM users WHERE id = :uid"),
            {"uid": user_id},
        )
        return str(name) if name else f"用户{user_id}"

    async def list_policies_with_products(
        self,
        user_id: int,
    ) -> list[tuple[Policy, Product]]:
        rows = await self.session.execute(
            select(Policy, Product)
            .join(Product, Product.id == Policy.product_id)
            .where(Policy.user_id == user_id)
            .order_by(Policy.effective_at.desc())
        )
        return [(row[0], row[1]) for row in rows.all()]

    async def list_claims_with_products(
        self,
        user_id: int,
    ) -> list[tuple[Claim, Product | None]]:
        rows = await self.session.execute(
            select(Claim, Product)
            .outerjoin(Product, Product.id == Claim.product_id)
            .where(Claim.user_id == user_id)
            .order_by(Claim.updated_at.desc())
        )
        return [(row[0], row[1]) for row in rows.all()]

    async def find_rate_table_items(self, product_ids: list[int] | None = None) -> dict[int, list[dict[str, Any]]]:
        sql = "SELECT product_id, plan_code, rate_type, age_min, age_max, gender, occupation_class, social_security, payment_period, coverage_period, coverage_amount, deductible, rate_value, premium FROM rate_table_items WHERE status = 'active'"
        params: dict[str, Any] = {}
        if product_ids:
            sql += " AND product_id = ANY(:pids)"
            params["pids"] = product_ids
        rows = (await self.session.execute(text(sql), params)).mappings().all()
        grouped: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            pid = int(row["product_id"])
            item = dict(row)
            for num_key in ("coverage_amount", "deductible", "rate_value", "premium"):
                if item.get(num_key) is not None:
                    item[num_key] = float(item[num_key])
            grouped.setdefault(pid, []).append(item)
        return grouped
