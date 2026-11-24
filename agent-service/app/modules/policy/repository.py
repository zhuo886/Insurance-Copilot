from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.insurance_plan.models import Policy
from app.modules.product.models import Product


class PolicyRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list_owned(
        self,
        user_id: int,
        status: str | None = "active",
    ) -> list[tuple[Policy, Product]]:
        stmt = (
            select(Policy, Product)
            .join(Product, Product.id == Policy.product_id)
            .where(Policy.user_id == user_id)
        )
        if status and status.lower() != "all":
            stmt = stmt.where(Policy.status == status)
        rows = await self.session.execute(stmt.order_by(Policy.effective_at.desc()))
        return [(row[0], row[1]) for row in rows.all()]

    async def find_owned_by_id(
        self,
        policy_id: UUID,
        user_id: int,
    ) -> tuple[Policy, Product] | None:
        row = (
            await self.session.execute(
                select(Policy, Product)
                .join(Product, Product.id == Policy.product_id)
                .where(Policy.id == policy_id, Policy.user_id == user_id)
            )
        ).first()
        return (row[0], row[1]) if row is not None else None
