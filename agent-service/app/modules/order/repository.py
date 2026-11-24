from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.insurance_plan.models import PremiumQuote
from app.modules.order.models import Order


class OrderRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list_owned(self, user_id: int) -> list[Order]:
        rows = await self.session.scalars(
            select(Order)
            .where(Order.user_id == user_id)
            .order_by(Order.created_at.desc())
        )
        return list(rows.all())

    async def find_owned_by_id(self, order_id: UUID, user_id: int) -> Order | None:
        return await self.session.scalar(
            select(Order).where(Order.id == order_id, Order.user_id == user_id)
        )

    async def find_by_idempotency_key(
        self,
        user_id: int,
        idempotency_key: str,
    ) -> Order | None:
        return await self.session.scalar(
            select(Order).where(
                Order.user_id == user_id,
                Order.idempotency_key == idempotency_key,
            )
        )

    async def find_quotes_by_ids(self, quote_ids: list[UUID]) -> list[PremiumQuote]:
        if not quote_ids:
            return []
        rows = await self.session.scalars(
            select(PremiumQuote).where(PremiumQuote.id.in_(quote_ids))
        )
        return list(rows.all())

    async def add_order(self, order: Order) -> Order:
        self.session.add(order)
        await self.session.flush()
        return order
