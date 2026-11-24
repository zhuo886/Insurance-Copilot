from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.insurance_plan.models import Claim, Policy


class ClaimRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list_owned(
        self,
        user_id: int,
        *,
        policy_id: UUID | None = None,
        status: str | None = None,
    ) -> list[tuple[Claim, Policy | None]]:
        stmt = (
            select(Claim, Policy)
            .outerjoin(Policy, Policy.id == Claim.policy_id)
            .where(Claim.user_id == user_id)
        )
        if policy_id is not None:
            stmt = stmt.where(Claim.policy_id == policy_id)
        if status:
            stmt = stmt.where(Claim.status == status)
        rows = await self.session.execute(stmt.order_by(Claim.updated_at.desc()))
        return [(row[0], row[1]) for row in rows.all()]

    async def find_owned_by_id(
        self,
        claim_id: UUID,
        user_id: int,
    ) -> tuple[Claim, Policy | None] | None:
        row = (
            await self.session.execute(
                select(Claim, Policy)
                .outerjoin(Policy, Policy.id == Claim.policy_id)
                .where(Claim.id == claim_id, Claim.user_id == user_id)
            )
        ).first()
        return (row[0], row[1]) if row is not None else None
