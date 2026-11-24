from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.insurance_plan.models import Policy
from app.modules.policy.exceptions import PolicyNotFoundError
from app.modules.policy.repository import PolicyRepository
from app.modules.policy.schemas import (
    PolicyListResponse,
    PolicyProductSummary,
    PolicyResponse,
)
from app.modules.product.models import Product


class PolicyService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repository = PolicyRepository(session)

    @staticmethod
    def _to_response(policy: Policy, product: Product) -> PolicyResponse:
        return PolicyResponse(
            id=policy.id,
            user_id=policy.user_id,
            product_id=policy.product_id,
            policy_number=policy.policy_number,
            application_no=policy.application_no,
            status=policy.status,
            effective_at=policy.effective_at,
            expires_at=policy.expires_at,
            holder_name=policy.holder_name,
            insured_name=policy.insured_name,
            insured_id_no_masked=policy.insured_id_no_masked,
            insured_phone=policy.insured_phone,
            coverage_amount=policy.coverage_amount,
            premium_amount=policy.premium_amount,
            payment_frequency=policy.payment_frequency,
            beneficiary=policy.beneficiary,
            policy_snapshot=policy.policy_snapshot,
            coverage_snapshot=policy.coverage_snapshot,
            product=PolicyProductSummary(
                id=product.id,
                name=product.name,
                category=product.category,
                insurer=product.insurer,
                image_url=product.image_url,
            ),
        )

    async def list_policies(
        self,
        *,
        user_id: int,
        status: str | None = "active",
    ) -> PolicyListResponse:
        rows = await self.repository.list_owned(user_id, status=status)
        return PolicyListResponse(
            items=[self._to_response(policy, product) for policy, product in rows]
        )

    async def get_policy(self, *, user_id: int, policy_id: UUID) -> PolicyResponse:
        row = await self.repository.find_owned_by_id(policy_id, user_id)
        if row is None:
            raise PolicyNotFoundError
        policy, product = row
        return self._to_response(policy, product)
