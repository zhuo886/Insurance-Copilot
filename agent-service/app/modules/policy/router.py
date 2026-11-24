from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.auth import get_current_user_id
from app.infra.database import get_session
from app.modules.policy.schemas import PolicyListResponse, PolicyResponse
from app.modules.policy.service import PolicyService

router = APIRouter(prefix="/api/v1/policies", tags=["保单管理接口"])


def get_service(session: AsyncSession = Depends(get_session)) -> PolicyService:
    return PolicyService(session)


@router.get("", response_model=PolicyListResponse)
async def list_policies(
    status: str | None = Query(default="active"),
    user_id: int = Depends(get_current_user_id),
    service: PolicyService = Depends(get_service),
) -> PolicyListResponse:
    """查询当前用户保单列表（默认 active，传 all 查询全部）"""
    return await service.list_policies(user_id=user_id, status=status)


@router.get("/{policy_id}", response_model=PolicyResponse)
async def get_policy_detail(
    policy_id: UUID,
    user_id: int = Depends(get_current_user_id),
    service: PolicyService = Depends(get_service),
) -> PolicyResponse:
    """查询当前用户指定保单详情"""
    return await service.get_policy(user_id=user_id, policy_id=policy_id)
