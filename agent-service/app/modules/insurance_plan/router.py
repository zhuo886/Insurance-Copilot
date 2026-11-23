from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header
from sqlalchemy.ext.asyncio import AsyncSession

from app.infra.database import get_session
from app.modules.insurance_plan.schemas import InsurancePlanResponse
from app.modules.insurance_plan.service import InsurancePlanService

router = APIRouter(prefix="/api/v1/insurance-plans", tags=["保险配置方案接口"])


def get_service(session: AsyncSession = Depends(get_session)) -> InsurancePlanService:
    return InsurancePlanService(session)


@router.get("/{plan_id}", response_model=InsurancePlanResponse)
async def get_insurance_plan(
    plan_id: UUID,
    user_id: Annotated[int, Header(alias="x-user-id")],
    service: InsurancePlanService = Depends(get_service),
) -> InsurancePlanResponse:
    """获取指定保险配置方案及其产品明细状态"""
    return await service.get_plan(user_id=user_id, plan_id=plan_id)


@router.patch("/{plan_id}/items/{item_id}/insured", response_model=InsurancePlanResponse)
async def mark_plan_item_insured(
    plan_id: UUID,
    item_id: UUID,
    user_id: Annotated[int, Header(alias="x-user-id")],
    service: InsurancePlanService = Depends(get_service),
) -> InsurancePlanResponse:
    """将方案内某项产品标记为已投保，并同步生成有效保单"""
    return await service.mark_item_insured(user_id=user_id, plan_id=plan_id, item_id=item_id)
