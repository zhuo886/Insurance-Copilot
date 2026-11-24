from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.auth import get_current_user_id
from app.infra.database import get_session
from app.modules.insurance_plan.schemas import (
    InsurancePlanListResponse,
    InsurancePlanResponse,
    SaveCurrentPlanRequest,
)
from app.modules.insurance_plan.service import InsurancePlanService

router = APIRouter(prefix="/api/v1/insurance-plans", tags=["保险配置方案接口"])


def get_service(session: AsyncSession = Depends(get_session)) -> InsurancePlanService:
    return InsurancePlanService(session)


@router.get("", response_model=InsurancePlanListResponse)
async def list_insurance_plans(
    user_id: int = Depends(get_current_user_id),
    service: InsurancePlanService = Depends(get_service),
) -> InsurancePlanListResponse:
    """查询当前用户保存的全部保险方案"""
    return await service.list_plans(user_id=user_id)


@router.get("/current", response_model=InsurancePlanResponse)
async def get_current_insurance_plan(
    user_id: int = Depends(get_current_user_id),
    service: InsurancePlanService = Depends(get_service),
) -> InsurancePlanResponse:
    """查询当前用户最近且尚未全部投保的保险方案"""
    return await service.get_current_plan(user_id=user_id)


@router.put("/current", response_model=InsurancePlanResponse)
async def save_current_insurance_plan(
    request: SaveCurrentPlanRequest,
    user_id: int = Depends(get_current_user_id),
    service: InsurancePlanService = Depends(get_service),
) -> InsurancePlanResponse:
    """创建或更新当前用户的保险方案及方案项"""
    return await service.save_current_plan(user_id=user_id, request=request)


@router.get("/{plan_id}", response_model=InsurancePlanResponse)
async def get_insurance_plan(
    plan_id: UUID,
    user_id: int = Depends(get_current_user_id),
    service: InsurancePlanService = Depends(get_service),
) -> InsurancePlanResponse:
    """获取指定保险配置方案及其产品明细状态"""
    return await service.get_plan(user_id=user_id, plan_id=plan_id)


@router.patch("/{plan_id}/items/{item_id}/insured", response_model=InsurancePlanResponse)
async def mark_plan_item_insured(
    plan_id: UUID,
    item_id: UUID,
    user_id: int = Depends(get_current_user_id),
    service: InsurancePlanService = Depends(get_service),
) -> InsurancePlanResponse:
    """将方案内某项产品标记为已投保，并同步生成完整快照有效保单"""
    return await service.mark_item_insured(user_id=user_id, plan_id=plan_id, item_id=item_id)
