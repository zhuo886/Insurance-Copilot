from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.auth import get_current_user_id
from app.infra.database import get_session
from app.modules.order.schemas import CreateOrderRequest, OrderListResponse, OrderResponse
from app.modules.order.service import OrderService

router = APIRouter(prefix="/api/v1/orders", tags=["投保订单接口"])


def get_service(session: AsyncSession = Depends(get_session)) -> OrderService:
    return OrderService(session)


@router.get("", response_model=OrderListResponse)
async def list_orders(
    user_id: int = Depends(get_current_user_id),
    service: OrderService = Depends(get_service),
) -> OrderListResponse:
    """查询当前用户投保订单列表"""
    return await service.list_orders(user_id=user_id)


@router.post("", response_model=OrderResponse)
async def create_order(
    request: CreateOrderRequest,
    user_id: int = Depends(get_current_user_id),
    service: OrderService = Depends(get_service),
) -> OrderResponse:
    """创建投保订单（关联保费试算报价并支持 idempotency_key 幂等提交）"""
    return await service.create_order(user_id=user_id, request=request)


@router.get("/{order_id}", response_model=OrderResponse)
async def get_order_detail(
    order_id: UUID,
    user_id: int = Depends(get_current_user_id),
    service: OrderService = Depends(get_service),
) -> OrderResponse:
    """查询指定投保订单详情"""
    return await service.get_order(user_id=user_id, order_id=order_id)


@router.post("/{order_id}/cancel", response_model=OrderResponse)
async def cancel_order(
    order_id: UUID,
    user_id: int = Depends(get_current_user_id),
    service: OrderService = Depends(get_service),
) -> OrderResponse:
    """取消指定投保订单"""
    return await service.cancel_order(user_id=user_id, order_id=order_id)
