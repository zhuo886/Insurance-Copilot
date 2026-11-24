from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.auth import get_optional_user_id
from app.infra.database import get_session
from app.modules.premium.schemas import PremiumCalculateRequest, PremiumCalculateResponse
from app.modules.premium.service import PremiumService

router = APIRouter(prefix="/api/v1/premium", tags=["保费试算接口"])


@router.post("/calculate", response_model=PremiumCalculateResponse)
async def calculate_premium(
    req: PremiumCalculateRequest,
    user_id: int | None = Depends(get_optional_user_id),
    session: AsyncSession = Depends(get_session),
) -> PremiumCalculateResponse:
    """保费试算接口（优先命中官方 PDF 费率表，缺失表时降级估算并明示风险提示）"""
    service = PremiumService(session)
    return await service.calculate(req, user_id=user_id)
