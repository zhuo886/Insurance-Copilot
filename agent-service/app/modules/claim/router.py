from uuid import UUID

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.auth import get_current_user_id
from app.infra.database import get_session
from app.modules.claim.schemas import ClaimGuideResponse, ClaimListResponse, ClaimResponse
from app.modules.claim.service import ClaimService

router = APIRouter(prefix="/api/v1/claims", tags=["理赔查询接口"])
claim_guide_router = APIRouter(prefix="/api/v1/claim-guides", tags=["理赔指引接口"])


def get_service(session: AsyncSession = Depends(get_session)) -> ClaimService:
    return ClaimService(session)


@router.get("", response_model=ClaimListResponse)
async def list_claims(
    policy_id: UUID | None = Query(default=None),
    status: str | None = Query(default=None),
    user_id: int = Depends(get_current_user_id),
    service: ClaimService = Depends(get_service),
) -> ClaimListResponse:
    """查询当前用户理赔记录列表"""
    return await service.list_claims(user_id=user_id, policy_id=policy_id, status=status)


@router.get("/{claim_id}", response_model=ClaimResponse)
async def get_claim_detail(
    claim_id: UUID,
    user_id: int = Depends(get_current_user_id),
    service: ClaimService = Depends(get_service),
) -> ClaimResponse:
    """查询当前用户指定理赔记录详情"""
    return await service.get_claim(user_id=user_id, claim_id=claim_id)


@claim_guide_router.get("/application-form")
async def download_application_form(
    service: ClaimService = Depends(get_service),
) -> FileResponse:
    """下载教学演示用标准理赔申请表 PDF"""
    path = service.get_application_form_path()
    return FileResponse(
        path=path,
        media_type="application/pdf",
        filename="claim-application-form.pdf",
    )


@claim_guide_router.get("/{policy_id}", response_model=ClaimGuideResponse)
async def get_claim_guide(
    policy_id: UUID,
    user_id: int = Depends(get_current_user_id),
    service: ClaimService = Depends(get_service),
) -> ClaimGuideResponse:
    """根据保单 ID 生成对应产品的理赔步骤、材料清单及申请表下载地址"""
    return await service.get_claim_guide(user_id=user_id, policy_id=policy_id)
