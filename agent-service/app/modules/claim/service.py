from pathlib import Path
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.claim.exceptions import ApplicationFormNotFoundError, ClaimNotFoundError
from app.modules.claim.repository import ClaimRepository
from app.modules.claim.schemas import (
    ClaimGuideMaterial,
    ClaimGuideResponse,
    ClaimGuideStep,
    ClaimListResponse,
    ClaimPolicySummary,
    ClaimResponse,
)
from app.modules.insurance_plan.models import Claim, Policy
from app.modules.policy.exceptions import PolicyNotFoundError
from app.modules.policy.repository import PolicyRepository

APPLICATION_FORM_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "assets" / "claim-application-form.pdf"
)

_STATUS_MESSAGES = {
    "submitted": "已提交理赔申请，等待承保公司受理",
    "reviewing": "保险公司正在审核材料，可能要求补充资料",
    "approved": "理赔审核已通过，正在安排赔付款项",
    "rejected": "理赔申请未通过审核，可联系承保公司复核",
    "paid": "理赔款项已完成支付",
}

_CATEGORY_EXTRA_MATERIALS = {
    "medical": [
        ClaimGuideMaterial(
            name="住院病历、出院小结及医疗费用原始收据/明细清单",
            required=True,
            remark="医保已结算的需提供医保结算单，涉及院外特药需附处方与购药发票。",
        ),
    ],
    "critical_illness": [
        ClaimGuideMaterial(
            name="二级及以上公立医院专科医生出具的重疾诊断证明书与病理/检查报告",
            required=True,
            remark="需包含病理切片、影像学及实验室检验等确诊依据。",
        ),
    ],
    "accident": [
        ClaimGuideMaterial(
            name="意外事故证明（交通事故认定书/工伤认定/派出所报案回执）及门急诊病历",
            required=True,
            remark="若申请伤残赔付，需提供有资质鉴定机构出具的伤残等级鉴定报告。",
        ),
    ],
    "life": [
        ClaimGuideMaterial(
            name="受益人身份证明、法定受益人关系证明或死亡/户籍注销证明",
            required=True,
            remark="分红型或减保类申请需同步提供保单合同与银行账户信息。",
        ),
    ],
}


class ClaimService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repository = ClaimRepository(session)
        self.policy_repository = PolicyRepository(session)

    @staticmethod
    def _to_response(claim: Claim, policy: Policy | None) -> ClaimResponse:
        return ClaimResponse(
            id=claim.id,
            user_id=claim.user_id,
            policy_id=claim.policy_id,
            claim_number=claim.claim_number,
            claim_type=claim.claim_type,
            amount=claim.amount,
            description=claim.description,
            status=claim.status,
            status_message=_STATUS_MESSAGES.get(claim.status, "处理中"),
            policy=(
                ClaimPolicySummary(id=policy.id, policy_number=policy.policy_number)
                if policy is not None
                else None
            ),
            submitted_at=claim.submitted_at,
            updated_at=claim.updated_at,
        )

    async def list_claims(
        self,
        *,
        user_id: int,
        policy_id: UUID | None = None,
        status: str | None = None,
    ) -> ClaimListResponse:
        rows = await self.repository.list_owned(user_id, policy_id=policy_id, status=status)
        return ClaimListResponse(
            items=[self._to_response(claim, policy) for claim, policy in rows]
        )

    async def get_claim(self, *, user_id: int, claim_id: UUID) -> ClaimResponse:
        row = await self.repository.find_owned_by_id(claim_id, user_id)
        if row is None:
            raise ClaimNotFoundError
        claim, policy = row
        return self._to_response(claim, policy)

    @staticmethod
    def get_application_form_path() -> Path:
        if not APPLICATION_FORM_PATH.exists():
            raise ApplicationFormNotFoundError
        return APPLICATION_FORM_PATH

    async def get_claim_guide(
        self,
        *,
        user_id: int,
        policy_id: UUID,
    ) -> ClaimGuideResponse:
        row = await self.policy_repository.find_owned_by_id(policy_id, user_id)
        if row is None:
            raise PolicyNotFoundError
        policy, product = row

        steps = [
            ClaimGuideStep(
                title="1. 及时报案通知",
                description=f"请在保险事故发生后尽快联系承保公司（{product.insurer}）客服热线报案登记保单号 {policy.policy_number}。",
            ),
            ClaimGuideStep(
                title="2. 准备理赔材料",
                description="对照下方材料清单收集就诊病历、诊断证明、费用清单及申请表。",
            ),
            ClaimGuideStep(
                title="3. 提交审核与理赔打款",
                description="向承保公司提交完整理赔材料；审核通过后理赔金将直接打入指定银行账户。",
            ),
        ]

        base_materials = [
            ClaimGuideMaterial(
                name="理赔申请书",
                required=True,
                remark="可点击下载地址获取标准教学版理赔申请表参考填写。",
            ),
            ClaimGuideMaterial(
                name="被保人及申请人有效身份证件、银行卡复印件",
                required=True,
                remark="需确认账户姓名与被保人或受益人一致。",
            ),
        ]
        extra = _CATEGORY_EXTRA_MATERIALS.get(product.category, [])

        notes = [
            "本商城仅提供理赔指引与申请表模板，不直接受理、审核或支付理赔款项。",
            "具体等待期、免赔额、报案时效与材料要求请以对应产品合同条款及承保公司要求为准。",
        ]
        return ClaimGuideResponse(
            policy_id=policy.id,
            product_id=product.id,
            product_name=product.name,
            download_uri="/api/v1/claim-guides/application-form",
            steps=steps,
            materials=[*base_materials, *extra],
            notes=notes,
        )
