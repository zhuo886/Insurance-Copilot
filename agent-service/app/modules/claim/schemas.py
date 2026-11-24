from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, Field


class ClaimPolicySummary(BaseModel):
    id: UUID
    policy_number: str


class ClaimResponse(BaseModel):
    id: UUID
    user_id: int
    policy_id: UUID | None = None
    claim_number: str | None = None
    claim_type: str
    amount: Decimal | None = None
    description: str | None = None
    status: str
    status_message: str
    policy: ClaimPolicySummary | None = None
    submitted_at: datetime
    updated_at: datetime


class ClaimListResponse(BaseModel):
    items: list[ClaimResponse] = Field(default_factory=list)


class ClaimGuideStep(BaseModel):
    title: str
    description: str


class ClaimGuideMaterial(BaseModel):
    name: str
    required: bool = True
    remark: str = ""


class ClaimGuideResponse(BaseModel):
    policy_id: UUID
    product_id: int
    product_name: str
    download_uri: str = "/api/v1/claim-guides/application-form"
    steps: list[ClaimGuideStep] = Field(default_factory=list)
    materials: list[ClaimGuideMaterial] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
