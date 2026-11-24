from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class PolicyProductSummary(BaseModel):
    id: int
    name: str
    category: str
    insurer: str | None = None
    image_url: str | None = None


class PolicyResponse(BaseModel):
    id: UUID
    user_id: int
    product_id: int
    policy_number: str
    application_no: str | None = None
    status: str
    effective_at: datetime
    expires_at: datetime | None = None
    holder_name: str | None = None
    insured_name: str | None = None
    insured_id_no_masked: str | None = None
    insured_phone: str | None = None
    coverage_amount: Decimal | None = None
    premium_amount: Decimal | None = None
    payment_frequency: str | None = None
    beneficiary: dict[str, Any] | None = None
    policy_snapshot: dict[str, Any] | None = None
    coverage_snapshot: dict[str, Any] | None = None
    product: PolicyProductSummary | None = None


class PolicyListResponse(BaseModel):
    items: list[PolicyResponse] = Field(default_factory=list)
