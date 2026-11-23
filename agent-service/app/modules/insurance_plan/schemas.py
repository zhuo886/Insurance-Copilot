from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class InsurancePlanItemInput(BaseModel):
    product_id: int
    priority: int = 1
    recommendation_reason: str | None = None
    annual_premium_budget: Decimal | None = None


class InsurancePlanItemResponse(BaseModel):
    id: UUID
    plan_id: UUID
    product_id: int
    name: str
    category: str
    image_url: str | None = None
    description: str | None = None
    highlights: list[str] = Field(default_factory=list)
    priority: int = 1
    recommendation_reason: str | None = None
    annual_premium_budget: Decimal | None = None
    status: str


class InsurancePlanResponse(BaseModel):
    id: UUID
    user_id: int
    plan_name: str
    summary: str | None = None
    insured_profile: dict[str, Any] = Field(default_factory=dict)
    annual_premium_budget: Decimal | None = None
    status: str
    created_at: datetime
    updated_at: datetime
    items: list[InsurancePlanItemResponse] = Field(default_factory=list)
