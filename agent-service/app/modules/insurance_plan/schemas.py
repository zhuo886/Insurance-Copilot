from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class InsurancePlanItemInput(BaseModel):
    product_id: int
    category: str | None = None
    priority: int = 1
    recommendation_reason: str | None = None
    annual_premium_budget: Decimal | None = None


class SaveCurrentPlanRequest(BaseModel):
    plan_name: str = Field(min_length=1, max_length=120)
    summary: str | None = None
    budget_note: str | None = None
    insured_profile: dict[str, Any] = Field(default_factory=dict)
    annual_premium_budget: Decimal | None = None
    items: list[InsurancePlanItemInput] = Field(min_length=1)


class PlanProductSummary(BaseModel):
    id: int
    name: str
    category: str
    insurer: str | None = None
    image_url: str | None = None


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
    product: PlanProductSummary | None = None


class InsurancePlanResponse(BaseModel):
    id: UUID
    user_id: int
    plan_name: str
    summary: str | None = None
    budget_note: str | None = None
    insured_profile: dict[str, Any] = Field(default_factory=dict)
    annual_premium_budget: Decimal | None = None
    version: int = 1
    status: str
    created_at: datetime
    updated_at: datetime
    items: list[InsurancePlanItemResponse] = Field(default_factory=list)


class InsurancePlanListResponse(BaseModel):
    items: list[InsurancePlanResponse] = Field(default_factory=list)
