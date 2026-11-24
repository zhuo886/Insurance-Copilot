from typing import Any

from pydantic import BaseModel, Field


class PremiumCalculateRequest(BaseModel):
    product_id: int | None = None
    scenario: str = "premium_by_coverage"
    answers: dict[str, Any] = Field(default_factory=dict)


class PremiumBreakdownItem(BaseModel):
    item: str
    name: str
    premium: str
    source: dict[str, Any] = Field(default_factory=dict)


class PremiumCalculateResponse(BaseModel):
    quote_id: str
    product_id: int
    plan_code: str
    pricing_mode: str
    annual_premium: str
    payment_frequency: str = "annual"
    breakdown: list[PremiumBreakdownItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    expires_at: str
