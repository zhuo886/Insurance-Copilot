from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class CreateOrderRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    quote_ids: list[UUID] = Field(
        default_factory=list,
        validation_alias=AliasChoices("quote_ids", "quoteIds"),
    )
    idempotency_key: str | None = None
    thread_id: UUID | None = Field(
        default=None,
        validation_alias=AliasChoices("thread_id", "threadId"),
    )
    confirmation_id: str | None = None
    items: list[dict[str, Any]] | None = None
    recommendation_snapshot: dict[str, Any] | None = None


class OrderResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: int
    thread_id: UUID | None = None
    order_number: str
    confirmation_id: str | None = None
    idempotency_key: str | None = None
    items: list[dict[str, Any]] = Field(default_factory=list)
    quote_ids: list[UUID] = Field(default_factory=list)
    total_premium: Decimal
    recommendation_snapshot: dict[str, Any] | None = None
    status: str
    created_at: datetime
    updated_at: datetime


class OrderListResponse(BaseModel):
    items: list[OrderResponse] = Field(default_factory=list)
