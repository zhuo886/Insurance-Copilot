from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import BigInteger, Numeric, String
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.common.models import Base, CreateAtMixin, UpdateAtMixin


class Order(Base, CreateAtMixin, UpdateAtMixin):
    """保险投保订单表（支持报价关联与幂等创建）"""

    __tablename__ = "orders"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    thread_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), nullable=True)
    order_number: Mapped[str] = mapped_column(String(40), unique=True, nullable=False)
    confirmation_id: Mapped[str | None] = mapped_column(String(60), unique=True, nullable=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(160), unique=True, nullable=True)
    idempotency_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    items: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, nullable=False)
    quote_ids: Mapped[list[UUID]] = mapped_column(
        ARRAY(PG_UUID(as_uuid=True)),
        default=list,
        server_default="{}",
        nullable=False,
    )
    total_premium: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    recommendation_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    status: Mapped[str] = mapped_column(
        String(30),
        default="pending_payment",
        server_default="pending_payment",
    )
