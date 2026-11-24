from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import BigInteger, DateTime, ForeignKey, Integer, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.common.models import Base, CreateAtMixin, UpdateAtMixin


class InsurancePlan(Base, CreateAtMixin, UpdateAtMixin):
    """用户确认保存的保险配置方案"""

    __tablename__ = "insurance_plans"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    plan_name: Mapped[str] = mapped_column(String(120), nullable=False)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    budget_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    insured_profile: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default="{}")
    annual_premium_budget: Mapped[Decimal | None] = mapped_column(Numeric(14, 2), nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    status: Mapped[str] = mapped_column(String(30), default="uninsured", server_default="uninsured")


class InsurancePlanItem(Base, CreateAtMixin, UpdateAtMixin):
    """保险配置方案中的单款产品明细"""

    __tablename__ = "insurance_plan_items"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    plan_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("insurance_plans.id", ondelete="CASCADE"),
        nullable=False,
    )
    product_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("products.id"), nullable=False)
    priority: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    recommendation_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    annual_premium_budget: Mapped[Decimal | None] = mapped_column(Numeric(14, 2), nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="uninsured", server_default="uninsured")


class Policy(Base, CreateAtMixin, UpdateAtMixin):
    """用户有效/历史保单（含 V14 投保与保障快照扩展字段）"""

    __tablename__ = "policies"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    product_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("products.id"), nullable=False)
    policy_number: Mapped[str] = mapped_column(String(60), unique=True, nullable=False)
    application_no: Mapped[str | None] = mapped_column(String(60), unique=True, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="active", server_default="active")
    effective_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    holder_name: Mapped[str | None] = mapped_column(String(80), nullable=True)
    insured_name: Mapped[str | None] = mapped_column(String(80), nullable=True)
    insured_id_no_masked: Mapped[str | None] = mapped_column(String(40), nullable=True)
    insured_phone: Mapped[str | None] = mapped_column(String(40), nullable=True)
    coverage_amount: Mapped[Decimal | None] = mapped_column(Numeric(14, 2), nullable=True)
    premium_amount: Mapped[Decimal | None] = mapped_column(Numeric(14, 2), nullable=True)
    payment_frequency: Mapped[str | None] = mapped_column(String(30), nullable=True)
    beneficiary: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    policy_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    coverage_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    paid_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    issued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Claim(Base, UpdateAtMixin):
    """用户理赔申请与进度记录（使用 submitted_at 与 updated_at）"""

    __tablename__ = "claims"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    policy_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("policies.id"),
        nullable=True,
    )
    product_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("products.id"), nullable=True)
    claim_number: Mapped[str | None] = mapped_column(String(60), nullable=True)
    claim_type: Mapped[str] = mapped_column(String(50), nullable=False)
    amount: Mapped[Decimal | None] = mapped_column(Numeric(14, 2), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="submitted", server_default="submitted")
    submitted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class PremiumQuote(Base, CreateAtMixin):
    """单产品保费试算结果持久化记录"""

    __tablename__ = "premium_quotes"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    product_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("products.id"), nullable=False)
    request_params: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    result: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    total_premium: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    pricing_mode: Mapped[str] = mapped_column(String(30), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
