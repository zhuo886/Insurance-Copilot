from decimal import Decimal

from sqlalchemy import ARRAY, BigInteger, Boolean, ForeignKey, Integer, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.common.models import Base, CreateAtMixin, UpdateAtMixin


class Product(Base, CreateAtMixin, UpdateAtMixin):
    """保险产品"""

    __tablename__ = "products"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    clause_name: Mapped[str] = mapped_column(String(300))
    category: Mapped[str] = mapped_column(String(50))
    insurer: Mapped[str] = mapped_column(String(120))
    image_url: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    min_premium: Mapped[Decimal | None] = mapped_column(Numeric(14, 2))
    max_premium: Mapped[Decimal | None] = mapped_column(Numeric(14, 2))
    target_group: Mapped[str | None] = mapped_column(String(300))
    highlights: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    status: Mapped[str] = mapped_column(
        String(30),
        default="active",
        server_default="active",
    )


class ProductFeature(Base, CreateAtMixin, UpdateAtMixin):
    """保险产品结构化承保与搜推特征扩展表（1:1 关联 products）"""

    __tablename__ = "product_features"

    product_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("products.id", ondelete="CASCADE"),
        primary_key=True,
    )
    min_age: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    max_age: Mapped[int] = mapped_column(Integer, default=65, server_default="65")
    max_occupation_class: Mapped[int] = mapped_column(Integer, default=4, server_default="4")
    waiting_period_days: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    guarantee_renewal_years: Mapped[int | None] = mapped_column(Integer, nullable=True)
    deductible_default: Mapped[Decimal] = mapped_column(
        Numeric(14, 2),
        default=Decimal("0.00"),
        server_default="0.00",
    )
    has_clause: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    is_featured: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    coverage_tags: Mapped[list[str]] = mapped_column(
        ARRAY(Text),
        default=list,
        server_default="{}",
    )
    health_disclosure_level: Mapped[str] = mapped_column(
        String(30),
        default="standard",
        server_default="standard",
    )
    insurer_tier: Mapped[str] = mapped_column(
        String(30),
        default="top",
        server_default="top",
    )
    base_rate_factor: Mapped[Decimal] = mapped_column(
        Numeric(8, 4),
        default=Decimal("1.0000"),
        server_default="1.0000",
    )
