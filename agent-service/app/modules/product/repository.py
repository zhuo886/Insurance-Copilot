from decimal import Decimal

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.product.models import Product, ProductFeature


class ProductRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def _has_features_table(self) -> bool:
        table_name = await self.session.scalar(text("SELECT to_regclass('public.product_features')"))
        return table_name is not None

    async def find_by_category(
        self,
        category: str | None = None,
        featured_only: bool = True,
    ) -> list[Product]:
        conditions = [Product.status == "active"]
        if category:
            conditions.append(Product.category == category)

        stmt = select(Product)
        if featured_only:
            if await self._has_features_table():
                stmt = stmt.join(ProductFeature, ProductFeature.product_id == Product.id)
                conditions.append(ProductFeature.is_featured.is_(True))
            else:
                conditions.append(Product.id <= 12)

        products = await self.session.scalars(
            stmt.where(*conditions).order_by(Product.id.asc())
        )
        return list(products.all())

    async def find_all_with_features(
        self,
        category: str | None = None,
    ) -> list[tuple[Product, ProductFeature | None]]:
        conditions = [Product.status == "active"]
        if category:
            conditions.append(Product.category == category)

        if await self._has_features_table():
            rows = await self.session.execute(
                select(Product, ProductFeature)
                .outerjoin(ProductFeature, ProductFeature.product_id == Product.id)
                .where(*conditions)
                .order_by(Product.id.asc())
            )
            return [(row[0], row[1]) for row in rows.all()]

        products = await self.session.scalars(
            select(Product).where(*conditions).order_by(Product.id.asc())
        )
        return [(product, None) for product in products.all()]

    async def find_limited_by_category(
        self,
        category: str,
        premium_min: Decimal | None,
        limit: int,
    ) -> list[Product]:
        conditions = [
            Product.status == "active",
            Product.category == category,
        ]
        if premium_min is not None:
            conditions.append(Product.min_premium < premium_min)

        products = await self.session.scalars(
            select(Product)
            .where(*conditions)
            .order_by(
                Product.min_premium.asc().nullslast(),
                Product.id.asc(),
            )
            .limit(limit)
        )
        return list(products.all())
