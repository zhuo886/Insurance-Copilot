from decimal import Decimal
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.product.catalog_data import load_builtin_catalog
from app.modules.product.exceptions import PricingSchemaNotFoundError, ProductNotFoundError
from app.modules.product.models import Product, ProductFeature
from app.modules.product.repository import ProductRepository
from app.modules.product.schemas import PricingSchemaResponse
from app.recsys.pricing import load_parsed_rate_tables

_PLAN_LABELS = {
    "standard": "标准/经典计划",
    "basic": "基础版",
    "deluxe": "典藏版",
    "premium": "尊贵/优享版",
    "supreme": "至尊/尊享版",
    "plan2": "计划二（0免赔升级）",
}


class ProductService:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.repository = ProductRepository(session)

    async def list_products(
        self,
        category: str | None = None,
        featured_only: bool = True,
    ) -> list[Product]:
        return await self.repository.find_by_category(
            category=category,
            featured_only=featured_only,
        )

    async def list_candidates(
        self,
        categories: list[str],
        premium_min: Decimal | None,
        limit_per_category: int,
    ) -> list[Product]:
        candidate_products: list[Product] = []

        for category in dict.fromkeys(categories):
            products = await self.repository.find_limited_by_category(
                category=category,
                premium_min=premium_min,
                limit=limit_per_category,
            )
            candidate_products.extend(products)

        return candidate_products

    async def get_product(self, product_id: int) -> Product:
        product = await self.repository.find_by_id(product_id)
        if product is None:
            raise ProductNotFoundError
        return product

    async def get_pricing_schema(
        self,
        product_id: int,
        scenario: str = "premium_by_coverage",
    ) -> PricingSchemaResponse:
        row = await self.repository.find_with_feature_by_id(product_id)
        if row is None:
            raise PricingSchemaNotFoundError
        product, feature = row

        db_schema = await self.repository.find_active_pricing_schema(product_id, scenario)
        if db_schema is not None:
            return PricingSchemaResponse(
                product_id=product_id,
                scenario=scenario,
                schema_version=db_schema["schema_version"],
                schema=db_schema["schema_json"],
            )

        schema_json = self._build_dynamic_pricing_schema(product, feature)
        return PricingSchemaResponse(
            product_id=product_id,
            scenario=scenario,
            schema_version="2026-official-table.v1",
            schema=schema_json,
        )

    @staticmethod
    def _build_dynamic_pricing_schema(
        product: Product,
        feature: ProductFeature | None,
    ) -> dict[str, Any]:
        builtin_map = {p.id: p for p in load_builtin_catalog()}
        fallback = builtin_map.get(product.id)

        min_age = feature.min_age if feature is not None else (fallback.min_age if fallback else 0)
        max_age = feature.max_age if feature is not None else (fallback.max_age if fallback else 70)
        max_occ = (
            feature.max_occupation_class
            if feature is not None
            else (fallback.max_occupation_class if fallback else 4)
        )

        rate_rows = load_parsed_rate_tables().get(product.id) or []
        plan_codes = list(dict.fromkeys(str(r.get("plan_code") or "standard") for r in rate_rows)) or [
            "standard"
        ]
        plan_options = [
            {"value": code, "label": _PLAN_LABELS.get(code, code)}
            for code in plan_codes
        ]

        fields: list[dict[str, Any]] = [
            {
                "key": "plan_code",
                "label": "保障计划",
                "type": "plan_select",
                "required": True,
                "options": plan_options,
            },
            {
                "key": "insured.age",
                "label": "被保人年龄",
                "type": "integer",
                "required": True,
                "min": min_age,
                "max": max_age,
            },
            {
                "key": "insured.gender",
                "label": "被保人性别",
                "type": "select",
                "required": True,
                "options": [
                    {"value": "male", "label": "男"},
                    {"value": "female", "label": "女"},
                ],
            },
            {
                "key": "insured.occupation_class",
                "label": "职业类别",
                "type": "select",
                "required": False,
                "options": [
                    {"value": i, "label": f"{i}类职业"}
                    for i in range(1, max_occ + 1)
                ],
            },
        ]

        if product.category in ("medical", "critical_illness"):
            fields.append(
                {
                    "key": "insured.social_security",
                    "label": "是否有医保/社保",
                    "type": "select",
                    "required": True,
                    "options": [
                        {"value": "yes", "label": "有社保"},
                        {"value": "no", "label": "无社保"},
                    ],
                }
            )

        if product.category == "critical_illness":
            fields.append(
                {
                    "key": "coverage.coverage_amount",
                    "label": "基本保险金额",
                    "type": "money",
                    "required": True,
                    "options": [100000, 300000, 500000],
                }
            )
            fields.append(
                {
                    "key": "coverage.payment_period",
                    "label": "缴费年限",
                    "type": "select",
                    "required": False,
                    "options": [
                        {"value": "20y", "label": "20年交"},
                        {"value": "15y", "label": "15年交"},
                        {"value": "10y", "label": "10年交"},
                    ],
                }
            )
        elif product.category == "life":
            fields.append(
                {
                    "key": "coverage.coverage_amount",
                    "label": "目标基本保额（可选，按费率表反算）",
                    "type": "money",
                    "required": False,
                    "options": [50000, 100000, 300000],
                }
            )

        return {
            "title": f"{product.name}保费试算",
            "fields": fields,
            "plan_options": plan_options,
            "rules": [],
        }
