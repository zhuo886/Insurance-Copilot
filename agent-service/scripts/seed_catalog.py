"""幂等建表与 50 款保险产品及结构化特征入库脚本。

用法：在 agent-service 目录执行
    uv run python scripts/seed_catalog.py
"""

import asyncio
import sys
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import text

from app.core.logging import get_logger
from app.infra.database import AsyncSessionFactory, close_database
from app.modules.product.catalog_data import load_builtin_catalog

logger = get_logger(__name__)

CREATE_PRODUCT_FEATURES_SQL = """
CREATE TABLE IF NOT EXISTS product_features (
    product_id BIGINT PRIMARY KEY REFERENCES products(id) ON DELETE CASCADE,
    min_age INT NOT NULL DEFAULT 0,
    max_age INT NOT NULL DEFAULT 65,
    max_occupation_class INT NOT NULL DEFAULT 4,
    waiting_period_days INT NOT NULL DEFAULT 0,
    guarantee_renewal_years INT,
    deductible_default DECIMAL(14, 2) NOT NULL DEFAULT 0.00,
    has_clause BOOLEAN NOT NULL DEFAULT FALSE,
    is_featured BOOLEAN NOT NULL DEFAULT FALSE,
    coverage_tags TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    health_disclosure_level VARCHAR(30) NOT NULL DEFAULT 'standard',
    insurer_tier VARCHAR(30) NOT NULL DEFAULT 'top',
    base_rate_factor DECIMAL(8, 4) NOT NULL DEFAULT 1.0000,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

UPSERT_PRODUCT_SQL = """
INSERT INTO products (
    id, name, clause_name, category, insurer, image_url,
    description, min_premium, max_premium, target_group, highlights, status
)
VALUES (
    :id, :name, :clause_name, :category, :insurer, :image_url,
    :description, :min_premium, :max_premium, :target_group, :highlights, :status
)
ON CONFLICT (id) DO UPDATE SET
    name = EXCLUDED.name,
    clause_name = EXCLUDED.clause_name,
    category = EXCLUDED.category,
    insurer = EXCLUDED.insurer,
    image_url = COALESCE(EXCLUDED.image_url, products.image_url),
    description = EXCLUDED.description,
    min_premium = EXCLUDED.min_premium,
    max_premium = EXCLUDED.max_premium,
    target_group = EXCLUDED.target_group,
    highlights = EXCLUDED.highlights,
    status = EXCLUDED.status,
    updated_at = CURRENT_TIMESTAMP;
"""

UPSERT_FEATURE_SQL = """
INSERT INTO product_features (
    product_id, min_age, max_age, max_occupation_class,
    waiting_period_days, guarantee_renewal_years, deductible_default,
    has_clause, is_featured, coverage_tags,
    health_disclosure_level, insurer_tier, base_rate_factor
)
VALUES (
    :product_id, :min_age, :max_age, :max_occupation_class,
    :waiting_period_days, :guarantee_renewal_years, :deductible_default,
    :has_clause, :is_featured, :coverage_tags,
    :health_disclosure_level, :insurer_tier, :base_rate_factor
)
ON CONFLICT (product_id) DO UPDATE SET
    min_age = EXCLUDED.min_age,
    max_age = EXCLUDED.max_age,
    max_occupation_class = EXCLUDED.max_occupation_class,
    waiting_period_days = EXCLUDED.waiting_period_days,
    guarantee_renewal_years = EXCLUDED.guarantee_renewal_years,
    deductible_default = EXCLUDED.deductible_default,
    has_clause = EXCLUDED.has_clause,
    is_featured = EXCLUDED.is_featured,
    coverage_tags = EXCLUDED.coverage_tags,
    health_disclosure_level = EXCLUDED.health_disclosure_level,
    insurer_tier = EXCLUDED.insurer_tier,
    base_rate_factor = EXCLUDED.base_rate_factor,
    updated_at = CURRENT_TIMESTAMP;
"""


async def seed_catalog() -> int:
    catalog = load_builtin_catalog()
    async with AsyncSessionFactory() as session:
        async with session.begin():
            await session.execute(text(CREATE_PRODUCT_FEATURES_SQL))

            for item in catalog:
                await session.execute(
                    text(UPSERT_PRODUCT_SQL),
                    {
                        "id": item.id,
                        "name": item.name,
                        "clause_name": item.clause_name,
                        "category": item.category,
                        "insurer": item.insurer,
                        "image_url": item.image_url,
                        "description": item.description,
                        "min_premium": item.min_premium,
                        "max_premium": item.max_premium,
                        "target_group": item.target_group,
                        "highlights": item.highlights,
                        "status": item.status,
                    },
                )
                await session.execute(
                    text(UPSERT_FEATURE_SQL),
                    {
                        "product_id": item.id,
                        "min_age": item.min_age,
                        "max_age": item.max_age,
                        "max_occupation_class": item.max_occupation_class,
                        "waiting_period_days": item.waiting_period_days,
                        "guarantee_renewal_years": item.guarantee_renewal_years,
                        "deductible_default": item.deductible_default,
                        "has_clause": item.has_clause,
                        "is_featured": item.is_featured,
                        "coverage_tags": item.coverage_tags,
                        "health_disclosure_level": item.health_disclosure_level,
                        "insurer_tier": item.insurer_tier,
                        "base_rate_factor": item.base_rate_factor,
                    },
                )

            await session.execute(
                text(
                    "SELECT setval(pg_get_serial_sequence('products', 'id'), "
                    "(SELECT COALESCE(MAX(id), 1) FROM products), true);"
                )
            )
    logger.info("50款产品及结构化特征幂等入库完成", count=len(catalog))
    return len(catalog)


async def main() -> None:
    try:
        await seed_catalog()
    finally:
        await close_database()


if __name__ == "__main__":
    asyncio.run(main())
