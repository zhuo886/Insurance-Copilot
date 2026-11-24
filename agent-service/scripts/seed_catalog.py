"""幂等建表与 12 款真实保险产品、官方 PDF 费率表及演示用户初始化脚本。

用法：在 agent-service 目录执行
    uv run python scripts/seed_catalog.py
"""

import asyncio
import json
from pathlib import Path
import sys
from uuid import NAMESPACE_DNS, uuid5

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import text

from app.common.auth import hash_password
from app.core.logging import get_logger
from app.infra.database import AsyncSessionFactory, close_database
from app.modules.product.catalog_data import load_builtin_catalog

logger = get_logger(__name__)

RATE_TABLES_PATH = PROJECT_ROOT / "data" / "parsed" / "rate_tables.json"

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

MIGRATE_INSURANCE_PLANS_SQL = """
ALTER TABLE insurance_plans
    ADD COLUMN IF NOT EXISTS budget_note TEXT,
    ADD COLUMN IF NOT EXISTS version INT NOT NULL DEFAULT 1;
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

INSERT_RATE_ITEM_SQL = """
INSERT INTO rate_table_items (
    id, product_id, plan_code, rate_type, age_min, age_max,
    gender, occupation_class, social_security, payment_period, coverage_period,
    coverage_amount, deductible, dimensions, rate_value, premium,
    source_page, source_table, status
)
VALUES (
    :id, :product_id, :plan_code, :rate_type, :age_min, :age_max,
    :gender, :occupation_class, :social_security, :payment_period, :coverage_period,
    :coverage_amount, :deductible, CAST(:dimensions AS jsonb), :rate_value, :premium,
    :source_page, :source_table, 'active'
);
"""


async def seed_catalog() -> int:
    catalog = load_builtin_catalog()
    max_real_id = max(item.id for item in catalog)

    async with AsyncSessionFactory() as session:
        async with session.begin():
            await session.execute(text(CREATE_PRODUCT_FEATURES_SQL))
            await session.execute(text(MIGRATE_INSURANCE_PLANS_SQL))

            # 安全清理历史上残存的虚构产品（id > 12），先删依赖表外键引用再删主表
            for cleanup_sql in (
                "DELETE FROM premium_quotes WHERE product_id > :max_id;",
                "DELETE FROM insurance_plan_items WHERE product_id > :max_id;",
                "DELETE FROM claims WHERE product_id > :max_id;",
                "DELETE FROM policies WHERE product_id > :max_id;",
                "DELETE FROM product_pricing_bindings WHERE product_id > :max_id;",
                "DELETE FROM product_pricing_schemas WHERE product_id > :max_id;",
                "DELETE FROM rate_table_items WHERE product_id > :max_id;",
                "DELETE FROM rate_plans WHERE product_id > :max_id;",
                "DELETE FROM rate_table_files WHERE product_id > :max_id;",
                "DELETE FROM product_features WHERE product_id > :max_id;",
                "DELETE FROM products WHERE id > :max_id;",
            ):
                await session.execute(text(cleanup_sql), {"max_id": max_real_id})

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

            # 同步 11/12 款官方 PDF 结构化费率表到 rate_table_items
            rate_rows_count = 0
            if RATE_TABLES_PATH.exists():
                tables: dict[str, list[dict]] = json.loads(
                    RATE_TABLES_PATH.read_text(encoding="utf-8")
                )
                await session.execute(text("DELETE FROM rate_table_items;"))
                batch_params: list[dict] = []
                active_pids: list[int] = []
                for pid_str, rows in tables.items():
                    pid = int(pid_str)
                    active_pids.append(pid)
                    for idx, row in enumerate(rows):
                        item_uuid = str(
                            uuid5(
                                NAMESPACE_DNS,
                                f"rate-item-{pid}-{idx}-{row.get('plan_code')}-{row.get('age_min')}-{row.get('gender')}-{row.get('social_security')}-{row.get('payment_period')}",
                            )
                        )
                        batch_params.append(
                            {
                                "id": item_uuid,
                                "product_id": pid,
                                "plan_code": row.get("plan_code") or "standard",
                                "rate_type": row.get("rate_type") or "fixed_plan_premium",
                                "age_min": row.get("age_min"),
                                "age_max": row.get("age_max"),
                                "gender": row.get("gender") or "all",
                                "occupation_class": str(row.get("occupation_class") or "all"),
                                "social_security": row.get("social_security") or "all",
                                "payment_period": row.get("payment_period"),
                                "coverage_period": row.get("coverage_period"),
                                "coverage_amount": row.get("coverage_amount"),
                                "deductible": row.get("deductible"),
                                "dimensions": json.dumps(
                                    {"source": "official_rate_table_pdf"},
                                    ensure_ascii=False,
                                ),
                                "rate_value": row.get("rate_value"),
                                "premium": row.get("premium"),
                                "source_page": row.get("source_page"),
                                "source_table": (row.get("source_table") or "")[:200],
                            }
                        )
                if batch_params:
                    await session.execute(text(INSERT_RATE_ITEM_SQL), batch_params)
                    rate_rows_count = len(batch_params)
                if active_pids:
                    await session.execute(
                        text(
                            "UPDATE rate_table_files "
                            "SET status = 'active', parser_version = 'official_pdf_v2', "
                            "activated_at = COALESCE(activated_at, CURRENT_TIMESTAMP) "
                            "WHERE product_id = ANY(:pids);"
                        ),
                        {"pids": active_pids},
                    )

            # 若 demo 账号仍为初始化占位密码哈希，则升级为真实 Argon2 哈希（演示口令 123456）
            await session.execute(
                text(
                    "UPDATE users SET password_hash = :pw_hash, updated_at = CURRENT_TIMESTAMP "
                    "WHERE id = 1 AND password_hash = '$2a$12$demo.password.hash.placeholder';"
                ),
                {"pw_hash": hash_password("123456")},
            )

    logger.info(
        "12款真实产品、官方费率表及基础配置幂等入库完成",
        product_count=len(catalog),
        rate_item_count=rate_rows_count,
    )
    return len(catalog)


async def main() -> None:
    try:
        await seed_catalog()
    finally:
        await close_database()


if __name__ == "__main__":
    asyncio.run(main())
