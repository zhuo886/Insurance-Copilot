from decimal import Decimal

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.infra.database import get_session
from app.modules.product.models import Product
from app.modules.product.schemas import ProductResponse
from app.modules.product.service import ProductService

router = APIRouter(prefix="/api/v1/products", tags=["products"])


@router.get("", response_model=list[ProductResponse])
async def list_products(
    category: str | None = None,
    featured_only: bool = True,
    session: AsyncSession = Depends(get_session),
) -> list[Product]:
    service = ProductService(session)
    return await service.list_products(category=category, featured_only=featured_only)


@router.get("/candidates", response_model=list[ProductResponse])
async def list_candidates(
    categories: list[str] = Query(min_length=1),
    premium_min: Decimal | None = None,
    limit_per_category: int = 5,
    session: AsyncSession = Depends(get_session),
) -> list[Product]:
    service = ProductService(session)
    return await service.list_candidates(
        categories=categories,
        premium_min=premium_min,
        limit_per_category=limit_per_category,
    )
