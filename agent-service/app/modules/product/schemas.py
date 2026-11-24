from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ProductResponse(BaseModel):
    """保险产品响应结果"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    clause_name: str
    category: str
    insurer: str
    image_url: str | None = None
    description: str | None = None
    min_premium: Decimal | None = None
    max_premium: Decimal | None = None
    target_group: str | None = None
    highlights: list[str] = Field(default_factory=list)
    status: str


class PricingSchemaResponse(BaseModel):
    """产品保费试算动态表单模板响应"""

    model_config = ConfigDict(populate_by_name=True)

    product_id: int
    scenario: str = "premium_by_coverage"
    schema_version: str
    schema_data: dict[str, Any] = Field(alias="schema", serialization_alias="schema")
