from decimal import Decimal
import json

from langchain.tools import tool
from fastapi.encoders import jsonable_encoder
from langgraph.config import get_stream_writer

from app.infra.database import AsyncSessionFactory
from app.modules.product.models import Product
from app.modules.product.service import ProductService

@tool
async def query_candidate_products(
    categories: list[str],
    premium_min: Decimal | None = None,
    limit_per_category: int = 5
):
    """
    根据险种和保费条件查询可用于推荐的候选保险产品。当用户咨询具体保险产品或需要保险产品推荐时使用。

    Args:
        categories: 产品分类列表，可选值为 medical、critical_illness、life、accident。
        premium_min: 最低保费上限，可选参数，只返回最低保费小于该值的产品。
        limit_per_category: 每个险种最多返回的产品数量，可选参数，默认5。
    """

    async with AsyncSessionFactory() as session:
        # 1.初始化Service
        service = ProductService(session)
        # 2.调用Service，得到候选产品
        products: list[Product] = await service.list_candidates(
            categories,
            premium_min,
            limit_per_category
        )
        # 3.返回给AI，最好把Product处理成json返回
        return jsonable_encoder(products)


@tool
async def query_product_clause(
        query: str,
        product_id: int,
        k: int=3
) -> str:
    """查询指定保险产品的条款内容。

    Args:
        query: 需要从保险条款中查询的问题。
        k: 返回的文档=数量
        product_id: 保险产品ID。
    """
    from app.rag import retriever

    chunks = await retriever.retrieve(query, k, product_id)
    if not chunks:
        return "没有查询到相关保险条款。"

    sources = {
        # 给文档编号，最终格式 {"ref-001": {"content": ""}}
        f"ref-{i:03d}": {
            "product_id": chunk.product_id,
            "clause_name": chunk.clause_name,
            "section_path": chunk.section_path,
            "content": chunk.content,
        }
        for i, chunk in enumerate(chunks, start=1)
    }

    # 把sources写到自定义事件中
    writer = get_stream_writer()
    writer({"type": "additional_info", "data": sources})

    # 以JSON格式返回
    return json.dumps(sources, ensure_ascii=False)

