import asyncio
import sys
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_milvus import BM25BuiltInFunction, Milvus
from langchain_openai import OpenAIEmbeddings

from app.core.config import APP_ROOT, settings
from app.core.logging import get_logger
from app.infra.database import AsyncSessionFactory, close_database
from app.modules.product.models import Product
from app.modules.product.repository import ProductRepository
from app.rag.pipeline import RAGPipeline

logger = get_logger(__name__)


def create_vector_store() -> Milvus:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-v4",
        api_key=settings.llm.api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        dimensions=1024,
        check_embedding_ctx_length=False,
        chunk_size=10,
    )
    return Milvus(
        embedding_function=embeddings,
        collection_name="insurance_collection",
        builtin_function=BM25BuiltInFunction(
            analyzer_params={"type": "chinese"}
        ),
        vector_field=["dense", "sparse"],
        connection_args={"uri": settings.rag.milvus_url},
        auto_id=True,
    )


def _has_local_clause_source(product: Product) -> bool:
    parsed_path = APP_ROOT / "data" / "parsed" / f"{product.id}.md"
    raw_path = APP_ROOT / "data" / "raw" / "kb" / product.clause_name
    return parsed_path.exists() or raw_path.exists()


async def find_clause_products() -> list[Product]:
    async with AsyncSessionFactory() as session:
        repository = ProductRepository(session)
        rows = await repository.find_all_with_features()
        products: list[Product] = []
        for product, feature in rows:
            has_clause = feature.has_clause if feature is not None else (product.id <= 12)
            if has_clause and _has_local_clause_source(product):
                products.append(product)
        return products


async def main() -> None:
    vector_store = create_vector_store()
    pipeline = RAGPipeline(vector_store)

    try:
        products = await find_clause_products()
        logger.info("开始构建RAG知识库", product_count=len(products))

        for index, product in enumerate(products, start=1):
            logger.info(
                "开始处理保险产品",
                progress=f"{index}/{len(products)}",
                product_id=product.id,
                product_name=product.name,
            )
            await pipeline.ingest(product)

        logger.info("RAG知识库构建完成", product_count=len(products))
    finally:
        pipeline.close()
        vector_store.client.close()
        await close_database()


if __name__ == "__main__":
    asyncio.run(main())
