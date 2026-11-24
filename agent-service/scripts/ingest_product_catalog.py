"""将全量 12 款真实产品的结构化语义文本写入 Milvus 产品目录集合（insurance_product_catalog）。

用法：在 agent-service 目录执行
    uv run python scripts/ingest_product_catalog.py
"""

import asyncio
import sys
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from langchain_milvus import BM25BuiltInFunction, Milvus
from langchain_openai import OpenAIEmbeddings
from pymilvus import MilvusClient

from app.core.config import settings
from app.core.logging import get_logger
from app.modules.product.catalog_data import load_builtin_catalog

logger = get_logger(__name__)

PRODUCT_CATALOG_COLLECTION = "insurance_product_catalog"


def create_product_vector_store(drop_old: bool = False) -> Milvus:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-v4",
        api_key=settings.llm.api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        dimensions=1024,
        check_embedding_ctx_length=False,
        chunk_size=10,
    )
    if drop_old:
        client = MilvusClient(uri=settings.rag.milvus_url)
        try:
            if client.has_collection(PRODUCT_CATALOG_COLLECTION):
                client.drop_collection(PRODUCT_CATALOG_COLLECTION)
        finally:
            client.close()

    return Milvus(
        embedding_function=embeddings,
        collection_name=PRODUCT_CATALOG_COLLECTION,
        builtin_function=BM25BuiltInFunction(analyzer_params={"type": "chinese"}),
        vector_field=["dense", "sparse"],
        connection_args={"uri": settings.rag.milvus_url},
        auto_id=True,
    )


def ingest_catalog_to_milvus() -> int:
    catalog = load_builtin_catalog()
    docs = [
        Document(
            page_content=item.to_semantic_text(),
            metadata={
                "product_id": item.id,
                "category": item.category,
                "is_featured": item.is_featured,
                "min_age": item.min_age,
                "max_age": item.max_age,
                "max_occupation_class": item.max_occupation_class,
            },
        )
        for item in catalog
    ]
    vector_store = create_product_vector_store(drop_old=True)
    try:
        vector_store.add_documents(docs)
    finally:
        vector_store.client.close()
    logger.info("产品目录向量与BM25索引构建完成", collection=PRODUCT_CATALOG_COLLECTION, count=len(docs))
    return len(docs)


if __name__ == "__main__":
    ingest_catalog_to_milvus()
