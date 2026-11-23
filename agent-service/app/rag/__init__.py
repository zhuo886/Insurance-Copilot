from langchain_milvus import BM25BuiltInFunction, Milvus
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

from app.core.config import settings
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import ParentChunkRetriever


# 1.初始化向量模型
_embeddings = OpenAIEmbeddings(
        model="text-embedding-v4",
        api_key=settings.llm.api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        dimensions=1024,                     # 必须与 collection 的 dense 维度一致
        check_embedding_ctx_length=False,    # 非 OpenAI 端点必须关掉
        chunk_size=10,                       # 百炼单次最多 10 条
    )
# 2.初始化VectorStore
_vector_store = Milvus(
    embedding_function=_embeddings,
    collection_name="insurance_collection",
    builtin_function=BM25BuiltInFunction(
        analyzer_params={"type": "chinese"}
    ),
    vector_field=["dense", "sparse"],
    connection_args={"uri": settings.rag.milvus_url},
    auto_id=True,
)

# 3.创建模块级Pipeline和Retriever
pipeline = RAGPipeline(_vector_store)
retriever = ParentChunkRetriever(_vector_store)


def close_rag() -> None:
    """关闭RAG组件"""
    pipeline.close()
    _vector_store.client.close()

__all__ = ['pipeline', 'retriever', 'close_rag']