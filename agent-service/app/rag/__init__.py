from typing import Any

_initialized = False
_vector_store: Any = None
_pipeline: Any = None
_retriever: Any = None


def _ensure_initialized() -> None:
    """延迟初始化 RAG 组件，避免纯离线单测导入子模块时强制连接 Milvus。"""
    global _initialized, _vector_store, _pipeline, _retriever
    if _initialized:
        return

    from langchain_milvus import BM25BuiltInFunction, Milvus
    from langchain_openai import OpenAIEmbeddings

    from app.core.config import settings
    from app.rag.pipeline import RAGPipeline
    from app.rag.retriever import ParentChunkRetriever

    embeddings = OpenAIEmbeddings(
        model="text-embedding-v4",
        api_key=settings.llm.api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        dimensions=1024,
        check_embedding_ctx_length=False,
        chunk_size=10,
    )
    _vector_store = Milvus(
        embedding_function=embeddings,
        collection_name="insurance_collection",
        builtin_function=BM25BuiltInFunction(
            analyzer_params={"type": "chinese"}
        ),
        vector_field=["dense", "sparse"],
        connection_args={"uri": settings.rag.milvus_url},
        auto_id=True,
    )
    _pipeline = RAGPipeline(_vector_store)
    _retriever = ParentChunkRetriever(_vector_store)
    globals()["_vector_store"] = _vector_store
    globals()["pipeline"] = _pipeline
    globals()["retriever"] = _retriever
    _initialized = True


def get_pipeline() -> Any:
    _ensure_initialized()
    return _pipeline


def get_retriever() -> Any:
    _ensure_initialized()
    return _retriever


def __getattr__(name: str) -> Any:
    if name == "_vector_store":
        _ensure_initialized()
        return _vector_store
    if name == "pipeline":
        return get_pipeline()
    if name == "retriever":
        return get_retriever()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def close_rag() -> None:
    """关闭 RAG 组件"""
    global _initialized
    if not _initialized:
        return
    _pipeline.close()
    _vector_store.client.close()
    _initialized = False


__all__ = ["pipeline", "retriever", "get_pipeline", "get_retriever", "close_rag"]
