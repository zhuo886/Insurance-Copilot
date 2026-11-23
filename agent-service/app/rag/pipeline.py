from pathlib import Path
from uuid import uuid4

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from mineru import MinerU

from app.core.config import APP_ROOT, settings
from app.infra.database import AsyncSessionFactory
from app.modules.product.models import Product
from app.rag.models import ParentChunk
from app.rag.repository import ParentChunkRepository
from app.core.logging import get_logger

logger = get_logger(__name__)


MARKDOWN_OPTIMIZATION_PROMPT = """
下面这份 Markdown 文档是从保险条款 PDF 解析得来，由于 PDF 中的各个小节是以表格形式存在，所以解析时出现错乱。你分析内容，帮我转为格式正确的 Markdown，特别是标题编号要正确。
- 标题等级要从 1 级标题开始，逐层增加，目录和文档名不计入标题等级。
- 输出结果中不要包含正文开始之前的部分。
- 输出结果只包含 Markdown 正文，不要解释或代码围栏。
- 不要修改保险条款。
""".strip()

# 解析缓存目录：解析一次，重复复用
CACHE_DIR = APP_ROOT / "data" / "parsed"
# MinerU 单文档最长等待秒数（SDK 默认仅 300 秒）
MINERU_TIMEOUT = 1800



class RAGPipeline:

    def __init__(self, vector_store: Milvus) -> None:
        self._vector_store = vector_store

        # 1.初始化文档解析工具
        self._mineru = MinerU(settings.rag.mineru_token)

        # 2.初始化大模型（provider 和 base_url 必须同时指定）
        self._model = init_chat_model(
            settings.llm.chat_model,
            model_provider="openai",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=settings.llm.api_key,
            extra_body={"thinking": {"type": "disabled"}},
            max_tokens=100000,
        )

        # 3.初始化文档切分工具
        self._markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
                ("####", "h4"),
            ],
            strip_headers=True,
        )
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "；", ";", "，", ","],
            keep_separator="end",
        )
        logger.info("RAG Pipeline 初始化完成✅️")

    async def ingest(self, product: Product) -> None:
        # 1.解析并优化保险条款（带缓存）
        markdown = self._parse_and_optimize(product)

        # 2.切分父块和子块
        parent_chunks, child_chunks = self._chunk(markdown, product)

        # 3.子块存入 Milvus
        self._save_child_chunks(product.id, child_chunks)

        # 4.父块存入 PostgreSQL
        await self._save_parent_chunks(product.id, parent_chunks)

    # ---------------- 解析与优化 ----------------

    def _parse_and_optimize(self, product: Product) -> str:
        """解析 PDF 并优化 Markdown，两级缓存：优化结果 > 原始解析结果 > 调用 MinerU"""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # 一级缓存：已经优化好的最终 Markdown，命中直接返回（最快）
        optimized_path = CACHE_DIR / f"{product.id}.md"
        if optimized_path.exists():
            logger.info(
                "命中最终缓存，跳过解析与优化",
                product_id=product.id,
                path=str(optimized_path),
            )
            return optimized_path.read_text(encoding="utf-8")

        # 二级缓存：MinerU 原始解析结果，命中则只做优化
        raw_path = CACHE_DIR / f"{product.id}.raw.md"
        if raw_path.exists():
            raw_md = raw_path.read_text(encoding="utf-8")
            logger.info("命中解析缓存，跳过 MinerU", product_id=product.id)
        else:
            file_path = Path(APP_ROOT / "data" / "raw" / "kb" / product.clause_name)
            if not file_path.exists():
                raise FileNotFoundError(f"条款文件不存在：{file_path}")

            logger.info(
                "调用 MinerU 解析",
                product_id=product.id,
                file=file_path.name,
                timeout=MINERU_TIMEOUT,
            )
            result = self._mineru.extract(str(file_path), timeout=MINERU_TIMEOUT)
            raw_md = result.markdown or ""
            if not raw_md.strip():
                raise ValueError(f"解析结果为空：{file_path.name}")

            raw_path.write_text(raw_md, encoding="utf-8")
            logger.info("解析完成并缓存", product_id=product.id, chars=len(raw_md))

        # 整体优化：整篇一次性交给模型，保持全局结构一致
        markdown = self._invoke_optimizer(raw_md)
        optimized_path.write_text(markdown, encoding="utf-8")
        logger.info("Markdown 优化完成并缓存", product_id=product.id, chars=len(markdown))
        return markdown

    def _invoke_optimizer(self, text: str) -> str:
        """整篇优化，并记录是否被 max_tokens 截断"""
        response = self._model.invoke(
            [
                {"role": "system", "content": MARKDOWN_OPTIMIZATION_PROMPT},
                {"role": "user", "content": text},
            ]
        )
        content = response.content
        content = content if isinstance(content, str) else str(content)

        finish_reason = (response.response_metadata or {}).get("finish_reason")
        logger.info(
            "优化返回",
            finish_reason=finish_reason,
            input_chars=len(text),
            output_chars=len(content),
        )
        if finish_reason == "length":
            logger.warning(
                "优化结果被截断，content 不完整",
                product_id_hint="检查 max_tokens 或改用「仅输出标题修正表」方案",
            )
        return content

    # ---------------- 切分 ----------------

    def _chunk(
        self,
        markdown: str,
        product: Product,
    ) -> tuple[list[ParentChunk], list[Document]]:
        # 1.按照Markdown标题切分父块
        sections = self._markdown_splitter.split_text(markdown)
        parent_chunks: list[ParentChunk] = []
        child_chunks: list[Document] = []

        # 2.遍历父块，继续切分子块
        for section in sections:
            section_path = list(section.metadata.values())
            parent = ParentChunk(
                id=uuid4(),
                product_id=product.id,
                clause_name=product.clause_name,
                section_path=section_path,
                content=section.page_content,
            )
            parent_chunks.append(parent)

            # 子块通过parent_id关联父块
            section.metadata = {
                "product_id": product.id,
                "parent_id": str(parent.id),
            }
            children = self._child_splitter.split_documents([section])

            # 把章节路径拼接到子块正文中，补充语义上下文
            section_header = "\n".join(section_path) + "\n"
            for child in children:
                child.page_content = section_header + child.page_content
            child_chunks.extend(children)

        return parent_chunks, child_chunks

    # ---------------- 入库 ----------------

    def _save_child_chunks(
        self,
        product_id: int,
        child_chunks: list[Document],
    ) -> None:
        # Collection存在时，先删除该产品的旧子块
        if self._vector_store.client.has_collection(
            self._vector_store.collection_name
        ):
            self._vector_store.delete(expr=f"product_id == {product_id}")
        # 再新增子块
        self._vector_store.add_documents(child_chunks)

    async def _save_parent_chunks(
        self,
        product_id: int,
        parent_chunks: list[ParentChunk],
    ) -> None:
        async with AsyncSessionFactory() as session:
            async with session.begin():
                repository = ParentChunkRepository(session)
                # 删除旧父块
                await repository.delete_by_product_id(product_id)
                # 添加新父块
                repository.add_all(parent_chunks)

    def close(self):
        """关闭mineru连接"""
        self._mineru.close()
