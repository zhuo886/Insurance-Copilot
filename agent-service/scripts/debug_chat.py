"""一次性排查脚本：配置 -> 数据 -> 检索 -> 工具 -> 图 -> 流式

用法：在 agent-service 目录执行
    uv run python scripts/debug_chat.py

说明：
- Windows 下 asyncio 默认用 ProactorEventLoop，psycopg 异步不支持，
  所以必须在导入 app 之前切到 SelectorEventLoop。
- checkpointer 的连接池是模块级单例，只能初始化一次、关闭一次，
  因此统一在 main() 里创建并向下传递。
"""

import asyncio
import sys
import time
import traceback
from pathlib import Path

# ---- Windows 事件循环修正（必须在其他导入之前设置）----
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

QUESTION = "达尔文12号的免赔额是多少"
PRODUCT_ID = 2                 # 达尔文12号重大疾病保险
RUN_TAG = str(int(time.time()))  # 每次运行用新的 thread_id，避免读到历史状态

FAILED = []


def step(no: str, title: str) -> None:
    print("\n" + "=" * 64)
    print(f"{no} {title}")
    print("=" * 64)


def mask(value) -> str:
    text = str(value or "")
    if not text:
        return "(空)"
    return text[:6] + "..." + text[-4:] if len(text) > 12 else "***"


def mark(ok: bool, name: str, detail: str = "") -> None:
    print(("  [OK] " if ok else "  [FAIL] ") + name + (f" -> {detail}" if detail else ""))
    if not ok:
        FAILED.append(f"{name}: {detail}")


async def check_config() -> None:
    step("①", "配置检查")
    from app.core.config import settings
    print("  chat_model :", settings.llm.chat_model)
    print("  api_key    :", mask(settings.llm.api_key))
    print("  milvus_url :", settings.rag.milvus_url)
    print("  mineru     :", mask(settings.rag.mineru_token))


async def check_postgres() -> list[int]:
    step("②", "PostgreSQL 父块数量")
    from sqlalchemy import text
    from app.infra.database import AsyncSessionFactory

    async with AsyncSessionFactory() as session:
        rows = (await session.execute(
            text("select product_id, count(*) from parent_chunks group by product_id order by product_id")
        )).all()

    if not rows:
        mark(False, "parent_chunks", "一条都没有")
        return []

    for pid, cnt in rows:
        print(f"  product {pid}: {cnt} 个父块")
    print("  合计:", sum(cnt for _, cnt in rows), "个父块")
    return [pid for pid, _ in rows]


def check_milvus(product_ids: list[int]) -> None:
    step("③", "Milvus 集合与子块数量")
    from pymilvus import MilvusClient
    from app.core.config import settings

    client = MilvusClient(uri=settings.rag.milvus_url)
    names = client.list_collections()
    print("  collections:", names)
    if "insurance_collection" not in names:
        mark(False, "insurance_collection", "不存在")
        return

    client.load_collection("insurance_collection")
    print("  stats:", client.get_collection_stats("insurance_collection"))

    total = 0
    empty = []
    for pid in range(1, 13):
        r = client.query(
            "insurance_collection",
            filter=f"product_id == {pid}",
            output_fields=["count(*)"],
        )
        cnt = r[0].get("count(*)") if r else 0
        total += cnt
        if cnt == 0:
            empty.append(pid)
        print(f"  product {pid}: {cnt} 个子块")
    print("  合计:", total, "个子块")

    if empty:
        mark(False, "milvus 子块", f"product {empty} 没有向量数据")
    missing = sorted(set(product_ids) - set(range(1, 13)))
    if missing:
        print("  提示：PG 中存在但不在 1~12 范围的产品:", missing)


async def check_retriever(product_ids: list[int]) -> None:
    step("④", "检索器（绕开 LLM 和图）")
    from app.rag import retriever

    for pid in product_ids:
        try:
            chunks = await retriever.retrieve("免赔额", k=2, product_id=pid)
            print(f"  product {pid}: {len(chunks)} 个父块")
            for c in chunks[:1]:
                print("      ", c.section_path, "|", c.content[:40].replace("\n", " "))
            if pid == PRODUCT_ID and not chunks:
                mark(False, "retrieve", f"product {pid} 返回空")
        except Exception as exc:
            mark(False, f"retrieve product {pid}", f"{type(exc).__name__}: {exc}")
            traceback.print_exc()


async def check_tool() -> None:
    step("⑤", "工具（绕开图）")
    from app.agents.tools import query_product_clause

    schema = query_product_clause.args or {}
    print("  工具参数:", list(schema.get("properties", {}).keys()))
    print("  必填参数:", schema.get("required", []))

    # 5.1 按模型实际可能给出的参数调用（故意不带 k）
    print("\n  -- 调用 A：只给 query + product_id（模拟模型漏参数）")
    try:
        result = await query_product_clause.ainvoke(
            {"query": "免赔额", "product_id": PRODUCT_ID}
        )
        print("  返回:", str(result)[:200])
        mark(True, "tool 调用A（不传 k）")
    except Exception as exc:
        mark(False, "tool 调用A（不传 k）", f"{type(exc).__name__}: {str(exc)[:160]}")

    # 5.2 补齐全部参数再试一次
    print("\n  -- 调用 B：补齐 k")
    try:
        result = await query_product_clause.ainvoke(
            {"query": "免赔额", "k": 2, "product_id": PRODUCT_ID}
        )
        print("  返回:", str(result)[:200])
        mark(True, "tool 调用B（传 k）")
    except Exception as exc:
        mark(False, "tool 调用B（传 k）", f"{type(exc).__name__}: {str(exc)[:160]}")
        traceback.print_exc()


async def check_graph(cp) -> None:
    step("⑥", "图（非流式，绕开 SSE）")
    from app.agents.orchestrator import init_insurance_orchestrator

    try:
        graph = init_insurance_orchestrator(cp)
        thread_id = f"debug-graph-{RUN_TAG}"
        result = await graph.ainvoke(
            {"messages": [("user", QUESTION)]},
            config={"configurable": {"thread_id": thread_id}},
        )
        for msg in result["messages"]:
            text = getattr(msg, "text", "") or ""
            label = type(msg).__name__
            print(f"  {label}: {text[:150]}")
            if getattr(msg, "tool_calls", None):
                print("      tool_calls:", [tc["name"] for tc in msg.tool_calls])
        mark(True, "graph.ainvoke")
    except Exception as exc:
        mark(False, "graph.ainvoke", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()


async def check_stream(cp) -> None:
    step("⑦", "流式（复现前端现象）")
    from app.agents.orchestrator import init_insurance_orchestrator

    try:
        graph = init_insurance_orchestrator(cp)
        thread_id = f"debug-stream-{RUN_TAG}"
        stream = await graph.astream_events(
            {"messages": [("user", QUESTION)]},
            {"configurable": {"thread_id": thread_id}},
            version="v3",
        )
        count = 0
        async for message in stream.messages:
            node = getattr(message, "node", None)
            texts = [t async for t in message.text]
            joined = "".join(texts)
            if joined:
                count += 1
                print(f"  [node={node}] {len(joined)} 字: {joined[:80]}")
        print("  共收到", count, "段文本")
        mark(count > 0, "astream_events", f"{count} 段文本")
    except Exception as exc:
        mark(False, "astream_events", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()


async def main() -> None:
    from app.infra.checkpointer import init_checkpointer, close_checkpointer
    from app.infra.database import close_database

    cp = None
    try:
        await check_config()
        product_ids = await check_postgres()
        check_milvus(product_ids)
        await check_retriever(product_ids)
        await check_tool()

        # checkpointer 全流程只初始化一次
        cp = await init_checkpointer()
        await check_graph(cp)
        await check_stream(cp)
    finally:
        if cp is not None:
            await close_checkpointer()
        await close_database()

        step("汇总", "失败项")
        if FAILED:
            for item in FAILED:
                print("  -", item)
        else:
            print("  全部通过")


if __name__ == "__main__":
    asyncio.run(main())
