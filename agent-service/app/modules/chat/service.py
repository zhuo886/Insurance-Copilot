from collections.abc import AsyncIterator
from typing import Any

from fastapi.sse import ServerSentEvent
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.stream import CustomTransformer
from langgraph.types import Command
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.modules.chat.schemas import ChatRequest
from app.modules.chat_thread.exceptions import ChatThreadNotFoundError
from app.modules.chat_thread.repository import ChatThreadRepository

logger = get_logger(__name__)

_LIST_KEYS = ("candidate_products", "candidates_products", "plan_items", "policies", "policy_items")


def _looks_like_policy(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    return bool(
        item.get("policy_number")
        or item.get("effective_at")
        or item.get("coverage_snapshot")
        or item.get("policy_snapshot")
    )


def merge_turn_additional_info(
    current: dict[str, Any],
    incoming: Any,
) -> dict[str, Any]:
    """聚合单轮对话中多个工具产出的附加信息（条款来源 / 候选产品 / 方案项 / 保单）。"""
    merged: dict[str, Any] = dict(current)
    if not incoming:
        return merged

    if isinstance(incoming, list):
        target_key = "policies" if any(_looks_like_policy(x) for x in incoming) else "candidate_products"
        existing_list = list(merged.get(target_key) or [])
        existing_list.extend(incoming)
        merged[target_key] = existing_list
        return merged

    if not isinstance(incoming, dict):
        return merged

    handled = False

    # 1. 条款引用来源（支持包装键或裸 {ref-001: {...}} 字典）
    sources_payload = incoming.get("clause_sources") or incoming.get("sources")
    if isinstance(sources_payload, dict):
        existing_sources = dict(merged.get("clause_sources") or {})
        existing_sources.update(sources_payload)
        merged["clause_sources"] = existing_sources
        handled = True

    # 2. 列表类卡片（候选产品 / 推荐方案 / 保单）
    for key in _LIST_KEYS:
        payload = incoming.get(key)
        if isinstance(payload, list):
            normalized_key = (
                "candidate_products"
                if key == "candidates_products"
                else ("policies" if key == "policy_items" else key)
            )
            existing_items = list(merged.get(normalized_key) or [])
            existing_items.extend(payload)
            merged[normalized_key] = existing_items
            handled = True

    if _looks_like_policy(incoming) and not handled:
        existing_policies = list(merged.get("policies") or [])
        existing_policies.append(incoming)
        merged["policies"] = existing_policies
        handled = True

    # 3. 兼容旧工具直接返回 {"ref-001": {"content": ...}} 的结构
    if not handled:
        existing_sources = dict(merged.get("clause_sources") or {})
        existing_sources.update(incoming)
        merged["clause_sources"] = existing_sources

    return merged


def extract_pending_interrupt(snapshot: Any) -> dict[str, Any] | None:
    """从 LangGraph StateSnapshot 中可靠提取挂起的 HITL interrupt 载荷。"""
    if snapshot is None:
        return None

    # 优先读取 LangGraph 提供的 snapshot.interrupts（如存在），再兜底遍历 tasks
    direct_interrupts = getattr(snapshot, "interrupts", None) or ()
    for intr in direct_interrupts:
        value = getattr(intr, "value", intr)
        if isinstance(value, dict):
            return value

    tasks = getattr(snapshot, "tasks", None) or ()
    for task in tasks:
        interrupts = getattr(task, "interrupts", None) or ()
        for intr in interrupts:
            value = getattr(intr, "value", intr)
            if isinstance(value, dict):
                return value
    return None


class ChatService:
    def __init__(self, session: AsyncSession, agent: CompiledStateGraph):
        self.repository = ChatThreadRepository(session)
        self.agent = agent

    async def stream_chat(
        self,
        user_id: int,
        request: ChatRequest,
    ) -> AsyncIterator[ServerSentEvent]:
        # 1. 校验会话归属权
        thread = await self.repository.find_owned(
            request.thread_id,
            user_id,
        )
        if thread is None:
            raise ChatThreadNotFoundError

        # 2. 准备图输入（支持普通消息与 HITL Command resume）并透传 user_id
        if request.decision is not None and not request.message:
            _input: Any = Command(resume=request.decision)
        else:
            _input = {"messages": [HumanMessage(content=request.message or "")]}

        config = {
            "configurable": {
                "thread_id": str(request.thread_id),
                "user_id": user_id,
            }
        }

        stream = await self.agent.astream_events(
            _input,
            config,
            version="v3",
            transformers=[CustomTransformer],
        )

        final_message_id: str | None = None
        additional_info: dict[str, Any] = {}
        streamed_ids: set[str] = set()

        async for event in stream:
            method = event.get("method")
            params = event.get("params") or {}
            logger.debug(
                "stream event",
                method=method,
                namespace=params.get("namespace"),
            )

            # 2.1 消息事件：只推送面向用户的文本增量
            if method == "messages":
                namespace = params.get("namespace") or []
                items = params.get("data") or []
                if not items:
                    continue

                payload = items[0]
                metadata = items[1] if len(items) > 1 else {}

                # 过滤父图内部 route 节点的结构化输出
                if (
                    not namespace
                    and isinstance(metadata, dict)
                    and metadata.get("langgraph_node") == "route"
                ):
                    continue

                if isinstance(payload, dict):
                    evt = payload.get("event")

                    if evt == "message-start":
                        final_message_id = payload.get("id") or final_message_id

                    elif evt == "content-block-delta":
                        delta = payload.get("delta") or {}
                        if delta.get("type") == "text-delta":
                            text = delta.get("text") or ""
                            if text:
                                if final_message_id:
                                    streamed_ids.add(final_message_id)
                                yield ServerSentEvent(data=text, event="message")

                    elif evt == "content-block-finish":
                        continue

                elif isinstance(payload, AIMessage):
                    message_id = getattr(payload, "id", None)
                    if message_id and message_id in streamed_ids:
                        continue
                    text = getattr(payload, "text", "") or ""
                    if text:
                        if message_id:
                            final_message_id = message_id
                            streamed_ids.add(message_id)
                        yield ServerSentEvent(data=text, event="message")

            # 2.2 custom 事件：累加并推送引用来源 / 产品 / 保单卡片
            elif method == "custom":
                data = params.get("data")
                if isinstance(data, dict) and data.get("type") == "additional_info":
                    incoming = data.get("data", {})
                    additional_info = merge_turn_additional_info(additional_info, incoming)
                    yield ServerSentEvent(data=incoming, event="additional_info")

        # 3. 持久化本轮附加信息到 Checkpointer 状态
        if final_message_id and additional_info:
            await self.agent.aupdate_state(
                config,
                {"additional_info": {final_message_id: additional_info}},
            )

        # 4. 流结束后检查快照是否存在挂起的 HITL interrupt，可靠推送给前端
        snapshot = await self.agent.aget_state(config)
        pending_interrupt = extract_pending_interrupt(snapshot)
        if pending_interrupt is not None:
            yield ServerSentEvent(data=pending_interrupt, event="interrupt")

        yield ServerSentEvent(data="[DONE]", event="done")
