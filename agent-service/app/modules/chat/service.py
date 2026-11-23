from collections.abc import AsyncIterator
from fastapi.sse import ServerSentEvent
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.stream import CustomTransformer
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.schemas import AdditionalInfo
from app.modules.chat.schemas import ChatRequest
from app.modules.chat_thread.exceptions import ChatThreadNotFoundError
from app.modules.chat_thread.repository import ChatThreadRepository
from app.core.logging import get_logger
logger = get_logger(__name__)

class ChatService:
    def __init__(self, session: AsyncSession, agent: CompiledStateGraph):
        self.repository = ChatThreadRepository(session)
        self.agent = agent

    async def stream_chat(
        self,
        user_id: int,
        request: ChatRequest,
    ) -> AsyncIterator[ServerSentEvent]:
        # 1.校验会话是否属于当前用户
        thread = await self.repository.find_owned(
            request.thread_id,
            user_id,
        )
        if thread is None:
            raise ChatThreadNotFoundError

        # 2.校验通过后，调用Agent

        # 2.1.准备Agent输入
        _input = {"messages": [HumanMessage(content=request.message)]}
        config = {"configurable": {"thread_id": str(request.thread_id)}}

        # 2.2.使用v3协议获取异步事件流
        stream = await self.agent.astream_events(
            _input,
            config,
            version="v3",
            transformers=[CustomTransformer],
        )

        # 2.2.1.记录AIMessage的ID和引用信息
        final_message_id: str | None = None
        additional_info: AdditionalInfo = {}

        # 已经通过 delta 逐字推送过的消息 id，避免 message-finish/AIMessage 重复推送
        streamed_ids: set[str] = set()

        async for event in stream:
            method = event.get("method")
            params = event.get("params") or {}
            # 只记录必要信息，避免把每个 token 都写进日志
            logger.debug(
                "stream event",
                method=method,
                namespace=params.get("namespace"),
            )

            # 2.3.1.消息事件：只推送面向用户的文本增量
            if method == "messages":
                namespace = params.get("namespace") or []
                items = params.get("data") or []
                if not items:
                    continue

                payload = items[0]
                metadata = items[1] if len(items) > 1 else {}

                # 父图的 route 节点属于内部逻辑，不推给前端（过滤掉 {"intent": ...}）
                if (
                    not namespace
                    and isinstance(metadata, dict)
                    and metadata.get("langgraph_node") == "route"
                ):
                    continue

                # 2.3.1.1.协议事件（流式增量）
                if isinstance(payload, dict):
                    evt = payload.get("event")

                    if evt == "message-start":
                        # 第一条片段会带上消息 id
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
                        # 这里带的是完整文本，前面已经逐字推送过，跳过避免重复
                        continue

                # 2.3.1.2.完整 AIMessage（模型没有流式输出时才会走到这里）
                elif isinstance(payload, AIMessage):
                    message_id = getattr(payload, "id", None)
                    if message_id and message_id in streamed_ids:
                        continue
                    text = getattr(payload, "text", "") or ""
                    if text:
                        yield ServerSentEvent(data=text, event="message")

            # 2.3.2.custom 事件：引用来源
            elif method == "custom":
                data = params.get("data")
                if isinstance(data, dict) and data.get("type") == "additional_info":
                    additional_info = data.get("data", {})
                    yield ServerSentEvent(data=additional_info, event="additional_info")

        if final_message_id and additional_info:
            # 将additional_info更新到state中
            await self.agent.aupdate_state(
                config,
                {"additional_info": {final_message_id: additional_info}}
            )
        # 2.6.返回结束标识
        yield ServerSentEvent(data="[DONE]", event="done")