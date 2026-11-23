from uuid import UUID

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.chat.service import extract_pending_interrupt
from app.modules.chat_thread.exceptions import ChatThreadNotFoundError
from app.modules.chat_thread.models import ChatThread
from app.modules.chat_thread.repository import ChatThreadRepository
from app.modules.chat_thread.schemas import ChatHistoryResponse, ChatMessageResponse


class ChatThreadService:

    def __init__(
        self,
        session: AsyncSession,
        agent: CompiledStateGraph,
    ):
        self.session = session
        self.repository = ChatThreadRepository(session)
        self.agent = agent

    async def add(self, user_id: int, title: str) -> ChatThread:
        async with self.session.begin():
            chat_thread = ChatThread(user_id=user_id, title=title)
            await self.repository.add(chat_thread)
            return chat_thread

    async def list_by_user(self, user_id: int) -> list[ChatThread]:
        return await self.repository.list_by_user(user_id)

    async def rename(
        self,
        thread_id: UUID,
        user_id: int,
        title: str,
    ) -> ChatThread:
        async with self.session.begin():
            thread = await self.repository.find_owned(thread_id, user_id)
            if thread is None:
                raise ChatThreadNotFoundError

            thread.title = title
            await self.session.flush()
            await self.session.refresh(thread)

        return thread

    async def delete(
        self,
        thread_id: UUID,
        user_id: int,
    ) -> None:
        async with self.session.begin():
            thread = await self.repository.find_owned(thread_id, user_id)
            if thread is None:
                raise ChatThreadNotFoundError

            await self.agent.checkpointer.adelete_thread(str(thread_id))
            await self.repository.delete(thread)

    async def get_history(
        self,
        thread_id: UUID,
        user_id: int,
    ) -> ChatHistoryResponse:
        # 1. 校验会话归属
        thread = await self.repository.find_owned(thread_id, user_id)
        if thread is None:
            raise ChatThreadNotFoundError

        # 2. 读取 Agent 最新快照
        config = {"configurable": {"thread_id": str(thread_id), "user_id": user_id}}
        snapshot = await self.agent.aget_state(config)
        values = getattr(snapshot, "values", {}) or {}
        additional_infos = values.get("additional_info", {}) or {}

        # 3. 过滤出面向用户的消息与附加卡片
        messages: list[ChatMessageResponse] = []
        for message in values.get("messages", []):
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                if getattr(message, "tool_calls", None):
                    continue
                role = "assistant"
            else:
                continue

            if message.text:
                messages.append(
                    ChatMessageResponse(
                        role=role,
                        content=message.text,
                        additional_info=additional_infos.get(message.id),
                    )
                )

        # 4. 恢复挂起的 HITL interrupt
        pending_interrupt = extract_pending_interrupt(snapshot)

        return ChatHistoryResponse(
            thread_id=thread_id,
            messages=messages,
            interrupt=pending_interrupt,
        )
