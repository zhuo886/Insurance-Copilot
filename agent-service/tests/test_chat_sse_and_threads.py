from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from app.modules.chat.schemas import ChatRequest
from app.modules.chat.service import ChatService
from app.modules.chat_thread.exceptions import ChatThreadNotFoundError
from app.modules.chat_thread.models import ChatThread
from app.modules.chat_thread.service import ChatThreadService


def _fake_session_ctx():
    class _Tx:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    return SimpleNamespace(
        begin=lambda: _Tx(),
        flush=AsyncMock(),
        refresh=AsyncMock(),
    )


async def test_stream_chat_raises_not_found_for_unowned_thread(monkeypatch):
    service = ChatService(session=_fake_session_ctx(), agent=SimpleNamespace())  # type: ignore[arg-type]
    monkeypatch.setattr(service.repository, "find_owned", AsyncMock(return_value=None))

    with pytest.raises(ChatThreadNotFoundError):
        async for _ in service.stream_chat(1, ChatRequest(thread_id=uuid4(), message="你好")):
            pass


async def test_stream_chat_filters_route_and_buffers_until_guardrail_passes(monkeypatch):
    tid = uuid4()
    events_log: list[dict] = []

    class FakeAgent:
        async def astream_events(self, input_payload, config, version="v3", transformers=None):
            events_log.append({"input": input_payload, "config": config})

            async def _gen():
                # 1. 父图 route 内部输出必须过滤
                yield {
                    "method": "messages",
                    "params": {
                        "namespace": [],
                        "data": [
                            {"event": "content-block-delta", "delta": {"type": "text-delta", "text": '{"intent":"recommendation_plan"}'}},
                            {"langgraph_node": "route"},
                        ],
                    },
                }
                # 2. 工具触发 guardrail_buffer
                yield {"method": "custom", "params": {"data": {"type": "guardrail_buffer"}}}
                yield {
                    "method": "custom",
                    "params": {
                        "data": {
                            "type": "additional_info",
                            "data": {"clause_sources": {"ref-001": {"content": "等待期90天"}}},
                        }
                    },
                }
                # 3. 首次未通过护栏的草稿（被新 msg_id 清空）
                yield {
                    "method": "messages",
                    "params": {
                        "namespace": ["recommendation_plan"],
                        "data": [{"event": "message-start", "id": "draft-1"}, {"langgraph_node": "advisor_agent"}],
                    },
                }
                yield {
                    "method": "messages",
                    "params": {
                        "namespace": ["recommendation_plan"],
                        "data": [
                            {"event": "content-block-delta", "delta": {"type": "text-delta", "text": "错误草稿[ref-999]"}},
                            {"langgraph_node": "advisor_agent"},
                        ],
                    },
                }
                # 4. 纠偏后正式回答
                yield {
                    "method": "messages",
                    "params": {
                        "namespace": ["recommendation_plan"],
                        "data": [{"event": "message-start", "id": "final-1"}, {"langgraph_node": "advisor_agent"}],
                    },
                }
                yield {
                    "method": "messages",
                    "params": {
                        "namespace": ["recommendation_plan"],
                        "data": [
                            {"event": "content-block-delta", "delta": {"type": "text-delta", "text": "合规回答[ref-001]"}},
                            {"langgraph_node": "advisor_agent"},
                        ],
                    },
                }

            return _gen()

        async def aget_state(self, _config):
            return SimpleNamespace(interrupts=(), tasks=())

        aupdate_state = AsyncMock()

    agent = FakeAgent()
    service = ChatService(session=_fake_session_ctx(), agent=agent)  # type: ignore[arg-type]
    monkeypatch.setattr(service.repository, "find_owned", AsyncMock(return_value=SimpleNamespace(id=tid)))

    sse_events = [
        ev async for ev in service.stream_chat(7, ChatRequest(thread_id=tid, message="查等待期"))
    ]

    msg_payloads = [e.data for e in sse_events if e.event == "message"]
    assert msg_payloads == ["合规回答[ref-001]"]
    assert not any("错误草稿" in str(e.data) for e in sse_events)
    assert sse_events[-1].event == "done" and sse_events[-1].data == "[DONE]"
    agent.aupdate_state.assert_awaited_once()
    update_dict = agent.aupdate_state.call_args.args[1]
    assert "final-1" in update_dict["additional_info"]


async def test_stream_chat_emits_interrupt_without_overwriting_state_and_supports_resume(monkeypatch):
    tid = uuid4()
    intr_payload = {"plan": {"plan_name": "基础方案"}, "confirm_info": "请确认"}
    captured_inputs: list = []

    class FakeHitlAgent:
        async def astream_events(self, input_payload, config, version="v3", transformers=None):
            captured_inputs.append(input_payload)

            async def _gen():
                yield {
                    "method": "messages",
                    "params": {
                        "namespace": ["recommendation_plan"],
                        "data": [
                            AIMessage(id="verify-1", content="确定性替换提示文案"),
                            {"langgraph_node": "verify_response"},
                        ],
                    },
                }

            return _gen()

        async def aget_state(self, _config):
            return SimpleNamespace(interrupts=(SimpleNamespace(value=intr_payload),), tasks=())

        aupdate_state = AsyncMock()

    agent = FakeHitlAgent()
    service = ChatService(session=_fake_session_ctx(), agent=agent)  # type: ignore[arg-type]
    monkeypatch.setattr(service.repository, "find_owned", AsyncMock(return_value=SimpleNamespace(id=tid)))

    # 带 HITL interrupt 时不得调用 aupdate_state 覆写挂起态
    res1 = [ev async for ev in service.stream_chat(5, ChatRequest(thread_id=tid, message="提交方案"))]
    assert [e.event for e in res1] == ["message", "interrupt", "done"]
    assert res1[1].data == intr_payload
    agent.aupdate_state.assert_not_called()

    # 仅传 decision 时以 Command(resume=...) 恢复执行
    await anext(service.stream_chat(5, ChatRequest(thread_id=tid, decision={"action": "approve"})).__aiter__(), None)
    assert isinstance(captured_inputs[-1], Command)
    assert captured_inputs[-1].resume == {"action": "approve"}


async def test_chat_thread_service_crud_and_history_recovery(monkeypatch):
    tid = uuid4()
    thread = ChatThread(id=tid, user_id=2, title="原会话标题")
    fake_checkpointer = SimpleNamespace(adelete_thread=AsyncMock())
    snapshot_msgs = [
        HumanMessage(content="看看重疾险"),
        AIMessage(
            id="tool-call-msg",
            content="调用检索",
            tool_calls=[{"name": "query_product_clause", "args": {}, "id": "c1", "type": "tool_call"}],
        ),
        AIMessage(id="ai-final", content="等待期为90天[ref-001]。"),
    ]
    fake_agent = SimpleNamespace(
        checkpointer=fake_checkpointer,
        aget_state=AsyncMock(
            return_value=SimpleNamespace(
                values={
                    "messages": snapshot_msgs,
                    "additional_info": {"ai-final": {"clause_sources": {"ref-001": {"content": "90天"}}}},
                },
                interrupts=(SimpleNamespace(value={"plan": {"plan_name": "待审"}}),),
                tasks=(),
            )
        ),
    )

    service = ChatThreadService(session=_fake_session_ctx(), agent=fake_agent)  # type: ignore[arg-type]
    monkeypatch.setattr(service.repository, "add", AsyncMock())
    monkeypatch.setattr(service.repository, "list_by_user", AsyncMock(return_value=[thread]))
    monkeypatch.setattr(service.repository, "delete", AsyncMock())

    created = await service.add(2, "新会话")
    assert created.user_id == 2 and created.title == "新会话"
    assert len(await service.list_by_user(2)) == 1

    # 非本人会话不可重命名或删除
    monkeypatch.setattr(service.repository, "find_owned", AsyncMock(return_value=None))
    with pytest.raises(ChatThreadNotFoundError):
        await service.rename(tid, 99, "越权标题")
    with pytest.raises(ChatThreadNotFoundError):
        await service.delete(tid, 99)

    # 本人操作成功且删除同时清理 checkpoint
    monkeypatch.setattr(service.repository, "find_owned", AsyncMock(return_value=thread))
    renamed = await service.rename(tid, 2, "医疗险咨询")
    assert renamed.title == "医疗险咨询"

    history = await service.get_history(tid, 2)
    assert len(history.messages) == 2
    assert [m.role for m in history.messages] == ["user", "assistant"]
    assert history.messages[1].additional_info is not None
    assert history.interrupt == {"plan": {"plan_name": "待审"}}

    await service.delete(tid, 2)
    fake_checkpointer.adelete_thread.assert_awaited_once_with(str(tid))
