from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END

from app.agents import orchestrator
from app.agents.chitchat import MAX_CHITCHAT_COUNT, ChitchatAgent
from app.agents.intent_router import (
    SOCIAL_RESPONSES,
    IntentRouter,
    match_social_intent,
    normalize_message,
    response_by_social_intent,
)
from app.agents.schemas import RouteResult


def test_normalize_message_strips_punctuation_and_lowercases():
    assert normalize_message("  Hello，世界！？ ") == "hello世界"
    assert normalize_message("谢谢你~！！") == "谢谢你"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("你好！", "greeting"),
        ("您好呀~", "greeting"),
        ("Hello!", "greeting"),
        ("在吗？", "greeting"),
        ("谢谢你，辛苦了", None),
        ("多谢！", "thanks"),
        ("感谢。", "thanks"),
        ("再见啦", None),
        ("拜拜！", "goodbye"),
        ("先这样。", "goodbye"),
        ("你好，我想买百万医疗险", None),
        ("谢谢，那达尔文12号等待期多久", None),
    ],
)
def test_match_social_intent_distinguishes_chitchat_from_business_queries(
    text: str,
    expected: str | None,
):
    assert match_social_intent(text) == expected


def test_response_by_social_intent_covers_all_templates():
    for key, tmpl in SOCIAL_RESPONSES.items():
        assert response_by_social_intent(key) == tmpl  # type: ignore[arg-type]


async def test_intent_router_bypasses_llm_on_social_messages():
    router = IntentRouter()

    class FailIfCalledModel:
        async def ainvoke(self, _messages):
            raise AssertionError("规则寒暄不应触发模型调用")

    router.model = FailIfCalledModel()
    result = await router.route("你好！", "")
    assert result == SOCIAL_RESPONSES["greeting"]


async def test_intent_router_returns_model_decision_and_falls_back_on_error():
    router = IntentRouter()

    class GoodModel:
        async def ainvoke(self, _messages):
            return RouteResult(intent="claim", reason="用户询问报案流程")

    router.model = GoodModel()
    res = await router.route("住院了怎么报案理赔", "上轮对话工作流:")
    assert isinstance(res, RouteResult)
    assert res.intent == "claim"

    class BrokenModel:
        async def ainvoke(self, _messages):
            raise RuntimeError("dashscope timeout")

    router.model = BrokenModel()
    fallback = await router.route("帮我看看这个情况", "")
    assert isinstance(fallback, RouteResult)
    assert fallback.intent == "fallback"


def test_find_previous_ai_message_ignores_tool_calls_and_latest_turn():
    messages = [
        HumanMessage(content="你好"),
        AIMessage(content="我是保险顾问"),
        HumanMessage(content="查条款"),
        AIMessage(
            content="中间工具调用",
            tool_calls=[{"name": "query_product_clause", "args": {}, "id": "t1", "type": "tool_call"}],
        ),
        HumanMessage(content="继续问"),
    ]
    assert orchestrator.find_previous_ai_message(messages) == "我是保险顾问"
    assert orchestrator.find_previous_ai_message([HumanMessage(content="首轮提问")]) == ""


async def test_route_node_direct_social_reply_and_five_way_dispatch(monkeypatch):
    async def fake_social_route(msg: str, ctx: str):
        return "固定问候"

    monkeypatch.setattr(orchestrator.intent_router, "route", fake_social_route)
    cmd_social = await orchestrator.route_node(
        {"messages": [HumanMessage(content="你好")], "active_workflow": "recommendation_plan"}
    )
    assert cmd_social.goto == END
    assert cmd_social.update["active_workflow"] == "chitchat"
    assert cmd_social.update["previous_workflow"] == "recommendation_plan"
    assert cmd_social.update["messages"][0].content == "固定问候"

    for target_intent in ("recommendation_plan", "claim", "chitchat", "human_handoff", "fallback"):
        async def fake_model_route(msg: str, ctx: str, intent=target_intent):
            return RouteResult(intent=intent, reason=f"route->{intent}")

        monkeypatch.setattr(orchestrator.intent_router, "route", fake_model_route)
        cmd = await orchestrator.route_node(
            {"messages": [HumanMessage(content="测试")], "active_workflow": "claim"}
        )
        assert cmd.goto == target_intent
        assert cmd.update["active_workflow"] == target_intent
        assert cmd.update["previous_workflow"] == "claim"


def test_human_handoff_and_fallback_nodes_format_context_cleanly():
    handoff = orchestrator.human_handoff_node(
        {
            "user_profile": {"age": 35, "annual_budget": 5000},
            "messages": [HumanMessage(content="我要投诉客服响应慢，马上安排人工专员处理")],
        }
    )
    handoff_text = handoff["messages"][0].content
    assert "35" in handoff_text
    assert "我要投诉客服响应慢" in handoff_text

    handoff_empty = orchestrator.human_handoff_node({"messages": []})
    assert "暂无" in handoff_empty["messages"][0].content

    fb_with_prev = orchestrator.fallback_node({"previous_workflow": "claim", "messages": []})
    assert "claim" in fb_with_prev["messages"][0].content

    fb_plain = orchestrator.fallback_node({"messages": []})
    assert "暂时没有明确理解" in fb_plain["messages"][0].content


async def test_chitchat_agent_counts_resets_and_enforces_limit():
    agent = ChitchatAgent()

    class FakeModel:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(self, _messages):
            self.calls += 1
            return AIMessage(content=f"闲聊第{self.calls}轮，顺便也可以了解重疾医疗保障。")

    fake_model = FakeModel()
    agent.model = fake_model

    # 上一轮非闲聊时，即使残留历史计数也重置归零
    res_reset = await agent.handle_chitchat(
        {
            "previous_workflow": "recommendation_plan",
            "chitchat_count": 2,
            "messages": [HumanMessage(content="今天天气不错")],
        }
    )
    assert res_reset["chitchat_count"] == 1
    assert fake_model.calls == 1

    # 连续闲聊累加
    res_cont = await agent.handle_chitchat(
        {
            "previous_workflow": "chitchat",
            "chitchat_count": 1,
            "messages": [HumanMessage(content="你喜欢吃什么")],
        }
    )
    assert res_cont["chitchat_count"] == 2
    assert fake_model.calls == 2

    # 达到 MAX_CHITCHAT_COUNT 直接触发边界收敛，不再请求 LLM
    res_limit = await agent.handle_chitchat(
        {
            "previous_workflow": "chitchat",
            "chitchat_count": MAX_CHITCHAT_COUNT,
            "messages": [HumanMessage(content="再讲个笑话吧")],
        }
    )
    assert fake_model.calls == 2
    assert "保险产品、投保和理赔咨询" in res_limit["messages"][0].content
