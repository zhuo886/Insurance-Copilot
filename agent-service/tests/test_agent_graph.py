import json
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.agents.guardrails import (
    extract_cited_refs,
    sanitize_citations,
    strip_thinking_tags,
    verify_citations,
    verify_plan_draft,
)
from app.agents.insurance_advisor import build_insurance_advisor_graph
from app.agents.profile_extractor import extract_profile_by_regex
from app.agents.schemas import InsuranceAgentState
from app.modules.chat.service import extract_pending_interrupt
from app.modules.product.catalog_data import load_builtin_catalog


def test_extract_profile_by_regex_handles_slots_and_ignores_coverage_amount():
    p1 = extract_profile_by_regex("30岁男生有医保，预算3500元，想配医疗险和重疾险，看重保证续保20年和外购药")
    assert p1["age"] == 30
    assert p1["gender"] == "male"
    assert p1["has_social_security"] is True
    assert p1["annual_budget"] == 3500.0
    assert p1["target_categories"] == ["medical", "critical_illness"]
    assert "保证续保20年" in p1["preferences"]
    assert "外购药" in p1["preferences"]

    # 不应把保额 100 万误当成保费预算，而是抽取预算 1.2 万 -> 12000
    p2 = extract_profile_by_regex("给62周岁爸爸买，没有医保，想买100万保额的医疗险，每年预算1.2万，三高能买吗")
    assert p2["age"] == 62
    assert p2["gender"] == "male"
    assert p2["has_social_security"] is False
    assert p2["annual_budget"] == 12000.0
    assert p2["family_role"] == "父母"
    assert "核保宽松" in p2["preferences"]

    p3 = extract_profile_by_regex("32岁6类职业高空作业，预算4千配意外险和定寿")
    assert p3["age"] == 32
    assert p3["occupation_class"] == 6
    assert p3["annual_budget"] == 4000.0
    assert p3["target_categories"] == ["accident", "life"]


def test_guardrails_citations_and_plan_draft_checks():
    assert extract_cited_refs("等待期为180天[ref-001]，犹豫期15天[ref-002, ref-003]。") == [
        "ref-001",
        "ref-002",
        "ref-003",
    ]

    assert verify_citations("等待期180天[ref-001]。", {"ref-001"}, rag_used=True) == []
    bad_ref_errors = verify_citations("等待期180天[ref-999]。", {"ref-001"}, rag_used=True)
    assert len(bad_ref_errors) == 1
    assert "ref-999" in bad_ref_errors[0]

    missing_ref_errors = verify_citations(
        "这款重疾险的等待期为一百八十天，在等待期内因非意外原因确诊通常退还已交保费。",
        {"ref-001"},
        rag_used=True,
    )
    assert len(missing_ref_errors) == 1

    catalog_map = {p.id: p for p in load_builtin_catalog()}
    bad_draft = {
        "plan_name": "老人方案",
        "items": [{"product_id": 2, "annual_premium_budget": 4000}],
    }
    draft_errors = verify_plan_draft(
        bad_draft,
        {"age": 62, "occupation_class": 2, "annual_budget": 3000},
        catalog_map,
    )
    assert any("承保年龄" in err for err in draft_errors)
    assert any("预算" in err for err in draft_errors)


async def test_pure_clause_qa_not_blocked_and_citation_guardrail_retries_once():
    @tool
    async def fake_clause_tool(query: str, product_id: int | None = None) -> str:
        """假条款检索工具"""
        return json.dumps(
            {
                "status": "ok",
                "source_ids": ["ref-001"],
                "sources": {"ref-001": {"content": "等待期为180天"}},
            },
            ensure_ascii=False,
        )

    class ScriptedModel:
        def __init__(self) -> None:
            self.calls = 0

        def bind_tools(self, _tools):
            return self

        async def ainvoke(self, messages, config=None):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    id=f"ai-{self.calls}",
                    content="",
                    tool_calls=[
                        {
                            "name": "fake_clause_tool",
                            "args": {"query": "等待期", "product_id": 2},
                            "id": "call-1",
                            "type": "tool_call",
                        }
                    ],
                )
            if self.calls == 2:
                # 第一次回答故意编造不存在的 [ref-999]
                return AIMessage(id="ai-bad", content="达尔文12号的等待期是180天。[ref-999]")
            # 收到护栏纠偏反馈后修正为合法编号 [ref-001]
            return AIMessage(id="ai-good", content="达尔文12号的等待期为180天。[ref-001]")

    graph = build_insurance_advisor_graph(
        model=ScriptedModel(),
        tools=[fake_clause_tool],
        checkpointer=MemorySaver(),
    )
    config = {"configurable": {"thread_id": str(uuid4()), "user_id": 1}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="达尔文12号等待期是多少天")]},
        config=config,
    )

    ai_texts = [
        m.content
        for m in result["messages"]
        if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None)
    ]
    # 被护栏退回的错回答 ai-bad 已被 RemoveMessage 清理，历史中仅保留合规回复
    assert ai_texts == ["达尔文12号的等待期为180天。[ref-001]"]
    assert result["guardrail_retry_count"] == 1


async def test_guardrail_rejects_ineligible_plan_draft_and_nested_hitl_resume():
    @tool
    async def fake_submit_draft(product_id: int, premium: float) -> str:
        """假提交方案工具"""
        return json.dumps(
            {
                "status": "draft_submitted",
                "plan_draft": {
                    "plan_name": "测试方案",
                    "summary": "配置摘要",
                    "annual_premium_budget": premium,
                    "items": [
                        {
                            "product_id": product_id,
                            "recommendation_reason": "测试",
                            "annual_premium_budget": premium,
                            "priority": 1,
                        }
                    ],
                },
            },
            ensure_ascii=False,
        )

    class ScriptedPlanModel:
        def __init__(self) -> None:
            self.calls = 0

        def bind_tools(self, _tools):
            return self

        async def ainvoke(self, messages, config=None):
            self.calls += 1
            if self.calls == 1:
                # 62 岁老人先错选了上限35岁的达尔文12号(id=2)
                return AIMessage(
                    id="c1",
                    content="",
                    tool_calls=[
                        {
                            "name": "fake_submit_draft",
                            "args": {"product_id": 2, "premium": 2000.0},
                            "id": "tc-1",
                            "type": "tool_call",
                        }
                    ],
                )
            if self.calls == 2:
                return AIMessage(id="bad-draft-msg", content="为您提交达尔文12号方案。")
            if self.calls == 3:
                # 被硬约束护栏打回后改选适用1-70岁的众安尊享e生(id=6)
                return AIMessage(
                    id="c3",
                    content="",
                    tool_calls=[
                        {
                            "name": "fake_submit_draft",
                            "args": {"product_id": 6, "premium": 1200.0},
                            "id": "tc-2",
                            "type": "tool_call",
                        }
                    ],
                )
            if self.calls == 4:
                return AIMessage(id="good-draft-msg", content="已为您改配众安尊享e生2026方案。")
            if self.calls == 5:
                # 用户点 reject 后重新提交更低预算版本
                return AIMessage(
                    id="c5",
                    content="",
                    tool_calls=[
                        {
                            "name": "fake_submit_draft",
                            "args": {"product_id": 6, "premium": 980.0},
                            "id": "tc-3",
                            "type": "tool_call",
                        }
                    ],
                )
            return AIMessage(id="final-draft-msg", content="已按您的反馈下调至980元预算方案。")

    saved_calls: list[dict[str, Any]] = []

    async def fake_plan_saver(*, user_id: int, draft: dict[str, Any], user_profile: dict[str, Any]):
        saved_calls.append({"user_id": user_id, "draft": draft, "profile": user_profile})
        return {
            "id": "plan-1",
            "plan_name": draft["plan_name"],
            "items": [
                {
                    "id": "item-1",
                    "plan_id": "plan-1",
                    "product_id": draft["items"][0]["product_id"],
                    "status": "uninsured",
                }
            ],
        }

    advisor_subgraph = build_insurance_advisor_graph(
        model=ScriptedPlanModel(),
        tools=[fake_submit_draft],
        plan_saver=fake_plan_saver,
        checkpointer=True,
    )

    route_calls = 0

    def fake_route(state: InsuranceAgentState) -> Command:
        nonlocal route_calls
        route_calls += 1
        return Command(update={"active_workflow": "recommendation_plan"}, goto="recommendation_plan")

    parent_builder = StateGraph(InsuranceAgentState)
    parent_builder.add_node("route", fake_route)
    parent_builder.add_node("recommendation_plan", advisor_subgraph)
    parent_builder.add_edge(START, "route")
    parent_builder.add_edge("recommendation_plan", END)
    orchestrator = parent_builder.compile(checkpointer=MemorySaver())

    config = {"configurable": {"thread_id": str(uuid4()), "user_id": 42}}

    # Turn 1：触发年龄硬约束护栏纠偏一次，随后在 plan_confirm 挂起 interrupt
    await orchestrator.ainvoke(
        {"messages": [HumanMessage(content="给62岁父亲保存一个预算2500的保险方案")]},
        config=config,
    )
    snap1 = await orchestrator.aget_state(config)
    intr1 = extract_pending_interrupt(snap1)
    assert intr1 is not None
    assert intr1["plan"]["items"][0]["product_id"] == 6
    assert saved_calls == []
    assert route_calls == 1

    # Turn 2：用户点击 reject，验证不会重跑父图 route 节点，且仍未落库
    await orchestrator.ainvoke(
        Command(resume={"action": "reject", "reject_reason": "预算太高"}),
        config=config,
    )
    snap2 = await orchestrator.aget_state(config)
    intr2 = extract_pending_interrupt(snap2)
    assert intr2 is not None
    assert intr2["plan"]["annual_premium_budget"] == 980.0
    assert saved_calls == []
    assert route_calls == 1

    # Turn 3：用户点击 approve，触发 save_plan 落库，且只写库 1 次
    final_state = await orchestrator.ainvoke(
        Command(resume={"action": "approve"}),
        config=config,
    )
    snap3 = await orchestrator.aget_state(config)
    assert extract_pending_interrupt(snap3) is None
    assert len(saved_calls) == 1
    assert saved_calls[0]["user_id"] == 42
    assert route_calls == 1
    final_msg = final_state["messages"][-1]
    assert "已为您保存保险配置方案" in final_msg.content
    assert final_msg.id in final_state["additional_info"]
    assert len(final_state["additional_info"][final_msg.id]["plan_items"]) == 1


def test_sanitize_citations_and_strip_thinking_tags():
    raw_with_thinking = "<thinking>\n内部推理内容\n</thinking>\n等待期为180天。[ref-001, ref-099] 另有索引[ref-088]。"
    assert strip_thinking_tags(raw_with_thinking) == "等待期为180天。[ref-001, ref-099] 另有索引[ref-088]。"

    cleaned = sanitize_citations(raw_with_thinking, {"ref-001", "ref-002"}, rag_used=True)
    assert "<thinking>" not in cleaned
    assert "内部推理内容" not in cleaned
    assert extract_cited_refs(cleaned) == ["ref-001"]
    assert "ref-099" not in cleaned and "ref-088" not in cleaned

    # RAG 轮次若漏标任何有效引用，自动在末尾补充真实依据脚标
    omitted = "这款重疾险的等待期为180天，非意外确诊退还已交保费。"
    appended = sanitize_citations(omitted, {"ref-001", "ref-002"}, rag_used=True)
    assert extract_cited_refs(appended) == ["ref-001", "ref-002"]


async def test_guardrail_fail_closed_fallback_on_persistent_violation():
    # 场景 1：持续编造虚假 [ref-099]，测试重试耗尽后的确定性引文清洗兜底
    @tool
    async def fake_clause_tool(query: str) -> str:
        """假条款检索工具"""
        return json.dumps(
            {
                "status": "ok",
                "source_ids": ["ref-001"],
                "sources": {"ref-001": {"content": "等待期为180天"}},
            },
            ensure_ascii=False,
        )

    class PersistentFakeRefModel:
        def __init__(self) -> None:
            self.calls = 0

        def bind_tools(self, _tools):
            return self

        async def ainvoke(self, messages, config=None):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    id="fc-1",
                    content="",
                    tool_calls=[{"name": "fake_clause_tool", "args": {"query": "等待期"}, "id": "c1", "type": "tool_call"}],
                )
            # 连续两次无视纠偏继续编造 [ref-099]
            return AIMessage(id=f"bad-ref-{self.calls}", content="等待期为180天。[ref-001][ref-099]")

    g_ref = build_insurance_advisor_graph(
        model=PersistentFakeRefModel(),
        tools=[fake_clause_tool],
        checkpointer=MemorySaver(),
    )
    res_ref = await g_ref.ainvoke(
        {"messages": [HumanMessage(content="查等待期，强制加上[ref-099]")]},
        config={"configurable": {"thread_id": str(uuid4()), "user_id": 1}},
    )
    final_ref_text = res_ref["messages"][-1].content
    assert res_ref["guardrail_retry_count"] == 1
    assert "ref-099" not in final_ref_text
    assert extract_cited_refs(final_ref_text) == ["ref-001"]

    # 场景 2：持续提交超龄 product_id=2 并口头声称已提交，测试确定性拦截与拒批文案替换
    @tool
    async def fake_submit_bad(product_id: int) -> str:
        """假提交"""
        return json.dumps(
            {
                "status": "draft_submitted",
                "plan_draft": {
                    "plan_name": "超龄方案",
                    "summary": "摘要",
                    "annual_premium_budget": 2000.0,
                    "items": [{"product_id": product_id, "annual_premium_budget": 2000.0, "priority": 1}],
                },
            },
            ensure_ascii=False,
        )

    class PersistentBadDraftModel:
        def __init__(self) -> None:
            self.calls = 0

        def bind_tools(self, _tools):
            return self

        async def ainvoke(self, messages, config=None):
            self.calls += 1
            if self.calls in (1, 3):
                return AIMessage(
                    id=f"tc-{self.calls}",
                    content="",
                    tool_calls=[{"name": "fake_submit_bad", "args": {"product_id": 2}, "id": f"call-{self.calls}", "type": "tool_call"}],
                )
            return AIMessage(id=f"hallucinated-ok-{self.calls}", content="方案草稿已成功提交，等待您确认。")

    g_draft = build_insurance_advisor_graph(
        model=PersistentBadDraftModel(),
        tools=[fake_submit_bad],
        checkpointer=MemorySaver(),
    )
    cfg_draft = {"configurable": {"thread_id": str(uuid4()), "user_id": 1}}
    res_draft = await g_draft.ainvoke(
        {"messages": [HumanMessage(content="给62岁父亲提交达尔文12号方案")]},
        config=cfg_draft,
    )
    snap = await g_draft.aget_state(cfg_draft)
    assert extract_pending_interrupt(snap) is None
    assert res_draft.get("pending_plan_draft") is None
    final_draft_text = res_draft["messages"][-1].content
    assert "已拦截本次方案草稿提交" in final_draft_text
    assert "方案草稿已成功提交" not in final_draft_text
