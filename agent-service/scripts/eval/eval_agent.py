"""Agent 编排 4 套架构真实消融评测脚本（基于 30 条纯问答/复合意图/对抗集）。

用法：在 agent-service 目录执行
    uv run python scripts/eval/eval_agent.py
"""

import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from langchain.chat_models import init_chat_model
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_core.outputs import LLMResult
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.agents.guardrails import extract_cited_refs, verify_plan_draft
from app.agents.insurance_advisor import (
    ADVISOR_SYSTEM_PROMPT,
    AdvisorState,
    _build_dynamic_system_prompt,
    _extract_latest_ai_message,
    _extract_latest_user_text,
    build_insurance_advisor_graph,
)
from app.agents.profile_extractor import extract_profile_by_regex
from app.agents.tools import (
    query_product_clause,
    quote_premium,
    recommend_insurance_portfolio,
    submit_plan_draft,
)
from app.core.config import settings
from app.core.logging import get_logger
from app.infra.database import close_database
from app.modules.chat.service import extract_pending_interrupt
from app.modules.insurance_plan.service import InsurancePlanService
from app.modules.product.catalog_data import load_builtin_catalog
from app.rag import close_rag

logger = get_logger(__name__)

CASES_PATH = Path(__file__).resolve().parent / "agent_cases.json"
REPORTS_DIR = PROJECT_ROOT / "reports"

ALL_TOOLS = [
    recommend_insurance_portfolio,
    query_product_clause,
    quote_premium,
    submit_plan_draft,
]

BASELINE_PROMPT = """
你是保险商城智能保险顾问，负责回答条款问题、推荐保险产品、试算保费并帮用户提交保险配置方案。
请注意：
1. 调用工具查资料、推荐或试算后再如实回答，尽量不要编造条款编号。
2. 当用户希望提交或保存方案草稿时，可调用 `submit_plan_draft`。
""".strip()

B1_ROUTER_PROMPT = """
你是刚性单路由分发器。请将用户请求严格单选分类到以下 3 个分支之一：
- clause_qa：纯粹只问保险产品条款细节；
- recommend：涉及产品推荐、组合配置、价格咨询或保费试算；
- plan_save：明确要求提交或保存配置方案草稿。
只能选择一个主分支。
""".strip()


class TokenTrackerCallback(AsyncCallbackHandler):
    """精准捕获被测架构真实 LLM Token 与调用次数（Judge 评测不挂载此回调）。"""

    def __init__(self) -> None:
        self.total_tokens = 0
        self.llm_calls = 0

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self.llm_calls += 1
        added = 0
        if response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    usage = getattr(msg, "usage_metadata", None) if msg is not None else None
                    if isinstance(usage, dict) and usage.get("total_tokens"):
                        added += int(usage["total_tokens"])
                    elif msg is not None and isinstance(getattr(msg, "response_metadata", None), dict):
                        tu = msg.response_metadata.get("token_usage") or msg.response_metadata.get("usage")
                        if isinstance(tu, dict) and tu.get("total_tokens"):
                            added += int(tu["total_tokens"])
        if added == 0 and isinstance(response.llm_output, dict):
            tu = response.llm_output.get("token_usage") or response.llm_output.get("usage") or {}
            if isinstance(tu, dict):
                added = int(
                    tu.get("total_tokens")
                    or ((tu.get("prompt_tokens") or 0) + (tu.get("completion_tokens") or 0))
                )
        self.total_tokens += added


class RigidRouteDecision(BaseModel):
    branch: Literal["clause_qa", "recommend", "plan_save"] = Field(description="单一路由分支")


class CriticDecision(BaseModel):
    passed: bool = Field(description="待审核回复与草稿是否完全合规且有真实依据")
    feedback: str = Field(default="", description="若不通过，给出具体修改要求；通过则留空")


class FaithfulnessScore(BaseModel):
    score: int = Field(ge=0, le=100, description="忠实度得分 0-100（100 表示所有陈述完全依据检索原文且无杜撰）")
    reason: str = Field(default="", description="一句话评分依据")


@dataclass
class CaseRunOutput:
    arch_name: str
    case_id: str
    category: str
    query: str
    answer_text: str = ""
    valid_source_ids: list[str] = field(default_factory=list)
    retrieved_sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    used_rag: bool = False
    used_recommend_or_quote: bool = False
    submitted_drafts: list[dict[str, Any]] = field(default_factory=list)
    released_draft: dict[str, Any] | None = None
    guardrail_retry_count: int = 0
    latency_ms: float = 0.0
    total_tokens: int = 0
    llm_calls: int = 0
    faithfulness: float | None = None
    error: str | None = None


def _create_chat_model():
    return init_chat_model(
        model=settings.llm.chat_model,
        model_provider="openai",
        api_key=settings.llm.api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        extra_body={"thinking": {"type": "disabled"}},
    )


async def _noop_plan_saver(
    *,
    user_id: int,
    draft: dict[str, Any],
    user_profile: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": "eval-plan-id",
        "plan_name": draft.get("plan_name", "评测方案"),
        "items": [
            {
                "id": f"eval-item-{idx}",
                "plan_id": "eval-plan-id",
                "product_id": int(item["product_id"]),
                "status": "uninsured",
            }
            for idx, item in enumerate(draft.get("items") or [], start=1)
        ],
    }


def _extract_tool_artifacts(messages: list[Any]) -> tuple[list[str], dict[str, dict[str, Any]], bool, bool, list[dict[str, Any]]]:
    valid_source_ids: list[str] = []
    retrieved_sources: dict[str, dict[str, Any]] = {}
    used_rag = False
    used_rec_or_quote = False
    submitted_drafts: list[dict[str, Any]] = []

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        raw = msg.content if isinstance(msg.content, str) else str(msg.content)
        try:
            parsed = json.loads(raw)
        except Exception:
            continue
        if not isinstance(parsed, dict):
            continue

        if msg.name == "query_product_clause" and parsed.get("status") == "ok":
            used_rag = True
            for sid in parsed.get("source_ids") or []:
                sid_str = str(sid)
                if sid_str not in valid_source_ids:
                    valid_source_ids.append(sid_str)
            sources = parsed.get("sources") or {}
            if isinstance(sources, dict):
                retrieved_sources.update(sources)
        elif msg.name in ("recommend_insurance_portfolio", "quote_premium") and parsed.get("status") == "ok":
            used_rec_or_quote = True
        elif msg.name == "submit_plan_draft" and parsed.get("status") == "draft_submitted":
            draft = parsed.get("plan_draft")
            if isinstance(draft, dict):
                submitted_drafts.append(draft)

    return valid_source_ids, retrieved_sources, used_rag, used_rec_or_quote, submitted_drafts


# ---------------------------------------------------------------------------
# 架构 A：原版单层 ReAct + Prompt 约束（Baseline）
# ---------------------------------------------------------------------------
def build_arch_a_graph():
    bound_model = _create_chat_model().bind_tools(ALL_TOOLS)
    tool_node = ToolNode(ALL_TOOLS)

    async def agent_node(state: AdvisorState, config: RunnableConfig) -> dict[str, Any]:
        msgs = [SystemMessage(content=BASELINE_PROMPT), *(state.get("messages") or [])]
        resp = await bound_model.ainvoke(msgs, config=config)
        return {"messages": [resp]}

    def route(state: AdvisorState) -> Literal["tools", "__end__"]:
        msgs = state.get("messages") or []
        if msgs and isinstance(msgs[-1], AIMessage) and getattr(msgs[-1], "tool_calls", None):
            return "tools"
        return END  # type: ignore[return-value]

    builder = StateGraph(AdvisorState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route)
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# 架构 B1：纯刚性状态机路由（Workflow-only：单分类 + 硬缺槽阻断 + 单分支单工具）
# ---------------------------------------------------------------------------
def build_arch_b1_graph():
    base = _create_chat_model()
    router_llm = base.with_structured_output(RigidRouteDecision)
    qa_model = base.bind_tools([query_product_clause])
    rec_model = base.bind_tools([recommend_insurance_portfolio, quote_premium])
    save_model = base.bind_tools([submit_plan_draft])

    qa_tools = ToolNode([query_product_clause])
    rec_tools = ToolNode([recommend_insurance_portfolio, quote_premium])
    save_tools = ToolNode([submit_plan_draft])

    async def router_node(
        state: AdvisorState, config: RunnableConfig
    ) -> Command[Literal["clause_qa", "slot_gate_rec", "slot_gate_save"]]:
        user_text = _extract_latest_user_text(state.get("messages") or [])
        decision = await router_llm.ainvoke(
            [SystemMessage(content=B1_ROUTER_PROMPT), HumanMessage(content=user_text)],
            config=config,
        )
        branch = decision.branch if isinstance(decision, RigidRouteDecision) else "clause_qa"
        if branch == "clause_qa":
            return Command(goto="clause_qa")
        if branch == "plan_save":
            return Command(goto="slot_gate_save")
        return Command(goto="slot_gate_rec")

    async def slot_gate_rec_node(
        state: AdvisorState,
    ) -> Command[Literal["rec_agent", "__end__"]]:
        user_text = _extract_latest_user_text(state.get("messages") or [])
        profile = extract_profile_by_regex(user_text)
        has_age = profile.get("age") is not None
        has_budget = profile.get("annual_budget") is not None
        has_cat = bool(profile.get("target_categories"))
        if not (has_age and (has_budget or has_cat)):
            return Command(
                update={
                    "messages": [
                        AIMessage(
                            id=str(uuid4()),
                            content="进入保险推荐或保费试算前，请先告知被保人年龄、是否有社保及年保费预算。",
                        )
                    ]
                },
                goto=END,
            )
        return Command(goto="rec_agent")

    async def slot_gate_save_node(
        state: AdvisorState,
    ) -> Command[Literal["save_agent", "__end__"]]:
        user_text = _extract_latest_user_text(state.get("messages") or [])
        profile = extract_profile_by_regex(user_text)
        missing = [
            label
            for key, label in (("age", "年龄"), ("occupation_class", "职业类别"), ("annual_budget", "年保费预算"))
            if profile.get(key) is None
        ]
        if missing:
            return Command(
                update={
                    "messages": [
                        AIMessage(
                            id=str(uuid4()),
                            content=f"保存保险方案前请先补充被保人完整信息：{'、'.join(missing)}。",
                        )
                    ]
                },
                goto=END,
            )
        return Command(goto="save_agent")

    def _make_agent(bound, prompt: str):
        async def _node(state: AdvisorState, config: RunnableConfig) -> dict[str, Any]:
            resp = await bound.ainvoke([SystemMessage(content=prompt), *(state.get("messages") or [])], config=config)
            return {"messages": [resp]}
        return _node

    def _make_route(tool_node_name: str):
        def _route(state: AdvisorState) -> str:
            msgs = state.get("messages") or []
            if msgs and isinstance(msgs[-1], AIMessage) and getattr(msgs[-1], "tool_calls", None):
                return tool_node_name
            return END
        return _route

    builder = StateGraph(AdvisorState)
    builder.add_node("router", router_node)
    builder.add_node("slot_gate_rec", slot_gate_rec_node)
    builder.add_node("slot_gate_save", slot_gate_save_node)
    builder.add_node("clause_qa", _make_agent(qa_model, "你是条款问答专员，只查询并回答保险条款问题。"))
    builder.add_node("qa_tools", qa_tools)
    builder.add_node("rec_agent", _make_agent(rec_model, "你是保险推荐试算专员，仅负责推荐组合与试算保费。"))
    builder.add_node("rec_tools", rec_tools)
    builder.add_node("save_agent", _make_agent(save_model, "你是方案提交专员，直接将确认的方案草稿调用 submit_plan_draft 提交。"))
    builder.add_node("save_tools", save_tools)

    builder.add_edge(START, "router")
    builder.add_conditional_edges("clause_qa", _make_route("qa_tools"))
    builder.add_edge("qa_tools", "clause_qa")
    builder.add_conditional_edges("rec_agent", _make_route("rec_tools"))
    builder.add_edge("rec_tools", "rec_agent")
    builder.add_conditional_edges("save_agent", _make_route("save_tools"))
    builder.add_edge("save_tools", "save_agent")
    return builder.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# 架构 B2：混合子图 + 全量 LLM-as-Critic 反思（每轮固定调一次 Critic LLM）
# ---------------------------------------------------------------------------
def build_arch_b2_graph(catalog_map: dict[int, Any]):
    base = _create_chat_model()
    bound_model = base.bind_tools(ALL_TOOLS)
    critic_llm = base.with_structured_output(CriticDecision)
    tool_node = ToolNode(ALL_TOOLS)

    async def extract_profile_node(state: AdvisorState) -> dict[str, Any]:
        latest = _extract_latest_user_text(state.get("messages") or [])
        return {
            "user_profile": extract_profile_by_regex(latest),
            "retrieved_source_ids": [],
            "pending_plan_draft": None,
            "guardrail_retry_count": 0,
            "guardrail_feedback": None,
            "last_rag_used": False,
            "verified_answer": None,
        }

    async def advisor_node(state: AdvisorState, config: RunnableConfig) -> dict[str, Any]:
        sys_prompt = _build_dynamic_system_prompt(state)
        resp = await bound_model.ainvoke([SystemMessage(content=sys_prompt), *(state.get("messages") or [])], config=config)
        return {"messages": [resp]}

    def route_agent(state: AdvisorState) -> Literal["tools", "llm_critic"]:
        msgs = state.get("messages") or []
        if msgs and isinstance(msgs[-1], AIMessage) and getattr(msgs[-1], "tool_calls", None):
            return "tools"
        return "llm_critic"

    async def collect_node(state: AdvisorState) -> dict[str, Any]:
        msgs = state.get("messages") or []
        sids = list(state.get("retrieved_source_ids") or [])
        draft = state.get("pending_plan_draft")
        rag_used = bool(state.get("last_rag_used"))
        recent: list[ToolMessage] = []
        for m in reversed(msgs):
            if isinstance(m, ToolMessage):
                recent.append(m)
            else:
                break
        for tm in reversed(recent):
            try:
                parsed = json.loads(tm.content if isinstance(tm.content, str) else str(tm.content))
            except Exception:
                continue
            if not isinstance(parsed, dict):
                continue
            if tm.name == "query_product_clause" or "source_ids" in parsed:
                rag_used = True
                for s in parsed.get("source_ids") or []:
                    if str(s) not in sids:
                        sids.append(str(s))
            if parsed.get("status") == "draft_submitted" and isinstance(parsed.get("plan_draft"), dict):
                draft = parsed["plan_draft"]
        return {"retrieved_source_ids": sids, "pending_plan_draft": draft, "last_rag_used": rag_used}

    async def llm_critic_node(state: AdvisorState, config: RunnableConfig) -> Command[Literal["advisor_agent", "__end__"]]:
        ai_msg = _extract_latest_ai_message(state.get("messages") or [])
        text = (ai_msg.text or str(ai_msg.content or "")) if ai_msg is not None else ""
        draft = state.get("pending_plan_draft")
        valid_ids = state.get("retrieved_source_ids") or []
        profile = state.get("user_profile") or {}
        retry_count = int(state.get("guardrail_retry_count") or 0)

        draft_brief: list[dict[str, Any]] = []
        if draft:
            for item in draft.get("items") or []:
                pid = int(item["product_id"])
                p = catalog_map.get(pid)
                draft_brief.append(
                    {
                        "product_id": pid,
                        "name": p.name if p else item.get("name"),
                        "allowed_age": f"{p.min_age}-{p.max_age}" if p else "unknown",
                        "max_occupation_class": p.max_occupation_class if p else "unknown",
                        "item_budget": item.get("annual_premium_budget"),
                    }
                )

        critic_prompt = (
            "你是严格的保险合规审稿官（LLM-Critic）。请核查以下 AI 回答与待提交草稿是否合规：\n"
            f"1. 用户原始请求：{_extract_latest_user_text(state.get('messages') or [])}\n"
            f"2. 已识别画像：{json.dumps(profile, ensure_ascii=False)}\n"
            f"3. 本轮是否调用条款检索={bool(state.get('last_rag_used'))}，合法条款编号={valid_ids}\n"
            f"4. 若已检索条款，回答正文中引用的 [ref-xxx] 编号必须完全来自合法列表且不得漏标。\n"
            f"5. 待提交方案草稿产品及承保条件：{json.dumps(draft_brief, ensure_ascii=False)}，总预估保费={draft.get('annual_premium_budget') if draft else None}\n"
            "   检查年龄、职业类别、年预算上限是否违规。\n"
            f"6. 待审 AI 回复正文：\n{text}"
        )
        decision = await critic_llm.ainvoke(
            [SystemMessage(content="请审稿并严格返回 JSON 判断。"), HumanMessage(content=critic_prompt)],
            config=config,
        )
        passed = decision.passed if isinstance(decision, CriticDecision) else True
        feedback = decision.feedback if isinstance(decision, CriticDecision) else ""

        if not passed and retry_count < 1:
            updates: dict[str, Any] = {
                "guardrail_feedback": feedback or "Critic 审稿未通过，请修正引文或产品约束。",
                "guardrail_retry_count": retry_count + 1,
                "pending_plan_draft": None if draft else state.get("pending_plan_draft"),
            }
            if ai_msg is not None and ai_msg.id:
                updates["messages"] = [RemoveMessage(id=ai_msg.id)]
            return Command(update=updates, goto="advisor_agent")

        if not passed and draft:
            return Command(update={"pending_plan_draft": None, "verified_answer": text}, goto=END)

        return Command(update={"verified_answer": text}, goto=END)

    builder = StateGraph(AdvisorState)
    builder.add_node("extract_profile", extract_profile_node)
    builder.add_node("advisor_agent", advisor_node)
    builder.add_node("tools", tool_node)
    builder.add_node("collect", collect_node)
    builder.add_node("llm_critic", llm_critic_node)
    builder.add_edge(START, "extract_profile")
    builder.add_edge("extract_profile", "advisor_agent")
    builder.add_conditional_edges("advisor_agent", route_agent)
    builder.add_edge("tools", "collect")
    builder.add_edge("collect", "advisor_agent")
    return builder.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# 统一运行入口与自动评判指标
# ---------------------------------------------------------------------------
async def run_one_case(
    arch_key: str,
    arch_name: str,
    graph: Any,
    case: dict[str, Any],
    judge_llm: Any,
    sem: asyncio.Semaphore,
) -> CaseRunOutput:
    async with sem:
        tracker = TokenTrackerCallback()
        cfg = {
            "configurable": {"thread_id": str(uuid4()), "user_id": 1},
            "callbacks": [tracker],
        }
        out = CaseRunOutput(
            arch_name=arch_name,
            case_id=case["id"],
            category=case["category"],
            query=case["query"],
        )
        t0 = time.perf_counter()
        try:
            await graph.ainvoke({"messages": [HumanMessage(content=case["query"])]}, config=cfg)
            snap = await graph.aget_state(cfg)
            values = getattr(snap, "values", {}) or {}
            messages = list(values.get("messages") or [])

            ai_msg = _extract_latest_ai_message(messages)
            out.answer_text = (ai_msg.text or str(ai_msg.content or "")) if ai_msg is not None else ""
            if not out.answer_text and values.get("verified_answer"):
                out.answer_text = str(values["verified_answer"])

            sids, sources, used_rag, used_rec_quote, drafts = _extract_tool_artifacts(messages)
            out.valid_source_ids = sids
            out.retrieved_sources = sources
            out.used_rag = used_rag
            out.used_recommend_or_quote = used_rec_quote
            out.submitted_drafts = drafts
            out.guardrail_retry_count = int(values.get("guardrail_retry_count") or 0)

            if arch_key in ("A", "B1"):
                out.released_draft = drafts[-1] if drafts else None
            else:
                out.released_draft = values.get("pending_plan_draft")
                pending_intr = extract_pending_interrupt(snap)
                if pending_intr and not out.released_draft and isinstance(pending_intr.get("plan"), dict):
                    out.released_draft = pending_intr["plan"]
        except Exception as exc:
            out.error = str(exc)
            logger.error("评测用例运行异常", arch=arch_name, case_id=case["id"], error=str(exc))
        finally:
            out.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            out.total_tokens = tracker.total_tokens
            out.llm_calls = tracker.llm_calls

        # 独立 Faithfulness Judge（不计入系统 Token 与耗时）
        if out.error is None and out.used_rag and out.retrieved_sources and out.answer_text:
            try:
                evidence_lines = [
                    f"[{sid}] ({info.get('clause_name', '')} | {info.get('section_path', '')}): {info.get('content', '')[:3000]}"
                    for sid, info in list(out.retrieved_sources.items())[:8]
                ]
                judge_prompt = (
                    f"用户问题：{case['query']}\n\n"
                    f"真实检索条款依据：\n{chr(10).join(evidence_lines)}\n\n"
                    f"被测系统回答：\n{out.answer_text}\n\n"
                    "评审要求：如果回答同时包含产品组合推荐/保费试算表格与条款细则解读，请仅针对保险条款细则"
                    "（如等待期、免赔额、外购药、续保条件、责任免除、理赔材料等）陈述部分评估其对上述真实条款原文的忠实度，"
                    "不要把搜推推荐的产品列表或保费试算数字误判为条款幻觉；"
                    "若出现编造 [ref-xxx] 编号或条款数值/条件与原文冲突则严厉扣分。\n"
                    "请以 0-100 打分：完全忠实且无杜撰给 90-100；轻微偏差或漏关键条件给 60-85；严重杜撰或与原文冲突给 0-50。"
                )
                score_obj = await judge_llm.ainvoke(
                    [SystemMessage(content="你是客观中立的 RAG 忠实度评审专家。"), HumanMessage(content=judge_prompt)]
                )
                out.faithfulness = float(score_obj.score)
            except Exception as exc:
                logger.warning("Faithfulness 评分跳过", case_id=case["id"], error=str(exc))

        return out


def compute_arch_metrics(
    arch_name: str,
    cases: list[dict[str, Any]],
    outputs: list[CaseRunOutput],
    catalog_map: dict[int, Any],
) -> dict[str, Any]:
    case_map = {c["id"]: c for c in cases}

    compound_scores: list[float] = []
    mis_followup_scores: list[float] = []
    valid_citation_scores: list[float] = []
    zero_hallucination_scores: list[float] = []
    constraint_pass_scores: list[float] = []
    faithfulness_scores: list[float] = []

    followup_cues = ("请先提供", "请先告知", "请先补充", "告诉我您的", "年龄和预算", "补充以下信息", "请问被保人")
    advice_cues = ("元", "¥", "保费", "推荐", "方案", "组合", "险")

    for out in outputs:
        case = case_map[out.case_id]
        text = out.answer_text or ""
        cited = extract_cited_refs(text)
        valid_set = set(out.valid_source_ids)

        # 1. 零虚构引用率（全集 30 条）
        has_fake_ref = any(r not in valid_set for r in cited)
        zero_hallucination_scores.append(0.0 if (out.error or has_fake_ref) else 1.0)

        # 2. 有效引用准确率（所有检索到真实条款的 RAG 轮次）
        if out.used_rag and valid_set:
            cit_ok = bool(cited) and not has_fake_ref
            valid_citation_scores.append(1.0 if cit_ok else 0.0)

        # 3. 纯问答误追问率（pure_clause_qa 10 条）
        if case["category"] == "pure_clause_qa":
            is_mis_followup = (not out.used_rag) or any(cue in text for cue in followup_cues)
            mis_followup_scores.append(1.0 if is_mis_followup else 0.0)

        # 4. 复合意图完成率（compound_intent 10 条）
        if case["category"] == "compound_intent":
            has_rec = out.used_recommend_or_quote and any(c in text for c in advice_cues)
            has_clause = out.used_rag and bool(valid_set) and (bool(cited) or len(text) >= 40)
            compound_scores.append(1.0 if (has_rec and has_clause) else 0.0)

        # 5. 硬约束合规率（对抗集 10 条 + 所有放行了草稿的用例）
        adv_type = case.get("adversarial_type")
        if case["category"] == "constraint_adversarial" or out.released_draft is not None:
            if out.error:
                constraint_pass_scores.append(0.0)
            elif adv_type in ("fake_ref", "omit_ref"):
                cit_ok = out.used_rag and bool(valid_set) and bool(cited) and not has_fake_ref
                constraint_pass_scores.append(1.0 if cit_ok else 0.0)
            else:
                if out.released_draft is None:
                    # 被成功拦阻，未放行任何违规方案
                    constraint_pass_scores.append(1.0)
                else:
                    errs = verify_plan_draft(
                        out.released_draft,
                        case.get("user_profile") or {},
                        catalog_map,
                    )
                    constraint_pass_scores.append(1.0 if not errs else 0.0)

        if out.faithfulness is not None:
            faithfulness_scores.append(out.faithfulness)

    latencies = [o.latency_ms for o in outputs]
    tokens = [o.total_tokens for o in outputs]
    retries = sum(1 for o in outputs if o.guardrail_retry_count > 0)
    errors = sum(1 for o in outputs if o.error is not None)

    return {
        "architecture": arch_name,
        "cases": len(outputs),
        "errors": errors,
        "compound_completion_rate": round(statistics.mean(compound_scores) * 100, 2) if compound_scores else 0.0,
        "pure_qa_mis_followup_rate": round(statistics.mean(mis_followup_scores) * 100, 2) if mis_followup_scores else 0.0,
        "valid_citation_precision": round(statistics.mean(valid_citation_scores) * 100, 2) if valid_citation_scores else 0.0,
        "zero_hallucination_ref_rate": round(statistics.mean(zero_hallucination_scores) * 100, 2),
        "hard_constraint_compliance_rate": round(statistics.mean(constraint_pass_scores) * 100, 2) if constraint_pass_scores else 0.0,
        "faithfulness": round(statistics.mean(faithfulness_scores), 2) if faithfulness_scores else 0.0,
        "faithfulness_samples": len(faithfulness_scores),
        "avg_latency_ms": round(statistics.mean(latencies), 2),
        "avg_tokens": round(statistics.mean(tokens), 1),
        "guardrail_retry_cases": retries,
    }


def render_markdown_report(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# Agent 编排 4 套架构真实消融对比报告",
        "",
        f"- 评测集规模：`{summaries[0]['cases']}` 条（10 条纯条款问答 + 10 条一句话复合意图 + 10 条硬约束/引文对抗）",
        f"- 基座模型：`{settings.llm.chat_model}`，真实调用本地 Postgres（50 款产品）+ Milvus 条款知识库",
        "- 统计口径：所有重试轮次（含 `RemoveMessage` 清掉的初稿与 B2 的 Critic 审稿）均如实计入总耗时与 Token 消耗；Faithfulness 由独立 Judge 依据真实检索原文评分",
        "",
        "| 对比架构 | 复合意图完成率 (%) | 纯问答误追问率 (%) ↓ | 有效引用准确率 (%) | 零虚构引用率 (%) | 硬约束合规率 (%) | 端到端 Faithfulness | 平均耗时 (ms) | 平均 Token 消耗 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for s in summaries:
        lines.append(
            f"| {s['architecture']} | {s['compound_completion_rate']:.2f} | "
            f"{s['pure_qa_mis_followup_rate']:.2f} | {s['valid_citation_precision']:.2f} | "
            f"{s['zero_hallucination_ref_rate']:.2f} | {s['hard_constraint_compliance_rate']:.2f} | "
            f"{s['faithfulness']:.2f} | {s['avg_latency_ms']:.2f} | {s['avg_tokens']:.1f} |"
        )
    lines.extend(["", "## 各架构补充运行明细", ""])
    for s in summaries:
        lines.append(
            f"- **{s['architecture']}**：异常 `{s['errors']}` 条、触发自检重试 `{s['guardrail_retry_cases']}` 条、"
            f"Faithfulness 有效评估 `{s['faithfulness_samples']}` 条"
        )
    return "\n".join(lines) + "\n"


async def main() -> None:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    catalog_map = {p.id: p for p in load_builtin_catalog()}

    # 评测期间避免把 120 次试算写入真实 PG 表
    async def _stub_save_quote(self, **kwargs: Any):
        return uuid4()

    orig_save_quote = InsurancePlanService.save_premium_quote
    InsurancePlanService.save_premium_quote = _stub_save_quote  # type: ignore[method-assign]

    judge_llm = _create_chat_model().with_structured_output(FaithfulnessScore)
    sem = asyncio.Semaphore(3)

    arch_specs = [
        ("A", "A. 单层 ReAct + Prompt 约束 (Baseline)", build_arch_a_graph()),
        ("B1", "B1. 纯刚性状态机路由 (Workflow-only)", build_arch_b1_graph()),
        ("B2", "B2. 混合子图 + 全量 LLM-Critic 反思", build_arch_b2_graph(catalog_map)),
        (
            "C",
            "C. 混合子图 + 确定性护栏按需单次纠偏 (最终方案)",
            build_insurance_advisor_graph(
                plan_saver=_noop_plan_saver,
                catalog_map=catalog_map,
                checkpointer=MemorySaver(),
            ),
        ),
    ]

    summaries: list[dict[str, Any]] = []
    all_details: dict[str, list[dict[str, Any]]] = {}

    try:
        for arch_key, arch_name, graph in arch_specs:
            logger.info("开始评测 Agent 架构", architecture=arch_name, cases=len(cases))
            outputs = await asyncio.gather(
                *(run_one_case(arch_key, arch_name, graph, case, judge_llm, sem) for case in cases)
            )
            summary = compute_arch_metrics(arch_name, cases, outputs, catalog_map)
            summaries.append(summary)
            all_details[arch_key] = [
                {
                    "case_id": o.case_id,
                    "category": o.category,
                    "used_rag": o.used_rag,
                    "used_recommend_or_quote": o.used_recommend_or_quote,
                    "valid_source_ids": o.valid_source_ids,
                    "cited_refs": extract_cited_refs(o.answer_text),
                    "released_draft": o.released_draft,
                    "guardrail_retry_count": o.guardrail_retry_count,
                    "faithfulness": o.faithfulness,
                    "latency_ms": o.latency_ms,
                    "total_tokens": o.total_tokens,
                    "llm_calls": o.llm_calls,
                    "error": o.error,
                    "answer_preview": o.answer_text[:280],
                }
                for o in outputs
            ]
            logger.info("完成评测 Agent 架构", **summary)
    finally:
        InsurancePlanService.save_premium_quote = orig_save_quote  # type: ignore[method-assign]
        close_rag()
        await close_database()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    md_text = render_markdown_report(summaries)
    (REPORTS_DIR / "agent_eval.md").write_text(md_text, encoding="utf-8")
    (REPORTS_DIR / "agent_eval.json").write_text(
        json.dumps({"summaries": summaries, "details": all_details}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n" + md_text)


if __name__ == "__main__":
    asyncio.run(main())
