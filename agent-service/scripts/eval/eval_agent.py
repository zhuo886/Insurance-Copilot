"""Agent 编排 4 套架构真实消融评测脚本（基于 12 款真实产品 + 异构强评审模型严评）。

用法：在 agent-service 目录执行
    uv run python scripts/eval/eval_agent.py
"""

import asyncio
import json
import os
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

from app.agents.guardrails import (
    extract_all_citation_tokens,
    extract_cited_refs,
    verify_plan_draft,
)
from app.agents.insurance_advisor import (
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
DEFAULT_JUDGE_MODEL = os.getenv("JUDGE_MODEL", "qwen-max")

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
只能选择一个主分支，请以严格合法的 JSON 格式返回 branch 字段。
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
    score: int = Field(ge=0, le=100, description="严格条款忠实度得分 0-100（100 分需每处条款数值与条件完全有原文支撑）")
    reason: str = Field(default="", description="逐条核对扣分说明")


@dataclass
class CaseRunOutput:
    arch_key: str
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
    faithfulness_reason: str = ""
    error: str | None = None


def _create_chat_model():
    return init_chat_model(
        model=settings.llm.chat_model,
        model_provider="openai",
        api_key=settings.llm.api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        extra_body={"thinking": {"type": "disabled"}},
    )


def _create_judge_model():
    """异构更强评审模型（默认 qwen-max），避免被测弱模型自评偏差。"""
    return init_chat_model(
        model=DEFAULT_JUDGE_MODEL,
        model_provider="openai",
        api_key=settings.llm.api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
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
        elif msg.name in ("recommend_insurance_portfolio", "quote_premium") and parsed.get("status") in ("ok", "ineligible"):
            used_rec_or_quote = True
        elif msg.name == "submit_plan_draft" and parsed.get("status") == "draft_submitted":
            draft = parsed.get("plan_draft")
            if isinstance(draft, dict):
                submitted_drafts.append(draft)

    return valid_source_ids, retrieved_sources, used_rag, used_rec_or_quote, submitted_drafts


# ---------------------------------------------------------------------------
# 架构 A / B1 / B2 构建
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
            if (tm.name == "query_product_clause" and parsed.get("status") == "ok") or parsed.get("source_ids"):
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
# 统一运行入口与异构 Judge 严格评判
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
            arch_key=arch_key,
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

        # 异构强模型严格 Faithfulness 评审（凡 expects_rag=True 未调 RAG 则严厉记 0 分）
        if case.get("expects_rag"):
            if out.error or not out.used_rag or not out.retrieved_sources or not out.answer_text.strip():
                out.faithfulness = 0.0
                out.faithfulness_reason = "未检索真实条款即直接作答（缺乏原文依据）" if not out.error else f"运行异常: {out.error}"
            else:
                try:
                    evidence_lines = [
                        f"[{sid}] ({info.get('clause_name', '')} | {info.get('section_path', '')}): {info.get('content', '')[:3000]}"
                        for sid, info in list(out.retrieved_sources.items())[:8]
                    ]
                    judge_prompt = (
                        f"用户问题：{case['query']}\n\n"
                        f"真实检索到的条款原文列表：\n{chr(10).join(evidence_lines)}\n\n"
                        f"被测系统回答：\n{out.answer_text}\n\n"
                        "严格扣分评审规则（从 100 分起扣）：\n"
                        "1. 请逐句核对回答中所有涉及条款数字、等待期、犹豫期、免赔额、赔付比例、责任免除、减保/退保/理赔材料规则的陈述。\n"
                        "2. 仅允许将上述真实检索条款中明确写出的条款细节视为有依据（注：如果问题还包含组合推荐/保费试算表格，推荐与价格数字不按条款幻觉扣分，但所有条款解读必须逐条核实）。\n"
                        "3. 扣分标准：\n"
                        "   - 每出现 1 处条款原文未提及或无法支撑的具体条款数字/条件断言：扣 20 分；\n"
                        "   - 每出现 1 处条款数值或赔付条件与原文矛盾冲突：扣 35 分；\n"
                        "   - 每出现 1 个虚假或不在上述 `[ref-xxx]` 列表里的引用编号：扣 30 分。\n"
                        "请严格公正打分（最低 0 分），并以合法 JSON 格式返回 `score` 与 `reason` 字段（在 reason 中简要写明每处扣分项或全部吻合依据）。"
                    )
                    score_obj = await judge_llm.ainvoke(
                        [
                            SystemMessage(content="你是极端严谨、客观中立的保险条款 RAG 忠实度审计专家。请严格以 JSON 格式输出 score（0-100 整数）与 reason 字段。"),
                            HumanMessage(content=judge_prompt),
                        ]
                    )
                    out.faithfulness = float(score_obj.score)
                    out.faithfulness_reason = str(getattr(score_obj, "reason", "") or "")
                except Exception as exc:
                    logger.warning("Faithfulness 评分跳过", case_id=case["id"], error=str(exc))

        return out


def compute_arch_metrics(
    arch_name: str,
    cases: list[dict[str, Any]],
    outputs: list[CaseRunOutput],
    catalog_map: dict[int, Any],
    common_rag_case_ids: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    case_map = {c["id"]: c for c in cases}

    compound_scores: list[float] = []
    rag_call_scores: list[float] = []
    mis_followup_scores: list[float] = []
    natural_citation_scores: list[float] = []
    zero_fake_ref_scores: list[float] = []
    zero_violation_scores: list[float] = []
    safe_resolution_scores: list[float] = []
    compliant_alt_draft_count = 0
    deterministic_block_count = 0
    faithfulness_all_scores: list[float] = []
    faithfulness_common_scores: list[float] = []
    weak_or_failed_cases: list[dict[str, Any]] = []

    followup_cues = ("请先提供", "请先告知", "请先补充", "告诉我您的", "年龄和预算", "补充以下信息", "请问被保人")
    advice_cues = ("元", "¥", "保费", "推荐", "方案", "组合", "险")
    refusal_cues = ("拦截", "不支持", "不符合", "超出", "无法", "不能", "超龄", "职业", "预算", "暂无", "不可投保", "拒绝")

    for out in outputs:
        case = case_map[out.case_id]
        text = out.answer_text or ""
        valid_set = set(out.valid_source_ids)
        cited_std = extract_cited_refs(text)
        all_tokens = extract_all_citation_tokens(text)
        invalid_tokens = [tok for tok in all_tokens if tok not in valid_set]

        # 1. 全集零虚构引用率（30 条全查，涵盖 [ref-f3b6f8e0] / [d64a1] 等任意伪造编号）
        zero_fake_ok = (not out.error) and (len(invalid_tokens) == 0)
        zero_fake_ref_scores.append(1.0 if zero_fake_ok else 0.0)

        # 2. 纯问答 RAG 调用率 vs 纯问答误追问率（拆开统计，不再混淆）
        if case["category"] == "pure_clause_qa":
            rag_call_scores.append(1.0 if out.used_rag else 0.0)
            is_mis_followup = any(cue in text for cue in followup_cues)
            mis_followup_scores.append(1.0 if is_mis_followup else 0.0)

        # 3. 自然自主标引率（所有检索到真实条款的期望 RAG 轮次，无后处理硬凑脚标）
        if case.get("expects_rag") and out.used_rag and valid_set:
            cit_ok = bool(cited_std) and not invalid_tokens
            natural_citation_scores.append(1.0 if cit_ok else 0.0)

        # 4. 复合意图完成率（10 条）
        if case["category"] == "compound_intent":
            has_rec = out.used_recommend_or_quote and any(c in text for c in advice_cues)
            has_clause = out.used_rag and bool(valid_set) and (bool(cited_std) and not invalid_tokens)
            compound_scores.append(1.0 if (has_rec and has_clause) else 0.0)

        # 5. 硬约束零违规率 & 合规解决率
        adv_type = case.get("adversarial_type")
        if case["category"] == "constraint_adversarial" or out.released_draft is not None:
            if out.error:
                zero_violation_scores.append(0.0)
                if case["category"] == "constraint_adversarial":
                    safe_resolution_scores.append(0.0)
            elif adv_type in ("fake_ref", "omit_ref"):
                cit_ok = out.used_rag and bool(valid_set) and bool(cited_std) and not invalid_tokens
                zero_violation_scores.append(1.0 if zero_fake_ok else 0.0)
                safe_resolution_scores.append(1.0 if cit_ok else 0.0)
            else:
                if out.released_draft is None:
                    zero_violation_scores.append(1.0)
                    has_clear_refusal = any(cue in text for cue in refusal_cues)
                    safe_resolution_scores.append(1.0 if has_clear_refusal else 0.0)
                    if has_clear_refusal:
                        deterministic_block_count += 1
                else:
                    errs = verify_plan_draft(
                        out.released_draft,
                        case.get("user_profile") or {},
                        catalog_map,
                    )
                    ok = len(errs) == 0
                    zero_violation_scores.append(1.0 if ok else 0.0)
                    if case["category"] == "constraint_adversarial":
                        safe_resolution_scores.append(1.0 if ok else 0.0)
                        if ok:
                            compliant_alt_draft_count += 1

        if out.faithfulness is not None:
            faithfulness_all_scores.append(out.faithfulness)
            if out.case_id in common_rag_case_ids:
                faithfulness_common_scores.append(out.faithfulness)

        # 记录异常、未满分忠实度或未达标边界 Case（用于报告坦诚剖析）
        issues: list[str] = []
        if out.error:
            issues.append(f"运行异常: {out.error}")
        if invalid_tokens:
            issues.append(f"出现伪造引文 {invalid_tokens}")
        if case.get("expects_rag") and not out.used_rag:
            issues.append("未调用 query_product_clause 检索即作答")
        if case.get("expects_rag") and out.used_rag and not cited_std:
            issues.append("检索到原文但正文未自主标注 [ref-xxx]")
        if out.faithfulness is not None and out.faithfulness < 90.0:
            issues.append(f"Faithfulness={out.faithfulness:.0f} ({out.faithfulness_reason[:80]})")
        if issues:
            weak_or_failed_cases.append({"case_id": out.case_id, "issues": issues})

    latencies = [o.latency_ms for o in outputs]
    tokens = [o.total_tokens for o in outputs]
    retries = sum(1 for o in outputs if o.guardrail_retry_count > 0)
    errors = sum(1 for o in outputs if o.error is not None)

    summary = {
        "architecture": arch_name,
        "arch_key": outputs[0].arch_key if outputs else "",
        "cases": len(outputs),
        "errors": errors,
        "compound_completion_rate": round(statistics.mean(compound_scores) * 100, 2) if compound_scores else 0.0,
        "pure_qa_rag_call_rate": round(statistics.mean(rag_call_scores) * 100, 2) if rag_call_scores else 0.0,
        "pure_qa_mis_followup_rate": round(statistics.mean(mis_followup_scores) * 100, 2) if mis_followup_scores else 0.0,
        "natural_citation_rate": round(statistics.mean(natural_citation_scores) * 100, 2) if natural_citation_scores else 0.0,
        "zero_fake_ref_rate": round(statistics.mean(zero_fake_ref_scores) * 100, 2),
        "zero_violation_rate": round(statistics.mean(zero_violation_scores) * 100, 2) if zero_violation_scores else 0.0,
        "safe_resolution_rate": round(statistics.mean(safe_resolution_scores) * 100, 2) if safe_resolution_scores else 0.0,
        "compliant_alt_draft_count": compliant_alt_draft_count,
        "deterministic_block_count": deterministic_block_count,
        "faithfulness_all": round(statistics.mean(faithfulness_all_scores), 2) if faithfulness_all_scores else 0.0,
        "faithfulness_common": round(statistics.mean(faithfulness_common_scores), 2) if faithfulness_common_scores else 0.0,
        "common_rag_samples": len(common_rag_case_ids),
        "avg_latency_ms": round(statistics.mean(latencies), 2),
        "avg_tokens": round(statistics.mean(tokens), 1),
        "guardrail_retry_cases": retries,
    }
    return summary, weak_or_failed_cases


def render_markdown_report(
    summaries: list[dict[str, Any]],
    c_weak_cases: list[dict[str, Any]],
) -> str:
    common_n = summaries[0]["common_rag_samples"] if summaries else 0
    lines = [
        "# Agent 编排 4 套架构真实消融对比报告（12 款真实产品 + 异构强模型严评）",
        "",
        f"- 评测集规模：`{summaries[0]['cases']}` 条（10 条纯条款问答 + 10 条一句话复合意图 + 10 条硬约束/引文对抗）",
        f"- 被测模型：`{settings.llm.chat_model}` | 异构 Faithfulness 评审模型：`{DEFAULT_JUDGE_MODEL}`",
        "- 统计口径声明：",
        "  1. 确定性引文清洗不再在文末强贴脚标，**自主有效标引率**如实反映模型自主遵循 `[ref-xxx]` 归因的能力；",
        "  2. **零虚构引用率**全集严查 `[ref-xxx]`/`[ref-f3b6f8e0]`/`[d64a1]` 等各类伪造编号；",
        "  3. **约束零违规率**与**对抗合规解决率**拆开披露（单列「提交合规替代草稿」与「确定性明确拒批」件数）；",
        f"  4. **Faithfulness（全集）**对未调用 RAG 直接凭记忆答条款记 `0.0` 分；**Faithfulness（公共交集）**仅统计 4 套架构均成功检索到条款的 `N={common_n}` 条公共用例。",
        "",
        "| 对比架构 | 复合意图完成 (%) | 纯问答调RAG (%) | 误追问 (%) ↓ | 自主标引率 (%) | 零虚构引用 (%) | 约束零违规 (%) | 对抗解决 (%) | Faith(全集) | Faith(交集) | 平均耗时 (ms) | 平均 Token |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for s in summaries:
        lines.append(
            f"| {s['architecture']} | {s['compound_completion_rate']:.2f} | {s['pure_qa_rag_call_rate']:.2f} | "
            f"{s['pure_qa_mis_followup_rate']:.2f} | {s['natural_citation_rate']:.2f} | "
            f"{s['zero_fake_ref_rate']:.2f} | {s['zero_violation_rate']:.2f} | "
            f"{s['safe_resolution_rate']:.2f} | {s['faithfulness_all']:.2f} | "
            f"{s['faithfulness_common']:.2f} | {s['avg_latency_ms']:.1f} | {s['avg_tokens']:.1f} |"
        )

    lines.extend(["", "## 各架构补充运行明细", ""])
    for s in summaries:
        lines.append(
            f"- **{s['architecture']}**：异常 `{s['errors']}` 条、触发护栏重试 `{s['guardrail_retry_cases']}` 条、"
            f"对抗草稿中 `{s['compliant_alt_draft_count']}` 条成功提交合规替代方案 / `{s['deterministic_block_count']}` 条给出确定性拒批说明"
        )

    lines.extend(["", "## 最终方案 C 边界与扣分 Case 诚实剖析", ""])
    if not c_weak_cases:
        lines.append("- 本轮评测中架构 C 全部期望 RAG 用例均检索原文并达标（`Faithfulness >= 90`）。")
    else:
        for item in c_weak_cases:
            lines.append(f"- **`{item['case_id']}`**：{'；'.join(item['issues'])}")

    return "\n".join(lines) + "\n"


async def main() -> None:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    catalog_map = {p.id: p for p in load_builtin_catalog()}

    async def _stub_save_quote(self, **kwargs: Any):
        return uuid4()

    orig_save_quote = InsurancePlanService.save_premium_quote
    InsurancePlanService.save_premium_quote = _stub_save_quote  # type: ignore[method-assign]

    judge_llm = _create_judge_model().with_structured_output(FaithfulnessScore)
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

    raw_outputs_by_arch: list[tuple[str, str, list[CaseRunOutput]]] = []
    try:
        for arch_key, arch_name, graph in arch_specs:
            logger.info("开始评测 Agent 架构", architecture=arch_name, cases=len(cases))
            outputs = await asyncio.gather(
                *(run_one_case(arch_key, arch_name, graph, case, judge_llm, sem) for case in cases)
            )
            raw_outputs_by_arch.append((arch_key, arch_name, outputs))
    finally:
        InsurancePlanService.save_premium_quote = orig_save_quote  # type: ignore[method-assign]
        close_rag()
        await close_database()

    # 计算 4 组架构均实际触发 RAG 的公共用例交集
    common_rag_ids: set[str] | None = None
    for _, _, outs in raw_outputs_by_arch:
        rag_ok_ids = {
            o.case_id
            for o in outs
            if o.error is None and o.used_rag and o.retrieved_sources and o.faithfulness is not None
        }
        common_rag_ids = rag_ok_ids if common_rag_ids is None else (common_rag_ids & rag_ok_ids)
    common_ids = common_rag_ids or set()

    summaries: list[dict[str, Any]] = []
    all_details: dict[str, list[dict[str, Any]]] = {}
    c_weak_cases: list[dict[str, Any]] = []

    for arch_key, arch_name, outputs in raw_outputs_by_arch:
        summary, weak_cases = compute_arch_metrics(
            arch_name,
            cases,
            outputs,
            catalog_map,
            common_ids,
        )
        summaries.append(summary)
        if arch_key == "C":
            c_weak_cases = weak_cases
        all_details[arch_key] = [
            {
                "case_id": o.case_id,
                "category": o.category,
                "used_rag": o.used_rag,
                "used_recommend_or_quote": o.used_recommend_or_quote,
                "valid_source_ids": o.valid_source_ids,
                "cited_refs": extract_cited_refs(o.answer_text),
                "all_citation_tokens": extract_all_citation_tokens(o.answer_text),
                "released_draft": o.released_draft,
                "guardrail_retry_count": o.guardrail_retry_count,
                "faithfulness": o.faithfulness,
                "faithfulness_reason": o.faithfulness_reason,
                "latency_ms": o.latency_ms,
                "total_tokens": o.total_tokens,
                "llm_calls": o.llm_calls,
                "error": o.error,
                "answer_preview": o.answer_text[:280],
            }
            for o in outputs
        ]
        logger.info("完成评测 Agent 架构", **summary)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    md_text = render_markdown_report(summaries, c_weak_cases)
    (REPORTS_DIR / "agent_eval.md").write_text(md_text, encoding="utf-8")
    (REPORTS_DIR / "agent_eval.json").write_text(
        json.dumps(
            {
                "summaries": summaries,
                "common_rag_case_ids": sorted(common_ids),
                "c_weak_cases": c_weak_cases,
                "details": all_details,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n" + md_text)


if __name__ == "__main__":
    asyncio.run(main())
