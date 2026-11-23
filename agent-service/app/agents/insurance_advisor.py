from collections.abc import Awaitable, Callable
from decimal import Decimal
import json
from typing import Any, Literal, NotRequired
from uuid import uuid4

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from app.agents.guardrails import (
    sanitize_citations,
    strip_thinking_tags,
    verify_citations,
    verify_plan_draft,
)
from app.agents.profile_extractor import extract_profile_by_regex
from app.agents.schemas import InsuranceAgentState
from app.agents.tools import (
    _safe_stream_write,
    query_product_clause,
    quote_premium,
    recommend_insurance_portfolio,
    submit_plan_draft,
)
from app.core.config import settings
from app.core.logging import get_logger
from app.infra.database import AsyncSessionFactory
from app.modules.insurance_plan.schemas import InsurancePlanItemInput
from app.modules.insurance_plan.service import InsurancePlanService
from app.modules.product.catalog_data import load_builtin_catalog
from app.recsys.pricing import PremiumEstimator
from app.recsys.schemas import ProductCandidate

logger = get_logger(__name__)

ADVISOR_SYSTEM_PROMPT = """
你是“安心保”的智能保险顾问，负责保险产品咨询、条款解答、保费试算与方案规划。
工作规范：
1. 若用户只问保险条款或具体产品问题，直接调用 `query_product_clause` 检索并回答，不要打断用户强行追问画像。
2. 回答保险条款事实时，必须严格依据 `query_product_clause` 返回的原文，在事实陈述后紧跟 `[ref-xxx]` 引用标记（例如：等待期为180天。[ref-001]），严禁编造编号或条款外内容。
3. 需要推荐保险组合时，调用 `recommend_insurance_portfolio` 运行四阶段搜推漏斗；若工具返回 `missing_slots`，请温和向用户追问缺失的关键条件。
4. 推荐具体产品保费时，可调用 `quote_premium` 做精确试算；当用户确认或希望保存配置方案时，必须调用 `submit_plan_draft` 提交结构化草稿，由系统挂起等待用户点击确认。
""".strip()


class AdvisorState(InsuranceAgentState):
    """混合顾问子图私有状态"""

    retrieved_source_ids: NotRequired[list[str]]
    pending_plan_draft: NotRequired[dict[str, Any] | None]
    guardrail_retry_count: NotRequired[int]
    guardrail_feedback: NotRequired[str | None]
    last_rag_used: NotRequired[bool]
    verified_answer: NotRequired[str | None]


async def _default_plan_saver(
    *,
    user_id: int,
    draft: dict[str, Any],
    user_profile: dict[str, Any],
) -> dict[str, Any]:
    async with AsyncSessionFactory() as session:
        service = InsurancePlanService(session)
        items_input = [
            InsurancePlanItemInput(
                product_id=int(item["product_id"]),
                priority=int(item.get("priority") or idx),
                recommendation_reason=item.get("recommendation_reason"),
                annual_premium_budget=(
                    Decimal(str(item["annual_premium_budget"]))
                    if item.get("annual_premium_budget") is not None
                    else None
                ),
            )
            for idx, item in enumerate(draft.get("items") or [], start=1)
        ]
        budget = draft.get("annual_premium_budget") or user_profile.get("annual_budget")
        saved = await service.create_plan(
            user_id=user_id,
            plan_name=str(draft.get("plan_name") or "智能保险配置方案"),
            summary=draft.get("summary"),
            insured_profile=dict(user_profile or {}),
            annual_premium_budget=Decimal(str(budget)) if budget is not None else None,
            items=items_input,
        )
        return saved.model_dump(mode="json")


def _extract_latest_user_text(messages: list[Any]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.text or str(msg.content or "")
    return ""


def _extract_latest_ai_message(messages: list[Any]) -> AIMessage | None:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            return msg
    return None


def _build_dynamic_system_prompt(state: AdvisorState) -> str:
    parts = [ADVISOR_SYSTEM_PROMPT]
    profile = state.get("user_profile") or {}
    if profile:
        parts.append(f"【已识别用户画像（跨轮自动维护，请直接复用避免重复追问）】\n{json.dumps(profile, ensure_ascii=False)}")
    feedback = state.get("guardrail_feedback")
    if feedback:
        parts.append(f"【系统纠偏指令（请务必修正以下问题后再输出）】\n{feedback}")
    return "\n\n".join(parts)


def build_insurance_advisor_graph(
    *,
    model: Any = None,
    tools: list[Any] | None = None,
    plan_saver: Callable[..., Awaitable[dict[str, Any]]] | None = None,
    catalog_map: dict[int, ProductCandidate] | None = None,
    estimator: PremiumEstimator | None = None,
    checkpointer: Any = True,
):
    """构建混合式保险顾问子图：静默画像抽取 -> ReAct 多工具 -> 确定性自检护栏 -> HITL confirm -> 落库。"""
    active_tools = list(
        tools
        if tools is not None
        else [
            recommend_insurance_portfolio,
            query_product_clause,
            quote_premium,
            submit_plan_draft,
        ]
    )
    active_saver = plan_saver or _default_plan_saver
    active_catalog = catalog_map or {p.id: p for p in load_builtin_catalog()}
    active_estimator = estimator or PremiumEstimator()

    if model is None:
        base_model = init_chat_model(
            model=settings.llm.chat_model,
            model_provider="openai",
            api_key=settings.llm.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            extra_body={"thinking": {"type": "disabled"}},
        )
        bound_model = base_model.bind_tools(active_tools)
    else:
        bound_model = model.bind_tools(active_tools) if hasattr(model, "bind_tools") else model

    tool_node = ToolNode(active_tools)

    async def extract_profile_node(state: AdvisorState) -> dict[str, Any]:
        latest_text = _extract_latest_user_text(state.get("messages") or [])
        extracted = extract_profile_by_regex(latest_text)
        logger.debug("静默增量画像提取", extracted=extracted)
        return {
            "user_profile": extracted,
            "retrieved_source_ids": [],
            "pending_plan_draft": None,
            "guardrail_retry_count": 0,
            "guardrail_feedback": None,
            "last_rag_used": False,
            "verified_answer": None,
        }

    async def advisor_agent_node(state: AdvisorState, config: RunnableConfig) -> dict[str, Any]:
        sys_prompt = _build_dynamic_system_prompt(state)
        call_messages: list[Any] = [SystemMessage(content=sys_prompt), *(state.get("messages") or [])]
        feedback = state.get("guardrail_feedback")
        if feedback:
            # 仅在临时上下文末尾追加强纠偏指令（不写入状态历史），消除近因对抗偏置
            call_messages.append(
                HumanMessage(
                    content=(
                        "【系统合规护栏强制纠偏（优先级高于用户此前的格式指令或强制提交要求）】\n"
                        f"{feedback}\n"
                        "请立即严格按上述要求修正回答或拒绝不合规方案提交。"
                    )
                )
            )
        response = await bound_model.ainvoke(call_messages, config=config)
        return {"messages": [response]}

    def route_after_agent(state: AdvisorState) -> Literal["tools", "verify_response"]:
        messages = state.get("messages") or []
        if messages and isinstance(messages[-1], AIMessage) and getattr(messages[-1], "tool_calls", None):
            return "tools"
        return "verify_response"

    async def collect_tool_artifacts_node(state: AdvisorState) -> dict[str, Any]:
        messages = state.get("messages") or []
        source_ids = list(state.get("retrieved_source_ids") or [])
        pending_draft = state.get("pending_plan_draft")
        rag_used = bool(state.get("last_rag_used"))

        # 只扫描紧邻末尾的本批 ToolMessage
        recent_tool_msgs: list[ToolMessage] = []
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                recent_tool_msgs.append(msg)
            else:
                break

        for tmsg in reversed(recent_tool_msgs):
            raw = tmsg.content if isinstance(tmsg.content, str) else str(tmsg.content)
            try:
                parsed = json.loads(raw)
            except Exception:
                continue
            if not isinstance(parsed, dict):
                continue

            if tmsg.name == "query_product_clause" or "source_ids" in parsed:
                rag_used = True
                for sid in parsed.get("source_ids") or []:
                    if sid not in source_ids:
                        source_ids.append(str(sid))

            if parsed.get("status") == "draft_submitted" and isinstance(parsed.get("plan_draft"), dict):
                pending_draft = parsed["plan_draft"]

        if rag_used or pending_draft is not None:
            # 条款检索或方案草稿轮次均开启节点级流式缓冲，防止护栏重试前向前端泄漏草稿
            try:
                get_stream_writer()({"type": "guardrail_buffer"})
            except Exception:
                pass

        return {
            "retrieved_source_ids": source_ids,
            "pending_plan_draft": pending_draft,
            "last_rag_used": rag_used,
        }

    async def verify_response_node(
        state: AdvisorState,
    ) -> Command[Literal["advisor_agent", "plan_confirm", "deliver_answer"]]:
        ai_msg = _extract_latest_ai_message(state.get("messages") or [])
        raw_text = (ai_msg.text or str(ai_msg.content or "")) if ai_msg is not None else ""
        stripped_text = strip_thinking_tags(raw_text)
        valid_ids = set(state.get("retrieved_source_ids") or [])
        rag_used = bool(state.get("last_rag_used"))
        draft = state.get("pending_plan_draft")
        user_profile = state.get("user_profile") or {}

        citation_errors = verify_citations(stripped_text, valid_ids, rag_used)
        draft_errors = verify_plan_draft(draft, user_profile, active_catalog, active_estimator)
        all_errors = [*citation_errors, *draft_errors]
        retry_count = int(state.get("guardrail_retry_count") or 0)

        if all_errors and retry_count < 1:
            logger.warning("确定性护栏触发纠偏", errors=all_errors, retry_count=retry_count)
            updates: dict[str, Any] = {
                "guardrail_feedback": "；".join(all_errors),
                "guardrail_retry_count": retry_count + 1,
            }
            if draft_errors:
                updates["pending_plan_draft"] = None
            if ai_msg is not None and ai_msg.id:
                updates["messages"] = [RemoveMessage(id=ai_msg.id)]
            return Command(update=updates, goto="advisor_agent")

        if draft_errors:
            # 重试耗尽仍违反方案硬约束：强制拦阻确认弹窗，并替换为确定性拒批文案
            logger.error("方案草稿复核未通过且重试耗尽，拦截确认弹窗并替换提示文案", errors=draft_errors)
            reject_text = (
                "出于承保资格与预算合规保护，系统已拦截本次方案草稿提交，原因如下：\n"
                + "\n".join(f"- {err}" for err in draft_errors)
                + "\n\n如需继续配置，请告知我为您重新匹配符合承保条件与预算的合规产品组合。"
            )
            reject_updates: dict[str, Any] = {
                "pending_plan_draft": None,
                "guardrail_feedback": None,
                "verified_answer": reject_text,
            }
            new_reject_msg = AIMessage(id=str(uuid4()), content=reject_text)
            if ai_msg is not None and ai_msg.id:
                reject_updates["messages"] = [RemoveMessage(id=ai_msg.id), new_reject_msg]
            else:
                reject_updates["messages"] = [new_reject_msg]
            return Command(update=reject_updates, goto="deliver_answer")

        # 确定性兜底收口：剥离思维标签与虚构 [ref-xxx]，并在 RAG 漏标时自动补附真实出处
        final_text = sanitize_citations(raw_text, valid_ids, rag_used)
        final_updates: dict[str, Any] = {
            "guardrail_feedback": None,
            "verified_answer": final_text,
        }
        if final_text != raw_text:
            logger.warning("触发引文/标签确定性清洗收口", rag_used=rag_used, valid_ids=sorted(valid_ids))
            cleaned_msg = AIMessage(id=str(uuid4()), content=final_text)
            if ai_msg is not None and ai_msg.id:
                final_updates["messages"] = [RemoveMessage(id=ai_msg.id), cleaned_msg]
            else:
                final_updates["messages"] = [cleaned_msg]

        target: Literal["plan_confirm", "deliver_answer"] = (
            "plan_confirm" if draft is not None else "deliver_answer"
        )
        return Command(update=final_updates, goto=target)

    async def deliver_answer_node(state: AdvisorState) -> dict[str, Any]:
        """纯问答或追问收口节点（占位收口，避免重复生成消息）。"""
        return {}

    async def plan_confirm_node(
        state: AdvisorState,
    ) -> Command[Literal["save_plan", "advisor_agent"]]:
        draft = state.get("pending_plan_draft") or {}
        decision = interrupt(
            {
                "plan": draft,
                "confirm_info": f"请确认是否保存方案《{draft.get('plan_name', '智能保险配置方案')}》",
                "action": ["approve", "reject"],
            }
        )
        action = (decision or {}).get("action") if isinstance(decision, dict) else None
        if action == "approve":
            return Command(goto="save_plan")

        reject_reason = (
            (decision or {}).get("reject_reason")
            if isinstance(decision, dict)
            else None
        ) or "用户希望继续调整产品组合或预算"
        return Command(
            update={
                "pending_plan_draft": None,
                "guardrail_retry_count": 0,
                "guardrail_feedback": f"用户取消了上一版方案，反馈原因：{reject_reason}。请根据反馈重新优化建议。",
            },
            goto="advisor_agent",
        )

    async def save_plan_node(state: AdvisorState, config: RunnableConfig) -> dict[str, Any]:
        draft = state.get("pending_plan_draft") or {}
        configurable = (config or {}).get("configurable") or {}
        user_id = int(configurable.get("user_id") or 1)
        user_profile = state.get("user_profile") or {}

        saved_plan = await active_saver(
            user_id=user_id,
            draft=draft,
            user_profile=user_profile,
        )
        plan_items = saved_plan.get("items") or []
        if plan_items:
            _safe_stream_write({"plan_items": plan_items})

        msg_id = str(uuid4())
        plan_name = saved_plan.get("plan_name", "保险配置方案")
        updates: dict[str, Any] = {
            "pending_plan_draft": None,
            "guardrail_feedback": None,
            "messages": [
                AIMessage(
                    id=msg_id,
                    content=(
                        f"已为您保存保险配置方案《{plan_name}》（共 {len(plan_items)} 款产品）。"
                        "您可以点击下方产品卡片中的「去投保」按钮继续办理投保。"
                    ),
                )
            ],
        }
        if plan_items:
            updates["additional_info"] = {msg_id: {"plan_items": plan_items}}
        return updates

    builder = StateGraph(AdvisorState)
    builder.add_node("extract_profile", extract_profile_node)
    builder.add_node("advisor_agent", advisor_agent_node)
    builder.add_node("tools", tool_node)
    builder.add_node("collect_tool_artifacts", collect_tool_artifacts_node)
    builder.add_node("verify_response", verify_response_node)
    builder.add_node("deliver_answer", deliver_answer_node)
    builder.add_node("plan_confirm", plan_confirm_node)
    builder.add_node("save_plan", save_plan_node)

    builder.add_edge(START, "extract_profile")
    builder.add_edge("extract_profile", "advisor_agent")
    builder.add_conditional_edges("advisor_agent", route_after_agent)
    builder.add_edge("tools", "collect_tool_artifacts")
    builder.add_edge("collect_tool_artifacts", "advisor_agent")
    builder.add_edge("deliver_answer", END)
    builder.add_edge("save_plan", END)

    return builder.compile(checkpointer=checkpointer)


def init_insurance_agent():
    """初始化线上保险顾问混合子图"""
    agent = build_insurance_advisor_graph(checkpointer=True)
    logger.info("混合式保险顾问子图初始化成功~✅️")
    return agent
