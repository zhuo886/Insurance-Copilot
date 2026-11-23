import json
import time

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.agents.chitchat import init_chitchat_agent
from app.agents.claim_agent import init_claim_agent
from app.agents.insurance_advisor import init_insurance_agent
from app.agents.intent_router import intent_router
from app.agents.schemas import InsuranceAgentState, Intent
from app.core.logging import get_logger

logger = get_logger(__name__)


async def route_node(
    state: InsuranceAgentState,
) -> Command[Intent]:
    """主图意图识别与工作流分发节点。"""
    t0 = time.perf_counter()
    previous_workflow = state.get("active_workflow", "")
    previous_ai_message = find_previous_ai_message(state["messages"])
    context = f"上轮对话工作流:{previous_workflow}\n上轮对话AI回复:{previous_ai_message}"
    latest_text = str(state["messages"][-1].content or "")
    result = await intent_router.route(latest_text, context)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    if isinstance(result, str):
        logger.info("规则寒暄直出", intent="chitchat", latency_ms=latency_ms)
        return Command(
            update={
                "messages": [AIMessage(content=result)],
                "previous_workflow": previous_workflow,
                "active_workflow": "chitchat",
            },
            goto=END,
        )

    logger.info(
        "意图识别路由完成",
        intent=result.intent,
        previous_workflow=previous_workflow,
        latency_ms=latency_ms,
        reason=result.reason,
    )
    return Command(
        update={
            "previous_workflow": previous_workflow,
            "active_workflow": result.intent,
        },
        goto=result.intent,
    )


def find_previous_ai_message(messages: list[BaseMessage]) -> str:
    """查找本轮用户消息之前最近一条可展示的 AI 回复"""
    for message in reversed(messages[:-1]):
        if not isinstance(message, AIMessage):
            continue
        if message.tool_calls:
            continue
        if message.text:
            return message.text
    return ""


def human_handoff_node(state: InsuranceAgentState) -> dict:
    """整理当前用户画像与最新诉求摘要后转接人工顾问。"""
    profile = state.get("user_profile") or {}
    latest_msg = ""
    for msg in reversed(state.get("messages") or []):
        if msg.type == "human":
            latest_msg = str(msg.content or "")
            break
    profile_summary = json.dumps(profile, ensure_ascii=False) if profile else "暂无"
    return {
        "messages": [
            AIMessage(
                content=(
                    f"已为您转接人工保险顾问并同步会话上下文（已识别画像：{profile_summary}；"
                    f"最新诉求：{latest_msg[:80]}）。人工客服接入前，您也可以继续补充产品或理赔问题。"
                )
            )
        ]
    }


def fallback_node(state: InsuranceAgentState) -> dict:
    """结合上轮工作流上下文给出明确服务边界引导。"""
    prev = state.get("previous_workflow") or ""
    hint = (
        f"您上一轮正在进行「{prev}」咨询，我可以继续为您处理。"
        if prev
        else ""
    )
    return {
        "messages": [
            AIMessage(
                content=(
                    f"{hint}目前我暂时没有明确理解您的具体需求。您可以告诉我想了解的保险产品与年龄预算，"
                    "或直接询问某款产品条款、保费试算与理赔进度。"
                )
            )
        ]
    }


def init_insurance_orchestrator(
    checkpointer: AsyncPostgresSaver,
):
    """初始化保险顾问主 Graph"""
    builder = StateGraph(InsuranceAgentState)
    builder.add_node("route", route_node)
    builder.add_node("chitchat", init_chitchat_agent())
    builder.add_node("recommendation_plan", init_insurance_agent())
    builder.add_node("claim", init_claim_agent())
    builder.add_node("human_handoff", human_handoff_node)
    builder.add_node("fallback", fallback_node)

    builder.add_edge(START, "route")
    builder.add_edge("chitchat", END)
    builder.add_edge("recommendation_plan", END)
    builder.add_edge("claim", END)
    builder.add_edge("human_handoff", END)
    builder.add_edge("fallback", END)

    return builder.compile(checkpointer=checkpointer)
