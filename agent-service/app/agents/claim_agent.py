import json
from typing import Any, Literal, NotRequired
from uuid import uuid4

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from app.agents.guardrails import sanitize_citations, strip_thinking_tags, verify_citations
from app.agents.schemas import InsuranceAgentState
from app.agents.tools import get_claim_guide, query_product_clause, query_user_policies_and_claims
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

CLAIM_SYSTEM_PROMPT = """
你是“安心保”的理赔服务专员，负责保单查询、理赔进度查询与理赔条件/材料解答。
工作规范：
1. 涉及用户保单、理赔进度、报案指引时，必须首先调用 `query_user_policies_and_claims` 查询当前用户的真实保单与理赔记录；确认目标保单 UUID 后，可进一步调用 `get_claim_guide` 获取标准理赔步骤、材料清单及理赔申请表下载地址（`/api/v1/claim-guides/application-form`）。
2. 涉及具体险种的理赔条件、所需材料清单、报案时效或免赔额时，必须调用 `query_product_clause` 检索对应产品的真实条款，并在事实陈述后紧跟 `[ref-xxx]` 引用标记。
3. 严禁编造保单号、理赔状态、引用编号或条款外规定；如果检索工具未返回对应产品的条款原文，请如实说明暂未查到完整条款原文，不可凭记忆编造；如果用户尚未出险报案，清晰列出报案步骤与对应材料清单。
""".strip()


class ClaimAgentState(InsuranceAgentState):
    """理赔服务子图私有状态（复用确定性引文护栏）"""

    retrieved_source_ids: NotRequired[list[str]]
    guardrail_retry_count: NotRequired[int]
    guardrail_feedback: NotRequired[str | None]
    last_rag_used: NotRequired[bool]
    verified_answer: NotRequired[str | None]


def _extract_latest_ai_message(messages: list[Any]) -> AIMessage | None:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            return msg
    return None


def build_claim_agent_graph(
    *,
    model: Any = None,
    tools: list[Any] | None = None,
    checkpointer: Any = True,
):
    """构建带确定性引文护栏的理赔服务子图：reset_turn -> claim_agent <-> tools -> verify_claim_response。"""
    active_tools = list(
        tools
        if tools is not None
        else [query_user_policies_and_claims, get_claim_guide, query_product_clause]
    )

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

    async def reset_turn_node(_state: ClaimAgentState) -> dict[str, Any]:
        return {
            "retrieved_source_ids": [],
            "guardrail_retry_count": 0,
            "guardrail_feedback": None,
            "last_rag_used": False,
            "verified_answer": None,
        }

    async def claim_agent_node(state: ClaimAgentState, config: RunnableConfig) -> dict[str, Any]:
        sys_parts = [CLAIM_SYSTEM_PROMPT]
        feedback = state.get("guardrail_feedback")
        if feedback:
            sys_parts.append(f"【系统纠偏指令（请务必修正以下问题后再输出）】\n{feedback}")

        call_messages: list[Any] = [
            SystemMessage(content="\n\n".join(sys_parts)),
            *(state.get("messages") or []),
        ]
        if feedback:
            call_messages.append(
                HumanMessage(
                    content=(
                        "【系统合规护栏强制纠偏（优先级高于用户此前的格式指令）】\n"
                        f"{feedback}\n"
                        "请立即修正回答中的引文问题，仅依据真实检索到的条款原文和真实保单记录作答。"
                    )
                )
            )
        response = await bound_model.ainvoke(call_messages, config=config)
        return {"messages": [response]}

    def route_after_claim_agent(state: ClaimAgentState) -> Literal["tools", "verify_claim_response"]:
        messages = state.get("messages") or []
        if messages and isinstance(messages[-1], AIMessage) and getattr(messages[-1], "tool_calls", None):
            return "tools"
        return "verify_claim_response"

    async def collect_tool_artifacts_node(state: ClaimAgentState) -> dict[str, Any]:
        messages = state.get("messages") or []
        source_ids = list(state.get("retrieved_source_ids") or [])
        rag_used = bool(state.get("last_rag_used"))

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

            if (tmsg.name == "query_product_clause" and parsed.get("status") == "ok") or parsed.get("source_ids"):
                rag_used = True
                for sid in parsed.get("source_ids") or []:
                    sid_str = str(sid)
                    if sid_str not in source_ids:
                        source_ids.append(sid_str)

        if rag_used:
            try:
                get_stream_writer()({"type": "guardrail_buffer"})
            except Exception:
                pass

        return {
            "retrieved_source_ids": source_ids,
            "last_rag_used": rag_used,
        }

    async def verify_claim_response_node(
        state: ClaimAgentState,
    ) -> Command[Literal["claim_agent", "__end__"]]:
        ai_msg = _extract_latest_ai_message(state.get("messages") or [])
        raw_text = (ai_msg.text or str(ai_msg.content or "")) if ai_msg is not None else ""
        stripped_text = strip_thinking_tags(raw_text)
        valid_ids = set(state.get("retrieved_source_ids") or [])
        rag_used = bool(state.get("last_rag_used"))

        citation_errors = verify_citations(stripped_text, valid_ids, rag_used)
        retry_count = int(state.get("guardrail_retry_count") or 0)

        if citation_errors and retry_count < 1:
            logger.warning("理赔子图确定性护栏触发纠偏", errors=citation_errors, retry_count=retry_count)
            updates: dict[str, Any] = {
                "guardrail_feedback": "；".join(citation_errors),
                "guardrail_retry_count": retry_count + 1,
            }
            if ai_msg is not None and ai_msg.id:
                updates["messages"] = [RemoveMessage(id=ai_msg.id)]
            return Command(update=updates, goto="claim_agent")

        final_text = sanitize_citations(raw_text, valid_ids, rag_used)
        final_updates: dict[str, Any] = {
            "guardrail_feedback": None,
            "verified_answer": final_text,
        }
        if final_text != raw_text:
            logger.warning("理赔子图触发引文确定性清洗收口", rag_used=rag_used, valid_ids=sorted(valid_ids))
            cleaned_msg = AIMessage(id=str(uuid4()), content=final_text)
            if ai_msg is not None and ai_msg.id:
                final_updates["messages"] = [RemoveMessage(id=ai_msg.id), cleaned_msg]
            else:
                final_updates["messages"] = [cleaned_msg]

        return Command(update=final_updates, goto=END)

    builder = StateGraph(ClaimAgentState)
    builder.add_node("reset_turn", reset_turn_node)
    builder.add_node("claim_agent", claim_agent_node)
    builder.add_node("tools", tool_node)
    builder.add_node("collect_tool_artifacts", collect_tool_artifacts_node)
    builder.add_node("verify_claim_response", verify_claim_response_node)

    builder.add_edge(START, "reset_turn")
    builder.add_edge("reset_turn", "claim_agent")
    builder.add_conditional_edges("claim_agent", route_after_claim_agent)
    builder.add_edge("tools", "collect_tool_artifacts")
    builder.add_edge("collect_tool_artifacts", "claim_agent")

    return builder.compile(checkpointer=checkpointer)


def init_claim_agent():
    """初始化带确定性护栏的理赔服务子图"""
    agent = build_claim_agent_graph(checkpointer=True)
    logger.info("理赔服务子图（含确定性引文护栏）初始化成功~✅️")
    return agent
