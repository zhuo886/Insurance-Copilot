from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from app.agents.tools import query_product_clause, query_user_policies_and_claims
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

CLAIM_SYSTEM_PROMPT = """
你是“安心保”的理赔服务专员，负责保单查询、理赔进度查询与理赔条件/材料解答。
工作规范：
1. 涉及用户保单、理赔进度、报案指引时，必须首先调用 `query_user_policies_and_claims` 查询当前用户的真实保单与理赔记录。
2. 涉及具体险种的理赔条件、所需材料清单、报案时效或免赔额时，调用 `query_product_clause` 检索对应产品的真实条款，并在事实陈述后紧跟 `[ref-xxx]` 引用标记。
3. 严禁编造保单号、理赔状态或条款外规定；如果用户尚未出险报案，清晰列出报案步骤与对应材料清单。
""".strip()


def init_claim_agent():
    """初始化理赔服务子 Agent"""
    model = init_chat_model(
        model=settings.llm.chat_model,
        model_provider="openai",
        api_key=settings.llm.api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        extra_body={"thinking": {"type": "disabled"}},
    )
    agent = create_agent(
        model=model,
        tools=[query_user_policies_and_claims, query_product_clause],
        system_prompt=CLAIM_SYSTEM_PROMPT,
        checkpointer=True,
    )
    logger.info("理赔服务子 Agent 初始化成功~✅️")
    return agent
