from typing import Any

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# 口语化用户提问 -> 保险条款规范术语映射词典
TERM_SYNONYM_MAP: dict[str, list[str]] = {
    "买完多久能赔": ["等待期", "观察期"],
    "多久生效": ["等待期", "观察期"],
    "多久能赔": ["等待期", "观察期"],
    "观察期": ["等待期"],
    "自费药": ["外购药品", "外购医疗器械", "医保目录外"],
    "院外药": ["外购药品", "外购医疗器械", "清单"],
    "外购药": ["外购药品", "外购医疗器械"],
    "特药": ["外购药品", "特定药品", "外购医疗器械"],
    "报销比例": ["给付比例", "费用补偿", "免赔额"],
    "住院报多少": ["住院医疗费用", "给付比例", "免赔额"],
    "报多少": ["给付比例", "保险金额", "费用补偿"],
    "起付线": ["免赔额"],
    "没有医保": ["未获得基本医疗保险", "公费医疗", "给付比例"],
    "没社保": ["未获得基本医疗保险", "公费医疗", "给付比例"],
    "哪些不赔": ["责任免除", "其他免责条款"],
    "不保什么": ["责任免除", "其他免责条款"],
    "不保不赔": ["责任免除", "其他免责条款"],
    "什么情况不赔": ["责任免除", "其他免责条款"],
    "除外": ["责任免除"],
    "怎么理赔": ["保险金申请", "保险事故通知", "保险金给付"],
    "理赔材料": ["保险金申请", "申请材料", "证明和资料"],
    "需要什么资料": ["保险金申请", "证明和资料"],
    "报案": ["保险事故通知"],
    "退保": ["犹豫期", "合同解除", "现金价值"],
    "后悔": ["犹豫期", "合同解除"],
    "全额退": ["犹豫期"],
    "忘交保费": ["宽限期", "效力中止"],
    "晚交": ["宽限期", "效力中止"],
    "断交": ["宽限期", "效力中止", "效力恢复"],
    "换工作": ["职业变更", "职业类别"],
    "转行": ["职业变更", "职业类别"],
    "续保": ["保证续保", "保险期间", "费率调整"],
    "明年还能买": ["保证续保", "不保证续保", "保险期间"],
    "涨价": ["费率调整", "费率可调"],
    "取钱": ["减保", "减少基本保险金额", "现金价值"],
    "退部分": ["减保", "减少基本保险金额"],
    "门急诊": ["住院前后门急诊", "特殊门诊", "门诊手术"],
    "住院前后": ["住院前后门急诊"],
    "摔伤": ["意外伤残", "残疾保险责任", "意外身故"],
    "残疾": ["意外伤残", "残疾保险金"],
    "转诊": ["异地转诊", "公共交通费用"],
}

HYDE_SYSTEM_PROMPT = """
你是资深保险条款专家。请针对用户关于保险条款的问题，用正式保险合同条款的写作风格生成一段约 120 字的假设性条款摘录（只输出纯条款正文，不要解释或废话）。
""".strip()

LLM_REWRITE_SYSTEM_PROMPT = """
你是保险检索 Query 规范化助手。请将用户的口语提问补充并改写为便于检索保险合同条款的简洁关键词序列，保留核心实体和险种词，只输出改写后的检索文本，不要解释。
""".strip()


class InsuranceQueryRewriter:
    """保险领域 Query 术语改写器（支持零延迟词典改写、可选 LLM 改写及 HyDE 对照）"""

    def __init__(self, synonym_map: dict[str, list[str]] | None = None) -> None:
        self.synonym_map = synonym_map or TERM_SYNONYM_MAP
        self._llm: Any = None

    def _get_llm(self) -> Any:
        if self._llm is None:
            from langchain.chat_models import init_chat_model

            self._llm = init_chat_model(
                model=settings.llm.chat_model,
                model_provider="openai",
                api_key=settings.llm.api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                extra_body={"thinking": {"type": "disabled"}},
            )
        return self._llm

    def rewrite_by_dict(self, query: str) -> str:
        """零延迟确定性词典改写：在原 query 后追加未出现的规范保险术语。"""
        cleaned = query.strip()
        if not cleaned:
            return ""

        expansions: list[str] = []
        for trigger, canonical_terms in self.synonym_map.items():
            if trigger in cleaned:
                for term in canonical_terms:
                    if term not in cleaned and term not in expansions:
                        expansions.append(term)

        if not expansions:
            return cleaned
        return f"{cleaned} {' '.join(expansions)}"

    async def rewrite(self, query: str, *, use_llm: bool = False) -> str:
        dict_rewritten = self.rewrite_by_dict(query)
        if not use_llm or not dict_rewritten:
            return dict_rewritten

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            llm = self._get_llm()
            response = await llm.ainvoke(
                [
                    SystemMessage(content=LLM_REWRITE_SYSTEM_PROMPT),
                    HumanMessage(content=f"原始提问：{query}\n术语补充：{dict_rewritten}"),
                ]
            )
            text = (getattr(response, "content", "") or "").strip()
            return text or dict_rewritten
        except Exception as exc:
            logger.warning("LLM Query改写失败，回退词典改写", error=str(exc))
            return dict_rewritten

    async def generate_hyde_passage(self, query: str) -> str:
        """生成假设性保险条款片段（用于 eval_rag 中 HyDE 竞品方案对比）。"""
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            llm = self._get_llm()
            response = await llm.ainvoke(
                [
                    SystemMessage(content=HYDE_SYSTEM_PROMPT),
                    HumanMessage(content=query),
                ]
            )
            text = (getattr(response, "content", "") or "").strip()
            return text or query
        except Exception as exc:
            logger.warning("HyDE 生成失败，回退原 Query", error=str(exc))
            return query
