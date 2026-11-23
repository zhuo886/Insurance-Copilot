import re
from typing import Any

from app.recsys.pricing import PremiumEstimator
from app.recsys.schemas import ProductCandidate, RecUserProfile

_REF_BLOCK_PATTERN = re.compile(r"\[(ref-\d{3,}(?:\s*[,，]\s*ref-\d{3,})*)\]")
_REF_ID_PATTERN = re.compile(r"ref-\d{3,}")
_THINKING_BLOCK_PATTERN = re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE)
_THINKING_UNCLOSED_PREFIX_PATTERN = re.compile(
    r"^<thinking>.*?(?:\n\s*\n|\Z)", re.DOTALL | re.IGNORECASE
)
_THINKING_TAG_PATTERN = re.compile(r"</?thinking>", re.IGNORECASE)
_NO_CLAUSE_PHRASES = ("未查询到", "没有查询到", "暂无完整", "未能查到", "未检索到", "没有查到")


def strip_thinking_tags(text: str) -> str:
    """移除模型偶发泄露的 <thinking> 思维链标签及未闭合前置思维块。"""
    raw = text or ""
    cleaned = _THINKING_BLOCK_PATTERN.sub("", raw)
    if "<thinking" in cleaned.lower():
        cleaned = _THINKING_UNCLOSED_PREFIX_PATTERN.sub("", cleaned.lstrip())
    cleaned = _THINKING_TAG_PATTERN.sub("", cleaned).strip()
    if not cleaned and raw.strip():
        return _THINKING_TAG_PATTERN.sub("", raw).strip()
    return cleaned


def extract_cited_refs(text: str) -> list[str]:
    """提取回复正文中出现的全部 [ref-xxx] 引用编号。"""
    refs: list[str] = []
    for block in _REF_BLOCK_PATTERN.findall(text or ""):
        for ref_id in _REF_ID_PATTERN.findall(block):
            if ref_id not in refs:
                refs.append(ref_id)
    return refs


def sanitize_citations(
    text: str,
    valid_source_ids: set[str],
    rag_used: bool,
) -> str:
    """确定性清洗思维标签与虚假引用编号，并在 RAG 实质回复缺引文时补全真实依据脚标。"""
    cleaned = strip_thinking_tags(text)

    def _filter_block(match: re.Match[str]) -> str:
        kept: list[str] = []
        for ref_id in _REF_ID_PATTERN.findall(match.group(1)):
            if ref_id in valid_source_ids and ref_id not in kept:
                kept.append(ref_id)
        return f"[{', '.join(kept)}]" if kept else ""

    cleaned = _REF_BLOCK_PATTERN.sub(_filter_block, cleaned)
    cleaned = _REF_ID_PATTERN.sub(
        lambda m: m.group(0) if m.group(0) in valid_source_ids else "",
        cleaned,
    )
    cleaned = re.sub(r"\[\s*[,，\s]*\]", "", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r" +([，。！？；,.!?;])", r"\1", cleaned).strip()

    if (
        rag_used
        and valid_source_ids
        and len(cleaned) >= 15
        and not extract_cited_refs(cleaned)
        and not any(phrase in cleaned for phrase in _NO_CLAUSE_PHRASES)
    ):
        refs_str = " ".join(f"[{sid}]" for sid in sorted(valid_source_ids)[:3])
        cleaned = f"{cleaned}\n\n（条款参考依据：{refs_str}）"

    return cleaned


def verify_citations(
    text: str,
    valid_source_ids: set[str],
    rag_used: bool,
) -> list[str]:
    """确定性引文校验：拦阻虚假引用编号，并提醒 RAG 轮次遗漏引用标记。"""
    errors: list[str] = []
    cited = extract_cited_refs(text)

    invalid = [ref_id for ref_id in cited if ref_id not in valid_source_ids]
    if invalid:
        errors.append(
            f"回答中引用了未检索到的虚假条款编号 {invalid}；合法编号仅限于 {sorted(valid_source_ids)}，"
            "请修正引用编号或删除无依据陈述。"
        )

    stripped = (text or "").strip()
    if (
        rag_used
        and valid_source_ids
        and len(stripped) >= 15
        and not cited
        and not any(phrase in stripped for phrase in _NO_CLAUSE_PHRASES)
    ):
        errors.append(
            f"本轮已检索到真实条款依据 {sorted(valid_source_ids)}，请在每个条款事实陈述后标注对应 [ref-xxx] 引用标记。"
        )
    return errors


def verify_plan_draft(
    plan_draft: dict[str, Any] | None,
    user_profile: dict[str, Any],
    catalog_map: dict[int, ProductCandidate],
    estimator: PremiumEstimator | None = None,
    budget_tolerance: float = 1.10,
) -> list[str]:
    """确定性方案硬约束复核：校验产品合法性、投保年龄/职业资格及总预算上限。"""
    if not plan_draft:
        return []

    items = plan_draft.get("items") or []
    if not items:
        return ["方案草稿未包含任何有效保险产品，请重新挑选产品后再提交方案。"]

    est = estimator or PremiumEstimator()
    age = user_profile.get("age")
    occ = user_profile.get("occupation_class")
    budget = user_profile.get("annual_budget")
    rec_profile = RecUserProfile(
        age=int(age) if age is not None else 30,
        gender=user_profile.get("gender") or "male",
        occupation_class=int(occ) if occ is not None else 1,
        has_social_security=(
            bool(user_profile["has_social_security"])
            if user_profile.get("has_social_security") is not None
            else True
        ),
        annual_budget=float(budget) if budget is not None else 0.0,
    )

    errors: list[str] = []
    total_estimated = 0.0
    for item in items:
        pid = int(item["product_id"])
        product = catalog_map.get(pid)
        if product is None:
            errors.append(f"方案草稿包含不存在的产品 ID={pid}。")
            continue

        if age is not None and (int(age) < product.min_age or int(age) > product.max_age):
            errors.append(
                f"产品《{product.name}》(ID={pid}) 承保年龄为 {product.min_age}-{product.max_age} 周岁，"
                f"不适用于当前被保人 {age} 周岁。"
            )
        if occ is not None and int(occ) > product.max_occupation_class:
            errors.append(
                f"产品《{product.name}》(ID={pid}) 最高承保 1-{product.max_occupation_class} 类职业，"
                f"不适用于当前 {occ} 类职业。"
            )

        item_budget = item.get("annual_premium_budget")
        if item_budget is not None:
            total_estimated += float(item_budget)
        else:
            total_estimated += est.estimate(product, rec_profile).annual_premium

    if budget is not None and float(budget) > 0 and total_estimated > float(budget) * budget_tolerance:
        errors.append(
            f"方案草稿预估年保费合计约 ¥{total_estimated:.0f}，超出用户年预算 ¥{float(budget):.0f} 容差上限，"
            "请调整组合以控制总保费。"
        )
    return errors
