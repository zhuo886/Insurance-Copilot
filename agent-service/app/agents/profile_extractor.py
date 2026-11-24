import re
from typing import Any

from app.recsys.recall import _DOMAIN_TOKENS

_AGE_PATTERNS = (
    re.compile(r"(?<!\d)(\d{1,3})\s*周?岁"),
    re.compile(r"年龄\s*[:：是为]?\s*(\d{1,3})(?!\d)"),
)

_OCCUPATION_PATTERN = re.compile(r"([1-6])\s*类(?:职业|工种)?")

_BUDGET_PATTERNS = (
    re.compile(
        r"(?:预算|每年|年缴|保费|控制在|不超过|大概花|大概|最多)[^\d，。；,\n]{0,8}?(\d+(?:\.\d+)?)\s*(万|千|k|K|元|块)?"
    ),
    re.compile(
        r"(\d+(?:\.\d+)?)\s*(万|千|k|K|元|块)\s*(?:预算|左右预算|的预算|每年|年缴|以内)"
    ),
)

_CATEGORY_RULES: list[tuple[tuple[str, ...], str]] = [
    (("医疗险", "百万医疗", "防癌医疗", "住院医疗", "门急诊", "特需医疗", "医疗"), "medical"),
    (("重疾险", "重大疾病", "防癌重疾", "重疾"), "critical_illness"),
    (("意外险", "学平险", "防跌倒", "猝死", "意外"), "accident"),
    (("定期寿险", "终身寿险", "增额终身寿", "增额寿", "定寿", "寿险"), "life"),
]

_ROLE_RULES: list[tuple[tuple[str, ...], str]] = [
    (("爸爸", "父亲", "妈妈", "母亲", "父母", "老人", "爷爷", "奶奶", "外公", "外婆"), "父母"),
    (("宝宝", "小孩", "少儿", "儿童", "孩子", "儿子", "女儿", "小学生", "中学生", "学生"), "少儿"),
    (("老公", "老婆", "丈夫", "妻子", "配偶", "夫妻"), "配偶"),
    (("家庭支柱", "顶梁柱", "房贷"), "家庭支柱"),
    (("本人", "自己", "我个人"), "本人"),
]

_PREFERENCE_RULES: list[tuple[tuple[str, ...], str]] = [
    (("保证续保20年", "保证续保"), "保证续保20年"),
    (("外购药", "自费药", "院外药", "靶向药", "特药"), "外购药"),
    (("0免赔", "零免赔"), "0免赔"),
    (("特需医疗", "特需", "国际部"), "特需医疗"),
    (("猝死",), "猝死保障"),
    (("高危职业", "高危"), "高危职业可保"),
    (("核保宽松", "免健告", "三高", "高血压", "糖尿病", "肺结节", "结节", "乙肝"), "核保宽松"),
    (("学平险",), "学平险"),
    (("高性价比", "性价比", "便宜"), "高性价比"),
    (("定期寿险", "定寿"), "定期寿险"),
    (("多次赔付", "多次赔"), "多次赔付"),
    (("等待期短",), "等待期短"),
    (("防跌倒", "老人意外"), "老人意外"),
    (("少儿特疾",), "少儿特疾"),
    (("商旅", "航空"), "商旅交通"),
    (("夫妻共保", "家庭保障", "全家"), "家庭保障"),
]


def extract_profile_by_regex(text: str) -> dict[str, Any]:
    """零延迟确定性用户画像槽位抽取，绝不阻断纯条款问答。"""
    cleaned = text.strip()
    if not cleaned:
        return {}

    extracted: dict[str, Any] = {}

    # 1. 年龄抽取
    for pattern in _AGE_PATTERNS:
        match = pattern.search(cleaned)
        if match:
            age_val = int(match.group(1))
            if 0 <= age_val <= 110:
                extracted["age"] = age_val
                break
    if "age" not in extracted and any(kw in cleaned for kw in ("半岁", "几个月", "新生儿", "满月")):
        extracted["age"] = 0

    # 2. 性别抽取
    if any(kw in cleaned for kw in ("男性", "男士", "男生", "男宝", "儿子", "爸爸", "父亲", "爷爷", "外公", "老公", "丈夫", "男")):
        if not any(neg in cleaned for neg in ("男女", "女生", "女性")):
            extracted["gender"] = "male"
    if (
        "gender" not in extracted
        and "男女" not in cleaned
        and any(
            kw in cleaned for kw in ("女性", "女士", "女生", "女宝", "女儿", "妈妈", "母亲", "奶奶", "外婆", "老婆", "妻子", "阿姨", "女")
        )
    ):
        extracted["gender"] = "female"

    # 3. 职业类别抽取
    occ_match = _OCCUPATION_PATTERN.search(cleaned)
    if occ_match:
        extracted["occupation_class"] = int(occ_match.group(1))
    elif any(kw in cleaned for kw in ("高空作业", "建筑工", "矿工", "大货车", "货车司机", "消防员")):
        extracted["occupation_class"] = 6
    elif any(kw in cleaned for kw in ("外卖", "骑手", "快递", "物流", "电工", "焊工")):
        extracted["occupation_class"] = 5

    # 4. 医保/社保状态
    if any(kw in cleaned for kw in ("没有医保", "无医保", "没交医保", "没有社保", "无社保", "没交社保", "没社保")):
        extracted["has_social_security"] = False
    elif any(kw in cleaned for kw in ("有医保", "有社保", "少儿医保", "交了医保", "交了社保", "有公费医疗")):
        extracted["has_social_security"] = True

    # 5. 年预算抽取（避免误匹配保额数字）
    for pattern in _BUDGET_PATTERNS:
        match = pattern.search(cleaned)
        if match:
            raw_num = float(match.group(1))
            unit = match.group(2) or ""
            span_start, span_end = match.span()
            surrounding = cleaned[max(0, span_start - 4) : min(len(cleaned), span_end + 4)]
            if "保额" in surrounding or "免赔" in surrounding:
                continue
            multiplier = 10000.0 if unit == "万" else (1000.0 if unit in ("千", "k", "K") else 1.0)
            budget = round(raw_num * multiplier, 2)
            if 100.0 <= budget <= 500000.0:
                extracted["annual_budget"] = budget
                break

    # 6. 目标险种抽取
    categories: list[str] = []
    for keywords, category in _CATEGORY_RULES:
        if any(kw in cleaned for kw in keywords) and category not in categories:
            categories.append(category)
    if categories:
        extracted["target_categories"] = categories

    # 7. 家庭角色
    for keywords, role in _ROLE_RULES:
        if any(kw in cleaned for kw in keywords):
            extracted["family_role"] = role
            break

    # 8. 保障偏好标签
    preferences: list[str] = []
    for keywords, tag in _PREFERENCE_RULES:
        if any(kw in cleaned for kw in keywords) and tag not in preferences:
            preferences.append(tag)
    for token in _DOMAIN_TOKENS:
        if token in cleaned and token not in preferences:
            preferences.append(token)
    if preferences:
        extracted["preferences"] = preferences

    return extracted
