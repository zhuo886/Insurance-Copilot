from app.recsys.recall import build_profile_query_text
from app.recsys.schemas import (
    PremiumEstimate,
    ProductCandidate,
    RecUserProfile,
    ScoreBreakdown,
    ScoredCandidate,
)

DEFAULT_SCORE_WEIGHTS = {
    "semantic_rrf": 0.25,
    "profile_fit": 0.25,
    "coverage_tag_match": 0.30,
    "budget_efficiency": 0.15,
    "clause_quality": 0.05,
}

# 从自然语言里映射核心偏好标签，确保用户即使没显式填 preferences 也能提取诉求（1对1精简映射，避免多别名膨胀）
_QUERY_PREFERENCE_HINTS: dict[str, list[str]] = {
    "保证续保": ["保证续保20年"],
    "外购药": ["外购药"],
    "0免赔": ["0免赔"],
    "特需": ["特需医疗"],
    "猝死": ["猝死保障"],
    "高危": ["高危职业可保"],
    "少儿": ["少儿特疾"],
    "老人": ["老人意外", "中老年医疗"],
    "父母": ["老人意外", "中老年医疗"],
    "三高": ["核保宽松"],
    "结节": ["核保宽松"],
    "性价比": ["高性价比"],
    "房贷": ["定期寿险"],
    "家庭": ["家庭保障"],
    "等待期短": ["等待期短"],
    "多次赔": ["多次赔付"],
}


def extract_query_preference_tags(profile: RecUserProfile) -> list[str]:
    tags: list[str] = list(profile.preferences)
    raw_text = " ".join([profile.query_text, *profile.preferences])
    for keyword, mapped_tags in _QUERY_PREFERENCE_HINTS.items():
        if keyword in raw_text:
            for tag in mapped_tags:
                if tag not in tags:
                    tags.append(tag)
    return tags


def _compute_profile_fit(product: ProductCandidate, profile: RecUserProfile) -> float:
    age_span = max(product.max_age - product.min_age, 1)
    center = (product.min_age + product.max_age) / 2.0
    distance_ratio = abs(profile.age - center) / (age_span / 2.0)
    age_score = max(0.2, 1.0 - 0.5 * min(distance_ratio, 1.2))

    # 人群专项加权
    segment_bonus = 0.0
    if profile.age <= 17 and product.max_age <= 24:
        segment_bonus += 0.25
    elif profile.age >= 55 and product.max_age >= 68:
        segment_bonus += 0.25
    elif 18 <= profile.age <= 45 and product.min_age >= 18 and product.max_age <= 60:
        segment_bonus += 0.15

    if profile.occupation_class >= 5 and product.max_occupation_class >= 6:
        segment_bonus += 0.25

    if not profile.has_social_security and "无社保友好" in product.coverage_tags:
        segment_bonus += 0.20

    # 性别错配微惩罚（如女性专款推荐给男性）
    tag_blob = " ".join([product.name, *product.highlights, *product.coverage_tags])
    if profile.gender == "male" and "女性" in tag_blob:
        segment_bonus -= 0.25

    # 健康告知正常的人群（未提及三高/结节/免健告）优先全责任医疗/重疾，防癌专项略降权
    raw_pref_text = " ".join([profile.query_text, *profile.preferences])
    needs_loose_uw = any(k in raw_pref_text for k in ("三高", "结节", "乙肝", "高血压", "糖尿病", "免健告", "核保宽松"))
    if not needs_loose_uw and profile.age < 60 and ("防癌医疗" in tag_blob or "防癌重疾" in tag_blob):
        segment_bonus -= 0.18
    elif needs_loose_uw and product.health_disclosure_level == "loose":
        segment_bonus += 0.20

    # 家庭支柱或中低预算推荐寿险时，定期寿险更贴合，高额分红储蓄寿险降权
    if product.category == "life" and profile.annual_budget > 0 and profile.annual_budget < 8000:
        if "定期寿险" in product.coverage_tags:
            segment_bonus += 0.20
        elif "分红型" in product.coverage_tags or "财富传承" in product.coverage_tags:
            segment_bonus -= 0.20

    return min(1.0, max(0.0, age_score * 0.75 + segment_bonus))


def _compute_tag_match(product: ProductCandidate, query_tags: list[str]) -> float:
    if not query_tags:
        return 0.55

    tag_set = " ".join([product.name, *product.highlights, *product.coverage_tags]).lower()
    desc_text = product.description.lower()
    hits = 0.0
    for tag in query_tags:
        t = tag.lower()
        if t in tag_set:
            hits += 1.0
        elif t in desc_text:
            hits += 0.6
    # 按品类单品通常最多承接 1~2 个用户跨险种偏好做分母归一化，避免跨险种多标签稀释单品得分
    denom = max(1.0, min(float(len(query_tags)), 2.0))
    return min(1.0, hits / denom)


def _compute_budget_efficiency(
    product: ProductCandidate,
    estimate: PremiumEstimate,
    profile: RecUserProfile,
    cat_min_prem: float,
    cat_max_prem: float,
) -> float:
    # 品类内价格分位（越便宜越具性价比，但不过度惩罚功能全的产品）
    if cat_max_prem > cat_min_prem:
        relative_price_score = 1.0 - 0.65 * (
            (estimate.annual_premium - cat_min_prem) / (cat_max_prem - cat_min_prem)
        )
    else:
        relative_price_score = 0.8

    if profile.annual_budget <= 0:
        return min(1.0, max(0.0, relative_price_score))

    num_cats = max(len(profile.target_categories) or 3, 1)
    # 估算每个险种在组合中的合理预算份额
    category_weight = {
        "critical_illness": 0.45,
        "medical": 0.25,
        "life": 0.25,
        "accident": 0.10,
    }.get(product.category, 1.0 / num_cats)
    target_share = profile.annual_budget * (1.0 if num_cats == 1 else max(0.12, category_weight))

    ratio = estimate.annual_premium / max(target_share, 1.0)
    if ratio <= 1.0:
        fit_score = 0.75 + 0.25 * (1.0 - abs(ratio - 0.75))
    elif ratio <= 1.35:
        fit_score = max(0.25, 0.75 - (ratio - 1.0) * 1.4)
    else:
        fit_score = 0.10

    return min(1.0, max(0.0, 0.4 * relative_price_score + 0.6 * fit_score))


def _compute_clause_quality(product: ProductCandidate) -> float:
    score = 0.25
    if product.has_clause:
        score += 0.12
    if product.category == "medical" and product.guarantee_renewal_years:
        capped_years = min(product.guarantee_renewal_years, 20)
        score += min(0.40, capped_years / 20.0 * 0.40)
    elif product.category == "critical_illness":
        score += 0.35 if product.waiting_period_days <= 90 else 0.15
    elif product.category in ("accident", "life"):
        score += 0.30 if product.waiting_period_days <= 90 else 0.15

    if product.insurer_tier == "top":
        score += 0.20
    elif product.insurer_tier == "joint":
        score += 0.12
    return min(1.0, score)


def score_candidates(
    eligible: list[tuple[ProductCandidate, PremiumEstimate]],
    profile: RecUserProfile,
    rrf_scores: dict[int, float],
    recall_ranks: dict[int, dict[str, int]] | None = None,
    weights: dict[str, float] | None = None,
) -> list[ScoredCandidate]:
    """Stage 3：5 维可解释多因子精排打分。"""
    if not eligible:
        return []

    w = weights or DEFAULT_SCORE_WEIGHTS
    ranks_map = recall_ranks or {}
    query_tags = extract_query_preference_tags(profile)

    max_rrf = max((rrf_scores.get(p.id, 0.0) for p, _ in eligible), default=1.0) or 1.0

    cat_prices: dict[str, list[float]] = {}
    for p, est in eligible:
        cat_prices.setdefault(p.category, []).append(est.annual_premium)
    cat_bounds = {cat: (min(vals), max(vals)) for cat, vals in cat_prices.items()}

    scored_list: list[ScoredCandidate] = []
    for product, estimate in eligible:
        raw_rrf = rrf_scores.get(product.id, 0.0)
        semantic_score = round(min(1.0, raw_rrf / max_rrf) if max_rrf > 0 else 0.5, 4)
        profile_fit = round(_compute_profile_fit(product, profile), 4)
        tag_match = round(_compute_tag_match(product, query_tags), 4)
        lo_p, hi_p = cat_bounds[product.category]
        budget_eff = round(_compute_budget_efficiency(product, estimate, profile, lo_p, hi_p), 4)
        clause_qual = round(_compute_clause_quality(product), 4)

        total = round(
            w["semantic_rrf"] * semantic_score
            + w["profile_fit"] * profile_fit
            + w["coverage_tag_match"] * tag_match
            + w["budget_efficiency"] * budget_eff
            + w["clause_quality"] * clause_qual,
            4,
        )

        scored_list.append(
            ScoredCandidate(
                product=product,
                estimate=estimate,
                rrf_score=round(raw_rrf, 6),
                recall_ranks=ranks_map.get(product.id, {}),
                scores=ScoreBreakdown(
                    semantic_rrf=semantic_score,
                    profile_fit=profile_fit,
                    coverage_tag_match=tag_match,
                    budget_efficiency=budget_eff,
                    clause_quality=clause_qual,
                    total_score=total,
                ),
            )
        )

    scored_list.sort(key=lambda item: (-item.scores.total_score, item.estimate.annual_premium, item.product.id))
    return scored_list
