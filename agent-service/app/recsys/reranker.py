from itertools import product as cartesian_product

from app.recsys.schemas import PortfolioResult, RecUserProfile, ScoredCandidate

_DEFAULT_CATEGORY_PRIORITY = ["medical", "accident", "critical_illness", "life"]


def _product_similarity(a: ScoredCandidate, b: ScoredCandidate) -> float:
    sim = 0.0
    if a.product.insurer == b.product.insurer:
        sim += 0.50
    if a.product.category == b.product.category:
        sim += 0.40
    if a.product.clause_name == b.product.clause_name:
        sim += 0.30
    return min(1.0, sim)


def mmr_diversify(
    scored: list[ScoredCandidate],
    lambda_param: float = 0.78,
    top_k: int = 12,
) -> list[ScoredCandidate]:
    """Stage 4a：MMR 多样性重排，抑制同险企、同条款或同品类扎堆。"""
    if len(scored) <= 1:
        return scored[:top_k]

    remaining = list(scored)
    selected: list[ScoredCandidate] = [remaining.pop(0)]

    while remaining and len(selected) < top_k:
        best_idx = 0
        best_mmr = float("-inf")
        for idx, cand in enumerate(remaining):
            max_sim = max(_product_similarity(cand, chosen) for chosen in selected)
            mmr_score = lambda_param * cand.scores.total_score - (1.0 - lambda_param) * max_sim
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx
        selected.append(remaining.pop(best_idx))

    return selected


def optimize_budget_portfolio(
    diversified: list[ScoredCandidate],
    profile: RecUserProfile,
    max_items: int = 4,
) -> PortfolioResult:
    """Stage 4b：预算约束下的多险种分组组合优化（最大化风险覆盖度与组合总分）。"""
    if not diversified:
        return PortfolioResult(
            missing_categories=list(profile.target_categories),
            within_budget=True,
        )

    target_cats = (
        list(dict.fromkeys(profile.target_categories))
        if profile.target_categories
        else [c for c in _DEFAULT_CATEGORY_PRIORITY if any(x.product.category == c for x in diversified)]
    )
    target_cats = target_cats[:max_items]

    # 若仅单一目标险种，取预算内最高分作为主选，其余作为备选
    if len(target_cats) == 1:
        cat_items = [x for x in diversified if x.product.category == target_cats[0]]
        budget = profile.annual_budget
        within = [x for x in cat_items if budget <= 0 or x.estimate.annual_premium <= budget]
        chosen = within[:1] if within else cat_items[:1]
        total_prem = round(sum(x.estimate.annual_premium for x in chosen), 2)
        within_budget = budget <= 0 or total_prem <= budget
        alternatives = [x for x in cat_items if x not in chosen][:3]
        return PortfolioResult(
            selected=chosen,
            alternatives=alternatives,
            ranked_candidates=diversified,
            total_estimated_premium=total_prem,
            covered_categories=[x.product.category for x in chosen],
            missing_categories=[] if chosen else target_cats,
            within_budget=within_budget,
            budget_gap=round(max(0.0, total_prem - budget), 2) if budget > 0 else 0.0,
        )

    # 多险种场景：每个品类最多取 Top-6 候选 + None（不选该品类），执行精确分组搜索
    groups: list[list[ScoredCandidate | None]] = []
    for cat in target_cats:
        cat_candidates = [x for x in diversified if x.product.category == cat][:6]
        groups.append([None, *cat_candidates])

    budget = profile.annual_budget
    best_combo: tuple[ScoredCandidate, ...] | None = None
    best_objective: tuple[int, float, float] = (-1, -1.0, float("-inf"))

    for raw_combo in cartesian_product(*groups):
        chosen = tuple(item for item in raw_combo if item is not None)
        if not chosen or len(chosen) > max_items:
            continue
        total_prem = sum(item.estimate.annual_premium for item in chosen)
        if budget > 0 and total_prem > budget:
            continue

        covered_count = len(chosen)
        total_score = sum(item.scores.total_score for item in chosen)
        # 相同覆盖数与分数下，优先预算利用率合理（贴近 85% 预算但绝不超支）
        budget_util_penalty = -abs(total_prem - 0.85 * budget) if budget > 0 else -total_prem
        objective = (covered_count, round(total_score, 6), round(budget_util_penalty, 2))
        if objective > best_objective:
            best_objective = objective
            best_combo = chosen

    if best_combo is not None:
        selected = list(best_combo)
        within_budget = True
        total_prem = round(sum(x.estimate.annual_premium for x in selected), 2)
        budget_gap = 0.0
    else:
        # 预算极端紧张连最便宜单品都超支时，优雅降级返回保障优先级最高的最轻单品并标出预算缺口
        priority_rank = {cat: idx for idx, cat in enumerate(_DEFAULT_CATEGORY_PRIORITY)}
        fallback_item = min(
            diversified,
            key=lambda x: (priority_rank.get(x.product.category, 99), x.estimate.annual_premium),
        )
        selected = [fallback_item]
        total_prem = round(fallback_item.estimate.annual_premium, 2)
        within_budget = budget <= 0 or total_prem <= budget
        budget_gap = round(max(0.0, total_prem - budget), 2) if budget > 0 else 0.0

    # 按保障核心优先级排序最终组合（医疗 -> 重疾 -> 意外 -> 寿险）
    display_order = {"medical": 0, "critical_illness": 1, "accident": 2, "life": 3}
    selected.sort(key=lambda x: display_order.get(x.product.category, 99))

    selected_ids = {x.product.id for x in selected}
    covered = [x.product.category for x in selected]
    missing = [cat for cat in target_cats if cat not in covered]
    alternatives = [x for x in diversified if x.product.id not in selected_ids][:4]

    return PortfolioResult(
        selected=selected,
        alternatives=alternatives,
        ranked_candidates=diversified,
        total_estimated_premium=total_prem,
        covered_categories=covered,
        missing_categories=missing,
        within_budget=within_budget,
        budget_gap=budget_gap,
    )
