from app.recsys.pricing import PremiumEstimator
from app.recsys.schemas import (
    FilterRejection,
    PremiumEstimate,
    ProductCandidate,
    RecUserProfile,
)


def filter_eligible_candidates(
    candidates: list[ProductCandidate],
    profile: RecUserProfile,
    estimator: PremiumEstimator,
    budget_tolerance: float = 1.15,
) -> tuple[list[tuple[ProductCandidate, PremiumEstimate]], list[FilterRejection]]:
    """Stage 2：基于结构化承保规则与单品预算上限做硬约束粗排过滤。"""
    eligible: list[tuple[ProductCandidate, PremiumEstimate]] = []
    rejections: list[FilterRejection] = []
    target_cats = set(profile.target_categories)

    for product in candidates:
        reasons: list[str] = []

        if target_cats and product.category not in target_cats:
            reasons.append(f"险种({product.category})不在目标范围({','.join(profile.target_categories)})")

        if profile.age < product.min_age or profile.age > product.max_age:
            reasons.append(f"被保人{profile.age}岁超出承保年龄({product.min_age}-{product.max_age}岁)")

        if profile.occupation_class > product.max_occupation_class:
            reasons.append(
                f"职业类别{profile.occupation_class}类超出承保上限(1-{product.max_occupation_class}类)"
            )

        estimate = estimator.estimate(product, profile)
        if profile.annual_budget > 0 and estimate.annual_premium > profile.annual_budget * budget_tolerance:
            reasons.append(
                f"预估年保费¥{estimate.annual_premium:.0f}超出预算容差上限(¥{profile.annual_budget * budget_tolerance:.0f})"
            )

        if reasons:
            rejections.append(
                FilterRejection(
                    product_id=product.id,
                    product_name=product.name,
                    category=product.category,
                    reasons=reasons,
                )
            )
        else:
            eligible.append((product, estimate))

    return eligible, rejections
