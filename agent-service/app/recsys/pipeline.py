import time
from typing import Sequence

from app.core.logging import get_logger
from app.recsys.filter import filter_eligible_candidates
from app.recsys.pricing import PremiumEstimator
from app.recsys.recall import (
    BM25KeywordRecall,
    OfflineSemanticRecall,
    RecallChannel,
    StructuredRuleRecall,
    rrf_fuse,
)
from app.recsys.reranker import mmr_diversify, optimize_budget_portfolio
from app.recsys.schemas import (
    FunnelStageMetric,
    FunnelTrace,
    PortfolioResult,
    ProductCandidate,
    RecUserProfile,
)
from app.recsys.scorer import score_candidates

logger = get_logger(__name__)


class RecSysPipeline:
    """四阶段工程化搜推漏斗：多路召回(RRF) -> 硬约束粗排 -> 5维可解释精排 -> MMR与预算组合优化"""

    def __init__(
        self,
        estimator: PremiumEstimator | None = None,
        recall_channels: Sequence[RecallChannel] | None = None,
        recall_top_k: int = 15,
    ) -> None:
        self.estimator = estimator or PremiumEstimator()
        self.recall_channels = list(
            recall_channels
            or (
                OfflineSemanticRecall(),
                BM25KeywordRecall(),
                StructuredRuleRecall(),
            )
        )
        self.recall_top_k = recall_top_k

    def recommend(
        self,
        profile: RecUserProfile,
        catalog: list[ProductCandidate],
        *,
        max_items: int = 4,
    ) -> PortfolioResult:
        total_start = time.perf_counter()
        stages: list[FunnelStageMetric] = []
        active_catalog = [item for item in catalog if item.status == "active"]
        catalog_map = {item.id: item for item in active_catalog}
        category_map = {item.id: item.category for item in active_catalog}

        # ---------------- Stage 1: 多路召回 + RRF 融合 ----------------
        t1 = time.perf_counter()
        ranked_lists: dict[str, list[int]] = {}
        channel_limit = max(self.recall_top_k + 5, 20)
        for channel in self.recall_channels:
            ranked_lists[channel.name] = channel.recall(
                profile=profile,
                candidates=active_catalog,
                limit=channel_limit,
            )

        fused = rrf_fuse(
            ranked_lists=ranked_lists,
            k=60,
            top_k=self.recall_top_k,
            category_map=category_map,
            target_categories=profile.target_categories,
            min_per_category=3,
        )
        recalled_products = [catalog_map[pid] for pid, _, _ in fused if pid in catalog_map]
        rrf_scores = {pid: score for pid, score, _ in fused}
        recall_ranks = {pid: ranks for pid, _, ranks in fused}
        stages.append(
            FunnelStageMetric(
                stage="recall_rrf",
                input_count=len(active_catalog),
                output_count=len(recalled_products),
                latency_ms=round((time.perf_counter() - t1) * 1000, 3),
                detail={"channels": list(ranked_lists.keys())},
            )
        )

        # ---------------- Stage 2: 承保资格硬约束粗排 ----------------
        t2 = time.perf_counter()
        eligible, rejections = filter_eligible_candidates(
            candidates=recalled_products,
            profile=profile,
            estimator=self.estimator,
        )
        stages.append(
            FunnelStageMetric(
                stage="hard_filter",
                input_count=len(recalled_products),
                output_count=len(eligible),
                latency_ms=round((time.perf_counter() - t2) * 1000, 3),
                detail={"rejected_count": len(rejections)},
            )
        )

        # ---------------- Stage 3: 5 维可解释多因子精排 ----------------
        t3 = time.perf_counter()
        scored = score_candidates(
            eligible=eligible,
            profile=profile,
            rrf_scores=rrf_scores,
            recall_ranks=recall_ranks,
        )
        stages.append(
            FunnelStageMetric(
                stage="multi_factor_rank",
                input_count=len(eligible),
                output_count=len(scored),
                latency_ms=round((time.perf_counter() - t3) * 1000, 3),
            )
        )

        # ---------------- Stage 4: MMR 多样性重排 + 预算组合优化 ----------------
        t4 = time.perf_counter()
        diversified = mmr_diversify(scored, lambda_param=0.78, top_k=12)
        portfolio = optimize_budget_portfolio(diversified, profile=profile, max_items=max_items)
        portfolio.ranked_candidates = scored
        portfolio.diversified_candidates = diversified
        stages.append(
            FunnelStageMetric(
                stage="mmr_portfolio_rerank",
                input_count=len(scored),
                output_count=len(portfolio.selected),
                latency_ms=round((time.perf_counter() - t4) * 1000, 3),
                detail={
                    "covered_categories": portfolio.covered_categories,
                    "total_estimated_premium": portfolio.total_estimated_premium,
                    "within_budget": portfolio.within_budget,
                },
            )
        )

        total_latency_ms = round((time.perf_counter() - total_start) * 1000, 3)
        portfolio.trace = FunnelTrace(
            stages=stages,
            total_latency_ms=total_latency_ms,
            rejections=rejections,
        )

        logger.info(
            "搜推漏斗执行完成",
            funnel_counts=[s.output_count for s in stages],
            total_latency_ms=total_latency_ms,
            selected_ids=[x.product.id for x in portfolio.selected],
            total_premium=portfolio.total_estimated_premium,
            within_budget=portfolio.within_budget,
        )
        return portfolio
