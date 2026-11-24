from typing import Any, Literal

from pydantic import BaseModel, Field


class ProductCandidate(BaseModel):
    """搜推漏斗统一产品候选表示（融合 products 与 product_features）"""

    id: int
    name: str
    clause_name: str
    category: Literal["medical", "critical_illness", "accident", "life"]
    insurer: str
    image_url: str | None = None
    description: str = ""
    min_premium: float = 0.0
    max_premium: float | None = None
    target_group: str = ""
    highlights: list[str] = Field(default_factory=list)
    status: str = "active"

    # 结构化承保与搜推特征
    min_age: int = 0
    max_age: int = 65
    max_occupation_class: int = 4
    waiting_period_days: int = 0
    guarantee_renewal_years: int | None = None
    deductible_default: float = 0.0
    has_clause: bool = False
    is_featured: bool = False
    coverage_tags: list[str] = Field(default_factory=list)
    health_disclosure_level: Literal["loose", "standard", "strict"] = "standard"
    insurer_tier: Literal["top", "joint", "internet"] = "top"
    base_rate_factor: float = 1.0

    def to_semantic_text(self) -> str:
        """构建产品目录检索用语义描述文本（兼顾 Dense 向量与 BM25 词法匹配）"""
        category_cn = {
            "medical": "医疗险 住院医疗 百万医疗",
            "critical_illness": "重疾险 重大疾病保险",
            "accident": "意外险 综合意外伤害",
            "life": "寿险 终身寿险 定期寿险",
        }.get(self.category, self.category)
        renewal_text = (
            f"保证续保{self.guarantee_renewal_years}年"
            if self.guarantee_renewal_years
            else "一年期或长期保障"
        )
        deductible_text = (
            "0免赔" if self.deductible_default == 0 else f"免赔额{int(self.deductible_default)}元"
        )
        return " | ".join(
            [
                self.name,
                category_cn,
                self.insurer,
                f"适用人群:{self.target_group}",
                f"投保年龄:{self.min_age}-{self.max_age}岁",
                f"承保职业:1-{self.max_occupation_class}类",
                f"等待期:{self.waiting_period_days}天",
                renewal_text,
                deductible_text,
                f"核心亮点:{' '.join(self.highlights)}",
                f"保障标签:{' '.join(self.coverage_tags)}",
                self.description,
            ]
        )


class RecUserProfile(BaseModel):
    """搜推漏斗输入用户画像"""

    age: int = Field(default=30, ge=0, le=120)
    gender: Literal["male", "female"] = "male"
    occupation_class: int = Field(default=1, ge=1, le=6)
    has_social_security: bool = True
    annual_budget: float = Field(default=3000.0, ge=0)
    target_categories: list[str] = Field(default_factory=list)
    family_role: str | None = None
    preferences: list[str] = Field(default_factory=list)
    coverage_amount: float | None = None
    query_text: str = ""


class PremiumEstimate(BaseModel):
    """保费试算结果"""

    product_id: int
    annual_premium: float
    pricing_mode: Literal["table", "estimated_formula"]
    plan_code: str = "standard"
    is_estimated: bool = False
    pricing_disclaimer: str = ""
    breakdown: dict[str, Any] = Field(default_factory=dict)


class FilterRejection(BaseModel):
    """硬约束过滤剔除记录"""

    product_id: int
    product_name: str
    category: str
    reasons: list[str] = Field(default_factory=list)


class ScoreBreakdown(BaseModel):
    """精排 5 维可解释得分明细"""

    semantic_rrf: float
    profile_fit: float
    coverage_tag_match: float
    budget_efficiency: float
    clause_quality: float
    total_score: float


class ScoredCandidate(BaseModel):
    """经过精排打分的候选产品"""

    product: ProductCandidate
    estimate: PremiumEstimate
    rrf_score: float = 0.0
    recall_ranks: dict[str, int] = Field(default_factory=dict)
    scores: ScoreBreakdown


class FunnelStageMetric(BaseModel):
    """单阶段漏斗可观测指标"""

    stage: str
    input_count: int
    output_count: int
    latency_ms: float
    detail: dict[str, Any] = Field(default_factory=dict)


class FunnelTrace(BaseModel):
    """全链路漏斗诊断追踪"""

    stages: list[FunnelStageMetric] = Field(default_factory=list)
    total_latency_ms: float = 0.0
    rejections: list[FilterRejection] = Field(default_factory=list)


class PortfolioResult(BaseModel):
    """四阶段搜推漏斗最终产出（含最优组合与备选列表）"""

    selected: list[ScoredCandidate] = Field(default_factory=list)
    alternatives: list[ScoredCandidate] = Field(default_factory=list)
    ranked_candidates: list[ScoredCandidate] = Field(default_factory=list)
    diversified_candidates: list[ScoredCandidate] = Field(default_factory=list)
    total_estimated_premium: float = 0.0
    covered_categories: list[str] = Field(default_factory=list)
    missing_categories: list[str] = Field(default_factory=list)
    within_budget: bool = True
    budget_gap: float = 0.0
    trace: FunnelTrace = Field(default_factory=FunnelTrace)
