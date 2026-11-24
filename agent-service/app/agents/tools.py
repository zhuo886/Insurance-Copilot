import json
from typing import Any, Literal

from langchain.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from app.core.logging import get_logger
from app.infra.database import AsyncSessionFactory
from app.modules.insurance_plan.service import InsurancePlanService
from app.modules.product.catalog_data import load_builtin_catalog
from app.modules.product.repository import ProductRepository
from app.rag import get_retriever
from app.recsys import (
    BM25KeywordRecall,
    MilvusProductRecall,
    PremiumEstimator,
    RecSysPipeline,
    RecUserProfile,
    StructuredRuleRecall,
)
from app.recsys.schemas import ProductCandidate

logger = get_logger(__name__)


def _safe_stream_write(payload: dict[str, Any]) -> None:
    """在流式上下文外（如单测或离线评测）安全忽略 stream writer 不可用异常。"""
    try:
        writer = get_stream_writer()
        writer({"type": "additional_info", "data": payload})
    except Exception:
        pass


def _get_user_id(config: RunnableConfig | None) -> int:
    configurable = (config or {}).get("configurable") or {}
    raw_uid = configurable.get("user_id")
    return int(raw_uid) if raw_uid is not None else 1


async def _load_catalog_candidates() -> list[ProductCandidate]:
    """优先从 PG 读取全量产品与结构化特征，不可用时无缝回退内置 12 款真实目录。"""
    try:
        async with AsyncSessionFactory() as session:
            repo = ProductRepository(session)
            rows = await repo.find_all_with_features()
        if rows:
            builtin_map = {item.id: item for item in load_builtin_catalog()}
            candidates: list[ProductCandidate] = []
            for product, feature in rows:
                fallback = builtin_map.get(product.id)
                candidates.append(
                    ProductCandidate(
                        id=product.id,
                        name=product.name,
                        clause_name=product.clause_name,
                        category=product.category,  # type: ignore[arg-type]
                        insurer=product.insurer,
                        image_url=product.image_url,
                        description=product.description or "",
                        min_premium=float(product.min_premium or 0.0),
                        max_premium=float(product.max_premium) if product.max_premium is not None else None,
                        target_group=product.target_group or "",
                        highlights=list(product.highlights or []),
                        status=product.status,
                        min_age=feature.min_age if feature else (fallback.min_age if fallback else 0),
                        max_age=feature.max_age if feature else (fallback.max_age if fallback else 65),
                        max_occupation_class=(
                            feature.max_occupation_class
                            if feature
                            else (fallback.max_occupation_class if fallback else 4)
                        ),
                        waiting_period_days=(
                            feature.waiting_period_days
                            if feature
                            else (fallback.waiting_period_days if fallback else 0)
                        ),
                        guarantee_renewal_years=(
                            feature.guarantee_renewal_years
                            if feature
                            else (fallback.guarantee_renewal_years if fallback else None)
                        ),
                        deductible_default=(
                            float(feature.deductible_default)
                            if feature
                            else (fallback.deductible_default if fallback else 0.0)
                        ),
                        has_clause=feature.has_clause if feature else (product.id <= 12),
                        is_featured=feature.is_featured if feature else (product.id <= 12),
                        coverage_tags=(
                            list(feature.coverage_tags or [])
                            if feature
                            else (fallback.coverage_tags if fallback else [])
                        ),
                        health_disclosure_level=(
                            feature.health_disclosure_level  # type: ignore[arg-type]
                            if feature
                            else (fallback.health_disclosure_level if fallback else "standard")
                        ),
                        insurer_tier=(
                            feature.insurer_tier  # type: ignore[arg-type]
                            if feature
                            else (fallback.insurer_tier if fallback else "top")
                        ),
                        base_rate_factor=(
                            float(feature.base_rate_factor)
                            if feature
                            else (fallback.base_rate_factor if fallback else 1.0)
                        ),
                    )
                )
            return candidates
    except Exception as exc:
        logger.warning("从数据库加载产品特征失败，回退内置 12 款目录", error=str(exc))

    return load_builtin_catalog()


async def _load_rate_tables(product_ids: list[int] | None = None) -> dict[int, list[dict[str, Any]]] | None:
    try:
        async with AsyncSessionFactory() as session:
            service = InsurancePlanService(session)
            items = await service.repository.find_rate_table_items(product_ids)
            return items or None
    except Exception:
        return None


@tool
async def recommend_insurance_portfolio(
    age: int | None = None,
    gender: Literal["male", "female"] = "male",
    occupation_class: int = 1,
    has_social_security: bool = True,
    annual_budget: float | None = None,
    target_categories: list[str] | None = None,
    preferences: list[str] | None = None,
    query_text: str = "",
) -> str:
    """运行四阶段搜推漏斗（多路召回RRF -> 承保硬过滤 -> 5维可解释精排 -> 预算组合优化），
    为用户提供符合年龄/职业资格和预算的保险组合推荐。

    Args:
        age: 被保人年龄（必填，未知时传 None 将触发缺槽提示）。
        gender: 被保人性别，male 或 female。
        occupation_class: 职业类别 1-6（默认 1 类）。
        has_social_security: 是否有医保/社保。
        annual_budget: 年缴保费预算（元），未知时可传 None。
        target_categories: 目标险种列表（medical / critical_illness / accident / life）。
        preferences: 保障偏好标签列表（如 保证续保20年、外购药、0免赔、猝死保障）。
        query_text: 用户需求自然语言描述。
    """
    missing: list[str] = []
    if age is None:
        missing.append("被保人年龄")
    if annual_budget is None and not target_categories:
        missing.append("年保费预算或关注的险种")
    if missing:
        return json.dumps(
            {
                "status": "missing_slots",
                "missing_fields": missing,
                "message": f"生成精准推荐方案前，请先向用户温和确认以下必要信息：{'、'.join(missing)}。",
            },
            ensure_ascii=False,
        )

    catalog = await _load_catalog_candidates()
    rate_tables = await _load_rate_tables()
    estimator = PremiumEstimator(rate_table_items=rate_tables)
    pipeline = RecSysPipeline(
        estimator=estimator,
        recall_channels=(
            MilvusProductRecall(),
            BM25KeywordRecall(),
            StructuredRuleRecall(),
        ),
        recall_top_k=15,
    )

    profile = RecUserProfile(
        age=int(age or 30),
        gender=gender,
        occupation_class=occupation_class,
        has_social_security=has_social_security,
        annual_budget=float(annual_budget or 0.0),
        target_categories=list(target_categories or []),
        preferences=list(preferences or []),
        query_text=query_text,
    )
    portfolio = pipeline.recommend(profile, catalog, max_items=4)

    candidate_cards = [
        {
            "id": item.product.id,
            "product_id": item.product.id,
            "name": item.product.name,
            "category": item.product.category,
            "insurer": item.product.insurer,
            "image_url": item.product.image_url,
            "description": item.product.description,
            "min_premium": item.estimate.annual_premium,
            "target_group": item.product.target_group,
            "highlights": item.product.highlights,
            "recommendation_reason": (
                f"综合匹配分 {item.scores.total_score:.2f}（估算年保费 ¥{item.estimate.annual_premium:.0f}）"
            ),
        }
        for item in portfolio.selected
    ]
    if candidate_cards:
        _safe_stream_write({"candidate_products": candidate_cards})

    payload = {
        "status": "ok",
        "selected": [
            {
                "product_id": x.product.id,
                "name": x.product.name,
                "category": x.product.category,
                "insurer": x.product.insurer,
                "has_clause": x.product.has_clause,
                "estimated_annual_premium": x.estimate.annual_premium,
                "pricing_mode": x.estimate.pricing_mode,
                "is_estimated": x.estimate.is_estimated,
                "pricing_disclaimer": x.estimate.pricing_disclaimer,
                "highlights": x.product.highlights,
                "score_breakdown": x.scores.model_dump(),
            }
            for x in portfolio.selected
        ],
        "alternatives": [
            {
                "product_id": x.product.id,
                "name": x.product.name,
                "category": x.product.category,
                "estimated_annual_premium": x.estimate.annual_premium,
                "total_score": x.scores.total_score,
            }
            for x in portfolio.alternatives[:3]
        ],
        "total_estimated_premium": portfolio.total_estimated_premium,
        "within_budget": portfolio.within_budget,
        "budget_gap": portfolio.budget_gap,
        "covered_categories": portfolio.covered_categories,
        "missing_categories": portfolio.missing_categories,
        "funnel_summary": [
            {
                "stage": s.stage,
                "in": s.input_count,
                "out": s.output_count,
                "ms": s.latency_ms,
            }
            for s in portfolio.trace.stages
        ],
    }
    return json.dumps(payload, ensure_ascii=False)


@tool
async def query_product_clause(
    query: str,
    product_id: int | None = None,
    product_ids: list[int] | None = None,
    k: int = 4,
) -> str:
    """查询一款或多款保险产品的条款原文（支持单品细则、多产品横向对比及全库条款检索）。

    Args:
        query: 关于保险条款的具体问题。
        product_id: 单款产品 ID（可选）。
        product_ids: 多款对比产品 ID 列表（可选，如 [4, 6]）。
        k: 返回的条款父块数量，默认 4。
    """
    chunks = await get_retriever().retrieve(
        query=query,
        k=k,
        product_id=product_id,
        product_ids=product_ids,
    )
    if not chunks:
        return json.dumps(
            {
                "status": "empty",
                "source_ids": [],
                "sources": {},
                "message": "未在条款知识库中检索到该产品的真实合同条款原文，暂不作条款级解读。",
            },
            ensure_ascii=False,
        )

    sources = {
        f"ref-{i:03d}": {
            "source_id": f"ref-{i:03d}",
            "product_id": chunk.product_id,
            "clause_name": chunk.clause_name,
            "section_path": chunk.section_path,
            "content": chunk.content,
        }
        for i, chunk in enumerate(chunks, start=1)
    }

    _safe_stream_write({"clause_sources": sources})

    return json.dumps(
        {
            "status": "ok",
            "source_ids": list(sources.keys()),
            "sources": sources,
        },
        ensure_ascii=False,
    )


@tool
async def quote_premium(
    product_id: int,
    age: int,
    gender: Literal["male", "female"] = "male",
    occupation_class: int = 1,
    has_social_security: bool = True,
    coverage_amount: float | None = None,
    deductible: float | None = None,
    plan_code: str = "standard",
    config: RunnableConfig | None = None,
) -> str:
    """精确试算指定保险产品的年缴保费（三级降级：费率表 -> Mock适配因子 -> 特征近似估算）并持久化试算记录。

    Args:
        product_id: 保险产品 ID。
        age: 被保人年龄。
        gender: 性别 male 或 female。
        occupation_class: 职业类别 1-6。
        has_social_security: 是否有医保/社保。
        coverage_amount: 保额（元，可选，如 500000）。
        deductible: 免赔额（元，可选，如 0 或 10000）。
        plan_code: 计划代码（默认 standard）。
    """
    catalog = {p.id: p for p in await _load_catalog_candidates()}
    product = catalog.get(product_id)
    if product is None:
        return json.dumps({"status": "error", "message": f"未找到产品 ID={product_id}"}, ensure_ascii=False)

    if age < product.min_age or age > product.max_age or occupation_class > product.max_occupation_class:
        return json.dumps(
            {
                "status": "ineligible",
                "product_id": product_id,
                "product_name": product.name,
                "message": (
                    f"被保人（{age}岁，{occupation_class}类职业）不符合《{product.name}》承保条件"
                    f"（适用年龄 {product.min_age}-{product.max_age}岁，1-{product.max_occupation_class}类职业）。"
                ),
            },
            ensure_ascii=False,
        )

    rate_tables = await _load_rate_tables([product_id])
    estimator = PremiumEstimator(rate_table_items=rate_tables)
    profile = RecUserProfile(
        age=age,
        gender=gender,
        occupation_class=occupation_class,
        has_social_security=has_social_security,
        coverage_amount=coverage_amount,
    )
    estimate = estimator.estimate(
        product,
        profile,
        plan_code=plan_code,
        coverage_amount=coverage_amount,
        deductible=deductible,
    )

    user_id = _get_user_id(config)
    req_params = {
        "age": age,
        "gender": gender,
        "occupation_class": occupation_class,
        "has_social_security": has_social_security,
        "coverage_amount": coverage_amount,
        "deductible": deductible,
        "plan_code": plan_code,
    }
    res_payload = {
        "product_id": product.id,
        "product_name": product.name,
        "annual_premium": estimate.annual_premium,
        "pricing_mode": estimate.pricing_mode,
        "is_estimated": estimate.is_estimated,
        "pricing_disclaimer": estimate.pricing_disclaimer,
        "plan_code": estimate.plan_code,
        "breakdown": estimate.breakdown,
    }
    try:
        async with AsyncSessionFactory() as session:
            service = InsurancePlanService(session)
            quote_id = await service.save_premium_quote(
                user_id=user_id,
                product_id=product.id,
                request_params=req_params,
                result=res_payload,
                total_premium=estimate.annual_premium,
                pricing_mode=estimate.pricing_mode,
            )
            res_payload["quote_id"] = str(quote_id)
    except Exception as exc:
        logger.warning("保费试算记录落库跳过", error=str(exc))

    return json.dumps({"status": "ok", **res_payload}, ensure_ascii=False)


@tool
async def submit_plan_draft(
    plan_name: str,
    summary: str,
    items: list[dict[str, Any]],
    annual_premium_budget: float | None = None,
) -> str:
    """当用户接受或希望保存推荐的保险配置方案时调用，提交结构化方案草稿进入确认审批流。

    Args:
        plan_name: 方案名称，如「30岁青年家庭基础保障方案」。
        summary: 方案总体配置思路与预算说明。
        items: 方案包含的产品列表，每项需包含 product_id（整数）、recommendation_reason（理由）、annual_premium_budget（预估年保费）、priority（优先级，从1开始）。
        annual_premium_budget: 方案合计预估年保费或总预算（元）。
    """
    catalog = {p.id: p for p in await _load_catalog_candidates()}
    enriched_items: list[dict[str, Any]] = []
    seen_pids: set[int] = set()
    total_prem = 0.0

    for idx, raw in enumerate(items, start=1):
        if not isinstance(raw, dict) or "product_id" not in raw:
            continue
        pid = int(raw["product_id"])
        if pid in seen_pids:
            continue
        seen_pids.add(pid)
        product = catalog.get(pid)
        item_prem = raw.get("annual_premium_budget")
        if item_prem is not None:
            total_prem += float(item_prem)
        enriched_items.append(
            {
                "product_id": pid,
                "name": product.name if product else raw.get("name", f"产品{pid}"),
                "category": product.category if product else raw.get("category", "medical"),
                "image_url": product.image_url if product else None,
                "description": product.description if product else "",
                "highlights": product.highlights if product else [],
                "priority": int(raw.get("priority") or idx),
                "recommendation_reason": raw.get("recommendation_reason") or "贴合当前保障需求",
                "annual_premium_budget": float(item_prem) if item_prem is not None else (
                    product.min_premium if product else None
                ),
            }
        )

    plan_draft = {
        "plan_name": plan_name,
        "summary": summary,
        "annual_premium_budget": (
            round(float(annual_premium_budget), 2)
            if annual_premium_budget is not None
            else round(total_prem, 2)
        ),
        "items": enriched_items,
    }
    return json.dumps({"status": "draft_submitted", "plan_draft": plan_draft}, ensure_ascii=False)


@tool
async def query_user_policies_and_claims(config: RunnableConfig | None = None) -> str:
    """查询当前用户的真实保单列表及历史理赔进度（自动按当前登录用户隔离），在理赔咨询时优先调用。"""
    user_id = _get_user_id(config)
    async with AsyncSessionFactory() as session:
        service = InsurancePlanService(session)
        data = await service.list_user_policies_and_claims(user_id)

    if data["policies"]:
        _safe_stream_write({"policies": data["policies"]})

    return json.dumps({"status": "ok", **data}, ensure_ascii=False)


@tool
async def get_claim_guide(
    policy_id: str,
    config: RunnableConfig | None = None,
) -> str:
    """根据用户保单 UUID 生成结构化理赔指引（报案步骤、所需材料清单与申请表 PDF 下载地址）。"""
    from uuid import UUID

    from app.modules.claim.service import ClaimService

    user_id = _get_user_id(config)
    try:
        pid = UUID(str(policy_id).strip())
    except ValueError:
        return json.dumps({"status": "error", "message": "policy_id 格式非法，需为保单 UUID"}, ensure_ascii=False)

    async with AsyncSessionFactory() as session:
        service = ClaimService(session)
        try:
            guide = await service.get_claim_guide(user_id=user_id, policy_id=pid)
        except Exception as exc:
            return json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False)

    return json.dumps({"status": "ok", **guide.model_dump(mode="json")}, ensure_ascii=False)


# 保留兼容别名，防止旧调试脚本引用报错
query_candidate_products = recommend_insurance_portfolio
