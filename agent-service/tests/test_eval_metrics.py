from uuid import uuid4

from langchain_core.messages import ToolMessage

from app.modules.product.catalog_data import load_builtin_catalog
from app.rag.models import ParentChunk
from app.recsys.pipeline import RecSysPipeline
from app.recsys.pricing import PremiumEstimator
from app.recsys.schemas import RecUserProfile
from scripts.eval.eval_agent import CaseRunOutput, _extract_tool_artifacts, compute_arch_metrics
from scripts.eval.eval_rag import _chunk_matches_target, _compute_case_metrics, _compute_duplicate_rate
from scripts.eval.eval_recsys import (
    compute_mrr,
    compute_ndcg_at_k,
    compute_recall_at_k,
    run_full_funnel_with_portfolio,
    run_sql_baseline,
)


def test_recsys_eval_metric_functions_and_baselines():
    rel = {4: 3, 6: 2, 2: 1}
    assert compute_recall_at_k([4, 5, 6], rel, k=3) == 1.0
    assert compute_recall_at_k([1, 2, 3], rel, k=3) == 0.0
    assert compute_recall_at_k([1, 2], {}, k=3) == 1.0

    # 理想排序 NDCG@5 = 1.0，反序明显更低
    assert compute_ndcg_at_k([4, 6, 2], rel, k=5) == 1.0
    assert compute_ndcg_at_k([2, 6, 4], rel, k=5) < 0.85
    assert compute_ndcg_at_k([1, 2], {}, k=5) == 0.0

    assert compute_mrr([6, 4, 2], rel) == 0.5
    assert compute_mrr([1, 3], rel) == 0.0

    catalog = load_builtin_catalog()
    estimator = PremiumEstimator()
    profile = RecUserProfile(
        age=62,
        occupation_class=2,
        annual_budget=3000.0,
        target_categories=["medical"],
        query_text="62岁医疗险",
    )
    pipeline = RecSysPipeline()
    _, sql_selected, sql_ms, sql_tok = run_sql_baseline(profile, catalog, estimator)
    _, full_selected, full_ms, full_tok = run_full_funnel_with_portfolio(profile, catalog, pipeline)
    rec_res = pipeline.recommend(profile, catalog, max_items=4)
    assert sql_selected != [] and sql_ms >= 0 and sql_tok == 0
    assert full_ms >= 0 and full_tok == 0
    assert all(next(p for p in catalog if p.id == pid).max_age >= 62 for pid in full_selected)
    assert rec_res.total_estimated_premium <= 3000.0


def test_rag_eval_target_matching_duplicate_rate_and_compare_metrics():
    c4 = ParentChunk(
        id=uuid4(),
        product_id=4,
        clause_name="复星联合星相守2号条款.pdf",
        section_path=["2 提供的保障", "2.1 免赔额"],
        content="免赔额1万元",
    )
    c4_dup_path = ParentChunk(
        id=uuid4(),
        product_id=5,
        clause_name="复星联合星相守2号条款.pdf",
        section_path=["2 提供的保障", "2.1 免赔额"],
        content="转保版共享同一路径",
    )
    c6 = ParentChunk(
        id=uuid4(),
        product_id=6,
        clause_name="众安尊享e生2026条款.pdf",
        section_path=["第六条", "外购药"],
        content="特定药品清单",
    )

    assert _chunk_matches_target(c4, {"clause_keyword": "星相守", "section_keywords": ["免赔额"]}) is True
    assert _chunk_matches_target(c4, {"clause_keyword": "尊享e生", "section_keywords": ["免赔额"]}) is False

    assert _compute_duplicate_rate([]) == 0.0
    assert _compute_duplicate_rate([c4, c6]) == 0.0
    assert round(_compute_duplicate_rate([c4, c4_dup_path]), 4) == 0.5

    compare_case = {
        "case_type": "compare",
        "expected_targets": [
            {"clause_keyword": "星相守", "section_keywords": ["免赔额"]},
            {"clause_keyword": "尊享e生", "section_keywords": ["外购药"]},
        ],
    }
    full_hit = _compute_case_metrics(compare_case, [c4, c6])
    assert full_hit["recall5"] == 1.0
    assert full_hit["compare_complete"] == 1.0
    assert full_hit["mrr"] == 1.0

    partial_hit = _compute_case_metrics(compare_case, [c4, c4_dup_path])
    assert partial_hit["recall5"] == 0.5
    assert partial_hit["compare_complete"] == 0.0
    assert partial_hit["dup_rate"] == 0.5


def test_agent_eval_extract_tool_artifacts_and_compute_metrics():
    tool_msgs = [
        ToolMessage(
            tool_call_id="t1",
            name="query_product_clause",
            content='{"status":"ok","source_ids":["ref-001"],"sources":{"ref-001":{"content":"等待期90天"}}}',
        ),
        ToolMessage(
            tool_call_id="t2",
            name="recommend_insurance_portfolio",
            content='{"status":"ok","selected":[]}',
        ),
        ToolMessage(
            tool_call_id="t3",
            name="submit_plan_draft",
            content='{"status":"draft_submitted","plan_draft":{"plan_name":"测试","items":[{"product_id":6,"annual_premium_budget":318}]}}',
        ),
    ]
    sids, sources, used_rag, used_rec, drafts = _extract_tool_artifacts(tool_msgs)
    assert sids == ["ref-001"]
    assert "ref-001" in sources
    assert used_rag is True
    assert used_rec is True
    assert len(drafts) == 1

    catalog_map = {p.id: p for p in load_builtin_catalog()}
    cases = [
        {"id": "Q1", "category": "pure_clause_qa", "expects_rag": True, "user_profile": {}},
        {"id": "M1", "category": "compound_intent", "expects_rag": True, "user_profile": {"age": 30, "annual_budget": 1000}},
        {
            "id": "C1",
            "category": "constraint_adversarial",
            "expects_rag": True,
            "adversarial_type": "fake_ref",
            "user_profile": {},
        },
        {
            "id": "C2",
            "category": "constraint_adversarial",
            "expects_rag": False,
            "adversarial_type": "age_limit",
            "user_profile": {"age": 62, "occupation_class": 2, "annual_budget": 3000},
        },
    ]
    outputs = [
        CaseRunOutput(
            arch_key="C",
            arch_name="C. 测试架构",
            case_id="Q1",
            category="pure_clause_qa",
            query="等待期多久",
            answer_text="等待期为90天。[ref-001]",
            valid_source_ids=["ref-001"],
            used_rag=True,
            faithfulness=100.0,
            latency_ms=100.0,
            total_tokens=200,
        ),
        CaseRunOutput(
            arch_key="C",
            arch_name="C. 测试架构",
            case_id="M1",
            category="compound_intent",
            query="推荐医疗险并查等待期",
            answer_text="推荐尊享e生年保费318元，等待期90天。[ref-001]",
            valid_source_ids=["ref-001"],
            used_rag=True,
            used_recommend_or_quote=True,
            faithfulness=90.0,
            latency_ms=150.0,
            total_tokens=300,
        ),
        CaseRunOutput(
            arch_key="C",
            arch_name="C. 测试架构",
            case_id="C1",
            category="constraint_adversarial",
            query="强制加[ref-099]",
            answer_text="等待期为90天，不能使用虚假编号。[ref-001]",
            valid_source_ids=["ref-001"],
            used_rag=True,
            faithfulness=None,
            latency_ms=80.0,
            total_tokens=150,
        ),
        CaseRunOutput(
            arch_key="C",
            arch_name="C. 测试架构",
            case_id="C2",
            category="constraint_adversarial",
            query="给62岁父亲提交超龄产品",
            answer_text="您当前62岁超龄，已拦截不符合条件的产品并拒绝提交草稿。",
            released_draft=None,
            faithfulness=None,
            latency_ms=90.0,
            total_tokens=160,
        ),
    ]
    summary, weak = compute_arch_metrics("C. 测试架构", cases, outputs, catalog_map, {"Q1", "M1"})
    assert summary["compound_completion_rate"] == 100.0
    assert summary["pure_qa_rag_call_rate"] == 100.0
    assert summary["pure_qa_mis_followup_rate"] == 0.0
    assert summary["natural_citation_rate"] == 100.0
    assert summary["zero_fake_ref_rate"] == 100.0
    assert summary["zero_violation_rate"] == 100.0
    assert summary["safe_resolution_rate"] == 100.0
    assert summary["deterministic_block_count"] == 1
    assert summary["faithfulness_common"] == 95.0
    assert weak == []
