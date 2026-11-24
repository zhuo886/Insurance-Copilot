import pytest

from app.agents.guardrails import (
    extract_all_citation_tokens,
    sanitize_citations,
    strip_thinking_tags,
    verify_citations,
    verify_plan_draft,
)
from app.agents.profile_extractor import extract_profile_by_regex
from app.modules.product.catalog_data import load_builtin_catalog


def test_extract_profile_empty_and_newborn_and_colon_age():
    assert extract_profile_by_regex("   ") == {}
    assert extract_profile_by_regex("给刚满月的新生儿女宝买保险")["age"] == 0
    assert extract_profile_by_regex("半岁男宝，有少儿医保")["has_social_security"] is True
    assert extract_profile_by_regex("年龄：45，预算6000元")["age"] == 45
    assert "age" not in extract_profile_by_regex("年龄：150岁")


@pytest.mark.parametrize(
    ("text", "expected_gender"),
    [
        ("28岁男性程序员", "male"),
        ("男女通用产品有什么", None),
        ("给妈妈挑一款医疗险", "female"),
        ("26岁女士预算3000", "female"),
    ],
)
def test_extract_profile_gender_disambiguation(text: str, expected_gender: str | None):
    res = extract_profile_by_regex(text)
    assert res.get("gender") == expected_gender


@pytest.mark.parametrize(
    ("text", "expected_occ"),
    [
        ("30岁3类职业预算4000", 3),
        ("货车司机长期跑高速", 6),
        ("一线消防员需要意外保障", 6),
        ("外卖骑手每天风里来雨里去", 5),
        ("工厂电工平时风险较高", 5),
    ],
)
def test_extract_profile_occupation_mapping(text: str, expected_occ: int):
    assert extract_profile_by_regex(text)["occupation_class"] == expected_occ


def test_extract_profile_budget_units_and_bounds():
    assert extract_profile_by_regex("预算5k配重疾险")["annual_budget"] == 5000.0
    assert extract_profile_by_regex("每年大概花0.8万")["annual_budget"] == 8000.0
    assert extract_profile_by_regex("保费控制在50元以内") == {}
    assert "annual_budget" not in extract_profile_by_regex("0免赔额医疗险怎么选")
    assert "annual_budget" not in extract_profile_by_regex("50万保额的重疾险有哪些")


def test_extract_profile_roles_categories_and_preferences():
    res = extract_profile_by_regex("家庭支柱有房贷，想看定期寿险和多次赔付重疾，需要特需医疗和保证续保")
    assert res["family_role"] == "家庭支柱"
    assert res["target_categories"] == ["medical", "critical_illness", "life"]
    assert "定期寿险" in res["preferences"]
    assert "多次赔付" in res["preferences"]
    assert "特需医疗" in res["preferences"]
    assert "保证续保20年" in res["preferences"]


def test_strip_thinking_tags_variants():
    assert strip_thinking_tags("<THINKING>草稿</thinking>\n正文结论") == "正文结论"
    assert strip_thinking_tags("<thinking>未闭合思维内容\n\n真实答案") == "真实答案"
    assert strip_thinking_tags("<thinking>仅内部内容</thinking>") == "仅内部内容"


def test_extract_all_citation_tokens_ignores_normal_brackets_and_captures_fake_formats():
    text = (
        "产品[推荐][2026][100万][abcdef][123456]不计入；"
        "但[ref-001]、[source-12]、[doc-abc]、[clause-1]、[d64a1]与裸 ref-009 均需识别。"
    )
    tokens = extract_all_citation_tokens(text)
    assert tokens == ["ref-001", "source-12", "doc-abc", "clause-1", "d64a1", "ref-009"]


def test_verify_citations_strictness_and_no_clause_exemptions():
    # 命中 RAG 时，非法编号拦截
    errs = verify_citations("等待期180天[ref-001, ref-009]。", {"ref-001"}, rag_used=True)
    assert len(errs) == 1 and "ref-009" in errs[0]

    # 命中 RAG 且有效回答 >= 15 字却未标注任何引文，必须提醒
    assert len(verify_citations("这款重疾险等待期是九十天并且自带特定疾病双倍赔付责任。", {"ref-001"}, rag_used=True)) == 1

    # 声明未查询到真实条款时豁免强制引文
    assert (
        verify_citations("知识库中暂未查询到该产品的完整合同条款，无法确认细节。", {"ref-001"}, rag_used=True)
        == []
    )

    # 短问答 < 15 字豁免强制引文
    assert verify_citations("等待期为90天。", {"ref-001"}, rag_used=True) == []

    # 未触发 RAG 且无任何伪编号应通过
    assert verify_citations("您好，请告诉我被保人年龄和预算。", set(), rag_used=False) == []


def test_sanitize_citations_preserves_valid_and_ordinary_brackets():
    raw = "条款[推荐]表明等待期180天[ref-001, d64a1, ref-999] ，且无裸引文 ref-888 。"
    cleaned = sanitize_citations(raw, {"ref-001"}, rag_used=True)
    assert "[推荐]" in cleaned
    assert "[ref-001]" in cleaned
    assert "d64a1" not in cleaned
    assert "ref-999" not in cleaned
    assert "ref-888" not in cleaned
    assert "  " not in cleaned


def test_verify_plan_draft_edge_cases():
    catalog_map = {p.id: p for p in load_builtin_catalog()}

    assert verify_plan_draft(None, {"age": 30}, catalog_map) == []
    assert "未包含任何有效保险产品" in verify_plan_draft({"items": []}, {"age": 30}, catalog_map)[0]

    missing_id = verify_plan_draft({"items": [{"product_id": 999}]}, {"age": 30}, catalog_map)
    assert any("不存在的产品 ID=999" in e for e in missing_id)

    # ID 3 医联有盟 18-59 周岁
    under_age = verify_plan_draft(
        {"items": [{"product_id": 3, "annual_premium_budget": 1000}]},
        {"age": 10, "occupation_class": 1, "annual_budget": 5000},
        catalog_map,
    )
    assert any("承保年龄为 18-59 周岁" in e for e in under_age)

    # ID 10 小蜜蜂6号 max_occupation_class=3
    over_occ = verify_plan_draft(
        {"items": [{"product_id": 10, "annual_premium_budget": 136}]},
        {"age": 30, "occupation_class": 5, "annual_budget": 2000},
        catalog_map,
    )
    assert any("最高承保 1-3 类职业" in e for e in over_occ)

    # 1.10 容差边界：预算 1000，1100 放行，1101 拦截
    within_tol = verify_plan_draft(
        {"items": [{"product_id": 6, "annual_premium_budget": 1100}]},
        {"age": 30, "occupation_class": 1, "annual_budget": 1000},
        catalog_map,
    )
    assert within_tol == []

    over_tol = verify_plan_draft(
        {"items": [{"product_id": 6, "annual_premium_budget": 1101}]},
        {"age": 30, "occupation_class": 1, "annual_budget": 1000},
        catalog_map,
    )
    assert any("超出用户年预算" in e for e in over_tol)

    # 省略单项 annual_premium_budget 时由估算器自动补齐校验
    auto_est = verify_plan_draft(
        {"items": [{"product_id": 6}]},
        {"age": 30, "occupation_class": 1, "annual_budget": 200},
        catalog_map,
    )
    assert any("超出用户年预算" in e for e in auto_est)
