from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.agents.schemas import (
    UserProfile,
    merge_additional_info,
    merge_user_profile,
)
from app.modules.chat.schemas import ChatRequest
from app.modules.chat.service import extract_pending_interrupt, merge_turn_additional_info
from app.modules.product.schemas import ProductResponse


def test_merge_additional_info_returns_new_dict_and_preserves_multi_types():
    left = {"msg-1": {"clause_sources": {"ref-001": {"content": "等待期90天"}}}}
    right = {
        "msg-1": {"candidate_products": [{"id": 4, "name": "星相守2号"}]},
        "msg-2": {"policies": [{"policy_number": "POL-001"}]},
    }
    merged = merge_additional_info(left, right)

    assert merged is not None
    assert "clause_sources" in merged["msg-1"]
    assert "candidate_products" in merged["msg-1"]
    assert merged["msg-2"]["policies"][0]["policy_number"] == "POL-001"


def test_merge_user_profile_incremental_update():
    left = UserProfile(
        age=30,
        target_categories=["medical"],
        preferences=["保证续保"],
    ).model_dump()
    right = {
        "age": None,
        "annual_budget": 4000.0,
        "target_categories": ["medical", "critical_illness"],
        "preferences": ["外购药", "保证续保"],
        "family_role": "  本人 ",
    }

    merged = merge_user_profile(left, right)
    assert merged["age"] == 30
    assert merged["annual_budget"] == 4000.0
    assert merged["target_categories"] == ["medical", "critical_illness"]
    assert merged["preferences"] == ["保证续保", "外购药"]
    assert merged["family_role"] == "本人"


def test_chat_request_accepts_message_or_decision_only():
    tid = uuid4()
    req_msg = ChatRequest(thread_id=tid, message="  推荐重疾险  ")
    assert req_msg.message == "推荐重疾险"

    req_dec = ChatRequest(thread_id=tid, decision={"action": "approve"})
    assert req_dec.decision == {"action": "approve"}

    with pytest.raises(ValidationError):
        ChatRequest(thread_id=tid, message="   ", decision=None)


def test_merge_turn_additional_info_aggregates_without_overwriting():
    turn = {}
    turn = merge_turn_additional_info(turn, {"ref-001": {"clause_name": "a.pdf", "content": "等待期"}})
    turn = merge_turn_additional_info(turn, {"candidate_products": [{"id": 2, "name": "达尔文12号"}]})
    turn = merge_turn_additional_info(turn, {"clause_sources": {"ref-002": {"clause_name": "b.pdf"}}})
    turn = merge_turn_additional_info(turn, {"policies": [{"policy_number": "POL-DEMO-2026-0001"}]})

    assert set(turn["clause_sources"].keys()) == {"ref-001", "ref-002"}
    assert len(turn["candidate_products"]) == 1
    assert len(turn["policies"]) == 1


def test_extract_pending_interrupt_reads_tasks():
    payload = {"plan": {"id": "p1"}, "confirm_info": "请确认保存方案", "action": ["approve", "reject"]}
    snapshot = SimpleNamespace(
        interrupts=(),
        tasks=(SimpleNamespace(interrupts=(SimpleNamespace(value=payload),)),),
    )
    extracted = extract_pending_interrupt(snapshot)
    assert extracted == payload


def test_product_response_allows_nullable_fields():
    resp = ProductResponse(
        id=13,
        name="测试产品",
        clause_name="test.md",
        category="medical",
        insurer="测试保司",
        status="active",
    )
    assert resp.image_url is None
    assert resp.highlights == []
