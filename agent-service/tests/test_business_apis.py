from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from app.common.auth import create_access_token, create_refresh_token, hash_password
from app.infra.database import get_session
from app.main import app
from app.modules.auth.models import User
from app.modules.insurance_plan.models import (
    Claim,
    InsurancePlan,
    InsurancePlanItem,
    Policy,
    PremiumQuote,
)
from app.modules.order.models import Order
from app.modules.product.catalog_data import load_builtin_catalog
from app.modules.product.models import Product, ProductFeature


def _build_inmemory_state():
    """构造无 Docker/Postgres 依赖的内存业务数据集，覆盖全套 REST API。"""
    now = datetime.now(timezone.utc)
    builtin = {p.id: p for p in load_builtin_catalog()}

    products: dict[int, Product] = {}
    features: dict[int, ProductFeature] = {}
    for pid, cand in builtin.items():
        products[pid] = Product(
            id=cand.id,
            name=cand.name,
            clause_name=cand.clause_name,
            category=cand.category,
            insurer=cand.insurer,
            image_url=cand.image_url,
            description=cand.description,
            min_premium=Decimal(str(cand.min_premium)),
            max_premium=Decimal(str(cand.max_premium)) if cand.max_premium is not None else None,
            target_group=cand.target_group,
            highlights=list(cand.highlights),
            status="active",
        )
        features[pid] = ProductFeature(
            product_id=cand.id,
            min_age=cand.min_age,
            max_age=cand.max_age,
            max_occupation_class=cand.max_occupation_class,
            waiting_period_days=cand.waiting_period_days,
            guarantee_renewal_years=cand.guarantee_renewal_years,
            deductible_default=Decimal(str(cand.deductible_default)),
            has_clause=cand.has_clause,
            is_featured=cand.is_featured,
            coverage_tags=list(cand.coverage_tags),
            health_disclosure_level=cand.health_disclosure_level,
            insurer_tier=cand.insurer_tier,
            base_rate_factor=Decimal(str(cand.base_rate_factor)),
        )

    demo_user = User(
        id=1,
        username="demo",
        email="demo@example.com",
        password_hash=hash_password("123456"),
        display_name="演示用户",
        status="active",
    )
    other_user = User(
        id=2,
        username="other_user",
        email="other@example.com",
        password_hash=hash_password("654321"),
        display_name="其他用户",
        status="active",
    )

    plan_id = uuid4()
    item_id = uuid4()
    plan = InsurancePlan(
        id=plan_id,
        user_id=1,
        plan_name="基础保障方案",
        summary="医疗+意外",
        budget_note="年缴参考",
        insured_profile={"name": "张三", "age": 30},
        annual_premium_budget=Decimal("1000.00"),
        version=1,
        status="uninsured",
        created_at=now,
        updated_at=now,
    )
    plan_item = InsurancePlanItem(
        id=item_id,
        plan_id=plan_id,
        product_id=6,
        priority=1,
        recommendation_reason="百万医疗主力产品",
        annual_premium_budget=Decimal("318.00"),
        status="uninsured",
        created_at=now,
        updated_at=now,
    )

    # 覆盖医疗 / 重疾 / 意外 / 寿险四类产品的已生效保单
    pol_medical_id = uuid4()
    pol_ci_id = uuid4()
    pol_acc_id = uuid4()
    pol_life_id = uuid4()

    def _make_pol(pid_uuid, product_id: int, num: str, cov: str, prem: str) -> Policy:
        return Policy(
            id=pid_uuid,
            user_id=1,
            product_id=product_id,
            policy_number=num,
            application_no=f"APP-{num}",
            status="active",
            effective_at=now - timedelta(days=30),
            expires_at=now + timedelta(days=335),
            holder_name="张三",
            insured_name="张三",
            insured_id_no_masked="110101********1234",
            insured_phone="138****8888",
            coverage_amount=Decimal(cov),
            premium_amount=Decimal(prem),
            payment_frequency="annual",
            beneficiary={"type": "法定受益人"},
            coverage_snapshot={"category": products[product_id].category},
        )

    policies = {
        pol_medical_id: _make_pol(pol_medical_id, 6, "POL-MED-001", "1000000.00", "318.00"),
        pol_ci_id: _make_pol(pol_ci_id, 2, "POL-CI-002", "500000.00", "1824.00"),
        pol_acc_id: _make_pol(pol_acc_id, 10, "POL-ACC-003", "1000000.00", "136.00"),
        pol_life_id: _make_pol(pol_life_id, 7, "POL-LIFE-004", "500000.00", "10000.00"),
    }

    claim_id = uuid4()
    claim = Claim(
        id=claim_id,
        user_id=1,
        policy_id=pol_medical_id,
        product_id=6,
        claim_number="CLM-DEMO-001",
        claim_type="medical",
        amount=Decimal("800.00"),
        description="住院医疗费报销",
        status="reviewing",
        submitted_at=now - timedelta(days=2),
        updated_at=now,
    )

    return {
        "next_user_id": 3,
        "users": {1: demo_user, 2: other_user},
        "products": products,
        "features": features,
        "plans": {plan_id: plan},
        "plan_items": {item_id: plan_item},
        "policies": policies,
        "policy_ids": {
            "medical": pol_medical_id,
            "critical_illness": pol_ci_id,
            "accident": pol_acc_id,
            "life": pol_life_id,
        },
        "claims": {claim_id: claim},
        "claim_id": claim_id,
        "plan_id": plan_id,
        "item_id": item_id,
        "quotes": {},
        "orders": {},
        "now": now,
    }


def _dummy_session():
    class _Tx:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    return SimpleNamespace(
        begin=lambda: _Tx(),
        add=lambda _obj: None,
        add_all=lambda _objs: None,
        flush=AsyncMock(),
    )


@pytest.fixture
async def api_env(monkeypatch):
    state = _build_inmemory_state()

    async def override_get_session():
        yield _dummy_session()

    app.dependency_overrides[get_session] = override_get_session

    from app.modules.auth.repository import AuthRepository
    from app.modules.claim.repository import ClaimRepository
    from app.modules.insurance_plan.repository import InsurancePlanRepository
    from app.modules.order.repository import OrderRepository
    from app.modules.policy.repository import PolicyRepository
    from app.modules.product.repository import ProductRepository

    async def fake_find_user_id(self, uid: int):
        return state["users"].get(uid)

    async def fake_find_user_login(self, identifier: str):
        return next(
            (u for u in state["users"].values() if u.username == identifier or u.email == identifier),
            None,
        )

    async def fake_exists_user(self, username: str, email: str):
        return any(u.username == username or u.email == email for u in state["users"].values())

    async def fake_add_user(self, user: User):
        user.id = state["next_user_id"]
        state["next_user_id"] += 1
        state["users"][user.id] = user
        return user

    monkeypatch.setattr(AuthRepository, "find_by_id", fake_find_user_id)
    monkeypatch.setattr(AuthRepository, "find_by_username_or_email", fake_find_user_login)
    monkeypatch.setattr(AuthRepository, "exists_username_or_email", fake_exists_user)
    monkeypatch.setattr(AuthRepository, "add_user", fake_add_user)

    async def fake_prod_by_id(self, pid: int):
        return state["products"].get(pid)

    async def fake_prod_feat(self, pid: int):
        p = state["products"].get(pid)
        return (p, state["features"].get(pid)) if p else None

    async def fake_active_schema(self, pid: int, scenario: str = "premium_by_coverage"):
        return None

    async def fake_prod_by_cat(self, category: str | None = None, featured_only: bool = True):
        items = [p for p in state["products"].values() if p.status == "active"]
        if category:
            items = [p for p in items if p.category == category]
        if featured_only:
            items = [p for p in items if state["features"][p.id].is_featured]
        return sorted(items, key=lambda p: p.id)

    async def fake_prod_limited(self, category: str, premium_min: Decimal | None, limit: int):
        items = [p for p in state["products"].values() if p.category == category and p.status == "active"]
        if premium_min is not None:
            items = [p for p in items if p.min_premium is not None and p.min_premium < premium_min]
        return sorted(items, key=lambda p: (p.min_premium or Decimal("0"), p.id))[:limit]

    monkeypatch.setattr(ProductRepository, "find_by_id", fake_prod_by_id)
    monkeypatch.setattr(ProductRepository, "find_with_feature_by_id", fake_prod_feat)
    monkeypatch.setattr(ProductRepository, "find_active_pricing_schema", fake_active_schema)
    monkeypatch.setattr(ProductRepository, "find_by_category", fake_prod_by_cat)
    monkeypatch.setattr(ProductRepository, "find_limited_by_category", fake_prod_limited)

    async def fake_list_plans(self, uid: int):
        return [p for p in state["plans"].values() if p.user_id == uid]

    async def fake_current_plan(self, uid: int):
        cands = [p for p in state["plans"].values() if p.user_id == uid and p.status != "insured"]
        return cands[0] if cands else None

    async def fake_owned_plan(self, pid, uid: int):
        p = state["plans"].get(pid)
        return p if p and p.user_id == uid else None

    async def fake_items_with_prods(self, pid):
        return [
            (it, state["products"][it.product_id])
            for it in state["plan_items"].values()
            if it.plan_id == pid
        ]

    async def fake_find_item(self, pid, iid):
        it = state["plan_items"].get(iid)
        return it if it and it.plan_id == pid else None

    async def fake_find_product(self, pid: int):
        return state["products"].get(pid)

    async def fake_list_items(self, pid):
        return [it for it in state["plan_items"].values() if it.plan_id == pid]

    async def fake_delete_items(self, pid):
        for k in [k for k, v in state["plan_items"].items() if v.plan_id == pid]:
            del state["plan_items"][k]

    async def fake_add_plan(self, plan: InsurancePlan):
        now = datetime.now(timezone.utc)
        plan.created_at = plan.created_at or now
        plan.updated_at = plan.updated_at or now
        state["plans"][plan.id] = plan

    async def fake_add_items(self, items: list[InsurancePlanItem]):
        now = datetime.now(timezone.utc)
        for it in items:
            it.created_at = it.created_at or now
            it.updated_at = it.updated_at or now
            state["plan_items"][it.id] = it

    async def fake_add_policy(self, policy: Policy):
        state["policies"][policy.id] = policy

    async def fake_add_quote(self, quote: PremiumQuote):
        state["quotes"][quote.id] = quote

    async def fake_user_name(self, uid: int):
        u = state["users"].get(uid)
        return u.display_name or u.username if u else "演示用户"

    async def fake_rate_tables(self, pids=None):
        return {}

    monkeypatch.setattr(InsurancePlanRepository, "list_owned_plans", fake_list_plans)
    monkeypatch.setattr(InsurancePlanRepository, "find_current_plan", fake_current_plan)
    monkeypatch.setattr(InsurancePlanRepository, "find_owned_plan", fake_owned_plan)
    monkeypatch.setattr(InsurancePlanRepository, "list_items_with_products", fake_items_with_prods)
    monkeypatch.setattr(InsurancePlanRepository, "find_item", fake_find_item)
    monkeypatch.setattr(InsurancePlanRepository, "find_product", fake_find_product)
    monkeypatch.setattr(InsurancePlanRepository, "list_plan_items", fake_list_items)
    monkeypatch.setattr(InsurancePlanRepository, "delete_plan_items", fake_delete_items)
    monkeypatch.setattr(InsurancePlanRepository, "add_plan", fake_add_plan)
    monkeypatch.setattr(InsurancePlanRepository, "add_items", fake_add_items)
    monkeypatch.setattr(InsurancePlanRepository, "add_policy", fake_add_policy)
    monkeypatch.setattr(InsurancePlanRepository, "add_quote", fake_add_quote)
    monkeypatch.setattr(InsurancePlanRepository, "get_user_display_name", fake_user_name)
    monkeypatch.setattr(InsurancePlanRepository, "find_rate_table_items", fake_rate_tables)

    async def fake_list_policies(self, uid: int, status: str | None = "active"):
        return [
            (pol, state["products"][pol.product_id])
            for pol in state["policies"].values()
            if pol.user_id == uid and (not status or status == "all" or pol.status == status)
        ]

    async def fake_find_policy(self, pol_id, uid: int):
        pol = state["policies"].get(pol_id)
        return (pol, state["products"][pol.product_id]) if pol and pol.user_id == uid else None

    async def fake_list_claims(self, uid: int, *, policy_id=None, status=None):
        res = []
        for cl in state["claims"].values():
            if cl.user_id != uid:
                continue
            if policy_id and cl.policy_id != policy_id:
                continue
            if status and cl.status != status:
                continue
            res.append((cl, state["policies"].get(cl.policy_id)))
        return res

    async def fake_find_claim(self, cid, uid: int):
        cl = state["claims"].get(cid)
        return (cl, state["policies"].get(cl.policy_id)) if cl and cl.user_id == uid else None

    async def fake_list_orders(self, uid: int):
        return [o for o in state["orders"].values() if o.user_id == uid]

    async def fake_find_order(self, oid, uid: int):
        o = state["orders"].get(oid)
        return o if o and o.user_id == uid else None

    async def fake_find_order_idem(self, uid: int, key: str):
        return next(
            (o for o in state["orders"].values() if o.user_id == uid and o.idempotency_key == key),
            None,
        )

    async def fake_find_quotes(self, qids):
        return [state["quotes"][q] for q in qids if q in state["quotes"]]

    async def fake_add_order(self, order: Order):
        now = datetime.now(timezone.utc)
        order.created_at = order.created_at or now
        order.updated_at = order.updated_at or now
        state["orders"][order.id] = order
        return order

    monkeypatch.setattr(PolicyRepository, "list_owned", fake_list_policies)
    monkeypatch.setattr(PolicyRepository, "find_owned_by_id", fake_find_policy)
    monkeypatch.setattr(ClaimRepository, "list_owned", fake_list_claims)
    monkeypatch.setattr(ClaimRepository, "find_owned_by_id", fake_find_claim)
    monkeypatch.setattr(OrderRepository, "list_owned", fake_list_orders)
    monkeypatch.setattr(OrderRepository, "find_owned_by_id", fake_find_order)
    monkeypatch.setattr(OrderRepository, "find_by_idempotency_key", fake_find_order_idem)
    monkeypatch.setattr(OrderRepository, "find_quotes_by_ids", fake_find_quotes)
    monkeypatch.setattr(OrderRepository, "add_order", fake_add_order)

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            yield client, state
    finally:
        app.dependency_overrides.clear()


def _auth_headers(user_id: int = 1) -> dict[str, str]:
    return {"Authorization": f"Bearer {create_access_token(user_id)}"}


async def test_health_check_endpoint(api_env):
    client, _ = api_env
    resp = await client.get("/healthy")
    assert resp.status_code == 200 and resp.json() == {"status": "ok"}


@pytest.mark.parametrize(
    "path",
    ["/api/v1/policies", "/api/v1/insurance-plans", "/api/v1/claims", "/api/v1/orders", "/api/v1/chat-threads"],
)
async def test_protected_endpoints_require_authentication(api_env, path: str):
    client, _ = api_env
    assert (await client.get(path)).status_code == 401


async def test_auth_register_validation_duplicate_and_success(api_env):
    client, _ = api_env
    bad_reg = await client.post("/api/v1/auth/register", json={"username": "", "email": "bad", "password": "12"})
    assert bad_reg.status_code == 400 and bad_reg.json()["code"] == "INVALID_REGISTER_REQUEST"

    dup_reg = await client.post(
        "/api/v1/auth/register",
        json={"username": "demo", "email": "new@example.com", "password": "secret123"},
    )
    assert dup_reg.status_code == 409 and dup_reg.json()["code"] == "USER_EXISTS"

    reg_ok = await client.post(
        "/api/v1/auth/register",
        json={"username": "alice", "email": "alice@example.com", "password": "secret123"},
    )
    assert reg_ok.status_code == 200 and reg_ok.json()["user"]["username"] == "alice"


async def test_auth_login_refresh_and_logout_lifecycle(api_env):
    client, _ = api_env
    bad_login = await client.post("/api/v1/auth/login", json={"username": "demo", "password": "wrong"})
    assert bad_login.status_code == 400 and bad_login.json()["code"] == "INVALID_CREDENTIALS"

    login_ok = await client.post("/api/v1/auth/login", json={"username": "demo", "password": "123456"})
    assert login_ok.status_code == 200
    assert "refresh_token" in login_ok.cookies

    # 清空 login_ok 写入 client cookie jar 的 refresh_token，验证无 Cookie 刷新返回 401
    client.cookies.clear()
    no_cookie_ref = await client.post("/api/v1/auth/refresh")
    assert no_cookie_ref.status_code == 401 and no_cookie_ref.json()["code"] == "INVALID_REFRESH_TOKEN"

    ref_ok = await client.post("/api/v1/auth/refresh", cookies={"refresh_token": create_refresh_token(1)})
    assert ref_ok.status_code == 200 and ref_ok.json()["user"]["id"] == 1

    logout_ok = await client.post("/api/v1/auth/logout")
    assert logout_ok.status_code == 200 and logout_ok.json() == {"ok": True}


async def test_auth_inactive_user_blocked_on_login_and_refresh(api_env):
    client, state = api_env
    state["users"][2].status = "disabled"

    bad_login = await client.post("/api/v1/auth/login", json={"username": "other_user", "password": "654321"})
    assert bad_login.status_code == 400 and bad_login.json()["code"] == "INACTIVE_USER"

    bad_refresh = await client.post("/api/v1/auth/refresh", cookies={"refresh_token": create_refresh_token(2)})
    assert bad_refresh.status_code == 400 and bad_refresh.json()["code"] == "INACTIVE_USER"


async def test_products_list_candidates_detail_and_pricing_schema(api_env):
    client, _ = api_env
    all_prods = await client.get("/api/v1/products")
    assert all_prods.status_code == 200 and len(all_prods.json()) == 12

    med_prods = await client.get("/api/v1/products?category=medical")
    assert med_prods.status_code == 200 and all(p["category"] == "medical" for p in med_prods.json())

    cands = await client.get("/api/v1/products/candidates?categories=medical&categories=accident&premium_min=500&limit_per_category=2")
    assert cands.status_code == 200 and len(cands.json()) >= 2

    p6 = await client.get("/api/v1/products/6")
    assert p6.status_code == 200 and p6.json()["name"] == "众安尊享e生百万医疗险2026版"
    assert (await client.get("/api/v1/products/999")).status_code == 404

    s6 = await client.get("/api/v1/products/6/pricing-schema")
    assert s6.status_code == 200 and len(s6.json()["schema"]["fields"]) >= 4
    assert (await client.get("/api/v1/products/999/pricing-schema")).status_code == 404


async def test_premium_calculate_table_estimated_and_validation_errors(api_env):
    client, _ = api_env
    calc_ok = await client.post(
        "/api/v1/premium/calculate",
        headers=_auth_headers(1),
        json={
            "product_id": 6,
            "answers": {"plan_code": "standard", "insured.age": 30, "insured.gender": "male", "insured.social_security": "yes"},
        },
    )
    assert calc_ok.status_code == 200
    assert calc_ok.json()["pricing_mode"] == "table" and calc_ok.json()["annual_premium"] == "318.00"

    calc_est = await client.post(
        "/api/v1/premium/calculate",
        json={"product_id": 1, "answers": {"insured.age": 5, "insured.gender": "female"}},
    )
    assert calc_est.status_code == 200 and calc_est.json()["pricing_mode"] == "estimated_formula"

    calc_ineligible = await client.post(
        "/api/v1/premium/calculate",
        json={"product_id": 2, "answers": {"insured.age": 62}},
    )
    assert calc_ineligible.status_code == 400 and calc_ineligible.json()["code"] == "INVALID_PRICING_PARAMS"

    missing_age = await client.post("/api/v1/premium/calculate", json={"product_id": 6, "answers": {}})
    assert missing_age.status_code == 400

    missing_prod = await client.post("/api/v1/premium/calculate", json={"product_id": 999, "answers": {"insured.age": 30}})
    assert missing_prod.status_code == 404


async def test_insurance_plan_crud_and_mark_insured_creates_policy(api_env):
    client, state = api_env
    headers = _auth_headers(1)

    assert (await client.get("/api/v1/insurance-plans", headers=headers)).status_code == 200
    assert (await client.get("/api/v1/insurance-plans/current", headers=headers)).status_code == 200
    assert (await client.get(f"/api/v1/insurance-plans/{state['plan_id']}", headers=headers)).status_code == 200

    upd_plan = await client.put(
        "/api/v1/insurance-plans/current",
        headers=headers,
        json={
            "plan_name": "升级保障方案",
            "summary": "医疗+意外组合",
            "budget_note": "年预算参考",
            "insured_profile": {"name": "张三", "age": 30, "budget": 2000},
            "items": [{"product_id": 4, "category": "medical", "priority": 1, "annual_premium_budget": 184}],
        },
    )
    assert upd_plan.status_code == 200 and upd_plan.json()["version"] == 2
    plan_id = upd_plan.json()["id"]
    item_id = upd_plan.json()["items"][0]["id"]

    insured_res = await client.patch(f"/api/v1/insurance-plans/{plan_id}/items/{item_id}/insured", headers=headers)
    assert insured_res.status_code == 200 and insured_res.json()["status"] == "insured"
    assert len(state["policies"]) == 5


async def test_policies_claims_and_four_category_claim_guides(api_env):
    client, state = api_env
    headers = _auth_headers(1)

    pols = await client.get("/api/v1/policies?status=active", headers=headers)
    assert pols.status_code == 200 and len(pols.json()["items"]) == 4

    med_pol_id = state["policy_ids"]["medical"]
    detail = await client.get(f"/api/v1/policies/{med_pol_id}", headers=headers)
    assert detail.status_code == 200 and detail.json()["policy_number"] == "POL-MED-001"

    claims = await client.get("/api/v1/claims?status=reviewing", headers=headers)
    assert claims.status_code == 200 and claims.json()["items"][0]["status_message"]
    claim_detail = await client.get(f"/api/v1/claims/{state['claim_id']}", headers=headers)
    assert claim_detail.status_code == 200 and claim_detail.json()["claim_number"] == "CLM-DEMO-001"

    # 四大险种各自理赔指南材料清单不同且均可生成
    for pid_uuid in state["policy_ids"].values():
        guide = await client.get(f"/api/v1/claim-guides/{pid_uuid}", headers=headers)
        assert guide.status_code == 200
        assert guide.json()["download_uri"] == "/api/v1/claim-guides/application-form"
        assert len(guide.json()["materials"]) >= 3

    pdf_resp = await client.get("/api/v1/claim-guides/application-form")
    assert pdf_resp.status_code == 200
    assert pdf_resp.headers["content-type"].startswith("application/pdf")
    assert pdf_resp.content.startswith(b"%PDF-")


async def test_orders_idempotency_invalid_quote_and_cancellation(api_env):
    client, _ = api_env
    headers = _auth_headers(1)

    calc = await client.post(
        "/api/v1/premium/calculate",
        headers=headers,
        json={"product_id": 6, "answers": {"insured.age": 30, "insured.social_security": "yes"}},
    )
    quote_id = calc.json()["quote_id"]

    bad_quote = await client.post(
        "/api/v1/orders",
        headers=headers,
        json={"quote_ids": [str(uuid4())], "idempotency_key": "bad-quote"},
    )
    assert bad_quote.status_code == 404 and bad_quote.json()["code"] == "QUOTE_NOT_FOUND"

    ord1 = await client.post(
        "/api/v1/orders",
        headers=headers,
        json={"quoteIds": [quote_id], "idempotency_key": "idem-001"},
    )
    ord2 = await client.post(
        "/api/v1/orders",
        headers=headers,
        json={"quote_ids": [quote_id], "idempotency_key": "idem-001"},
    )
    assert ord1.status_code == 200 and ord2.status_code == 200
    assert ord1.json()["id"] == ord2.json()["id"] and ord1.json()["total_premium"] == "318.00"
    order_id = ord1.json()["id"]

    assert (await client.get(f"/api/v1/orders/{order_id}", headers=headers)).status_code == 200
    assert len((await client.get("/api/v1/orders", headers=headers)).json()["items"]) == 1

    cancel_res = await client.post(f"/api/v1/orders/{order_id}/cancel", headers=headers)
    assert cancel_res.status_code == 200 and cancel_res.json()["status"] == "cancelled"


async def test_cross_user_resource_isolation_returns_404(api_env):
    client, state = api_env
    user1_headers = _auth_headers(1)
    user2_headers = _auth_headers(2)

    calc = await client.post(
        "/api/v1/premium/calculate",
        headers=user1_headers,
        json={"product_id": 6, "answers": {"insured.age": 30}},
    )
    order = await client.post(
        "/api/v1/orders",
        headers=user1_headers,
        json={"quote_ids": [calc.json()["quote_id"]], "idempotency_key": "u1-only"},
    )
    order_id = order.json()["id"]

    assert (await client.get(f"/api/v1/insurance-plans/{state['plan_id']}", headers=user2_headers)).status_code == 404
    assert (await client.get(f"/api/v1/policies/{state['policy_ids']['medical']}", headers=user2_headers)).status_code == 404
    assert (await client.get(f"/api/v1/claims/{state['claim_id']}", headers=user2_headers)).status_code == 404
    assert (await client.get(f"/api/v1/claim-guides/{state['policy_ids']['medical']}", headers=user2_headers)).status_code == 404
    assert (await client.get(f"/api/v1/orders/{order_id}", headers=user2_headers)).status_code == 404
    assert (await client.post(f"/api/v1/orders/{order_id}/cancel", headers=user2_headers)).status_code == 404
