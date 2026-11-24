from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from httpx import ASGITransport, AsyncClient

from app.common.auth import create_refresh_token, hash_password
from app.infra.database import get_session
from app.main import app
from app.modules.auth.models import User
from app.modules.insurance_plan.models import Claim, InsurancePlan, InsurancePlanItem, Policy, PremiumQuote
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

    policy_id = uuid4()
    policy = Policy(
        id=policy_id,
        user_id=1,
        product_id=6,
        policy_number="POL-DEMO-2026-0001",
        application_no="APP-DEMO-2026-0001",
        status="active",
        effective_at=now - timedelta(days=30),
        expires_at=now + timedelta(days=335),
        holder_name="张三",
        insured_name="张三",
        insured_id_no_masked="110101********1234",
        insured_phone="138****8888",
        coverage_amount=Decimal("1000000.00"),
        premium_amount=Decimal("318.00"),
        payment_frequency="annual",
        beneficiary={"type": "法定受益人"},
        coverage_snapshot={"category": "medical"},
    )

    claim_id = uuid4()
    claim = Claim(
        id=claim_id,
        user_id=1,
        policy_id=policy_id,
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
        "next_user_id": 2,
        "users": {1: demo_user},
        "products": products,
        "features": features,
        "plans": {plan_id: plan},
        "plan_items": {item_id: plan_item},
        "policies": {policy_id: policy},
        "claims": {claim_id: claim},
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


async def test_auth_products_premium_plans_policies_claims_and_orders_end_to_end(monkeypatch):
    state = _build_inmemory_state()

    async def override_get_session():
        yield _dummy_session()

    app.dependency_overrides[get_session] = override_get_session

    # ---- AuthRepository monkeypatch ----
    from app.modules.auth.repository import AuthRepository

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

    # ---- ProductRepository monkeypatch ----
    from app.modules.product.repository import ProductRepository

    async def fake_prod_by_id(self, pid: int):
        return state["products"].get(pid)

    async def fake_prod_feat(self, pid: int):
        p = state["products"].get(pid)
        return (p, state["features"].get(pid)) if p else None

    async def fake_active_schema(self, pid: int, scenario: str = "premium_by_coverage"):
        return None

    monkeypatch.setattr(ProductRepository, "find_by_id", fake_prod_by_id)
    monkeypatch.setattr(ProductRepository, "find_with_feature_by_id", fake_prod_feat)
    monkeypatch.setattr(ProductRepository, "find_active_pricing_schema", fake_active_schema)

    # ---- InsurancePlanRepository monkeypatch ----
    from app.modules.insurance_plan.repository import InsurancePlanRepository

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

    # ---- PolicyRepository / ClaimRepository / OrderRepository monkeypatch ----
    from app.modules.policy.repository import PolicyRepository
    from app.modules.claim.repository import ClaimRepository
    from app.modules.order.repository import OrderRepository

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
            # 1. 未鉴权拦截验证
            assert (await client.get("/api/v1/policies")).status_code == 401
            assert (await client.get("/api/v1/insurance-plans")).status_code == 401
            assert (await client.get("/api/v1/claims")).status_code == 401
            assert (await client.get("/api/v1/orders")).status_code == 401

            # 2. 注册 & 查重 & 参数校验
            bad_reg = await client.post("/api/v1/auth/register", json={"username": "", "email": "bad", "password": "12"})
            assert bad_reg.status_code == 400
            assert bad_reg.json()["code"] == "INVALID_REGISTER_REQUEST"

            dup_reg = await client.post(
                "/api/v1/auth/register",
                json={"username": "demo", "email": "new@example.com", "password": "secret123"},
            )
            assert dup_reg.status_code == 409
            assert dup_reg.json()["code"] == "USER_EXISTS"

            reg_ok = await client.post(
                "/api/v1/auth/register",
                json={"username": "alice", "email": "alice@example.com", "password": "secret123"},
            )
            assert reg_ok.status_code == 200
            assert reg_ok.json()["user"]["username"] == "alice"

            # 3. 登录 + Refresh Cookie + 退出
            bad_login = await client.post("/api/v1/auth/login", json={"username": "demo", "password": "wrong"})
            assert bad_login.status_code == 400
            assert bad_login.json()["code"] == "INVALID_CREDENTIALS"

            login_ok = await client.post("/api/v1/auth/login", json={"username": "demo", "password": "123456"})
            assert login_ok.status_code == 200
            token = login_ok.json()["access_token"]
            auth_headers = {"Authorization": f"Bearer {token}"}
            assert "refresh_token" in login_ok.cookies

            ref_ok = await client.post("/api/v1/auth/refresh", cookies={"refresh_token": create_refresh_token(1)})
            assert ref_ok.status_code == 200
            assert ref_ok.json()["user"]["id"] == 1

            logout_ok = await client.post("/api/v1/auth/logout")
            assert logout_ok.status_code == 200 and logout_ok.json() == {"ok": True}

            # 4. 产品详情 & 试算模板
            p6 = await client.get("/api/v1/products/6")
            assert p6.status_code == 200 and p6.json()["name"] == "众安尊享e生百万医疗险2026版"
            assert (await client.get("/api/v1/products/999")).status_code == 404

            s6 = await client.get("/api/v1/products/6/pricing-schema")
            assert s6.status_code == 200 and len(s6.json()["schema"]["fields"]) >= 4
            assert (await client.get("/api/v1/products/999/pricing-schema")).status_code == 404

            # 5. 保费试算（真实官方 PDF 表 & ID=1 降级估算 & 资格校验）
            calc_ok = await client.post(
                "/api/v1/premium/calculate",
                headers=auth_headers,
                json={
                    "product_id": 6,
                    "answers": {
                        "plan_code": "standard",
                        "insured.age": 30,
                        "insured.gender": "male",
                        "insured.social_security": "yes",
                    },
                },
            )
            assert calc_ok.status_code == 200
            calc_body = calc_ok.json()
            assert calc_body["pricing_mode"] == "table"
            assert calc_body["annual_premium"] == "318.00"
            quote_id = calc_body["quote_id"]

            calc_est = await client.post(
                "/api/v1/premium/calculate",
                json={"product_id": 1, "answers": {"insured.age": 5, "insured.gender": "female"}},
            )
            assert calc_est.status_code == 200
            assert calc_est.json()["pricing_mode"] == "estimated_formula"

            calc_ineligible = await client.post(
                "/api/v1/premium/calculate",
                json={"product_id": 2, "answers": {"insured.age": 62}},
            )
            assert calc_ineligible.status_code == 400
            assert calc_ineligible.json()["code"] == "INVALID_PRICING_PARAMS"

            # 6. 保险方案列表 / 当前 / 更新保存 / 标记已投保生成 V14 保单快照
            plans_list = await client.get("/api/v1/insurance-plans", headers=auth_headers)
            assert plans_list.status_code == 200 and len(plans_list.json()["items"]) == 1

            cur_plan = await client.get("/api/v1/insurance-plans/current", headers=auth_headers)
            assert cur_plan.status_code == 200

            upd_plan = await client.put(
                "/api/v1/insurance-plans/current",
                headers=auth_headers,
                json={
                    "plan_name": "升级保障方案",
                    "summary": "医疗+意外组合",
                    "budget_note": "年预算参考",
                    "insured_profile": {"name": "张三", "age": 30, "budget": 2000},
                    "items": [
                        {"product_id": 4, "category": "medical", "priority": 1, "annual_premium_budget": 184}
                    ],
                },
            )
            assert upd_plan.status_code == 200
            assert upd_plan.json()["version"] == 2
            new_item_id = upd_plan.json()["items"][0]["id"]
            plan_id = upd_plan.json()["id"]

            insured_res = await client.patch(
                f"/api/v1/insurance-plans/{plan_id}/items/{new_item_id}/insured",
                headers=auth_headers,
            )
            assert insured_res.status_code == 200
            assert insured_res.json()["status"] == "insured"

            # 7. 保单与理赔/理赔指引/PDF申请表下载
            pols = await client.get("/api/v1/policies?status=active", headers=auth_headers)
            assert pols.status_code == 200 and len(pols.json()["items"]) == 2
            first_pol_id = pols.json()["items"][0]["id"]
            assert (await client.get(f"/api/v1/policies/{first_pol_id}", headers=auth_headers)).status_code == 200

            claims = await client.get("/api/v1/claims?status=reviewing", headers=auth_headers)
            assert claims.status_code == 200 and claims.json()["items"][0]["status_message"]

            guide = await client.get(f"/api/v1/claim-guides/{first_pol_id}", headers=auth_headers)
            assert guide.status_code == 200
            assert guide.json()["download_uri"] == "/api/v1/claim-guides/application-form"
            assert len(guide.json()["steps"]) >= 3

            pdf_resp = await client.get("/api/v1/claim-guides/application-form")
            assert pdf_resp.status_code == 200
            assert pdf_resp.headers["content-type"].startswith("application/pdf")
            assert pdf_resp.content.startswith(b"%PDF-")

            # 8. 订单幂等创建 / 查询 / 取消
            ord_res1 = await client.post(
                "/api/v1/orders",
                headers=auth_headers,
                json={"quoteIds": [quote_id], "idempotency_key": "idem-001"},
            )
            ord_res2 = await client.post(
                "/api/v1/orders",
                headers=auth_headers,
                json={"quote_ids": [quote_id], "idempotency_key": "idem-001"},
            )
            assert ord_res1.status_code == 200 and ord_res2.status_code == 200
            assert ord_res1.json()["id"] == ord_res2.json()["id"]
            assert ord_res1.json()["total_premium"] == "318.00"
            order_id = ord_res1.json()["id"]

            ord_list = await client.get("/api/v1/orders", headers=auth_headers)
            assert ord_list.status_code == 200 and len(ord_list.json()["items"]) == 1

            cancel_res = await client.post(f"/api/v1/orders/{order_id}/cancel", headers=auth_headers)
            assert cancel_res.status_code == 200
            assert cancel_res.json()["status"] == "cancelled"
    finally:
        app.dependency_overrides.clear()
