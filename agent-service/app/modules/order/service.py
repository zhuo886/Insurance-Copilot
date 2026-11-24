from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.order.exceptions import OrderNotFoundError, QuoteNotFoundError
from app.modules.order.models import Order
from app.modules.order.repository import OrderRepository
from app.modules.order.schemas import CreateOrderRequest, OrderListResponse, OrderResponse


class OrderService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repository = OrderRepository(session)

    async def list_orders(self, *, user_id: int) -> OrderListResponse:
        orders = await self.repository.list_owned(user_id)
        return OrderListResponse(
            items=[OrderResponse.model_validate(order) for order in orders]
        )

    async def get_order(self, *, user_id: int, order_id: UUID) -> OrderResponse:
        order = await self.repository.find_owned_by_id(order_id, user_id)
        if order is None:
            raise OrderNotFoundError
        return OrderResponse.model_validate(order)

    async def create_order(
        self,
        *,
        user_id: int,
        request: CreateOrderRequest,
    ) -> OrderResponse:
        if request.idempotency_key:
            existing = await self.repository.find_by_idempotency_key(
                user_id,
                request.idempotency_key,
            )
            if existing is not None:
                return OrderResponse.model_validate(existing)

        all_quote_ids: list[UUID] = list(dict.fromkeys(request.quote_ids))
        for item in request.items or []:
            qid_raw = item.get("quote_id")
            if qid_raw:
                try:
                    qid = UUID(str(qid_raw))
                except ValueError as exc:
                    raise QuoteNotFoundError from exc
                if qid not in all_quote_ids:
                    all_quote_ids.append(qid)

        quotes = await self.repository.find_quotes_by_ids(all_quote_ids)
        quote_map = {q.id: q for q in quotes}
        for qid in all_quote_ids:
            if qid not in quote_map:
                raise QuoteNotFoundError

        total_premium = sum((q.total_premium for q in quotes), Decimal("0.00"))
        if request.items:
            resolved_items: list[dict[str, Any]] = []
            for item in request.items:
                entry = dict(item)
                if "product_id" not in entry and entry.get("quote_id"):
                    q = quote_map.get(UUID(str(entry["quote_id"])))
                    if q is not None:
                        entry["product_id"] = q.product_id
                resolved_items.append(entry)
                if not entry.get("quote_id") and entry.get("premium") is not None:
                    total_premium += Decimal(str(entry["premium"]))
        else:
            resolved_items = [
                {"quote_id": str(q.id), "product_id": q.product_id}
                for q in quotes
            ]

        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "user_id": user_id,
                    "quote_ids": [str(q) for q in all_quote_ids],
                    "items": resolved_items,
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()

        now = datetime.now(timezone.utc)
        order_id = uuid4()
        order_number = f"ORD-{now:%Y%m%d%H%M%S}-{order_id.hex[:8]}"

        async with self.session.begin():
            order = Order(
                id=order_id,
                user_id=user_id,
                thread_id=request.thread_id,
                order_number=order_number,
                confirmation_id=request.confirmation_id,
                idempotency_key=request.idempotency_key,
                idempotency_fingerprint=fingerprint,
                items=resolved_items,
                quote_ids=all_quote_ids,
                total_premium=total_premium,
                recommendation_snapshot=request.recommendation_snapshot,
                status="pending_payment",
            )
            await self.repository.add_order(order)

        return await self.get_order(user_id=user_id, order_id=order_id)

    async def cancel_order(self, *, user_id: int, order_id: UUID) -> OrderResponse:
        async with self.session.begin():
            order = await self.repository.find_owned_by_id(order_id, user_id)
            if order is None:
                raise OrderNotFoundError
            order.status = "cancelled"
        return await self.get_order(user_id=user_id, order_id=order_id)
