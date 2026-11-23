from typing import Any
from uuid import UUID

from pydantic import BaseModel, model_validator


class ChatRequest(BaseModel):
    """基础对话与 HITL 审批恢复请求"""

    thread_id: UUID
    message: str | None = None
    decision: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_message_or_decision(self) -> "ChatRequest":
        cleaned_message = self.message.strip() if isinstance(self.message, str) else None
        self.message = cleaned_message or None
        has_decision = bool(self.decision)
        if not self.message and not has_decision:
            raise ValueError("message 和 decision 不能同时为空")
        return self
