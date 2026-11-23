from typing import Annotated, Any, Literal, NotRequired

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

# 路由意图常量
Intent = Literal[
    "chitchat",
    "recommendation_plan",
    "claim",
    "human_handoff",
    "fallback",
]

AdditionalInfo = dict[str, Any]


class UserProfile(BaseModel):
    """结构化保险用户画像（跨轮增量维护）"""

    age: int | None = Field(default=None, ge=0, le=120, description="被保人年龄")
    gender: Literal["male", "female"] | None = Field(default=None, description="被保人性别")
    occupation_class: int | None = Field(default=None, ge=1, le=6, description="职业类别 1-6")
    has_social_security: bool | None = Field(default=None, description="是否有医保/社保")
    annual_budget: float | None = Field(default=None, ge=0, description="年缴保费预算（元）")
    target_categories: list[str] = Field(
        default_factory=list,
        description="关注险种列表：medical/critical_illness/accident/life",
    )
    family_role: str | None = Field(default=None, description="家庭角色，如本人/少儿/父母/配偶")
    preferences: list[str] = Field(
        default_factory=list,
        description="保障偏好标签，如保证续保、外购药、0免赔、特需医疗、猝死保障等",
    )


class RouteResult(BaseModel):
    """意图识别结果"""

    intent: Intent = Field(description="本轮消息应该进入的工作流")
    reason: str = Field(description="判断该意图的理由")


def merge_user_profile(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
) -> dict[str, Any]:
    """跨轮增量合并用户画像：新值中的有效字段覆盖旧值，列表字段去重累加。"""
    merged: dict[str, Any] = dict(left or {})
    if not right:
        return merged

    for key, value in right.items():
        if value is None:
            continue
        if isinstance(value, list):
            existing = merged.get(key)
            base_list = list(existing) if isinstance(existing, list) else []
            for item in value:
                if item is not None and item not in base_list:
                    base_list.append(item)
            merged[key] = base_list
        elif isinstance(value, str):
            stripped = value.strip()
            if stripped:
                merged[key] = stripped
        else:
            merged[key] = value
    return merged


def _merge_message_payload(
    existing: dict[str, Any],
    incoming: dict[str, Any],
) -> dict[str, Any]:
    """合并同一条 AI 消息下的多类型附加信息，防止后者覆盖前者。"""
    result = dict(existing)
    for key, value in incoming.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = {**result[key], **value}
        elif key in result and isinstance(result[key], list) and isinstance(value, list):
            result[key] = [*result[key], *value]
        else:
            result[key] = value
    return result


def merge_additional_info(
    left: dict[str, AdditionalInfo] | None,
    right: dict[str, AdditionalInfo] | None,
) -> dict[str, AdditionalInfo]:
    """按 AI 消息 ID 合并引用与卡片信息，返回新字典避免就地 update 返回 None。"""
    merged: dict[str, AdditionalInfo] = dict(left or {})
    if not right:
        return merged

    for message_id, payload in right.items():
        if not isinstance(payload, dict):
            continue
        if message_id in merged and isinstance(merged[message_id], dict):
            merged[message_id] = _merge_message_payload(merged[message_id], payload)
        else:
            merged[message_id] = dict(payload)
    return merged


class InsuranceAgentState(MessagesState):
    """保险顾问主 Graph 状态"""

    previous_workflow: NotRequired[Intent]
    active_workflow: NotRequired[Intent]
    user_profile: Annotated[dict[str, Any], merge_user_profile]
    additional_info: Annotated[dict[str, AdditionalInfo], merge_additional_info]
