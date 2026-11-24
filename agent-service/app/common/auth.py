from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from fastapi import Header
from pwdlib import PasswordHash

from app.core.config import settings
from app.core.exceptions import ApplicationError

_PASSWORD_HASHER = PasswordHash.recommended()
_JWT_ALGORITHM = "HS256"


class UnauthorizedError(ApplicationError):
    status_code = 401
    code = "UNAUTHORIZED"
    message = "Invalid or expired token"


def hash_password(password: str) -> str:
    """使用 Argon2 对用户口令进行单向哈希。"""
    return _PASSWORD_HASHER.hash(password)


def verify_password(plain_password: str, password_hash: str) -> bool:
    """校验明文口令与存储哈希是否匹配。"""
    if not plain_password or not password_hash:
        return False
    try:
        return _PASSWORD_HASHER.verify(plain_password, password_hash)
    except Exception:
        return False


def create_access_token(user_id: int) -> str:
    """签发访问令牌（Access Token）。"""
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "type": "access",
        "iat": now,
        "exp": now + timedelta(minutes=settings.app.jwt_access_expire_minutes),
    }
    return jwt.encode(payload, settings.app.jwt_secret_key, algorithm=_JWT_ALGORITHM)


def create_refresh_token(user_id: int) -> str:
    """签发刷新令牌（Refresh Token，用于 Cookie 续期）。"""
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "type": "refresh",
        "iat": now,
        "exp": now + timedelta(days=settings.app.jwt_refresh_expire_days),
    }
    return jwt.encode(payload, settings.app.jwt_secret_key, algorithm=_JWT_ALGORITHM)


def decode_token(token: str, *, expected_type: str) -> int:
    """解析并验证 JWT，返回对应的 user_id。"""
    try:
        payload = jwt.decode(
            token,
            settings.app.jwt_secret_key,
            algorithms=[_JWT_ALGORITHM],
        )
    except jwt.PyJWTError as exc:
        raise UnauthorizedError("Invalid or expired token") from exc

    if payload.get("type") != expected_type:
        raise UnauthorizedError("Invalid token type")
    sub = payload.get("sub")
    if sub is None or not str(sub).isdigit():
        raise UnauthorizedError("Invalid token subject")
    return int(sub)


def get_current_user_id(
    authorization: str | None = Header(default=None),
    x_user_id: int | None = Header(default=None, alias="x-user-id"),
) -> int:
    """统一认证依赖：优先校验 Authorization Bearer JWT，未带 Bearer 时向下兼容 x-user-id。"""
    if authorization is not None:
        scheme, _, token = authorization.strip().partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise UnauthorizedError("Invalid Authorization header")
        return decode_token(token.strip(), expected_type="access")

    if x_user_id is not None and x_user_id > 0:
        return int(x_user_id)

    raise UnauthorizedError("Authentication required")


def get_optional_user_id(
    authorization: str | None = Header(default=None),
    x_user_id: int | None = Header(default=None, alias="x-user-id"),
) -> int | None:
    """可选认证依赖（用于试算等公开但登录后需关联用户的接口）。"""
    if authorization is not None:
        scheme, _, token = authorization.strip().partition(" ")
        if scheme.lower() == "bearer" and token.strip():
            try:
                return decode_token(token.strip(), expected_type="access")
            except UnauthorizedError:
                return None
        return None
    if x_user_id is not None and x_user_id > 0:
        return int(x_user_id)
    return None
