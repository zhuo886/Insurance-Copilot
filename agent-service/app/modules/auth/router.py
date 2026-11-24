from fastapi import APIRouter, Cookie, Depends, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.infra.database import get_session
from app.modules.auth.schemas import (
    AuthTokenResponse,
    LoginRequest,
    LogoutResponse,
    RegisterRequest,
)
from app.modules.auth.service import AuthService

router = APIRouter(prefix="/api/v1/auth", tags=["用户认证接口"])

_REFRESH_COOKIE_NAME = "refresh_token"


def get_service(session: AsyncSession = Depends(get_session)) -> AuthService:
    return AuthService(session)


def _set_refresh_cookie(response: Response, refresh_token: str) -> None:
    response.set_cookie(
        key=_REFRESH_COOKIE_NAME,
        value=refresh_token,
        httponly=True,
        samesite="lax",
        path="/",
        max_age=settings.app.jwt_refresh_expire_days * 86400,
    )


@router.post("/register", response_model=AuthTokenResponse)
async def register(
    req: RegisterRequest,
    response: Response,
    service: AuthService = Depends(get_service),
) -> AuthTokenResponse:
    """用户注册"""
    payload, refresh_token = await service.register(req)
    _set_refresh_cookie(response, refresh_token)
    return payload


@router.post("/login", response_model=AuthTokenResponse)
async def login(
    req: LoginRequest,
    response: Response,
    service: AuthService = Depends(get_service),
) -> AuthTokenResponse:
    """用户登录"""
    payload, refresh_token = await service.login(req)
    _set_refresh_cookie(response, refresh_token)
    return payload


@router.post("/refresh", response_model=AuthTokenResponse)
async def refresh(
    response: Response,
    refresh_token: str | None = Cookie(default=None, alias=_REFRESH_COOKIE_NAME),
    service: AuthService = Depends(get_service),
) -> AuthTokenResponse:
    """使用 Refresh Token Cookie 刷新 access token"""
    payload, new_refresh_token = await service.refresh(refresh_token)
    _set_refresh_cookie(response, new_refresh_token)
    return payload


@router.post("/logout", response_model=LogoutResponse)
async def logout(response: Response) -> LogoutResponse:
    """退出登录并清除 Refresh Token Cookie"""
    response.delete_cookie(key=_REFRESH_COOKIE_NAME, path="/")
    return LogoutResponse(ok=True)
