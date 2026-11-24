from sqlalchemy.ext.asyncio import AsyncSession

from app.common.auth import (
    UnauthorizedError,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.modules.auth.exceptions import (
    InactiveUserError,
    InvalidCredentialsError,
    InvalidRefreshTokenError,
    InvalidRegisterRequestError,
    UserExistsError,
    UserNotFoundError,
)
from app.modules.auth.models import User
from app.modules.auth.repository import AuthRepository
from app.modules.auth.schemas import AuthTokenResponse, LoginRequest, RegisterRequest, UserResponse

_DEMO_PLACEHOLDER_HASH = "$2a$12$demo.password.hash.placeholder"


class AuthService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repository = AuthRepository(session)

    async def register(self, req: RegisterRequest) -> tuple[AuthTokenResponse, str]:
        username = (req.username or "").strip()
        email = (req.email or "").strip()
        password = req.password or ""
        if not username or "@" not in email or len(password) < 6:
            raise InvalidRegisterRequestError

        async with self.session.begin():
            if await self.repository.exists_username_or_email(username, email):
                raise UserExistsError
            user = User(
                username=username,
                email=email,
                password_hash=hash_password(password),
                display_name=username,
                status="active",
            )
            await self.repository.add_user(user)

        return self._issue_tokens(user)

    async def login(self, req: LoginRequest) -> tuple[AuthTokenResponse, str]:
        identifier = (req.username or "").strip()
        password = req.password or ""
        if not identifier or not password:
            raise InvalidCredentialsError

        async with self.session.begin():
            user = await self.repository.find_by_username_or_email(identifier)
            if user is None:
                raise InvalidCredentialsError

            # 兼容 init.sql 演示账号首次登录：验证演示口令并升级为 Argon2 哈希
            if user.password_hash == _DEMO_PLACEHOLDER_HASH and password == "123456":
                user.password_hash = hash_password(password)
            elif not verify_password(password, user.password_hash):
                raise InvalidCredentialsError

            if user.status != "active":
                raise InactiveUserError

        return self._issue_tokens(user)

    async def refresh(self, refresh_token: str | None) -> tuple[AuthTokenResponse, str]:
        if not refresh_token:
            raise InvalidRefreshTokenError
        try:
            user_id = decode_token(refresh_token, expected_type="refresh")
        except UnauthorizedError as exc:
            raise InvalidRefreshTokenError from exc

        user = await self.repository.find_by_id(user_id)
        if user is None:
            raise UserNotFoundError
        if user.status != "active":
            raise InactiveUserError

        return self._issue_tokens(user)

    @staticmethod
    def _issue_tokens(user: User) -> tuple[AuthTokenResponse, str]:
        access_token = create_access_token(user.id)
        refresh_token = create_refresh_token(user.id)
        payload = AuthTokenResponse(
            access_token=access_token,
            token_type="Bearer",
            user=UserResponse.model_validate(user),
        )
        return payload, refresh_token
