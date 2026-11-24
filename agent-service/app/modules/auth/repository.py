from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.auth.models import User


class AuthRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def find_by_id(self, user_id: int) -> User | None:
        return await self.session.scalar(select(User).where(User.id == user_id))

    async def find_by_username_or_email(self, identifier: str) -> User | None:
        return await self.session.scalar(
            select(User).where(
                or_(User.username == identifier, User.email == identifier)
            )
        )

    async def exists_username_or_email(self, username: str, email: str) -> bool:
        existing = await self.session.scalar(
            select(User.id).where(
                or_(User.username == username, User.email == email)
            )
        )
        return existing is not None

    async def add_user(self, user: User) -> User:
        self.session.add(user)
        await self.session.flush()
        return user
