import logging
from datetime import datetime, timezone

from sqlalchemy import DateTime, Integer, String, delete, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Mapped, mapped_column

from llm_common.postgres_store import Base, SessionLocal

logger = logging.getLogger(__name__)


class LoginSession(Base):
    __tablename__ = "login_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    token: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    email: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


async def save_login_session(email: str, token: str, expires_at: datetime) -> None:
    normalized_expire_at = expires_at
    if normalized_expire_at.tzinfo is None:
        normalized_expire_at = normalized_expire_at.replace(tzinfo=timezone.utc)

    try:
        async with SessionLocal() as session:
            existing = await session.scalar(
                select(LoginSession).where(LoginSession.email == email)
            )
            if existing:
                existing.token = token
                existing.expires_at = normalized_expire_at
            else:
                session.add(
                    LoginSession(
                        token=token,
                        email=email,
                        expires_at=normalized_expire_at,
                    )
                )
            await session.commit()
    except SQLAlchemyError:
        logger.exception("Failed to save login session to PostgreSQL")
        raise


async def get_login_session_by_token(token: str) -> LoginSession | None:
    try:
        async with SessionLocal() as session:
            return await session.scalar(
                select(LoginSession).where(LoginSession.token == token)
            )
    except SQLAlchemyError:
        logger.exception("Failed to query login session by token")
        raise


async def delete_login_session_by_token(token: str) -> None:
    try:
        async with SessionLocal() as session:
            await session.execute(delete(LoginSession).where(LoginSession.token == token))
            await session.commit()
    except SQLAlchemyError:
        logger.exception("Failed to delete login session by token")
        raise
