import logging
import os
from datetime import datetime, timezone

from sqlalchemy import DateTime, Integer, String, delete, select
from sqlalchemy.engine import URL
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

logger = logging.getLogger(__name__)

PG_USER = os.getenv("PG_USER") or os.getenv("USER", "postgres")
logger.info(f"PG_USER: {PG_USER}")
PG_PASSWORD = os.getenv("PG_PASSWORD", "")
PG_DATABASE = os.getenv("PG_DATABASE", "postgres")
PG_HOST = os.getenv("PG_HOST", "127.0.0.1")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_SOCKET_DIR = os.getenv("PG_SOCKET_DIR", "/tmp/postgresql")
PG_POOL_MIN_CONN = int(os.getenv("PG_POOL_MIN_CONN", "1"))
PG_POOL_MAX_CONN = int(os.getenv("PG_POOL_MAX_CONN", "10"))


class Base(DeclarativeBase):
    pass


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


def _build_connection_url() -> URL:
    if PG_SOCKET_DIR:
        return URL.create(
            drivername="postgresql+asyncpg",
            username=PG_USER,
            password=PG_PASSWORD,
            database=PG_DATABASE,
            query={"host": PG_SOCKET_DIR, "port": str(PG_PORT)},
        )

    return URL.create(
        drivername="postgresql+asyncpg",
        username=PG_USER,
        password=PG_PASSWORD,
        host=PG_HOST,
        port=PG_PORT,
        database=PG_DATABASE,
    )


def create_engine_and_session() -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(
        _build_connection_url(),
        pool_size=PG_POOL_MIN_CONN,
        max_overflow=max(PG_POOL_MAX_CONN - PG_POOL_MIN_CONN, 0),
    )
    return engine, async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)


pg_engine, SessionLocal = create_engine_and_session()


async def init_postgres() -> None:
    try:
        async with pg_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except SQLAlchemyError:
        logger.exception("Failed to initialize PostgreSQL schema")
        raise


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
