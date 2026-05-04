import importlib
import logging
import os
from collections.abc import Sequence

from sqlalchemy.engine import URL
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

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


async def init_postgres(*, ensure_models: Sequence[str] = ()) -> None:
    """Create ORM tables for models registered on ``Base``.

    Import modules that declare mapped classes on ``Base`` before calling, or pass
    their dotted names in ``ensure_models`` so they are loaded before ``create_all``.
    """
    for name in ensure_models:
        importlib.import_module(name)
    try:
        async with pg_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except SQLAlchemyError:
        logger.exception("Failed to initialize PostgreSQL schema")
        raise
