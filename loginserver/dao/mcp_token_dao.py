import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, Integer, String, Text, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Mapped, mapped_column

from llm_common.postgres_store import Base, SessionLocal

logger = logging.getLogger(__name__)


class McpUserToken(Base):
    __tablename__ = "mcp_user_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    token: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    config_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


async def save_mcp_user_token(
    *,
    email: str,
    token: str,
    expires_at: datetime,
    config_json: str,
) -> None:
    normalized = expires_at
    if normalized.tzinfo is None:
        normalized = normalized.replace(tzinfo=timezone.utc)

    try:
        async with SessionLocal() as session:
            existing = await session.scalar(
                select(McpUserToken).where(McpUserToken.email == email.lower())
            )
            if existing:
                existing.token = token
                existing.expires_at = normalized
                existing.config_json = config_json
            else:
                session.add(
                    McpUserToken(
                        email=email.lower(),
                        token=token,
                        expires_at=normalized,
                        config_json=config_json,
                    )
                )
            await session.commit()
    except SQLAlchemyError:
        logger.exception("Failed to save MCP user token to PostgreSQL")
        raise


async def get_mcp_user_token_by_email(email: str) -> Optional[McpUserToken]:
    try:
        async with SessionLocal() as session:
            return await session.scalar(
                select(McpUserToken).where(McpUserToken.email == email.lower())
            )
    except SQLAlchemyError:
        logger.exception("Failed to query MCP user token")
        raise
