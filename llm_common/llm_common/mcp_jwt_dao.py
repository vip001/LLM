"""MCP JWT 配置（issuer / audience / RSA PEM）持久化到 PostgreSQL，与 ``login_dao`` 同风格。"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from fastmcp.server.auth.providers.jwt import RSAKeyPair
from pydantic import SecretStr
from sqlalchemy import Integer, String, Text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Mapped, mapped_column

from llm_common.postgres_store import Base, SessionLocal

logger = logging.getLogger(__name__)

_MCP_JWT_CONFIG_ID = 1


class McpJwtConfig(Base):
    __tablename__ = "mcp_jwt_config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    issuer: Mapped[str] = mapped_column(String(1024), nullable=False)
    audience: Mapped[str] = mapped_column(String(1024), nullable=False)
    public_key_pem: Mapped[str] = mapped_column(Text, nullable=False)
    private_key_pem: Mapped[str] = mapped_column(Text, nullable=False)


async def get_mcp_jwt_config() -> McpJwtConfig | None:
    try:
        async with SessionLocal() as session:
            return await session.get(McpJwtConfig, _MCP_JWT_CONFIG_ID)
    except SQLAlchemyError:
        logger.exception("Failed to load MCP JWT config from PostgreSQL")
        raise


async def save_mcp_jwt_config(
    *,
    issuer: str,
    audience: str,
    public_key_pem: str,
    private_key_pem: str,
) -> None:
    try:
        async with SessionLocal() as session:
            existing = await session.get(McpJwtConfig, _MCP_JWT_CONFIG_ID)
            if existing:
                existing.issuer = issuer
                existing.audience = audience
                existing.public_key_pem = public_key_pem
                existing.private_key_pem = private_key_pem
            else:
                session.add(
                    McpJwtConfig(
                        id=_MCP_JWT_CONFIG_ID,
                        issuer=issuer,
                        audience=audience,
                        public_key_pem=public_key_pem,
                        private_key_pem=private_key_pem,
                    )
                )
            await session.commit()
    except SQLAlchemyError:
        logger.exception("Failed to save MCP JWT config to PostgreSQL")
        raise


async def mint_mcp_access_token(
    *,
    subject: str,
    expires_in_seconds: int,
) -> tuple[str, datetime]:
    """从库中读取 RSA 与 iss/aud，签发供 MCP Streamable HTTP 使用的 Bearer JWT。"""
    row = await get_mcp_jwt_config()
    if row is None:
        raise ValueError("MCP JWT 尚未初始化，请先启动 MCP 服务完成密钥落库")
    priv = (row.private_key_pem or "").strip()
    pub = (row.public_key_pem or "").strip()
    iss = (row.issuer or "").strip()
    aud = (row.audience or "").strip()
    if not priv or not pub or "PRIVATE KEY" not in priv or "PUBLIC KEY" not in pub:
        raise ValueError("MCP JWT 密钥配置无效")
    if not iss or not aud:
        raise ValueError("MCP JWT issuer/audience 未配置")

    pair = RSAKeyPair(private_key=SecretStr(priv), public_key=pub)
    token = pair.create_token(
        subject=subject,
        issuer=iss,
        audience=aud,
        expires_in_seconds=expires_in_seconds,
    )
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in_seconds)
    return token, expires_at
