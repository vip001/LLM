"""MCP 访问令牌：/auth/mcp-token 签发、查询与登录态中间件。"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Callable, Optional

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

try:
    from llm_common.mcp_jwt_dao import mint_mcp_access_token
    from .dao.mcp_token_dao import get_mcp_user_token_by_email, save_mcp_user_token
    from .dao.redis_store import jwt_blacklist_key, mcp_token_cache_key, redis_client, session_key
    from .jwt_api import authorization_looks_like_jwt, decode_jwt_token
except ImportError:
    from llm_common.mcp_jwt_dao import mint_mcp_access_token
    from dao.mcp_token_dao import get_mcp_user_token_by_email, save_mcp_user_token
    from dao.redis_store import jwt_blacklist_key, mcp_token_cache_key, redis_client, session_key
    from jwt_api import authorization_looks_like_jwt, decode_jwt_token

logger = logging.getLogger(__name__)

MCP_ACCESS_TOKEN_EXPIRE_SECONDS = 7 * 24 * 60 * 60
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8001/mcp")


class CreateMcpTokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_at: str
    config: str


def _mcp_token_redis_ttl_seconds(expires_at: datetime) -> int:
    now = datetime.now(timezone.utc)
    exp = expires_at if expires_at.tzinfo else expires_at.replace(tzinfo=timezone.utc)
    remaining = int((exp - now).total_seconds())
    return remaining if remaining > 0 else 0


async def _store_mcp_token_in_redis(
    email: str,
    *,
    access_token: str,
    expires_at: datetime,
    config_formatted: str,
) -> None:
    ttl = _mcp_token_redis_ttl_seconds(expires_at)
    if ttl <= 0:
        return
    expires_at_str = expires_at.isoformat()
    payload = json.dumps(
        {
            "access_token": access_token,
            "expires_at": expires_at_str,
            "config": config_formatted,
        },
        ensure_ascii=False,
    )
    await redis_client.setex(mcp_token_cache_key(email), ttl, payload)


def _response_from_cached_mcp_json(data: dict) -> CreateMcpTokenResponse:
    return CreateMcpTokenResponse(
        access_token=data["access_token"],
        token_type="Bearer",
        expires_at=data["expires_at"],
        config=data["config"],
    )


async def _resolve_authenticated_email_from_header(
    authorization: Optional[str],
    *,
    extract_bearer_token: Callable[[Optional[str]], str],
) -> str:
    token = extract_bearer_token(authorization)
    if authorization_looks_like_jwt(token):
        if await redis_client.get(jwt_blacklist_key(token)):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="JWT 已退出登录，请重新登录",
            )
        payload = decode_jwt_token(token)
        return str(payload["sub"])
    email = await redis_client.get(session_key(token))
    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="登录已失效，请重新登录",
        )
    return email


def make_mcp_token_login_middleware(
    extract_bearer_token: Callable[[Optional[str]], str],
) -> type[BaseHTTPMiddleware]:
    """在业务处理前校验 MCP token 路由的登录态（POST / GET），结果写入 ``request.state``。"""

    class _McpTokenLoginMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            if request.url.path == "/auth/mcp-token" and request.method in ("POST", "GET"):
                auth = request.headers.get("Authorization")
                email = await _resolve_authenticated_email_from_header(
                    auth,
                    extract_bearer_token=extract_bearer_token,
                )
                request.state.mcp_token_user_email = email
            return await call_next(request)

    return _McpTokenLoginMiddleware


def create_mcp_token_router(
    *,
    mcp_access_token_expire_seconds: int = MCP_ACCESS_TOKEN_EXPIRE_SECONDS,
    mcp_server_url: str = MCP_SERVER_URL,
) -> APIRouter:
    router = APIRouter()

    @router.post("/auth/mcp-token", response_model=CreateMcpTokenResponse)
    async def create_mcp_token(request: Request) -> CreateMcpTokenResponse:
        email = getattr(request.state, "mcp_token_user_email", None)
        if not isinstance(email, str) or not email:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="MCP token 路由未经过登录校验中间件",
            )
        try:
            access_token, expires_at = await mint_mcp_access_token(
                subject=email,
                expires_in_seconds=mcp_access_token_expire_seconds,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        mcp_client_config = {
            "transport": "streamable_http",
            "url": mcp_server_url,
            "headers": {"Authorization": f"Bearer {access_token}"},
        }
        config_str = json.dumps(mcp_client_config, ensure_ascii=False, indent=2)
        try:
            await save_mcp_user_token(
                email=email,
                token=access_token,
                expires_at=expires_at,
                config_json=config_str,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="MCP token 保存失败，请稍后重试",
            ) from exc
        try:
            await _store_mcp_token_in_redis(
                email,
                access_token=access_token,
                expires_at=expires_at,
                config_formatted=config_str,
            )
        except Exception:
            logger.warning("MCP token 写入 Redis 缓存失败", exc_info=True)
        return CreateMcpTokenResponse(
            access_token=access_token,
            token_type="Bearer",
            expires_at=expires_at.isoformat(),
            config=config_str,
        )

    @router.get("/auth/mcp-token", response_model=CreateMcpTokenResponse)
    async def get_mcp_token(request: Request) -> CreateMcpTokenResponse:
        """返回当前用户最近一次签发的 MCP token（优先 Redis，未命中则用 PostgreSQL 回填缓存）。"""
        email = getattr(request.state, "mcp_token_user_email", None)
        if not isinstance(email, str) or not email:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="MCP token 路由未经过登录校验中间件",
            )
        cached = await redis_client.get(mcp_token_cache_key(email))
        if cached:
            try:
                return _response_from_cached_mcp_json(json.loads(cached))
            except (KeyError, TypeError, json.JSONDecodeError):
                pass

        row = await get_mcp_user_token_by_email(email)
        now = datetime.now(timezone.utc)
        if row is None or row.expires_at <= now:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="暂无有效 MCP token，请先通过 POST /auth/mcp-token 签发",
            )
        try:
            await _store_mcp_token_in_redis(
                email,
                access_token=row.token,
                expires_at=row.expires_at,
                config_formatted=row.config_json,
            )
        except Exception:
            logger.warning("MCP token Redis 回填失败", exc_info=True)
        return CreateMcpTokenResponse(
            access_token=row.token,
            token_type="Bearer",
            expires_at=row.expires_at.isoformat(),
            config=row.config_json,
        )

    return router
