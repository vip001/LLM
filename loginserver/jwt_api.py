"""登录服务使用的用户 JWT（/auth/jwt/*）签发、校验与路由。"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional, Type

import jwt
from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel

try:
    from .dao.redis_store import code_key, jwt_blacklist_key, redis_client
except ImportError:
    from dao.redis_store import code_key, jwt_blacklist_key, redis_client

JWT_SECRET = os.getenv("JWT_SECRET", "replace-me-in-production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_SECONDS = int(os.getenv("JWT_EXPIRE_SECONDS", "7200"))


class JWTLoginResponse(BaseModel):
    access_token: str
    token_type: str
    expires_at: str
    user_email: str


def authorization_looks_like_jwt(token: str) -> bool:
    parts = token.split(".")
    return len(parts) == 3 and all(parts)


def create_jwt_token(email: str, expires_at: datetime) -> str:
    payload = {
        "sub": email,
        "iat": int(datetime.now(timezone.utc).timestamp()),
        "exp": int(expires_at.timestamp()),
        "type": "access",
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_jwt_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="JWT 已过期，请重新登录",
        ) from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="JWT 无效",
        ) from exc

    subject = payload.get("sub")
    if not isinstance(subject, str) or not subject:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="JWT 缺少用户标识",
        )
    return payload


def create_jwt_router(
    *,
    login_request_model: Type[BaseModel],
    current_user_response_model: Type[BaseModel],
    logout_response_model: Type[BaseModel],
    extract_bearer_token: Callable[[Optional[str]], str],
    email_regex: re.Pattern[str],
) -> APIRouter:
    router = APIRouter()

    @router.post("/auth/jwt/login", response_model=JWTLoginResponse)
    async def jwt_login(
        payload: login_request_model,  # pyright: ignore[reportInvalidTypeForm]
    ) -> JWTLoginResponse:
        email = payload.email.strip().lower()
        input_code = payload.code.strip()
        if not email_regex.match(email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="邮箱格式不正确",
            )

        saved_code = await redis_client.get(code_key(email))
        if not saved_code or saved_code != input_code:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="验证码错误或已过期",
            )

        expires_at = datetime.now(timezone.utc) + timedelta(seconds=JWT_EXPIRE_SECONDS)
        access_token = create_jwt_token(email=email, expires_at=expires_at)
        await redis_client.delete(code_key(email))

        return JWTLoginResponse(
            access_token=access_token,
            token_type="Bearer",
            expires_at=expires_at.isoformat(),
            user_email=email,
        )

    @router.get("/auth/jwt/me", response_model=current_user_response_model)
    async def jwt_me(
        authorization: Optional[str] = Header(default=None),
    ) -> current_user_response_model:  # pyright: ignore[reportInvalidTypeForm]
        token = extract_bearer_token(authorization)
        if await redis_client.get(jwt_blacklist_key(token)):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="JWT 已退出登录，请重新登录",
            )

        payload = decode_jwt_token(token)
        expires_at = datetime.fromtimestamp(payload["exp"], tz=timezone.utc).isoformat()
        return current_user_response_model(
            email=payload["sub"],
            token_type="Bearer",
            expires_at=expires_at,
        )

    @router.post("/auth/jwt/logout", response_model=logout_response_model)
    async def jwt_logout(
        authorization: Optional[str] = Header(default=None),
    ) -> logout_response_model:  # pyright: ignore[reportInvalidTypeForm]
        token = extract_bearer_token(authorization)
        payload = decode_jwt_token(token)
        exp_timestamp = int(payload["exp"])
        ttl_seconds = exp_timestamp - int(datetime.now(timezone.utc).timestamp())
        if ttl_seconds > 0:
            await redis_client.setex(jwt_blacklist_key(token), ttl_seconds, "1")
        return logout_response_model(message="已退出登录")

    return router
