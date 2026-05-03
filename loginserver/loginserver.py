import os
import re
import secrets
from contextlib import asynccontextmanager
import smtplib
import logging
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from random import randint
from typing import Optional
import jwt
from fastapi import FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

try:
    # Package import when running from project root: uvicorn loginserver:app
    from llm_common.postgres_store import init_postgres
    from .login_dao import (
        delete_login_session_by_token,
        get_login_session_by_token,
        save_login_session,
    )
    from .redis_store import code_key, jwt_blacklist_key, redis_client, session_key

    _PG_ENSURE_MODELS: tuple[str, ...] = ("loginserver.login_dao",)
except ImportError:
    # Module import when running inside loginserver directory: uvicorn loginserver:app
    from llm_common.postgres_store import init_postgres
    from login_dao import (
        delete_login_session_by_token,
        get_login_session_by_token,
        save_login_session,
    )
    from redis_store import code_key, jwt_blacklist_key, redis_client, session_key

    _PG_ENSURE_MODELS = ("login_dao",)

EMAIL_REGEX = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
CODE_EXPIRE_SECONDS = int(os.getenv("CODE_EXPIRE_SECONDS", "60"))
SESSION_EXPIRE_SECONDS = int(os.getenv("SESSION_EXPIRE_SECONDS", "604800"))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "1") == "1"
MAIL_FROM = os.getenv("MAIL_FROM", SMTP_USER)
JWT_SECRET = os.getenv("JWT_SECRET", "replace-me-in-production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_SECONDS = int(os.getenv("JWT_EXPIRE_SECONDS", "7200"))


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    await redis_client.ping()
    await init_postgres(ensure_models=_PG_ENSURE_MODELS)
    yield


app = FastAPI(title="Login Server", version="1.0.0", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[item.strip() for item in ALLOWED_ORIGINS.split(",") if item.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class SendCodeRequest(BaseModel):
    email: EmailStr


class SendCodeResponse(BaseModel):
    message: str
    expires_in: int = Field(..., description="验证码有效时长，秒")
    # 开发阶段可返回验证码，生产环境建议关闭
    code: Optional[str] = None


class LoginRequest(BaseModel):
    email: EmailStr
    code: str = Field(..., min_length=4, max_length=8)


class LoginResponse(BaseModel):
    token: str
    token_type: str
    expires_at: str
    user_email: str


class CurrentUserResponse(BaseModel):
    email: str
    token_type: str
    expires_at: Optional[str] = None


class LogoutResponse(BaseModel):
    message: str


class JWTLoginResponse(BaseModel):
    access_token: str
    token_type: str
    expires_at: str
    user_email: str


def send_verification_email(email: str, code: str) -> None:
    if not SMTP_HOST or not MAIL_FROM:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="邮件服务未配置，请联系管理员",
        )

    message = EmailMessage()
    message["Subject"] = "您的登录验证码"
    message["From"] = MAIL_FROM
    message["To"] = email
    message.set_content(
        f"您的验证码是：{code}\n"
        f"有效期 {CODE_EXPIRE_SECONDS} 秒，请勿泄露给他人。"
    )

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as smtp:
            if SMTP_USE_TLS:
                smtp.starttls()
            if SMTP_USER:
                smtp.login(SMTP_USER, SMTP_PASSWORD)
            smtp.send_message(message)
    except smtplib.SMTPException as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="验证码邮件发送失败，请稍后重试",
        ) from exc


def extract_bearer_token(authorization: Optional[str]) -> str:
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少 Authorization 头",
        )

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization 格式错误，应为 Bearer <token>",
        )
    return token.strip()


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


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/auth/send-code", response_model=SendCodeResponse)
async def send_code(payload: SendCodeRequest) -> SendCodeResponse:
    email = payload.email.strip().lower()
    if not EMAIL_REGEX.match(email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="邮箱格式不正确",
        )

    code = f"{randint(0, 999999):06d}"
    await redis_client.setex(code_key(email), CODE_EXPIRE_SECONDS, code)

   # send_verification_email(email, code)

    return SendCodeResponse(
        message="验证码已发送",
        expires_in=CODE_EXPIRE_SECONDS,
        code=code if os.getenv("ENABLE_DEBUG_CODE", "1") == "1" else None,
    )


@app.post("/auth/login", response_model=LoginResponse)
async def login(payload: LoginRequest) -> LoginResponse:
    email = payload.email.strip().lower()
    input_code = payload.code.strip()
    if not EMAIL_REGEX.match(email):
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

    token = secrets.token_urlsafe(32)
    expire_at = datetime.now(timezone.utc) + timedelta(seconds=SESSION_EXPIRE_SECONDS)
    await redis_client.setex(session_key(token), SESSION_EXPIRE_SECONDS, email)
    try:
        await save_login_session(email=email, token=token, expires_at=expire_at)
    except Exception as exc:
        await redis_client.delete(session_key(token))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="登录会话保存失败，请稍后重试",
        ) from exc
    await redis_client.delete(code_key(email))

    return LoginResponse(
        token=token,
        token_type="Bearer",
        expires_at=expire_at.isoformat(),
        user_email=email,
    )


@app.get("/auth/me", response_model=CurrentUserResponse)
async def get_login_info(
    authorization: Optional[str] = Header(default=None),
) -> CurrentUserResponse:
    token = extract_bearer_token(authorization)
    email = await redis_client.get(session_key(token))
    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="登录已失效，请重新登录",
        )

    expires_at_value: Optional[str] = None
    db_session = await get_login_session_by_token(token)
    if db_session:
        expires_at_value = db_session.expires_at.isoformat()

    return CurrentUserResponse(
        email=email,
        token_type="Bearer",
        expires_at=expires_at_value,
    )


@app.post("/auth/logout", response_model=LogoutResponse)
async def logout(authorization: Optional[str] = Header(default=None)) -> LogoutResponse:
    token = extract_bearer_token(authorization)
    if not await redis_client.get(session_key(token)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="登录已失效，请重新登录",
        )

    await redis_client.delete(session_key(token))
    await delete_login_session_by_token(token)
    return LogoutResponse(message="已退出登录")


@app.post("/auth/jwt/login", response_model=JWTLoginResponse)
async def jwt_login(payload: LoginRequest) -> JWTLoginResponse:
    email = payload.email.strip().lower()
    input_code = payload.code.strip()
    if not EMAIL_REGEX.match(email):
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


@app.get("/auth/jwt/me", response_model=CurrentUserResponse)
async def jwt_me(authorization: Optional[str] = Header(default=None)) -> CurrentUserResponse:
    token = extract_bearer_token(authorization)
    if await redis_client.get(jwt_blacklist_key(token)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="JWT 已退出登录，请重新登录",
        )

    payload = decode_jwt_token(token)
    expires_at = datetime.fromtimestamp(payload["exp"], tz=timezone.utc).isoformat()
    return CurrentUserResponse(
        email=payload["sub"],
        token_type="Bearer",
        expires_at=expires_at,
    )


@app.post("/auth/jwt/logout", response_model=LogoutResponse)
async def jwt_logout(authorization: Optional[str] = Header(default=None)) -> LogoutResponse:
    token = extract_bearer_token(authorization)
    payload = decode_jwt_token(token)
    exp_timestamp = int(payload["exp"])
    ttl_seconds = exp_timestamp - int(datetime.now(timezone.utc).timestamp())
    if ttl_seconds > 0:
        await redis_client.setex(jwt_blacklist_key(token), ttl_seconds, "1")
    return LogoutResponse(message="已退出登录")
