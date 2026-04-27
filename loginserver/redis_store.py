import os
import logging

import redis.asyncio as redis

logger = logging.getLogger(__name__)

# 默认 TCP 连接；若配置了 UDS，会优先尝试 UDS。
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
REDIS_SOCKET_PATH = os.getenv("REDIS_SOCKET_PATH", "/tmp/redis.sock")
REDIS_PREFIX = os.getenv("REDIS_PREFIX", "loginserver")


def create_redis_client() -> redis.Redis:
    if REDIS_SOCKET_PATH:
        logger.info("Using Redis UDS socket: %s", REDIS_SOCKET_PATH)
        return redis.Redis(
            unix_socket_path=REDIS_SOCKET_PATH,
            decode_responses=True,
        )

    logger.info("Using Redis URL: %s", REDIS_URL)
    return redis.from_url(REDIS_URL, decode_responses=True)


redis_client = create_redis_client()


def code_key(email: str) -> str:
    return f"{REDIS_PREFIX}:code:{email.lower()}"


def session_key(token: str) -> str:
    return f"{REDIS_PREFIX}:session:{token}"


def jwt_blacklist_key(token: str) -> str:
    return f"{REDIS_PREFIX}:jwt:blacklist:{token}"
