"""
FastMCP entry: expose RAG retrieval `contexts` (and `prompt_context`) using the same
pipeline as `rag_graph.retrieve_node` / `QwenRagService.retrieve_context`.

JWT: RSA 密钥对与 ``issuer`` / ``audience`` 存于 PostgreSQL 表 ``mcp_jwt_config``（首启时生成
并写入；可通过环境变量 ``MCP_JWT_ISSUER`` / ``MCP_JWT_AUDIENCE`` 指定默认值）。
``MCP_JWT_ISSUER`` / ``MCP_JWT_AUDIENCE`` 运行中取自数据库行，须与签发端一致。
导入 ``jwt_key_pair`` 可签发 Bearer（如 ``test_mcpclient``）。
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from fastmcp import FastMCP
from fastmcp.server.auth import JWTVerifier
from fastmcp.server.auth.providers.jwt import RSAKeyPair
from pydantic import SecretStr

# Loads repo ``.env`` before ``postgres_store`` reads ``PG_*`` at import time.
from llm_common.mcp_jwt_dao import get_mcp_jwt_config, save_mcp_jwt_config
from llm_common.rag.qwen_rag_service import QwenRagService
from llm_common.postgres_store import init_postgres

_DEFAULT_MCP_JWT_ISSUER = "https://fastmcp.example.com"
_DEFAULT_MCP_JWT_AUDIENCE = "Android黄金屋社区"


def _default_issuer_audience() -> tuple[str, str]:
    issuer = os.getenv("MCP_JWT_ISSUER", _DEFAULT_MCP_JWT_ISSUER).strip() or _DEFAULT_MCP_JWT_ISSUER
    audience = (
        os.getenv("MCP_JWT_AUDIENCE", _DEFAULT_MCP_JWT_AUDIENCE).strip()
        or _DEFAULT_MCP_JWT_AUDIENCE
    )
    return issuer, audience


async def _bootstrap_mcp_jwt_async() -> tuple[str, str, RSAKeyPair]:
    await init_postgres(ensure_models=("llm_common.mcp_jwt_dao",))
    row = await get_mcp_jwt_config()
    default_issuer, default_audience = _default_issuer_audience()
    if row is not None:
        priv = row.private_key_pem
        pub = row.public_key_pem
        iss = (row.issuer or "").strip()
        aud = (row.audience or "").strip()
        if (
            iss
            and aud
            and isinstance(priv, str)
            and isinstance(pub, str)
            and "PRIVATE KEY" in priv
            and "PUBLIC KEY" in pub
        ):
            return iss, aud, RSAKeyPair(private_key=SecretStr(priv), public_key=pub)
    pair = RSAKeyPair.generate()
    await save_mcp_jwt_config(
        issuer=default_issuer,
        audience=default_audience,
        public_key_pem=pair.public_key,
        private_key_pem=pair.private_key.get_secret_value(),
    )
    return default_issuer, default_audience, pair


MCP_JWT_ISSUER, MCP_JWT_AUDIENCE, jwt_key_pair = asyncio.run(_bootstrap_mcp_jwt_async())

auth = JWTVerifier(
    public_key=jwt_key_pair.public_key,
    issuer=MCP_JWT_ISSUER,
    audience=MCP_JWT_AUDIENCE,
)

mcp = FastMCP(
    name="llm-rag-contexts",
    instructions=(
        "Provides knowledge-base retrieval: structured `contexts` plus a flattened "
        "`prompt_context` string, aligned with the project's RAG graph retrieve step."
    ),
    auth=auth,
)

_service: QwenRagService | None = None


def _get_service() -> QwenRagService:
    global _service
    if _service is None:
        _service = QwenRagService()
    return _service


@mcp.tool(
    name="retrieve_rag_contexts",
    description=(
        "Run vector/text retrieval with optional query enhancement (query2doc, hyde, …), "
        "merge same-page images, then return `prompt_context` and serialized `contexts` "
        "(type, text, metadata, optional image_data) for downstream LLM use."
    ),
)
def retrieve_rag_contexts(
    query: str,
    k: int = 4,
    enhance_strategy: str = "query2doc",
) -> dict[str, Any]:
    """
    Args:
        query: User question or search query.
        k: Max primary chunks to retrieve before page-image merge.
        enhance_strategy: Passed to query_enhance (e.g. query2doc, hyde, none).

    Returns:
        ``prompt_context`` (flattened string for system prompt) and ``contexts``
        (serialized chunks: type, text, metadata, optional image_data).
    """
    svc = _get_service()
    prompt_context, contexts = svc.retrieve_context(
        query=(query or "").strip(),
        k=int(k),
        enhance_strategy=(enhance_strategy or "query2doc").strip(),
    )
    return {
        "prompt_context": prompt_context,
        "contexts": json.dumps(contexts),
    }


def main() -> None:
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8001, path="/mcp")


if __name__ == "__main__":
    main()
