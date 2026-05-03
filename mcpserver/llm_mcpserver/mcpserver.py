"""
FastMCP entry: expose RAG retrieval `contexts` (and `prompt_context`) using the same
pipeline as `rag_graph.retrieve_node` / `QwenRagService.retrieve_context`.

JWT: dev keys are created with ``RSAKeyPair.generate()`` on first run and reused from
``mcpserver/.mcp_jwt_dev_keys.json`` so separate server and client processes share the pair.
``MCP_JWT_ISSUER`` must match ``RSAKeyPair.create_token`` defaults. Import ``jwt_key_pair``
to mint bearer tokens (e.g. from ``test_mcpclient``).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from fastmcp.server.auth import JWTVerifier
from fastmcp.server.auth.providers.jwt import RSAKeyPair
from pydantic import SecretStr

from llm_common.rag.qwen_rag_service import QwenRagService

MCP_JWT_ISSUER = "https://fastmcp.example.com"
MCP_JWT_AUDIENCE = "Android黄金屋社区"

_DEV_JWT_KEYS_PATH = Path(__file__).resolve().parent.parent / ".mcp_jwt_dev_keys.json"


def _dev_jwt_key_pair() -> RSAKeyPair:
    if _DEV_JWT_KEYS_PATH.is_file():
        try:
            data = json.loads(_DEV_JWT_KEYS_PATH.read_text(encoding="utf-8"))
            priv = data.get("private_pem")
            pub = data.get("public_pem")
            if (
                isinstance(priv, str)
                and isinstance(pub, str)
                and "PRIVATE KEY" in priv
                and "PUBLIC KEY" in pub
            ):
                return RSAKeyPair(private_key=SecretStr(priv), public_key=pub)
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    pair = RSAKeyPair.generate()
    try:
        _DEV_JWT_KEYS_PATH.write_text(
            json.dumps(
                {
                    "private_pem": pair.private_key.get_secret_value(),
                    "public_pem": pair.public_key,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except OSError:
        pass
    return pair


jwt_key_pair = _dev_jwt_key_pair()

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
