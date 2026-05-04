"""
CLI client for the FastMCP server in ``llm_mcpserver.mcpserver`` (streamable HTTP),
using ``langchain-mcp-adapters`` ``MultiServerMCPClient`` to load LangChain tools.

Start the server first, e.g. from repo root::

    python -m llm_mcpserver.mcpserver

Then list tools::

    python mcpserver/test_mcpclient.py

Optional env: ``MCP_URL`` (default ``http://127.0.0.1:8001/mcp``). HTTP client uses
``trust_env=False`` so system ``HTTP_PROXY`` does not send localhost traffic through a
proxy (which often yields 502).

Uses the same ``jwt_key_pair`` / issuer / audience as ``llm_mcpserver.mcpserver`` (loaded
from PostgreSQL ``mcp_jwt_config`` after ``RSAKeyPair.generate()`` on first run) to mint a bearer JWT.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any, cast

import httpx
from langchain_core.messages import ToolMessage
from langchain_core.messages.tool import tool_call
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import Connection, McpHttpClientFactory
from mcp.shared._httpx_utils import MCP_DEFAULT_SSE_READ_TIMEOUT, MCP_DEFAULT_TIMEOUT

from llm_mcpserver.mcpserver import MCP_JWT_AUDIENCE, MCP_JWT_ISSUER, jwt_key_pair

DEFAULT_MCP_URL = os.environ.get("MCP_URL", "http://127.0.0.1:8001/mcp")


def _mcp_httpx_client_factory(
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout | None = None,
    auth: httpx.Auth | None = None,
) -> httpx.AsyncClient:
    """Match MCP default timeouts; ``trust_env=False`` avoids proxying ``127.0.0.1``."""
    if timeout is None:
        timeout = httpx.Timeout(MCP_DEFAULT_TIMEOUT, read=MCP_DEFAULT_SSE_READ_TIMEOUT)
    kwargs: dict[str, Any] = {
        "follow_redirects": True,
        "trust_env": False,
        "timeout": timeout,
    }
    if headers is not None:
        kwargs["headers"] = headers
    if auth is not None:
        kwargs["auth"] = auth
    return httpx.AsyncClient(**kwargs)
DEFAULT_TOOL = "retrieve_rag_contexts"


def parse_mcp_tool_text_content(raw: Any) -> dict[str, Any]:
    """从 ``tool.ainvoke`` 的返回值里取出 MCP 工具写在 *content* 里的整段 JSON 并解析。

    LangChain MCP 工具在 ``tool_call_id`` 为 ``None`` 时只返回 ``content``（常见为
    ``[{"type":"text","text": "{...}"}]``），``structuredContent`` 不会出现在返回值里。
    本函数把其中所有 ``text`` 块拼起来做一次 ``json.loads``。

    ``retrieve_rag_contexts`` 的 ``contexts`` 字段本身是 ``json.dumps`` 后的字符串，
    需要再 ``json.loads`` 一次才是 chunk 列表；解析结果放在 ``contexts_list``。
    """
    if isinstance(raw, str):
        text = raw
    elif isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        text = "".join(parts)
    else:
        msg = f"Unexpected tool return type for content parsing: {type(raw)}"
        raise TypeError(msg)

    data: dict[str, Any] = json.loads(text)
    ctx = data.get("contexts")
    if isinstance(ctx, str):
        data["contexts_list"] = json.loads(ctx)
    return data


def structured_content_from_tool_message(msg: ToolMessage) -> dict[str, Any]:
    """从带 ``tool_call_id`` 调用得到的 ``ToolMessage`` 里取出 MCP ``structuredContent``。

    ``langchain_mcp_adapters`` 把 MCP 的 ``structuredContent`` 放在
    ``artifact["structured_content"]``（与 ``MCPToolArtifact`` 一致）。
    """
    art = msg.artifact
    if not isinstance(art, dict) or "structured_content" not in art:
        raise TypeError(f"Unexpected artifact shape: {type(art)!r} {art!r}")
    sc = art["structured_content"]
    if not isinstance(sc, dict):
        raise TypeError(f"structured_content must be dict, got {type(sc)}")
    return sc


def _mcp_client(url: str) -> MultiServerMCPClient:
    # Streamable HTTP (same wire protocol as FastMCP ``streamable-http`` / ``http`` in adapters).
    token = jwt_key_pair.create_token(
        issuer=MCP_JWT_ISSUER,
        audience=MCP_JWT_AUDIENCE,
    )
    factory: McpHttpClientFactory = _mcp_httpx_client_factory
    cfg: dict[str, Any] = {
        "transport": "streamable_http",
        "url": url,
        "headers": {"Authorization": f"Bearer {token}"},
        "httpx_client_factory": factory,
    }
    rag_mcp: Connection = cast(Connection, cfg)
    return MultiServerMCPClient({"rag_mcp": rag_mcp})


async def _list_tools(client: MultiServerMCPClient) -> list[dict[str, Any]]:
    print("list tools")
    tools = await client.get_tools()
    rows: list[dict[str, Any]] = []
    for t in tools:
        rows.append(
            {
                "name": t.name,
                "description": (t.description or "").strip(),
            }
        )
    return rows

async def _call_retrieve(client: MultiServerMCPClient) -> Any:
    tools = await client.get_tools()
    tool = next(t for t in tools if t.name == "retrieve_rag_contexts")
    return await tool.ainvoke(
        {
            "query": "MMKV的用法",
            "k": 4,
            "enhance_strategy": "query2doc",  # 可改为 "hyde" 或 "none"
        }
    )


async def _call_retrieve_as_tool_message(
    client: MultiServerMCPClient,
    *,
    tool_call_id: str = "cli-example-1",
) -> ToolMessage:
    """用 ``ToolCall`` 传入 ``id``，返回值包成 ``ToolMessage``，MCP 结构化结果在 ``artifact``。

    不能写成 ``ainvoke({...}, tool_call_id=...)``：``_prep_run_args`` 里已固定传入
    ``tool_call_id``，再放到 ``kwargs`` 会与 ``dict(..., tool_call_id=..., **kwargs)``
    冲突而报错。
    """
    tools = await client.get_tools()
    tool = next(t for t in tools if t.name == "retrieve_rag_contexts")
    out = await tool.ainvoke(
        tool_call(
            name=tool.name,
            args={
                "query": "MMKV的用法",
                "k": 4,
                "enhance_strategy": "query2doc",
            },
            id=tool_call_id,
        ),
    )
    if not isinstance(out, ToolMessage):
        msg = f"Expected ToolMessage when tool_call_id is set, got {type(out)}"
        raise TypeError(msg)
    return out


async def main_async() -> None:
    client = _mcp_client(DEFAULT_MCP_URL)
    rows = await _list_tools(client)
    print("mcp get_tools ----")
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print("--- Example A: ainvoke without tool_call_id → parse content text JSON ---")
    raw = await _call_retrieve(client)
    print("raw type:", type(raw))
    parsed = parse_mcp_tool_text_content(raw)
    print("keys:", sorted(parsed.keys()))
    print("prompt_context preview:", (parsed.get("prompt_context") or "")[:200])
    cl = parsed.get("contexts_list")
    print("contexts_list len:", len(cl) if isinstance(cl, list) else cl)

    print("--- Example B: ainvoke with tool_call_id → ToolMessage + artifact ---")
    msg = await _call_retrieve_as_tool_message(client)
    print("ToolMessage.name:", msg.name, "tool_call_id:", msg.tool_call_id)
    structured = structured_content_from_tool_message(msg)
    print("structured_content keys:", sorted(structured.keys()))
    ctx_raw = structured.get("contexts")
    if isinstance(ctx_raw, str):
        ctx_list = json.loads(ctx_raw)
        print("contexts (from structured_content) chunk count:", len(ctx_list))
    print("prompt_context preview:", (structured.get("prompt_context") or "")[:200])


if __name__ == "__main__":
    asyncio.run(main_async())
    