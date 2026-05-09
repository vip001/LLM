"""
与 Flask /ask、gRPC AskService 共用的 RAG 调用与流式字节编码（RAG\\x01 头 + 正文）。
"""
from __future__ import annotations

import json
import struct
from typing import Any, Iterator

from llm_common.rag.qwen_rag_service import QwenRagService

_STREAM_REFS_MAGIC = b"RAG\x01"


def to_stream_bytes(chunk: Any) -> bytes:
    if chunk is None:
        return b""
    if isinstance(chunk, bytes):
        return chunk
    if isinstance(chunk, str):
        return chunk.encode("utf-8")
    return str(chunk).encode("utf-8")


def stream_ask_body(
    rag: QwenRagService,
    query: str,
    *,
    trace: bool,
    thread_id: str | None,
) -> Iterator[bytes]:
    contexts, stream_gen, meta = rag.ask_stream(query, trace=trace, thread_id=thread_id)
    refs_payload: dict[str, Any] = {
        "contexts": contexts,
        "sessionId": meta["sessionId"],
    }
    if trace and meta.get("trace") is not None:
        refs_payload["trace"] = meta["trace"]
    refs_bytes = json.dumps(refs_payload, ensure_ascii=False).encode("utf-8")
    header = _STREAM_REFS_MAGIC + struct.pack(">I", len(refs_bytes)) + refs_bytes
    yield header
    first_chunk = None
    try:
        first_chunk = next(stream_gen)
    except StopIteration:
        pass
    if first_chunk:
        yield to_stream_bytes(first_chunk)
    for part in stream_gen:
        yield to_stream_bytes(part)


def once_ask_payload(
    rag: QwenRagService,
    query: str,
    *,
    trace: bool,
    thread_id: str | None,
) -> dict[str, Any]:
    return rag.ask_once(query, trace=trace, thread_id=thread_id)
