"""
gRPC 流式 RAG 问答服务（与 Flask /ask 语义一致），基于 asyncio（grpc.aio）。

启动（需在 server 目录、已安装依赖）：
  python grpc_ask_server.py
  或: GRPC_BIND=0.0.0.0 GRPC_PORT=50051 python grpc_ask_server.py

与 gunicorn 同容器并行时使用 run_services.sh / Dockerfile CMD。
"""
from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
import grpc
from grpc import aio as grpc_aio
from ask_handlers import once_ask_payload, stream_ask_body
from grpc_generated.proto import ask_pb2, ask_pb2_grpc
from llm_common.rag.qwen_rag_service import QwenRagService


def _ollama_502_message() -> str:
    return (
        "DashScope / 嵌入服务调用失败。请检查：\n"
        "  1. 环境变量 DASHSCOPE_API_KEY 是否已配置（项目根 .env）\n"
        "  2. 网络与百炼配额是否正常\n"
        "  3. 若错误来自向量嵌入，请检查 EmbeddingProvider（Ollama 嵌入需本机 ollama 运行）"
    )


def _is_ollama_502(e: BaseException) -> bool:
    if getattr(e, "status_code", None) == 502:
        return True
    if "502" in str(e) or "502 Bad Gateway" in str(e):
        return True
    return False


def _session_id(raw: str) -> str | None:
    s = (raw or "").strip()
    return s or None


async def _stream_ask_body_async(
    loop: asyncio.AbstractEventLoop,
    rag: QwenRagService,
    query: str,
    *,
    trace: bool,
    thread_id: str | None,
) -> AsyncIterator[bytes]:
    """在线程中跑同步 `stream_ask_body`，经队列交给 async 生成器，避免阻塞事件循环。"""
    q: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=32)
    err: list[BaseException] = []

    def producer() -> None:
        try:
            for chunk in stream_ask_body(rag, query, trace=trace, thread_id=thread_id):
                asyncio.run_coroutine_threadsafe(q.put(chunk), loop).result()
        except BaseException as e:
            err.append(e)
        finally:
            asyncio.run_coroutine_threadsafe(q.put(None), loop).result()
    
    task = asyncio.create_task(asyncio.to_thread(producer))
    try:
        while True:
            chunk = await q.get()
            if chunk is None:
                break
            yield chunk
    finally:
        await task
    if err:
        raise err[0]


class AskGrpcServicer(ask_pb2_grpc.AskServiceServicer):
    def __init__(self, rag: QwenRagService) -> None:
        self._rag = rag

    async def AskStream(
        self, request: ask_pb2.AskRequest, context: grpc_aio.ServicerContext
    ):
        query = (request.query or "").strip()
        if not query:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "缺少参数 query")
        session_id = _session_id(request.session_id)
        trace = request.trace
        loop = asyncio.get_running_loop()
        try:
            async for chunk in _stream_ask_body_async(
                loop, self._rag, query, trace=trace, thread_id=session_id
            ):
                yield ask_pb2.AskStreamChunk(body_chunk=chunk)
        except ValueError as e:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except RuntimeError as e:
            await context.abort(grpc.StatusCode.UNAVAILABLE, str(e))
        except Exception as e:
            if _is_ollama_502(e):
                await context.abort(
                    grpc.StatusCode.UNAVAILABLE, _ollama_502_message()
                )
            else:
                await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def AskOnce(
        self, request: ask_pb2.AskRequest, context: grpc_aio.ServicerContext
    ) -> ask_pb2.AskOnceResponse:
        query = (request.query or "").strip()
        if not query:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "缺少参数 query")
        session_id = _session_id(request.session_id)
        trace = request.trace
        try:
            payload = await asyncio.to_thread(
                once_ask_payload,
                self._rag,
                query,
                trace=trace,
                thread_id=session_id,
            )
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            return ask_pb2.AskOnceResponse(json_body=raw)
        except ValueError as e:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except RuntimeError as e:
            await context.abort(grpc.StatusCode.UNAVAILABLE, str(e))
        except Exception as e:
            if _is_ollama_502(e):
                await context.abort(
                    grpc.StatusCode.UNAVAILABLE, _ollama_502_message()
                )
            else:
                await context.abort(grpc.StatusCode.INTERNAL, str(e))


async def serve() -> None:
    bind = (os.environ.get("GRPC_BIND") or "0.0.0.0").strip()
    port = int((os.environ.get("GRPC_PORT") or "50051").strip() or "50051")
    rag = QwenRagService()
    server = grpc_aio.server()
    ask_pb2_grpc.add_AskServiceServicer_to_server(AskGrpcServicer(rag), server)
    server.add_insecure_port(f"{bind}:{port}")
    await server.start()
    print(f"gRPC AskService (asyncio) listening on {bind}:{port}", flush=True)
    await server.wait_for_termination()


if __name__ == "__main__":
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass
