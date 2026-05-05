"""
使用阿里云百炼 DashScope 的 Qwen 对话模型，通过 Flask 提供 RAG 问答 API。
对话与检索逻辑见 llm_common.rag.qwen_rag_service.QwenRagService，DashScope 客户端见 llm_common.rag.dashscope_llm.DashScopeLLMClient。

运行方式（需已安装共享包：pip install -e ../llm_common；可在 server 目录下启动）：
  cd /Users/xiafeng/PythonProject/llm/server
  ../.venv/bin/python ollama_qwen.py
  启动 HTTP：../.venv/bin/python ollama_qwen.py serve

API 使用：
  GET  /ask?query=MMKV的用法
  POST /ask   Body: {"query": "MMKV的用法"}

多轮对话（LangGraph checkpointer + 服务端会话）：
  - 首次请求可不传 session_id，响应 JSON 含 session_id，或流式时见首包 JSON 字段 session_id / 响应头 X-Session-Id。
  - 后续请求传入同一 session_id：Query 参数、JSON 字段 session_id，或请求头 X-Session-Id。

前置条件：
1. pip 已安装 dashscope、langchain-community（见 server/requirements.txt）
2. 项目根目录 .env 中配置 DASHSCOPE_API_KEY（与官方文档一致）
3. 可选环境变量：
   - DASHSCOPE_BASE_HTTP_API_URL：DashScope HTTP 基址，默认与官方文档一致
     https://dashscope.aliyuncs.com/api/v1（其他地域可覆盖）
   - DASHSCOPE_CHAT_MODEL：文本对话模型，默认 qwen3.5-plus（百炼 DashScope）
   - DASHSCOPE_VL_MODEL：检索结果含图片时使用，默认 qwen3.5-plus（走 MultiModalConversation，与文档示例一致）
4. 向量检索 / HyDE 仍使用 EmbeddingProvider（如 Ollama 或 DashScope 嵌入），与对话模型独立
5. 多轮会话持久化（生产）：环境变量 LANGGRAPH_CHECKPOINT_DB_URI 指向 PostgreSQL（如 Docker 默认
   postgresql://loginserver:loginserver@postgres:5432/loginserver?sslmode=disable，与 loginserver 应用共用库）。未设置时使用内存 checkpoint。
   可选 LANGGRAPH_CHECKPOINT_POOL_MAX 调整连接池大小（默认 5）。
"""
# 必须在 import 任何使用 OpenMP 的库（torch/faiss/numpy 等）之前设置，避免多份 libomp 冲突导致 OMP Error #15
import json
import os
import struct
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from flask import Flask, request, jsonify, Response
from typing import Any


def _session_id_from_request() -> str | None:
    h = (request.headers.get("X-Session-Id") or request.headers.get("X-Session-ID") or "").strip()
    if h:
        return h
    if request.method == "GET":
        sid = (request.args.get("session_id") or request.args.get("sessionId") or "").strip()
        return sid or None
    body = request.get_json(silent=True) or {}
    sid = (body.get("session_id") or body.get("sessionId") or "")
    if isinstance(sid, str) and sid.strip():
        return sid.strip()
    return None

from llm_common.rag.qwen_rag_service import QwenRagService

_rag = QwenRagService()


def _parse_stream_flag(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError("参数 stream 只能是 true/false")


def _parse_trace_flag(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError("参数 trace 只能是 true/false")


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


OLLAMA_502_RETRY_TIMES = 3
OLLAMA_502_RETRY_DELAYS = (2, 4, 6)

# 流式响应：魔数 + uint32(JSON 字节长度) + UTF-8 JSON({"contexts": [...]}) + 正文 token（UTF-8）
_STREAM_REFS_MAGIC = b"RAG\x01"


def _to_stream_bytes(chunk: Any) -> bytes:
    if chunk is None:
        return b""
    if isinstance(chunk, bytes):
        return chunk
    if isinstance(chunk, str):
        return chunk.encode("utf-8")
    return str(chunk).encode("utf-8")


class AskHttpController:
    """Flask /ask 路由：解析请求参数，调用 RAG 服务，返回 JSON 或 text/plain 流。"""

    def __init__(self, rag: QwenRagService) -> None:
        self._rag = rag

    def ask(self) -> Response | tuple[Any, int]:
        body: dict[str, Any] = {}
        if request.method == "GET":
            query = request.args.get("query", "").strip()
            stream = _parse_stream_flag(request.args.get("stream"), default=True)
            trace = _parse_trace_flag(request.args.get("trace"), default=False)
        else:
            body = request.get_json(silent=True) or {}
            query = (body.get("query") or "").strip()
            stream = _parse_stream_flag(body.get("stream"), default=True)
            trace = _parse_trace_flag(body.get("trace"), default=False)

        if not query:
            return jsonify({"error": "缺少参数 query"}), 400
        session_id = _session_id_from_request()
        try:
            if stream:
                contexts, stream_gen, meta = self._rag.ask_stream(
                    query, trace=trace, thread_id=session_id
                )
                refs_payload: dict[str, Any] = {
                    "contexts": contexts,
                    "session_id": meta["session_id"],
                }
                if trace and meta.get("trace") is not None:
                    refs_payload["trace"] = meta["trace"]
                refs_bytes = json.dumps(refs_payload, ensure_ascii=False).encode("utf-8")
                header = _STREAM_REFS_MAGIC + struct.pack(">I", len(refs_bytes)) + refs_bytes

                def stream_refs_then_text():
                    yield header
                    first_chunk = None
                    try:
                        first_chunk = next(stream_gen)
                    except StopIteration:
                        pass
                    if first_chunk:
                        yield _to_stream_bytes(first_chunk)
                    for part in stream_gen:
                        yield _to_stream_bytes(part)

                return Response(
                    stream_refs_then_text(),
                    mimetype="application/octet-stream",
                    headers={"X-Session-Id": meta["session_id"]},
                )
            print(f"rag askOnce: {query}")
            payload = self._rag.ask_once(query, trace=trace, thread_id=session_id)
            resp = jsonify(payload)
            resp.headers["X-Session-Id"] = payload.get("session_id", "")
            return resp
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 502
        except Exception as e:
            if _is_ollama_502(e):
                return jsonify({"error": _ollama_502_message()}), 502
            return jsonify({"error": str(e)}), 500


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["JSON_AS_ASCII"] = False

    ask_controller = AskHttpController(_rag)

    @app.route("/ask", methods=["GET", "POST"])
    def api_ask():
        return ask_controller.ask()

    @app.route("/")
    def health():
        return jsonify({"status": "ok"})

    return app


app = create_app()


def main() -> None:
    import sys
    last_e: BaseException | None = None
    for attempt in range(OLLAMA_502_RETRY_TIMES):
        try:
            _ctx, _gen, _meta = _rag.ask_stream("MMKV和sharedpreferences的性能对比")
            for part in _gen:
                print(part, end="", flush=True)
            print()
            return
        except Exception as e:
            last_e = e
            if _is_ollama_502(e) and attempt < OLLAMA_502_RETRY_TIMES - 1:
                time.sleep(OLLAMA_502_RETRY_DELAYS[attempt])
                continue
            if _is_ollama_502(e):
                print(_ollama_502_message(), file=sys.stderr)
                raise SystemExit(1) from e
            raise
    if last_e is not None:
        raise last_e


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        app.run(
            host=os.getenv("FLASK_HOST", "0.0.0.0"),
            port=int(os.getenv("FLASK_PORT", "5000")),
            debug=False,
        )
    else:
        main()
