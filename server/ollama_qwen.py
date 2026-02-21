"""
使用 LangChain (langchain-ollama) 加载本地通过 Ollama 部署的 Qwen2.5:7b_instruct 模型，
并通过 Flask 提供 API 服务。

运行方式（必须在项目根目录且使用本项目的 venv）：
  cd /Users/xiafeng/PythonProject/llm
  .venv/bin/python ollama_qwen.py
  或先激活虚拟环境：source .venv/bin/activate  再执行：python ollama_qwen.py

API 使用：
  GET  /ask?query=MMKV的用法
  POST /ask   Body: {"query": "MMKV的用法"}

前置条件：
1. 已安装并运行 Ollama：https://ollama.ai
2. 已拉取模型：ollama pull qwen2.5:7b-instruct
   （若你使用的 tag 是 qwen2.5:7b_instruct，下面 model 名改为 "qwen2.5:7b_instruct" 即可）
"""

from flask import Flask, request, jsonify, Response

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from vector_db import VectorDB


app = Flask(__name__)
# 让 jsonify 直接输出中文，而不是 \uXXXX 形式的 Unicode 转义
app.config["JSON_AS_ASCII"] = False

# 复用 LLM 与链（避免每次请求重建）
_llm = None
_llm_chain = None
_json_parser = None


def _get_chain():
    global _llm, _llm_chain, _json_parser
    if _llm_chain is None:
        _llm = ChatOllama(
            model="qwen2.5:7b-instruct",
            base_url="http://localhost:11434",
            temperature=0.7,
        )
        messages = ChatPromptTemplate.from_messages([
            ("system", "你是一个有帮助的助手。请根据以下 context 回答用户的问题。如果 context 没有相关信息，请回答不知道，不要自己编造。\n\n"
             "context:\n{context}"),
            ("human", "{question}"),
        ])
        _llm_chain = messages | _llm
    return _llm_chain


def _get_context(query: str, k: int = 4) -> str:
    """RAG 检索，返回 context 文本；异常时抛出 RuntimeError 或原异常。"""
    db = VectorDB()
    try:
        docs = db.search(query, k=k)
    except Exception as e:
        err = str(e).strip()
        if "502" in err or "ResponseError" in err:
            raise RuntimeError(
                "Ollama 嵌入服务异常 (502)。请检查：\n"
                "  1. Ollama 是否在运行：终端执行 ollama list\n"
                "  2. 是否已拉取嵌入模型：ollama pull bge-m3\n"
                "  3. 若仍报错，可重启 Ollama 后再试。"
            ) from e
        raise
    return "\n\n".join(doc.page_content for doc in docs) if docs else "（未检索到相关文档）"


def ask_stream(query: str, k: int = 4):
    """根据 query 做 RAG 检索并流式调用 LLM，逐块 yield 文本。"""
    context = _get_context(query, k=k)
    chain = _get_chain()
    for chunk in chain.stream({
        "context": context,
        "question": query,
    }):
        content = getattr(chunk, "content", None)
        if content:
            yield content


@app.route("/ask", methods=["GET", "POST"])
def api_ask():
    """接收 query 参数：GET 用 ?query=xxx，POST 用 JSON body {"query": "xxx"}。流式返回纯文本。"""
    if request.method == "GET":
        query = request.args.get("query", "").strip()
    else:
        body = request.get_json(silent=True) or {}
        query = (body.get("query") or "").strip()

    if not query:
        return jsonify({"error": "缺少参数 query"}), 400

    try:
        return Response(
            ask_stream(query),
            mimetype="text/plain; charset=utf-8",
        )
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def health():
    return jsonify({"status": "ok"})


def main():
    # 命令行直接跑一次（兼容原有用法）
    for part in ask_stream("MMKV的用法"):
        print(part, end="", flush=True)
    print()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        app.run(host="0.0.0.0", port=5000, debug=True)
    else:
        main()
