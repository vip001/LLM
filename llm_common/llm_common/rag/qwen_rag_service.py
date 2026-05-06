"""
RAG 问答编排：向量检索、上下文与消息构造、调用 DashScope 对话模型（非 HTTP）。
"""
# 在 import 可能触发 OpenMP 的库之前设置（与 ollama_qwen 入口一致）
import atexit
import os
import traceback
import uuid

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
from typing import Any, Iterator, cast

from llm_common.paths import PathsUtil

PathsUtil.load_repo_dotenv()

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import Conn, PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from llm_common.rag.dashscope_llm import DashScopeLLMClient
from llm_common.rag.embedding_provider import EmbeddingProvider
from llm_common.rag.query_enhance import get_strategy
from llm_common.rag.rag_graph import RagState, build_rag_graph
from llm_common.rag.vector_db import VectorDB


_CONTEXTUALIZE_SYSTEM = """你是检索查询改写助手。根据对话历史与「当前用户句」，生成一条自包含的检索查询：
补全代词与省略所指，使不依赖上文也能被检索系统理解。
只输出这一条查询本身：不要解释、不要引号、不要前后缀、不要多余换行。"""


def _build_langgraph_checkpointer() -> BaseCheckpointSaver[str]:
    """
    生产环境：设置 LANGGRAPH_CHECKPOINT_DB_URI（postgresql://…），使用 PostgresSaver。
    未设置时回退 InMemorySaver（本地开发）。
    """
    uri = (os.environ.get("LANGGRAPH_CHECKPOINT_DB_URI") or "").strip()
    if not uri:
        return InMemorySaver()
    max_size = int((os.environ.get("LANGGRAPH_CHECKPOINT_POOL_MAX") or "5").strip() or "5")
    pool = ConnectionPool(
        conninfo=uri,
        min_size=1,
        max_size=max(1, max_size),
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
    )
    saver = PostgresSaver(cast(Conn, pool))
    saver.setup()

    def _close_pool() -> None:
        try:
            pool.close()
        except Exception:
            pass

    atexit.register(_close_pool)
    return saver


class QwenRagService:
    """检索 + 查询增强 + Qwen 单次 / 流式作答。"""

    def __init__(self) -> None:
        self._llm = DashScopeLLMClient()
        self._embedding_provider = EmbeddingProvider()
        self._checkpointer = _build_langgraph_checkpointer()
        self._graph = build_rag_graph(self, self._checkpointer)

    def get_embeddings(self) -> Embeddings:
        return self._embedding_provider.get_embeddings()

    def get_text_llm_for_strategy(self):
        """供 query_enhance 等需要 Generation 接口的组件使用。"""
        return self._llm.get_text_model()

    @staticmethod
    def _max_distance_threshold() -> float | None:
        """
        FAISS 的 score 通常是距离（越小越相关）。
        固定阈值：仅保留 score <= 阈值的结果。
        经验建议可先用 0.9；召回太少可适当放宽到 1.1 左右。
        """
        return 0.9

    @staticmethod
    def _filter_docs_by_max_distance(
        docs_with_score: list[tuple[Document, float]],
        *,
        max_distance: float | None,
    ) -> list[Document]:
        if max_distance is None:
            return [doc for doc, _ in docs_with_score]
        return [doc for doc, score in docs_with_score if score <= max_distance]

    @staticmethod
    def _get_system_prompt(context: str) -> str:
        return (
            "你是一个有帮助的助手。请根据以下 context 回答用户的问题。"
            "如果提供了图片，请同时结合图片内容与 context 作答。"
            "如果 context 和图片里都没有相关信息，请回答不知道，不要自己编造。\n\n"
            f"context:\n{context}"
        )

    @staticmethod
    def _is_supported_image_ref(value: Any) -> bool:
        if not isinstance(value, str):
            return False
        text = value.strip()
        return text.startswith("data:image/") or text.startswith("http://") or text.startswith("https://")

    @classmethod
    def extract_image_refs(cls, contexts: list[dict[str, Any]]) -> list[str]:
        seen: set[str] = set()
        image_refs: list[str] = []
        for item in contexts:
            if item.get("type") != "image":
                continue
            raw = item.get("image_data")
            if not isinstance(raw, str) or not cls._is_supported_image_ref(raw):
                continue
            if raw in seen:
                continue
            seen.add(raw)
            image_refs.append(raw)
        return image_refs

    def _build_llm_messages(
        self,
        query: str,
        context: str,
        contexts: list[dict[str, Any]],
        *,
        prior_messages: list[BaseMessage] | None = None,
    ) -> list[BaseMessage]:
        image_refs = self.extract_image_refs(contexts)
        system = SystemMessage(content=self._get_system_prompt(context))
        if image_refs:
            user_text = (
                f"用户问题：{query}\n\n"
                "下面附有与检索结果对应的图片，请结合这些图片和上面的 context 一起回答。"
            )
            blocks: list[str | dict] = [{"type": "text", "text": user_text}]
            for image_ref in image_refs:
                blocks.append(
                    {"type": "image_url", "image_url": {"url": image_ref}},
                )
            user_msg: BaseMessage = HumanMessage(content=blocks)
        else:
            user_msg = HumanMessage(content=query)
        if not prior_messages:
            return [system, user_msg]
        trimmed: list[BaseMessage] = []
        for m in prior_messages[-10:]:
            if isinstance(m, (HumanMessage, AIMessage)):
                trimmed.append(m)
        return [system, *trimmed, user_msg]

    def contextualize_retrieval_query(
        self,
        messages: list[BaseMessage],
        latest_user_text: str,
        *,
        max_history_messages: int = 6,
    ) -> str:
        q = (latest_user_text or "").strip()
        if not q:
            return ""
        prior = list(messages)
        if prior and isinstance(prior[-1], HumanMessage):
            prior = prior[:-1]
        if not prior:
            return q
        lines: list[str] = []
        for m in prior[-max_history_messages:]:
            if isinstance(m, HumanMessage):
                role = "用户"
            elif isinstance(m, AIMessage):
                role = "助手"
            else:
                continue
            text = self.message_content_to_text(m.content).strip()
            if text:
                lines.append(f"{role}: {text}")
        hist = "\n".join(lines).strip()
        if not hist:
            return q
        user_block = f"对话历史：\n{hist}\n\n当前用户句：{q}"
        out = self._llm.invoke(
            [
                SystemMessage(content=_CONTEXTUALIZE_SYSTEM),
                HumanMessage(content=user_block),
            ]
        )
        rewritten = self.message_content_to_text(getattr(out, "content", out)).strip()
        print("rewritten:", rewritten,";originquery:",q)
        return rewritten or q

    def _search_docs(
        self,
        query: str,
        k: int = 4,
        *,
        filter_: dict[str, Any] | None = None,
        fetch_k: int = 20,
    ) -> list[Document]:
        db = VectorDB(embeddings=self.get_embeddings())
        try:
            docs_with_score = db.search_with_score(query, k=k, filter_=filter_)
            return self._filter_docs_by_max_distance(
                docs_with_score,
                max_distance=self._max_distance_threshold(),
            )
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

    def _search_docs_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        *,
        filter_: dict[str, Any] | None = None,
        fetch_k: int = 20,
    ) -> list[Document]:
        db = VectorDB(embeddings=self.get_embeddings())
        try:
            docs_with_score = db.search_by_vector_with_score(
                embedding,
                k=k,
                filter_=filter_,
                fetch_k=fetch_k,
            )
            return self._filter_docs_by_max_distance(
                docs_with_score,
                max_distance=self._max_distance_threshold(),
            )
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

    @staticmethod
    def prompt_context_from_docs(docs: list[Document]) -> str:
        if not docs:
            return "（未检索到相关文档）"

        items: list[str] = []
        for idx, doc in enumerate[Document](docs, start=1):
            metadata = doc.metadata or {}
            source = metadata.get("source") or "unknown"
            page = metadata.get("page")
            page_info = f", page={page}" if page is not None else ""
            if metadata.get("type") == "image":
                header = f"[{idx}] 图片资料 source={source}{page_info}"
                body = doc.page_content.strip() or "（该图片无关联文字上下文）"
                items.append(f"{header}\n相关文字：{body}")
            else:
                header = f"[{idx}] 文本资料 source={source}{page_info}"
                body = doc.page_content.strip() or "（空文本）"
                items.append(f"{header}\n{body}")
        return "\n\n".join(items)

    @staticmethod
    def serialize_context_docs(docs: list[Document]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for doc in docs:
            metadata = dict(doc.metadata or {})
            item_type = metadata.get("type") or "text"
            image_data = metadata.pop("image_data", None)
            item: dict[str, Any] = {
                "type": item_type,
                "text": doc.page_content or "",
                "metadata": metadata,
            }
            if item_type == "image" and image_data:
                item["image_data"] = image_data
            results.append(item)
        return results

    @staticmethod
    def _doc_page_key(doc: Document) -> tuple[str, int] | None:
        """非图片 chunk 的 (规范化路径, 页码)，用于与同页图片对齐。"""
        m = doc.metadata or {}
        if m.get("type") == "image":
            return None
        src = m.get("source")
        if not src or m.get("page") is None:
            return None
        try:
            return str(Path(src).resolve()), int(m["page"])
        except (TypeError, ValueError, OSError):
            return None

    def _image_docs_for_text_pages(
        self,
        text_docs: list[Document],
        *,
        query: str | None = None,
        query_embedding: list[float] | None = None,
        max_images: int = 4,
        fetch_k: int = 128,
    ) -> list[Document]:
        """
        文本检索命中页上的图表向量往往排在全局 top-k 之后（文本与 query 更像）。
        在足够大的 fetch_k 内按 query 相似度筛出图片，再只保留「当前文本命中页」上的图。
        """
        page_keys: set[tuple[str, int]] = set()
        for d in text_docs:
            pk = self._doc_page_key(d)
            if pk:
                page_keys.add(pk)
        if not page_keys:
            return []

        db = VectorDB(embeddings=self.get_embeddings())
        cap = max(max_images * 4, 16)
        try:
            if query_embedding is not None:
                candidates = db.search_by_vector(
                    query_embedding,
                    k=cap,
                    filter_={"type": "image"},
                    fetch_k=fetch_k,
                )
            elif query:
                candidates = db.search(
                    query,
                    k=cap,
                    filter_={"type": "image"},
                    fetch_k=fetch_k,
                )
            else:
                return []
        except Exception:
            return []

        out: list[Document] = []
        seen_uri: set[str] = set()
        for d in candidates:
            m = d.metadata or {}
            src, page = m.get("source"), m.get("page")
            if not src or page is None:
                continue
            try:
                key = (str(Path(src).resolve()), int(page))
            except (TypeError, ValueError, OSError):
                continue
            if key not in page_keys:
                continue
            uri = m.get("image_data")
            if isinstance(uri, str) and uri in seen_uri:
                continue
            if isinstance(uri, str):
                seen_uri.add(uri)
            out.append(d)
            if len(out) >= max_images:
                break
        return out

    def retrieve_context(
        self,
        query: str,
        k: int = 4,
        enhance_strategy: str = "query2doc",
    ) -> tuple[str, list[dict[str, Any]]]:
        strategy_name = (enhance_strategy or "query2doc").strip().lower()
        strategy = get_strategy(
            strategy_name,
            llm=self.get_text_llm_for_strategy(),
            base_embeddings=self.get_embeddings() if strategy_name == "hyde" else None,
        )
        inp = strategy.get_retrieval_input(query)
        if inp.embedding is not None:
            print("search_by_vector (HyDE)")
            docs = self._search_docs_by_vector(inp.embedding, k=k)
            extra_images = self._image_docs_for_text_pages(
                docs,
                query_embedding=inp.embedding,
            )
        else:
            print("search_query:", inp.text)
            docs = self._search_docs(inp.text or query, k=k)
            extra_images = self._image_docs_for_text_pages(
                docs,
                query=inp.text or query,
            )
        if extra_images:
            docs = list[Document](docs) + extra_images

        prompt_context = self.prompt_context_from_docs(docs)
        return prompt_context, self.serialize_context_docs(docs)

    @staticmethod
    def message_content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            return "".join(parts)
        return str(content or "")

    @staticmethod
    def _rag_graph_invoke_input(
        query: str,
        k: int,
        *,
        stream: bool = False,
    ) -> RagState:
        """LangGraph 入口 state：须含 query，供 prepare / contextualize / generate 等节点使用。"""
        out: RagState = {
            "messages": [HumanMessage(content=query)],
            "k": k,
            "max_retries": 1,
            "query": query,
        }
        if stream:
            out["stream"] = True
        return out

    def ask_once(
        self,
        query: str,
        k: int = 4,
        trace: bool = False,
        *,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        tid = thread_id or uuid.uuid4().hex
        cfg: RunnableConfig = {"configurable": {"thread_id": tid}}
        result = self._graph.invoke(
            self._rag_graph_invoke_input(query, k),
            config=cfg,
        )
        contexts = result.get("contexts", [])
        out = {
            "answer": result.get("answer", ""),
            "contexts": contexts,
            "sessionId": tid,
        }
        if trace:
            out["trace"] = self._build_trace(result)
        return out

    @staticmethod
    def _build_trace(result: dict[str, Any]) -> dict[str, Any]:
        contexts = result.get("contexts") or []
        retrieval_embedding = result.get("retrieval_embedding")
        retrieval_mode = "vector" if retrieval_embedding is not None else "text"
        return {
            "retrieval_mode": retrieval_mode,
            "retrieval_text": result.get("retrieval_text", ""),
            "retrieval_query": (result.get("retrieval_query") or "").strip(),
            "contexts_count": len(contexts),
            "image_refs_count": len(QwenRagService.extract_image_refs(contexts)),
            "confidence": float(result.get("confidence", 0.0)),
            "retry_count": int(result.get("retry_count", 0)),
            "max_retries": int(result.get("max_retries", 1)),
            "refusal_reason": result.get("refusal_reason", ""),
            "needs_retry": bool(result.get("needs_retry", False)),
        }

    def ask_stream(
        self,
        query: str,
        k: int = 4,
        trace: bool = False,
        *,
        thread_id: str | None = None,
    ) -> tuple[list[dict[str, Any]], Iterator[Any], dict[str, Any]]:
        """
        返回 (检索上下文列表, 模型输出文本片段迭代器, meta)。
        meta 含 sessionId；若 trace 为 True 则另含 trace 字段。
        """
        tid = thread_id or uuid.uuid4().hex
        compiled = self._graph
        cfg: RunnableConfig = {"configurable": {"thread_id": tid}}
        result = compiled.invoke(
            self._rag_graph_invoke_input(query, k, stream=True),
            config=cfg,
        )
        contexts = result.get("contexts", [])
        messages = result.get("llm_messages", [])
        meta: dict[str, Any] = {"sessionId": tid}
        if trace:
            meta["trace"] = self._build_trace(result)
        print("images:", len(self.extract_image_refs(contexts)))
        refusal_answer = (result.get("answer") or "").strip()
        gen = self._llm.stream(messages) if messages else None

        def text_iter() -> Iterator[Any]:
            parts: list[str] = []
            if refusal_answer:
                yield refusal_answer
                compiled.update_state(
                    cfg,
                    {"messages": [AIMessage(content=refusal_answer)]},
                )
                return
            if gen is None:
                return
            while True:
                try:
                    chunk = next(gen)
                    text = self._llm.chunk_to_text(chunk)
                    if text:
                        parts.append(text)
                        yield text
                except StopIteration:
                    full = "".join(parts)
                    if full.strip():
                        compiled.update_state(
                            cfg,
                            {"messages": [AIMessage(content=full)]},
                        )
                    return
                except Exception:
                    traceback.print_exc()
                    full = "".join(parts)
                    if full.strip():
                        compiled.update_state(
                            cfg,
                            {"messages": [AIMessage(content=full)]},
                        )
                    return

        return contexts, text_iter(), meta
