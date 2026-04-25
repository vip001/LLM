"""
向量数据库操作类，提供增删改查方法，供 RAG 检索使用。

封装 LangChain FAISS 向量库，支持：
- add: 新增文档
- delete: 按 ID 删除文档
- update: 更新文档（先删后加）
- search: 相似度检索（RAG 查询）

RAG 使用示例：
    from vector_db import VectorDB
    from langchain_core.documents import Document

    db = VectorDB()  # 加载已有 chromastore

    # 方式一：直接检索
    docs = db.search("用户问题", k=4)

    # 方式二：接入 LangChain RAG 链
    retriever = db.as_retriever(search_kwargs={"k": 4})
    # chain = create_retrieval_chain(retriever, llm_chain)
"""
import time
from pathlib import Path
from typing import Any
import os
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

try:
    from ollama import ResponseError as OllamaResponseError
except ImportError:
    OllamaResponseError = None  # type: ignore[misc, assignment]

# 默认配置（与 embedding_provider 默认模型一致，供未传入 embeddings 时使用）
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_STORE_DIR = PROJECT_ROOT / "chromastore"
DEFAULT_EMBED_MODEL = "fervent_mcclintock/Qwen3-VL-Embedding-2B:F16"
DEFAULT_BASE_URL = "http://localhost:11434"


class VectorDB:
    """向量数据库操作类，提供增删改查接口供 RAG 使用。"""

    def __init__(
        self,
        store_path: str | Path = DEFAULT_STORE_DIR,
        embeddings: Embeddings | None = None,
        embed_model: str = DEFAULT_EMBED_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        force_new: bool = False,
    ) -> None:
        """
        初始化向量数据库。

        Args:
            store_path: 持久化目录路径
            embeddings: 嵌入模型，若不传则使用 OllamaEmbeddings
            embed_model: Ollama 嵌入模型名（仅当 embeddings 为 None 时生效）
            base_url: Ollama API 地址（仅当 embeddings 为 None 时生效）
            force_new: 若为 True，即使本地有数据也创建新库（不加载已有数据）
        """
        self.store_path = Path(store_path)
        self.embeddings = embeddings or OllamaEmbeddings(
            model=embed_model,
            base_url=base_url,
        )
        self.force_new = force_new
        self._store: FAISS | None = None

    # 502 通常来自 Ollama 嵌入服务未就绪或过载，重试次数与间隔
    _OLLAMA_RETRY_TIMES = 3
    _OLLAMA_RETRY_DELAYS = (1.0, 2.0, 3.0)  # 秒

    def _is_ollama_502(self, e: BaseException) -> bool:
        """判断是否为 Ollama 返回的 502 或等价错误。"""
        if OllamaResponseError is not None and isinstance(e, OllamaResponseError):
            return getattr(e, "status_code", None) == 502
        return "502" in str(e)

    def _call_with_ollama_retry(self, fn, *args, **kwargs):
        """对依赖 Ollama 嵌入的调用做重试，失败时抛出带排查提示的异常。"""
        last_error = None
        for attempt in range(self._OLLAMA_RETRY_TIMES):
            try:
                return fn(*args, **kwargs)
            except BaseException as e:
                last_error = e
                if not self._is_ollama_502(e) or attempt == self._OLLAMA_RETRY_TIMES - 1:
                    break
            time.sleep(self._OLLAMA_RETRY_DELAYS[attempt])
        msg = (
            "Ollama 嵌入服务异常 (502)。请检查：\n"
            "  1. Ollama 是否在运行：终端执行 ollama list\n"
            "  2. 是否已拉取嵌入模型：ollama pull fervent_mcclintock/Qwen3-VL-Embedding-2B:F16\n"
            "  3. 若仍报错，可重启 Ollama 后再试。"
        )
        raise RuntimeError(msg) from last_error

    def _ensure_loaded(self) -> FAISS:
        """确保向量库已加载，若存在则从本地加载，否则返回空实例。"""
        if self._store is not None:
            return self._store

        path = self.store_path
        if (
            not self.force_new
            and path.exists()
            and (path / "index.faiss").exists()
        ):
            self._store = FAISS.load_local(
                str(path),
                self.embeddings,
                index_name="index",
                allow_dangerous_deserialization=True,
            )
        else:
            self._store = FAISS.from_texts(
                ["__placeholder__"],
                self.embeddings,
                metadatas=[{}],
                ids=["__placeholder__"],
            )
            self._store.delete(ids=["__placeholder__"])
        return self._store

    def add(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
    ) -> list[str]:
        """
        新增文档到向量库。

        Args:
            documents: LangChain Document 列表
            ids: 可选的自定义 ID 列表，长度需与 documents 一致

        Returns:
            新增文档的 ID 列表
        """
        store = self._ensure_loaded()
        return self._call_with_ollama_retry(
            store.add_documents,
            documents,
            ids=ids,
        )

    def add_embeddings(
        self,
        text_embeddings: list[tuple[str, list[float]]],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """
        将已计算好的 (文本, 向量) 对加入向量库，用于图片等多模态内容（先外部算向量再写入）。

        Args:
            text_embeddings: [(page_content, embedding), ...]，page_content 用于检索结果展示
            metadatas: 与 text_embeddings 等长的元数据列表
            ids: 可选 ID 列表

        Returns:
            写入后的 ID 列表
        """
        store = self._ensure_loaded()
        return self._call_with_ollama_retry(
            store.add_embeddings,
            text_embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    def delete(self, ids: list[str]) -> bool:
        """
        按 ID 删除文档。

        Args:
            ids: 要删除的文档 ID 列表

        Returns:
            删除是否成功
        """
        store = self._ensure_loaded()
        result = store.delete(ids=ids)
        return result is True

    def update(
        self,
        document_id: str,
        document: Document,
    ) -> str:
        """
        更新文档（先删后加）。

        Args:
            document_id: 要更新的文档 ID
            document: 新的文档内容

        Returns:
            更新后的文档 ID（可能与原 ID 相同）
        """
        self.delete(ids=[document_id])
        new_ids = self.add([document], ids=[document_id])
        return new_ids[0]

    def update_batch(
        self,
        updates: list[tuple[str, Document]],
    ) -> list[str]:
        """
        批量更新文档。

        Args:
            updates: (document_id, document) 元组列表

        Returns:
            更新后的文档 ID 列表
        """
        ids_to_delete = [uid for uid, _ in updates]
        self.delete(ids=ids_to_delete)
        docs = [doc for _, doc in updates]
        ids = [uid for uid, _ in updates]
        return self.add(docs, ids=ids)

    def search(
        self,
        query: str,
        k: int = 4,
        filter_: dict[str, Any] | None = None,
        fetch_k: int = 20,
    ) -> list[Document]:
        """
        相似度检索，用于 RAG 查询。

        Args:
            query: 查询文本
            k: 返回的文档数量
            filter_: 元数据过滤条件，如 {"source": "doc1.pdf"}
            fetch_k: MMR 搜索时的预取数量

        Returns:
            与查询最相关的 Document 列表
        """
        store = self._ensure_loaded()
        return self._call_with_ollama_retry(
            store.similarity_search,
            query,
            k=k,
            filter=filter_,
            fetch_k=fetch_k,
        )

    def search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter_: dict[str, Any] | None = None,
        fetch_k: int = 20,
    ) -> list[Document]:
        """
        按向量相似度检索，用于 HyDE 等已预计算 embedding 的场景。

        Args:
            embedding: 查询向量
            k: 返回的文档数量
            filter_: 元数据过滤条件
            fetch_k: 预取数量

        Returns:
            与向量最相关的 Document 列表
        """
        store = self._ensure_loaded()
        return self._call_with_ollama_retry(
            store.similarity_search_by_vector,
            embedding,
            k=k,
            filter=filter_,
            fetch_k=fetch_k,
        )

    def search_with_score(
        self,
        query: str,
        k: int = 4,
        filter_: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """
        相似度检索并返回分数。

        Args:
            query: 查询文本
            k: 返回的文档数量
            filter_: 元数据过滤条件

        Returns:
            (Document, score) 元组列表
        """
        store = self._ensure_loaded()
        return self._call_with_ollama_retry(
            store.similarity_search_with_score,
            query,
            k=k,
            filter=filter_,
        )

    def search_by_vector_with_score(
        self,
        embedding: list[float],
        k: int = 4,
        filter_: dict[str, Any] | None = None,
        fetch_k: int = 20,
    ) -> list[tuple[Document, float]]:
        """
        按向量相似度检索并返回分数（用于 HyDE 等场景的阈值过滤）。

        Args:
            embedding: 查询向量
            k: 返回的文档数量
            filter_: 元数据过滤条件
            fetch_k: 预取数量

        Returns:
            (Document, score) 元组列表
        """
        store = self._ensure_loaded()
        return self._call_with_ollama_retry(
            store.similarity_search_with_score_by_vector,
            embedding,
            k=k,
            filter=filter_,
            fetch_k=fetch_k,
        )

    def save(self, path: str | Path | None = None) -> None:
        """
        持久化向量库到本地。

        Args:
            path: 保存路径，默认使用初始化时的 store_path
        """
        store = self._ensure_loaded()
        target = Path(path) if path is not None else self.store_path
        target = target.resolve()
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise RuntimeError(
                f"无法创建向量库目录: {target}，请检查挂载路径与写权限。原始错误: {e}"
            ) from e

        if not target.exists() or not target.is_dir():
            raise RuntimeError(f"向量库存储路径不可用: {target}（不是有效目录）")

        try:
            store.save_local(str(target), index_name="index")
        except RuntimeError as e:
            raise RuntimeError(
                f"写入 FAISS 失败，目标目录: {target}。"
                "请检查该目录是否真实存在、已正确挂载并具有写权限。"
                f"原始错误: {e}"
            ) from e

    def as_retriever(self, **kwargs: Any):
        """
        获取 LangChain Retriever，可直接用于 RAG 链。

        Args:
            **kwargs: 传入 as_retriever 的参数，如 search_type="mmr", search_kwargs={"k": 4}

        Returns:
            LangChain Retriever
        """
        store = self._ensure_loaded()
        return store.as_retriever(**kwargs)
