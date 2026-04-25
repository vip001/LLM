"""
嵌入模型工具类：按名称获取与向量库共用的 Embeddings 实例（带缓存）。
支持 DashScope 多模态向量与 Ollama 嵌入模型（如 bge-m3、Qwen3-VL-Embedding-2B）。

使用 fervent_mcclintock/Qwen3-VL-Embedding-2B:F16 做向量查询：
  1. 确保 Ollama 已拉取该模型：ollama pull fervent_mcclintock/Qwen3-VL-Embedding-2B:F16
  2. 确认模型支持 /api/embed：运行 python server/check_ollama_embed.py 查看本机模型及 embed 是否可用
  3. 若 400：部分 VL 模型在 Ollama 中仅打包为 chat，不支持 embed，可改用纯文本 embedding 模型：
     ollama pull qwen3-embedding 或 ollama pull nomic-ai/text-embed-v1.5，
     并把下方 EMBED_MODEL_QWEN 改为对应模型名（如 qwen3-embedding）。
"""

from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# 从项目根目录加载 .env（含 DASHSCOPE_API_KEY 等）
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import httpx
from langchain_core.embeddings import Embeddings

from vector_db import DEFAULT_BASE_URL


# 默认嵌入模型名：Ollama 上的 Qwen3-VL-Embedding-2B（支持文本与多模态，可配置 dimensions）
EMBED_MODEL_QWEN = "fervent_mcclintock/Qwen3-VL-Embedding-2B:F16"
# DashScope 多模态向量模型（需 DASHSCOPE_API_KEY）
EMBED_MODEL_DASHSCOPE = "tongyi-embedding-vision-plus"
# Qwen3-VL 支持 64–2048 维；设为 None 则不传 dimensions（部分部署下传 dimensions 会 400）
QWEN_EMBED_DIMENSIONS: int | None = None

DEFAULT_EMBED_MODEL_NAME = EMBED_MODEL_DASHSCOPE

class DirectOllamaEmbeddings(Embeddings):
    """
    使用 httpx 直连 Ollama /api/embed，不经过 ollama 包，避免被 OLLAMA_HOST/代理劫持。
    请求始终发往传入的 base_url。
    """

    def __init__(
        self,
        model: str,
        base_url: str = DEFAULT_BASE_URL,
        dimensions: int | None = None,
        timeout: float = 60.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.dimensions = dimensions
        self.timeout = timeout

    def _request(self, texts: list[str]) -> list[list[float]]:
        url = f"{self.base_url}/api/embed"
        payload = {
            "model": self.model,
            "input": texts[0] if len(texts) == 1 else texts,
            "truncate": True,
        }
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        with httpx.Client(timeout=self.timeout, trust_env=False) as client:
            r = client.post(url, json=payload)
            if r.status_code != 200:
                err_detail = r.text
                try:
                    err_detail = r.json()
                except Exception:
                    pass
                raise RuntimeError(
                    f"Ollama /api/embed 请求失败 status={r.status_code} model={self.model!r} "
                    f"base_url={self.base_url}\n响应: {err_detail}"
                ) from None
            data = r.json()
        return data["embeddings"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._request(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._request([text])[0]


class DashScopeMultimodalEmbeddings(Embeddings):
    """
    使用阿里云百炼 DashScope SDK 的 Qwen 多模态向量模型（tongyi-embedding-vision-plus）。
    支持文本、图片 URL/Base64；支持「文本+图片」融合向量，便于图文联合检索。
    API Key 从环境变量 DASHSCOPE_API_KEY 读取（已由 load_dotenv 加载 .env）。
    """

    def __init__(self, model: str = EMBED_MODEL_DASHSCOPE, dimension: int | None = None):
        self.model = model
        self.dimension = dimension  # tongyi-embedding-vision-plus 支持 64,128,256,512,1024,1152

    def _call_sdk(self, texts: list[str]) -> list[list[float]]:
        from http import HTTPStatus
        import dashscope
        input_list = [{"text": t} for t in texts]
        kwargs = {"model": self.model, "input": input_list}
        if self.dimension is not None:
            kwargs["parameters"] = {"dimension": self.dimension}
        resp = dashscope.MultiModalEmbedding.call(**kwargs)
        if resp.status_code != HTTPStatus.OK:
            raise RuntimeError(
                f"DashScope MultiModalEmbedding 调用失败 status_code={resp.status_code} "
                f"code={getattr(resp, 'code', '')} message={getattr(resp, 'message', '')}"
            )
        emb_list = getattr(resp.output, "embeddings", None) or (resp.output.get("embeddings", []) if isinstance(resp.output, dict) else [])
        def _idx_emb(e, i):
            if isinstance(e, dict):
                return e.get("index", i), e.get("embedding", [])
            return getattr(e, "index", i), getattr(e, "embedding", [])
        by_index = dict[Any | int, Any | list[Any]](_idx_emb(e, i) for i, e in enumerate(emb_list))
        return [by_index[i] for i in range(len(texts))]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._call_sdk(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._call_sdk([text])[0]

    def embed_images_with_context(
        self,
        items: list[tuple[str, str]],
    ) -> list[list[float]]:
        """
        对「文本上下文 + 图片」生成融合向量，便于后续文本检索能命中图片内容。
        items: [(context_text, image_data_uri), ...]，image_data_uri 格式为 data:image/png;base64,xxx 或图片 URL。
        单张图片不超过 3MB；tongyi-embedding-vision-plus 支持 URL 或 Base64。
        """
        if not items:
            return []
        from http import HTTPStatus
        import dashscope
        input_list = []
        for ctx, img in items:
            entry = {"image": img}
            if ctx and ctx.strip():
                entry["text"] = ctx.strip()[:1024]  # 模型限制 1024 tokens，截断
            input_list.append(entry)
        kwargs = {"model": self.model, "input": input_list}
        if self.dimension is not None:
            kwargs["parameters"] = {"dimension": self.dimension}
        resp = dashscope.MultiModalEmbedding.call(**kwargs)
        if resp.status_code != HTTPStatus.OK:
            raise RuntimeError(
                f"DashScope MultiModalEmbedding 图片向量化失败 status_code={resp.status_code} "
                f"code={getattr(resp, 'code', '')} message={getattr(resp, 'message', '')}"
            )
        emb_list = getattr(resp.output, "embeddings", None) or (resp.output.get("embeddings", []) if isinstance(resp.output, dict) else [])
        def _idx_emb(e, i):
            if isinstance(e, dict):
                return e.get("index", i), e.get("embedding", [])
            return getattr(e, "index", i), getattr(e, "embedding", [])
        by_index = dict(_idx_emb(e, i) for i, e in enumerate(emb_list))
        return [by_index[i] for i in range(len(items))]


class EmbeddingProvider:
    """获取与向量库共用的嵌入模型，供 RAG / HyDE 等使用。模型在内部固定为 DEFAULT_EMBED_MODEL_NAME。"""

    def __init__(self, base_url: str = DEFAULT_BASE_URL):
        self.base_url = base_url
        self._cache: dict[str, Embeddings] = {}

    def get_embeddings(self) -> Embeddings:
        """获取嵌入模型实例（带缓存）。使用模块内指定的默认模型（DEFAULT_EMBED_MODEL_NAME）。"""
        model = DEFAULT_EMBED_MODEL_NAME
        if model not in self._cache:
            if model == EMBED_MODEL_DASHSCOPE:
                self._cache[model] = DashScopeMultimodalEmbeddings(model=model)
            elif model == EMBED_MODEL_QWEN:
                self._cache[model] = DirectOllamaEmbeddings(
                    model=model,
                    base_url=self.base_url,
                    dimensions=QWEN_EMBED_DIMENSIONS,
                )
            else:
                self._cache[model] = DirectOllamaEmbeddings(
                    model=model,
                    base_url=self.base_url,
                )
        return self._cache[model]
