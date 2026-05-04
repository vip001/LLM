"""
查询增强策略：在 RAG 检索前对用户 query 进行扩展，提升召回。

策略模式：
- Query2Doc：根据问题生成「假设的文档片段」，与原始 query 拼接成文本后做检索（文本 → 由检索器嵌入再查）。
- HyDE (Hypothetical Document Embeddings)：先用 LLM 根据问题生成「假设答案文档」，
  再对假设文档做嵌入，并与 query 的嵌入取平均。用得到的向量做相似度检索（by vector）。

使用方式：
    from query_enhance import get_strategy, RetrievalInput

    strategy = get_strategy("query2doc", llm=my_llm)
    inp = strategy.get_retrieval_input("MMKV的用法")
    # 若 inp.text 非空则 db.search(inp.text)；若 inp.embedding 非空则 db.search_by_vector(inp.embedding)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate


@dataclass
class RetrievalInput:
    """
    检索输入：要么用文本检索（query2doc），要么用向量检索（hyde）。
    仅 one of text / embedding 应有值。
    """
    text: str | None = None
    embedding: list[float] | None = None

    def __post_init__(self) -> None:
        # 必须且只能设置 text 或 embedding 之一
        if (self.text is not None) == (self.embedding is not None):
            raise ValueError("RetrievalInput 必须且只能设置 text 或 embedding 之一")


class QueryEnhanceStrategy(ABC):
    """查询增强策略抽象基类：根据 query 得到检索输入（文本或向量）。"""

    @abstractmethod
    def get_retrieval_input(self, query: str) -> RetrievalInput:
        """
        根据用户问题得到用于检索的输入（文本或向量）。

        Args:
            query: 用户原始问题

        Returns:
            若为文本策略则 text 有值；若为 HyDE 则 embedding 有值
        """
        pass


class Query2DocStrategy(QueryEnhanceStrategy):
    """
    Query2Doc 策略：让 LLM 根据问题生成一段「假设的文档片段」，
    与原始 query 拼接成文本，由检索端嵌入后做相似度检索。
    """

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm
        self._prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "你是一个文档摘要助手。根据用户的问题，用 1～3 句话写出一段「假设的文档片段」——"
                "就像某份资料里可能回答该问题的一小段文字。只输出这段假设文档，不要解释、不要重复问题。",
            ),
            ("human", "{question}"),
        ])
        self._chain = self._prompt | self._llm

    def get_retrieval_input(self, query: str) -> RetrievalInput:
        try:
            msg = self._chain.invoke({"question": query})
            doc = (getattr(msg, "content", None) or "").strip()
        except Exception as e:
            print(f"Query2Doc error: {e}")
            doc = ""
        text = f"{query}\n{doc}" if doc else query
        print(f"Query2Doc: {text}")
        return RetrievalInput(text=text)


class HyDEStrategy(QueryEnhanceStrategy):
    """
    HyDE (Hypothetical Document Embeddings) 策略：
    1) 用 LLM 根据问题生成「假设的答案文档」；
    2) 用 base_embeddings 对假设文档做嵌入得到 hyde_embedding；
    3) 对原始 query 做嵌入，与 hyde_embedding 取平均作为最终检索向量；
    4) 用该向量做 similarity_search_by_vector，而非文本检索。
    """

    def __init__(
        self,
        llm: BaseChatModel,
        base_embeddings: Embeddings,
    ) -> None:
        self._llm = llm
        self._base_embeddings = base_embeddings
        self._prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "你是一名问答助手，请根据问题简短写出可能出现在文档中的答案内容（1～3 句话）。"
                "只输出这段假设的答案文字，不要重复问题、不要加「答：」等前缀。",
            ),
            ("human", "{question}"),
        ])
        self._chain = self._prompt | self._llm

    def get_retrieval_input(self, query: str) -> RetrievalInput:
        try:
            msg = self._chain.invoke({"question": query})
            hypo_doc = (getattr(msg, "content", None) or "").strip()
        except Exception:
            hypo_doc = ""
        if not hypo_doc:
            # 生成失败时退化为对原始 query 做嵌入
            vec = self._base_embeddings.embed_query(query)
            return RetrievalInput(embedding=vec)

        hyde_embedding = self._base_embeddings.embed_query(hypo_doc)
        query_embedding = self._base_embeddings.embed_query(query)
        try:
            import numpy as np
            result = (np.array(query_embedding) + np.array(hyde_embedding)) / 2.0
            embedding = result.tolist()
        except ImportError:
            embedding = [
                (q + h) / 2.0 for q, h in zip[tuple[float, float]](query_embedding, hyde_embedding)
            ]
        return RetrievalInput(embedding=embedding)


StrategyName = Literal["query2doc", "hyde"]
_DEFAULT_STRATEGY: StrategyName = "query2doc"


def get_strategy(
    name: str | StrategyName = _DEFAULT_STRATEGY,
    llm: BaseChatModel | None = None,
    base_embeddings: Embeddings | None = None,
) -> QueryEnhanceStrategy:
    """
    根据名称返回对应的查询增强策略实例。

    Args:
        name: 策略名称，"query2doc" | "hyde"
        llm: 用于生成假设文档/答案的 LLM，由调用方注入
        base_embeddings: 用于 HyDE 的嵌入模型（对假设文档与 query 做嵌入并取平均），仅 hyde 需要

    Returns:
        查询增强策略实例

    Raises:
        ValueError: name 不合法，或缺少 llm / (hyde 时的 base_embeddings)
    """
    if llm is None:
        raise ValueError("get_strategy 需要传入 llm 参数")
    n = (name or "").strip().lower()
    if n == "query2doc":
        return Query2DocStrategy(llm)
    if n == "hyde":
        if base_embeddings is None:
            raise ValueError("HyDE 策略需要传入 base_embeddings 参数（与向量库使用同一嵌入模型）")
        return HyDEStrategy(llm, base_embeddings)
    raise ValueError(f"不支持的查询增强策略: {name!r}，可选: query2doc, hyde")
