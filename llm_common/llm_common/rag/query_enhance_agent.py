"""
使用 LangGraph 预置的 tool-calling agent，在 Query2Doc 与 HyDE 两种查询增强之间由模型择一调用。

说明：LangGraph 当前提供的预置入口为 ``create_react_agent``；若未来版本导出 ``create_agent``，
则优先使用 ``create_agent``，否则回退到 ``create_react_agent``（二者在本场景等价）。
直接构造 ``Query2DocStrategy`` / ``HyDEStrategy`` 实例并封装为工具。
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool

try:
    from langgraph.prebuilt import create_agent as _create_agent  # type: ignore[attr-defined, unused-ignore]
except ImportError:  # LangGraph 0.6.x 等版本仅有 create_react_agent
    from langgraph.prebuilt import create_react_agent as _create_agent

from llm_common.rag.query_enhance import HyDEStrategy, Query2DocStrategy, RetrievalInput

if TYPE_CHECKING:
    from llm_common.rag.qwen_rag_service import QwenRagService


def _tool_result_to_retrieval_input(messages: list[Any]) -> RetrievalInput | None:
    for m in reversed(messages):
        if not isinstance(m, ToolMessage):
            continue
        raw = m.content
        if isinstance(raw, dict):
            data = raw
        else:
            if not isinstance(raw, str) or not raw.strip():
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
        kind = data.get("kind")
        if kind == "text":
            text = data.get("retrieval_text")
            if isinstance(text, str) and text.strip():
                return RetrievalInput(text=text.strip())
        elif kind == "vector":
            emb = data.get("embedding")
            if isinstance(emb, list) and emb:
                return RetrievalInput(embedding=[float(x) for x in emb])
    return None


_AGENT_SYSTEM_PROMPT = """你是 RAG 管道里的「查询增强」调度器。用户会提供一条需要从知识库检索的用户问题。

你必须只通过调用**一个**工具来完成增强（二选一），不要自己编造检索用的文本或向量：
- enhance_with_query2doc：生成假设文档片段并与原问题拼接，走**文本检索**（嵌入在检索端完成）。适合术语解释、步骤、API/配置名明确、偏「可关键词命中」的问题。
- enhance_with_hyde：生成假设答案再嵌入并与原问题向量平均，走**向量检索**。适合表述模糊、口语化、与文档措辞差异大、更依赖语义近邻召回的问题。

规则：恰好调用其中一个工具一次即可；不要同时调用两个工具；不要输出 JSON 或增强结果正文，只通过工具返回数据。"""


class QueryEnhanceToolAgent:
    """
    将 Query2Doc 与 HyDE 注册为工具，由 LLM 通过预置 agent 图择一调用，得到 ``RetrievalInput``。
    """

    def __init__(
        self,
        llm: BaseChatModel,
        base_embeddings: Embeddings,
        *,
        recursion_limit: int = 6,
    ) -> None:
        self._recursion_limit = recursion_limit
        q2d = Query2DocStrategy(llm)
        hyde = HyDEStrategy(llm, base_embeddings)

        @tool(return_direct=True)
        def enhance_with_query2doc(user_query: str) -> str:
            """Query2Doc：假设文档片段 + 原问题拼接，用于文本检索。适合表述清晰、可关键词命中的问题。"""
            inp = q2d.get_retrieval_input(user_query.strip())
            text = inp.text or user_query.strip()
            return json.dumps(
                {"kind": "text", "retrieval_text": text, "chosen": "query2doc"},
                ensure_ascii=False,
            )

        @tool(return_direct=True)
        def enhance_with_hyde(user_query: str) -> str:
            """HyDE：假设答案嵌入与原问题向量平均，用于向量检索。适合语义模糊、需近邻召回的问题。"""
            inp = hyde.get_retrieval_input(user_query.strip())
            if inp.embedding is None:
                raise RuntimeError("HyDE 工具未得到有效 embedding")
            return json.dumps(
                {
                    "kind": "vector",
                    "embedding": inp.embedding,
                    "chosen": "hyde",
                },
                ensure_ascii=False,
            )

        self._graph = _create_agent(
            llm,
            [enhance_with_query2doc, enhance_with_hyde],
            prompt=_AGENT_SYSTEM_PROMPT,
        )

    @classmethod
    def from_rag_service(cls, service: QwenRagService, **kwargs: Any) -> QueryEnhanceToolAgent:
        return cls(
            service.get_text_llm_for_strategy(),
            service.get_embeddings(),
            **kwargs,
        )

    def get_retrieval_input(self, user_query: str) -> RetrievalInput:
        """
        运行 agent，解析最后一次工具返回，得到 ``RetrievalInput``。
        若未解析到工具结果，则退化为对原问题做文本检索（``RetrievalInput(text=query)``）。
        """
        q = (user_query or "").strip()
        if not q:
            return RetrievalInput(text="")

        user_turn = (
            "请为下面的用户问题选择唯一的查询增强方式，并调用对应工具完成增强。\n\n"
            f"用户问题：\n{q}"
        )
        result = self._graph.invoke(
            {"messages": [("user", user_turn)]},
            config={"recursion_limit": self._recursion_limit},
        )
        messages = result.get("messages") or []
        parsed = _tool_result_to_retrieval_input(messages)
        if parsed is not None:
            return parsed
        return RetrievalInput(text=q)
