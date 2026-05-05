"""
使用 LangGraph ``create_react_agent`` 与 ``ToolNode``（``handle_tool_errors=True``）构建 tool-calling agent，
在 Query2Doc 与 HyDE 两种查询增强之间由模型择一调用。

直接构造 ``Query2DocStrategy`` / ``HyDEStrategy`` 实例并封装为工具。
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, create_react_agent

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


def _failed_tool_names(messages: list[Any]) -> set[str]:
    """收集本轮对话里执行失败的工具名（``ToolMessage.status == \"error\"``）。"""
    names: set[str] = set()
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        if m.status != "error":
            continue
        name = getattr(m, "name", None)
        if isinstance(name, str) and name:
            names.add(name)
    return names


_AGENT_SYSTEM_PROMPT = """你是 RAG 管道里的「查询增强」调度器。用户会提供一条需要从知识库检索的用户问题。

你必须只通过调用**一个**工具来完成增强（二选一），不要自己编造检索用的文本或向量：
- enhance_with_query2doc：生成假设文档片段并与原问题拼接，走**文本检索**（嵌入在检索端完成）。适合术语解释、步骤、API/配置名明确、偏「可关键词命中」的问题。
- enhance_with_hyde：生成假设答案再嵌入并与原问题向量平均，走**向量检索**。适合表述模糊、口语化、与文档措辞差异大、更依赖语义近邻召回的问题。

规则：恰好调用其中一个工具一次即可；不要同时调用两个工具；不要输出 JSON 或增强结果正文，只通过工具返回数据。
若上一轮工具调用失败（工具结果为错误状态），你必须改调**另一个**工具，禁止再次调用已失败过的那个工具。
若两个工具均已失败，则无法再调用工具；请简短结束回复，不要编造检索用的文本或向量。"""


class QueryEnhanceToolAgent:
    """
    将 Query2Doc 与 HyDE 注册为工具，由 LLM 通过 ``create_react_agent`` + ``ToolNode`` 编译图择一调用，得到 ``RetrievalInput``。
    使用动态 ``bind_tools``：若某工具已产生 ``status=\"error\"`` 的 ``ToolMessage``（如 ``ToolNode`` 在 ``handle_tool_errors`` 下写入），下一轮模型侧不再暴露该工具，从机制上避免重复调用同一失败工具。
    两个工具均失败后不再绑定工具（返回裸 ``llm``），ReAct 在无 ``tool_calls`` 时结束；``get_retrieval_input`` 未解析到成功工具结果时退化为原文文本检索。
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

        @tool(return_direct=False)
        def enhance_with_query2doc(user_query: str) -> str:
            """Query2Doc：假设文档片段 + 原问题拼接，用于文本检索。适合表述清晰、可关键词命中的问题。"""
            inp = q2d.get_retrieval_input(user_query.strip())
            text = inp.text or user_query.strip()
            return json.dumps(
                {"kind": "text", "retrieval_text": text, "chosen": "query2doc"},
                ensure_ascii=False,
            )

        @tool(return_direct=False)
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

        lc_tools = [enhance_with_query2doc, enhance_with_hyde]
        tool_node = ToolNode(lc_tools, handle_tool_errors=True)

        def select_model(state: Any, runtime: Any) -> Any:
            msgs = list(state.get("messages") or [])
            banned = _failed_tool_names(msgs)
            allowed = [t for t in lc_tools if t.name not in banned]
            if not allowed:
                return llm
            return llm.bind_tools(allowed)

        self._graph = create_react_agent(
            select_model,
            tool_node,
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
        若未解析到成功工具结果（含两个工具均失败、模型未调用工具等情况），则退化为对原问题做文本检索（``RetrievalInput(text=query)``）。
        """
        q = (user_query or "").strip()
        if not q:
            return RetrievalInput(text="")

        user_turn = (
            "请为下面的用户问题选择唯一的查询增强方式，并调用对应工具完成增强。\n\n"
            f"用户问题：\n{q}"
        )
        result = self._graph.invoke(
            {"messages": [HumanMessage(content=user_turn)]},
            config={"recursion_limit": self._recursion_limit},
        )
        messages = result.get("messages") or []
        parsed = _tool_result_to_retrieval_input(messages)
        if parsed is not None:
            return parsed
        return RetrievalInput(text=q)
