from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, TypedDict, cast

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from llm_common.rag.query_enhance_agent import QueryEnhanceToolAgent

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver

    from llm_common.rag.qwen_rag_service import QwenRagService


class RagState(TypedDict, total=False):
    messages: Annotated[list[Any], add_messages]
    query: str
    retrieval_query: str
    k: int
    max_retries: int
    retry_count: int
    stream: bool
    retrieval_text: str
    retrieval_embedding: list[float] | None

    docs: list[Document]
    contexts: list[dict[str, Any]]
    prompt_context: str
    llm_messages: list[Any]

    answer: str
    confidence: float
    needs_retry: bool
    refusal_reason: str


def build_rag_graph(
    service: "QwenRagService",
    checkpointer: "BaseCheckpointSaver[str] | None" = None,
):
    enhance_agent = QueryEnhanceToolAgent.from_rag_service(service)

    def prepare_node(state: RagState) -> RagState:
        q = (state.get("query") or "").strip()
        return {
            "query": q,
            "k": int(state.get("k", 4)),
            "max_retries": int(state.get("max_retries", 1)),
            "retry_count": int(state.get("retry_count", 0)),
            "stream": bool(state.get("stream", False)),
            "needs_retry": False,
            "confidence": 0.0,
            "refusal_reason": "",
        }

    def contextualize_node(state: RagState) -> RagState:
        msgs = state.get("messages") or []
        q = state.get("query") or ""
        rq = service.contextualize_retrieval_query(msgs, q)
        return {"retrieval_query": rq}

    def enhance_query_node(state: RagState) -> RagState:
        q = (state.get("retrieval_query") or state.get("query") or "").strip()
        inp = enhance_agent.get_retrieval_input(q)
        return {
            "retrieval_text": inp.text or q,
            "retrieval_embedding": inp.embedding,
        }

    def retrieve_node(state: RagState) -> RagState:
        k = state.get("k", 4)
        retrieval_embedding = state.get("retrieval_embedding")
        retrieval_text = state.get("retrieval_text") or (state.get("query") or "")

        if retrieval_embedding is not None:
            docs = service._search_docs_by_vector(retrieval_embedding, k=k)
            extra_images = service._image_docs_for_text_pages(
                docs,
                query_embedding=retrieval_embedding,
            )
        else:
            docs = service._search_docs(retrieval_text, k=k)
            extra_images = service._image_docs_for_text_pages(
                docs,
                query=retrieval_text,
            )

        if extra_images:
            docs = list(docs) + extra_images

        prompt_context = service.prompt_context_from_docs(docs)
        contexts = service.serialize_context_docs(docs)
        return {
            "docs": docs,
            "prompt_context": prompt_context,
            "contexts": contexts,
        }

    def retrieval_guard_node(state: RagState) -> RagState:
        contexts = state.get("contexts") or []
        if contexts:
            return {"refusal_reason": ""}
        return {
            "refusal_reason": "未检索到足够上下文，请补充更具体的问题（如文档名、功能名、报错信息）。",
            "answer": "我暂时无法从知识库中找到相关内容。请补充更具体的信息后再试。",
            "confidence": 0.0,
            "needs_retry": False,
        }

    def generate_node(state: RagState) -> RagState:
        msgs = state.get("messages") or []
        prior: list[BaseMessage] | None = None
        if len(msgs) >= 2:
            prior = []
            for m in msgs[:-1]:
                if isinstance(m, (HumanMessage, AIMessage)):
                    prior.append(m)
            if not prior:
                prior = None
        messages = service._build_llm_messages(
            state.get("query") or "",
            state.get("prompt_context", "（未检索到相关文档）"),
            state.get("contexts", []),
            prior_messages=prior,
        )
        if state.get("stream", False):
            return {"llm_messages": messages}
        result = service._llm.invoke(cast(list[BaseMessage], messages))
        answer = service.message_content_to_text(getattr(result, "content", result))
        return {"llm_messages": messages, "answer": answer}

    def self_check_node(state: RagState) -> RagState:
        answer = (state.get("answer") or "").strip()
        contexts = state.get("contexts") or []
        retry_count = int(state.get("retry_count", 0))
        max_retries = int(state.get("max_retries", 1))

        if not answer:
            needs_retry = retry_count < max_retries
            return {"confidence": 0.0, "needs_retry": needs_retry}

        has_uncertain_phrase = any(
            token in answer for token in ("不确定", "可能", "无法确认", "不知道")
        )
        if not contexts:
            confidence = 0.2
        elif has_uncertain_phrase:
            confidence = 0.5
        else:
            confidence = 0.9

        needs_retry = confidence < 0.4 and retry_count < max_retries
        return {"confidence": confidence, "needs_retry": needs_retry}

    def retry_node(state: RagState) -> RagState:
        retry_count = int(state.get("retry_count", 0)) + 1
        return {
            "retry_count": retry_count,
            "needs_retry": False,
        }

    def finalize_node(state: RagState) -> RagState:
        if state.get("stream", False):
            # 流式：助手回合在 QwenRagService 流结束（或拒答短路）后 update_state，避免与 finalize 重复写入。
            return {}
        ans = (state.get("answer") or "").strip()
        if ans:
            return {"messages": [AIMessage(content=ans)]}
        reason = state.get("refusal_reason") or "当前信息不足，无法给出可靠回答。"
        return {"answer": reason, "messages": [AIMessage(content=reason)]}

    def route_after_retrieval_guard(state: RagState) -> str:
        if state.get("refusal_reason"):
            return "finalize"
        return "generate"

    def route_after_self_check(state: RagState) -> str:
        if state.get("needs_retry"):
            return "retry"
        return "finalize"

    def route_after_generate(state: RagState) -> str:
        if state.get("stream", False):
            return "finalize"
        return "self_check"

    graph = StateGraph(RagState)
    graph.add_node("prepare", prepare_node)
    graph.add_node("contextualize", contextualize_node)
    graph.add_node("enhance_query", enhance_query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("retrieval_guard", retrieval_guard_node)
    graph.add_node("generate", generate_node)
    graph.add_node("self_check", self_check_node)
    graph.add_node("retry", retry_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "contextualize")
    graph.add_edge("contextualize", "enhance_query")
    graph.add_edge("enhance_query", "retrieve")
    graph.add_edge("retrieve", "retrieval_guard")
    graph.add_conditional_edges(
        "retrieval_guard",
        route_after_retrieval_guard,
        {
            "generate": "generate",
            "finalize": "finalize",
        },
    )
    graph.add_conditional_edges(
        "generate",
        route_after_generate,
        {
            "self_check": "self_check",
            "finalize": "finalize",
        },
    )
    graph.add_conditional_edges(
        "self_check",
        route_after_self_check,
        {
            "retry": "retry",
            "finalize": "finalize",
        },
    )
    graph.add_edge("retry", "enhance_query")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)
