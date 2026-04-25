from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from query_enhance import get_strategy

if TYPE_CHECKING:
    from qwen_rag_service import QwenRagService


class RagState(TypedDict, total=False):
    query: str
    k: int
    enhance_strategy: str
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


def build_rag_graph(service: "QwenRagService"):
    def prepare_node(state: RagState) -> RagState:
        return {
            "query": (state.get("query") or "").strip(),
            "k": int(state.get("k", 4)),
            "enhance_strategy": (state.get("enhance_strategy") or "query2doc").strip().lower(),
            "max_retries": int(state.get("max_retries", 1)),
            "retry_count": int(state.get("retry_count", 0)),
            "stream": bool(state.get("stream", False)),
            "needs_retry": False,
            "confidence": 0.0,
            "refusal_reason": "",
        }

    def enhance_query_node(state: RagState) -> RagState:
        strategy_name = state.get("enhance_strategy") or "query2doc"
        strategy = get_strategy(
            strategy_name,
            llm=service.get_text_llm_for_strategy(),
            base_embeddings=service.get_embeddings() if strategy_name == "hyde" else None,
        )
        inp = strategy.get_retrieval_input(state["query"])
        return {
            "retrieval_text": inp.text or state["query"],
            "retrieval_embedding": inp.embedding,
        }

    def retrieve_node(state: RagState) -> RagState:
        k = state.get("k", 4)
        retrieval_embedding = state.get("retrieval_embedding")
        retrieval_text = state.get("retrieval_text") or state["query"]

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
        messages = service._build_llm_messages(
            state["query"],
            state.get("prompt_context", "（未检索到相关文档）"),
            state.get("contexts", []),
        )
        if state.get("stream", False):
            return {"llm_messages": messages}
        result = service._llm.invoke(messages)
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
        fallback_strategy = state.get("enhance_strategy") or "query2doc"
        if fallback_strategy == "hyde":
            fallback_strategy = "query2doc"
        return {
            "retry_count": retry_count,
            "enhance_strategy": fallback_strategy,
            "needs_retry": False,
        }

    def finalize_node(state: RagState) -> RagState:
        if state.get("answer"):
            return {}
        if state.get("stream", False) and not state.get("refusal_reason"):
            # Streaming mode emits answer chunks outside graph state; avoid premature fallback.
            return {}
        reason = state.get("refusal_reason") or "当前信息不足，无法给出可靠回答。"
        return {"answer": reason}

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
    graph.add_node("enhance_query", enhance_query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("retrieval_guard", retrieval_guard_node)
    graph.add_node("generate", generate_node)
    graph.add_node("self_check", self_check_node)
    graph.add_node("retry", retry_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "enhance_query")
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
    return graph.compile()
