"""
DashScope（百炼）对话客户端：ChatTongyi 实例、多模态消息适配、按消息选择文本 / VL 模型并调用。
"""
from __future__ import annotations

import os
from typing import Any, Iterator

from langchain_community.chat_models import ChatTongyi
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage


def configure_dashscope_sdk() -> None:
    """与百炼文档一致：设置 base_http_api_url；api_key 来自环境变量 DASHSCOPE_API_KEY。"""
    import dashscope

    base = (
        os.getenv("DASHSCOPE_BASE_HTTP_API_URL") or "https://dashscope.aliyuncs.com/api/v1"
    ).strip()
    dashscope.base_http_api_url = base


configure_dashscope_sdk()

_MULTIMODAL_CONVERSATION_MODELS = frozenset({"qwen3.5-plus"})
_DEFAULT_TEXT_MODEL = "qwen-plus"
_DEFAULT_VL_MODEL = "qwen-plus"


def _ensure_multimodal_conversation_client(llm: BaseChatModel) -> None:
    import dashscope

    name = getattr(llm, "model_name", None) or getattr(llm, "model", "")
    if isinstance(name, str) and name.strip() in _MULTIMODAL_CONVERSATION_MODELS:
        llm.client = dashscope.MultiModalConversation


class DashScopeLLMClient:
    """管理文本与多模态 ChatTongyi 实例，并统一 invoke / stream 入口。"""

    def __init__(self) -> None:
        self._llm: BaseChatModel | None = None
        self._llm_vl: BaseChatModel | None = None

    def get_text_model(self) -> BaseChatModel:
        if self._llm is None:
            model = (os.getenv("DASHSCOPE_CHAT_MODEL") or _DEFAULT_TEXT_MODEL).strip()
            print(f"Using model: {model}")
            self._llm = ChatTongyi(
                model=model,
                model_kwargs={"temperature": 0.7},
            )
        return self._llm

    def get_vl_model(self) -> BaseChatModel:
        if self._llm_vl is None:
            model = (os.getenv("DASHSCOPE_VL_MODEL") or _DEFAULT_VL_MODEL).strip()
            self._llm_vl = ChatTongyi(
                model=model,
                model_kwargs={"temperature": 0.7},
            )
            _ensure_multimodal_conversation_client(self._llm_vl)
        return self._llm_vl

    @staticmethod
    def human_message_has_image_url(msg: BaseMessage) -> bool:
        if not isinstance(msg, HumanMessage):
            return False
        c = msg.content
        if not isinstance(c, list):
            return False
        for block in c:
            if isinstance(block, dict) and block.get("type") == "image_url":
                return True
        return False

    @classmethod
    def messages_need_vl(cls, messages: list[BaseMessage]) -> bool:
        return any(cls.human_message_has_image_url(m) for m in messages)

    @staticmethod
    def _openai_blocks_to_dashscope_multimodal(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
        texts: list[str] = []
        images: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                texts.append(str(block.get("text", "")))
            elif block.get("type") == "image_url":
                url = (block.get("image_url") or {}).get("url")
                if isinstance(url, str) and url.strip():
                    images.append(url.strip())
        parts: list[dict[str, Any]] = [{"image": u} for u in images]
        body = "".join(texts)
        if body:
            parts.append({"text": body})
        return parts

    @classmethod
    def adapt_messages_for_dashscope_vl(cls, messages: list[BaseMessage]) -> list[BaseMessage]:
        out: list[BaseMessage] = []
        for m in messages:
            if isinstance(m, HumanMessage) and isinstance(m.content, list):
                out.append(
                    HumanMessage(content=cls._openai_blocks_to_dashscope_multimodal(m.content))
                )
            else:
                out.append(m)
        return out

    def model_for_messages(self, messages: list[BaseMessage]) -> BaseChatModel:
        if self.messages_need_vl(messages):
            return self.get_vl_model()
        return self.get_text_model()

    def prepare_messages(self, messages: list[BaseMessage]) -> tuple[BaseChatModel, list[BaseMessage]]:
        to_send = (
            self.adapt_messages_for_dashscope_vl(messages)
            if self.messages_need_vl(messages)
            else messages
        )
        return self.model_for_messages(messages), to_send

    @staticmethod
    def _raise_with_better_hint(e: Exception, llm: BaseChatModel) -> None:
        msg = str(e)
        if "InvalidParameter" in msg and "url error" in msg:
            model_name = getattr(llm, "model_name", None) or getattr(llm, "model", "unknown")
            raise ValueError(
                f"{msg}\n"
                "提示：当前 DashScope 模型可能与 ChatTongyi 的调用接口不兼容。"
                f"当前模型: {model_name}。"
                "请尝试将 DASHSCOPE_CHAT_MODEL / DASHSCOPE_VL_MODEL 设置为 qwen-plus 或 qwen-max。"
            ) from e
        raise e

    def invoke(self, messages: list[BaseMessage]) -> BaseMessage:
        llm, to_send = self.prepare_messages(messages)
        try:
            return llm.invoke(to_send)
        except Exception as e:
            self._raise_with_better_hint(e, llm)

    def stream(self, messages: list[BaseMessage]) -> Iterator[BaseMessage]:
        llm, to_send = self.prepare_messages(messages)
        try:
            return llm.stream(to_send)
        except Exception as e:
            self._raise_with_better_hint(e, llm)

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    t = item.get("type")
                    if t == "text":
                        parts.append(str(item.get("text", "")))
                    elif "text" in item and item.get("text") is not None:
                        parts.append(str(item.get("text")))
            return "".join(parts)
        if isinstance(content, dict):
            # 兼容部分 SDK 在增量里返回 {"text": "..."} 的情况
            if content.get("type") == "text":
                return str(content.get("text", ""))
            if content.get("text") is not None:
                return str(content.get("text"))
        return ""

    @classmethod
    def chunk_to_text(cls, chunk: Any) -> str:
        """
        统一解析流式 chunk 文本，兼容：
        1) chunk.content（str / list[dict]）
        2) chunk.additional_kwargs.reasoning_content（Qwen 推理流常见）
        """
        text = cls._content_to_text(getattr(chunk, "content", None))
        if text:
            return text

        ak = getattr(chunk, "additional_kwargs", None)
        if isinstance(ak, dict):
            for key in ("reasoning_content", "content", "text", "output_text", "delta"):
                value = ak.get(key)
                parsed = cls._content_to_text(value)
                if parsed:
                    return parsed
                if isinstance(value, str) and value:
                    return value
        return ""
