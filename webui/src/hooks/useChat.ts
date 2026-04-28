"use client";

import {
  useState,
  useRef,
  useEffect,
  FormEvent,
} from "react";
import type { ChatMessage } from "../types/chat";
import { SIDEBAR_CHATS, WELCOME_TEXT } from "../lib/chat/constants";
import { parseErrorBody } from "../lib/chat/utils";
import { readRagStreamBody } from "../lib/ragStream";

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    { id: "welcome", role: "assistant", content: WELCOME_TEXT },
  ]);
  const [activeSidebarId, setActiveSidebarId] = useState<string>("new");
  const [chatTitle, setChatTitle] = useState("新对话");
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saveHint, setSaveHint] = useState(false);
  const messagesScrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = messagesScrollRef.current;
    if (!el) return;
    el.scrollTo({
      top: el.scrollHeight,
      behavior: loading ? "auto" : "smooth",
    });
  }, [messages, loading]);

  function resetChat() {
    setMessages([{ id: "welcome", role: "assistant", content: WELCOME_TEXT }]);
    setError(null);
    setChatTitle("新对话");
    setActiveSidebarId("new");
  }

  function clearChat() {
    resetChat();
  }

  function saveChat() {
    try {
      const payload = { title: chatTitle, messages, savedAt: Date.now() };
      localStorage.setItem("ai-assistant-chat", JSON.stringify(payload));
      setSaveHint(true);
      window.setTimeout(() => setSaveHint(false), 2000);
    } catch {
      setError("无法保存到本地");
    }
  }

  function selectSidebar(id: string) {
    setActiveSidebarId(id);
    if (id === "new") {
      resetChat();
      return;
    }
    const item = SIDEBAR_CHATS.find((c) => c.id === id);
    if (item) setChatTitle(item.label);
  }

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    const q = query.trim();
    if (!q || loading) return;

    setError(null);

    const userId =
      typeof crypto !== "undefined" && crypto.randomUUID
        ? crypto.randomUUID()
        : `u-${Date.now()}`;
    const asstId =
      typeof crypto !== "undefined" && crypto.randomUUID
        ? crypto.randomUUID()
        : `a-${Date.now()}`;

    const wasOnlyWelcome =
      messages.length === 1 && messages[0]?.id === "welcome";
    if (wasOnlyWelcome) {
      setChatTitle(q.length > 18 ? `${q.slice(0, 18)}…` : q);
    }

    setMessages((m) => {
      const base = m.filter((x) => x.id !== "welcome");
      return [
        ...base,
        { id: userId, role: "user", content: q },
        { id: asstId, role: "assistant", content: "" },
      ];
    });

    setQuery("");
    setLoading(true);

    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, stream: true }),
      });

      if (!res.ok) {
        const raw = await res.text().catch(() => "");
        const d = parseErrorBody(raw);
        const msg = d?.error ?? `请求失败 ${res.status}`;
        const code = d?.code ? ` [${d.code}]` : "";
        const detail =
          d?.detail != null
            ? `\n\n详情: ${JSON.stringify(d.detail, null, 2)}`
            : "";
        setError(`${msg}${code}${detail}`);
        setMessages((m) => m.filter((x) => x.id !== asstId));
        return;
      }

      const reader = res.body?.getReader();
      if (!reader) {
        setError("无法读取响应流");
        setMessages((m) => m.filter((x) => x.id !== asstId));
        return;
      }

      await readRagStreamBody(reader, {
        onRefs: (contexts) => {
          setMessages((m) => {
            const i = m.findIndex((x) => x.id === asstId);
            if (i === -1) return m;
            const copy = [...m];
            copy[i] = { ...copy[i], refs: contexts };
            return copy;
          });
        },
        onAnswerDelta: (delta) => {
          setMessages((m) => {
            const i = m.findIndex((x) => x.id === asstId);
            if (i === -1) return m;
            const copy = [...m];
            copy[i] = { ...copy[i], content: `${copy[i].content}${delta}` };
            return copy;
          });
        },
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "网络错误");
      setMessages((m) => m.filter((x) => x.id !== asstId));
    } finally {
      setLoading(false);
    }
  }

  return {
    messages,
    activeSidebarId,
    chatTitle,
    query,
    setQuery,
    loading,
    error,
    saveHint,
    messagesScrollRef,
    clearChat,
    saveChat,
    selectSidebar,
    onSubmit,
  };
}
