"use client";

import {
  createContext,
  createElement,
  useContext,
  useState,
  useRef,
  useEffect,
  useCallback,
  FormEvent,
  type ReactNode,
} from "react";
import type { ChatMessage } from "../types/chat";
import { SIDEBAR_CHATS, WELCOME_TEXT } from "../lib/chat/constants";
import { parseErrorBody } from "../lib/chat/utils";
import { readRagStreamBody } from "../lib/ragStream";

type ChatContextValue = ReturnType<typeof useChatState>;

const ChatContext = createContext<ChatContextValue | null>(null);

function useChatState() {
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
  /** 用户是否在底部附近：上翻读历史时为 false，暂停跟滚 */
  const stickToBottomRef = useRef(true);
  const suppressStickFromProgrammaticScrollRef = useRef(false);
  const scrollToBottomRafRef = useRef<number | null>(null);
  const loadingRef = useRef(loading);
  loadingRef.current = loading;

  /** 后端 LangGraph thread，多轮 RAG / 代词消解依赖此 ID 回传 */
  const ragSessionIdRef = useRef<string | null>(null);

  const scheduleScrollToBottom = useCallback(
    (opts?: { layoutOnly?: boolean }) => {
    const el = messagesScrollRef.current;
    if (!el || !stickToBottomRef.current) return;

    if (scrollToBottomRafRef.current != null) {
      cancelAnimationFrame(scrollToBottomRafRef.current);
    }
    scrollToBottomRafRef.current = requestAnimationFrame(() => {
      scrollToBottomRafRef.current = null;
      const target = messagesScrollRef.current;
      if (!target || !stickToBottomRef.current) return;

      const useSmooth =
        !loadingRef.current && !opts?.layoutOnly;
      suppressStickFromProgrammaticScrollRef.current = true;
      target.scrollTo({
        top: target.scrollHeight,
        behavior: useSmooth ? "smooth" : "auto",
      });
      window.setTimeout(
        () => {
          suppressStickFromProgrammaticScrollRef.current = false;
        },
        useSmooth ? 450 : 0,
      );
    });
  },
  [],
);

  const handleMessagesScroll = useCallback(() => {
    if (suppressStickFromProgrammaticScrollRef.current) return;
    const el = messagesScrollRef.current;
    if (!el) return;
    const slack = 80;
    const dist = el.scrollHeight - el.scrollTop - el.clientHeight;
    stickToBottomRef.current = dist <= slack;
  }, []);

  useEffect(() => {
    scheduleScrollToBottom();
    return () => {
      if (scrollToBottomRafRef.current != null) {
        cancelAnimationFrame(scrollToBottomRafRef.current);
        scrollToBottomRafRef.current = null;
      }
    };
  }, [messages, loading, scheduleScrollToBottom]);

  useEffect(() => {
    const scroll = messagesScrollRef.current;
    if (!scroll) return;

    const onLayoutChange = () => {
      if (!stickToBottomRef.current) return;
      scheduleScrollToBottom({ layoutOnly: true });
    };

    const ro = new ResizeObserver(onLayoutChange);
    ro.observe(scroll);

    const mo = new MutationObserver(onLayoutChange);
    mo.observe(scroll, { childList: true, subtree: true });

    return () => {
      ro.disconnect();
      mo.disconnect();
    };
  }, [scheduleScrollToBottom]);

  function resetChat() {
    stickToBottomRef.current = true;
    setMessages([{ id: "welcome", role: "assistant", content: WELCOME_TEXT }]);
    setError(null);
    setChatTitle("新对话");
    setActiveSidebarId("new");
    ragSessionIdRef.current = null;
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

    stickToBottomRef.current = true;

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
      const sid = ragSessionIdRef.current;
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: q,
          stream: true,
          ...(sid ? { sessionId: sid } : {}),
        }),
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

      const streamResult = await readRagStreamBody(reader, {
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
      const nextSid = streamResult.sessionId;
      if (nextSid) {
        ragSessionIdRef.current = nextSid;
      }
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
    handleMessagesScroll,
    clearChat,
    saveChat,
    selectSidebar,
    onSubmit,
  };
}

export function ChatProvider({ children }: { children: ReactNode }) {
  const value = useChatState();
  return createElement(ChatContext.Provider, { value }, children);
}

export function useChat(): ChatContextValue {
  const v = useContext(ChatContext);
  if (!v) {
    throw new Error("useChat must be used within ChatProvider");
  }
  return v;
}
