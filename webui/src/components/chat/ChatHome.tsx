"use client";

import { useChat } from "../../hooks/useChat";
import { AppNav } from "./AppNav";
import { ChatSidebar } from "./ChatSidebar";
import { ChatHeader } from "./ChatHeader";
import { ChatMessages } from "./ChatMessages";
import { ChatComposer } from "./ChatComposer";

export function ChatHome() {
  const {
    messages,
    activeSidebarId,
    chatTitle,
    query,
    setQuery,
    loading,
    error,
    refs,
    saveHint,
    messagesScrollRef,
    clearChat,
    saveChat,
    selectSidebar,
    onSubmit,
  } = useChat();

  return (
    <div className="h-dvh min-h-0 flex flex-col  overflow-hidden bg-[#f7f7f7] font-sans text-[#333]">
      <AppNav />

      <div className="flex flex-1 min-h-0">
        <ChatSidebar
          activeSidebarId={activeSidebarId}
          onSelect={selectSidebar}
        />

        <section className="flex-1 flex flex-col min-w-0 min-h-0 bg-white">
          <ChatHeader
            chatTitle={chatTitle}
            onClear={clearChat}
            onSave={saveChat}
          />

          {saveHint && (
            <div className="shrink-0 mx-4 mt-2 text-sm text-[#0066cc]">
              已保存到浏览器本地
            </div>
          )}

          <ChatMessages
            messages={messages}
            loading={loading}
            refs={refs}
            messagesScrollRef={messagesScrollRef}
          />

          {error && (
            <div className="shrink-0 mx-4 mb-2 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-red-700 text-sm">
              {error}
            </div>
          )}

          <ChatComposer
            query={query}
            onQueryChange={setQuery}
            loading={loading}
            onSubmit={onSubmit}
          />
        </section>
      </div>
    </div>
  );
}
