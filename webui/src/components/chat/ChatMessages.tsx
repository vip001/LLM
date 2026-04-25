import type { RefObject } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatMessage, RagContextItem } from "../../types/chat";
import { markdownBubbleClass } from "../../lib/chat/constants";
import { RefSources } from "./RefSources";

type ChatMessagesProps = {
  messages: ChatMessage[];
  loading: boolean;
  refs: RagContextItem[];
  messagesScrollRef: RefObject<HTMLDivElement | null>;
};

export function ChatMessages({
  messages,
  loading,
  refs,
  messagesScrollRef,
}: ChatMessagesProps) {
  return (
    <div
      ref={messagesScrollRef}
      className="flex-1 min-h-0 overflow-y-auto overflow-x-hidden px-8 py-8 flex flex-col gap-6"
    >
      {messages.map((msg) => {
        if (msg.role === "user") {
          return (
            <div
              key={msg.id}
              className="self-end max-w-[70%] px-4 py-3 rounded-xl bg-[#0066cc] text-white leading-relaxed"
            >
              {msg.content}
            </div>
          );
        }
        const isLast = msg.id === messages[messages.length - 1]?.id;
        const showTyping = loading && isLast && !msg.content;
        return (
          <div
            key={msg.id}
            className="self-start max-w-[70%] px-4 py-3 rounded-xl bg-[#f0f7ff] text-[#333] leading-relaxed"
          >
            {showTyping ? (
              <span className="text-[#666] text-sm">正在生成回答…</span>
            ) : (
              <div className={markdownBubbleClass}>
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {msg.content}
                </ReactMarkdown>
                {loading && isLast && (
                  <span
                    className="inline-block w-0.5 h-4 ml-0.5 bg-[#0066cc] animate-pulse align-middle"
                    aria-hidden
                  />
                )}
              </div>
            )}
          </div>
        );
      })}

      {refs.length > 0 && !loading && <RefSources refs={refs} />}
    </div>
  );
}
