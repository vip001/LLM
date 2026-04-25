import type { FormEvent } from "react";

type ChatComposerProps = {
  query: string;
  onQueryChange: (value: string) => void;
  loading: boolean;
  onSubmit: (e: FormEvent) => void;
};

export function ChatComposer({
  query,
  onQueryChange,
  loading,
  onSubmit,
}: ChatComposerProps) {
  return (
    <form
      onSubmit={onSubmit}
      className="shrink-0 p-4 border-t border-[#e5e5e5] bg-white flex gap-4 items-center"
    >
      <input
        type="text"
        value={query}
        onChange={(e) => onQueryChange(e.target.value)}
        placeholder="输入你的问题..."
        disabled={loading}
        className="flex-1 min-w-0 min-h-[48px] px-5 py-3.5 border border-[#e5e5e5] rounded-[24px] text-base bg-white text-[#333] placeholder:text-[#999] focus:outline-none focus:border-[#0066cc] focus:ring-2 focus:ring-[#0066cc]/10"
      />
      <button
        type="submit"
        disabled={loading || !query.trim()}
        className="inline-flex items-center justify-center min-w-28 shrink-0 px-8 py-3.5 min-h-[48px] bg-[#0066cc] text-white border-0 rounded-[24px] text-base font-medium cursor-pointer hover:bg-[#0052a3] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        {loading ? "发送中…" : "发送"}
      </button>
    </form>
  );
}
