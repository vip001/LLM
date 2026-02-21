"use client";
import { useState, FormEvent } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export default function Home() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [streamText, setStreamText] = useState("");

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (!query.trim()) return;
    setError(null);
    setStreamText("");
    setLoading(true);
    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query.trim() }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        const d = data as { error?: string; code?: string; detail?: unknown };
        const msg = d?.error ?? `请求失败 ${res.status}`;
        const code = d?.code ? ` [${d.code}]` : "";
        const detail =
          d?.detail != null
            ? `\n\n详情: ${JSON.stringify(d.detail, null, 2)}`
            : "";
        setError(`${msg}${code}${detail}`);
        return;
      }

      const reader = res.body?.getReader();
      const decoder = new TextDecoder();
      if (!reader) {
        setError("无法读取响应流");
        return;
      }
      let acc = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        acc += decoder.decode(value, { stream: true });
        setStreamText(acc);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "网络错误");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="max-w-2xl mx-auto px-4 py-12">
      <h1 className="text-2xl font-semibold text-white mb-2">RAG 问答</h1>
      <p className="text-zinc-400 text-sm mb-8">
        输入问题，将基于本地文档检索并调用模型回答
      </p>

      <form onSubmit={onSubmit} className="mb-8">
        <div className="flex gap-2">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="例如：MMKV的用法"
            className="flex-1 rounded-lg border border-(--border) bg-(--card) px-4 py-3 text-white placeholder:text-zinc-500 focus:outline-none focus:ring-2 focus:ring-(--accent)"
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="rounded-lg bg-(--accent) px-5 py-3 font-medium text-white hover:bg-(--accent-hover) disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? "查询中…" : "提问"}
          </button>
        </div>
      </form>

      {error && (
        <div className="rounded-lg border border-red-500/50 bg-red-500/10 px-4 py-3 text-red-300 text-sm mb-6">
          {error}
        </div>
      )}

      {(streamText || loading) && (
        <section className="rounded-lg border border-(--border) bg-(--card) p-4">
          <h2 className="text-sm font-medium text-zinc-400 mb-2">回答</h2>
          <div className="text-white [&_pre]:rounded-md [&_pre]:bg-zinc-800/80 [&_pre]:p-3 [&_pre]:overflow-x-auto [&_code]:rounded [&_code]:bg-zinc-800/80 [&_code]:px-1.5 [&_code]:py-0.5 [&_code]:text-sm [&_pre_code]:bg-transparent [&_pre_code]:p-0 [&_ul]:list-disc [&_ul]:pl-6 [&_ol]:list-decimal [&_ol]:pl-6 [&_h1]:text-xl [&_h2]:text-lg [&_h3]:text-base [&_h1,h2,h3]:font-semibold [&_a]:text-(--accent) [&_a]:underline">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {streamText}
            </ReactMarkdown>
            {loading && (
              <span className="inline-block w-2 h-4 ml-0.5 bg-(--accent) animate-pulse" aria-hidden />
            )}
          </div>
        </section>
      )}
    </main>
  );
}
