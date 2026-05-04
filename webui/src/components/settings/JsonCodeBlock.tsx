"use client";

import { useCallback, useState } from "react";

type JsonCodeBlockProps = {
  text: string;
  className?: string;
};

export function JsonCodeBlock({ text, className = "" }: JsonCodeBlockProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  }, [text]);

  return (
    <div className={`overflow-hidden rounded-lg border border-[#e2e8f0] bg-[#f1f5f9] shadow-sm ${className}`}>
      <div className="flex items-center justify-between gap-3 border-b border-[#e2e8f0] bg-[#e8eef4] px-3 py-2">
        <span className="text-xs font-medium tracking-wide text-[#64748b]">JSON</span>
        <button
          type="button"
          onClick={() => void handleCopy()}
          className="rounded-md border border-[#cbd5e1] bg-white px-2.5 py-1 text-xs font-medium text-[#334155] transition hover:bg-[#f8fafc]"
        >
          {copied ? "已复制" : "复制"}
        </button>
      </div>
      <pre className="max-h-[min(60vh,28rem)] overflow-auto p-3 text-left text-[13px] leading-relaxed text-[#1e293b]">
        <code className="font-mono">{text}</code>
      </pre>
    </div>
  );
}
