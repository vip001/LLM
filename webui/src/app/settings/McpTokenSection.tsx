"use client";

import { useCallback, useEffect, useState } from "react";

import { JsonCodeBlock } from "../../components/settings/JsonCodeBlock";
import { AUTH_BASE_URL } from "../../lib/authBaseUrl";

const AUTH_TOKEN_KEY = "auth_token";

export function McpTokenSection() {
  const [configText, setConfigText] = useState("");
  const [loading, setLoading] = useState(false);
  const [initializing, setInitializing] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const run = async () => {
      const token = localStorage.getItem(AUTH_TOKEN_KEY);
      if (!token) {
        setInitializing(false);
        return;
      }
      try {
        const response = await fetch(`${AUTH_BASE_URL}/mcp-token`, {
          method: "GET",
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });
        const data = (await response.json()) as { config?: string; detail?: string };
        if (response.ok) {
          if (typeof data.config === "string" && data.config.trim()) {
            setConfigText(data.config);
          }
          return;
        }
        if (response.status === 404) {
          return;
        }
        setError(typeof data.detail === "string" ? data.detail : `请求失败（${response.status}）`);
      } catch (e) {
        setError(e instanceof Error ? e.message : "网络错误");
      } finally {
        setInitializing(false);
      }
    };
    void run();
  }, []);

  const generate = useCallback(async () => {
    const token = localStorage.getItem(AUTH_TOKEN_KEY);
    if (!token) {
      setError("请先登录后再生成 MCP Token。");
      setConfigText("");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const response = await fetch(`${AUTH_BASE_URL}/mcp-token`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      const data = (await response.json()) as { config?: string; detail?: string };
      if (!response.ok) {
        setConfigText("");
        setError(typeof data.detail === "string" ? data.detail : `请求失败（${response.status}）`);
        return;
      }
      if (typeof data.config !== "string" || !data.config.trim()) {
        setConfigText("");
        setError("响应中缺少 config 字段。");
        return;
      }
      setConfigText(data.config);
    } catch (e) {
      setConfigText("");
      setError(e instanceof Error ? e.message : "网络错误");
    } finally {
      setLoading(false);
    }
  }, []);

  return (
    <div className="border-t border-[#eee] px-[5%] py-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-base font-semibold text-[#333]">知识库mcp服务token</h2>
        <button
          type="button"
          disabled={loading || initializing}
          onClick={() => void generate()}
          className="rounded-lg bg-[#0066cc] px-4 py-2 text-sm font-medium text-white shadow-sm transition hover:bg-[#0052a3] disabled:cursor-not-allowed disabled:opacity-60"
        >
          {loading ? "生成中…" : initializing ? "加载中…" : "生成"}
        </button>
      </div>
      {error ? <p className="mt-3 text-sm text-red-600">{error}</p> : null}
      {configText ? (
        <div className="mt-4">
          <JsonCodeBlock text={configText} />
        </div>
      ) : null}
    </div>
  );
}
