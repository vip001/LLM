import { existsSync } from "node:fs";

import { NextRequest } from "next/server";

const isDev = process.env.NODE_ENV === "development";

/**
 * 1) 环境变量 FLASK_ASK_URL（键名拼接，减少被构建器内联）
 * 2) Docker 内默认走 compose 服务名 server（与 docker-compose 中 Flask 服务名一致）
 * 3) 非 Docker 的生产回退固定网段 IP；本地开发默认 127.0.0.1
 */
function flaskAskUrl(): string {
  const k = "FLASK" + "_" + "ASK" + "_" + "URL";
  const u = (process.env as Record<string, string | undefined>)[k];
  if (typeof u === "string" && u.trim()) return u.trim();
  if (existsSync("/.dockerenv")) {
    return "http://server:5000/ask";
  }
  if (process.env.NODE_ENV === "production") {
    return "http://172.28.240.11:5000/ask";
  }
  return "http://127.0.0.1:5000/ask";
}

export const revalidate = 10;
export const dynamic = "force-dynamic";

/** 开发环境下在响应里带上更多诊断信息，便于排查 500 */
function errPayload(
  code: string,
  message: string,
  detail?: Record<string, unknown>
) {
  const payload: Record<string, unknown> = {
    error: message,
    code,
    ...(isDev && detail && { detail }),
  };
  return payload;
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const query = (body?.query ?? "").trim();
    const stream = body?.stream ?? true;
    const strategy =
      (body?.strategy ?? body?.enhance_strategy ?? "query2doc").toString().trim() ||
      "query2doc";
    if (!query) {
      return Response.json(
        errPayload("MISSING_QUERY", "缺少参数 query"),
        { status: 400 }
      );
    }

    const target = flaskAskUrl();
    let res: Response;
    try {
      res = await fetch(target, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, stream, strategy }),
      });
    } catch (fetchErr) {
      const msg = fetchErr instanceof Error ? fetchErr.message : String(fetchErr);
      console.error("[api/ask] 请求 Flask 失败:", msg, "\nURL:", target);
      return Response.json(
        errPayload(
          "FLASK_UNREACHABLE",
          "无法连接后端服务，请确认 Flask 已启动且 FLASK_ASK_URL 正确",
          isDev ? { url: target, raw: msg } : undefined
        ),
        { status: 502 }
      );
    }

    if (!res.ok) {
      let data: unknown;
      try {
        const text = await res.text();
        data = text.trim() ? JSON.parse(text) : null;
      } catch {
        data = null;
      }
      console.error("[api/ask] Flask 业务错误:", res.status, data);
      return Response.json(
        errPayload(
          "FLASK_ERROR",
          (data as { error?: string })?.error ?? `后端错误 ${res.status}`,
          isDev ? { status: res.status, body: data } : undefined
        ),
        { status: res.status >= 400 ? res.status : 502 }
      );
    }

    return new Response(res.body, {
      status: 200,
      headers: {
        "Content-Type": res.headers.get("Content-Type") ?? "text/plain; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
        Connection: "keep-alive",
        "X-Accel-Buffering": "no",
      },
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "代理请求失败";
    const stack = err instanceof Error ? err.stack : undefined;
    console.error("[api/ask] 未预期错误:", message, stack);
    return Response.json(
      errPayload(
        "PROXY_ERROR",
        message,
        isDev && stack ? { stack: stack.split("\n").slice(0, 5) } : undefined
      ),
      { status: 500 }
    );
  }
}
