import { NextRequest, NextResponse } from "next/server";

const FLASK_ASK_URL =
  process.env.FLASK_ASK_URL ?? "http://127.0.0.1:5000/ask";
const isDev = process.env.NODE_ENV === "development";

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
    if (!query) {
      return Response.json(
        errPayload("MISSING_QUERY", "缺少参数 query"),
        { status: 400 }
      );
    }

    let res: Response;
    try {
      res = await fetch(FLASK_ASK_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
    } catch (fetchErr) {
      const msg = fetchErr instanceof Error ? fetchErr.message : String(fetchErr);
      console.error("[api/ask] 请求 Flask 失败:", msg, "\nURL:", FLASK_ASK_URL);
      return Response.json(
        errPayload(
          "FLASK_UNREACHABLE",
          "无法连接后端服务，请确认 Flask 已启动且 FLASK_ASK_URL 正确",
          isDev ? { url: FLASK_ASK_URL, raw: msg } : undefined
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
