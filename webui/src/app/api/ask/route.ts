import { existsSync } from "node:fs";

import { NextRequest } from "next/server";
import * as grpc from "@grpc/grpc-js";

import { askViaGrpc, grpcCodeToHttpStatus } from "@/lib/grpcAsk";

const isDev = process.env.NODE_ENV === "development";

/**
 * 设置 GRPC_ASK_ADDR（如 server:50051）时，Next 通过 gRPC 流式调用后端，与 Flask /ask 流式体格式一致。
 * 未设置时回退 HTTP（FLASK_ASK_URL）。
 */
function grpcAskAddr(): string | null {
  const u = process.env.GRPC_ASK_ADDR;
  if (typeof u === "string" && u.trim()) return u.trim();
  else return "127.0.0.1:50051"
}

/**
 * 1) 环境变量 FLASK_ASK_URL
 * 2) Docker 内默认走 compose 服务名 server（与 docker-compose 中 Flask 服务名一致）
 * 3) 非 Docker 的生产回退固定网段 IP；本地开发默认 127.0.0.1
 */
function flaskAskUrl(): string {
  const u = process.env.FLASK_ASK_URL;
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
    const trace = Boolean(body?.trace);
    const sessionId =
      typeof body?.sessionId === "string" ? body.sessionId.trim() : "";
    if (!query) {
      return Response.json(
        errPayload("MISSING_QUERY", "缺少参数 query"),
        { status: 400 }
      );
    }

    const grpcAddr = grpcAskAddr();
    if (grpcAddr) {
      try {
        console.log("grpcAddr:",grpcAddr);
        return await askViaGrpc({
          address: grpcAddr,
          query,
          stream,
          trace,
          sessionId,
        });
      } catch (grpcErr) {
        const ge = grpcErr as grpc.ServiceError;
        const code = typeof ge?.code === "number" ? ge.code : grpc.status.UNKNOWN;
        const status = grpcCodeToHttpStatus(code);
        const msg = ge?.message || "gRPC 调用失败";
        console.error("[api/ask] gRPC 错误:", code, msg, grpcAddr);
        return Response.json(
          errPayload(
            "GRPC_ERROR",
            msg,
            isDev ? { address: grpcAddr, code } : undefined
          ),
          { status: status >= 400 ? status : 502 }
        );
      }
    }

    const target = flaskAskUrl();
    const upstreamHeaders: Record<string, string> = {
      "Content-Type": "application/json",
    };
    const flaskBody: Record<string, unknown> = { query, stream, trace };
    if (sessionId) {
      flaskBody.sessionId = sessionId;
    }
    let res: Response;
    try {
      res = await fetch(target, {
        method: "POST",
        headers: upstreamHeaders,
        body: JSON.stringify(flaskBody),
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

    const passthrough = new Headers({
      "Content-Type": res.headers.get("Content-Type") ?? "text/plain; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    });
    return new Response(res.body, {
      status: 200,
      headers: passthrough,
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
