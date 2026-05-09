import * as grpc from "@grpc/grpc-js";

import { AskRequest, AskServiceClient } from "@/gen/ask";

export function grpcCodeToHttpStatus(code: grpc.status): number {
  if (code === grpc.status.INVALID_ARGUMENT) return 400;
  if (code === grpc.status.UNAVAILABLE) return 502;
  if (code === grpc.status.DEADLINE_EXCEEDED) return 504;
  return 500;
}

function bodyChunkToUint8(chunk: Buffer | Uint8Array): Uint8Array {
  if (Buffer.isBuffer(chunk)) return new Uint8Array(chunk);
  return new Uint8Array(chunk);
}

const streamDeadlineMs = 600_000;

/**
 * 通过 gRPC 调用 RAG；流式时返回与 Flask /ask 相同的 application/octet-stream 字节序列。
 */
export function askViaGrpc(params: {
  address: string;
  query: string;
  stream: boolean;
  trace: boolean;
  sessionId: string;
}): Promise<Response> {
  const client = new AskServiceClient(params.address, grpc.credentials.createInsecure());
  const metadata = new grpc.Metadata();
  const callOpts: Partial<grpc.CallOptions> = {
    deadline: new Date(Date.now() + streamDeadlineMs),
  };
  const req = AskRequest.create({
    query: params.query,
    stream: params.stream,
    trace: params.trace,
    sessionId: params.sessionId,
  });

  if (!params.stream) {
    return new Promise((resolve, reject) => {
      client.askOnce(req, metadata, callOpts, (err, res) => {
        client.close();
        if (err) {
          reject(err);
          return;
        }
        const raw = res?.jsonBody;
        if (!raw?.length) {
          reject(new Error("AskOnce 返回为空"));
          return;
        }
        try {
          const json: unknown = JSON.parse(Buffer.from(raw).toString("utf-8"));
          resolve(Response.json(json));
        } catch {
          reject(new Error("AskOnce 返回非 JSON"));
        }
      });
    });
  }

  const webStream = new ReadableStream<Uint8Array>({
    start(controller) {
      const call = client.askStream(req, metadata, callOpts);
      call.on("data", (msg) => {
        const c = msg.bodyChunk;
        if (c && c.length) controller.enqueue(bodyChunkToUint8(c));
      });
      call.on("error", (err: Error) => {
        client.close();
        controller.error(err);
      });
      call.on("end", () => {
        client.close();
        controller.close();
      });
    },
  });

  return Promise.resolve(
    new Response(webStream, {
      status: 200,
      headers: {
        "Content-Type": "application/octet-stream",
        "Cache-Control": "no-cache, no-transform",
        Connection: "keep-alive",
        "X-Accel-Buffering": "no",
      },
    })
  );
}
