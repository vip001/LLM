import type { RagContextItem } from "../types/chat";

const STREAM_MAGIC = new Uint8Array([0x52, 0x41, 0x47, 0x01]);

function concatBytes(a: Uint8Array, b: Uint8Array): Uint8Array {
  const out = new Uint8Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}

function startsWithMagic(buf: Uint8Array): boolean {
  if (buf.length < 4) return false;
  return (
    buf[0] === STREAM_MAGIC[0] &&
    buf[1] === STREAM_MAGIC[1] &&
    buf[2] === STREAM_MAGIC[2] &&
    buf[3] === STREAM_MAGIC[3]
  );
}

export type RagStreamCallbacks = {
  onRefs?: (contexts: RagContextItem[]) => void;
  onAnswerDelta?: (fullAnswer: string) => void;
};

export async function readRagStreamBody(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  cb?: RagStreamCallbacks
): Promise<{ contexts: RagContextItem[]; answerText: string }> {
  const decoder = new TextDecoder();
  let buffer: Uint8Array<ArrayBufferLike> = new Uint8Array(0);

  const readChunk = async (): Promise<boolean> => {
    const { done, value } = await reader.read();
    if (value && value.length) {
      buffer = concatBytes(buffer, new Uint8Array(value));
    }
    return !done;
  };

  while (buffer.length < 4 && (await readChunk())) {}

  if (!startsWithMagic(buffer)) {
    while (await readChunk()) {}
    const answerText = decoder.decode(buffer);
    cb?.onAnswerDelta?.(answerText);
    return { contexts: [], answerText };
  }

  while (buffer.length < 8 && (await readChunk())) {}

  const jsonLen =
    (buffer[4] << 24) | (buffer[5] << 16) | (buffer[6] << 8) | buffer[7];
  if (jsonLen < 0 || jsonLen > 50 * 1024 * 1024) {
    throw new Error("无效的引用元数据长度");
  }

  while (buffer.length < 8 + jsonLen && (await readChunk())) {}

  const jsonBytes = buffer.subarray(8, 8 + jsonLen);
  let meta: { contexts?: RagContextItem[] };
  try {
    meta = JSON.parse(decoder.decode(jsonBytes)) as { contexts?: RagContextItem[] };
  } catch {
    throw new Error("无法解析引用元数据 JSON");
  }

  const contexts = meta.contexts ?? [];
  cb?.onRefs?.(contexts);

  const tail = buffer.subarray(8 + jsonLen);
  buffer = new Uint8Array(0);

  let answerText = decoder.decode(tail, { stream: true });
  cb?.onAnswerDelta?.(answerText);
  for (;;) {
    const more = await readChunk();
    if (buffer.length) {
      answerText += decoder.decode(buffer, { stream: true });
      cb?.onAnswerDelta?.(answerText);
      buffer = new Uint8Array(0);
    }
    if (!more) break;
  }
  answerText += decoder.decode();
  cb?.onAnswerDelta?.(answerText);
  return { contexts, answerText };
}
