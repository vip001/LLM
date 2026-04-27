import type { RagContextItem } from "../types/chat";
import { ByteBuffer } from "./byteBuffer";

const STREAM_MAGIC = new Uint8Array([0x52, 0x41, 0x47, 0x01]);

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
  onAnswerDelta?: (delta: string) => void;
};

export async function readRagStreamBody(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  cb?: RagStreamCallbacks
): Promise<{ contexts: RagContextItem[]; answerText: string }> {
  const decoder = new TextDecoder();
  const buffer = new ByteBuffer();

  const readChunk = async (): Promise<boolean> => {
    const { done, value } = await reader.read();
    if (value && value.length) {
      buffer.append(value);
    }
    return !done;
  };

  while (buffer.length < 4 && (await readChunk())) {}
  let buffered = buffer.view();

  if (!startsWithMagic(buffered)) {
    while (await readChunk()) {}
    buffered = buffer.view();
    const answerText = decoder.decode(buffered);
    if (answerText) cb?.onAnswerDelta?.(answerText);
    return { contexts: [], answerText };
  }

  while (buffer.length < 8 && (await readChunk())) {}
  buffered = buffer.view();

  const jsonLen =
    (buffered[4] << 24) | (buffered[5] << 16) | (buffered[6] << 8) | buffered[7];
  if (jsonLen < 0 || jsonLen > 50 * 1024 * 1024) {
    throw new Error("无效的引用元数据长度");
  }

  while (buffer.length < 8 + jsonLen && (await readChunk())) {}
  buffered = buffer.view();

  const jsonBytes = buffered.subarray(8, 8 + jsonLen);
  let meta: { contexts?: RagContextItem[] };
  try {
    meta = JSON.parse(decoder.decode(jsonBytes)) as { contexts?: RagContextItem[] };
  } catch {
    throw new Error("无法解析引用元数据 JSON");
  }

  const contexts = meta.contexts ?? [];
  cb?.onRefs?.(contexts);

  const tail = buffered.subarray(8 + jsonLen);
  const answerChunks: string[] = [];
  const firstDelta = decoder.decode(tail, { stream: true });
  if (firstDelta) {
    answerChunks.push(firstDelta);
    cb?.onAnswerDelta?.(firstDelta);
  }
  for (;;) {
    const { done, value } = await reader.read();
    if (value && value.length) {
      const delta = decoder.decode(value, { stream: true });
      if (delta) {
        answerChunks.push(delta);
        cb?.onAnswerDelta?.(delta);
      }
    }
    if (done) break;
  }
  const flushDelta = decoder.decode();
  if (flushDelta) {
    answerChunks.push(flushDelta);
    cb?.onAnswerDelta?.(flushDelta);
  }
  const answerText = answerChunks.join("");
  return { contexts, answerText };
}
