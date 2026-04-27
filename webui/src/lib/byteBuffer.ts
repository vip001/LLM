export class ByteBuffer {
  private buf: Uint8Array;
  private used = 0;

  constructor(initialCapacity = 1024) {
    this.buf = new Uint8Array(initialCapacity);
  }

  get length(): number {
    return this.used;
  }

  view(): Uint8Array {
    return this.buf.subarray(0, this.used);
  }

  append(chunk: Uint8Array): void {
    if (!chunk.length) return;
    this.ensureCapacity(this.used + chunk.length);
    this.buf.set(chunk, this.used);
    this.used += chunk.length;
  }

  private ensureCapacity(required: number): void {
    if (required <= this.buf.length) return;
    let next = this.buf.length || 1;
    while (next < required) next <<= 1;
    const grown = new Uint8Array(next);
    grown.set(this.buf.subarray(0, this.used), 0);
    this.buf = grown;
  }
}
