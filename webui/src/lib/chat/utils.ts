export function sourceLabel(source: unknown): string {
  if (typeof source !== "string" || !source) return "unknown";
  const parts = source.split(/[/\\]/);
  return parts[parts.length - 1] || source;
}

export function parseErrorBody(raw: string): {
  error?: string;
  code?: string;
  detail?: unknown;
} {
  const text = raw.trim();
  if (!text) return {};
  try {
    return JSON.parse(text) as { error?: string; code?: string; detail?: unknown };
  } catch {
    return { error: text };
  }
}
