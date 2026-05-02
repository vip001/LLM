/** Same-origin `/auth` in Docker/nginx; local dev defaults to loginserver. */
export const AUTH_BASE_URL =
  process.env.NEXT_PUBLIC_AUTH_BASE_URL?.trim() || "http://127.0.0.1:8000/auth";
