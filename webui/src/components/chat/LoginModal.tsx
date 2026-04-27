"use client";

import { useEffect, useState } from "react";

type LoginModalProps = {
  open: boolean;
  onClose: () => void;
  onLoginSuccess: (payload: { token: string; email: string }) => void;
};

const COUNTDOWN_SECONDS = 60;
const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

const AUTH_BASE_URL = process.env.NEXT_PUBLIC_AUTH_BASE_URL?.trim() || "/auth";

type SendCodeResponse = {
  message: string;
  expires_in: number;
  code?: string | null;
};

type LoginResponse = {
  token: string;
  token_type: string;
  expires_at: string;
  user_email: string;
};

export function LoginModal({ open, onClose, onLoginSuccess }: LoginModalProps) {
  const [email, setEmail] = useState("");
  const [code, setCode] = useState("");
  const [countdown, setCountdown] = useState(0);
  const [toast, setToast] = useState("");
  const [sendingCode, setSendingCode] = useState(false);
  const [loggingIn, setLoggingIn] = useState(false);

  const showToast = (message: string) => {
    setToast(message);
    window.setTimeout(() => {
      setToast("");
    }, 2200);
  };

  useEffect(() => {
    if (countdown <= 0) {
      return;
    }

    const timer = window.setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          window.clearInterval(timer);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => window.clearInterval(timer);
  }, [countdown]);

  useEffect(() => {
    if (!open) {
      setCode("");
    }
  }, [open]);

  if (!open) {
    return null;
  }

  const normalizedEmail = email.trim();
  const isEmailValid = EMAIL_REGEX.test(normalizedEmail);
  const canSendCode = isEmailValid && countdown === 0 && !sendingCode;
  const canLogin = email.trim().length > 0 && code.trim().length > 0 && !loggingIn;

  const handleSendCode = async () => {
    if (!canSendCode) {
      return;
    }
    setSendingCode(true);
    try {
      const response = await fetch(`${AUTH_BASE_URL}/send-code`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email: normalizedEmail }),
      });
      const data = (await response.json()) as SendCodeResponse & { detail?: string };
      if (!response.ok) {
        showToast(data.detail || "验证码发送失败");
        return;
      }
      if (data.code) {
        setCode(data.code);
      }
      setCountdown(COUNTDOWN_SECONDS);
    } catch (err) {
      showToast(err instanceof Error ? err.message : "验证码发送失败");
    } finally {
      setSendingCode(false);
    }
  };

  const handleLogin = async () => {
    if (!canLogin) {
      return;
    }
    setLoggingIn(true);
    try {
      const response = await fetch(`${AUTH_BASE_URL}/login`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          email: normalizedEmail,
          code: code.trim(),
        }),
      });
      const data = (await response.json()) as LoginResponse & { detail?: string };
      if (!response.ok) {
        showToast(data.detail || "登录失败");
        return;
      }
      onLoginSuccess({ token: data.token, email: data.user_email });
      onClose();
    } catch (err) {
      showToast(err instanceof Error ? err.message : "登录失败");
    } finally {
      setLoggingIn(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-200 flex items-center justify-center bg-black/35 px-4"
      onClick={onClose}
    >
      {toast && (
        <div className="pointer-events-none fixed top-5 left-1/2 z-300 -translate-x-1/2 rounded-lg bg-black/85 px-4 py-2 text-sm text-white shadow-lg">
          {toast}
        </div>
      )}
      <div
        className="w-full max-w-md rounded-2xl border border-[#e5e5e5] bg-white p-6 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-5 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-[#333]">邮箱登录</h2>
          <button
            type="button"
            onClick={onClose}
            className="text-[#999] transition-colors hover:text-[#666]"
            aria-label="关闭登录弹窗"
          >
            ✕
          </button>
        </div>
        <div className="space-y-4">
          <label className="block">
            <span className="mb-2 block text-sm text-[#666]">邮箱</span>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="请输入邮箱地址"
              className="w-full rounded-lg border border-[#e5e5e5] bg-white px-3 py-2.5 text-sm text-[#333] outline-none transition-colors placeholder:text-[#aaa] focus:border-[#0066cc]"
            />
          </label>

          <label className="block">
            <span className="mb-2 block text-sm text-[#666]">邮箱验证码</span>
            <div className="flex gap-2">
              <input
                type="text"
                value={code}
                onChange={(e) => setCode(e.target.value)}
                placeholder="请输入验证码"
                className="min-w-0 flex-1 rounded-lg border border-[#e5e5e5] bg-white px-3 py-2.5 text-sm text-[#333] outline-none transition-colors placeholder:text-[#aaa] focus:border-[#0066cc]"
              />
              <button
                type="button"
                onClick={handleSendCode}
                disabled={!canSendCode}
                className="shrink-0 rounded-lg border border-[#0066cc] px-3 py-2.5 text-sm font-medium text-[#0066cc] transition-colors disabled:cursor-not-allowed disabled:border-[#b7d4f4] disabled:text-[#b7d4f4] hover:bg-[#f2f8ff]"
              >
                {sendingCode ? "发送中..." : countdown > 0 ? `${countdown}s` : "发送验证码"}
              </button>
            </div>
          </label>
        </div>

        <button
          type="button"
          onClick={handleLogin}
          disabled={!canLogin}
          className="mt-6 w-full rounded-lg bg-[#0066cc] px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-[#005bb7] disabled:cursor-not-allowed disabled:bg-[#9ec4ea]"
        >
          {loggingIn ? "登录中..." : "登录"}
        </button>
      </div>
    </div>
  );
}
