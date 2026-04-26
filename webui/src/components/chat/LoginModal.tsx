"use client";

import { useEffect, useState } from "react";

type LoginModalProps = {
  open: boolean;
  onClose: () => void;
};

const COUNTDOWN_SECONDS = 60;
const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export function LoginModal({ open, onClose }: LoginModalProps) {
  const [email, setEmail] = useState("");
  const [code, setCode] = useState("");
  const [countdown, setCountdown] = useState(0);

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
  const canSendCode = isEmailValid && countdown === 0;
  const canLogin = email.trim().length > 0 && code.trim().length > 0;

  const handleSendCode = () => {
    if (!canSendCode) {
      return;
    }
    setCountdown(COUNTDOWN_SECONDS);
  };

  const handleLogin = () => {
    if (!canLogin) {
      return;
    }
    onClose();
  };

  return (
    <div
      className="fixed inset-0 z-200 flex items-center justify-center bg-black/35 px-4"
      onClick={onClose}
    >
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
                {countdown > 0 ? `${countdown}s` : "发送验证码"}
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
          登录
        </button>
      </div>
    </div>
  );
}
