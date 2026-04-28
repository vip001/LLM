"use client";

import { type MouseEvent, useEffect, useState } from "react";
import { LoginModal } from "./LoginModal";
import { LogoutConfirmModal } from "./LogoutConfirmModal";

const AUTH_BASE_URL = process.env.NEXT_PUBLIC_AUTH_BASE_URL?.trim() || "/auth";
const AUTH_TOKEN_KEY = "auth_token";
const USER_EMAIL_KEY = "auth_user_email";

export function AppNav() {
  const [loginOpen, setLoginOpen] = useState(false);
  const [logoutConfirmOpen, setLogoutConfirmOpen] = useState(false);
  const [loggingOut, setLoggingOut] = useState(false);
  const [token, setToken] = useState("");
  const [email, setEmail] = useState("");

  useEffect(() => {
    const storedToken = localStorage.getItem(AUTH_TOKEN_KEY);
    const storedEmail = localStorage.getItem(USER_EMAIL_KEY);
    if (!storedToken) {
      return;
    }

    const validateLogin = async () => {
      try {
        const response = await fetch(`${AUTH_BASE_URL}/me`, {
          headers: {
            Authorization: `Bearer ${storedToken}`,
          },
        });
        const data = (await response.json()) as { email?: string; detail?: string };
        if (!response.ok || !data.email) {
          throw new Error(data.detail || "登录状态已失效");
        }
        setToken(storedToken);
        setEmail(data.email || storedEmail || "");
        localStorage.setItem(USER_EMAIL_KEY, data.email);
      } catch {
        localStorage.removeItem(AUTH_TOKEN_KEY);
        localStorage.removeItem(USER_EMAIL_KEY);
      }
    };

    void validateLogin();
  }, []);

  const handleLoginSuccess = (payload: { token: string; email: string }) => {
    setToken(payload.token);
    setEmail(payload.email);
    localStorage.setItem(AUTH_TOKEN_KEY, payload.token);
    localStorage.setItem(USER_EMAIL_KEY, payload.email);
  };

  const handleAuthAction = (event: MouseEvent<HTMLAnchorElement>) => {
    event.preventDefault();
    if (!token) {
      setLoginOpen(true);
      return;
    }
    setLogoutConfirmOpen(true);
  };

  const performLogout = async () => {
    if (!token || loggingOut) {
      return;
    }
    setLoggingOut(true);
    try {
      await fetch(`${AUTH_BASE_URL}/logout`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
    } finally {
      setLoggingOut(false);
      setToken("");
      setEmail("");
      localStorage.removeItem(AUTH_TOKEN_KEY);
      localStorage.removeItem(USER_EMAIL_KEY);
      setLoginOpen(false);
      setLogoutConfirmOpen(false);
    }
  };

  const closeLogoutConfirm = () => {
    if (!loggingOut) {
      setLogoutConfirmOpen(false);
    }
  };

  return (
    <>
      <nav className="shrink-0 flex items-center justify-between px-[5%] py-4 bg-white border-b border-[#e5e5e5] z-100">
        <div className="text-xl font-bold text-[#0066cc]">AI助手</div>
        <ul className="flex  list-none  gap-8 m-0 p-0">
          {email && (
            <li className="text-[#666] text-sm max-w-[240px] truncate" title={email}>
              {email}
            </li>
          )}
          <li>
            <a
              href="#"
              className="no-underline text-[#333] font-medium hover:text-[#0066cc] transition-colors"
              onClick={handleAuthAction}
            >
              {token ? "退出登录" : "登录"}
            </a>
          </li>
          <li>
            <a
              href="#"
              className="no-underline text-[#333] font-medium hover:text-[#0066cc] transition-colors"
              onClick={(e) => {
                e.preventDefault();
              }}
            >
              功能
            </a>
          </li>
          <li>
            <a
              href="#"
              className="no-underline text-[#333] font-medium hover:text-[#0066cc] transition-colors"
              onClick={(e) => e.preventDefault()}
            >
              帮助
            </a>
          </li>
        </ul>
      </nav>

      <LoginModal
        open={loginOpen}
        onClose={() => setLoginOpen(false)}
        onLoginSuccess={handleLoginSuccess}
      />

      <LogoutConfirmModal
        open={logoutConfirmOpen}
        loggingOut={loggingOut}
        onClose={closeLogoutConfirm}
        onConfirm={performLogout}
      />
    </>
  );
}
