"use client";

import { useState } from "react";
import { LoginModal } from "./LoginModal";

export function AppNav() {
  const [loginOpen, setLoginOpen] = useState(false);

  return (
    <>
      <nav className="shrink-0 flex items-center justify-between px-[5%] py-4 bg-white border-b border-[#e5e5e5] z-100">
        <div className="text-xl font-bold text-[#0066cc]">AI助手</div>
        <ul className="flex  list-none  gap-8 m-0 p-0">
          <li>
            <a
              href="#"
              className="no-underline text-[#333] font-medium hover:text-[#0066cc] transition-colors"
              onClick={(e) => e.preventDefault()}
            >
              首页
            </a>
          </li>
          <li>
            <a
              href="#"
              className="no-underline text-[#333] font-medium hover:text-[#0066cc] transition-colors"
              onClick={(e) => {
                e.preventDefault();
                setLoginOpen(true);
              }}
            >
              登录
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

      <LoginModal open={loginOpen} onClose={() => setLoginOpen(false)} />
    </>
  );
}
