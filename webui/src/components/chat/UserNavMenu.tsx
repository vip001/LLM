"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";

type UserNavMenuProps = {
  email: string;
  onRequestLogout: () => void;
};

export function UserNavMenu({ email, onRequestLogout }: UserNavMenuProps) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) {
      return;
    }
    const onPointerDown = (event: PointerEvent) => {
      const el = rootRef.current;
      if (el && !el.contains(event.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("pointerdown", onPointerDown);
    return () => document.removeEventListener("pointerdown", onPointerDown);
  }, [open]);

  const openLogoutFromMenu = () => {
    setOpen(false);
    onRequestLogout();
  };

  return (
    <li className="relative">
      <div ref={rootRef} className="relative">
        <button
          type="button"
          title={email || undefined}
          className="flex max-w-[240px] cursor-pointer items-center gap-1 border-0 bg-transparent p-0 text-left text-sm text-[#666] hover:text-[#0066cc] transition-colors truncate"
          aria-expanded={open}
          aria-haspopup="menu"
          onClick={() => setOpen((v) => !v)}
        >
          <span className="truncate">{email || "账户"}</span>
          <span className="shrink-0 text-[12px] text-[#999]" aria-hidden>
            ▾
          </span>
        </button>
        {open ? (
          <div
            role="menu"
            className="absolute right-0 top-full z-200 mt-1 min-w-[140px] rounded-md border border-[#e5e5e5] bg-white py-1 shadow-md"
          >
            <Link
              href="/settings"
              role="menuitem"
              className="block px-3 py-2 text-sm text-[#333] no-underline hover:bg-[#f5f5f5]"
              onClick={() => setOpen(false)}
            >
              设置
            </Link>
            <button
              type="button"
              role="menuitem"
              className="w-full cursor-pointer border-0 bg-transparent px-3 py-2 text-left text-sm text-[#333] hover:bg-[#f5f5f5]"
              onClick={openLogoutFromMenu}
            >
              退出登录
            </button>
          </div>
        ) : null}
      </div>
    </li>
  );
}
