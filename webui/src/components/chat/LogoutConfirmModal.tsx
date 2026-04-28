"use client";

type LogoutConfirmModalProps = {
  open: boolean;
  loggingOut: boolean;
  onClose: () => void;
  onConfirm: () => void;
};

export function LogoutConfirmModal({
  open,
  loggingOut,
  onClose,
  onConfirm,
}: LogoutConfirmModalProps) {
  if (!open) {
    return null;
  }

  return (
    <div
      className="fixed inset-0 z-200 flex items-center justify-center bg-black/35 px-4"
      onClick={onClose}
      role="presentation"
    >
      <div
        className="w-full max-w-sm rounded-2xl border border-[#e5e5e5] bg-white p-6 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 className="text-lg font-semibold text-[#333]">确认退出</h2>
        <p className="mt-3 text-sm leading-relaxed text-[#666]">
          确定要退出当前账号吗？
        </p>
        <div className="mt-6 flex justify-end gap-3">
          <button
            type="button"
            onClick={onClose}
            disabled={loggingOut}
            className="rounded-lg border border-[#e5e5e5] px-4 py-2 text-sm font-medium text-[#333] transition-colors hover:bg-[#f5f5f5] disabled:cursor-not-allowed disabled:opacity-60"
          >
            取消
          </button>
          <button
            type="button"
            onClick={() => void onConfirm()}
            disabled={loggingOut}
            className="rounded-lg bg-[#0066cc] px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-[#005bb7] disabled:cursor-not-allowed disabled:bg-[#9ec4ea]"
          >
            {loggingOut ? "退出中..." : "退出登录"}
          </button>
        </div>
      </div>
    </div>
  );
}
