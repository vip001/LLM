type ChatHeaderProps = {
  chatTitle: string;
  onClear: () => void;
  onSave: () => void;
};

export function ChatHeader({ chatTitle, onClear, onSave }: ChatHeaderProps) {
  return (
    <div className="shrink-0 px-4 py-4 border-b border-[#e5e5e5] flex items-center justify-between gap-4">
      <div className="text-lg font-semibold text-[#333] truncate">{chatTitle}</div>
      <div className="flex gap-3 shrink-0">
        <button
          type="button"
          onClick={onClear}
          className="px-4 py-2 text-sm border border-[#e5e5e5] bg-white rounded-md cursor-pointer hover:bg-[#f0f7ff] hover:border-[#0066cc] transition-colors"
        >
          清空
        </button>
        <button
          type="button"
          onClick={onSave}
          className="px-4 py-2 text-sm border border-[#e5e5e5] bg-white rounded-md cursor-pointer hover:bg-[#f0f7ff] hover:border-[#0066cc] transition-colors"
        >
          保存
        </button>
      </div>
    </div>
  );
}
