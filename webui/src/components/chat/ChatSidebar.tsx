import { SIDEBAR_CHATS } from "../../lib/chat/constants";

type ChatSidebarProps = {
  activeSidebarId: string;
  onSelect: (id: string) => void;
};

export function ChatSidebar({ activeSidebarId, onSelect }: ChatSidebarProps) {
  return (
    <aside className="hidden md:block w-[250px] shrink-0 bg-white border-r border-[#e5e5e5] p-4 overflow-y-auto">
      <h3 className="m-0 mb-4 text-[#666] text-sm font-semibold uppercase tracking-wide">
        最近对话
      </h3>
      <ul className="list-none m-0 p-0">
        {SIDEBAR_CHATS.map((item) => {
          const active = activeSidebarId === item.id;
          return (
            <li key={item.id}>
              <button
                type="button"
                onClick={() => onSelect(item.id)}
                className={`w-full text-left px-3 py-3 mb-2 rounded-lg text-sm cursor-pointer transition-colors border-0 ${
                  active
                    ? "bg-[#e6f2ff] text-[#0066cc]"
                    : "bg-transparent text-[#333] hover:bg-[#f0f7ff]"
                }`}
              >
                {item.label}
              </button>
            </li>
          );
        })}
      </ul>
    </aside>
  );
}
