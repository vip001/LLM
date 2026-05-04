import { McpTokenSection } from "./McpTokenSection";

export default function SettingsPage() {
  return (
    <div className="h-dvh min-h-0 flex flex-col overflow-hidden bg-[#f7f7f7] font-sans text-[#333]">
      <main className="flex-1 min-h-0 overflow-auto bg-white">
        <div className="px-[5%] py-4">
          <h1 className="text-xl font-semibold text-[#333]">设置</h1>
        </div>
        <McpTokenSection />
      </main>
    </div>
  );
}
