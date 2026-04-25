export const SIDEBAR_CHATS = [
  { id: "new", label: "新对话" },
  { id: "py", label: "Python编程指南" },
  { id: "travel", label: "旅行攻略" },
  { id: "health", label: "健康饮食" },
] as const;

export const WELCOME_TEXT =
  "你好！我是你的AI助手，有什么可以帮你的吗？";

export const markdownBubbleClass =
  "text-[#333] leading-relaxed [&_pre]:rounded-lg [&_pre]:bg-[#eef2f6] [&_pre]:p-3 [&_pre]:overflow-x-auto [&_code]:rounded [&_code]:bg-[#eef2f6] [&_code]:px-1.5 [&_code]:py-0.5 [&_code]:text-sm [&_pre_code]:bg-transparent [&_pre_code]:p-0 [&_ul]:list-disc [&_ul]:pl-6 [&_ol]:list-decimal [&_ol]:pl-6 [&_h1]:text-xl [&_h2]:text-lg [&_h3]:text-base [&_h1,h2,h3]:font-semibold [&_a]:text-[#0066cc] [&_a]:underline";
