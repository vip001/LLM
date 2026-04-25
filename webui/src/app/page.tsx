"use client";

import { ChatHome } from "../components/chat/ChatHome";

export type { RagContextItem, ChatMessage } from "../types/chat";

export default function Home() {
  return <ChatHome />;
}
