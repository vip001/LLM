export type RagContextItem = {
  type: string;
  text: string;
  metadata: Record<string, unknown>;
  image_data?: string;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
};
