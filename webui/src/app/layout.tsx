import type { Metadata } from "next";
import "./globals.css";

// export const metadata: Metadata = {
//   title: "RAG 问答",
//   description: "基于本地 RAG 的问答界面",
// };
export async function generateMetadata(params:{ title: string }) {
  return {
    title: params.title || "default",
    description: "基于本地 RAG 的问答界面",
  };
}
export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body className="antialiased min-h-screen">{children}</body>
    </html>
  );
}
