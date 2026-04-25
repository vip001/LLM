import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI助手",
  description: "基于本地 RAG 的问答界面",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  
  return (
    <html lang="zh-CN">
      <body className="antialiased h-dvh min-h-0 overflow-hidden font-sans">
        {children}
      </body>
    </html>
  );
}
