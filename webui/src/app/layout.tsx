import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Android黄金屋社区",
  description: "基于RAG的Android问答社区",
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
