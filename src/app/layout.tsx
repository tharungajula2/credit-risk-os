import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { getAllNotes } from "@/lib/markdown";
import Sidebar from "@/components/Sidebar";
import MobileHeader from "@/components/MobileHeader";
import VianAIChat from "@/components/VianAIChat";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Quant OS",
  description: "A Zettelkasten-powered knowledge base for credit risk.",
};

export default async function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const notes = await getAllNotes();

  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
        style={{
          background: "#050505",
          color: "#f0f0f5",
          margin: 0,
        }}
      >
        {/* ── OS Shell Wrapper ── */}
        <div className="flex w-full min-h-screen">
          <Sidebar notes={notes} />

          {/* ── Main content column ── */}
          <div className="flex-1 flex flex-col min-w-0 transition-all duration-300">
            {/* Mobile header — only visible below lg */}
            <MobileHeader />

            {/* Page content */}
            <main className="flex-1 relative">
              {children}
            </main>
          </div>
        </div>
        <VianAIChat />
      </body>
    </html>
  );
}
