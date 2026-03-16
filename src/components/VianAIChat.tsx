// @ts-nocheck
"use client";

import { useState, useRef, useEffect } from "react";
import { useChat } from "@ai-sdk/react";
import { BrainCircuit, X, Lock, Send, Unlock } from "lucide-react";

export default function VianAIChat() {
  const [isOpen, setIsOpen] = useState(false);
  const [isUnlocked, setIsUnlocked] = useState(false);
  const [passwordInput, setPasswordInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const [input, setInput] = useState("");
  const { messages, sendMessage, status, error } = useChat();

  const isLoading = status === "submitted" || status === "streaming";

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInput(e.target.value);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;
    sendMessage({ text: input });
    setInput("");
  };

  const handlePasswordSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (passwordInput.toLowerCase() === "del26") {
      setIsUnlocked(true);
    } else {
      setPasswordInput("");
    }
  };

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  return (
    <>
      {/* ── Floating Toggle Button ── */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`fixed bottom-6 right-6 z-50 h-14 w-14 rounded-full flex items-center justify-center shadow-lg transition-all duration-300 cursor-pointer ${isOpen ? "bg-gray-800 text-white" : "bg-blue-600 hover:bg-blue-500 text-white"
          }`}
        title="Toggle Vian AI"
      >
        {isOpen ? <X size={24} /> : <BrainCircuit size={24} />}
      </button>

      {/* ── Chat Window ── */}
      {isOpen && (
        <div
          className="fixed bottom-24 right-2 sm:right-6 w-[calc(100vw-16px)] sm:w-[400px] h-[600px] max-h-[80vh] backdrop-blur-2xl bg-black/80 border border-white/10 rounded-2xl flex flex-col shadow-[0_0_40px_rgba(0,0,0,0.5)] overflow-hidden z-[100]"
        >
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-white/10 bg-white/[0.03]">
            <div className="flex items-center gap-2">
              <BrainCircuit size={18} className="text-blue-400" />
              <span className="font-bold text-white">Vian AI</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
              {isUnlocked ? (
                <Unlock size={14} className="text-emerald-400" />
              ) : (
                <Lock size={14} className="text-red-500" />
              )}
            </div>
          </div>

          {/* Body */}
          <div className="flex-1 overflow-y-auto px-5 py-6 space-y-6 flex flex-col relative w-full h-full text-white min-h-0">
            {!isUnlocked ? (
              <div className="flex-1 flex flex-col items-center justify-center w-full">
                <Lock size={48} className="text-red-500 mb-4 opacity-80" />
                <h3 className="text-red-500 font-mono text-[10px] md:text-xs font-bold tracking-widest mb-6 text-center leading-relaxed px-4">
                  Vian AI TERMINAL LOCKED.<br />ENTER CLEARANCE CODE.
                </h3>
                <form onSubmit={handlePasswordSubmit} className="flex flex-col gap-3 w-full max-w-[220px]">
                  <input
                    type="password"
                    value={passwordInput}
                    onChange={(e) => setPasswordInput(e.target.value)}
                    placeholder="PASSWORD..."
                    className="w-full bg-black/50 border border-red-900/50 rounded px-3 py-2 text-xs font-mono text-red-500 text-center outline-none focus:border-red-500 transition-colors placeholder:text-red-900/50"
                    autoFocus
                  />
                  <button type="submit" className="w-full bg-red-900/30 hover:bg-red-900/60 text-red-500 text-xs font-mono font-bold py-2 rounded transition-colors border border-red-900/50">
                    AUTHENTICATE
                  </button>
                </form>
              </div>
            ) : (
              <div className="flex flex-col gap-4">
                {messages.length === 0 && (
                  <div className="text-center text-[11px] text-white/40 my-6 font-mono border border-white/10 bg-white/5 p-3 rounded mx-4">
                    {">"} RAG SYSTEMS ONLINE.<br /> {">"} KNOWLEDGE BASE LOADED.
                  </div>
                )}
                {messages.map((m: any) => {
                  const isUser = m.role === 'user';
                  return (
                    <div
                      key={m.id}
                      className={isUser
                        ? 'text-sm whitespace-pre-wrap bg-blue-600 text-white px-5 py-3 rounded-2xl rounded-tr-sm w-fit max-w-[85%] ml-auto shadow-md leading-relaxed'
                        : 'text-sm whitespace-pre-wrap bg-[#1a1a24] text-gray-200 px-5 py-3 rounded-2xl rounded-tl-sm w-fit max-w-[85%] mr-auto border border-white/5 shadow-sm leading-relaxed'
                      }
                    >
                      {m.parts
                        ? m.parts.map((p: any, i: number) => (p.type === 'text' ? <span key={i}>{p.text}</span> : null))
                        : m.text || m.content}
                    </div>
                  );
                })}
                {isLoading && (
                  <div className="flex justify-start w-full">
                    <div className="bg-white/5 text-gray-400 border border-white/5 rounded-2xl rounded-tl-sm px-4 py-3 text-[0.8rem] font-mono animate-pulse mr-auto w-fit max-w-[85%]">
                      Vian AI is typing...
                    </div>
                  </div>
                )}
                {error && <div className="text-red-500 p-4 font-mono text-sm">Error: {error.message}</div>}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>

          {/* Input Footer */}
          {isUnlocked && (
            <div className="p-4 bg-black/60 backdrop-blur-md border-t border-white/10">
              <form onSubmit={handleSubmit} className="flex items-center gap-2 bg-[#111116] border border-white/10 p-1.5 rounded-full w-full">
                <input
                  className="flex-1 bg-transparent px-4 py-2 text-sm text-white focus:outline-none placeholder:text-white/30"
                  value={input}
                  onChange={handleInputChange}
                  placeholder="Ask Vian AI..."
                />
                <button
                  type="submit"
                  disabled={isLoading || !input?.trim()}
                  className="w-9 h-9 flex items-center justify-center bg-blue-600 rounded-full hover:bg-blue-500 shrink-0 text-white transition-colors disabled:opacity-50 disabled:hover:bg-blue-600 cursor-pointer"
                >
                  <Send size={16} />
                </button>
              </form>
            </div>
          )}
        </div>
      )}
    </>
  );
}
