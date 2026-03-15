"use client";

import { useState, useMemo, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Search, FileText, X, Brain, ChevronLeft, ChevronRight } from "lucide-react";
import type { NoteListItem } from "@/lib/markdown";

interface SidebarProps {
  notes: NoteListItem[];
}

export default function Sidebar({ notes }: SidebarProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [isOpen, setIsOpen] = useState(false);
  const [isDesktopOpen, setIsDesktopOpen] = useState(true);
  const pathname = usePathname();

  // ── Listen for toggle-sidebar events from MobileHeader ──
  useEffect(() => {
    const handler = () => setIsOpen((prev) => !prev);
    window.addEventListener("toggle-sidebar", handler);
    return () => window.removeEventListener("toggle-sidebar", handler);
  }, []);

  // Close on route change (mobile)
  useEffect(() => {
    setIsOpen(false);
  }, [pathname]);

  // ── Filter + group by cluster ──
  const grouped = useMemo(() => {
    const q = searchQuery.toLowerCase().trim();
    const filtered = q
      ? notes.filter((n) =>
          (n.frontmatter.title ?? n.slug).toLowerCase().includes(q)
        )
      : notes;

    const groups: Record<string, NoteListItem[]> = {};
    for (const note of filtered) {
      const cluster = note.frontmatter.cluster || "Uncategorized";
      if (!groups[cluster]) groups[cluster] = [];
      groups[cluster].push(note);
    }

    return Object.entries(groups).sort(([a], [b]) => {
      if (a === "Uncategorized") return 1;
      if (b === "Uncategorized") return -1;
      return a.localeCompare(b);
    });
  }, [notes, searchQuery]);

  const totalFiltered = grouped.reduce((sum, [, items]) => sum + items.length, 0);

  /* ═══════════════════════════════════
     Shared sidebar inner content
     ═══════════════════════════════════ */
  const sidebarContent = (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        overflow: "hidden",
      }}
    >
      {/* ── Premium ID Card Branding (Top) ── */}
      <Link href="/" style={{ textDecoration: "none", display: "block" }}>
        <div
          style={{
            padding: "1.25rem 1.25rem 1rem",
            background: "linear-gradient(180deg, rgba(255,255,255,0.03) 0%, transparent 100%)",
            borderBottom: "1px solid rgba(255,255,255,0.06)",
            transition: "background 0.2s ease",
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = "linear-gradient(180deg, rgba(45,105,230,0.08) 0%, transparent 100%)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = "linear-gradient(180deg, rgba(255,255,255,0.03) 0%, transparent 100%)";
          }}
        >
        <div 
          style={{
            display: "flex",
            alignItems: "center",
            gap: "0.75rem",
          }}
        >
          {/* Avatar / Initial */}
          <div
            style={{
              width: "2.5rem",
              height: "2.5rem",
              borderRadius: "0.6rem",
              background: "linear-gradient(135deg, rgba(96,165,250,0.2), rgba(96,165,250,0.05))",
              border: "1px solid rgba(96,165,250,0.3)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexShrink: 0,
            }}
          >
            <span style={{ fontSize: "1.1rem", fontWeight: 700, color: "#60a5fa" }}>T</span>
          </div>
          
          {/* Details */}
          <div>
            <p
              style={{
                margin: 0,
                fontSize: "0.6rem",
                color: "rgba(255,255,255,0.4)",
                letterSpacing: "0.06em",
                textTransform: "uppercase",
                fontWeight: 600,
              }}
            >
              Architected by
            </p>
            <p
              style={{
                margin: "0.1rem 0 0",
                fontSize: "0.85rem",
                fontWeight: 700,
                color: "rgba(255,255,255,0.95)",
                letterSpacing: "-0.01em",
              }}
            >
              Tharun Gajula
            </p>
            <p
              style={{
                margin: "0.1rem 0 0",
                fontSize: "0.65rem",
                fontWeight: 500,
                color: "#60a5fa",
                opacity: 0.9,
              }}
            >
              AI & Risk Quant
            </p>
          </div>
        </div>
      </div>
      </Link>

      {/* ── Search ── */}
      <div style={{ padding: "0.75rem 1rem" }}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "0.5rem",
            padding: "0.45rem 0.65rem",
            borderRadius: "0.5rem",
            background: "rgba(255,255,255,0.035)",
            border: "1px solid rgba(255,255,255,0.06)",
            transition: "border-color 0.2s ease",
          }}
        >
          <Search
            size={13}
            style={{ color: "rgba(255,255,255,0.2)", flexShrink: 0 }}
          />
          <input
            type="text"
            placeholder="Search notes…"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            style={{
              flex: 1,
              background: "transparent",
              border: "none",
              outline: "none",
              fontSize: "0.73rem",
              color: "rgba(255,255,255,0.65)",
              width: "100%",
              fontFamily: "inherit",
            }}
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery("")}
              style={{
                background: "none",
                border: "none",
                cursor: "pointer",
                padding: 0,
                color: "rgba(255,255,255,0.2)",
                display: "flex",
              }}
            >
              <X size={12} />
            </button>
          )}
        </div>
        {searchQuery && (
          <p
            style={{
              fontSize: "0.6rem",
              color: "rgba(255,255,255,0.2)",
              marginTop: "0.3rem",
              paddingLeft: "0.25rem",
              fontFamily: "var(--font-geist-mono), monospace",
            }}
          >
            {totalFiltered} result{totalFiltered !== 1 ? "s" : ""}
          </p>
        )}
      </div>

      {/* ── Note list (scrollable) ── */}
      <nav
        className="sidebar-scroll"
        style={{
          flex: 1,
          overflowY: "auto",
          padding: "0.25rem 0.75rem 1.5rem",
        }}
      >
        {grouped.length === 0 ? (
          <p
            style={{
              fontSize: "0.72rem",
              color: "rgba(255,255,255,0.15)",
              textAlign: "center",
              marginTop: "2rem",
            }}
          >
            No matching notes.
          </p>
        ) : (
          grouped.map(([cluster, items]) => (
            <div key={cluster} style={{ marginBottom: "1.1rem" }}>
              {/* Cluster subheader */}
              <p
                style={{
                  fontSize: "10px",
                  fontWeight: 700,
                  letterSpacing: "0.15em",
                  textTransform: "uppercase",
                  color: "#6b7280",
                  padding: "0.4rem 0.5rem 0.25rem",
                  marginBottom: "0.15rem",
                }}
              >
                {cluster}
              </p>

              {/* Note links */}
              {items.map((note) => {
                const href = `/notes/${note.slug}`;
                const isActive = pathname === href;

                return (
                  <Link
                    key={note.slug}
                    href={href}
                    onClick={() => setIsOpen(false)}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "0.45rem",
                      padding: "0.625rem 0.5rem",
                      borderRadius: "0.375rem",
                      fontSize: "0.875rem",
                      fontWeight: isActive ? 600 : 450,
                      textDecoration: "none",
                      color: isActive
                        ? "rgba(255,255,255,0.95)"
                        : "rgba(255,255,255,0.38)",
                      background: isActive
                        ? "rgba(96,165,250,0.1)"
                        : "transparent",
                      borderLeft: isActive
                        ? "2px solid rgba(96,165,250,0.6)"
                        : "2px solid transparent",
                      transition: "all 0.15s ease",
                      lineHeight: 1.35,
                    }}
                    onMouseEnter={(e) => {
                      if (!isActive) {
                        e.currentTarget.style.background =
                          "rgba(255,255,255,0.03)";
                        e.currentTarget.style.color =
                          "rgba(255,255,255,0.75)";
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (!isActive) {
                        e.currentTarget.style.background = "transparent";
                        e.currentTarget.style.color =
                          "rgba(255,255,255,0.38)";
                      }
                    }}
                  >
                    <FileText
                      size={12}
                      style={{
                        flexShrink: 0,
                        opacity: isActive ? 0.7 : 0.25,
                      }}
                    />
                    <span
                      style={{
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {note.frontmatter.title ?? note.slug}
                    </span>
                  </Link>
                );
              })}
            </div>
          ))
        )}
      </nav>


    </div>
  );

  /* ═══════════════════════════════════
     Render
     ═══════════════════════════════════ */
  return (
    <>
      {/* ── Desktop sidebar (always visible ≥ 1024px) ── */}
      <aside
        style={{
          position: "sticky",
          top: 0,
          height: "100vh",
          width: isDesktopOpen ? "18rem" : "0",
          zIndex: 40,
          background: "rgba(5,5,5,0.97)",
          backdropFilter: "blur(24px)",
          WebkitBackdropFilter: "blur(24px)",
          borderRight: isDesktopOpen ? "1px solid rgba(255,255,255,0.05)" : "none",
          transition: "width 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
          flexShrink: 0,
          alignSelf: "flex-start",
        }}
        className="hidden lg:block relative"
      >
        {/* Toggle Button Container */}
        <div 
          style={{
            position: "absolute",
            top: "1.25rem",
            right: "-0.5rem",
            zIndex: 50,
            transform: "translateX(100%)"
          }}
        >
          <button
            onClick={() => setIsDesktopOpen(!isDesktopOpen)}
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              width: "2rem",
              height: "2rem",
              background: "rgba(5,5,5,0.8)",
              border: "1px solid rgba(255,255,255,0.1)",
              borderRadius: "0.5rem",
              color: "rgba(255,255,255,0.5)",
              cursor: "pointer",
              backdropFilter: "blur(12px)",
              transition: "all 0.2s ease"
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.color = "#fff";
              e.currentTarget.style.background = "rgba(255,255,255,0.05)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.color = "rgba(255,255,255,0.5)";
              e.currentTarget.style.background = "rgba(5,5,5,0.8)";
            }}
          >
            {isDesktopOpen ? <ChevronLeft size={16} /> : <ChevronRight size={16} />}
          </button>
        </div>

        {/* Content wrapper with fixed width to prevent squishing during transition */}
        <div style={{ 
          width: "18rem", 
          height: "100%", 
          overflow: "hidden", 
          opacity: isDesktopOpen ? 1 : 0, 
          transition: "opacity 0.2s ease",
          display: "flex",
          flexDirection: "column"
        }}>
          {sidebarContent}
        </div>
      </aside>

      {/* ── Mobile: Overlay backdrop ── */}
      {isOpen && (
        <div
          onClick={() => setIsOpen(false)}
          style={{
            position: "fixed",
            inset: 0,
            zIndex: 45,
            background: "rgba(0,0,0,0.65)",
            backdropFilter: "blur(6px)",
          }}
          className="lg:hidden"
        />
      )}

      {/* ── Mobile: Slide-in drawer ── */}
      <aside
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          bottom: 0,
          width: "18rem",
          zIndex: 50,
          background: "rgba(5,5,5,0.98)",
          backdropFilter: "blur(24px)",
          WebkitBackdropFilter: "blur(24px)",
          borderRight: "1px solid rgba(255,255,255,0.05)",
          transform: isOpen ? "translateX(0)" : "translateX(-100%)",
          transition: "transform 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
        }}
        className="lg:hidden flex flex-col"
      >
        {/* Close button */}
        <div
          style={{
            display: "flex",
            justifyContent: "flex-end",
            padding: "0.75rem",
          }}
        >
          <button
            onClick={() => setIsOpen(false)}
            style={{
              background: "none",
              border: "none",
              cursor: "pointer",
              color: "rgba(255,255,255,0.4)",
              padding: "0.35rem",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              borderRadius: "0.375rem",
            }}
          >
            <X size={20} />
          </button>
        </div>
        {sidebarContent}
      </aside>
    </>
  );
}
