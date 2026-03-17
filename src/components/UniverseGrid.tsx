"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { Lock } from "lucide-react";
import type { NoteListItem } from "@/lib/markdown";

interface UniverseGridProps {
  notes: NoteListItem[];
}

/* ── Cluster palette ── */
const PALETTE: Record<string, { badge: string; glowColor: string }> = {
  "00 — System Atlas": { badge: "text-blue-300 bg-blue-500/10 border-blue-400/25", glowColor: "rgba(59,130,246,0.07)" },
  "01 — End-to-End Credit Risk Modeling": { badge: "text-emerald-300 bg-emerald-500/10 border-emerald-400/25", glowColor: "rgba(16,185,129,0.07)" },
  "02 — Model Monitoring, Validation, and Stability": { badge: "text-teal-300 bg-teal-500/10 border-teal-400/25", glowColor: "rgba(20,184,166,0.07)" },
  "03 — Loss, Exposure, and Recovery Modeling": { badge: "text-pink-300 bg-pink-500/10 border-pink-400/25", glowColor: "rgba(236,72,153,0.07)" },
  "04 — Applied Machine Learning Projects": { badge: "text-amber-300 bg-amber-500/10 border-amber-400/25", glowColor: "rgba(245,158,11,0.07)" },
  "05 — Sandbox Environment": { badge: "text-gray-300 bg-gray-500/10 border-gray-400/25", glowColor: "rgba(156,163,175,0.07)" },
  "Others": { badge: "text-gray-300 bg-gray-500/10 border-gray-400/25", glowColor: "rgba(156,163,175,0.07)" },
};
const DEFAULT = { badge: "text-indigo-300 bg-indigo-500/10 border-indigo-400/25", glowColor: "rgba(99,102,241,0.07)" };

function palette(cluster?: string) {
  if (!cluster) return DEFAULT;
  return PALETTE[cluster] ?? DEFAULT;
}

/* ── Stagger variants ── */
const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.09, delayChildren: 0.05 } },
};

const card = {
  hidden: { opacity: 0, y: 20 },
  show: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.5,
      ease: [0.25, 0.46, 0.45, 0.94] as [number, number, number, number],
    },
  },
};

export default function UniverseGrid({ notes }: UniverseGridProps) {
  if (notes.length === 0) {
    return (
      <p style={{ textAlign: "center", color: "rgba(240,240,245,0.35)", marginTop: "4rem", fontSize: "0.875rem" }}>
        No notes found in <code>brain/</code>. Create a Markdown file to get started.
      </p>
    );
  }

  const sortedNotes = [...notes].sort((a, b) => {
    const clusterA = a.frontmatter.cluster || "Others";
    const clusterB = b.frontmatter.cluster || "Others";
    
    if (clusterA !== clusterB) {
      if (clusterA === "Others") return 1;
      if (clusterB === "Others") return -1;
      return clusterA.localeCompare(clusterB);
    }
    
    const titleA = a.frontmatter.title || a.slug;
    const titleB = b.frontmatter.title || b.slug;
    return titleA.localeCompare(titleB);
  });

  return (
    <motion.ul
      variants={container}
      initial="hidden"
      animate="show"
      style={{
        listStyle: "none",
        display: "grid",
        gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
        gap: "1.25rem",
        padding: 0,
        margin: 0,
        width: "100%",
      }}
    >
      {sortedNotes.map((note) => {
        const { badge, glowColor } = palette(note.frontmatter.cluster);

        return (
          <motion.li key={note.slug} variants={card} style={{ minWidth: 0 }}>
            <Link 
              href={`/notes/${note.slug}`} 
              style={{ textDecoration: "none", display: "block", height: "100%" }}
            >

              {/* ── Glass Card ── */}
              <div
                className="group"
                style={{
                  position: "relative",
                  display: "flex",
                  flexDirection: "column",
                  minHeight: "190px",
                  height: "100%",
                  padding: "1.5rem",
                  borderRadius: "1.5rem",
                  overflow: "hidden",
                  /* Glass material */
                  backdropFilter: "blur(24px)",
                  WebkitBackdropFilter: "blur(24px)",
                  background: "rgba(255,255,255,0.03)",
                  border: "1px solid rgba(255,255,255,0.08)",
                  boxShadow: "0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.07)",
                  transition: "background 0.4s ease, border-color 0.4s ease, box-shadow 0.4s ease",
                  cursor: "pointer",
                  opacity: 1,
                }}
                onMouseEnter={(e) => {
                  const el = e.currentTarget;
                  el.style.background = "rgba(255,255,255,0.055)";
                  el.style.borderColor = "rgba(255,255,255,0.14)";
                  el.style.boxShadow = "0 12px 48px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.12), 0 0 40px rgba(255,255,255,0.04)";
                }}
                onMouseLeave={(e) => {
                  const el = e.currentTarget;
                  el.style.background = "rgba(255,255,255,0.03)";
                  el.style.borderColor = "rgba(255,255,255,0.08)";
                  el.style.boxShadow = "0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.07)";
                }}
              >
                {/* Cluster-tinted glow inside card */}
                <div
                  aria-hidden
                  style={{
                    position: "absolute",
                    inset: 0,
                    background: `radial-gradient(ellipse at top left, ${glowColor} 0%, transparent 55%)`,
                    pointerEvents: "none",
                    borderRadius: "1.5rem",
                  }}
                />

                {/* Top specular edge */}
                <div
                  aria-hidden
                  style={{
                    position: "absolute",
                    top: 0, left: "10%", right: "10%",
                    height: "1px",
                    background: "linear-gradient(90deg, transparent, rgba(255,255,255,0.14), transparent)",
                    pointerEvents: "none",
                  }}
                />

                {/* ── Top row: badge + arrow ── */}
                <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: "0.5rem", marginBottom: "0.875rem", position: "relative" }}>
                  {note.frontmatter.cluster ? (
                    <span
                      className={[
                        "inline-flex items-center px-2.5 py-0.5 rounded-full border",
                        "text-[10px] font-bold uppercase tracking-[0.18em]",
                        badge,
                      ].join(" ")}
                    >
                      {note.frontmatter.cluster}
                    </span>
                  ) : <span />}

                  <span style={{ color: "rgba(255,255,255,0.2)", fontSize: "1.1rem", lineHeight: 1, transition: "color 0.25s" }} aria-hidden>↗</span>
                </div>

                {/* ── Title ── */}
                <h2
                  style={{
                    position: "relative",
                    fontSize: "1rem",
                    fontWeight: 600,
                    lineHeight: 1.45,
                    color: "rgba(255,255,255,0.82)",
                    margin: 0,
                  }}
                >
                  {note.frontmatter.title ?? note.slug}
                </h2>

                {/* ── Spacer: pushes footer to bottom ── */}
                <div style={{ flex: 1 }} />
                
                {/* ── Progress Bar ── */}
                <div style={{ marginTop: "1rem" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.25rem" }}>
                    <span style={{ fontSize: "0.6rem", fontWeight: 600, color: "rgba(255,255,255,0.4)", textTransform: "uppercase", letterSpacing: "0.05em" }}>Reading Sync</span>
                    <span style={{ fontSize: "0.6rem", fontWeight: 700, color: "rgba(255,255,255,0.7)" }}>{note.frontmatter.progress || 0}%</span>
                  </div>
                  <div style={{ height: "4px", width: "100%", background: "rgba(255,255,255,0.05)", borderRadius: "2px", overflow: "hidden" }}>
                    <div 
                      style={{ 
                        height: "100%", 
                        width: `${note.frontmatter.progress || 0}%`, 
                        background: "linear-gradient(90deg, rgba(255,255,255,0.2), #fff)", 
                        boxShadow: "0 0 8px rgba(255,255,255,0.5)",
                        transition: "width 0.8s cubic-bezier(0.16, 1, 0.3, 1)"
                      }} 
                    />
                  </div>
                </div>

                {/* ── Footer row ── */}
                <div
                  style={{
                    position: "relative",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    gap: "0.5rem",
                    marginTop: "1.25rem",
                    paddingTop: "0.875rem",
                    borderTop: "1px solid rgba(255,255,255,0.06)",
                  }}
                >
                  <span style={{ fontSize: "0.7rem", color: "rgba(240,240,245,0.3)", lineHeight: 1 }}>
                    {note.frontmatter.date
                      ? new Date(note.frontmatter.date).toLocaleDateString("en-US", {
                          year: "numeric", month: "short", day: "numeric",
                        })
                      : "—"}
                  </span>
                  <span style={{ fontFamily: "monospace", fontSize: "0.65rem", color: "rgba(240,240,245,0.18)", lineHeight: 1 }}>
                    {note.slug}
                  </span>
                </div>

              </div>
            </Link>
          </motion.li>
        );
      })}
    </motion.ul>
  );
}
