"use client";

import { useState, useEffect } from "react";
import { LayoutGrid, Network, Briefcase, User } from "lucide-react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import UniverseGrid from "@/components/UniverseGrid";
import GraphView from "@/components/GraphView";
import ProjectsView from "@/components/ProjectsView";
import type { NoteListItem, GraphData, ProjectListItem } from "@/lib/markdown";

interface HomeDashboardProps {
  notes: NoteListItem[];
  graphData: GraphData;
  projects: ProjectListItem[];
}

export default function HomeDashboard({
  notes,
  graphData,
  projects,
}: HomeDashboardProps) {
  const [view, setView] = useState<"grid" | "graph" | "projects">("graph");
  const searchParams = useSearchParams();

  // ── Sync View State from URL Query ──
  useEffect(() => {
    const urlView = searchParams.get("view");
    if (urlView === "projects") {
      setView("projects");
    }
  }, [searchParams]);

  return (
    <div className="flex flex-col w-full" style={{ minHeight: "85vh" }}>
      {/* ── Top Bar: Title + Toggle ── */}
      <div
        className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4 px-4 pt-8 pb-8"
      >
        {/* Branding */}
        <div>
          <p
            style={{
              fontSize: "0.65rem",
              fontWeight: 600,
              letterSpacing: "0.25em",
              textTransform: "uppercase",
              color: "rgba(255,255,255,0.25)",
              marginBottom: "0.4rem",
            }}
          >
            Risk · Quant · Frameworks
          </p>
          <h1
            style={{
              fontSize: "clamp(1.6rem, 4vw, 2.4rem)",
              fontWeight: 700,
              letterSpacing: "-0.02em",
              lineHeight: 1.1,
              color: "#fff",
              margin: 0,
              display: "flex",
              alignItems: "center",
              gap: "1rem",
            }}
          >
            Credit Risk OS
            
            {/* ── OS Sync Mastery Pill ── */}
            <div
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "0.5rem",
                padding: "0.25rem 0.75rem",
                borderRadius: "99px",
                background: "rgba(16, 185, 129, 0.1)",
                border: "1px solid rgba(16, 185, 129, 0.2)",
                boxShadow: "0 0 12px rgba(16, 185, 129, 0.15)",
                transform: "translateY(-2px)",
              }}
            >
              {/* Pulsing Dot */}
              <div
                style={{
                  width: "6px",
                  height: "6px",
                  borderRadius: "50%",
                  background: "#34d399",
                  boxShadow: "0 0 8px #34d399",
                  animation: "pulse-green 2s infinite ease-in-out",
                }}
              />
              <style>{`
                @keyframes pulse-green {
                  0%   { opacity: 0.5; transform: scale(0.9); }
                  50%  { opacity: 1; transform: scale(1.2); }
                  100% { opacity: 0.5; transform: scale(0.9); }
                }
              `}</style>
              
              <span
                style={{
                  fontSize: "0.75rem",
                  fontWeight: 700,
                  letterSpacing: "0.1em",
                  color: "#34d399",
                  textTransform: "uppercase",
                }}
              >
                OS Sync: {notes.length > 0 ? Math.round(notes.reduce((acc, n) => acc + (n.frontmatter.progress || 0), 0) / notes.length) : 0}%
              </span>
            </div>
          </h1>
          <p
            style={{
              marginTop: "0.4rem",
              fontSize: "0.82rem",
              color: "rgba(255,255,255,0.35)",
              fontWeight: 500,
            }}
          >
            A living knowledge graph.{" "}
            <span style={{ color: "rgba(255,255,255,0.55)" }}>
              {notes.length} nodes indexed.
            </span>
          </p>
        </div>

        {/* View Toggle Pills */}
        <div
          style={{
            display: "flex",
            gap: "0.25rem",
            padding: "0.25rem",
            borderRadius: "0.75rem",
            background: "rgba(255,255,255,0.04)",
            border: "1px solid rgba(255,255,255,0.08)",
            alignSelf: "flex-start",
          }}
        >
          <button
            onClick={() => setView("grid")}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "0.4rem",
              padding: "0.45rem 0.85rem",
              borderRadius: "0.55rem",
              fontSize: "0.75rem",
              fontWeight: 600,
              border: "none",
              cursor: "pointer",
              transition: "all 0.2s ease",
              background:
                view === "grid"
                  ? "rgba(255,255,255,0.1)"
                  : "transparent",
              color:
                view === "grid"
                  ? "#fff"
                  : "rgba(255,255,255,0.3)",
            }}
          >
            <LayoutGrid size={14} />
            Grid
          </button>
          <button
            onClick={() => setView("graph")}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "0.4rem",
              padding: "0.45rem 0.85rem",
              borderRadius: "0.55rem",
              fontSize: "0.75rem",
              fontWeight: 600,
              border: "none",
              cursor: "pointer",
              transition: "all 0.2s ease",
              background:
                view === "graph"
                  ? "rgba(255,255,255,0.1)"
                  : "transparent",
              color:
                view === "graph"
                  ? "#fff"
                  : "rgba(255,255,255,0.3)",
            }}
          >
            <Network size={14} />
            Graph
          </button>
          <button
            onClick={() => setView("projects")}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "0.4rem",
              padding: "0.45rem 0.85rem",
              borderRadius: "0.55rem",
              fontSize: "0.75rem",
              fontWeight: 600,
              border: "none",
              cursor: "pointer",
              transition: "all 0.2s ease",
              background:
                view === "projects"
                  ? "rgba(255,255,255,0.1)"
                  : "transparent",
              color:
                view === "projects"
                  ? "#fff"
                  : "rgba(255,255,255,0.3)",
            }}
          >
            <Briefcase size={14} />
            Projects
          </button>

          <div style={{ width: "1px", background: "rgba(255,255,255,0.1)", margin: "0.25rem", alignSelf: "stretch" }} />

          <Link
            href="/notes/Tharun-Kumar-Gajula"
            style={{
              display: "flex",
              alignItems: "center",
              gap: "0.4rem",
              padding: "0.45rem 0.85rem",
              borderRadius: "0.55rem",
              fontSize: "0.75rem",
              fontWeight: 600,
              border: "none",
              cursor: "pointer",
              transition: "all 0.2s ease",
              background: "transparent",
              color: "rgba(255,255,255,0.3)",
              textDecoration: "none",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.color = "#fff";
              e.currentTarget.style.background = "rgba(255,255,255,0.05)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.color = "rgba(255,255,255,0.3)";
              e.currentTarget.style.background = "transparent";
            }}
          >
            <User size={14} />
            Profile
          </Link>
        </div>
      </div>

      {/* ── Content Area ── */}
      <div
        className="flex-1 w-full relative overflow-hidden rounded-2xl mx-auto"
        style={{
          border: "1px solid rgba(255,255,255,0.06)",
          background: "rgba(5,5,5,0.3)",
          minHeight: "75vh",
        }}
      >
        {view === "grid" ? (
          <div
            className="w-full h-full overflow-y-auto p-4"
            style={{ minHeight: "75vh" }}
          >
            <UniverseGrid notes={notes} />
          </div>
        ) : view === "graph" ? (
          <div style={{ height: "75vh", width: "100%" }}>
            <GraphView graphData={graphData} />
          </div>
        ) : (
          <ProjectsView projects={projects} />
        )}
      </div>
    </div>
  );
}
