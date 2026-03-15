"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { ProjectListItem } from "@/lib/markdown";

interface ProjectsViewProps {
  projects: ProjectListItem[];
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

export default function ProjectsView({ projects }: ProjectsViewProps) {
  if (!projects || projects.length === 0) {
    return (
      <div className="flex h-full w-full items-center justify-center text-gray-500 font-mono text-sm">
        No projects found in brain/portfolio/
      </div>
    );
  }

  return (
    <div
      className="w-full h-full overflow-y-auto p-4 md:p-8"
      style={{ minHeight: "75vh" }}
    >
      <motion.div 
        variants={container}
        initial="hidden"
        animate="show"
        className="max-w-4xl mx-auto flex flex-col gap-6 pb-20"
      >
        {projects.map((project) => {
          const glowColor = "rgba(99,102,241,0.07)"; // Indigo glow for projects

          return (
            <motion.div key={project.slug} variants={card} className="group relative block w-full">
              <Link
                href={`/portfolio/${project.slug}`}
                style={{ textDecoration: "none", display: "block" }}
              >
                {/* ── Glass Card matching UniverseGrid ── */}
                <div
                  style={{
                    position: "relative",
                    display: "flex",
                    flexDirection: "column",
                    padding: "2rem",
                    borderRadius: "1.5rem",
                    overflow: "hidden",
                    backdropFilter: "blur(24px)",
                    WebkitBackdropFilter: "blur(24px)",
                    background: "rgba(255,255,255,0.03)",
                    border: "1px solid rgba(255,255,255,0.08)",
                    boxShadow: "0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.07)",
                    transition: "background 0.4s ease, border-color 0.4s ease, box-shadow 0.4s ease",
                    cursor: "pointer",
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
                  {/* Tinted glow inside card */}
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

                  {/* ── Top row: Title + arrow ── */}
                  <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: "0.5rem", marginBottom: "0.875rem", position: "relative" }}>
                    <h2
                      style={{
                        position: "relative",
                        fontSize: "1.3rem",
                        fontWeight: 600,
                        lineHeight: 1.45,
                        color: "rgba(255,255,255,0.82)",
                        margin: 0,
                      }}
                      className="group-hover:text-indigo-300 transition-colors"
                    >
                      {project.frontmatter.title}
                    </h2>
                    <span style={{ color: "rgba(255,255,255,0.2)", fontSize: "1.1rem", lineHeight: 1, transition: "color 0.25s" }}
                      className="group-hover:text-white"
                      aria-hidden>↗</span>
                  </div>

                  {/* ── Summary ── */}
                  <p 
                    style={{
                      position: "relative",
                      fontSize: "0.95rem",
                      color: "rgba(255,255,255,0.55)",
                      lineHeight: 1.6,
                      margin: "0 0 1.5rem 0",
                    }}
                  >
                    {project.frontmatter.summary}
                  </p>

                  <div style={{ flex: 1 }} />

                  {/* ── Tags and Date Footer ── */}
                  <div
                    style={{
                      position: "relative",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      flexWrap: "wrap",
                      gap: "0.5rem",
                      marginTop: "0.5rem",
                      paddingTop: "1rem",
                      borderTop: "1px solid rgba(255,255,255,0.06)",
                    }}
                  >
                    <div className="flex flex-wrap gap-2">
                      {project.frontmatter.tags && project.frontmatter.tags.map((tag) => (
                        <span
                          key={tag}
                          className="inline-flex items-center px-2.5 py-0.5 rounded-full border text-[10px] font-bold uppercase tracking-[0.18em] text-indigo-300 bg-indigo-500/10 border-indigo-400/25"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>

                    <span style={{ fontSize: "0.75rem", fontFamily: "monospace", color: "rgba(240,240,245,0.3)", lineHeight: 1 }}>
                      {project.frontmatter.date 
                        ? new Date(project.frontmatter.date).toLocaleDateString("en-US", {
                            year: "numeric", month: "short", day: "numeric"
                          })
                        : "—"}
                    </span>
                  </div>
                </div>
              </Link>
            </motion.div>
          );
        })}
      </motion.div>
    </div>
  );
}
