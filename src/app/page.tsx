import { Suspense } from "react";
import { getAllNotes, getGraphData, getAllProjects } from "@/lib/markdown";
import HomeDashboard from "@/components/HomeDashboard";

export const metadata = {
  title: "Quant OS",
  description: "A Zettelkasten-powered knowledge base for credit risk.",
};

export default async function Home() {
  const [notes, graphData, projects] = await Promise.all([
    getAllNotes(),
    getGraphData(),
    getAllProjects(),
  ]);

  return (
    <div style={{ position: "relative" }}>

      {/* ── Ambient orbs — FIXED position, truly behind everything ── */}
      <div
        aria-hidden
        style={{
          position: "fixed",
          top: "-120px",
          left: "-80px",
          width: "600px",
          height: "600px",
          borderRadius: "50%",
          background:
            "radial-gradient(circle, rgba(59,130,246,0.22) 0%, transparent 65%)",
          filter: "blur(90px)",
          pointerEvents: "none",
          zIndex: 0,
        }}
      />
      <div
        aria-hidden
        style={{
          position: "fixed",
          top: "10%",
          right: "-100px",
          width: "500px",
          height: "500px",
          borderRadius: "50%",
          background:
            "radial-gradient(circle, rgba(99,102,241,0.2) 0%, transparent 65%)",
          filter: "blur(100px)",
          pointerEvents: "none",
          zIndex: 0,
        }}
      />
      <div
        aria-hidden
        style={{
          position: "fixed",
          bottom: "0",
          left: "30%",
          width: "420px",
          height: "420px",
          borderRadius: "50%",
          background:
            "radial-gradient(circle, rgba(20,184,166,0.14) 0%, transparent 65%)",
          filter: "blur(110px)",
          pointerEvents: "none",
          zIndex: 0,
        }}
      />

      {/* ── Page content ── */}
      <main
        style={{
          position: "relative",
          zIndex: 1,
          width: "100%",
          padding: "1rem",
        }}
      >
        <Suspense fallback={<div style={{ padding: "2rem", color: "rgba(255,255,255,0.5)", fontSize: "0.85rem", textAlign: "center" }}>Initializing neural map...</div>}>
          <HomeDashboard notes={notes} graphData={graphData} projects={projects} />
        </Suspense>
      </main>
    </div>
  );
}
