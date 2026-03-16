"use client";

import { useRef, useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import dynamic from "next/dynamic";
import { ZoomIn, ZoomOut, Target } from "lucide-react";
import type { GraphData } from "@/lib/markdown";

// CRITICAL: react-force-graph-2d uses Canvas + window — must disable SSR
const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), {
  ssr: false,
  loading: () => (
    <div
      style={{
        height: "100%",
        width: "100%",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        color: "rgba(255,255,255,0.2)",
        fontSize: "0.8rem",
        fontFamily: "var(--font-geist-mono), monospace",
        letterSpacing: "0.1em",
      }}
    >
      Initializing neural map…
    </div>
  ),
});

// ── Premium cluster color palette ──
const CLUSTER_COLORS: Record<string, string> = {
  "System Atlas": "#ffffff",
  "Phase 1: The Macro Picture": "#60a5fa",
  "Phase 2: Data Architecture": "#a78bfa",
  "Phase 3: The Algorithmic Engine": "#2dd4bf",
  "Phase 4: Model Validation": "#f472b6",
  "Phase 5: Production Monitoring": "#fbbf24",
  "Phase 6: Deployment Lifecycle": "#34d399",
  "Quantitative Portfolio": "#f87171",
  Others: "#9ca3af",
};

function getColor(group: string, id: string): string {
  if (id === "Tharun-Kumar-Gajula") return "#ffffff"; // Distinct white color for profile
  return CLUSTER_COLORS[group] ?? "#9ca3af";
}

function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

interface GraphViewProps {
  graphData: GraphData;
}

export default function GraphView({ graphData }: GraphViewProps) {
  const router = useRouter();
  const fgRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);

  // ── Sizing to fill container ──
  useEffect(() => {
    function measure() {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        setDimensions({
          width: rect.width,
          height: rect.height,
        });
      }
    }
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, []);

  const handleNodeClick = useCallback(
    (node: any) => {
      if (node?.id) {
        const isLocked = node.group === "Phase 5: Production Monitoring" || node.group === "Phase 6: Deployment Lifecycle";
        if (!isLocked) {
          router.push(`/notes/${node.id}`);
        }
      }
    },
    [router]
  );

  // ── View Controls ──
  const zoomIn = () => {
    if (!fgRef.current) return;
    const currentZoom = fgRef.current.zoom();
    fgRef.current.zoom(currentZoom * 1.5, 400);
  };

  const zoomOut = () => {
    if (!fgRef.current) return;
    const currentZoom = fgRef.current.zoom();
    fgRef.current.zoom(currentZoom / 1.5, 400);
  };

  const resetView = () => {
    if (!fgRef.current) return;
    fgRef.current.zoomToFit(800, 60);
  };

  return (
    <div
      ref={containerRef}
      style={{ position: "relative", width: "100%", height: "100%" }}
    >
      {/* ── Graph Canvas ── */}
      <ForceGraph2D
        ref={fgRef}
        graphData={graphData}
        width={dimensions.width}
        height={dimensions.height}
        backgroundColor="transparent"
        /* ── Node painting with small permanent labels ── */
        nodeCanvasObject={(node, ctx, globalScale) => {
          const n = node as {
            id: string;
            name: string;
            group: string;
            x?: number;
            y?: number;
            fx?: number;
            fy?: number;
          };

          // Fix the central node strictly to the center of the universe
          if (n.id === "Tharun-Kumar-Gajula") {
            n.fx = 0;
            n.fy = 0;
          }

          const x = n.x ?? 0;
          const y = n.y ?? 0;
          const color = getColor(n.group, n.id);
          const isHovered = hoveredNode === n.id;
          const isCentralNode = n.id === "Tharun-Kumar-Gajula";
          
          const baseRadius = isCentralNode ? 8 : 4.5;
          const hoverRadius = isCentralNode ? 10 : 7;
          const radius = isHovered ? hoverRadius : baseRadius;

          // Outer glow
          const glowRadius = isHovered ? (isCentralNode ? 24 : 18) : (isCentralNode ? 16 : 10);
          const grad = ctx.createRadialGradient(x, y, 0, x, y, glowRadius);
          grad.addColorStop(0, hexToRgba(color, isHovered ? 0.35 : 0.12));
          grad.addColorStop(1, hexToRgba(color, 0));
          ctx.beginPath();
          ctx.arc(x, y, glowRadius, 0, 2 * Math.PI);
          ctx.fillStyle = grad;
          ctx.fill();

          // Core dot
          ctx.beginPath();
          ctx.arc(x, y, radius, 0, 2 * Math.PI);
          ctx.fillStyle = color;
          ctx.fill();

          // Specular highlight
          ctx.beginPath();
          ctx.arc(x, y - radius * 0.25, radius * 0.35, 0, 2 * Math.PI);
          ctx.fillStyle = "rgba(255,255,255,0.25)";
          ctx.fill();

          // ── Small permanent label ──
          const fontSize = isHovered
            ? Math.min(13 / globalScale, 11)
            : Math.min(10 / globalScale, 7);
          ctx.font = `${isHovered ? 600 : 400} ${fontSize}px Inter, system-ui, sans-serif`;
          ctx.textAlign = "center";
          ctx.textBaseline = "top";
          ctx.fillStyle = isHovered
            ? "rgba(255,255,255,0.92)"
            : "rgba(255,255,255,0.28)";

          if (isHovered) {
            ctx.shadowColor = "rgba(0,0,0,0.8)";
            ctx.shadowBlur = 4;
          }
          
          const isLocked = n.group === "Phase 5: Production Monitoring" || n.group === "Phase 6: Deployment Lifecycle";
          const displayName = isLocked ? `🔒 ${n.name}` : n.name;
          
          ctx.fillText(displayName, x, y + radius + 3);
          ctx.shadowBlur = 0;
        }}
        nodePointerAreaPaint={(node, color, ctx) => {
          const n = node as { x?: number; y?: number };
          ctx.beginPath();
          ctx.arc(n.x ?? 0, n.y ?? 0, 12, 0, 2 * Math.PI);
          ctx.fillStyle = color;
          ctx.fill();
        }}
        /* ── Link styling ── */
        linkColor={() => "rgba(255,255,255,0.06)"}
        linkWidth={0.8}
        /* ── Interactions ── */
        onNodeClick={handleNodeClick}
        onNodeHover={(node: any) =>
          setHoveredNode(node ? node.id : null)
        }
        nodeLabel={(node) => (node as { name: string }).name}
        /* ── Physics ── */
        cooldownTicks={120}
        d3AlphaDecay={0.02}
        d3VelocityDecay={0.3}
      />

      {/* ── Zoom Controls (Bottom Right) ── */}
      <div className="absolute bottom-28 right-6 flex flex-col gap-2 z-20">
        {[
          { icon: <Target size={16} />, fn: resetView, label: "Reset" },
          { icon: <ZoomIn size={16} />, fn: zoomIn, label: "Zoom In" },
          { icon: <ZoomOut size={16} />, fn: zoomOut, label: "Zoom Out" },
        ].map((btn) => (
          <button
            key={btn.label}
            onClick={btn.fn}
            title={btn.label}
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              width: "2.25rem",
              height: "2.25rem",
              borderRadius: "0.6rem",
              border: "1px solid rgba(255,255,255,0.08)",
              background: "rgba(5,5,5,0.6)",
              backdropFilter: "blur(12px)",
              color: "rgba(255,255,255,0.5)",
              cursor: "pointer",
              transition: "all 0.15s ease",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.color = "#fff";
              e.currentTarget.style.background = "rgba(255,255,255,0.08)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.color = "rgba(255,255,255,0.5)";
              e.currentTarget.style.background = "rgba(5,5,5,0.6)";
            }}
          >
            {btn.icon}
          </button>
        ))}
      </div>
    </div>
  );
}
