import { MDXRemote } from "next-mdx-remote/rsc";
import { getProjectBySlug } from "@/lib/markdown";
import { notFound } from "next/navigation";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";

/* ── Plugins ── */
import remarkMath from "remark-math";
import remarkGfm from "remark-gfm";
import remarkBreaks from "remark-breaks";
import rehypeKatex from "rehype-katex";
import rehypePrettyCode from "rehype-pretty-code";

/* ── KaTeX CSS ── */
import "katex/dist/katex.min.css";

interface PortfolioPageProps {
  params: Promise<{ slug: string }>;
}

export default async function PortfolioPage({ params }: PortfolioPageProps) {
  const { slug } = await params;
  const project = await getProjectBySlug(slug);

  if (!project) notFound();

  const { frontmatter, content } = project;
  const tags: string[] = Array.isArray(frontmatter.tags) ? frontmatter.tags : [];

  // ── Safeguard MDX compilation ──
  // MDX will crash if it sees "<0", "<1", etc. because it tries to parse a JSX tag.
  // We globally replace "< followed by a digit" with "< space digit".
  const safeContent = content.replace(/<(\d)/g, '< $1');

  return (
    <div
      className="selection:bg-indigo-500/30"
      style={{ color: "#d1d5db" }}
    >
      <div
        style={{
          maxWidth: "56rem",
          margin: "0 auto",
          padding: "3rem 1.5rem",
          width: "100%",
        }}
      >
        {/* ── Back link ── */}
        <Link
          href="/?view=projects"
          className="group"
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: "0.4rem",
            marginBottom: "3rem",
            fontSize: "0.8rem",
            fontWeight: 500,
            letterSpacing: "0.03em",
            color: "rgba(255,255,255,0.4)",
            textDecoration: "none",
            transition: "color 0.2s ease",
          }}
        >
          <ArrowLeft
            size={14}
            style={{ transition: "transform 0.2s ease" }}
            className="group-hover:-translate-x-0.5"
          />
          Back to Top Level
        </Link>

        {/* ══════════════════════════════════════════════════════════════
            PORTFOLIO PROPERTIES PANEL
            ══════════════════════════════════════════════════════════════ */}
        <div
          style={{
            marginBottom: "2.5rem",
            padding: "1.5rem",
            borderRadius: "1rem",
            background: "rgba(255,255,255,0.025)",
            border: "1px solid rgba(255,255,255,0.07)",
            backdropFilter: "blur(12px)",
            boxShadow: "0 12px 48px rgba(0,0,0,0.2), inset 0 1px 0 rgba(255,255,255,0.05)",
          }}
        >
          {/* Title */}
          <h1
            style={{
              fontSize: "clamp(1.8rem, 5vw, 2.8rem)",
              fontWeight: 700,
              letterSpacing: "-0.02em",
              lineHeight: 1.15,
              color: "#ffffff",
              margin: 0,
            }}
          >
            {frontmatter.title ?? slug}
          </h1>

          <p
            style={{
              marginTop: "1.25rem",
              fontSize: "1.05rem",
              lineHeight: 1.6,
              color: "rgba(255,255,255,0.65)",
              fontWeight: 500,
            }}
          >
            {frontmatter.summary}
          </p>

          {/* Metadata rows */}
          <div
            style={{
              marginTop: "2rem",
              display: "flex",
              flexDirection: "column",
              gap: "0.8rem",
              paddingTop: "1.5rem",
              borderTop: "1px solid rgba(255,255,255,0.08)",
            }}
          >
            {/* Date */}
            {frontmatter.date && (
              <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
                <span style={{ fontSize: "0.7rem", color: "rgba(255,255,255,0.3)", textTransform: "uppercase", letterSpacing: "0.15em", fontWeight: 600, minWidth: "4rem" }}>
                  Timeline
                </span>
                <span style={{ fontSize: "0.85rem", color: "rgba(240,240,245,0.7)", fontFamily: "monospace" }}>
                  {new Date(frontmatter.date).toLocaleDateString("en-US", {
                    year: "numeric",
                    month: "long",
                    day: "numeric",
                  })}
                </span>
              </div>
            )}

            {/* Tags */}
            {tags.length > 0 && (
              <div style={{ display: "flex", alignItems: "flex-start", gap: "0.75rem" }}>
                <span style={{ fontSize: "0.7rem", color: "rgba(255,255,255,0.3)", textTransform: "uppercase", letterSpacing: "0.15em", fontWeight: 600, minWidth: "4rem", paddingTop: "0.2rem" }}>
                  Tech Stack
                </span>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "0.35rem" }}>
                  {tags.map((tag) => (
                    <span
                      key={tag}
                      style={{
                        display: "inline-block",
                        padding: "0.2rem 0.6rem",
                        borderRadius: "999px",
                        border: "1px solid rgba(99,102,241,0.25)",
                        background: "rgba(99,102,241,0.08)",
                        fontSize: "0.68rem",
                        color: "#a5b4fc",
                        fontWeight: 600,
                        letterSpacing: "0.12em",
                        textTransform: "uppercase",
                      }}
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* ══════════════════════════════════════════════════════════════
            MDX ARTICLE
            ══════════════════════════════════════════════════════════════ */}
        <article
          className={[
            "prose prose-invert prose-lg max-w-none",
            "prose-headings:text-white prose-headings:font-bold",
            "prose-h1:text-4xl",
            "prose-h2:text-2xl prose-h2:mt-12 prose-h2:border-b prose-h2:border-white/10 prose-h2:pb-2",
            "prose-a:text-indigo-400 prose-a:no-underline hover:prose-a:underline",
            "prose-strong:text-white",
            "prose-code:before:hidden prose-code:after:hidden prose-code:bg-white/10 prose-code:text-indigo-300 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded-md prose-code:font-normal",
            "prose-pre:bg-[#0d1117] prose-pre:border prose-pre:border-white/10 prose-pre:rounded-xl prose-pre:p-4 prose-pre:w-full prose-pre:overflow-x-auto",
            "prose-th:text-white prose-td:text-gray-300",
            "prose-blockquote:border-l-indigo-500/40 prose-blockquote:text-gray-400",
            "prose-hr:border-white/10",
            "prose-ul:list-disc",
            "prose-li:text-gray-300",
            "prose-table:w-full prose-table:border-collapse prose-table:border prose-table:border-white/10 prose-th:bg-white/5 prose-th:p-3 prose-th:border-b prose-th:border-white/10 prose-th:text-left prose-td:p-3 prose-td:border-b prose-td:border-white/5",
          ].join(" ")}
        >
          <MDXRemote
            source={safeContent}
            options={{
              mdxOptions: {
                remarkPlugins: [remarkMath, remarkGfm, remarkBreaks],
                rehypePlugins: [
                  rehypeKatex,
                  [
                    rehypePrettyCode,
                    {
                      theme: "github-dark",
                      keepBackground: true,
                    },
                  ],
                ],
              },
            }}
          />
        </article>
      </div>
    </div>
  );
}
