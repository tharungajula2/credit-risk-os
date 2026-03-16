# Quant OS - Architecture & Context

**Project Name:** Quant OS
**Phase:** Core Initialized (Strike 15 Save State)

## Tech Stack
- **Framework:** Next.js (App Router)
- **Styling:** Tailwind CSS v4, custom CSS-in-JS (via inline styles for precise dynamic bounds)
- **Animations:** dynamic inline CSS transitions
- **Visualization:** `react-force-graph-2d`
- **Markdown Processing:** `gray-matter`, `next-mdx-remote`, `remark-math`, `remark-gfm`, `rehype-katex`, `rehype-pretty-code`
- **Database:** Local Obsidian Vault (`/brain` directory) containing raw Markdown files

## Core System Architecture

Quant OS operates essentially as a headless knowledge-graph interface built over a local ecosystem of Markdown files. It functions similarly to Obsidian but lives natively on the web.

### 1. The Brain (Database)
The `brain/` directory acts as the sole database. Every `.md` file represents a node in the knowledge graph. These files contain:
- Standard YAML Frontmatter (`title`, `date`, `tags`, `cluster` mapped to Phase 1-6 categories).
- Raw markdown content encompassing text, MathJax equations, code blocks, and standard tables (purged of any narrative hooks to maintain academic purity).
- `[[Wikilinks]]` pointing to other notes within the vault. The markdown parser natively strips `.md` extensions for graph resilience.

### 1.5 The Shelf (Workspace Utilities)
The `shelf/temp_files/` directory acts as an explicit quarantine zone for auxiliary scripts (Node.js parsers, link fixers, AI hydration scripts). This guarantees the root directory remains strictly focused on Next.js (`package.json`, `next.config.ts`, `app/`).

### 2. The Local Knowledge Engine (`lib/markdown.ts`)
This subsystem is the backbone of the OS, responsible for translating static files into a relational database:
- **`getAllNotes()`**: Traverses the `brain/` directory, extracts frontmatter using `gray-matter`, calculates word counts, and generates URL-friendly slugs.
- **Wikilink Parsing**: During note retrieval, Regex pipelines intercept `[[Linked Note]]` syntax and convert it to standard markdown links `[Linked Note](/notes/Linked-Note)`, matching the generated slug format.
- **`getBacklinks(currentSlug)`**: Calculates bi-directional relationships dynamically by scanning the raw content of all notes for wikilinks pointing to the active note.
- **`getGraphData()`**: Computes the 2D network topology, generating an array of `nodes` and mapping `links` derived from wikilink strings to node IDs.

### 3. The UI Architecture and Layout
The OS layout consists of three primary, non-overlapping pillars functioning in a fluid flexbox layout:
- **RootLayout (`app/layout.tsx`)**: Establishes a flex wrapper to coordinate the `Sidebar` and the main `flex-1` content view.
- **The Navigation Engine (`components/Sidebar.tsx`)**: The control center. It intercepts `getAllNotes()`, grouping files dynamically by `cluster`. It features a desktop-only toggle that relies on flex-shrink (`width` from `18rem` to `0`) pushing the main canvas cleanly.
- **The Graph Canvas (`components/GraphView.tsx`)**: The primary interface. By wrapping `react-force-graph-2d` inside a constrained 75vh layout block (`HomeDashboard.tsx`), it maps nodes into an interactive physics simulation. The visual appearance of nodes is deeply customized via Canvas HTML5 API (`nodeCanvasObject`), providing colored glowing orbs and permanent text labels.
- **The Reading View (`app/notes/[slug]/page.tsx`)**: Standard notes are hydrated dynamically using `next-mdx-remote`. The `remark` and `rehype` processing pipelines parse tables, equations, and code syntax, styled deeply with standard Tailwind Typography (`prose-table`), producing structured Obsidian-grade tables.

## Purpose of this File
**DO NOT DELETE.**
This document exists to provide the complete, current architectural context to any new AI agent interacting with this repository. Development can be resumed seamlessly by reading this document as the ground truth of the application's structure, syntax processing patterns, layout constraints, CSS definitions, and data models without requiring the original system prompts or chat history.
