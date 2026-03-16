# CREDIT RISK OS - SYSTEM ARCHITECTURE & CONTEXT
Credit Risk OS is an advanced spatial learning environment and professional quantitative risk portfolio showcase. It utilizes a local Markdown-driven Zettelkasten system as a headless CMS to render complex risk knowledge and project case studies into a 2D physics graph and interactive grid layout.

## 1. Tech Stack & Core Libraries
- **Next.js (v16.1.6 App Router)**: The core React framework providing server-side rendering, dynamic filesystem routing (`app/notes/[slug]` and `app/portfolio/[slug]`), and static build optimizations.
- **Tailwind CSS v4 & @tailwindcss/typography**: Used for all baseline styling and precise typography overrides, creating the dark void layout and rendering markdown content akin to Obsidian.
- **Framer Motion**: Powers the hardware-accelerated stagger animations, layout transitions, and fluid card interactions across the grid views.
- **react-force-graph-2d**: A client-side canvas renderer that transforms markdown internal links into a visually interactive 2D neural network physics map.
- **Vercel AI SDK (v6)**: Powers the `CrosAIChat` component, utilizing `ToolLoopAgent` and `createAgentUIStreamResponse` for an integrated, context-aware AI chatbot assistant running on Google's `gemini-2.5-flash-lite`.
- **gray-matter**: Parses frontmatter (YAML) from local `.md` files to extract metadata.
- **next-mdx-remote**: The rendering engine converting raw markdown body content into styled React Server Components dynamically.
- **Remark & Rehype Ecosystem (`remark-gfm`, `remark-math`, `rehype-katex`, `rehype-pretty-code`)**: A suite of markdown plugins crucial for parsing complex mathematical LaTeX equations, GitHub-flavored tables, and syntax-highlighted code blocks natively.

## 2. Directory Structure & Data Flow
The system operates completely file-based without an external database, reading from local Markdown structures:
- `brain/`: Contains the core Zettelkasten knowledge notes (e.g., Credit Risk concepts, probability models). Files here populate the Universe Grid and the active 2D Physics Graph. (The Feynman Hook narrative layer was manually purged from all 21 files to leave pure academic content).
- `brain/portfolio/`: Strictly separated from the core graph. Contains structured project summaries defining the user's professional portfolio to be rendered exclusively in the Projects view.
- `shelf/temp_files/`: A newly designated workspace folder added to `.gitignore`. It contains all auxiliary data processing scripts to keep the root directory pristine. MOST IMPORTANTLY, it contains **`format_frontmatter.js`** — a pure JavaScript executable designed to reliably and safely bulk-edit markdown YAML frontmatter without falling victim to regex/newline system anomalies.
- `src/lib/markdown.ts`: The absolute data layer of the app. It specifically exports separate logical getters (`getAllNotes()` vs `getAllProjects()`) ensuring standard notes and portfolio cases never bleed into each other. It handles the complex `convertWikilinks` logic.
- `src/app/notes/[slug]/page.tsx` & `src/app/portfolio/[slug]/page.tsx`: Dynamic server routes that fetch and render the specific markdown payload.

## 3. Core Features & UI/UX Guidelines
- **"Liquid Glass" Aesthetic**: The application follows a strict premium dark "void" constraint (`#050505` background). Interactions rely on deep Gaussian blurs (`backdropFilter: "blur(24px)"`), translucent surface whites (`rgba(255,255,255,0.03)`), ambient node glows, and specular highlights.
- **CROS AI Terminal**: A fully functional, password-protected (code: *del26*, case-insensitive) RAG Chat terminal floating in the UI. It features a stunning mobile-responsive 'Liquid Glass' chat interface with pixel-perfect padding (`px-5 py-3`), shadow-mapped user/AI bubbles, and a unified pill-shaped input form (`bg-[#111116] border-white/10`).
- **Navigation Architecture**: The `Sidebar` component acts as a `sticky` desktop navigation column and gracefully collapses into a bottom `MobileHeader` drawer on smaller viewports.
- **The Triptych View System**: `HomeDashboard.tsx` employs a 3-way React state toggle (managed via URL query parameters) allowing seamless pivoting between: Graph, Grid, and Projects.
- **Graph Control Synergy**: The Graph view's zoom and reset controls are carefully layered (`z-20`, `bottom-28`) avoiding collision with the floating CROS AI chat toggle dot.

## 4. Strict Engineering Constraints (For Future AI Agents)
- **DO NOT** alter the markdown parsing logic (`lib/markdown.ts`) without explicit permission.
- **DO NOT** use absolute positioning for main layout containers; rely on Flexbox/Grid to prevent overlaps (Exception: The floating Chat UI and Graph Controls are absolutely positioned with strict spacing rules to avoid collision).
- **DO NOT** change the Tailwind v4 base configuration.
- **MARKDOWN RULE**: All tables, code blocks, and math equations (`$$`) in the `.md` files must have a blank line above and below them to render correctly.

## 5. Recent Architectural Victories (March Hand-off)
- **Knowledge Base Refinement & Purge**: The 21+ node Zettelkasten was manually purged of extraneous narrative elements ("The Feynman Hook") to maintain a highly professional, academic tone strictly aligned with quantitative risk management.
- **Root Directory Cleanliness**: All legacy `.js` and `.bat` utility scripts were explicitly quarantined into a git-ignored `/shelf/temp_files/` directory, leaving the root perfectly clean with only Next.js execution logic.
- **Graph Link Resilience**: Upgraded the internal markdown parser (`src/lib/markdown.ts`). It now transparently strips `.md` extensions from internal `[[wikilinks]]`. This means if the user accidentally links `[[Probability-of-Default.md]]`, the graph engine will seamlessly clean the slug and maintain the physical 3D node connection without dropping it.
- **The Graph's Gravitational Center**: The profile node `Tharun-Kumar-Gajula.md` has been programmatically pinned to the exact mathematical center of the ForceGraph (`x: 0, y: 0`) and rendered entirely white, serving as the permanent anchoring star for the entire portfolio.
- **The 6-Phase Knowledge Journey**: The `cluster` YAML frontmatter properties across all 21 notes were updated to cleanly categorize the grid and color-code the 3D map into a structured 6-phase journey:
  1. Phase 1. Bank Loss Engine
  2. Phase 2. Regulatory Skeleton
  3. Phase 3. Core Credit Risk Trinity
  4. Phase 4. Model Build & Validate
  5. Phase 5. Hard Portfolios & Stress
  6. Phase 6. Broader Risk Domains
  7. Phase 7. Sandbox (Used for experimental learning notes and temporary ideations)
  8. Others

- **JS Ecosystem Standardization & Bulk Markdown Scripting**: Previously, bulk-updating Markdown files (like synchronizing dates or appending `progress: 0`) using terminal hooks like PowerShell or Python proved disastrous due to CLI execution swallowing and Windows/Unix `\r\n` line-end mismatches. The system now universally relies on pure node `fs` JavaScript execution. 
   - **CRITICAL NOTE FOR FUTURE SESSIONS**: If you ever need to bulk-modify brain markdown files or frontmatter, **DO NOT** use generic terminal commands. **ALWAYS** refer to and execute/modify `shelf/temp_files/format_frontmatter.js`. This script splits the document via robust native arrays, explicitly checking `---` bounds to manipulate frontmatter cleanly, representing the absolute gold-standard template for programmatic data entry in this repository.

## 6. CROS AI Chat Interface Learnings (UI/UX Engineering Triumphs)
Building the `CrosAIChat` overlay was a complex exercise in CSS architecture to ensure it integrated seamlessly over the 3D Graph and Universe Grid without breaking layout constraints. These final learnings represent what *perfectly* worked:
- **Liquid Glass Pixel Perfection**: The glassmorphic chat container achieves its premium aesthetic through a direct `backdropFilter: "blur(24px)"`, an ultra-thin border (`border-white/10`), and a highly translucent dark background (`bg-[#0a0a0c]/80` or `rgba(255,255,255,0.03)`). It is vital to maintain this exact class combination; solid, non-transparent hex colors immediately destroy the spatial depth illusion. Both user and AI message bubbles also rely on precise drop shadows (`shadow-[0_4px_12px_rgba(0,0,0,0.5)]`).
- **Z-Index & Canvas Layering**: The `react-force-graph-2d` canvas naturally dominates the DOM. The Chat Toggle Orb must maintain a strictly defined `z-50` via fixed positioning (`bottom-6 right-6`), and the open Chat Window must also sit in a fixed `z-50` overlay to avoid rendering beneath the physics engine.
- **Graph Control Synergy**: To avoid overlapping with the Chat Orb, the Graph controls (Zoom In/Out/Target) were strictly moved to `bottom-28 right-6` with `z-20`. **Do not change these coordinates**; they are the exact mathematical offset required to prevent collision.
- **Scroll Container Architecture**: The Chat Message area uses `overflow-y-auto` paired with specific height constraints (`max-h-[60vh]` on mobile, `max-h-[500px]` on desktop) inside the Liquid Glass pane to prevent the entire page window from scrolling out of bounds behind the conversation. 
- **Avoiding Absolute Layout Bleed**: Aside from the Chat UI and Graph Controls, the main layout strictly avoids `position: absolute`. It was confirmed that leveraging Tailwind's Flexbox and CSS Grid natively is the absolute requirement for the core dashboard to organically adapt without content bleeding heavily over the Sidebar or Mobile Header.

## 7. Critical Active Blocker: Vercel AI SDK v6 vs TypeScript (Chat UI)
**Current Status**: The `CrosAIChat.tsx` frontend UI is currently broken and failing the Next.js production build (`npm run build`). The chat UI is temporarily paused.

**The Problem**:
We recently migrated the backend `src/app/api/chat/route.ts` to utilize the brand new Vercel AI SDK **v6** features (`createAgentUIStreamResponse`, `ToolLoopAgent`). The backend successfully compiles and returns a 200 OK. 
However, the frontend `useChat()` hook from the older version of the `@ai-sdk/react` library is catastrophically clashing with the new v6 types.

**Specifically**:
`Type error: Property 'append' does not exist on type 'UseChatHelpers<UIMessage<unknown, UIDataTypes, UITools>>'`

This indicates a severe mismatch between how `ai` (the v6 backend package) defines messages vs how `@ai-sdk/react` (the frontend package) expects to consume them. 

**Next Chat Action Item**:
When resuming, the very first task MUST BE to definitively resolve the Vercel AI SDK v6 type mismatch in `CrosAIChat.tsx`. Do not attempt to style or polish the chat UI until `useChat()` is cleanly importing and mapping messages according to the strict v6 standard, without TypeScript throwing `append` or `m.parts` errors.

---

## 8. Vercel Deployment vs Localhost (The TypeScript Strictness Rule)
A critical learning regarding deploying updates (like Markdown frontmatter or new components): **Vercel will silently fail deployments and serve stale code if the Next.js `npm run build` process encounters a TypeScript error.**

- **The Illusion of `npm run dev`**: When testing locally via `npm run dev`, Next.js acts gracefully. It will swallow TypeScript errors and render your new files (e.g., `Phase 7. Sandbox`) seamlessly.
- **The Reality of Production (`npm run build`)**: Vercel triggers a strict production build. If *any* component in your project has a type mismatch (such as the `CrosAIChat.tsx` SDK v6 error), Vercel instantly aborts the deployment. It will not show a "Broken Page" to users; it simply refuses to update, leaving you wondering why your new Markdown files or UI updates aren't showing up live.
- **The Temporary Fix (The Nuclear Option)**: If you need to force a deployment through while a component is genuinely broken at the type level, you can inject `// @ts-nocheck` at the very top of the failing file. This temporarily mutes the typechecker, allows `npm run build` to pass, and forces Vercel to sync your new Markdown data/UI to production. 
- **Rule of Thumb**: Always run `npm run build` locally before pushing to GitHub if you are unsure why Vercel isn't updating.
