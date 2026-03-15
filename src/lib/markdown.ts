import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';

// brain/ lives at the project root, one level above src/
const BRAIN_DIR = path.join(process.cwd(), 'brain');
const PORTFOLIO_DIR = path.join(process.cwd(), 'brain', 'portfolio');

export interface NoteFrontmatter {
  title: string;
  date: string;
  cluster: string;
  tags?: string[];
  [key: string]: unknown;
}

export interface NoteListItem {
  slug: string;
  frontmatter: NoteFrontmatter;
}

export interface NoteDetail {
  slug: string;
  frontmatter: NoteFrontmatter;
  content: string;        // wikilinks already converted to markdown links
  rawContent: string;     // original content (for backlink scanning)
}

export interface ProjectFrontmatter {
  title: string;
  date: string;
  summary: string;
  tags?: string[];
  [key: string]: unknown;
}

export interface ProjectListItem {
  slug: string;
  frontmatter: ProjectFrontmatter;
}

export interface ProjectDetail {
  slug: string;
  frontmatter: ProjectFrontmatter;
  content: string;
}

export interface BacklinkItem {
  slug: string;
  title: string;
  cluster?: string;
}

// ─── Wikilink → Markdown link converter ─────────────────────────────
// Converts [[Some Concept Name]] → [Some Concept Name](/notes/Some-Concept-Name)
// Also handles [[slug|Display Text]] → [Display Text](/notes/slug)
function convertWikilinks(markdown: string): string {
  return markdown.replace(
    /\[\[([^\]|]+?)(?:\|([^\]]+?))?\]\]/g,
    (_match, target: string, display?: string) => {
      const slug = target.trim().replace(/\s+/g, '-');
      const label = (display ?? target).trim();
      return `[${label}](/notes/${slug})`;
    }
  );
}

// ─── Public API ─────────────────────────────────────────────────────

/**
 * Scans brain/ and returns an array of { slug, frontmatter } for every .md file.
 * Returns an empty array if the directory is missing or contains no .md files.
 */
export async function getAllNotes(): Promise<NoteListItem[]> {
  if (!fs.existsSync(BRAIN_DIR)) {
    console.warn(`[markdown] brain/ directory not found at: ${BRAIN_DIR}`);
    return [];
  }

  // Explicitly ignore the portfolio directory (readdir non-recursive handles it, but just to be safe)
  const files = fs.readdirSync(BRAIN_DIR).filter((f) => f.endsWith('.md'));

  if (files.length === 0) {
    console.warn('[markdown] brain/ directory exists but contains no .md files.');
    return [];
  }

  return files.map((filename) => {
    const slug = filename.replace(/\.md$/, '');
    const raw = fs.readFileSync(path.join(BRAIN_DIR, filename), 'utf-8');
    const { data } = matter(raw);

    return {
      slug,
      frontmatter: data as NoteFrontmatter,
    };
  });
}

/**
 * Reads a single .md file by slug and returns its frontmatter,
 * wikilink-converted content, and raw content for backlink scanning.
 * Returns null if the file does not exist.
 */
export async function getNoteBySlug(slug: string): Promise<NoteDetail | null> {
  const filePath = path.join(BRAIN_DIR, `${slug}.md`);

  if (!fs.existsSync(filePath)) {
    console.warn(`[markdown] File not found: ${filePath}`);
    return null;
  }

  const raw = fs.readFileSync(filePath, 'utf-8');
  const { data, content } = matter(raw);

  return {
    slug,
    frontmatter: data as NoteFrontmatter,
    content: convertWikilinks(content),
    rawContent: content,
  };
}

/**
 * Finds all notes that contain a [[wikilink]] pointing to the given slug.
 * Compares against slug (hyphenated filename) and title (spaced display name).
 */
export async function getBacklinks(currentSlug: string): Promise<BacklinkItem[]> {
  if (!fs.existsSync(BRAIN_DIR)) return [];

  const files = fs.readdirSync(BRAIN_DIR).filter((f) => f.endsWith('.md'));
  const backlinks: BacklinkItem[] = [];

  // Match patterns: [[Current-Slug]] or [[Current Slug]] or [[Title With Spaces]]
  // We build a set of possible link targets for this slug
  const displayName = currentSlug.replace(/-/g, ' ');
  const targets = new Set([
    currentSlug,                        // e.g. "Probability-of-Default"
    displayName,                        // e.g. "Probability of Default"
    currentSlug.replace(/-/g, '-'),     // identity, just in case
  ]);

  for (const filename of files) {
    const otherSlug = filename.replace(/\.md$/, '');

    // Don't count self-references
    if (otherSlug === currentSlug) continue;

    const raw = fs.readFileSync(path.join(BRAIN_DIR, filename), 'utf-8');
    const { data, content } = matter(raw);

    // Check if the raw content contains any [[target]] wikilink pointing to us
    const hasBacklink = Array.from(targets).some((t) => {
      // Case-insensitive check for [[target]]
      const re = new RegExp(`\\[\\[${escapeRegex(t)}(\\|[^\\]]*)?\\]\\]`, 'i');
      return re.test(content);
    });

    if (hasBacklink) {
      backlinks.push({
        slug: otherSlug,
        title: (data as NoteFrontmatter).title ?? otherSlug,
        cluster: (data as NoteFrontmatter).cluster,
      });
    }
  }

  return backlinks;
}

// ─── Graph Data Types & Builder ─────────────────────────────────────

export interface GraphNode {
  id: string;
  name: string;
  val: number;
  group: string;
}

export interface GraphLink {
  source: string;
  target: string;
}

export interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

/**
 * Builds a force-graph dataset: nodes from all notes, links from [[wikilinks]].
 */
export async function getGraphData(): Promise<GraphData> {
  if (!fs.existsSync(BRAIN_DIR)) return { nodes: [], links: [] };

  const files = fs.readdirSync(BRAIN_DIR).filter((f) => f.endsWith('.md'));
  const nodes: GraphNode[] = [];
  const links: GraphLink[] = [];
  const slugSet = new Set<string>();

  // First pass: build node list and slug set
  for (const filename of files) {
    const slug = filename.replace(/\.md$/, '');
    const raw = fs.readFileSync(path.join(BRAIN_DIR, filename), 'utf-8');
    const { data } = matter(raw);
    const fm = data as NoteFrontmatter;

    slugSet.add(slug);
    nodes.push({
      id: slug,
      name: fm.title ?? slug,
      val: 1,
      group: fm.cluster ?? 'Uncategorized',
    });
  }

  // Second pass: extract [[wikilinks]] to build links
  for (const filename of files) {
    const sourceSlug = filename.replace(/\.md$/, '');
    const raw = fs.readFileSync(path.join(BRAIN_DIR, filename), 'utf-8');
    const { content } = matter(raw);

    // Find all [[Target]] or [[Target|Display]] wikilinks
    const wikiRe = /\[\[([^\]|]+?)(?:\|[^\]]*?)?\]\]/g;
    let m: RegExpExecArray | null;

    const seen = new Set<string>(); // dedupe links within same file
    while ((m = wikiRe.exec(content)) !== null) {
      const targetSlug = m[1].trim().replace(/\s+/g, '-');
      // Only create link if target exists and isn't self
      if (targetSlug !== sourceSlug && slugSet.has(targetSlug) && !seen.has(targetSlug)) {
        seen.add(targetSlug);
        links.push({ source: sourceSlug, target: targetSlug });
      }
    }
  }

  return { nodes, links };
}

/** Escape special regex characters in a string */
function escapeRegex(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// ─── Portfolio Data Fetching ────────────────────────────────────────

/**
 * Scans brain/portfolio/ and returns an array of { slug, frontmatter } for every .md file.
 * Specific for distinct Project representations.
 */
export async function getAllProjects(): Promise<ProjectListItem[]> {
  if (!fs.existsSync(PORTFOLIO_DIR)) {
    console.warn(`[markdown] Portfolio directory not found at: ${PORTFOLIO_DIR}`);
    return [];
  }

  const files = fs.readdirSync(PORTFOLIO_DIR).filter((f) => f.endsWith('.md'));

  if (files.length === 0) {
    console.warn('[markdown] Portfolio directory exists but contains no .md files.');
    return [];
  }

  return files.map((filename) => {
    const slug = filename.replace(/\.md$/, '');
    const raw = fs.readFileSync(path.join(PORTFOLIO_DIR, filename), 'utf-8');
    const { data } = matter(raw);

    return {
      slug,
      frontmatter: data as ProjectFrontmatter,
    };
  });
}

/**
 * Reads a single .md file from the portfolio directory by slug.
 * Used for dynamic routing rendering the Projects view pages.
 */
export async function getProjectBySlug(slug: string): Promise<ProjectDetail | null> {
  const filePath = path.join(PORTFOLIO_DIR, `${slug}.md`);

  if (!fs.existsSync(filePath)) {
    console.warn(`[markdown] Portfolio file not found: ${filePath}`);
    return null;
  }

  const raw = fs.readFileSync(filePath, 'utf-8');
  const { data, content } = matter(raw);

  return {
    slug,
    frontmatter: data as ProjectFrontmatter,
    content: convertWikilinks(content),
  };
}
