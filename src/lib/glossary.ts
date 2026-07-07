import { findPage } from "./content";

export interface GlossaryEntry {
  /** Canonical term as written in the glossary table. */
  term: string;
  /** Plain-text definition. */
  definition: string;
}

/** Strip the small set of inline markdown markers we expect in table cells. */
function stripInlineMarkdown(value: string): string {
  return value
    .replace(/`([^`]*)`/g, "$1") // inline code
    .replace(/\*\*([^*]+)\*\*/g, "$1") // bold
    .replace(/\*([^*]+)\*/g, "$1") // italic
    .replace(/__([^_]+)__/g, "$1") // bold (underscores)
    .replace(/_([^_]+)_/g, "$1") // italic (underscores)
    .trim();
}

/** Split a markdown table row into trimmed cell values. */
function parseRow(line: string): string[] | null {
  const trimmed = line.trim();
  if (!trimmed.startsWith("|")) return null;
  // Drop the leading/trailing pipe, then split on unescaped pipes.
  const inner = trimmed.replace(/^\|/, "").replace(/\|\s*$/, "");
  return inner.split("|").map((c) => c.trim());
}

/** A separator row looks like | :--- | :--- |. */
function isSeparatorRow(cells: string[]): boolean {
  return cells.every((c) => /^:?-{3,}:?$/.test(c.replace(/\s/g, "")));
}

let cache: Map<string, GlossaryEntry> | null = null;

/**
 * Parse the first two-column markdown table in the glossary page into a
 * case-insensitive lookup of term -> definition. Result is memoised.
 */
export function getGlossary(): Map<string, GlossaryEntry> {
  if (cache) return cache;

  const map = new Map<string, GlossaryEntry>();
  const page = findPage("glossary");
  if (!page) {
    cache = map;
    return map;
  }

  const lines = page.body.split(/\r?\n/);
  // The glossary page can hold several term tables (e.g. one per chapter).
  // `inTable` is true once we've passed a table's header separator; any
  // non-table line (blank line, heading, `---` rule) ends the current table
  // so the next table's header row is skipped instead of parsed as an entry.
  let inTable = false;

  for (const line of lines) {
    const cells = parseRow(line);
    if (!cells || cells.length < 2) {
      inTable = false;
      continue;
    }

    if (isSeparatorRow(cells)) {
      inTable = true;
      continue;
    }

    // Skip the literal "Term | Definition" header row that precedes each
    // separator — only data rows below a separator become entries.
    if (!inTable) continue;

    const term = stripInlineMarkdown(cells[0]);
    const definition = stripInlineMarkdown(cells[1]);
    if (!term || !definition) continue;

    map.set(term.toLowerCase(), { term, definition });
  }

  cache = map;
  return map;
}

/** Look up a definition by term (case-insensitive). */
export function lookupTerm(term: string): GlossaryEntry | undefined {
  return getGlossary().get(term.trim().toLowerCase());
}

const HTML_ESCAPE: Record<string, string> = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
};

function escapeHtml(value: string): string {
  return value.replace(/[&<>"]/g, (c) => HTML_ESCAPE[c]);
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

let patternCache: RegExp | null = null;

/**
 * One regex that matches any glossary term (longest first so multi-word terms
 * win), captured in group 1, with an optional trailing percent marker in
 * group 2. Word boundaries keep "Markdown" from matching inside "Markdownish".
 */
function buildPattern(glossary: Map<string, GlossaryEntry>): RegExp {
  if (patternCache) return patternCache;
  const terms = [...glossary.values()]
    .map((e) => e.term)
    .sort((a, b) => b.length - a.length)
    .map(escapeRegExp);
  patternCache = new RegExp(
    `(?<![\\p{L}\\p{N}])((?:${terms.join("|")})s?)(?![\\p{L}\\p{N}])(%?)`,
    "giu",
  );
  return patternCache;
}

function spanFor(entry: GlossaryEntry, displayed: string): string {
  return `<span class="glossary-term" data-term="${escapeHtml(
    entry.term,
  )}">${escapeHtml(displayed)}</span>`;
}

function processSegment(
  text: string,
  glossary: Map<string, GlossaryEntry>,
  pattern: RegExp,
  linked: Set<string>,
): string {
  pattern.lastIndex = 0;
  return text.replace(pattern, (full, termText: string, pct: string) => {
    // Only an explicit `%` marker creates a link — no automatic matching, so
    // there are never false positives on ordinary prose.
    if (pct !== "%") return full;
    const key = termText.replace(/\s+/g, " ").trim().toLowerCase();

    let entry = glossary.get(key);
    let canonicalKey = key;

    // Fallback: If not found and ends with 's', try the singular form
    if (!entry && key.endsWith("s")) {
      canonicalKey = key.slice(0, -1);
      entry = glossary.get(canonicalKey);
    }

    if (!entry) return full;
    linked.add(canonicalKey);
    return spanFor(entry, termText);
  });
}

/** Resolve a term key (with singular fallback) and record it as linked. */
function resolveEntry(
  rawTerm: string,
  glossary: Map<string, GlossaryEntry>,
  linked: Set<string>,
): GlossaryEntry | undefined {
  const key = rawTerm.replace(/\s+/g, " ").trim().toLowerCase();
  let entry = glossary.get(key);
  let canonicalKey = key;
  if (!entry && key.endsWith("s")) {
    canonicalKey = key.slice(0, -1);
    entry = glossary.get(canonicalKey);
  }
  if (!entry) return undefined;
  linked.add(canonicalKey);
  return entry;
}

/**
 * Matches an inline-formatted token (code, bold, or italic) whose `%` marker
 * sits either just inside the closing delimiter (`` `term%` ``, `*term%*`) or
 * just outside it (`` `term`% ``, `*term*%`). Group 1 is the delimiter, group 2
 * the inner text, group 3 the optional trailing marker.
 */
const FORMATTED_MARKER = /(\*\*|__|\*|_|`)([\s\S]+?)\1(%?)/g;

function processLine(
  line: string,
  glossary: Map<string, GlossaryEntry>,
  pattern: RegExp,
  linked: Set<string>,
): string {
  // Leave markdown links untouched; everything else is processed.
  const parts = line.split(/(\[[^\]]*\]\([^)]*\))/g);
  for (let i = 0; i < parts.length; i++) {
    if (i % 2 === 1) continue; // odd parts are links

    // 1) Formatted terms with the `%` marker inside or outside the delimiters.
    parts[i] = parts[i].replace(
      FORMATTED_MARKER,
      (full, delim: string, inner: string, trailing: string) => {
        let term: string;
        if (inner.endsWith("%")) {
          term = inner.slice(0, -1);
        } else if (trailing === "%") {
          term = inner;
        } else {
          return full; // no marker — leave the formatting for markdown to render
        }
        const entry = resolveEntry(term.trim(), glossary, linked);
        if (!entry) return full;
        return spanFor(entry, `${delim}${term}${delim}`);
      },
    );

    // 2) Plain-text terms with a trailing `%`.
    parts[i] = processSegment(parts[i], glossary, pattern, linked);
    // Unescape \% to %
    parts[i] = parts[i].replace(/\\%/g, "%");
  }
  return parts.join("");
}

/**
 * Turn glossary terms in a markdown source into glossary `<span>` HTML.
 *
 * - Only the explicit `Term%` marker creates a link; the `%` itself is removed
 *   from the output. There is no automatic matching, so ordinary prose never
 *   produces false-positive links.
 * - Matching is case-insensitive and multi-word terms win over shorter ones.
 * - Headings, table rows, fenced code, inline code, and links are skipped.
 */
export function applyGlossaryMarkers(source: string): string {
  const glossary = getGlossary();
  if (glossary.size === 0) return source;

  const pattern = buildPattern(glossary);
  const linked = new Set<string>();

  const lines = source.split(/\r?\n/);
  let inFence = false;
  for (let i = 0; i < lines.length; i++) {
    if (/^\s*(```|~~~)/.test(lines[i])) {
      inFence = !inFence;
      continue;
    }
    if (inFence) continue;

    lines[i] = processLine(lines[i], glossary, pattern, linked);
  }
  return lines.join("\n");
}
