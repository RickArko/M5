# M5 llm-wiki

A persistent, LLM-curated knowledge base for the M5 Forecasting project.

## What this is

A compounding artifact: every research session, every ingested doc, and
every question answered makes the next session faster. Pattern is from
[Karpathy's llm-wiki gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).

You curate raw sources. Claude does the summarizing, cross-referencing,
and bookkeeping.

## How to use

| Command | What it does |
|---|---|
| `/wiki ingest <path-or-url>` | Add a new source: copy to `raw/`, summarize, update entity / concept / topic pages, refresh `index.md`. |
| `/wiki query <question>` | Search the wiki and synthesize an answer with `[[wikilinks]]`. |
| `/wiki lint` | Surface broken links, orphans, stale claims, missing concept pages. |
| `/wiki log [N]` | Show the last N log entries. |
| `/wiki index` | Rebuild `index.md` from the file tree. |
| `/wiki init` | Bootstrap or repair the wiki tree. |

## Layout

| Path | Purpose |
|---|---|
| `AGENTS.md` | Schema — read first every session. |
| `README.md` | This file. |
| `index.md` | Auto-maintained catalog of pages. |
| `log.md` | Append-only activity log. |
| `raw/` | Immutable source materials — never edited after writing. |
| `pages/sources/` | One-page summary per raw source. |
| `pages/entities/` | Named things — libraries, datasets, people, repos. |
| `pages/concepts/` | Ideas, methods, abstractions (e.g. `wrmsse`, `tweedie-loss`). |
| `pages/topics/` | Synthesized analyses, comparisons, deep-dives. |

## Conventions

- Filenames: lowercase, hyphenated slugs.
- Cross-references: `[[wikilinks]]` relative to `pages/`.
- Frontmatter (`type`, `created`, `updated`, `tags`) mandatory on every
  page under `pages/`.
- See [`AGENTS.md`](AGENTS.md) for the full schema.
