---
name: wiki-schema
description: Operational schema for the .claude/wiki/ tree — read at the start of every wiki session.
---

# AGENTS.md — Wiki schema

Source of truth for how `.claude/wiki/` is structured and maintained.
Read this at the start of every wiki session. Overrides the `/wiki`
skill description if the two disagree.

## Layout

```
.claude/wiki/
├── AGENTS.md       this file — schema
├── README.md       human entry point
├── index.md        content catalog (auto-maintained)
├── log.md          append-only activity log
├── raw/            immutable source materials — never modify
└── pages/
    ├── entities/   named things (libraries, datasets, people, repos)
    ├── concepts/   ideas, methods, abstractions
    ├── topics/     synthesized analyses, comparisons, deep-dives
    └── sources/    one-page summary per item in raw/
```

## Page conventions

- Filenames are lowercase, hyphenated slugs: `wrmsse.md`, `nixtla-stack.md`.
- One topic per page. Split rather than grow past ~300 lines.
- Cross-reference via `[[wikilinks]]` using paths relative to `pages/`,
  e.g. `[[concepts/wrmsse]]`, `[[entities/lightgbm]]`.
- Citations from raw sources use `([[sources/<slug>]])` after the claim.
- Frontmatter is mandatory on every `pages/` file.

## Frontmatter spec

```yaml
---
type: source | entity | concept | topic
created: YYYY-MM-DD
updated: YYYY-MM-DD
tags: [tag1, tag2]
---
```

Source pages add one extra field: `source: raw/<filename>`.

## Workflows

### Ingest
1. Confirm the raw file lives under `raw/`. Copy it in if needed —
   never modify a `raw/` file once written.
2. Read source. Extract claims, entities, dates, contradictions.
3. Write `pages/sources/<slug>.md`: TL;DR, Key claims, Entities
   mentioned, Open questions, Citations.
4. Update or create entity / concept / topic pages, linking with
   `[[wikilinks]]`.
5. Update `index.md`.
6. Append to `log.md`: `## [YYYY-MM-DD] ingest | <Source Title>` plus
   touched pages.
7. On contradiction: add a `## Contested` section to the affected page
   and surface it to the user. Never silently overwrite.

### Query
1. Read `index.md` first to find candidate pages.
2. `Grep` / `Glob` over `pages/` to surface anything the index missed.
3. Synthesize, cite with `[[wikilinks]]`.
4. If the answer is reusable, offer to file it as `topics/<slug>.md`.
5. Append to `log.md`.

### Lint
Report (do not fix without asking):
- broken `[[wikilinks]]`
- orphan pages (zero inbound links)
- pages with `updated:` older than the newest source on the same topic
- terms appearing in 3+ pages with no entity/concept page
- frontmatter drift
- empty directories under `pages/`

### Index rebuild
Regenerate `index.md` from the actual page tree. Preserve human-edited
prose above the first `## ` heading.

## Hard rules

- **Never modify `raw/`.** Sources are immutable. Corrections live in
  wiki pages, not in raw files.
- **Never edit `log.md` history.** Append-only. Corrections are new
  entries.
- **Frontmatter is mandatory** on every page under `pages/`.
- **Cite or link.** Claims from sources need a source citation;
  references to wiki concepts use `[[wikilinks]]`.
- **No secrets in the wiki.** Use Bitwarden references (`bw://...`)
  for anything sensitive — the wiki is committed to git.
- **Compound, don't replicate.** If a page already answers a question,
  link to it rather than re-deriving the answer.

## Related

- `README.md` — human-facing entry point
- `index.md` — content catalog
- `log.md` — chronological record
- Karpathy's pattern — https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f
