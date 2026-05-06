# Wiki log

Append-only chronological record of wiki activity. Newest entries at the
bottom. Never edit historical entries — file corrections as new entries.

## [2026-05-05] init | bootstrap empty wiki tree

Created `.claude/wiki/` skeleton:

- `AGENTS.md` — schema
- `README.md` — human entry point
- `index.md` — empty catalog (auto-maintained)
- `log.md` — this file
- `raw/` — empty
- `pages/{entities,concepts,topics,sources}/` — empty, `.gitkeep` placeholders

No content ingested yet. Next step: `/wiki ingest <path>` to add the first
source, or hand-author a concept page (e.g. `pages/concepts/wrmsse.md`)
for a topic already understood from the codebase.
