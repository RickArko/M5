# GEMINI.md — M5

Guidance for the Gemini CLI when working in this repo. Mirrors the rules
in `CLAUDE.md` and `AGENTS.md` so behavior is consistent across providers.

## Project Overview

_Replace with a 2–4 sentence description._

## Key Commands

```bash
# Replace with this repo's most-used commands.
make help
```

## Default Working Mode

- Prefer implementation over speculation.
- Decompose substantial work into atomic tasks.
- Call out blockers and assumptions explicitly.

## Output Preferences

- Concise summaries first, then details.
- Include exact file paths and commands.
- Show verification status and any gaps.

## AI-enrichment

This repo is enriched with `ai-enrich`. The Gemini CLI loads context
from `.gemini/settings.json` → `contextFiles`, which by default includes
`GEMINI.md` and `.claude/wiki/AGENTS.md`. The wiki schema and conventions
apply equally to Gemini-driven work.

## Active focus briefs

Per-edit context lives at `.claude/.prompts/`. Catalog:
`.claude/.prompts/INDEX.md`. Schema: `.claude/.prompts/AGENTS.md`. If
the user names a slug or asks to resume a workstream, read the matching
prompt before doing anything else; the prompt links **out** to beads /
sessions / wiki / files rather than inlining their content.
