# AGENTS.md — M5

Durable rules for any AI agent working in this repo. Loaded by Claude
Code, Codex, Gemini CLI, and other agents that read AGENTS-style files.

## Default working mode

- Prefer implementation over speculation.
- Decompose substantial work into atomic tasks with clear file ownership.
- Call out blockers, risks, and assumptions explicitly.
- Run narrow relevant tests and lint where practical.
- Say clearly when verification could not be run.

## Architecture preferences

- _Replace with this repo's preferences. Examples:_
- Raw data stores remain source of truth; derived stores are rebuildable.
- Preserve human-authored markdown during regeneration.
- Prefer reusable tooling over one-off notebooks or ad hoc scripts.

## Output preferences

- Concise summaries first, then details on request.
- Include exact file paths and commands.
- Show verification status and any gaps.

## High-value paths

- _List the directories where most useful work happens, e.g. `src/`,
  `tests/`, `documentation/`, `.claude/wiki/pages/`._

## AI-enrichment

- Wiki: `.claude/wiki/` — see `.claude/wiki/AGENTS.md` for schema.
- Condensed context: `AI-CONTEXT.md` at repo root — produced by
  `/ai-condense`. The file starts with the marker
  `# ** ~ AI-Condensed Context File ~ TOKEN-CONDENSED!`. Do not edit
  by hand. Re-run `/ai-condense <path>` to refresh.
- Provider config: `.ai-enrich.toml`.
