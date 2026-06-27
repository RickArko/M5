# GEMINI.md — M5 project context for Gemini CLI

The canonical agent guide for this project lives in
[`AGENTS.md`](AGENTS.md) (the cross-tool [agents.md](https://agents.md)
spec file). The extensive contributor guide for AI agents is
[`docs/developer/AGENTS.md`](docs/developer/AGENTS.md). The token-optimized
full repo context is [`AI-CONTEXT.md`](AI-CONTEXT.md).

This file exists so Gemini CLI auto-loads project context on session
start. Everything below is a thin extract of `AGENTS.md` — read that
file if you need the full picture.

## Quick context

- **What:** Reproducible Kaggle M5 Forecasting – Accuracy solution
  (28-day daily forecast, 30,490 series).
- **Stack:** Python 3.12, `uv`, Nixtla (`statsforecast`, `mlforecast`,
  `hierarchicalforecast`) + LightGBM, FastAPI for serving.
- **Entrypoint:** `Makefile` (Linux / macOS / WSL only).
- **CLI:** `uv run m5 <download|prep|cv|forecast|train|serve>`.

## Useful at-mentions for Gemini sessions

Gemini CLI lets you inline files at the prompt with `@`. Recommended
context loads:

```
> @AI-CONTEXT.md  what does m5.cv.lgbm_cv guarantee?
> @AGENTS.md @src/m5/models/lgbm.py  walk me through the lag transforms.
> @docs/developer/AGENTS.md  what's the PR convention for new features?
```

To preload everything once for a session:

```
> @AGENTS.md @AI-CONTEXT.md @CLAUDE.md
```

## Commands worth knowing

```bash
make help          # every Make target with its docstring
make check         # lint + typecheck + test (CI entry)
make test-fast     # ~5 s — best inner-loop target
uv run m5 --help   # full CLI surface
```

## Hard rules (summary)

Full list in [`AGENTS.md`](AGENTS.md#hard-rules-for-ai-agents). The
short version:

1. **Determinism is a contract** — never add randomness without seeding.
2. **RAM-constrained dev box** — never run `make prep` / `cv-*` against
   the full 30,490 series. Cap with
   `M5_N_SERIES=200 M5_LAST_N_DAYS=120 M5_N_WINDOWS=1`.
3. **Don't extend the feature menu without a CV diff.**
4. **Don't hand-edit** `AI-CONTEXT.md` (regenerated) or `uv.lock`
   (managed by `uv`).
5. **Don't push or open PRs without user confirmation.**

## Where to read more

| File | Purpose |
|---|---|
| [`AGENTS.md`](AGENTS.md) | Cross-tool agent contract (canonical). |
| [`AI-CONTEXT.md`](AI-CONTEXT.md) | Token-optimized full repo context. |
| [`docs/developer/AGENTS.md`](docs/developer/AGENTS.md) | Extensive agent contributor guide (per-harness setup, workflow, examples). |
| [`README.md`](README.md) | Happy path. |
| [`CLAUDE.md`](CLAUDE.md) | Claude Code's auto-loaded equivalent. |
