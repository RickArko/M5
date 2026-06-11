# AGENTS.md — M5 project guide for AI coding agents

This file follows the [agents.md](https://agents.md) spec. Most modern
agentic CLIs (OpenAI Codex, opencode, factory.ai droids, …) auto-discover
it. Aider needs `--read AGENTS.md`; Gemini CLI uses
[`GEMINI.md`](GEMINI.md) which points back here. Claude Code's
auto-loaded equivalent is [`CLAUDE.md`](CLAUDE.md) — overlapping content,
read either.

The full agent contributor guide is
[`docs/developer/AGENTS.md`](docs/developer/AGENTS.md). The
token-optimized full repo context is
[`AI-CONTEXT.md`](AI-CONTEXT.md) — drop it into a system prompt and the
agent has the whole map.

## Project

Reproducible Kaggle **M5 Forecasting – Accuracy** solution: 28-day daily
forecast for 30,490 Walmart item × store series. Stack: Nixtla
(`statsforecast`, `mlforecast`, `utilsforecast`, `hierarchicalforecast`,
`datasetsforecast`) + LightGBM, glued by a small `m5` Python package and
a Typer CLI. Python **3.12** only, deps managed by **`uv`**, entrypoint
is the `Makefile` (Linux / macOS / WSL — no PowerShell path).

## Setup

```bash
make bootstrap     # install uv, sync deps, seed .env, download data (idempotent)
make install       # uv sync --all-groups + register Jupyter kernel + pre-commit
```

If `command not found: uv` after bootstrap, open a new terminal or
`source ~/.local/bin/env`.

## Build / test / lint commands

```bash
make help          # print every target with its docstring
make check         # lint + typecheck + test (CI entry point)
make lint          # ruff check
make fmt           # ruff format + autofix
make typecheck     # mypy on src/m5
make test          # full pytest suite (smoke + unit + integration)
make test-fast     # smoke + unit only (~5 s) — best inner-loop target
make test-smoke    # ~1 s — imports, CLI help, package metadata
```

CLI surface (every Make target above pipeline-* is a thin wrapper):

```bash
uv run m5 --help
uv run m5 download
uv run m5 prep --last-n-days 400 --n-series -1
uv run m5 cv lgbm  --horizon 28 --n-windows 3
uv run m5 forecast lgbm --horizon 28
uv run m5 cv toto  --horizon 28 --n-windows 3   # needs --group toto
uv run m5 forecast toto                           # needs --group toto
uv run m5 train
uv run m5 serve     # FastAPI on http://localhost:8000
```

## TOTO (zero-shot foundation model)

TOTO is DataDog's time series foundation model — no training required, forecasts
directly from the last 512 days of context.  Requires the ``toto`` dep group:

```bash
uv run --group toto m5 cv toto --horizon 28 --n-windows 1
uv run --group toto m5 forecast toto --horizon 28
```

TOTO commands are **not** capped like stats/lgbm — the model processes every
series in batches, so subsample with ``M5_N_SERIES`` if RAM/VRAM is tight.
Always run with ``--group toto`` (managed separately because ``toto-models``
pulls PyTorch).

## Pipeline commands (capped for dev hardware)

```bash
# Cheap: run end-to-end on a subsample (~1–2 min)
M5_N_SERIES=500 M5_LAST_N_DAYS=200 M5_N_WINDOWS=1 make prep cv-lgbm

# Expensive: full data — only on remote / cloud nodes
make prep cv-lgbm
```

The dev box this repo lives on is RAM-constrained — never run `prep` /
`cv-*` against the full 30,490 series locally. Honor the env caps.

## Architecture (module map)

```
src/m5/
  config.py      → SETTINGS (frozen dataclass, env-driven). Single source of truth.
  data.py        → load + melt → Nixtla long frame.
  features.py    → date / snap / event-flag / price-norm features.
  evaluation.py  → WRMSSE: compute_components + wrmsse / wrmsse_for_models.
  hierarchy.py   → 12-level M5 spec around hierarchicalforecast.
  cv.py          → stats_cv / lgbm_cv / hier_cv. Always seeds first.
  models/stats.py        → Theta + AutoETS + SeasonalNaive @ season_length=7.
  models/lgbm.py         → MLForecast + LightGBM (Tweedie, deterministic).
  models/toto.py         → TOTO 2.0 zero-shot foundation model (DataDog).
  models/hierarchical.py → Theta + BU / TD / MinT reconcilers.
  cli.py         → typer app: download | prep | cv | forecast | train | serve.
  serve/         → FastAPI service (see README.md "Serving the model").
```

**Data flow.** `m5 download` → `data/m5/datasets/*.csv` → `m5 prep` →
`data/processed/long.parquet` → `m5 cv <model>` →
`artifacts/cv_<model>.parquet` → `m5 forecast <model>` →
`forecasts/forecast_<model>.parquet`.

## Conventions to preserve

- **Schema is Nixtla long-frame.** `unique_id` (string `"{item_id}_{store_id}"`),
  `ds` (datetime64), `y` (float32). Statics: `item_id, dept_id, cat_id,
  store_id, state_id`. Time-varying: `sell_price`, `snap_{CA,TX,WI}` +
  collapsed `snap`, `event_*` + collapsed `is_event`, `price_norm`,
  `price_change_pct`. Don't reintroduce `id` or `_evaluation` — those
  are stripped legacy columns.
- **Reproducibility is a contract.** Every `cv` / `forecast` / `train`
  runner calls `set_global_seed()` before fitting. LightGBM uses
  `deterministic=True, force_row_wise=True, seed=SETTINGS.seed`. Same
  inputs → same WRMSSE. If two runs disagree, that's a bug.
- **Tweedie for LGBM** (sparse count-like retail data). Don't switch
  objectives without a CV diff.
- **Drop leading zeros once** (`_drop_leading_zeros` in `data.py`); not
  downstream.
- **Trailing window default = 400 days** (`M5_LAST_N_DAYS`). Full
  ~1941-day history is supported but slow.
- **Feature menu is deliberately small.** Prefer extending `mlforecast`
  lag / rolling configs in `models/lgbm.py` over adding columns in
  `features.py`. Adding a feature requires a CV diff in the PR body.
- **Configuration via env / `.env` only** (see `.env.example`). Don't
  hardcode paths or seeds.
- **`data/processed/long.parquet` is the only handoff** between prep
  and modelling.

## Hard rules for AI agents

1. Don't break determinism — every model entry calls `set_global_seed()`.
2. Don't run full `prep` / `cv-*` on dev hardware — it OOMs. Cap with
   `M5_N_SERIES` / `M5_LAST_N_DAYS` / `M5_N_WINDOWS`.
3. Don't add columns to the feature menu without a CV diff.
4. Don't reintroduce `id` or `_evaluation` columns.
5. Don't hand-edit `AI-CONTEXT.md` — regenerate via `/ai-condense`.
6. Don't hand-edit `uv.lock` — use `uv add` / `uv sync --upgrade-package`.
7. Don't bypass pre-commit hooks (`--no-verify`). If a hook fails, fix
   and create a new commit.
8. Don't push or create PRs without explicit user confirmation, even if
   tests pass.

## Code style

- **Python 3.12.** Modern syntax (PEP 604 unions, `from __future__ import
  annotations` already in source where needed).
- **Ruff** owns lint + format (`line-length = 110`, `target-version =
  "py312"`). Selected rules: `E F W I B UP SIM PD RUF`.
- **mypy** strict on `src/m5` only (`strict_optional`, `warn_unused_ignores`,
  `warn_redundant_casts`).
- **No emojis** in code or comments unless the user asks.
- **Comments only for non-obvious why** — avoid restating what the code
  does. Identifiers should carry the "what".
- **Don't add features, refactor, or abstract beyond the task.**

## Test guidance

- Tiers: `smoke` (~1 s), `unit` (~5 s), `integration` (~30 s),
  `slow` (skip in tight loops).
- New tests depend on `tests/conftest.py::toy_long` (3 series × 200
  days). Don't synthesize data ad-hoc.
- Inner loop: `make test-fast` after every edit.
- Pre-PR: `make check`.
- For model / feature / evaluation changes, capped CV diff in the PR
  body is required.

## Where to read more

| File | Purpose |
|---|---|
| [`AI-CONTEXT.md`](AI-CONTEXT.md) | Token-optimized full repo context. Drop into a system prompt. |
| [`docs/developer/AGENTS.md`](docs/developer/AGENTS.md) | Extensive contributor guide for agents (per-harness setup, workflow, examples). |
| [`README.md`](README.md) | Happy path, CLI surface, serving. |
| [`docs/developer/ARCHITECTURE.md`](docs/developer/ARCHITECTURE.md) | Module map + data flow + design tenets. |
| [`docs/developer/DEVELOPMENT.md`](docs/developer/DEVELOPMENT.md) | Daily workflow, debugging, notebooks. |
| [`docs/developer/SETUP.md`](docs/developer/SETUP.md) | First-time install (WSL, uv, VSCode). |
| [`docs/developer/TROUBLESHOOTING.md`](docs/developer/TROUBLESHOOTING.md) | Error matrix. |
| [`CLAUDE.md`](CLAUDE.md) | Claude Code-specific notes (overlaps this file). |
| [`WriteUp.md`](WriteUp.md) | Canonical methodology + EDA highlights. |
