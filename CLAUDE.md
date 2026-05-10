# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> Shares conventions with [`AGENTS.md`](AGENTS.md) (the cross-tool agent
> contract used by Codex / opencode / Aider) and
> [`docs/developer/AGENTS.md`](docs/developer/AGENTS.md) (the extensive
> agent contributor guide — per-harness setup, workflow, examples). If
> the three ever disagree, this file wins for Claude Code; otherwise
> keep them in sync.

## Repo identity

Reproducible Kaggle **M5 Forecasting – Accuracy** solution: 28-day daily forecast for 30,490 Walmart item × store series. Stack: Nixtla (`statsforecast`, `mlforecast`, `utilsforecast`, `hierarchicalforecast`, `datasetsforecast`) + LightGBM, glued by a small `m5` Python package and a Typer CLI. Python **3.12** only, deps managed by **`uv`**, entrypoint is the `Makefile` (Linux/macOS/WSL — no PowerShell path).

## Common commands

The `Makefile` is the canonical entrypoint; targets are thin wrappers over the `m5` Typer CLI.

```bash
make bootstrap                    # one-shot: install uv, sync deps, seed .env, download data
make install                      # uv sync --all-groups + register Jupyter kernel
make lint | fmt | typecheck | test | cov | check
make download | prep              # CLI: m5 download, m5 prep
make cv-stats | cv-lgbm           # rolling-origin CV (writes artifacts/cv_<model>.parquet)
make forecast-stats | forecast-lgbm  # train on all data, write forecasts/forecast_<model>.parquet
make notebook                     # jupyter lab (notebook dep group)
make clean | clean-all
```

Override CV knobs on the CLI: `make cv-lgbm HORIZON=28 WINDOWS=3`.

Direct CLI usage:

```bash
uv run m5 prep --last-n-days 400 --n-series -1
uv run m5 cv lgbm --horizon 28 --n-windows 3
uv run pytest tests/test_evaluation.py::test_perfect_forecast_scores_zero
uv run ruff check . && uv run mypy
```

## Architecture

```
src/m5/
  config.py      → SETTINGS (frozen dataclass, env-driven). Single source of truth for paths/seed/horizon.
  data.py        → load_calendar/prices/sales + build_long_frame. Wide → Nixtla long ('unique_id','ds','y' + statics + price + snap + events).
  features.py    → date / snap / event-flag / price-norm features. Lags & rolls live in models/lgbm.py, not here.
  evaluation.py  → WRMSSE: compute_components(weights, scales) + wrmsse / wrmsse_for_models.
  hierarchy.py   → M5_LEVELS_SPEC (12 levels) + build_hierarchy / extract_bottom around hierarchicalforecast.utils.aggregate.
  cv.py          → stats_cv / lgbm_cv / hier_cv. Always seeds before .cross_validation().
  models/stats.py → Theta + AutoETS('ZNA') + SeasonalNaive @ season_length=7.
  models/lgbm.py  → MLForecast + LightGBM (Tweedie, deterministic). lags=(7,14,28), rolls=(7,28), Differences([1]).
  models/hierarchical.py → Theta base at every level + 4 reconcilers (BU / TD-fp / MinTrace OLS / MinTrace shrink).
  cli.py          → typer app: download | prep | cv | forecast.
  plots.py        → tiny matplotlib axis-formatter helpers.
```

**Data flow.** `m5 download` → `data/m5/datasets/*.csv` → `m5 prep` → `data/processed/long.parquet` → `m5 cv <model>` → `artifacts/cv_<model>.parquet` → `m5 forecast <model>` → `forecasts/forecast_<model>.parquet`. All those dirs come from `SETTINGS` and are auto-created by `SETTINGS.ensure_dirs()`.

## Conventions to preserve

- **Schema is Nixtla long-frame**: `unique_id` (string `"{item_id}_{store_id}"`), `ds` (datetime64[ns]), `y` (float32). Statics: `item_id, dept_id, cat_id, store_id, state_id`. Time-varying: `sell_price`, `snap_{CA,TX,WI}` + collapsed `snap`, `event_*` + collapsed `is_event`, `price_norm`, `price_change_pct`. Don't reintroduce `id` or `_evaluation` suffixes — those are legacy from the original dataset and have been stripped.
- **Determinism**: every `cv` / `forecast` runner calls `set_global_seed()` before fitting. LightGBM uses `deterministic=True, force_row_wise=True, seed=SETTINGS.seed`. Don't add randomness without seeding it.
- **Tweedie for LGBM** (count-like retail demand). Don't switch objectives without an experiment.
- **Drop leading zeros per series** (pre-stocking). Already handled in `_drop_leading_zeros`; don't double-do it downstream.
- **Trailing window default = 400 days** (`M5_LAST_N_DAYS`). The full 1941-day history is supported but slow.
- **Feature menu is deliberately small.** If you're tempted to add a feature, prefer extending `mlforecast` lag/rolling configs in `models/lgbm.py` over piling onto `features.py`.
- **Configuration via env vars only** (`.env` / `.env.example`). Don't hardcode paths or seeds in modules.

## Caveats

- `m5.evaluation.wrmsse` scores the **bottom level only** (item × store). The 12 official M5 levels are now wired through `m5.hierarchy` (used by `models/hierarchical.py` + `cv.hier_cv`); `hier_cv` returns bottom-level rows by default so `wrmsse_for_models` consumes it directly. To score per-level instead, call `hier_cv(..., bottom_only=False)` and group by `tags`.
- The hierarchical pipeline uses Theta as the base learner at every level. LightGBM-as-base is deferred — price/snap don't aggregate cleanly to upper levels and need their own treatment.
- `tests/conftest.py::toy_long` is the M5-shaped synthetic fixture every test reuses; new tests should depend on it instead of building their own.
- Notebook outputs are excluded from ruff (`extend-exclude = ["notebooks/*.ipynb"]`); don't rely on lint to catch issues there.
- `mypy` only walks `src/m5` (notebooks and tests are unchecked).

## When extending

- New model → add `src/m5/models/<x>.py` with `build_<x>_forecaster` + `fit_predict_<x>` mirroring `stats.py` / `lgbm.py` / `hierarchical.py`, register it in `models/__init__.py`, add a `<x>_cv` runner in `cv.py`, and wire into `cli.cv` / `cli.forecast`'s `if/elif` chain.
- New feature → add to `features.py`, include in `build_feature_frame`, and update the `keep_cols` list in both `models/lgbm.py::fit_predict_lgbm` and `cv.py::lgbm_cv` so the feature actually reaches the model.
- Reference doc for methodology and metric: [`WriteUp.md`](WriteUp.md).
