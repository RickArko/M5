# Architecture

A short tour of *why* the package is shaped the way it is.

## Tenets

1. **Get close to world-class with as few features as possible.** The 2020 M5
   winners had hundreds of features and deep ensembles. We're after a
   defensible minimum: three model families, a small calendar/price feature
   menu, and reproducible CV.
2. **Logic in the package, demos in the notebooks.** Every notebook should be
   replaceable by a CLI call.
3. **One blessed shell.** Linux/macOS/WSL with `make` and `bash`. No
   parallel Windows-native code path to maintain.
4. **Reproducible by default.** Every script seeds the world before doing
   work; LightGBM runs in deterministic mode.

## Stack

| Layer        | Choice                              | Why                                                 |
|--------------|-------------------------------------|-----------------------------------------------------|
| Env / build  | `uv` (`uv_build` backend)           | Fast, lockfile-first, replaces pip+venv+pip-tools   |
| Numerics     | `numpy`, `pandas`, `polars`         | pandas for Nixtla compat, polars for I/O speed      |
| Forecasting  | Nixtla (`statsforecast`, `mlforecast`, `utilsforecast`, `datasetsforecast`, `hierarchicalforecast`) | One coherent API for stats + ML forecasting; standardised `unique_id, ds, y` schema |
| ML model     | LightGBM (Tweedie objective)        | Strong on count retail data; CPU-fast; deterministic |
| CLI          | Typer                               | Modern argparse with autocomplete + types           |
| Logs         | loguru                              | One-line config, structured output                  |
| Lint/format  | ruff (native server)                | Replaces black + isort + flake8 + most plugins      |
| Types        | mypy                                | Catches the dumb stuff in `src/m5`                  |
| Tests        | pytest + pytest-cov                 | Standard Python                                      |
| Notebooks    | Jupyter Lab via the `notebook` group | Optional install, kernel auto-registered           |

Why **not** Prophet, NeuralForecast, AutoGluon, ray? They're great, but the
goal here is "few features, defensible baseline". Adding them would muddy the
story without changing the leaderboard much for the effort budget.

## Data flow

```text
data/m5/datasets/                 (raw CSVs from datasetsforecast.M5)
        │   calendar.csv
        │   sell_prices.csv
        │   sales_train_evaluation.csv
        ▼
m5.data.{load_calendar,load_prices,load_sales}
        │
        ▼
m5.data.build_long_frame          (melt → Nixtla schema)
        │
        ▼
data/processed/long.parquet       (`unique_id, ds, y` + features)
        │
        ▼
m5.features.build_feature_frame   (date / snap / event / price)
        │
        ┌─────────────────┴────────────────┐
        ▼                                  ▼
m5.models.stats.fit_predict_stats   m5.models.lgbm.fit_predict_lgbm
        │                                  │
        └──────────────► m5.cv ◄───────────┘
                          │
                          ▼
                  artifacts/cv_<model>.parquet
                          │
                          ▼
                  m5.evaluation.wrmsse_for_models  ──► leaderboard
```

## Schema convention (Nixtla)

| Column      | Type            | Meaning                                  |
|-------------|-----------------|------------------------------------------|
| `unique_id` | `category` / str| Series id = `item_id + "_" + store_id`   |
| `ds`        | `datetime64[ns]`| Day                                      |
| `y`         | `float32`       | Daily unit sales                         |
| static      | `category` / str| `item_id`, `dept_id`, `cat_id`, `store_id`, `state_id` |
| time-varying| numeric / cat   | `sell_price`, snap, events, …            |

Both `statsforecast` and `mlforecast` consume this schema directly. We never
maintain a parallel "wide" representation outside of the raw CSV load step.

## Feature menu (deliberately small)

| Family    | Features                                                      |
|-----------|---------------------------------------------------------------|
| Date      | `dayofweek`, `day`, `week`, `month`, `year`, `is_weekend`     |
| Calendar  | `snap` (per-row state-correct), `is_event` (binary)           |
| Price     | `sell_price`, `price_norm` (per-series), `price_change_pct`   |
| Lags      | 7, 14, 28                                                     |
| Rolls     | RollingMean(7), RollingMean(28), lagged by 1                  |
| Static    | `item_id`, `dept_id`, `cat_id`, `store_id`, `state_id`        |

That's it. No fourier terms, no holiday distances, no Cartesian event
encodings. If a model is missing a signal, audit before adding — the cost of
features is in their drift, not their compute.

## Cross-validation

`statsforecast.StatsForecast.cross_validation` and
`mlforecast.MLForecast.cross_validation` share the rolling-origin semantics:
walk forward in steps of `step_size` for `n_windows` windows of length `h`.
Defaults: `h=28, n_windows=3, step_size=h`. We seed once per call.

## Evaluation — WRMSSE

`m5.evaluation` implements the bottom-level (item × store) WRMSSE directly:

- **Weights** — trailing 28-day dollar sales (or unit sales if `sell_price`
  is unavailable) per series, normalised to sum to 1.
- **Scales** — in-sample MSE of the naive-1 differenced series (drops series
  with zero scale).
- **Score** — `Σ_i w_i · sqrt(MSE_i / scale_i)` over each forecast column.

Hierarchical aggregation across the 12 M5 levels can be added by computing
weights at each level and reusing `wrmsse_from_components`. We default to
bottom-level because that's what the leaderboard uses.

## Reproducibility contract

- `m5.config.SETTINGS` is a frozen dataclass.
- `m5.config.set_global_seed` seeds Python, NumPy, and `PYTHONHASHSEED`.
- LightGBM uses `deterministic=True`, `force_row_wise=True`, and a fixed seed.
- `uv.lock` pins every dep transitively.
- `data/processed/long.parquet` is the only handoff between prep and modelling.

If you discover a place where two runs disagree, that's a bug.
