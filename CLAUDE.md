# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

M5 Forecasting Challenge solution — predicts 28-day sales forecasts for 30,490 item-store combinations using 5 years of Walmart daily sales data across 10 stores in 3 states (CA, TX, WI).

## Setup

```bash
bash install.sh
```

This runs `uv sync --all-groups` (editable install), registers a Jupyter kernel named "m5", downloads the M5 dataset, and processes it into parquet files.

## Common Commands

```bash
uv sync --all-groups          # Install/update all deps (editable)
uv run python src/process.py  # Reprocess raw data into parquet
uv run python src/score.py    # Score model submissions
uv run ruff check src/        # Lint
uv run pytest                 # Run tests
```

Run notebooks with the "m5" Jupyter kernel.

## Architecture

### Data Pipeline

`src/generate_data.py` → downloads raw CSVs via `datasetsforecast.m5.M5.load()` into `data/m5/`
`src/process.py` → loads calendar/prices/sales, merges into long-format features, outputs `data/train.snap.parquet` and `data/fit.snap.parquet` (last 400 days)

### Notebooks (run in order)

1. `notebooks/EDA.ipynb` — exploratory analysis
2. `notebooks/NaiveForecast.ipynb` — naive baseline
3. `notebooks/Forecast.ipynb` — statistical baselines (statsforecast)
4. `notebooks/MLForecast.ipynb` — LightGBM regression via mlforecast
5. `notebooks/LinearRegression.ipynb` — linear regression approach
6. `notebooks/Score.ipynb` — WRMSSE evaluation across all models

### Key Modules

- **process.py**: `create_m5_fit_data()` melts wide sales → long format, merges calendar + prices. `filter_data()` removes leading zeros. `create_future_features()` builds the 28-day forecast feature frame. `make_submission()` pivots predictions back to wide format.
- **score.py**: `score_submission()` computes WRMSSE (Weighted Root Mean Squared Scaled Error) across 12 hierarchical levels.
- **plots.py**: `@save_figure(filepath)` decorator auto-saves matplotlib/seaborn plots. Includes formatting helpers.

### Data Hierarchy (12 aggregation levels)

Network → State → Store → Category → Department → Item → Item×Store (30,490 bottom-level series)

### Evaluation Metric

WRMSSE — errors are normalized by historical MSE per series and weighted by sales contribution. Evaluated across all 12 hierarchy levels.

## Key Dependencies

- **polars** for data processing (not pandas for main pipeline)
- **statsforecast** / **mlforecast** for time-series models
- **lightgbm** as the ML backend
- **loguru** for logging
- **ruff** for linting (line-length: 120, isort profile: black)

## Conventions

- Python >=3.10,<3.11
- Line length: 120
- Parquet files use `.snap.parquet` suffix (snappy compression)
- `data/` and `DS-CaseStudy-SalesForecast/` are gitignored
