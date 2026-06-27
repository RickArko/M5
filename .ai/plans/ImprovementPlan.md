# M5 Forecasting — High-Leverage Improvements Plan

## Overview

This document captures the two highest-impact improvements implemented to move
the baseline closer to the M5 competition winner's accuracy:

1. **Feature Expansion** — richer exogenous signals (mean encodings, calendar
distance, price stats, release date)
2. **Store-Level Segmentation** — training separate LightGBM models per store
(store, store+category, store+department)

Both follow the competition's winning methodology (YJ_STU) and the M5 paper
findings: *cross-learning with segmented models* and *feature-rich global models*.

---

## 1. Feature Expansion (Phase 2)

### What Changed

Added five new feature families in `src/m5/features.py`:

| Feature Family | Columns | Description |
|---------------|---------|-------------|
| **Mean Encodings** | `cat_id_mean_dow`, `dept_id_mean_dow`, `store_id_mean_dow`, `state_id_mean_dow`, `store_cat_mean_dow` | Historical mean sales by (group, day-of-week). Pre-computed from full history and merged back — no future leakage. |
| **Calendar Distance** | `days_to_next_event`, `days_since_last_event`, `week_of_month` | Days until / since nearest event, and week-of-month (1–5). |
| **Price Statistics** | `price_mean`, `price_min`, `price_max`, `price_rank_in_store` | Per-series historical price aggregates, and daily price percentile rank within store. |
| **Release Date** | `days_since_release` | Days since first non-zero sale per series. |

### Why These Help

- **Mean encodings** give the model a baseline expectation for each group × day-of-week combination. This is especially powerful for intermittent series where individual history is sparse.
- **Calendar distance** captures the ramp-up / ramp-down around holidays (e.g., sales spike 2–3 days before Christmas).
- **Price stats** encode the price sensitivity of each item relative to its own history and its peers.
- **Release date** separates newly introduced items from mature ones with stable demand patterns.

### How to Use

```bash
# After `make prep`, the new features are computed automatically in long.parquet
make prep

# Run CV with expanded features (same command as before)
make cv-lgbm

# Or with a capped sample for fast iteration
M5_N_SERIES=5000 M5_LAST_N_DAYS=400 M5_N_WINDOWS=1 make cv-lgbm
```

### Configuration

The `configs/m5/lgbm.yaml` now lists all new dynamic features under
`task.dynamic_cols`. The CLI's `lgbm_cv` and `segmented_cv` functions auto-detect
which of these columns exist in the dataframe and include them.

---

## 2. Store-Level Segmentation (Phase 1)

### What Changed

Added `src/m5/models/segmented.py` with three segmentation strategies:

| Segmentation | # Models | Keys | Description |
|-------------|---------|------|-------------|
| `store` | 10 | `store_id` | One model per store (CA_1, CA_2, ..., WI_3) |
| `store_cat` | 30 | `store_id`, `cat_id` | One model per store × category |
| `store_dept` | 70 | `store_id`, `dept_id` | One model per store × department |

Each segment is a completely independent `MLForecast` + LightGBM model, fitted on
only the series belonging to that segment. Predictions are concatenated back
into a single long frame.

### Why This Helps

The M5 winner (YJ_STU) used exactly this approach:

> "The winning solution considered an equal weighted combination of various
> LightGBM models that were trained to produce forecasts for the product-store
> series using data per store (10 models), store-category (30 models), and
> store-department (70 models)."

Different stores have different demand patterns:
- California stores are larger and more stable
- Texas has higher variability
- Wisconsin is smaller and more intermittent

A single global model must learn all these patterns simultaneously, diluting
store-specific signals. Segmented models learn localized patterns directly.

### How to Use

```bash
# Cross-validate 10 store-level models
make cv-segmented
# or
make cv-store

# Cross-validate 30 store-category models
make cv-store-cat

# Cross-validate 70 store-department models
make cv-store-dept

# Forecast with segmented models
make forecast-segmented
make forecast-store-cat
make forecast-store-dept
```

You can also run from the CLI directly:

```bash
uv run m5 cv segmented --horizon 28 --n-windows 3
uv run m5 cv store_cat --horizon 28 --n-windows 1
uv run m5 forecast store_dept --horizon 28
```

---

## 3. Combined Impact (The Winning Formula)

The M5 winner combined **both** segmentation and feature expansion, plus:

- **Recursive + direct forecasting** (average of both approaches)
- **Ensemble of ensembles** (average of store, store-cat, store-dept outputs)

## How to Measure Lift

The `m5 score` command already compares multiple models side-by-side. After
running any set of CVs, score them together and the report will contain WRMSSE
by model, fold, horizon, segment, and statistical significance.

### Quick Comparison (One Command)

```bash
# Runs baseline + new model CVs, scores them, and prints a comparison table
make compare
```

This will output something like:

```
| Model   | WRMSSE   | Lift vs Baseline |
|---------|----------|------------------|
| store   | 0.543210 | +12.34%          |
| lgbm    | 0.620000 | +0.00%           |
| stats   | 0.680000 | -9.68%           |
```

### Step-by-Step Comparison

1. **Run your baseline CVs**:
   ```bash
   M5_N_SERIES=5000 M5_N_WINDOWS=1 make cv-lgbm cv-stats
   ```

2. **Run your new model CVs**:
   ```bash
   M5_N_SERIES=5000 M5_N_WINDOWS=1 make cv-store cv-store-cat
   ```

3. **Score them together**:
   ```bash
   make score-all
   # or explicitly
   uv run m5 score --model lgbm --model stats --model store --model store_cat
   ```

4. **View the comparison table**:
   ```bash
   uv run python scripts/compare_scores.py reports --baseline lgbm
   ```

5. **Open the full report** (HTML with figures):
   ```bash
   open reports/report.html  # or xdg-open / firefox / etc.
   ```

### What the Report Shows

| Report Section | What It Tells You |
|----------------|-------------------|
| **Headline WRMSSE** | Overall score per model — the single number to compare |
| **Per-Fold WRMSSE** | Is the improvement consistent across all CV windows? |
| **Per-Horizon WRMSSE** | Does the model help more on day 1 or day 28? |
| **Per-Segment (store/cat/dept/state)** | Which segments benefit most? |
| **FVA (Forecast Value Add)** | How much value does each model add over SeasonalNaive? |
| **Paired Bootstrap P-Values** | Is the improvement statistically significant? |

### Recommended Next Steps

1. **Tune Tweedie variance power** per segment:
   The winner used `tweedie_variance_power=1.1` for some stores and `1.2` for
   others. This is configurable per segment via `build_lgbm_forecaster()` or
   a custom recipe YAML.

2. **Build a simple ensemble** averaging `cv_lgbm` + `cv_store` + `cv_stats`:
   ```bash
   M5_N_SERIES=5000 M5_N_WINDOWS=1 make cv-lgbm cv-store cv-stats
   make score-all
   ```

3. **Add direct multi-step forecasting**:
   Train 28 separate models (one per horizon day) and average with recursive
   predictions. This is the last major piece of the winner's approach.

---

## 4. Repository Changes Summary

| File | Change |
|------|--------|
| `src/m5/features.py` | Added `add_mean_encoding_features`, `add_calendar_features`, `add_price_stats`, `add_release_features` |
| `src/m5/models/segmented.py` | New module: segmented LightGBM fit/predict/CV |
| `src/m5/models/__init__.py` | Exported segmented functions |
| `src/m5/cv.py` | Updated `lgbm_cv` to include new dynamic features; imported `segmented_cv` |
| `src/m5/cli.py` | Added `segmented`, `store`, `store_cat`, `store_dept` to `cv` and `forecast` |
| `configs/m5/lgbm.yaml` | Added 13 new dynamic columns to `task.dynamic_cols` |
| `Makefile` | Added `cv-segmented`, `cv-store`, `cv-store-cat`, `cv-store-dept`, `forecast-segmented`, etc. |

---

## 5. References

- **M5 Winning Solution Paper**: *Simple averaging of direct and recursive forecasts via LightGBM* (Yeonjun In, 2022)
- **M5 Competition Results**: *M5 accuracy competition: Results, findings, and conclusions* (Makridakis et al., 2022)
- **Christophe Nicault's Write-up**: [M5 Forecasting Accuracy Competition](https://www.christophenicault.com/post/m5_forecasting_accuracy/)
- **Artefact Engineering Blog**: [Sales forecasting in retail: what we learned from the M5 competition](https://medium.com/artefact-engineering-and-data-science/sales-forecasting-in-retail-what-we-learned-from-the-m5-competition-445c5911e2f6)

---

> **Note**: Always run capped experiments (`M5_N_SERIES=5000 M5_N_WINDOWS=1`) for
> fast iteration. Full runs against all 30,490 series are memory-intensive and
> should only run on remote/cloud nodes.
