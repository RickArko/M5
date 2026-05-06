# ** ~ AI-Condensed Context File ~ TOKEN-CONDENSED! DO NOT ALTER WITHOUT `/@ai-condense`**
---

> Source: entire repo `/home/ricka/Git/GitHub/M5/` (M5 Forecasting Accuracy — Nixtla + LightGBM solution).
> Condensation rules: preserve signatures, docstrings, core control flow, key constants. Drop inline comments, `from __future__`, boilerplate `if __name__ == "__main__":`, license headers, prose duplication.

---

## 0. Project metadata (`pyproject.toml`)

```toml
[project]
name = "m5"
version = "0.1.0"
description = "Reproducible M5 Forecasting Accuracy solution — Nixtla + LightGBM + Theta/ETS"
requires-python = ">=3.12,<3.13"
authors = [{ name = "Rick Arko", email = "rick.arko17@gmail.com" }]
license = { text = "MIT" }

dependencies = [
    # Forecasting (Nixtla)
    "statsforecast>=2.0.2", "mlforecast>=1.0.2", "utilsforecast>=0.2.12",
    "datasetsforecast>=1.0.0", "hierarchicalforecast>=1.2.0",
    # ML / numerics
    "lightgbm>=4.5.0", "numpy>=2.0", "pandas>=2.2", "polars>=1.21",
    "pyarrow>=18.0", "scikit-learn>=1.5",
    # Plumbing
    "typer>=0.15", "loguru>=0.7.2", "python-dotenv>=1.0", "tqdm>=4.67",
    # Plotting
    "matplotlib>=3.9", "seaborn>=0.13",
]

[project.scripts]
m5 = "m5.cli:app"

[build-system]
requires = ["uv_build>=0.5,<0.10"]
build-backend = "uv_build"

[dependency-groups]
dev = ["pytest>=8.3", "pytest-cov>=6.0", "ruff>=0.8", "mypy>=1.13",
       "pandas-stubs>=2.2", "types-tqdm>=4.67"]
notebook = ["jupyter>=1.1", "ipykernel>=6.29,<7", "jupyterlab>=4.3", "plotly>=5.24"]

[tool.ruff]
line-length = 110
target-version = "py312"
extend-exclude = ["notebooks/*.ipynb"]
[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP", "SIM", "PD", "RUF"]
ignore = ["E501", "B008", "RUF002"]  # B008: Typer pattern; RUF002: unicode in docstrings
[tool.ruff.lint.per-file-ignores]
"tests/*" = ["B", "PD"]
"src/m5/cli.py" = ["B008"]

[tool.mypy]
python_version = "3.12"
strict_optional = true
warn_unused_ignores = true
warn_redundant_casts = true
ignore_missing_imports = true
files = ["src/m5"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short --strict-markers"

[tool.coverage.run]
source = ["src/m5"]
branch = true
```

---

## 1. Repo layout

```
M5/
├── Makefile                  # canonical entrypoint (Linux/macOS/WSL)
├── pyproject.toml            # uv-managed deps; ruff/mypy/pytest config
├── .env.example              # DATA_DIR, M5_SEED, M5_HORIZON, M5_N_WINDOWS, M5_LAST_N_DAYS, M5_N_SERIES, LOG_LEVEL
├── .python-version           # 3.12
├── scripts/
│   ├── bootstrap.sh          # one-shot setup (idempotent)
│   └── download_data.sh      # cron-friendly data refresh
├── src/m5/
│   ├── __init__.py           # __version__ via importlib.metadata
│   ├── config.py             # Settings dataclass + set_global_seed
│   ├── data.py               # load_calendar/prices/sales + build_long_frame + reduce_mem_usage
│   ├── features.py           # add_date/snap/event/price_features + build_feature_frame
│   ├── evaluation.py         # WRMSSE — compute_components / wrmsse / wrmsse_for_models
│   ├── cv.py                 # stats_cv / lgbm_cv (rolling-origin)
│   ├── cli.py                # Typer CLI: download | prep | cv | forecast
│   ├── logging.py            # loguru wrapper
│   ├── plots.py              # matplotlib helpers
│   └── models/
│       ├── stats.py          # Theta + AutoETS + SeasonalNaive
│       └── lgbm.py           # LightGBM (Tweedie) via mlforecast
├── notebooks/                # 00_run_pipeline + 01_eda…06_score
├── tests/                    # pytest unit + smoke tests
├── plots/                    # static images
└── data/                     # raw + processed (gitignored)
```

---

## 2. Tenets / design (from `docs/developer/ARCHITECTURE.md`)

1. **Get close to world-class with as few features as possible.** Three model families + small calendar/price menu + reproducible CV. Original winners had hundreds of features and deep ensembles; this is a defensible minimum.
2. **Logic in the package, demos in the notebooks.** Every notebook should be replaceable by a CLI call.
3. **One blessed shell.** Linux/macOS/WSL with `make`+`bash`. No PowerShell/cmd path.
4. **Reproducible by default.** Every script seeds before work. LightGBM in deterministic mode. `uv.lock` pins everything transitively. `data/processed/long.parquet` is the only handoff between prep and modelling. If two runs disagree, that's a bug.

### Data flow

```text
data/m5/datasets/                 (raw CSVs from datasetsforecast.M5)
  calendar.csv, sell_prices.csv, sales_train_evaluation.csv
        ▼  m5.data.{load_calendar, load_prices, load_sales}
        ▼  m5.data.build_long_frame   (melt → Nixtla schema)
data/processed/long.parquet            (unique_id, ds, y + features)
        ▼  m5.features.build_feature_frame  (date / snap / event / price)
        ┌────────────────┴────────────────┐
        ▼                                 ▼
m5.models.stats.fit_predict_stats   m5.models.lgbm.fit_predict_lgbm
        └────────────► m5.cv ◄────────────┘
                          ▼
                  artifacts/cv_<model>.parquet
                          ▼
                  m5.evaluation.wrmsse_for_models  ──► leaderboard
```

### Nixtla schema convention

| Column      | Type             | Meaning                                                |
|-------------|------------------|--------------------------------------------------------|
| `unique_id` | `category` / str | Series id = `item_id + "_" + store_id`                 |
| `ds`        | `datetime64[ns]` | Day                                                    |
| `y`         | `float32`        | Daily unit sales                                       |
| static      | `category`       | `item_id`, `dept_id`, `cat_id`, `store_id`, `state_id` |
| time-varying| numeric / cat    | `sell_price`, snap, events, …                          |

### Feature menu (deliberately small — no Fourier, no holiday distances, no Cartesian event encodings)

| Family   | Features                                                    |
|----------|-------------------------------------------------------------|
| Date     | `dayofweek`, `day`, `week`, `month`, `year`, `is_weekend`   |
| Calendar | `snap` (per-row state-correct), `is_event` (binary)         |
| Price    | `sell_price`, `price_norm` (per-series), `price_change_pct` |
| Lags     | 7, 14, 28                                                   |
| Rolls    | RollingMean(7), RollingMean(28), lagged by 1                |
| Static   | `item_id`, `dept_id`, `cat_id`, `store_id`, `state_id`      |

---

## 3. Source — `src/m5/`

> Common imports across modules (listed once): `from __future__ import annotations`, `numpy as np`, `pandas as pd`, `from pathlib import Path`. Module-specific imports shown inline.

### 3.1 `__init__.py`

```python
"""M5 Forecasting Accuracy — reproducible Nixtla + LightGBM solution."""
from importlib.metadata import PackageNotFoundError, version
try:
    __version__ = version("m5")
except PackageNotFoundError:  # editable install before metadata is built
    __version__ = "0.0.0+local"
__all__ = ["__version__"]
```

### 3.2 `config.py` — paths, seeds, env-driven Settings

```python
"""Centralised paths, seeds, and run-time settings.

Settings are read from environment variables (with `.env` support) and
exposed as a frozen dataclass so every module gets the same view.
"""
import os, random
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv(override=False)
REPO_ROOT = Path(__file__).resolve().parents[2]

def _env_int(key: str, default: int) -> int: ...      # parse env int, blank/missing → default
def _env_path(key: str, default: Path) -> Path: ...   # parse env path with ~ expansion

@dataclass(frozen=True)
class Settings:
    """Run-time configuration. Override via env vars (see `.env.example`)."""
    seed: int           = field(default_factory=lambda: _env_int("M5_SEED", 42))
    horizon: int        = field(default_factory=lambda: _env_int("M5_HORIZON", 28))
    n_windows: int      = field(default_factory=lambda: _env_int("M5_N_WINDOWS", 3))
    last_n_days: int    = field(default_factory=lambda: _env_int("M5_LAST_N_DAYS", 400))
    n_series: int       = field(default_factory=lambda: _env_int("M5_N_SERIES", -1))
    data_dir: Path      = field(default_factory=lambda: _env_path("DATA_DIR", REPO_ROOT / "data"))

    @property
    def raw_dir(self) -> Path:        return self.data_dir / "m5" / "datasets"
    @property
    def processed_dir(self) -> Path:  return self.data_dir / "processed"
    @property
    def artifacts_dir(self) -> Path:  return REPO_ROOT / "artifacts"
    @property
    def forecasts_dir(self) -> Path:  return REPO_ROOT / "forecasts"

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.processed_dir, self.artifacts_dir, self.forecasts_dir):
            p.mkdir(parents=True, exist_ok=True)

SETTINGS = Settings()

def set_global_seed(seed: int | None = None) -> int:
    """Seed Python, NumPy, and (if importable) LightGBM/PyTorch for reproducibility."""
    s = SETTINGS.seed if seed is None else seed
    random.seed(s); np.random.seed(s); os.environ["PYTHONHASHSEED"] = str(s)
    return s
```

### 3.3 `logging.py` — loguru wrapper

```python
"""Tiny loguru wrapper so every module logs the same way."""
import sys
from loguru import logger as _logger

def configure(level: str | None = None) -> None:
    _logger.remove()
    _logger.add(
        sys.stderr,
        level=level or os.getenv("LOG_LEVEL", "INFO"),
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

configure()
logger = _logger
__all__ = ["configure", "logger"]
```

### 3.4 `data.py` — Load + melt → Nixtla long frame

```python
"""Load and shape M5 raw CSVs into a Nixtla-compatible long frame.

Schema convention (Nixtla):
    unique_id : series id (item_id + "_" + store_id)
    ds        : datestamp (datetime64[ns])
    y         : target (float32)
    plus exogenous columns (calendar/price/snap features)
"""
from m5.config import SETTINGS
from m5.logging import logger

ID_COLS    = ["unique_id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
EVENT_COLS = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]

def _calendar_dtypes() -> dict[str, str | type]:
    # wm_yr_wk: uint16; event_name/type_1/2: category; snap_CA/TX/WI: uint8
    ...

def load_calendar(raw_dir: Path) -> pd.DataFrame:
    # read calendar.csv with typed dtypes; parse_dates=["date"]
    # fill NaN events with literal "none" and add "d" Categorical column ("d_1"..."d_N")
    ...

def load_prices(raw_dir: Path) -> pd.DataFrame:
    # read sell_prices.csv; store_id/item_id as category, wm_yr_wk uint16, sell_price float32
    ...

def load_sales(raw_dir: Path, prices: pd.DataFrame, n_days: int = 1941) -> pd.DataFrame:
    """Load wide sales (one column per d_*) using the *evaluation* split."""
    # dtypes: id/item_id/dept_id/cat_id/store_id/state_id all category; d_1..d_n_days → float32
    # prefer sales_train_evaluation.csv; fall back to sales_train_validation.csv
    # add unique_id = Categorical(item_id + "_" + store_id)
    ...

def reduce_mem_usage(df: pd.DataFrame, *, verbose: bool = True) -> pd.DataFrame:
    """Down-cast numeric columns to the smallest dtype that fits."""
    int_kinds   = (np.int8, np.int16, np.int32, np.int64)
    float_kinds = (np.float32, np.float64)
    start = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        kind = df[col].dtype.kind
        if kind == "i":
            cmin, cmax = df[col].min(), df[col].max()
            for t in int_kinds:
                if cmin >= np.iinfo(t).min and cmax <= np.iinfo(t).max:
                    df[col] = df[col].astype(t); break
        elif kind == "f":
            cmin, cmax = df[col].min(), df[col].max()
            for t in float_kinds:
                if cmin >= np.finfo(t).min and cmax <= np.finfo(t).max:
                    df[col] = df[col].astype(t); break
    end = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        logger.info(f"reduce_mem_usage: {start:,.1f} MB → {end:,.1f} MB ({100*(start-end)/start:.1f}% drop)")
    return df

def build_long_frame(
    sales: pd.DataFrame,
    cal: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    last_n_days: int | None = None,
    n_series: int | None = None,
) -> pd.DataFrame:
    """Melt wide sales → Nixtla long frame, attach calendar + price features.

    Returns columns: ``unique_id, ds, y`` + static and time-varying covariates.
    """
    # 1) Optional subsample of N series (using SETTINGS.seed for reproducibility)
    # 2) melt wide → long on id_vars=[unique_id, item_id, dept_id, cat_id, store_id, state_id]
    # 3) merge calendar on "d", merge prices on [store_id, item_id, wm_yr_wk]
    # 4) rename "date" → "ds"; sort by [unique_id, ds]
    # 5) _drop_leading_zeros; optional last_n_days trim; cast y → float32
    ...

def _drop_leading_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """Remove leading zero observations within each series (item not yet stocked)."""
    has_started = df.groupby("unique_id", observed=True)["y"].transform(lambda s: s.gt(0).cummax())
    return df[has_started.astype(bool)].reset_index(drop=True)

def split_train_horizon(df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/holdout split that mirrors the M5 evaluation window."""
    cutoff  = df["ds"].max() - pd.Timedelta(days=horizon)
    train   = df[df["ds"] <= cutoff].copy()
    holdout = df[df["ds"] >  cutoff].copy()
    return train, holdout
```

### 3.5 `features.py` — Minimal feature pipeline

```python
"""Minimal feature set — date features, snap, event flag, price normaliser.

Philosophy: keep the menu short. Lags/rolls are configured directly on the
MLForecast model (see :mod:`m5.models.lgbm`) so there's exactly one place
where temporal features are defined.
"""

DATE_FEATURE_COLS = ("dayofweek", "day", "week", "month", "year", "is_weekend")
SNAP_COLS         = ("snap_CA", "snap_TX", "snap_WI")
PRICE_COLS        = ("sell_price", "price_norm", "price_change_pct")

def add_date_features(df: pd.DataFrame, *, ds_col: str = "ds") -> pd.DataFrame:
    # dayofweek, day, week (isocalendar), month, year, is_weekend  — int8/int16 dtypes
    ...

def add_snap_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row snap flag for the row's state — collapses 3 columns into 1."""
    # If state_id is missing → no-op. For each state in (CA, TX, WI), look up snap_<STATE>
    # column on rows where state_id == STATE; result stored as int8 column "snap".
    ...

def add_event_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Single binary flag for ``any event today`` — drops sparse multi-hot encoding."""
    # is_event = (event_name_1 != "none").astype(int8)
    ...

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-series price normalisation and week-over-week change."""
    grp = df.groupby("unique_id", observed=True)["sell_price"]
    df["price_norm"]        = (df["sell_price"] / grp.transform("mean")).astype(np.float32)
    df["price_change_pct"]  = grp.pct_change(fill_method=None).fillna(0).astype(np.float32)
    return df

def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full minimal feature pipeline in place-friendly order."""
    return add_price_features(add_event_flag(add_snap_flag(add_date_features(df))))

def static_features(df: pd.DataFrame) -> pd.DataFrame:
    """One row per series with category-level static features (ML-Forecast format)."""
    cols = ["unique_id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    have = [c for c in cols if c in df.columns]
    return df.drop_duplicates("unique_id")[have].reset_index(drop=True)
```

### 3.6 `evaluation.py` — WRMSSE (M5 official metric)

```python
"""WRMSSE — Weighted Root Mean Squared Scaled Error (M5 official metric).

Implements the bottom-level (item × store) score directly. Hierarchical
aggregation across the 12 M5 levels can be added by precomputing series
weights at each level and reusing :func:`wrmsse_from_components`.

Reference: https://mofc.unic.ac.cy/m5-competition/
"""
from dataclasses import dataclass

@dataclass
class WRMSSEComponents:
    """Per-series weights and scales — cached so we score many models cheaply."""
    weights: pd.Series  # by unique_id, sums to 1
    scales:  pd.Series  # by unique_id, > 0

def compute_components(
    train: pd.DataFrame,
    *,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    price_col: str | None = "sell_price",
) -> WRMSSEComponents:
    """Compute series-level weights (by trailing dollar sales) and scales (Naive-1 MSE).

    Args:
        train: Long training frame ending at the day before the forecast window.
        price_col: If provided, weights are dollar-sales (units * price); else unit-sales.
    """
    df  = train.sort_values([id_col, time_col])
    rev = df[target_col] * df[price_col].fillna(0) if price_col and price_col in df.columns else df[target_col]
    last_28 = (df.assign(_rev=rev)
                 .groupby(id_col, observed=True).tail(28)
                 .groupby(id_col, observed=True)["_rev"].sum())
    weights = (last_28 / last_28.sum()).rename("weight")
    diffs   = df.groupby(id_col, observed=True)[target_col].diff()
    scales  = diffs.pow(2).groupby(df[id_col], observed=True).mean().rename("scale")
    scales  = scales.replace({0.0: np.nan}).dropna()
    common  = weights.index.intersection(scales.index)
    return WRMSSEComponents(weights=weights.loc[common], scales=scales.loc[common])

def wrmsse(
    truth: pd.DataFrame,
    forecast: pd.DataFrame,
    components: WRMSSEComponents,
    *,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    forecast_col: str = "y_hat",
) -> float:
    """Score a single forecast column against ground truth."""
    merged = truth[[id_col, time_col, target_col]].merge(
        forecast[[id_col, time_col, forecast_col]], on=[id_col, time_col], how="inner"
    )
    if merged.empty:
        raise ValueError("No overlapping rows between truth and forecast.")
    err_sq         = (merged[target_col] - merged[forecast_col]).pow(2)
    mse_per_series = err_sq.groupby(merged[id_col], observed=True).mean()
    common         = components.weights.index.intersection(mse_per_series.index)
    rmsse          = np.sqrt(mse_per_series.loc[common] / components.scales.loc[common])
    return float((components.weights.loc[common] * rmsse).sum())

def wrmsse_for_models(
    truth: pd.DataFrame,
    forecasts: pd.DataFrame,
    components: WRMSSEComponents,
    *,
    model_cols: list[str] | None = None,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
) -> pd.Series:
    """Score every model column in a wide Nixtla-style forecast frame."""
    # default model_cols = forecasts.columns - {id_col, time_col, target_col, "cutoff"}
    # returns pd.Series sorted ascending by score
    ...
```

**Mathematical form.** Bottom-level WRMSSE:

```
WRMSSE = sqrt( (1/h) Σ_t Σ_i  w_i · ( (ŷ_{i,t} - y_{i,t}) / σ_i )² )
```

- `w_i` ∝ trailing-28-day dollar sales (or unit sales if no price), normalised to sum to 1
- `σ_i²` = in-sample naive-1 differenced MSE (series with σ=0 are dropped)
- Implementation aggregates per-series MSE first, then takes `sqrt(MSE_i / scale_i) · w_i`

### 3.7 `cv.py` — Reproducible rolling-origin CV

```python
"""Reproducible cross-validation using Nixtla's rolling-origin CV.

Both ``StatsForecast`` and ``MLForecast`` expose ``.cross_validation(...)``
with the same semantics: walk forward in steps of size ``step_size`` for
``n_windows`` windows of length ``h``. We always seed first.
"""
from m5.config import SETTINGS, set_global_seed
from m5.logging import logger
from m5.models.lgbm import build_lgbm_forecaster
from m5.models.stats import build_stats_forecaster

def stats_cv(
    df: pd.DataFrame, *,
    h: int = SETTINGS.horizon,
    n_windows: int = SETTINGS.n_windows,
    step_size: int | None = None,
    season_length: int = 7,
) -> pd.DataFrame:
    """Run rolling-origin CV with the statistical model bundle."""
    set_global_seed()
    sf = build_stats_forecaster(season_length=season_length)
    return sf.cross_validation(df=df[["unique_id", "ds", "y"]], h=h, n_windows=n_windows,
                               step_size=step_size or h)

def lgbm_cv(
    df: pd.DataFrame, *,
    h: int = SETTINGS.horizon,
    n_windows: int = SETTINGS.n_windows,
    step_size: int | None = None,
    static_cols: tuple[str, ...] = ("item_id", "dept_id", "cat_id", "store_id", "state_id"),
) -> pd.DataFrame:
    """Run rolling-origin CV with the LightGBM global model."""
    set_global_seed()
    fcst            = build_lgbm_forecaster()
    static_features = [c for c in static_cols if c in df.columns]
    keep = ["unique_id", "ds", "y"] + [c for c in ("snap", "is_event", "price_norm", "price_change_pct")
                                        if c in df.columns]
    return fcst.cross_validation(df=df[keep], h=h, n_windows=n_windows,
                                  step_size=step_size or h,
                                  static_features=static_features)
```

### 3.8 `models/stats.py` — Theta + AutoETS + SeasonalNaive

```python
"""Statistical baselines via Nixtla ``statsforecast``: Theta, AutoETS, SeasonalNaive.

These are univariate, fast on CPU, and very strong on M5 — Theta in particular
was a top-tier baseline in the original competition.
"""
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, SeasonalNaive, Theta
from m5.config import SETTINGS

DEFAULT_FREQ   = "D"
DEFAULT_SEASON = 7  # weekly seasonality on daily retail data

def build_stats_forecaster(*, season_length: int = DEFAULT_SEASON, freq: str = DEFAULT_FREQ,
                           n_jobs: int = -1) -> StatsForecast:
    """Theta + AutoETS + SeasonalNaive (the seasonal naive is the canonical M5 baseline)."""
    models = [
        Theta(season_length=season_length, alias="Theta"),
        AutoETS(season_length=season_length, model="ZNA", alias="AutoETS"),
        SeasonalNaive(season_length=season_length, alias="SeasonalNaive"),
    ]
    return StatsForecast(models=models, freq=freq, n_jobs=n_jobs)

def fit_predict_stats(df: pd.DataFrame, *,
                      horizon: int = SETTINGS.horizon,
                      season_length: int = DEFAULT_SEASON) -> pd.DataFrame:
    """Train statistical baselines and return a long forecast frame."""
    sf = build_stats_forecaster(season_length=season_length)
    sf.fit(df=df[["unique_id", "ds", "y"]])
    return sf.predict(h=horizon)
```

### 3.9 `models/lgbm.py` — LightGBM global model via `mlforecast`

```python
"""LightGBM global model via Nixtla ``mlforecast``.

We keep the feature menu deliberately small: lags 7/14/28 + 7-day rolling mean,
date features, snap flag, single event flag, normalised price. No mega-blender.
"""
import lightgbm as lgb
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean
from mlforecast.target_transforms import Differences
from m5.config import SETTINGS
from m5.features import build_feature_frame, static_features

DEFAULT_LAGS:  tuple[int, ...] = (7, 14, 28)
DEFAULT_ROLLS: tuple[int, ...] = (7, 28)

def lgbm_params(seed: int = SETTINGS.seed) -> dict[str, object]:
    """Sensible LightGBM defaults for daily retail count data (Tweedie)."""
    return {
        "objective": "tweedie", "tweedie_variance_power": 1.1, "metric": "rmse",
        "learning_rate": 0.05, "num_leaves": 128, "min_data_in_leaf": 100,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
        "n_estimators": 1500, "verbosity": -1,
        "seed": seed, "deterministic": True, "force_row_wise": True,
    }

def build_lgbm_forecaster(*,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    rolling_windows: tuple[int, ...] = DEFAULT_ROLLS,
    freq: str = "D", n_jobs: int = -1, seed: int = SETTINGS.seed,
) -> MLForecast:
    """Construct an MLForecast with LightGBM and minimal date features."""
    lag_transforms = {lag: [RollingMean(window_size=w) for w in rolling_windows] for lag in (1,)}
    model = lgb.LGBMRegressor(**lgbm_params(seed=seed), n_jobs=n_jobs)
    return MLForecast(
        models={"LGBM": model}, freq=freq, lags=list(lags),
        lag_transforms=lag_transforms,
        date_features=["dayofweek", "day", "week", "month", "year"],
        target_transforms=[Differences([1])],
        num_threads=n_jobs,
    )

def fit_predict_lgbm(df: pd.DataFrame, *,
    horizon: int = SETTINGS.horizon,
    static_cols: tuple[str, ...] = ("item_id", "dept_id", "cat_id", "store_id", "state_id"),
) -> pd.DataFrame:
    """End-to-end fit+predict; ``df`` must have ``unique_id, ds, y`` + features."""
    df       = build_feature_frame(df.copy())
    statics  = static_features(df)
    keep_cols = ["unique_id", "ds", "y"] + [c for c in ("snap", "is_event", "price_norm", "price_change_pct")
                                              if c in df.columns]
    fcst = build_lgbm_forecaster()
    fcst.fit(df[keep_cols], static_features=[c for c in static_cols if c in statics.columns])
    return fcst.predict(h=horizon)
```

### 3.10 `models/__init__.py`

```python
"""Forecast models for the M5 task."""
from m5.models.lgbm  import build_lgbm_forecaster, fit_predict_lgbm
from m5.models.stats import build_stats_forecaster, fit_predict_stats
__all__ = ["build_lgbm_forecaster", "build_stats_forecaster",
           "fit_predict_lgbm", "fit_predict_stats"]
```

### 3.11 `cli.py` — Typer CLI: `m5 download | prep | cv | forecast`

```python
"""Typer CLI: ``m5 download | prep | cv | forecast | score``."""
import time
import typer
from m5.config import SETTINGS, set_global_seed
from m5.logging import logger

app = typer.Typer(add_completion=False, help="M5 forecasting toolkit.")

@app.command()
def download() -> None:
    """Download the M5 raw dataset via ``datasetsforecast``."""
    from datasetsforecast.m5 import M5
    SETTINGS.ensure_dirs()
    M5.load(directory=str(SETTINGS.data_dir))

@app.command()
def prep(
    last_n_days: int = typer.Option(SETTINGS.last_n_days, help="Trailing window of training data."),
    n_series:    int = typer.Option(SETTINGS.n_series,    help="Subsample N series (-1 = all)."),
    out:         Path = typer.Option(None, help="Output parquet path (default: data/processed/long.parquet)."),
) -> None:
    """Build the long-format training frame and write it to parquet."""
    from m5.data import build_long_frame, load_calendar, load_prices, load_sales, reduce_mem_usage
    set_global_seed(); SETTINGS.ensure_dirs()
    cal    = load_calendar(SETTINGS.raw_dir)
    prices = load_prices(SETTINGS.raw_dir)
    sales  = load_sales(SETTINGS.raw_dir, prices)
    long   = build_long_frame(sales, cal, prices,
                               last_n_days=last_n_days,
                               n_series=n_series if n_series > 0 else None)
    long   = reduce_mem_usage(long)
    out_path = out or (SETTINGS.processed_dir / "long.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    long.to_parquet(out_path, index=False)

@app.command()
def cv(
    model:     str  = typer.Argument("stats", help="One of: stats, lgbm."),
    horizon:   int  = typer.Option(SETTINGS.horizon),
    n_windows: int  = typer.Option(SETTINGS.n_windows),
    long_path: Path = typer.Option(None, help="Path to processed long parquet."),
) -> None:
    """Run reproducible rolling-origin cross-validation."""
    from m5.cv import lgbm_cv, stats_cv
    from m5.evaluation import compute_components, wrmsse_for_models
    long_path = long_path or SETTINGS.processed_dir / "long.parquet"
    df        = pd.read_parquet(long_path)
    runner    = {"stats": stats_cv, "lgbm": lgbm_cv}.get(model)
    if runner is None:
        raise typer.BadParameter(f"Unknown model: {model!r}. Use 'stats' or 'lgbm'.")
    cv_df      = runner(df, h=horizon, n_windows=n_windows)
    components = compute_components(df[df["ds"] < cv_df["ds"].min()])
    truth      = cv_df.rename(columns={"y": "y"})[["unique_id", "ds", "y"]]
    scores     = wrmsse_for_models(truth, cv_df, components)
    out = SETTINGS.artifacts_dir / f"cv_{model}.parquet"
    cv_df.to_parquet(out, index=False)

@app.command()
def forecast(
    model:     str  = typer.Argument("stats"),
    horizon:   int  = typer.Option(SETTINGS.horizon),
    long_path: Path = typer.Option(None),
) -> None:
    """Train on all available data and emit a future forecast."""
    from m5.models.lgbm  import fit_predict_lgbm
    from m5.models.stats import fit_predict_stats
    long_path = long_path or SETTINGS.processed_dir / "long.parquet"
    df        = pd.read_parquet(long_path)
    runner    = {"stats": fit_predict_stats, "lgbm": fit_predict_lgbm}.get(model)
    if runner is None:
        raise typer.BadParameter(f"Unknown model: {model!r}.")
    out_df = runner(df, horizon=horizon)
    out    = SETTINGS.forecasts_dir / f"forecast_{model}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False)

def main() -> None:  # pragma: no cover
    app()
```

### 3.12 `plots.py` — matplotlib helpers (notebook use)

```python
"""Small matplotlib helpers used by the notebooks."""
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter

def format_yaxis_thousands(ax: Axes) -> None:   # 12345 → "12,345"
def format_yaxis_percentage(ax: Axes) -> None:  # 0.42 → "42%"
```

---

## 4. Tests — `tests/`

### 4.1 `conftest.py` — shared fixture

```python
"""Shared fixtures: synthetic but M5-shaped data."""
import pytest

@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)

@pytest.fixture()
def toy_long(rng: np.random.Generator) -> pd.DataFrame:
    """A small Nixtla-shaped frame: 3 series × 200 days with weekly seasonality."""
    # 3 series: FOODS_1_001_CA_1, FOODS_1_002_CA_1, HOUSEHOLD_1_001_TX_1
    # y = clip(5 + 3*sin(2π/7 · t) + linspace(0,4) + N(0,1), 0, ∞)
    # columns: unique_id, ds, y, item_id, dept_id, cat_id, store_id, state_id, sell_price
    ...
```

### 4.2 `test_data.py`

```python
from m5.data import reduce_mem_usage, split_train_horizon

def test_split_train_horizon_uses_last_h_days(toy_long): ...
    # holdout has horizon distinct ds, no (id, ds) overlap with train

def test_reduce_mem_usage_lowers_memory(): ...
    # int64/float64 frame is downcast → memory shrinks, dtypes change
```

### 4.3 `test_features.py`

```python
from m5.features import (add_date_features, add_event_flag,
                          add_price_features, add_snap_flag, build_feature_frame)

def test_date_features_attach_expected_cols(toy_long): ...
    # all 6 DATE_FEATURE_COLS present; dayofweek ∈ [0,6]; is_weekend ∈ {0,1}

def test_snap_flag_uses_state_specific_column(toy_long): ...
    # snap_CA=1 → CA rows snap=1; TX rows (snap_TX=0) snap=0

def test_event_flag_default_zero(toy_long): ...
    # event_name_1="none" → is_event all 0

def test_price_features_norm_around_one(toy_long): ...
    # per-series mean of price_norm == 1.0

def test_build_feature_frame_idempotent_columns(toy_long): ...
    # output contains {dayofweek, snap, is_event, price_norm}
```

### 4.4 `test_evaluation.py`

```python
from m5.evaluation import compute_components, wrmsse, wrmsse_for_models

def test_components_have_normalised_weights(toy_long): ...
    # weights.sum() ≈ 1; scales > 0

def test_perfect_forecast_scores_zero(toy_long): ...
    # forecast = holdout truth → wrmsse == 0.0

def test_naive_constant_zero_forecast_is_positive(toy_long): ...
    # forecast all zero → wrmsse > 0

def test_wrmsse_for_models_ranks_better_lower(toy_long): ...
    # wrmsse(Perfect) < wrmsse(Bad); Perfect == 0.0
```

### 4.5 `test_models_smoke.py`

```python
"""Smoke tests — these import the model bundles but skip live fitting unless
optional heavy deps are installed. CI runs them; local dev too if the env is set up.
"""
import importlib.util
def _have(pkg: str) -> bool: return importlib.util.find_spec(pkg) is not None

@pytest.mark.skipif(not _have("statsforecast"), reason="statsforecast not installed")
def test_stats_forecaster_builds(): ...  # 3 models bundled

@pytest.mark.skipif(not _have("statsforecast"), reason="statsforecast not installed")
def test_stats_fit_predict_smoke(toy_long): ...  # horizon=7 → 7 rows per series

@pytest.mark.skipif(not (_have("mlforecast") and _have("lightgbm")), reason="mlforecast / lightgbm missing")
def test_lgbm_forecaster_builds(): ...  # "LGBM" key present in fcst.models
```

---

## 5. Tooling

### 5.1 `Makefile` — single entrypoint (Linux/macOS/WSL)

```makefile
# Vars: UV?=uv, HORIZON?=28, WINDOWS?=3, MODEL?=stats
# .DEFAULT_GOAL := help

help            ## Print self-documented target list
bootstrap       ## First-time setup → `bash scripts/bootstrap.sh`
install         ## `uv sync --all-groups` + register Jupyter kernel "Python (m5)"

# Quality
lint            ## ruff check .
fmt             ## ruff format . && ruff check --fix .
typecheck       ## mypy
test            ## pytest
cov             ## pytest --cov=m5 --cov-report=term-missing --cov-report=html
check           ## lint + typecheck + test (CI entry)

# Pipeline
download        ## uv run m5 download
prep            ## uv run m5 prep
cv-stats        ## uv run m5 cv stats   --horizon $(HORIZON) --n-windows $(WINDOWS)
cv-lgbm         ## uv run m5 cv lgbm    --horizon $(HORIZON) --n-windows $(WINDOWS)
forecast-stats  ## uv run m5 forecast stats --horizon $(HORIZON)
forecast-lgbm   ## uv run m5 forecast lgbm  --horizon $(HORIZON)

# Notebooks
notebook        ## uv run --group notebook jupyter lab

# Cleanup
clean           ## remove build/, dist/, *.egg-info, .pytest_cache, .coverage, htmlcov,
                ##   .ruff_cache, .mypy_cache, __pycache__, *.pyc
clean-all       ## clean + rm -rf .venv data/processed forecasts artifacts
```

### 5.2 `scripts/bootstrap.sh` (idempotent)

```bash
#!/usr/bin/env bash
# 1) Install uv if missing (curl from https://astral.sh/uv/install.sh)
# 2) uv sync --all-groups
# 3) cp .env.example .env  (if .env not present)
# 4) Register Jupyter kernel: python -m ipykernel install --user --name m5 --display-name "Python (m5)"
# 5) If data/m5/datasets is missing → uv run m5 download (~250 MB, one-time)
# Prints next-step hints (make prep / cv-stats / cv-lgbm / notebook).
```

### 5.3 `scripts/download_data.sh`

```bash
#!/usr/bin/env bash
# Thin wrapper for cron/CI: cd to repo root → `uv run m5 download`
```

### 5.4 `.env.example`

```dotenv
DATA_DIR=data
M5_SEED=42
M5_HORIZON=28          # M5 evaluation window
M5_N_WINDOWS=3         # rolling-origin CV windows
M5_LAST_N_DAYS=400     # trailing window of training data
M5_N_SERIES=-1         # subsample (-1 = all 30,490)
LOG_LEVEL=INFO         # TRACE | DEBUG | INFO | WARNING | ERROR
```

---

## 6. Methodology + EDA highlights (from `WriteUp.md`)

- **30,490 SKUs × 10 stores ⇒ 42,842 series across all aggregations.** ~1,941 daily observations per series.
- **3 data families:** demand (historic sales), prices (historic+future), date features (holidays/events).
- **Hierarchy levels (12 total in M5):** Network → State → Category → Store → Department → Item → Item×Store.
  - Network: trend + seasonality; Christmas is the lone outlier (Walmart closed).
  - State: CA > TX > WI by size; WI > CA > TX by variability.
  - Category: FOODS > HOUSEHOLD > HOBBIES; FOODS perishable → forecast accuracy matters most.
  - Store: high within-state variability (urban vs rural).
  - Bottom level (item × store) is highly intermittent.
- **Price distributions by category:** FOODS shifted log-normal; HOBBIES bi-modal log-normal; HOUSEHOLD shifted log-normal.

### Approach (5 steps)

1. **EDA** — identify calendar effects that move sales (weekly cycle, holidays, SNAP days, Christmas closure). `notebooks/01_eda.ipynb`.
2. **Naive baselines** — `SeasonalNaive(7)` is the canonical M5 baseline; included in every CV run.
3. **Statistical** — `Theta` + `AutoETS` (univariate, fast, strong on M5).
4. **LightGBM global** — single Tweedie regressor over all series via `mlforecast`; lags 7/14/28; rolling means 7/28; minimal feature menu.
5. **Reproducible CV** — rolling-origin with `h=28, n_windows=3`, seeded globally; results → `artifacts/cv_<model>.parquet`.
6. **Evaluation** — bottom-level **WRMSSE** (item × store) in `src/m5/evaluation.py`.

---

## 7. Reproducibility contract

- `m5.config.SETTINGS` is a frozen dataclass.
- `m5.config.set_global_seed` seeds `random`, NumPy, and `PYTHONHASHSEED` (default seed 42 from `M5_SEED`).
- LightGBM uses `deterministic=True`, `force_row_wise=True`, and a fixed seed.
- `uv.lock` pins every dep transitively.
- `data/processed/long.parquet` is the only handoff between prep and modelling.
- **If two runs of `make cv-lgbm` produce different WRMSSE numbers, that's a bug — file an issue with the diff.**

---

## 8. Common pitfalls (from `TROUBLESHOOTING.md`)

| Symptom | Resolution |
|---|---|
| `command not found: uv` after bootstrap | `source ~/.local/bin/env` or open a new terminal |
| VSCode warns about deprecated `ruff-lsp` | Remove `ruff.*` keys from User settings; workspace already uses `ruff.nativeServer: "on"` |
| Notebook can't see *Python (m5)* kernel | `make install` re-registers it |
| `ModuleNotFoundError: No module named 'm5'` in notebook | Wrong kernel, or pkg not installed editable → `make install` |
| `mlforecast: num_threads must be -1 or a positive integer` | Update; fix is `num_threads=n_jobs` (not `n_jobs if n_jobs > 0 else 0`) in `src/m5/models/lgbm.py` |
| Tests pass locally but fail CI | Hardcoded path → use `SETTINGS.data_dir`; wall-clock seed → use `set_global_seed()`; sort sets before comparing |
| `data/m5/datasets/` empty | `make download` (~250 MB) |
| RAM blows up during `make prep` | `M5_LAST_N_DAYS=200 M5_N_SERIES=5000 make prep` |
| LightGBM non-deterministic | Confirm `seed`/`deterministic`/`force_row_wise` not overridden in `lgbm_params`; confirm `set_global_seed()` called (every CV/forecast entry point does on entry) |
| "Add this feature" requests | Repo tenet is **fewer features, not more**. Audit signal (CV diff) before column. |

---

## 9. Notebooks (in `notebooks/`)

- `00_run_pipeline.ipynb` — minimal end-to-end driver calling `src/m5`. Use to sanity-check a fresh install.
- `01_eda.ipynb` — hierarchical EDA + calendar / price visualisations.
- `02_naive_forecast.ipynb` — `SeasonalNaive(7)` baseline.
- `03_stats_forecast.ipynb` — `Theta` + `AutoETS`.
- `04_linear_regression.ipynb` — linear baseline / sanity check.
- `05_mlforecast_lgbm.ipynb` — LightGBM global model exploration.
- `06_score.ipynb` — leaderboard rendering from `artifacts/cv_*.parquet`.

Notebook hygiene: heavy lifting goes into `src/m5`, not cells. Always select the **Python (m5)** kernel.

---

## 10. CLI reference

```bash
uv run m5 --help
uv run m5 download
uv run m5 prep --last-n-days 400 --n-series -1
uv run m5 cv stats --horizon 28 --n-windows 3
uv run m5 cv lgbm  --horizon 28 --n-windows 3
uv run m5 forecast stats --horizon 28
uv run m5 forecast lgbm  --horizon 28
```

Every CLI flag mirrors an env var from `.env.example`, so you can set once per shell:

```bash
M5_N_SERIES=500 make prep
M5_N_WINDOWS=1 make cv-lgbm
LOG_LEVEL=DEBUG make prep
```
