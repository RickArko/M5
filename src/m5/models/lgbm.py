"""LightGBM global model via Nixtla ``mlforecast``.

We keep the feature menu deliberately small: lags 7/14/28 + 7-day rolling mean,
date features, snap flag, single event flag, normalised price. No mega-blender.
"""

from __future__ import annotations

import time
from typing import Any

import lightgbm as lgb
import pandas as pd
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean

from m5.config import SETTINGS
from m5.features import build_feature_frame
from m5.logging import logger

DEFAULT_LAGS: tuple[int, ...] = (7, 14, 28)
DEFAULT_ROLLS: tuple[int, ...] = (7, 28)


def encode_static_categoricals(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """LightGBM rejects ``object`` dtype — coerce static id columns to ``category``."""
    out = df.copy()
    for c in cols:
        if c in out.columns and out[c].dtype == "object":
            out[c] = out[c].astype("category")
    return out


def lgbm_params(seed: int = SETTINGS.seed) -> dict[str, Any]:
    """Sensible LightGBM defaults for daily retail count data (Tweedie)."""
    return {
        "objective": "tweedie",
        "tweedie_variance_power": 1.1,
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 128,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "n_estimators": 1500,
        "verbosity": -1,
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
    }


def build_lgbm_forecaster(
    *,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    rolling_windows: tuple[int, ...] = DEFAULT_ROLLS,
    freq: str = "D",
    n_jobs: int = -1,
    seed: int = SETTINGS.seed,
) -> MLForecast:
    """Construct an MLForecast with LightGBM and minimal date features."""
    lag_transforms: dict[int, list[Any]] = {
        1: [RollingMean(window_size=w) for w in rolling_windows]
    }
    model = lgb.LGBMRegressor(**lgbm_params(seed=seed), n_jobs=n_jobs)
    return MLForecast(
        models={"LGBM": model},
        freq=freq,
        lags=list(lags),
        lag_transforms=lag_transforms,
        date_features=["dayofweek", "day", "week", "month", "year"],
        num_threads=n_jobs,
    )


def fit_predict_lgbm(
    df: pd.DataFrame,
    *,
    horizon: int = SETTINGS.horizon,
    static_cols: tuple[str, ...] = ("item_id", "dept_id", "cat_id", "store_id", "state_id"),
    X_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """End-to-end fit+predict.

    If ``X_df`` (future calendar + prices) is provided, dynamic exogenous
    features are used; otherwise the model is trained on lags + date features
    + static categoricals only, so ``.predict(h)`` works without a future frame.
    """
    n_rows, n_series = len(df), df["unique_id"].nunique()
    logger.info(f"fit_predict_lgbm: features → {n_series:,d} series × {n_rows // max(n_series, 1):,d} rows each")
    t0 = time.time()
    df = build_feature_frame(df.copy())
    statics_present = [c for c in static_cols if c in df.columns]
    df = encode_static_categoricals(df, statics_present)

    dynamic_cols: list[str] = []
    if X_df is not None:
        dynamic_cols = [c for c in ("snap", "is_event", "price_norm", "price_change_pct") if c in df.columns]

    keep_cols = ["unique_id", "ds", "y", *statics_present, *dynamic_cols]
    fcst = build_lgbm_forecaster()
    logger.info(f"fit_predict_lgbm: fitting LightGBM (h={horizon}, dynamic={len(dynamic_cols)} cols)")
    fcst.fit(df[keep_cols], static_features=statics_present)
    logger.info(f"fit_predict_lgbm: fit done in {time.time() - t0:.1f}s — predicting")
    out = fcst.predict(h=horizon, X_df=X_df) if X_df is not None else fcst.predict(h=horizon)
    logger.info(f"fit_predict_lgbm: total {time.time() - t0:.1f}s, {len(out):,d} forecast rows")
    return out
