"""LightGBM global model via Nixtla ``mlforecast``.

We keep the feature menu deliberately small: lags 7/14/28 + 7-day rolling mean,
date features, snap flag, single event flag, normalised price. No mega-blender.
"""

from __future__ import annotations

import lightgbm as lgb
import pandas as pd
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean
from mlforecast.target_transforms import Differences

from m5.config import SETTINGS
from m5.features import build_feature_frame, static_features

DEFAULT_LAGS: tuple[int, ...] = (7, 14, 28)
DEFAULT_ROLLS: tuple[int, ...] = (7, 28)


def lgbm_params(seed: int = SETTINGS.seed) -> dict[str, object]:
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
    lag_transforms = {lag: [RollingMean(window_size=w) for w in rolling_windows] for lag in (1,)}
    model = lgb.LGBMRegressor(**lgbm_params(seed=seed), n_jobs=n_jobs)
    return MLForecast(
        models={"LGBM": model},
        freq=freq,
        lags=list(lags),
        lag_transforms=lag_transforms,
        date_features=["dayofweek", "day", "week", "month", "year"],
        target_transforms=[Differences([1])],
        num_threads=n_jobs,
    )


def fit_predict_lgbm(
    df: pd.DataFrame,
    *,
    horizon: int = SETTINGS.horizon,
    static_cols: tuple[str, ...] = ("item_id", "dept_id", "cat_id", "store_id", "state_id"),
) -> pd.DataFrame:
    """End-to-end fit+predict; ``df`` must have ``unique_id, ds, y`` + features."""
    df = build_feature_frame(df.copy())
    statics = static_features(df)

    keep_cols = ["unique_id", "ds", "y"]
    keep_cols += [c for c in ("snap", "is_event", "price_norm", "price_change_pct") if c in df.columns]
    fcst = build_lgbm_forecaster()
    fcst.fit(
        df[keep_cols],
        static_features=[c for c in static_cols if c in statics.columns],
    )
    return fcst.predict(h=horizon)
