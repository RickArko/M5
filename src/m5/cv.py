"""Reproducible cross-validation using Nixtla's rolling-origin CV.

Both ``StatsForecast`` and ``MLForecast`` expose ``.cross_validation(...)``
with the same semantics: walk forward in steps of size ``step_size`` for
``n_windows`` windows of length ``h``. We always seed first.
"""

from __future__ import annotations

import pandas as pd

from m5.config import SETTINGS, set_global_seed
from m5.logging import logger
from m5.models.lgbm import build_lgbm_forecaster
from m5.models.stats import build_stats_forecaster


def stats_cv(
    df: pd.DataFrame,
    *,
    h: int = SETTINGS.horizon,
    n_windows: int = SETTINGS.n_windows,
    step_size: int | None = None,
    season_length: int = 7,
) -> pd.DataFrame:
    """Run rolling-origin CV with the statistical model bundle."""
    set_global_seed()
    sf = build_stats_forecaster(season_length=season_length)
    logger.info(f"stats_cv: h={h} n_windows={n_windows} step={step_size or h}")
    return sf.cross_validation(
        df=df[["unique_id", "ds", "y"]],
        h=h,
        n_windows=n_windows,
        step_size=step_size or h,
    )


def lgbm_cv(
    df: pd.DataFrame,
    *,
    h: int = SETTINGS.horizon,
    n_windows: int = SETTINGS.n_windows,
    step_size: int | None = None,
    static_cols: tuple[str, ...] = ("item_id", "dept_id", "cat_id", "store_id", "state_id"),
) -> pd.DataFrame:
    """Run rolling-origin CV with the LightGBM global model."""
    set_global_seed()
    fcst = build_lgbm_forecaster()
    static_features = [c for c in static_cols if c in df.columns]
    logger.info(f"lgbm_cv: h={h} n_windows={n_windows} step={step_size or h}")
    keep = ["unique_id", "ds", "y"] + [c for c in ("snap", "is_event", "price_norm", "price_change_pct") if c in df.columns]
    return fcst.cross_validation(
        df=df[keep],
        h=h,
        n_windows=n_windows,
        step_size=step_size or h,
        static_features=static_features,
    )
