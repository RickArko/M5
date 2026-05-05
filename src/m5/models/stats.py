"""Statistical baselines via Nixtla ``statsforecast``: Theta, AutoETS, SeasonalNaive.

These are univariate, fast on CPU, and very strong on M5 — Theta in particular
was a top-tier baseline in the original competition.

The canonical bundle lives in ``configs/m5/stats.yaml`` and is loaded through
:mod:`m5.recipes`. The functions below are thin back-compat wrappers.
"""

from __future__ import annotations

import time

import pandas as pd
from statsforecast import StatsForecast

from m5.config import SETTINGS
from m5.logging import logger
from m5.recipes import (
    STATS_RECIPE_PATH,
    Recipe,
    build_stats_from_recipe,
)

DEFAULT_FREQ = "D"
DEFAULT_SEASON = 7  # weekly seasonality on daily retail data


def _load_stats_recipe() -> Recipe:
    return Recipe.from_yaml(STATS_RECIPE_PATH)


def build_stats_forecaster(
    *,
    season_length: int = DEFAULT_SEASON,
    freq: str = DEFAULT_FREQ,
    n_jobs: int = -1,
) -> StatsForecast:
    """Theta + AutoETS + SeasonalNaive bundle from ``configs/m5/stats.yaml``."""
    recipe = _load_stats_recipe()
    assert recipe.model.kind == "stats"

    # Apply per-call overrides by patching every model spec's season_length.
    new_specs = [
        spec.model_copy(update={"season_length": season_length})
        for spec in recipe.model.models
    ]
    new_model = recipe.model.model_copy(update={"models": new_specs})
    new_task = recipe.task.model_copy(update={"freq": freq})
    recipe = recipe.model_copy(update={"task": new_task, "model": new_model})
    return build_stats_from_recipe(recipe, n_jobs=n_jobs)


def fit_predict_stats(
    df: pd.DataFrame,
    *,
    horizon: int = SETTINGS.horizon,
    season_length: int = DEFAULT_SEASON,
) -> pd.DataFrame:
    """Train statistical baselines and return a long forecast frame."""
    n_series = df["unique_id"].nunique()
    logger.info(f"fit_predict_stats: Theta+AutoETS+SeasonalNaive on {n_series:,d} series, h={horizon}")
    t0 = time.time()
    sf = build_stats_forecaster(season_length=season_length)
    sf.fit(df=df[["unique_id", "ds", "y"]])
    logger.info(f"fit_predict_stats: fit done in {time.time() - t0:.1f}s — predicting")
    out = sf.predict(h=horizon)
    logger.info(f"fit_predict_stats: total {time.time() - t0:.1f}s, {len(out):,d} forecast rows")
    return out
