"""Statistical baselines via Nixtla ``statsforecast``: Theta, AutoETS, SeasonalNaive.

These are univariate, fast on CPU, and very strong on M5 — Theta in particular
was a top-tier baseline in the original competition.
"""

from __future__ import annotations

import time

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, SeasonalNaive, Theta

from m5.config import SETTINGS
from m5.logging import logger

DEFAULT_FREQ = "D"
DEFAULT_SEASON = 7  # weekly seasonality on daily retail data


def build_stats_forecaster(
    *,
    season_length: int = DEFAULT_SEASON,
    freq: str = DEFAULT_FREQ,
    n_jobs: int = -1,
) -> StatsForecast:
    """Theta + AutoETS + SeasonalNaive (the seasonal naive is the canonical M5 baseline)."""
    models = [
        Theta(season_length=season_length, alias="Theta"),
        AutoETS(season_length=season_length, model="ZNA", alias="AutoETS"),
        SeasonalNaive(season_length=season_length, alias="SeasonalNaive"),
    ]
    return StatsForecast(models=models, freq=freq, n_jobs=n_jobs)


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
