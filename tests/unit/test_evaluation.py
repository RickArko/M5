from __future__ import annotations

import numpy as np
import pandas as pd

from m5.evaluation import compute_components, wrmsse, wrmsse_for_models


def _split(df: pd.DataFrame, h: int = 14) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = df["ds"].max() - pd.Timedelta(days=h)
    return df[df["ds"] <= cutoff], df[df["ds"] > cutoff]


def test_components_have_normalised_weights(toy_long: pd.DataFrame) -> None:
    train, _ = _split(toy_long)
    comp = compute_components(train)
    assert abs(comp.weights.sum() - 1.0) < 1e-6
    assert (comp.scales > 0).all()


def test_perfect_forecast_scores_zero(toy_long: pd.DataFrame) -> None:
    train, holdout = _split(toy_long)
    comp = compute_components(train)
    forecast = holdout.rename(columns={"y": "y_hat"})[["unique_id", "ds", "y_hat"]]
    score = wrmsse(holdout, forecast, comp)
    assert score == 0.0


def test_naive_constant_zero_forecast_is_positive(toy_long: pd.DataFrame) -> None:
    train, holdout = _split(toy_long)
    comp = compute_components(train)
    forecast = holdout.assign(y_hat=0.0)[["unique_id", "ds", "y_hat"]]
    assert wrmsse(holdout, forecast, comp) > 0


def test_wrmsse_for_models_ranks_better_lower(toy_long: pd.DataFrame) -> None:
    train, holdout = _split(toy_long)
    comp = compute_components(train)
    perfect = holdout["y"].to_numpy()
    bad = perfect + np.full_like(perfect, 5.0)
    fc = holdout[["unique_id", "ds", "y"]].copy()
    fc["Perfect"] = perfect
    fc["Bad"] = bad
    scores = wrmsse_for_models(holdout, fc, comp, model_cols=["Perfect", "Bad"])
    assert scores["Perfect"] < scores["Bad"]
    assert scores["Perfect"] == 0.0
