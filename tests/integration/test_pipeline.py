"""Integration: end-to-end glue from raw long frame → features → WRMSSE."""

from __future__ import annotations

import importlib.util

import pandas as pd
import pytest

from m5.evaluation import compute_components, wrmsse_for_models
from m5.features import build_feature_frame


def _have(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


def test_features_pipeline_preserves_required_schema(toy_with_calendar: pd.DataFrame) -> None:
    out = build_feature_frame(toy_with_calendar.copy())
    required = {"unique_id", "ds", "y", "dayofweek", "snap", "is_event", "price_norm"}
    assert required.issubset(out.columns)
    assert out["unique_id"].nunique() == toy_with_calendar["unique_id"].nunique()


@pytest.mark.slow
@pytest.mark.skipif(not _have("statsforecast"), reason="statsforecast not installed")
def test_pipeline_stats_cv_then_score(toy_long: pd.DataFrame) -> None:
    """Full loop: fit stats CV → compute WRMSSE components → score every model column."""
    from m5.cv import stats_cv

    cv_df = stats_cv(toy_long, h=14, n_windows=1, season_length=7)
    history = toy_long[toy_long["ds"] < cv_df["ds"].min()]
    components = compute_components(history)

    truth = cv_df[["unique_id", "ds", "y"]]
    scores = wrmsse_for_models(truth, cv_df, components)
    assert len(scores) == 3  # Theta, AutoETS, SeasonalNaive
    assert (scores >= 0).all()
