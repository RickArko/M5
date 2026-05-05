"""Smoke tests — these import the model bundles but skip live fitting unless
optional heavy deps are installed. CI runs them; local dev too if the env is set up.
"""

from __future__ import annotations

import importlib.util

import pandas as pd
import pytest


def _have(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


@pytest.mark.skipif(not _have("statsforecast"), reason="statsforecast not installed")
def test_stats_forecaster_builds() -> None:
    from m5.models.stats import build_stats_forecaster

    sf = build_stats_forecaster(season_length=7)
    assert sf is not None
    assert len(sf.models) == 3


@pytest.mark.skipif(not _have("statsforecast"), reason="statsforecast not installed")
def test_stats_fit_predict_smoke(toy_long: pd.DataFrame) -> None:
    from m5.models.stats import fit_predict_stats

    out = fit_predict_stats(toy_long, horizon=7, season_length=7)
    assert {"unique_id", "ds"}.issubset(out.columns)
    assert out["unique_id"].nunique() == toy_long["unique_id"].nunique()
    assert (out.groupby("unique_id").size() == 7).all()


@pytest.mark.skipif(not (_have("mlforecast") and _have("lightgbm")), reason="mlforecast / lightgbm missing")
def test_lgbm_forecaster_builds() -> None:
    from m5.models.lgbm import build_lgbm_forecaster

    fcst = build_lgbm_forecaster()
    assert fcst is not None
    assert "LGBM" in fcst.models
