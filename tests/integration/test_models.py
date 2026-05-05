"""Integration: build forecasters and run small fit/predict cycles on toy data."""

from __future__ import annotations

import importlib.util

import pandas as pd
import pytest


def _have(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


# ---------------------------------------------------------------------
# Statistical bundle (Theta + AutoETS + SeasonalNaive)
# ---------------------------------------------------------------------
@pytest.mark.skipif(not _have("statsforecast"), reason="statsforecast not installed")
def test_stats_forecaster_has_three_models() -> None:
    from m5.models.stats import build_stats_forecaster

    sf = build_stats_forecaster(season_length=7)
    assert len(sf.models) == 3
    aliases = {m.alias for m in sf.models}
    assert {"Theta", "AutoETS", "SeasonalNaive"}.issubset(aliases)


@pytest.mark.slow
@pytest.mark.skipif(not _have("statsforecast"), reason="statsforecast not installed")
def test_stats_fit_predict_returns_one_row_per_series_per_step(toy_long: pd.DataFrame) -> None:
    from m5.models.stats import fit_predict_stats

    out = fit_predict_stats(toy_long, horizon=7, season_length=7)
    assert {"unique_id", "ds"}.issubset(out.columns)
    assert out["unique_id"].nunique() == toy_long["unique_id"].nunique()
    assert (out.groupby("unique_id").size() == 7).all()


# ---------------------------------------------------------------------
# LightGBM via mlforecast
# ---------------------------------------------------------------------
@pytest.mark.skipif(not (_have("mlforecast") and _have("lightgbm")), reason="mlforecast / lightgbm missing")
def test_lgbm_forecaster_builds() -> None:
    from m5.models.lgbm import build_lgbm_forecaster

    fcst = build_lgbm_forecaster()
    assert fcst is not None
    assert "LGBM" in fcst.models


@pytest.mark.slow
@pytest.mark.skipif(not (_have("mlforecast") and _have("lightgbm")), reason="mlforecast / lightgbm missing")
def test_lgbm_fit_predict_on_toy(toy_with_calendar: pd.DataFrame) -> None:
    from m5.models.lgbm import fit_predict_lgbm

    out = fit_predict_lgbm(toy_with_calendar, horizon=7)
    assert {"unique_id", "ds", "LGBM"}.issubset(out.columns)
    assert (out.groupby("unique_id").size() == 7).all()


# ---------------------------------------------------------------------
# Hierarchical: Theta base + 4 reconcilers
# ---------------------------------------------------------------------
@pytest.mark.skipif(
    not (_have("statsforecast") and _have("hierarchicalforecast")),
    reason="statsforecast / hierarchicalforecast missing",
)
def test_hier_reconciler_bundle_has_four_methods() -> None:
    from m5.models.hierarchical import build_hier_reconcilers

    assert len(build_hier_reconcilers()) == 4


@pytest.mark.slow
@pytest.mark.skipif(
    not (_have("statsforecast") and _have("hierarchicalforecast")),
    reason="statsforecast / hierarchicalforecast missing",
)
def test_hier_fit_predict_returns_bottom_level_with_reconciler_columns(
    toy_long: pd.DataFrame,
) -> None:
    from m5.models.hierarchical import fit_predict_hier

    out = fit_predict_hier(toy_long, horizon=7, season_length=7, bottom_only=True)
    assert {"unique_id", "ds", "Theta", "Theta/BottomUp"}.issubset(out.columns)
    # 4 reconcilers + the base Theta column
    reconciler_cols = [c for c in out.columns if c.startswith("Theta/")]
    assert len(reconciler_cols) == 4
    # bottom level: original ids restored, one row per (series, day)
    assert set(out["unique_id"]) == set(toy_long["unique_id"])
    assert (out.groupby("unique_id").size() == 7).all()
