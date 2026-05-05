"""Integration: rolling-origin CV runners produce the expected shape."""

from __future__ import annotations

import importlib.util

import pandas as pd
import pytest


def _have(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


@pytest.mark.slow
@pytest.mark.skipif(not _have("statsforecast"), reason="statsforecast not installed")
def test_stats_cv_yields_two_windows(toy_long: pd.DataFrame) -> None:
    from m5.cv import stats_cv

    cv_df = stats_cv(toy_long, h=7, n_windows=2, season_length=7)
    expected_cols = {"unique_id", "ds", "cutoff", "y"}
    assert expected_cols.issubset(cv_df.columns)
    assert cv_df["cutoff"].nunique() == 2


@pytest.mark.slow
@pytest.mark.skipif(not (_have("mlforecast") and _have("lightgbm")), reason="mlforecast / lightgbm missing")
def test_lgbm_cv_runs_without_static_cols(toy_with_calendar: pd.DataFrame) -> None:
    from m5.cv import lgbm_cv

    cv_df = lgbm_cv(toy_with_calendar, h=7, n_windows=2)
    assert {"unique_id", "ds", "cutoff", "y", "LGBM"}.issubset(cv_df.columns)
    assert cv_df["cutoff"].nunique() == 2


@pytest.mark.slow
@pytest.mark.skipif(not _have("statsforecast"), reason="statsforecast not installed")
def test_cv_is_deterministic_under_fixed_seed(toy_long: pd.DataFrame) -> None:
    """Two calls with the same seed must produce identical predictions."""
    from m5.cv import stats_cv

    a = stats_cv(toy_long, h=7, n_windows=1, season_length=7).sort_values(["unique_id", "ds"])
    b = stats_cv(toy_long, h=7, n_windows=1, season_length=7).sort_values(["unique_id", "ds"])
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


@pytest.mark.slow
@pytest.mark.skipif(
    not (_have("statsforecast") and _have("hierarchicalforecast")),
    reason="statsforecast / hierarchicalforecast missing",
)
def test_hier_cv_emits_reconciled_columns_at_bottom_level(toy_long: pd.DataFrame) -> None:
    from m5.cv import hier_cv

    cv_df = hier_cv(toy_long, h=7, n_windows=2, season_length=7, bottom_only=True)
    assert {"unique_id", "ds", "cutoff", "y", "Theta"}.issubset(cv_df.columns)
    assert cv_df["cutoff"].nunique() == 2
    # ids restored to project convention; 4 reconcilers materialised
    assert set(cv_df["unique_id"]) == set(toy_long["unique_id"])
    assert sum(c.startswith("Theta/") for c in cv_df.columns) == 4
