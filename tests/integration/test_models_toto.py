"""Integration tests for the TOTO zero-shot foundation model wrapper.

These tests verify that the TOTO model can be built and produces valid
forecast outputs.  They require ``toto-models`` and a working PyTorch
installation (CPU-only is fine for CI).

Marked ``integration`` and ``slow`` because model download + inference
is heavier than a typical unit test.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from m5.evaluation import compute_components, wrmsse_for_models


def _have(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


pytestmark = [
    pytest.mark.skipif(not _have("toto2"), reason="toto-models not installed (pip install toto-models)"),
    pytest.mark.skipif(not _have("torch"), reason="torch not installed"),
    pytest.mark.integration,
    pytest.mark.slow,
]


class TestTotoModelBuild:
    """Verify the model can be loaded from HuggingFace."""

    def test_build_toto_model(self) -> None:
        from m5.models.toto import build_toto_model

        model = build_toto_model()
        assert model is not None
        # Should be in eval mode
        assert not model.training


class TestTotoForecast:
    """Verify forecast produces valid Nixtla-shaped output."""

    def test_forecast_returns_expected_columns(self, toy_long: pd.DataFrame) -> None:
        from m5.models.toto import toto_forecast

        fcast = toto_forecast(toy_long, horizon=7, context_length=100, batch_size=3)
        assert isinstance(fcast, pd.DataFrame)
        assert "unique_id" in fcast.columns
        assert "ds" in fcast.columns
        assert "TOTO" in fcast.columns
        n_series = toy_long["unique_id"].nunique()
        assert len(fcast) == n_series * 7

    def test_forecast_values_are_finite(self, toy_long: pd.DataFrame) -> None:
        from m5.models.toto import toto_forecast

        fcast = toto_forecast(toy_long, horizon=7, context_length=100, batch_size=3)
        vals = fcast["TOTO"].values
        assert not np.any(np.isnan(vals)), "NaN values in TOTO forecast"
        assert not np.any(np.isinf(vals)), "Inf values in TOTO forecast"

    def test_forecast_reproducible(self, toy_long: pd.DataFrame) -> None:
        from m5.models.toto import toto_forecast

        f1 = toto_forecast(toy_long, horizon=7, context_length=100, batch_size=3)
        f2 = toto_forecast(toy_long, horizon=7, context_length=100, batch_size=3)
        pd.testing.assert_frame_equal(f1, f2)


class TestTotoCV:
    """Verify CV produces a Nixtla-format CV frame."""

    def test_cv_returns_expected_columns(self, toy_long: pd.DataFrame) -> None:
        from m5.models.toto import toto_cv

        cv_df = toto_cv(toy_long, h=7, n_windows=2, step_size=7, context_length=100, batch_size=3)
        assert "unique_id" in cv_df.columns
        assert "ds" in cv_df.columns
        assert "cutoff" in cv_df.columns
        assert "y" in cv_df.columns
        assert "TOTO" in cv_df.columns
        n_series = toy_long["unique_id"].nunique()
        assert len(cv_df) == n_series * 7 * 2  # 2 windows x 7 days x N series

    def test_cv_scorable(self, toy_long: pd.DataFrame) -> None:
        from m5.models.toto import toto_cv

        cv_df = toto_cv(toy_long, h=7, n_windows=2, step_size=7, context_length=100, batch_size=3)
        train_pre_cv = toy_long[toy_long["ds"] < cv_df["ds"].min()]
        components = compute_components(train_pre_cv)
        truth = cv_df[["unique_id", "ds", "y"]]
        scores = wrmsse_for_models(truth, cv_df, components)
        assert isinstance(scores, pd.Series)
        assert "TOTO" in scores.index
        assert scores["TOTO"] > 0  # not a perfect forecast
