"""Unit tests for conformal prediction intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from m5.conformal import (
    ConformalCalibrator,
    average_interval_width,
    compute_residuals,
    coverage,
)


# ---------------------------------------------------------------------------
# ConformalCalibrator — fit + predict
# ---------------------------------------------------------------------------
class TestConformalCalibrator:
    def test_fit_pooled_computes_quantiles(self, toy_cv: pd.DataFrame) -> None:
        cal = ConformalCalibrator(alpha=0.1, method="pooled")
        cal.fit(toy_cv, model_col="Perfect")
        assert cal.horizon == 14
        assert cal.model_col == "Perfect"
        assert cal._pooled_quantiles is not None
        assert len(cal._pooled_quantiles) == 14  # one per step
        assert (cal._pooled_quantiles >= 0).all()

    def test_fit_grouped_requires_group_col(self, toy_cv: pd.DataFrame) -> None:
        cal = ConformalCalibrator(alpha=0.1, method="grouped")
        with pytest.raises(ValueError, match="group_col"):
            cal.fit(toy_cv, model_col="Perfect")

    def test_fit_grouped_with_valid_group(self, toy_cv: pd.DataFrame) -> None:
        cal = ConformalCalibrator(alpha=0.1, method="grouped", group_col="cat_id")
        cal.fit(toy_cv, model_col="Perfect")
        assert cal._group_quantiles is not None
        # Two groups: FOODS, HOUSEHOLD
        groups = cal._group_quantiles["cat_id"].unique()
        assert len(groups) == 2

    def test_predict_adds_interval_columns(self, toy_cv: pd.DataFrame) -> None:
        cal = ConformalCalibrator(alpha=0.1, method="pooled")
        cal.fit(toy_cv, model_col="Perfect")
        result = cal.predict(toy_cv)
        assert "Perfect_lo" in result.columns
        assert "Perfect_hi" in result.columns
        assert len(result) == len(toy_cv)

    def test_predict_lower_upper_order(self, toy_cv: pd.DataFrame) -> None:
        cal = ConformalCalibrator(alpha=0.1, method="pooled")
        cal.fit(toy_cv, model_col="Perfect")
        result = cal.predict(toy_cv)
        assert (result["Perfect_lo"] <= result["Perfect"]).all()
        assert (result["Perfect_hi"] >= result["Perfect"]).all()

    def test_predict_forecast_col_override(self, toy_cv: pd.DataFrame) -> None:
        cal = ConformalCalibrator(alpha=0.1, method="pooled")
        cal.fit(toy_cv, model_col="Perfect")
        result = cal.predict(toy_cv, forecast_col="Biased")
        assert "Biased_lo" in result.columns
        assert "Biased_hi" in result.columns

    def test_predict_lo_clipped_to_zero(self, toy_cv: pd.DataFrame) -> None:
        cal = ConformalCalibrator(alpha=0.5, method="pooled")
        cal.fit(toy_cv, model_col="Biased")
        result = cal.predict(toy_cv)
        lo_col = "Biased_lo"
        hi_col = "Biased_hi"
        assert (result[lo_col] >= 0).all()
        assert (result[hi_col] >= result[lo_col]).all()

    def test_perfect_forecast_coverage_at_least_nominal(self, toy_cv: pd.DataFrame) -> None:
        cal = ConformalCalibrator(alpha=0.05, method="pooled")
        cal.fit(toy_cv, model_col="Perfect")
        result = cal.predict(toy_cv)
        cov = coverage(toy_cv, result, lo_col="Perfect_lo", hi_col="Perfect_hi")
        # Perfect forecast has zero residuals → intervals should be tight but
        # coverage should still be >= 1 - alpha (all points inside ±0)
        assert cov >= 1 - 0.05 - 1e-10

    def test_scaled_method_produces_valid_intervals(self, toy_cv: pd.DataFrame) -> None:
        cal = ConformalCalibrator(alpha=0.1, method="scaled")
        cal.fit(toy_cv, model_col="Biased")
        result = cal.predict(toy_cv)
        assert "Biased_lo" in result.columns
        assert "Biased_hi" in result.columns
        assert (result["Biased_lo"] <= result["Biased"]).all()
        assert (result["Biased_hi"] >= result["Biased"]).all()

    def test_predict_raises_without_fit(self) -> None:
        cal = ConformalCalibrator(alpha=0.1)
        toy = pd.DataFrame({"unique_id": ["a"], "ds": pd.to_datetime(["2020-01-01"]), "y": [1.0], "f": [1.0]})
        with pytest.raises(ValueError, match="No forecast column"):
            cal.predict(toy)


# ---------------------------------------------------------------------------
# compute_residuals
# ---------------------------------------------------------------------------
class TestComputeResiduals:
    def test_returns_expected_columns(self, toy_cv: pd.DataFrame) -> None:
        res = compute_residuals(toy_cv, model_col="Perfect")
        for c in ("unique_id", "ds", "cutoff", "y", "Perfect", "step", "residual"):
            assert c in res.columns

    def test_perfect_model_has_zero_residuals(self, toy_cv: pd.DataFrame) -> None:
        res = compute_residuals(toy_cv, model_col="Perfect")
        assert np.abs(res["residual"]).max() < 1e-10

    def test_biased_model_has_positive_residuals(self, toy_cv: pd.DataFrame) -> None:
        res = compute_residuals(toy_cv, model_col="Biased")
        # Biased = y + 1.5 → residual should be -1.5
        assert np.abs(res["residual"].mean() + 1.5) < 0.01

    def test_step_is_positive(self, toy_cv: pd.DataFrame) -> None:
        res = compute_residuals(toy_cv, model_col="Perfect")
        assert (res["step"] >= 0).all()
        assert res["step"].nunique() == 14


# ---------------------------------------------------------------------------
# coverage / average_interval_width
# ---------------------------------------------------------------------------
class TestCoverage:
    def test_perfect_intervals_have_full_coverage(self, toy_cv: pd.DataFrame) -> None:
        cal = ConformalCalibrator(alpha=0.1, method="pooled")
        cal.fit(toy_cv, model_col="Perfect")
        pred = cal.predict(toy_cv)
        cov = coverage(toy_cv, pred, lo_col="Perfect_lo", hi_col="Perfect_hi")
        assert cov == 1.0

    def test_coverage_is_zero_when_all_outside(self, toy_cv: pd.DataFrame) -> None:
        pred = toy_cv.copy()
        pred["y_hat_lo"] = pred["y"] + 1
        pred["y_hat_hi"] = pred["y"] + 2
        cov = coverage(toy_cv, pred, lo_col="y_hat_lo", hi_col="y_hat_hi")
        assert cov == 0.0


class TestAverageIntervalWidth:
    def test_returns_positive(self) -> None:
        df = pd.DataFrame({"lo": [1.0, 2.0], "hi": [3.0, 5.0]})
        w = average_interval_width(df, lo_col="lo", hi_col="hi")
        assert w == pytest.approx(2.5)

    def test_zero_width(self) -> None:
        df = pd.DataFrame({"lo": [1.0, 2.0], "hi": [1.0, 2.0]})
        w = average_interval_width(df, lo_col="lo", hi_col="hi")
        assert w == 0.0
