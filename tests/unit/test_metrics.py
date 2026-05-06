from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from m5.metrics import aggregate_series_metrics, naive_scale, per_series_metrics


def _split(df: pd.DataFrame, h: int = 14) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = df["ds"].max() - pd.Timedelta(days=h)
    return df[df["ds"] <= cutoff], df[df["ds"] > cutoff]


def test_naive_scale_positive_for_non_constant(toy_long: pd.DataFrame) -> None:
    train, _ = _split(toy_long)
    s = naive_scale(train, season_length=1)
    assert (s > 0).all()
    assert s.index.name == "unique_id"


def test_naive_scale_drops_constant_series() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["a"] * 10 + ["b"] * 10,
            "ds": list(pd.date_range("2020-01-01", periods=10)) * 2,
            "y": [3.0] * 10 + list(np.arange(10, dtype=float)),
        }
    )
    s = naive_scale(df)
    assert "a" not in s.index  # constant → dropped
    assert s.loc["b"] > 0


def test_perfect_forecast_zero_metrics(toy_long: pd.DataFrame) -> None:
    _, holdout = _split(toy_long)
    truth = holdout[["unique_id", "ds", "y"]]
    forecast = holdout.rename(columns={"y": "y_hat"})[["unique_id", "ds", "y_hat"]]
    out = per_series_metrics(truth, forecast)
    assert (out["rmse"] == 0).all()
    assert (out["mae"] == 0).all()
    assert (out["bias"] == 0).all()
    assert (out["smape"] == 0).all()


def test_biased_forecast_recovers_bias(toy_long: pd.DataFrame) -> None:
    _, holdout = _split(toy_long)
    bias_const = 1.5
    truth = holdout[["unique_id", "ds", "y"]]
    forecast = holdout.assign(y_hat=holdout["y"] + bias_const)[["unique_id", "ds", "y_hat"]]
    out = per_series_metrics(truth, forecast)
    assert np.allclose(out["bias"], bias_const, atol=1e-6)
    assert np.allclose(out["mae"], bias_const, atol=1e-6)
    assert np.allclose(out["rmse"], bias_const, atol=1e-6)


def test_mase_uses_provided_scales(toy_long: pd.DataFrame) -> None:
    train, holdout = _split(toy_long)
    scales = naive_scale(train)
    truth = holdout[["unique_id", "ds", "y"]]
    forecast = holdout.assign(y_hat=holdout["y"] + 1.0)[["unique_id", "ds", "y_hat"]]
    out = per_series_metrics(truth, forecast, scales=scales)
    assert "mase" in out.columns
    common = out.index.intersection(scales.index)
    assert np.allclose(out.loc[common, "mase"], (out.loc[common, "mae"] / scales.loc[common]).to_numpy())


def test_per_series_metrics_raises_on_empty_overlap() -> None:
    truth = pd.DataFrame({"unique_id": ["a"], "ds": [pd.Timestamp("2020-01-01")], "y": [1.0]})
    forecast = pd.DataFrame({"unique_id": ["b"], "ds": [pd.Timestamp("2020-01-02")], "y_hat": [1.0]})
    with pytest.raises(ValueError, match="No overlapping rows"):
        per_series_metrics(truth, forecast)


def test_aggregate_uses_weights() -> None:
    per_series = pd.DataFrame(
        {"rmse": [1.0, 4.0], "mae": [1.0, 4.0], "smape": [0.0, 0.0], "bias": [1.0, -1.0]},
        index=pd.Index(["a", "b"], name="unique_id"),
    )
    weights = pd.Series([0.9, 0.1], index=pd.Index(["a", "b"], name="unique_id"))
    agg = aggregate_series_metrics(per_series, weights=weights)
    assert agg["rmse"] == pytest.approx(0.9 * 1.0 + 0.1 * 4.0)
    assert agg["bias"] == pytest.approx(0.9 * 1.0 + 0.1 * -1.0)


def test_aggregate_unweighted_is_simple_mean() -> None:
    per_series = pd.DataFrame(
        {"rmse": [1.0, 3.0], "mae": [1.0, 3.0], "smape": [0.0, 0.0], "bias": [0.0, 0.0]},
        index=pd.Index(["a", "b"], name="unique_id"),
    )
    agg = aggregate_series_metrics(per_series)
    assert agg["rmse"] == pytest.approx(2.0)
