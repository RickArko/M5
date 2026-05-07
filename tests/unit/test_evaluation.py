from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from m5.evaluation import (
    M5_FORECAST_END_DAY,
    M5_FORECAST_START_DAY,
    M5_HORIZON,
    compute_components,
    make_submission,
    wrmsse,
    wrmsse_for_models,
)


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


def _toy_preds(h: int = 14, n_series: int = 3) -> pd.DataFrame:
    """Simple long forecast frame: ``n_series`` × ``h`` days × one ``y_hat`` column."""
    series = [f"FOODS_1_{i:03d}_CA_1" for i in range(1, n_series + 1)]
    dates = pd.date_range("2025-01-01", periods=h, freq="D")
    rows = [(s, d, float(i)) for s in series for i, d in enumerate(dates)]
    return pd.DataFrame(rows, columns=["unique_id", "ds", "y_hat"])


def test_make_submission_kaggle_layout_columns_and_shape() -> None:
    preds = _toy_preds(h=14, n_series=3)
    sub = make_submission(preds, h=14)
    assert list(sub.columns) == [f"F{i + 1}" for i in range(14)]
    assert sub.shape == (3, 14)
    assert sub.index.name == "unique_id"


def test_make_submission_d_index_layout() -> None:
    preds = _toy_preds(h=M5_HORIZON, n_series=2)
    sub = make_submission(preds, layout="d_index")
    assert sub.columns[0] == f"d_{M5_FORECAST_START_DAY}"
    assert sub.columns[-1] == f"d_{M5_FORECAST_END_DAY}"


def test_make_submission_infers_value_col_when_unique() -> None:
    preds = _toy_preds(h=7, n_series=1).rename(columns={"y_hat": "LGBM"})
    sub = make_submission(preds, h=7)
    np.testing.assert_array_equal(sub.iloc[0].to_numpy(), np.arange(7, dtype=float))


def test_make_submission_requires_explicit_value_col_when_ambiguous() -> None:
    preds = _toy_preds(h=7).assign(other=0.0)
    with pytest.raises(ValueError, match="value_col"):
        make_submission(preds, h=7)


def test_make_submission_rejects_unknown_layout() -> None:
    preds = _toy_preds(h=7)
    with pytest.raises(ValueError, match="Unknown layout"):
        make_submission(preds, h=7, layout="weird")  # type: ignore[arg-type]
