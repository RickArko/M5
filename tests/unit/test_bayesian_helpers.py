"""Unit tests for Bayesian model helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from m5.models.bayesian import (
    classify_intermittency,
    intermittency_profiles,
    pick_intermittency_examples,
    route_demand_class,
    series_zero_rate,
)


@pytest.mark.unit
def test_series_zero_rate(toy_with_calendar: pd.DataFrame) -> None:
    rates = series_zero_rate(toy_with_calendar)
    assert (rates >= 0).all() and (rates <= 1).all()
    assert len(rates) == toy_with_calendar["unique_id"].nunique()


@pytest.mark.unit
def test_pick_intermittency_examples(toy_with_calendar: pd.DataFrame) -> None:
    high, low = pick_intermittency_examples(toy_with_calendar, min_train_days=50)
    rates = series_zero_rate(toy_with_calendar)
    assert rates[high] >= rates[low]


@pytest.mark.unit
def test_classify_intermittency_quadrants() -> None:
    assert classify_intermittency(1.0, 0.1) == "smooth"
    assert classify_intermittency(2.0, 0.1) == "intermittent"
    assert classify_intermittency(1.0, 1.0) == "erratic"
    assert classify_intermittency(2.0, 1.0) == "lumpy"


@pytest.mark.unit
def test_route_demand_class() -> None:
    assert route_demand_class("smooth") == ("negbin", False)
    assert route_demand_class("intermittent") == ("zinb", True)
    assert route_demand_class("lumpy") == ("zinb", True)


@pytest.mark.unit
def test_intermittency_profiles_synthetic() -> None:
    dates = pd.date_range("2020-01-01", periods=120, freq="D")
    y_smooth = np.clip(3 + np.sin(np.arange(120) * 2 * np.pi / 7), 0, None)
    y_intermittent = np.where(np.arange(120) % 5 == 0, 4, 0).astype(float)
    rows = []
    for uid, y in (("smooth", y_smooth), ("intermittent", y_intermittent)):
        for d, v in zip(dates, y, strict=True):
            rows.append({"unique_id": uid, "ds": d, "y": float(v), "dept_id": "FOODS_1"})
    df = pd.DataFrame(rows)
    prof = intermittency_profiles(df)
    assert prof.loc["smooth", "demand_class"] == "smooth"
    assert prof.loc["intermittent", "demand_class"] == "intermittent"
