"""Shared fixtures and per-directory markers.

Layout:
    tests/smoke/        marker: smoke
    tests/unit/         marker: unit
    tests/integration/  marker: integration

Fixtures live here so any tier can use them. Don't put logic in cells.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------
# Auto-mark by directory — keeps test files free of `pytestmark = ...`.
# ---------------------------------------------------------------------
_TIER_MARKERS = {
    "smoke": pytest.mark.smoke,
    "unit": pytest.mark.unit,
    "integration": pytest.mark.integration,
}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    tests_root = Path(__file__).parent
    for item in items:
        try:
            rel = Path(item.fspath).relative_to(tests_root)
        except ValueError:
            continue
        tier = rel.parts[0] if rel.parts else ""
        marker = _TIER_MARKERS.get(tier)
        if marker is not None:
            item.add_marker(marker)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def toy_long(rng: np.random.Generator) -> pd.DataFrame:
    """A small Nixtla-shaped frame: 3 series × 200 days with weekly seasonality."""
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    rows = []
    for sid in ("FOODS_1_001_CA_1", "FOODS_1_002_CA_1", "HOUSEHOLD_1_001_TX_1"):
        seasonal = 5 + 3 * np.sin(np.arange(200) * 2 * np.pi / 7)
        trend = np.linspace(0, 4, 200)
        noise = rng.normal(0, 1, 200)
        y = np.clip(seasonal + trend + noise, 0, None).astype(np.float32)
        for d, v in zip(dates, y, strict=True):
            rows.append(
                {
                    "unique_id": sid,
                    "ds": d,
                    "y": v,
                    "item_id": "_".join(sid.split("_")[:3]),
                    "dept_id": "_".join(sid.split("_")[:2]),
                    "cat_id": sid.split("_")[0],
                    "store_id": "_".join(sid.split("_")[3:5]),
                    "state_id": sid.split("_")[3],
                    "sell_price": 1.0 + 0.1 * (sid.endswith("CA_1")),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def toy_with_calendar(toy_long: pd.DataFrame) -> pd.DataFrame:
    """``toy_long`` + the calendar columns the feature pipeline expects."""
    df = toy_long.copy()
    df["snap_CA"] = (df["ds"].dt.day <= 10).astype("int8")
    df["snap_TX"] = (df["ds"].dt.day.between(11, 20)).astype("int8")
    df["snap_WI"] = (df["ds"].dt.day > 20).astype("int8")
    df["event_name_1"] = "none"
    df["event_type_1"] = "none"
    return df


@pytest.fixture()
def small_holdout(toy_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/holdout split with a 14-day evaluation window."""
    cutoff = toy_long["ds"].max() - pd.Timedelta(days=14)
    return toy_long[toy_long["ds"] <= cutoff].copy(), toy_long[toy_long["ds"] > cutoff].copy()
