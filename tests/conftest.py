"""Shared fixtures: synthetic but M5-shaped data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


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
