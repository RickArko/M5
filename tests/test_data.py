from __future__ import annotations

import numpy as np
import pandas as pd

from m5.data import reduce_mem_usage, split_train_horizon


def test_split_train_horizon_uses_last_h_days(toy_long: pd.DataFrame) -> None:
    train, holdout = split_train_horizon(toy_long, horizon=14)
    assert holdout["ds"].nunique() == 14
    assert train["ds"].max() < holdout["ds"].min()
    # No overlap in (id, ds) pairs
    overlap = train.merge(holdout, on=["unique_id", "ds"])
    assert overlap.empty


def test_reduce_mem_usage_lowers_memory() -> None:
    df = pd.DataFrame({
        "a": np.arange(100, dtype=np.int64),
        "b": np.linspace(0, 1, 100, dtype=np.float64),
    })
    before = df.memory_usage(deep=True).sum()
    out = reduce_mem_usage(df, verbose=False)
    after = out.memory_usage(deep=True).sum()
    assert after < before
    assert out["a"].dtype != np.int64
