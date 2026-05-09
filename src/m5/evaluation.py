"""WRMSSE — Weighted Root Mean Squared Scaled Error (M5 official metric).

Implements the bottom-level (item × store) score directly. Hierarchical
aggregation across the 12 M5 levels can be added by precomputing series
weights at each level and reusing :func:`wrmsse_from_components`.

Reference: https://mofc.unic.ac.cy/m5-competition/
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

# M5 competition day-index constants — fixed by the competition split, not env.
# d_1 .. d_1941 = train, d_1942 .. d_1969 = held-out evaluation window.
M5_TRAIN_END_DAY: int = 1941
M5_FORECAST_START_DAY: int = 1942
M5_FORECAST_END_DAY: int = 1969
M5_HORIZON: int = M5_FORECAST_END_DAY - M5_FORECAST_START_DAY + 1  # 28


@dataclass
class WRMSSEComponents:
    """Per-series weights and scales — cached so we score many models cheaply."""

    weights: pd.Series  # by unique_id, sums to 1
    scales: pd.Series  # by unique_id, > 0


def compute_components(
    train: pd.DataFrame,
    *,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    price_col: str | None = "sell_price",
) -> WRMSSEComponents:
    """Compute series-level weights (by trailing dollar sales) and scales (Naive-1 MSE).

    Args:
        train: Long training frame ending at the day before the forecast window.
        price_col: If provided, weights are dollar-sales (units * price); else unit-sales.
    """
    df = train.sort_values([id_col, time_col])

    if price_col and price_col in df.columns:
        rev = df[target_col] * df[price_col].fillna(0)
    else:
        rev = df[target_col]

    last_28 = (
        df.assign(_rev=rev)
        .groupby(id_col, observed=True)
        .tail(28)
        .groupby(id_col, observed=True)["_rev"]
        .sum()
    )
    weights = (last_28 / last_28.sum()).rename("weight")

    diffs = df.groupby(id_col, observed=True)[target_col].diff()
    scales = diffs.pow(2).groupby(df[id_col], observed=True).mean().rename("scale")
    scales = scales.replace({0.0: np.nan}).dropna()

    common = weights.index.intersection(scales.index)
    return WRMSSEComponents(weights=weights.loc[common], scales=scales.loc[common])


def wrmsse(
    truth: pd.DataFrame,
    forecast: pd.DataFrame,
    components: WRMSSEComponents,
    *,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    forecast_col: str = "y_hat",
) -> float:
    """Score a single forecast column against ground truth."""
    merged = truth[[id_col, time_col, target_col]].merge(
        forecast[[id_col, time_col, forecast_col]],
        on=[id_col, time_col],
        how="inner",
    )
    if merged.empty:
        raise ValueError("No overlapping rows between truth and forecast.")

    err_sq = (merged[target_col] - merged[forecast_col]).pow(2)
    mse_per_series = err_sq.groupby(merged[id_col], observed=True).mean()

    common = components.weights.index.intersection(mse_per_series.index)
    if len(common) == 0:
        raise ValueError(
            "WRMSSE components share no unique_ids with the forecast — likely the "
            "training frame used for compute_components doesn't match the CV frame. "
            f"forecast has {mse_per_series.index.nunique()} series; "
            f"components have {components.weights.index.nunique()} series."
        )
    rmsse = np.sqrt(mse_per_series.loc[common] / components.scales.loc[common])
    return float((components.weights.loc[common] * rmsse).sum())


def wrmsse_for_models(
    truth: pd.DataFrame,
    forecasts: pd.DataFrame,
    components: WRMSSEComponents,
    *,
    model_cols: list[str] | None = None,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
) -> pd.Series:
    """Score every model column in a wide Nixtla-style forecast frame."""
    if model_cols is None:
        excluded = {id_col, time_col, target_col, "cutoff"}
        model_cols = [c for c in forecasts.columns if c not in excluded]

    scores = {
        m: wrmsse(truth, forecasts.rename(columns={m: "y_hat"}), components, forecast_col="y_hat")
        for m in model_cols
    }
    return pd.Series(scores, name="wrmsse").sort_values()


def make_submission(
    preds: pd.DataFrame,
    *,
    h: int = M5_HORIZON,
    id_col: str = "unique_id",
    time_col: str = "ds",
    value_col: str | None = None,
    layout: Literal["kaggle", "d_index"] = "kaggle",
) -> pd.DataFrame:
    """Pivot a long forecast frame into the wide M5 submission layout.

    Args:
        preds: Long frame with columns ``id_col``, ``time_col``, ``value_col``.
        h: forecast horizon (column count).
        value_col: forecast column. If ``None``, inferred as the only column not
            in ``(id_col, time_col)`` — set explicitly when several model columns
            sit alongside in the same frame.
        layout: ``"kaggle"`` → ``F1..Fh`` (what ``M5Evaluation.evaluate`` expects).
            ``"d_index"`` → ``d_1942..d_{1942+h-1}`` (raw competition day index).

    Returns:
        Wide DataFrame indexed by ``id_col`` with ``h`` columns, sorted by id.
    """
    if value_col is None:
        candidates = [c for c in preds.columns if c not in (id_col, time_col)]
        if len(candidates) != 1:
            raise ValueError(f"value_col not given and could not be inferred; candidates: {candidates}")
        value_col = candidates[0]

    wide = preds.pivot_table(index=id_col, columns=time_col, values=value_col, observed=True)
    if layout == "kaggle":
        wide.columns = [f"F{i + 1}" for i in range(h)]
    elif layout == "d_index":
        wide.columns = [f"d_{M5_FORECAST_START_DAY + i}" for i in range(h)]
    else:
        raise ValueError(f"Unknown layout: {layout!r}. Use 'kaggle' or 'd_index'.")

    wide.columns.name = None
    wide.index.name = id_col
    return wide
