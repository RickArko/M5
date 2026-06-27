"""WRMSSE \u2014 Weighted Root Mean Squared Scaled Error (M5 official metric).

Implements the bottom-level (item \u00d7 store) score directly. Hierarchical
aggregation across the 12 M5 levels can be added by precomputing series
weights at each level and reusing :func:`wrmsse_from_components`.

Reference: https://mofc.unic.ac.cy/m5-competition/
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from m5.backend import from_native, nw, to_pandas

# M5 competition day-index constants \u2014 fixed by the competition split, not env.
# d_1 .. d_1941 = train, d_1942 .. d_1969 = held-out evaluation window.
M5_TRAIN_END_DAY: int = 1941
M5_FORECAST_START_DAY: int = 1942
M5_FORECAST_END_DAY: int = 1969
M5_HORIZON: int = M5_FORECAST_END_DAY - M5_FORECAST_START_DAY + 1  # 28


@dataclass
class WRMSSEComponents:
    """Per-series weights and scales \u2014 cached so we score many models cheaply.

    ``weights`` and ``scales`` are pandas Series indexed by ``unique_id``.
    """

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

    Uses the narwhals backend internally for all DataFrame operations.
    Returns WRMSSEComponents with pandas Series for backward compatibility.

    Args:
        train: Long training frame ending at the day before the forecast window.
        price_col: If provided, weights are dollar-sales (units * price); else unit-sales.
    """
    df = from_native(train)

    # Compute revenue vector.
    if price_col and price_col in df.columns:
        rev = nw.col(target_col) * nw.col(price_col).fill_null(0)
    else:
        rev = nw.col(target_col)

    df = df.with_columns(rev.alias("_rev"))
    df = df.sort([id_col, time_col])

    # --- Weights: trailing 28-day dollar sales per series ---
    # Sort descending by time within each group, keep first 28 (= most recent).
    sorted_desc = df.sort([id_col, time_col], descending=[False, True])
    last_28 = (
        sorted_desc.with_columns(nw.col(time_col).cum_count().over(id_col).alias("_nw_rn"))
        .filter(nw.col("_nw_rn") <= 28)
        .drop(["_nw_rn"])
    )

    # Sum revenue per series for the trailing 28 days.
    rev_sum = last_28.group_by(id_col).agg(nw.col("_rev").sum().alias("_rev_sum"))
    total_rev = rev_sum.select(nw.col("_rev_sum").sum()).item()
    weights = to_pandas(
        rev_sum.with_columns((nw.col("_rev_sum") / total_rev).alias("weight")).select(id_col, "weight")
    ).set_index(id_col)["weight"]

    # --- Scales: Naive-1 in-sample MSE ---
    # Diff y within each group → squared → mean (pre-square so group_by is a simple mean).
    df_no_rev = df.drop(["_rev"])
    diffs = df_no_rev.with_columns(nw.col(target_col).diff().over(id_col).alias("_diff"))
    diffs = diffs.with_columns((nw.col("_diff") ** 2).alias("_diff_sq"))
    scales_nw = diffs.group_by(id_col).agg(nw.col("_diff_sq").mean().alias("scale"))
    scales_nw = scales_nw.filter(nw.col("scale").is_null() | (nw.col("scale") != 0.0))
    scales = to_pandas(scales_nw.select(id_col, "scale")).set_index(id_col)["scale"]

    # --- Align ---
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
    truth_nw = from_native(truth)
    forecast_nw = from_native(forecast)

    merged = truth_nw.select(id_col, time_col, target_col).join(
        forecast_nw.select(id_col, time_col, forecast_col),
        on=[id_col, time_col],
        how="inner",
    )
    if merged.select(nw.len()).item() == 0:
        raise ValueError("No overlapping rows between truth and forecast.")

    # Per-series MSE.
    err_sq = (nw.col(target_col) - nw.col(forecast_col)) ** 2
    mse_series = to_pandas(
        merged.with_columns(err_sq.alias("_err_sq"))
        .group_by(id_col)
        .agg(nw.col("_err_sq").mean().alias("mse"))
        .select(id_col, "mse")
    ).set_index(id_col)["mse"]

    common = components.weights.index.intersection(mse_series.index)
    if len(common) == 0:
        raise ValueError(
            "WRMSSE components share no unique_ids with the forecast \u2014 likely the "
            "training frame used for compute_components doesn't match the CV frame. "
            f"forecast has {len(mse_series)} series; "
            f"components have {len(components.weights)} series."
        )
    rmsse = np.sqrt(mse_series.loc[common] / components.scales.loc[common])
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


def accuracy_for_models(
    truth: pd.DataFrame,
    forecasts: pd.DataFrame,
    *,
    model_cols: list[str] | None = None,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
) -> pd.DataFrame:
    """Per-model MAE, RMSE, sMAPE, and signed bias on the bottom level."""
    if model_cols is None:
        excluded = {id_col, time_col, target_col, "cutoff"}
        model_cols = [c for c in forecasts.columns if c not in excluded]

    merged = truth[[id_col, time_col, target_col]].merge(
        forecasts[[id_col, time_col, *model_cols]],
        on=[id_col, time_col],
        how="inner",
    )
    if merged.empty:
        raise ValueError("No overlapping rows between truth and forecasts.")

    rows: list[dict[str, float | str]] = []
    denom = merged[target_col].abs().mean()
    for model in model_cols:
        err = merged[model] - merged[target_col]
        rows.append(
            {
                "model": model,
                "MAE": float(np.abs(err).mean()),
                "RMSE": float(np.sqrt((err**2).mean())),
                "sMAPE": float(
                    (
                        2 * np.abs(err) / (merged[target_col].abs() + merged[model].abs()).clip(lower=1e-8)
                    ).mean()
                ),
                "bias": float(err.mean()),
                "bias_pct_of_mean_y": float(err.mean() / denom) if denom else 0.0,
                "n_obs": float(len(merged)),
            }
        )
    return pd.DataFrame(rows).set_index("model").sort_values("MAE")


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
        value_col: Forecast column. If ``None``, inferred as the only column
            not in ``(id_col, time_col)``.
        layout: ``"kaggle"`` \u2192 ``F1..Fh``. ``"d_index"`` \u2192 ``d_1942..``.

    Returns:
        Wide DataFrame indexed by ``id_col`` with ``h`` columns, sorted by id.
    """
    preds_nw = from_native(preds)

    if value_col is None:
        candidates = [c for c in preds_nw.columns if c not in (id_col, time_col)]
        if len(candidates) != 1:
            raise ValueError(f"value_col not given and could not be inferred; candidates: {candidates}")
        value_col = candidates[0]

    wide_nw = preds_nw.pivot(  # noqa: PD010  — narwhals API, not pandas
        values=value_col,
        index=id_col,
        on=time_col,
        sort_columns=True,
    )
    wide = to_pandas(wide_nw).set_index(id_col)

    if layout == "kaggle":
        wide.columns = [f"F{i + 1}" for i in range(h)]
    elif layout == "d_index":
        wide.columns = [f"d_{M5_FORECAST_START_DAY + i}" for i in range(h)]
    else:
        raise ValueError(f"Unknown layout: {layout!r}. Use 'kaggle' or 'd_index'.")

    wide.index.name = id_col
    return wide
