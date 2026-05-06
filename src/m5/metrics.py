"""Point-forecast metrics on Nixtla long frames.

These return *per-series* values so they compose cleanly with the per-fold,
per-horizon, per-segment, and per-level aggregators in :mod:`m5.scoring`.
WRMSSE itself lives in :mod:`m5.evaluation`; this module is for the lighter
metrics (RMSE, MAE, sMAPE, bias, MASE) that round out the leaderboard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9

__all__ = [
    "aggregate_series_metrics",
    "naive_scale",
    "per_series_metrics",
]


def naive_scale(
    train: pd.DataFrame,
    *,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    season_length: int = 1,
) -> pd.Series:
    """Per-series mean absolute m-step naive differences (MASE denominator).

    For ``season_length=1`` this is the textbook MASE scale; for ``7`` it's
    the seasonal MASE used in M-competitions on weekly retail data.
    Series whose scale is zero (constant target) are dropped — MASE is
    undefined there.
    """
    df = train.sort_values([id_col, time_col])
    diffs = df.groupby(id_col, observed=True)[target_col].diff(season_length)
    scale = diffs.abs().groupby(df[id_col], observed=True).mean()
    scale = scale.replace({0.0: np.nan}).dropna()
    return scale.rename("naive_scale")


def per_series_metrics(
    truth: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    forecast_col: str = "y_hat",
    scales: pd.Series | None = None,
) -> pd.DataFrame:
    """Per-series RMSE, MAE, sMAPE, bias (mean error). MASE if ``scales`` given.

    sMAPE is reported in [0, 2] form (``2|err| / (|y|+|yhat|)``) — multiply by
    100 for the more familiar percent. Bias is signed: positive = over-forecast.
    """
    merged = truth[[id_col, time_col, target_col]].merge(
        forecast[[id_col, time_col, forecast_col]],
        on=[id_col, time_col],
        how="inner",
    )
    if merged.empty:
        raise ValueError("No overlapping rows between truth and forecast.")

    err = merged[forecast_col] - merged[target_col]
    abs_err = err.abs()
    sq_err = err.pow(2)
    denom = (merged[target_col].abs() + merged[forecast_col].abs()).clip(lower=EPS)
    smape_obs = 2.0 * abs_err / denom

    grouper = merged[id_col]
    out = pd.DataFrame(
        {
            "rmse": np.sqrt(sq_err.groupby(grouper, observed=True).mean()),
            "mae": abs_err.groupby(grouper, observed=True).mean(),
            "smape": smape_obs.groupby(grouper, observed=True).mean(),
            "bias": err.groupby(grouper, observed=True).mean(),
            "n_obs": merged.groupby(grouper, observed=True).size(),
        }
    )
    if scales is not None:
        common = out.index.intersection(scales.index)
        out["mase"] = np.nan
        out.loc[common, "mase"] = (out.loc[common, "mae"] / scales.loc[common]).to_numpy()
    return out


def aggregate_series_metrics(
    per_series: pd.DataFrame,
    *,
    weights: pd.Series | None = None,
) -> pd.Series:
    """Reduce per-series metrics to scalars; weighted if ``weights`` given.

    Bias is averaged in *signed* space (so over- and under-forecasts cancel).
    Scale-free metrics (sMAPE, MASE) and absolute-error metrics (RMSE, MAE)
    use the same weighting; this keeps the leaderboard internally consistent.
    """
    cols = [c for c in ("rmse", "mae", "smape", "bias", "mase") if c in per_series.columns]
    if weights is None:
        return per_series[cols].mean()
    common = per_series.index.intersection(weights.index)
    w = weights.loc[common]
    w = w / w.sum() if w.sum() > 0 else w
    return pd.Series(
        {c: float((per_series.loc[common, c] * w).sum()) for c in cols},
        name="weighted",
    )
