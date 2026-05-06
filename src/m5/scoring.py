"""Multi-axis scoring composer for CV outputs.

Consumes the wide CV frames written by :mod:`m5.cv` (``unique_id, ds, cutoff,
y, model_a, model_b, ...``) and produces tidy DataFrames the reporting layer
can render directly. Every axis (fold, horizon, segment, level) returns the
same ``(model, ..., wrmsse[, rmse, mae, smape, bias])`` schema so figures stay
boring and consistent.

WRMSSE itself is delegated to :mod:`m5.evaluation`; this module just slices,
aggregates, and re-normalises components for each axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from m5.evaluation import WRMSSEComponents, wrmsse
from m5.hierarchy import M5_LEVELS_SPEC, TOTAL_COL, TOTAL_VALUE
from m5.metrics import aggregate_series_metrics, naive_scale, per_series_metrics

__all__ = [
    "ScoringInputs",
    "bias_variance_decomposition",
    "discover_models",
    "error_concentration",
    "headline_scores",
    "paired_bootstrap_pvalues",
    "per_fold_scores",
    "per_horizon_scores",
    "per_level_scores",
    "per_segment_scores",
    "residuals_long",
]

_PROTECTED = frozenset({"unique_id", "ds", "y", "cutoff"})


@dataclass(frozen=True)
class ScoringInputs:
    """Container the CLI builds once and threads through every axis."""

    cv_df: pd.DataFrame  # wide: unique_id, ds, cutoff, y, <model>, ...
    train: pd.DataFrame  # long training frame ending before first CV cutoff
    statics: pd.DataFrame  # one row per unique_id with attribute columns
    components: WRMSSEComponents
    models: list[str]


def discover_models(cv_df: pd.DataFrame, *, exclude: tuple[str, ...] = ()) -> list[str]:
    """Forecast columns in a wide CV frame, excluding protected names."""
    skip = _PROTECTED | set(exclude)
    return [c for c in cv_df.columns if c not in skip]


def _scales_from_train(train: pd.DataFrame) -> pd.Series:
    return naive_scale(train, season_length=1)


def headline_scores(inp: ScoringInputs) -> pd.DataFrame:
    """One row per model: WRMSSE + weighted RMSE/MAE/sMAPE/bias/MASE."""
    truth = inp.cv_df[["unique_id", "ds", "y"]]
    scales = _scales_from_train(inp.train)
    rows: list[dict[str, object]] = []
    for m in inp.models:
        ps = per_series_metrics(
            truth,
            inp.cv_df.rename(columns={m: "y_hat"}),
            forecast_col="y_hat",
            scales=scales,
        )
        agg = aggregate_series_metrics(ps, weights=inp.components.weights)
        wr = wrmsse(truth, inp.cv_df.rename(columns={m: "y_hat"}), inp.components)
        row: dict[str, object] = {"model": m, "wrmsse": wr}
        row.update({str(k): v for k, v in agg.to_dict().items()})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("wrmsse", kind="stable").reset_index(drop=True)


def per_fold_scores(inp: ScoringInputs) -> pd.DataFrame:
    """WRMSSE per (model, cutoff)."""
    rows = []
    for cutoff, fold in inp.cv_df.groupby("cutoff", observed=True):
        truth = fold[["unique_id", "ds", "y"]]
        ts = pd.Timestamp(cutoff)  # type: ignore[arg-type]
        for m in inp.models:
            try:
                wr = wrmsse(truth, fold.rename(columns={m: "y_hat"}), inp.components)
            except ValueError:
                continue
            rows.append({"model": m, "cutoff": ts, "wrmsse": wr})
    return pd.DataFrame(rows)


def per_horizon_scores(inp: ScoringInputs) -> pd.DataFrame:
    """WRMSSE per (model, lead-time-in-days)."""
    df = inp.cv_df.copy()
    df["h"] = (df["ds"] - df["cutoff"]).dt.days.astype(int)
    rows = []
    for h, group in df.groupby("h"):
        truth = group[["unique_id", "ds", "y"]]
        h_int = int(h)  # type: ignore[arg-type]
        for m in inp.models:
            try:
                wr = wrmsse(truth, group.rename(columns={m: "y_hat"}), inp.components)
            except ValueError:
                continue
            rows.append({"model": m, "h": h_int, "wrmsse": wr})
    return pd.DataFrame(rows).sort_values(["model", "h"]).reset_index(drop=True)


def per_segment_scores(
    inp: ScoringInputs,
    segment_col: str,
) -> pd.DataFrame:
    """WRMSSE within each value of ``segment_col`` (weights re-normalised)."""
    if segment_col not in inp.statics.columns:
        return pd.DataFrame(columns=["model", "segment", "wrmsse", "n_series"])

    if segment_col in inp.cv_df.columns:
        df = inp.cv_df
    else:
        df = inp.cv_df.merge(
            inp.statics[["unique_id", segment_col]].drop_duplicates("unique_id"),
            on="unique_id",
            how="left",
        )
    rows = []
    for seg, group in df.groupby(segment_col, observed=True):
        ids = pd.Index(group["unique_id"].unique())
        common = inp.components.weights.index.intersection(ids)
        if len(common) == 0 or inp.components.weights.loc[common].sum() == 0:
            continue
        w = inp.components.weights.loc[common]
        w = w / w.sum()
        seg_components = WRMSSEComponents(weights=w, scales=inp.components.scales)
        truth = group[["unique_id", "ds", "y"]]
        for m in inp.models:
            try:
                wr = wrmsse(truth, group.rename(columns={m: "y_hat"}), seg_components)
            except ValueError:
                continue
            rows.append({"model": m, "segment": str(seg), "wrmsse": wr, "n_series": len(common)})
    return pd.DataFrame(rows)


def _level_components(level_train: pd.DataFrame) -> WRMSSEComponents:
    """Build WRMSSE components for an *already-aggregated* training frame."""
    df = level_train.sort_values(["unique_id", "ds"])
    last_28 = (
        df.groupby("unique_id", observed=True).tail(28).groupby("unique_id", observed=True)["_rev"].sum()
    )
    total = last_28.sum()
    weights = (last_28 / total).rename("weight") if total > 0 else last_28.rename("weight")
    diffs = df.groupby("unique_id", observed=True)["y"].diff()
    scales = diffs.pow(2).groupby(df["unique_id"], observed=True).mean().rename("scale")
    scales = scales.replace({0.0: np.nan}).dropna()
    common = weights.index.intersection(scales.index)
    return WRMSSEComponents(weights=weights.loc[common], scales=scales.loc[common])


def per_level_scores(inp: ScoringInputs) -> pd.DataFrame:
    """WRMSSE at each of the 12 M5 levels (bottom forecasts aggregated up).

    Aggregating bottom-level forecasts upward and scoring at each level is the
    canonical M5 evaluation. Per-level weights are dollar-sales of the
    aggregated series; per-level scales are naive-1 differenced MSE of the
    aggregated training series.
    """
    needed = {col for level in M5_LEVELS_SPEC for col in level} - {TOTAL_COL}
    missing = needed - set(inp.statics.columns)
    if missing:
        raise ValueError(f"Statics missing columns required for hierarchy: {sorted(missing)}")

    statics = inp.statics.drop_duplicates("unique_id")
    cv_missing = [c for c in needed if c not in inp.cv_df.columns]
    if cv_missing:
        cv = inp.cv_df.merge(statics[["unique_id", *cv_missing]], on="unique_id", how="left")
    else:
        cv = inp.cv_df.copy()
    cv[TOTAL_COL] = TOTAL_VALUE
    if "sell_price" in inp.train.columns:
        train_keep = ["unique_id", "ds", "y", "sell_price"]
    else:
        train_keep = ["unique_id", "ds", "y"]
    train_missing = [c for c in needed if c not in inp.train.columns]
    if train_missing:
        train = inp.train[train_keep].merge(
            statics[["unique_id", *train_missing]], on="unique_id", how="left"
        )
    else:
        train = inp.train[[*train_keep, *needed]].copy()
    train[TOTAL_COL] = TOTAL_VALUE
    train["_rev"] = (
        train["y"] * train["sell_price"].fillna(0) if "sell_price" in train.columns else train["y"]
    )

    rows = []
    for level_spec in M5_LEVELS_SPEC:
        level_name = "/".join(c for c in level_spec if c != TOTAL_COL) or TOTAL_COL
        agg_train = (
            train.groupby([*level_spec, "ds"], observed=True)
            .agg(y=("y", "sum"), _rev=("_rev", "sum"))
            .reset_index()
        )
        agg_train["unique_id"] = agg_train[level_spec].astype(str).agg("/".join, axis=1)
        components = _level_components(agg_train)

        agg_cv_groups = cv.groupby([*level_spec, "ds"], observed=True)
        agg_cv = agg_cv_groups.agg(y=("y", "sum")).reset_index()
        for m in inp.models:
            agg_cv[m] = agg_cv_groups[m].sum().to_numpy()
        agg_cv["unique_id"] = agg_cv[level_spec].astype(str).agg("/".join, axis=1)

        truth = agg_cv[["unique_id", "ds", "y"]]
        for m in inp.models:
            try:
                wr = wrmsse(truth, agg_cv.rename(columns={m: "y_hat"}), components)
            except ValueError:
                continue
            rows.append(
                {
                    "model": m,
                    "level": level_name,
                    "level_idx": M5_LEVELS_SPEC.index(level_spec),
                    "n_series": len(components.weights),
                    "wrmsse": wr,
                }
            )
    return pd.DataFrame(rows)


def bias_variance_decomposition(inp: ScoringInputs) -> pd.DataFrame:
    """Pool residuals across all (series, time) and split MSE = bias² + variance.

    The decomposition is exact in expectation: ``MSE = mean(r)² + var(r)``
    where ``r = ŷ - y``. A model with high bias (systematically over/under)
    looks different on this chart from one whose errors are well-centred but
    noisy, even if the WRMSSE is the same.
    """
    truth = inp.cv_df[["unique_id", "ds", "y"]]
    rows = []
    for m in inp.models:
        merged = truth.merge(inp.cv_df[["unique_id", "ds", m]], on=["unique_id", "ds"])
        residuals = (merged[m] - merged["y"]).to_numpy()
        bias = float(residuals.mean())
        variance = float(residuals.var(ddof=0))
        mse = float((residuals**2).mean())
        rows.append(
            {
                "model": m,
                "bias": bias,
                "bias_sq": bias**2,
                "variance": variance,
                "mse": mse,
                "rmse": float(np.sqrt(mse)),
            }
        )
    return pd.DataFrame(rows).sort_values("mse").reset_index(drop=True)


def error_concentration(inp: ScoringInputs) -> pd.DataFrame:
    """Lorenz-style per-model curve: cumulative weight vs. cumulative error share.

    For each model, sort series by descending dollar weight, then plot the
    cumulative weight on x against the cumulative share of weighted RMSSE on
    y. The diagonal means errors are proportional to weight; bowing above the
    diagonal means the heaviest series carry an outsized share of the error.
    """
    truth = inp.cv_df[["unique_id", "ds", "y"]]
    weights = inp.components.weights
    scales = inp.components.scales
    rows = []
    for m in inp.models:
        merged = truth.merge(inp.cv_df[["unique_id", "ds", m]], on=["unique_id", "ds"])
        sq_err = (merged[m] - merged["y"]).pow(2)
        mse_per_series = sq_err.groupby(merged["unique_id"], observed=True).mean()
        common = weights.index.intersection(mse_per_series.index).intersection(scales.index)
        rmsse = np.sqrt(mse_per_series.loc[common] / scales.loc[common])
        contribution = (weights.loc[common] * rmsse).sort_values(ascending=False)
        total_contrib = float(contribution.sum())
        if total_contrib <= 0:
            continue  # perfect forecast — nothing to concentrate
        weight_sorted = weights.loc[contribution.index]
        cum_weight = weight_sorted.cumsum().to_numpy() / weight_sorted.sum()
        cum_error = contribution.cumsum().to_numpy() / total_contrib
        for r, (w, e) in enumerate(zip(cum_weight, cum_error, strict=True), start=1):
            rows.append({"model": m, "rank": r, "cum_weight": float(w), "cum_error_share": float(e)})
    return pd.DataFrame(rows)


def paired_bootstrap_pvalues(
    inp: ScoringInputs,
    *,
    n_iter: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Pairwise p-values for "model A is no better than model B".

    Resamples series with replacement; computes weighted mean RMSSE
    differential ``d = RMSSE_A - RMSSE_B``; reports the right-tail probability
    of seeing ``d ≥ 0`` under the bootstrap distribution. Small p ⇒ strong
    evidence A beats B.
    """
    rng = np.random.default_rng(seed)
    truth = inp.cv_df[["unique_id", "ds", "y"]]
    weights_full = inp.components.weights
    scales_full = inp.components.scales

    rmsse_per_model: dict[str, pd.Series] = {}
    common_idx: pd.Index | None = None
    for m in inp.models:
        merged = truth.merge(inp.cv_df[["unique_id", "ds", m]], on=["unique_id", "ds"])
        mse_s = (merged[m] - merged["y"]).pow(2).groupby(merged["unique_id"], observed=True).mean()
        idx = mse_s.index.intersection(weights_full.index).intersection(scales_full.index)
        common_idx = idx if common_idx is None else common_idx.intersection(idx)
        ratio = (mse_s.reindex(idx) / scales_full.loc[idx]).to_numpy()
        rmsse_per_model[m] = pd.Series(np.sqrt(ratio), index=idx)

    if common_idx is None or len(common_idx) == 0:
        return pd.DataFrame(np.nan, index=inp.models, columns=inp.models)

    common_idx = pd.Index(common_idx)
    w = weights_full.loc[common_idx].to_numpy()
    w = w / w.sum() if w.sum() > 0 else w
    rmsse_arr: dict[str, np.ndarray] = {m: rmsse_per_model[m].loc[common_idx].to_numpy() for m in inp.models}

    n_series = len(common_idx)
    # One shared resample matrix → all model pairs use the same draws (paired).
    samples = rng.integers(0, n_series, size=(n_iter, n_series))

    pvalues = pd.DataFrame(np.eye(len(inp.models)), index=inp.models, columns=inp.models)
    for i, m1 in enumerate(inp.models):
        for j, m2 in enumerate(inp.models):
            if i == j:
                continue
            diff = rmsse_arr[m1] - rmsse_arr[m2]
            boot = (w[samples] * diff[samples]).sum(axis=1)
            pvalues.loc[m1, m2] = float((boot >= 0).mean())
    return pvalues


def residuals_long(inp: ScoringInputs) -> pd.DataFrame:
    """Tidy residual frame for diagnostic plots: model, ds, residual."""
    truth = inp.cv_df[["unique_id", "ds", "y"]]
    rows = []
    for m in inp.models:
        merged = truth.merge(inp.cv_df[["unique_id", "ds", m]], on=["unique_id", "ds"])
        merged["residual"] = merged[m] - merged["y"]
        merged["model"] = m
        rows.append(merged[["model", "unique_id", "ds", "y", m, "residual"]].rename(columns={m: "y_hat"}))
    return pd.concat(rows, ignore_index=True)
