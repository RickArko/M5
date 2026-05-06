"""The 12-figure principal-ML-scientist pack.

Every builder returns a :class:`matplotlib.figure.Figure` and never writes to
disk — :func:`m5.reporting.save.save_figure` handles persistence. Builders
take tidy DataFrames straight off :mod:`m5.scoring`; they don't recompute
metrics.

Figure contract
---------------
* Each builder is named ``fig_*`` and returns one Figure (or ``None`` when
  the input frame is empty / inapplicable; the caller handles the skip).
* Stable model colours via :func:`m5.reporting.style.model_color`.
* Annotations point out the headline finding (best model, biggest gap, etc.)
  so a glance-only reader can extract the lede without reading the caption.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from m5.reporting.style import apply_style, model_color

__all__ = [
    "FigureBundle",
    "build_all_figures",
    "fig_bias_variance",
    "fig_calibration",
    "fig_forecast_examples",
    "fig_leaderboard",
    "fig_lorenz",
    "fig_per_fold_stability",
    "fig_per_horizon_curve",
    "fig_per_level_heatmap",
    "fig_per_segment",
    "fig_residual_histogram",
    "fig_residuals_time",
    "fig_significance_matrix",
]


# --------------------------------------------------------------------------- #
# Figure 01 — headline leaderboard                                            #
# --------------------------------------------------------------------------- #
def fig_leaderboard(headline: pd.DataFrame) -> Figure:
    """Horizontal bar chart of WRMSSE per model with secondary metrics annotated."""
    df = headline.sort_values("wrmsse")
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.55 * len(df) + 1.5)))
    colors = [model_color(m) for m in df["model"]]
    bars = ax.barh(df["model"], df["wrmsse"], color=colors, edgecolor="white")
    ax.invert_yaxis()
    ax.set_xlabel("WRMSSE (lower is better)")
    ax.set_title("Model leaderboard — bottom-level WRMSSE")
    best_w = float(df["wrmsse"].iloc[0])
    for bar, (_, row) in zip(bars, df.iterrows(), strict=False):
        delta = (row["wrmsse"] - best_w) / best_w if best_w > 0 else 0.0
        suffix = "" if row["model"] == df["model"].iloc[0] else f"  (+{delta:.1%})"
        ax.text(
            bar.get_width() * 1.005,
            bar.get_y() + bar.get_height() / 2,
            f"{row['wrmsse']:.4f}{suffix}",
            va="center",
            fontsize=9,
        )
    ax.set_xlim(0, df["wrmsse"].max() * 1.18)
    ax.margins(y=0.05)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure 02 — per-level WRMSSE heatmap                                        #
# --------------------------------------------------------------------------- #
def fig_per_level_heatmap(per_level: pd.DataFrame) -> Figure | None:
    """Heatmap of WRMSSE × {12 M5 levels, models}."""
    if per_level.empty:
        return None
    pivot = per_level.pivot_table(index="level", columns="model", values="wrmsse")
    if "level_idx" in per_level.columns:
        order = per_level.drop_duplicates("level").sort_values("level_idx")["level"].tolist()
        pivot = pivot.reindex(order)
    fig, ax = plt.subplots(figsize=(1.6 + 1.0 * pivot.shape[1], 0.45 * pivot.shape[0] + 1.5))
    im = ax.imshow(pivot.values, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_title("WRMSSE by M5 level (lower is better)")
    pivot_arr = pivot.to_numpy()
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot_arr[i, j]
            if not np.isfinite(v):
                continue
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="WRMSSE")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure 03 — per-fold stability                                              #
# --------------------------------------------------------------------------- #
def fig_per_fold_stability(per_fold: pd.DataFrame) -> Figure | None:
    """Per-fold WRMSSE strip + median bar — shows variance across CV windows."""
    if per_fold.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.35 * per_fold["model"].nunique() + 2.5)))
    models_sorted = per_fold.groupby("model")["wrmsse"].mean().sort_values().index.tolist()
    for i, m in enumerate(models_sorted):
        scores = per_fold.loc[per_fold["model"] == m, "wrmsse"].to_numpy()
        c = model_color(m)
        ax.scatter(scores, np.full_like(scores, i, dtype=float), color=c, s=40, alpha=0.85, zorder=3)
        ax.scatter([scores.mean()], [i], color=c, marker="|", s=600, linewidth=2.5, zorder=4)
        if scores.size > 1:
            lo, hi = scores.min(), scores.max()
            ax.hlines(i, lo, hi, color=c, alpha=0.35, linewidth=2, zorder=2)
    ax.set_yticks(range(len(models_sorted)))
    ax.set_yticklabels(models_sorted)
    ax.invert_yaxis()
    ax.set_xlabel("WRMSSE per CV fold (▮ = mean)")
    ax.set_title("Per-fold stability")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure 04 — per-horizon error curve                                         #
# --------------------------------------------------------------------------- #
def fig_per_horizon_curve(per_horizon: pd.DataFrame) -> Figure | None:
    """WRMSSE vs lead-time (1..H) per model — does error degrade with horizon?"""
    if per_horizon.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m, group in per_horizon.groupby("model"):
        g = group.sort_values("h")
        ax.plot(g["h"].to_numpy(), g["wrmsse"].to_numpy(), marker="o", label=m, color=model_color(str(m)))
    ax.set_xlabel("Lead-time (days from cutoff)")
    ax.set_ylabel("WRMSSE")
    ax.set_title("Error growth by forecast horizon")
    ax.legend(loc="best", title="model")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure 05 — per-segment small-multiples                                     #
# --------------------------------------------------------------------------- #
def fig_per_segment(
    segment_frames: Mapping[str, pd.DataFrame],
) -> Figure | None:
    """Faceted bars: WRMSSE per model within each segment of each cut.

    ``segment_frames`` maps ``cat_id``/``store_id``/``state_id``/``dept_id``
    (or any other static col) to the corresponding scoring DataFrame.
    """
    panels = [(cut, df) for cut, df in segment_frames.items() if not df.empty]
    if not panels:
        return None
    n = len(panels)
    fig, axes = plt.subplots(
        n, 1, figsize=(9, 1.6 + 1.4 * n + 0.18 * sum(p[1]["segment"].nunique() for p in panels))
    )
    axes_list = [axes] if n == 1 else list(axes)
    for ax, (cut, df) in zip(axes_list, panels, strict=True):
        pivot = df.pivot_table(index="segment", columns="model", values="wrmsse")
        pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
        x = np.arange(len(pivot.index))
        width = 0.8 / max(1, pivot.shape[1])
        for i, m in enumerate(pivot.columns):
            ax.bar(
                x + i * width - 0.4 + width / 2,
                pivot[m].to_numpy(),
                width=width,
                color=model_color(m),
                label=m,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=20, ha="right")
        ax.set_ylabel("WRMSSE")
        ax.set_title(f"by {cut}")
    axes_list[0].legend(loc="upper right", title="model", ncol=min(4, len(pivot.columns)))
    fig.suptitle("Per-segment leaderboard", y=1.0, fontsize=13)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure 06 — bias / variance scatter                                         #
# --------------------------------------------------------------------------- #
def fig_bias_variance(bv: pd.DataFrame) -> Figure | None:
    """Scatter of variance vs bias² with iso-MSE contours."""
    if bv.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for _, row in bv.iterrows():
        c = model_color(row["model"])
        ax.scatter(row["bias_sq"], row["variance"], s=110, color=c, edgecolor="white", zorder=3)
        ax.annotate(
            row["model"],
            (row["bias_sq"], row["variance"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
        )
    # Iso-MSE contours: bias² + variance = const
    if len(bv) > 0:
        lo = 0.0
        hi = float(max(bv["mse"].max(), bv["bias_sq"].max() + bv["variance"].max())) * 1.05
        if hi > 0:
            xs = np.linspace(lo, hi, 200)
            for q in (0.25, 0.5, 0.75, 1.0):
                level = q * hi
                ys = level - xs
                m = ys >= 0
                ax.plot(xs[m], ys[m], color="#bbbbbb", linewidth=0.8, linestyle="--", zorder=1)
                if m.any():
                    ax.text(
                        xs[m][-1] * 0.98,
                        ys[m][-1] * 0.02,
                        f"MSE={level:.2g}",
                        fontsize=7,
                        color="#888888",
                    )
            ax.set_xlim(0, hi)
            ax.set_ylim(0, hi)
    ax.set_xlabel("bias² (systematic error)")
    ax.set_ylabel("variance (residual noise)")
    ax.set_title("Bias / variance decomposition (pooled across all (series, time))")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure 07 — pairwise significance matrix                                    #
# --------------------------------------------------------------------------- #
def fig_significance_matrix(pvalues: pd.DataFrame) -> Figure | None:
    """Pairwise p-values that "row beats column" via paired bootstrap."""
    if pvalues.empty:
        return None
    fig, ax = plt.subplots(figsize=(1.6 + 0.7 * pvalues.shape[1], 1.6 + 0.7 * pvalues.shape[0]))
    im = ax.imshow(pvalues.values, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(pvalues.shape[1]))
    ax.set_xticklabels(pvalues.columns, rotation=30, ha="right")
    ax.set_yticks(range(pvalues.shape[0]))
    ax.set_yticklabels(pvalues.index)
    pvalues_arr = pvalues.to_numpy()
    for i in range(pvalues.shape[0]):
        for j in range(pvalues.shape[1]):
            v = pvalues_arr[i, j]
            if i == j:
                txt = "—"
            else:
                txt = "<0.001" if v < 0.001 else f"{v:.3f}"
            ax.text(j, i, txt, ha="center", va="center", color="black", fontsize=8)
    ax.set_xlabel("model B")
    ax.set_ylabel("model A")
    ax.set_title("P(A no better than B)  —  green = A wins")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure 08 — residuals over time                                             #
# --------------------------------------------------------------------------- #
def fig_residuals_time(residuals: pd.DataFrame) -> Figure | None:
    """Per-day mean residual per model — drift, level errors, regime breaks."""
    if residuals.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for m, group in residuals.groupby("model"):
        daily = group.groupby("ds", observed=True)["residual"].mean()
        ax.plot(daily.index, daily.to_numpy(), label=m, color=model_color(str(m)), linewidth=1.4)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("date")
    ax.set_ylabel("mean residual (ŷ − y)")
    ax.set_title("Residual drift over the holdout window")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.legend(loc="best", title="model")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure 09 — residual histogram + QQ                                         #
# --------------------------------------------------------------------------- #
def fig_residual_histogram(residuals: pd.DataFrame) -> Figure | None:
    """Per-model residual distribution: histogram + normal-QQ small multiples."""
    if residuals.empty:
        return None
    models = list(residuals["model"].unique())
    n = len(models)
    fig, axes = plt.subplots(n, 2, figsize=(9, 2.4 * n + 0.5), squeeze=False)
    for i, m in enumerate(models):
        r = residuals.loc[residuals["model"] == m, "residual"].dropna().to_numpy()
        c = model_color(str(m))
        ax_h, ax_q = axes[i]
        if r.size:
            bins = min(60, max(10, int(np.sqrt(r.size))))
            try:
                ax_h.hist(r, bins=bins, color=c, alpha=0.85, edgecolor="white")
            except ValueError:
                # Constant or near-constant residuals — show a single bar.
                ax_h.axvline(float(r[0]), color=c, linewidth=2)
        ax_h.axvline(0, color="black", linewidth=0.8)
        ax_h.set_title(f"{m} — residual distribution")
        ax_h.set_xlabel("residual")
        if r.size:
            r_sorted = np.sort(r)
            quantiles = (np.arange(1, len(r_sorted) + 1) - 0.5) / len(r_sorted)
            from scipy.stats import norm

            theoretical = norm.ppf(quantiles)
            ax_q.scatter(theoretical, r_sorted, s=4, color=c, alpha=0.5)
            mn, mx = float(theoretical.min()), float(theoretical.max())
            slope = float(np.std(r))
            intercept = float(np.mean(r))
            ax_q.plot(
                [mn, mx], [intercept + slope * mn, intercept + slope * mx], color="black", linewidth=0.8
            )
        ax_q.set_title(f"{m} — normal Q-Q")
        ax_q.set_xlabel("theoretical quantiles")
        ax_q.set_ylabel("residual quantiles")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure 10 — calibration / quantile bias                                     #
# --------------------------------------------------------------------------- #
def fig_calibration(residuals: pd.DataFrame) -> Figure | None:
    """Quantile-bias plot: predicted-quantile vs realised-quantile per model.

    For point forecasts (no PIs in the CV frame), this falls back to a binned
    over/under plot — predicted-volume decile vs realised mean residual. It's
    the closest interpretable analogue when intervals aren't available.
    """
    if residuals.empty or "y_hat" not in residuals.columns:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, group in residuals.groupby("model"):
        g = group.dropna(subset=["y_hat", "y", "residual"]).copy()
        if g.empty:
            continue
        g["bin"] = pd.qcut(g["y_hat"], q=10, labels=False, duplicates="drop")
        binned = g.groupby("bin", observed=True).agg(
            yhat=("y_hat", "mean"), y=("y", "mean"), resid=("residual", "mean")
        )
        ax.plot(binned["yhat"], binned["resid"], marker="o", color=model_color(str(m)), label=m)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("forecast magnitude (decile midpoint)")
    ax.set_ylabel("mean residual within decile")
    ax.set_title("Calibration — bias by forecast decile")
    ax.legend(loc="best", title="model")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure 11 — forecast vs actual exemplars                                    #
# --------------------------------------------------------------------------- #
def _select_exemplar_series(train: pd.DataFrame, n_per_tier: int = 2) -> list[str]:
    """Pick six series spanning {high, mid, low volume} × {regular, intermittent}."""
    grouped = train.groupby("unique_id", observed=True)
    summary = pd.DataFrame(
        {
            "mean_y": grouped["y"].mean(),
            "zero_share": grouped["y"].apply(lambda s: float((s == 0).mean())),
        }
    ).dropna()
    if summary.empty:
        return list(train["unique_id"].unique()[:6])
    ranked = summary.sort_values("mean_y", ascending=False)
    n = len(ranked)
    tiers = {
        "high": ranked.iloc[: max(1, n // 3)],
        "mid": ranked.iloc[max(1, n // 3) : max(2, 2 * n // 3)],
        "low": ranked.iloc[max(2, 2 * n // 3) :],
    }
    picks: list[str] = []
    for tier_df in tiers.values():
        if tier_df.empty:
            continue
        regular = tier_df.sort_values("zero_share").index.tolist()
        intermittent = tier_df.sort_values("zero_share", ascending=False).index.tolist()
        if regular:
            picks.append(regular[0])
        if intermittent and (not regular or intermittent[0] != regular[0]):
            picks.append(intermittent[0])
        if len(picks) >= n_per_tier * 3:
            break
    seen: list[str] = []
    for sid in picks:
        if sid not in seen:
            seen.append(sid)
    return seen[: n_per_tier * 3]


def fig_forecast_examples(
    cv_df: pd.DataFrame,
    train: pd.DataFrame,
    models: list[str],
    *,
    history_days: int = 56,
) -> Figure | None:
    """3×2 small-multiples of forecast vs actual on six representative series."""
    series = _select_exemplar_series(train)
    if not series:
        return None
    rows = (len(series) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(11, 2.6 * rows + 0.4), squeeze=False)

    for ax, sid in zip(axes.flatten(), series, strict=False):
        hist = train[train["unique_id"] == sid].sort_values("ds")
        if hist.empty:
            ax.set_visible(False)
            continue
        cutoff_max = cv_df.loc[cv_df["unique_id"] == sid, "ds"].min()
        if pd.isna(cutoff_max):
            cutoff_max = hist["ds"].max()
        recent = hist[hist["ds"] >= cutoff_max - pd.Timedelta(days=history_days)]
        ax.plot(recent["ds"], recent["y"], color="#444", linewidth=1.2, label="actual (history)", alpha=0.7)
        fold = cv_df[cv_df["unique_id"] == sid].sort_values("ds")
        ax.plot(fold["ds"], fold["y"], color="black", linewidth=1.6, label="actual")
        for m in models:
            if m not in fold.columns:
                continue
            ax.plot(fold["ds"], fold[m], color=model_color(m), linewidth=1.4, alpha=0.9, label=m)
        ax.axvspan(fold["ds"].min(), fold["ds"].max(), color="#ffe6cc", alpha=0.35, zorder=0)
        ax.set_title(sid, fontsize=10)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=4))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    for ax in axes.flatten()[len(series) :]:
        ax.set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5), bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Forecast vs actual — representative series", y=1.04, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# --------------------------------------------------------------------------- #
# Figure 12 — Lorenz error concentration                                      #
# --------------------------------------------------------------------------- #
def fig_lorenz(error_curves: pd.DataFrame) -> Figure | None:
    """Per-model cumulative-error vs cumulative-weight (Lorenz-style)."""
    if error_curves.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=1, label="proportional")
    for m, group in error_curves.groupby("model"):
        g = group.sort_values("rank")
        ax.plot(
            g["cum_weight"].to_numpy(), g["cum_error_share"].to_numpy(), label=m, color=model_color(str(m))
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("cumulative dollar-weight (series, sorted descending by error contribution)")
    ax.set_ylabel("cumulative share of weighted RMSSE")
    ax.set_title("Where is the error concentrated?")
    ax.legend(loc="lower right", title="model")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Bundle + dispatcher                                                          #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FigureBundle:
    """All figures + the captions/insights the report stitcher consumes."""

    figures: dict[str, Figure]
    captions: dict[str, str]
    insights: dict[str, str]

    def __iter__(self):
        return iter(self.figures.items())


_FIGURE_ORDER: tuple[tuple[str, str], ...] = (
    ("01_leaderboard", "Headline leaderboard"),
    ("02_per_level_heatmap", "Per-level WRMSSE (12 M5 levels)"),
    ("03_per_fold_stability", "Per-fold stability"),
    ("04_per_horizon_curve", "Error growth by lead-time"),
    ("05_per_segment", "Per-segment leaderboard"),
    ("06_bias_variance", "Bias / variance decomposition"),
    ("07_significance_matrix", "Pairwise paired-bootstrap p-values"),
    ("08_residuals_time", "Residuals over time"),
    ("09_residual_histogram", "Residual distribution + Q-Q"),
    ("10_calibration", "Calibration by forecast decile"),
    ("11_forecast_examples", "Forecast vs actual exemplars"),
    ("12_lorenz", "Error concentration (Lorenz)"),
)

_DEFAULT_CAPTIONS: dict[str, str] = {
    "01_leaderboard": (
        "Bottom-level WRMSSE per model. Lower is better; relative gap to the leader is annotated."
    ),
    "02_per_level_heatmap": (
        "Bottom-level forecasts aggregated up to each of the 12 official M5 "
        "levels and re-scored. Shows which levels each model wins or loses."
    ),
    "03_per_fold_stability": (
        "Per-fold WRMSSE across the rolling-origin CV windows. Tight clusters "
        "mean the model generalises stably across time; spread means luck."
    ),
    "04_per_horizon_curve": (
        "WRMSSE at each lead-time within the H-day horizon. Steep slopes flag "
        "models whose accuracy decays fast — relevant for ops planning."
    ),
    "05_per_segment": (
        "Leaderboard re-computed within each value of selected static cuts "
        "(category, state, store). Surfaces segment-specific winners."
    ),
    "06_bias_variance": (
        "Residual MSE decomposed into bias² and variance. Two models with the "
        "same WRMSSE can sit in very different places here — a high-bias model "
        "is calibratable, a high-variance one needs better signal."
    ),
    "07_significance_matrix": (
        "Pairwise paired-bootstrap probability that row's WRMSSE is no better "
        "than column's. Green cells flag wins with statistical evidence."
    ),
    "08_residuals_time": (
        "Daily mean residual (ŷ − y) per model over the holdout. Slopes mean "
        "drift; bias above/below zero means consistent over/under-forecasting."
    ),
    "09_residual_histogram": (
        "Residual distribution and normal Q-Q per model. Heavy tails / skew "
        "show up here even when the WRMSSE looks healthy."
    ),
    "10_calibration": (
        "Mean residual within each decile of forecast magnitude. A flat line "
        "at zero is well-calibrated; tilt = systematic over/under at certain "
        "forecast ranges."
    ),
    "11_forecast_examples": (
        "Six representative series spanning high/mid/low volume × "
        "regular/intermittent. Sanity check beyond aggregate metrics."
    ),
    "12_lorenz": (
        "Cumulative dollar-weight vs cumulative share of weighted RMSSE. "
        "Bowing above the diagonal means the heaviest series carry an "
        "outsized share of the error."
    ),
}


def _build_insights(
    headline: pd.DataFrame,
    per_fold: pd.DataFrame,
    per_horizon: pd.DataFrame,
    per_level: pd.DataFrame,
    bv: pd.DataFrame,
    pvalues: pd.DataFrame | None,
) -> dict[str, str]:
    out: dict[str, str] = {}
    if not headline.empty:
        best = headline.iloc[0]
        worst = headline.iloc[-1]
        gap = (worst["wrmsse"] - best["wrmsse"]) / best["wrmsse"] if best["wrmsse"] > 0 else 0.0
        out["01_leaderboard"] = (
            f"**{best['model']}** leads with WRMSSE = {best['wrmsse']:.4f}; "
            f"{worst['model']} trails by {gap:.1%}."
        )
    if not per_level.empty:
        per_level_best = per_level.sort_values("wrmsse").groupby("level").head(1)
        winners = per_level_best["model"].value_counts()
        out["02_per_level_heatmap"] = (
            f"Best model wins {winners.iloc[0]} of {len(per_level['level'].unique())} M5 "
            f"levels: **{winners.index[0]}**."
        )
    if not per_fold.empty:
        stab = per_fold.groupby("model")["wrmsse"].std().sort_values()
        out["03_per_fold_stability"] = (
            f"Most stable across folds: **{stab.index[0]}** "
            f"(std = {stab.iloc[0]:.4f}); least: {stab.index[-1]} ({stab.iloc[-1]:.4f})."
        )
    if not per_horizon.empty:
        hmin = per_horizon["h"].min()
        hmax = per_horizon["h"].max()
        slope = (
            per_horizon.groupby("model")
            .apply(  # type: ignore[call-overload]
                lambda g: float(np.polyfit(g["h"], g["wrmsse"], 1)[0] if len(g) > 1 else 0.0),
                include_groups=False,
            )
            .sort_values()
        )
        out["04_per_horizon_curve"] = (
            f"Best degradation slope (h={hmin}→{hmax}): **{slope.index[0]}** "
            f"({slope.iloc[0]:+.5f} WRMSSE/day)."
        )
    if not bv.empty:
        most_biased = bv.iloc[bv["bias_sq"].argmax()]
        out["06_bias_variance"] = (
            f"Most-biased model: **{most_biased['model']}** (bias = {most_biased['bias']:+.3f})."
        )
    if pvalues is not None and not pvalues.empty and len(pvalues) >= 2:
        # Find the smallest p-value off-diagonal (clearest win).
        mat = pvalues.where(~np.eye(len(pvalues), dtype=bool))
        if mat.notna().any().any():
            best_pair = mat.stack().idxmin()
            row, col = (str(best_pair[0]), str(best_pair[1])) if isinstance(best_pair, tuple) else ("", "")
            if row and col:
                p = mat.loc[row, col]
                out["07_significance_matrix"] = (
                    f"Strongest evidence: **{row}** beats **{col}** with p = "
                    f"{p:.4f} (paired bootstrap, 1000 resamples)."
                )
    return out


def build_all_figures(
    *,
    headline: pd.DataFrame,
    per_fold: pd.DataFrame,
    per_horizon: pd.DataFrame,
    per_level: pd.DataFrame,
    segment_frames: Mapping[str, pd.DataFrame],
    bv: pd.DataFrame,
    pvalues: pd.DataFrame | None,
    residuals: pd.DataFrame,
    error_curves: pd.DataFrame,
    cv_df: pd.DataFrame,
    train: pd.DataFrame,
    models: list[str],
) -> FigureBundle:
    """Build every figure that has data; skip cleanly when an input is empty."""
    apply_style()
    builders: list[tuple[str, Callable[[], Figure | None]]] = [
        ("01_leaderboard", lambda: fig_leaderboard(headline)),
        ("02_per_level_heatmap", lambda: fig_per_level_heatmap(per_level)),
        ("03_per_fold_stability", lambda: fig_per_fold_stability(per_fold)),
        ("04_per_horizon_curve", lambda: fig_per_horizon_curve(per_horizon)),
        ("05_per_segment", lambda: fig_per_segment(segment_frames)),
        ("06_bias_variance", lambda: fig_bias_variance(bv)),
        ("07_significance_matrix", lambda: fig_significance_matrix(pvalues) if pvalues is not None else None),
        ("08_residuals_time", lambda: fig_residuals_time(residuals)),
        ("09_residual_histogram", lambda: fig_residual_histogram(residuals)),
        ("10_calibration", lambda: fig_calibration(residuals)),
        ("11_forecast_examples", lambda: fig_forecast_examples(cv_df, train, models)),
        ("12_lorenz", lambda: fig_lorenz(error_curves)),
    ]
    figures: dict[str, Figure] = {}
    for name, fn in builders:
        fig = fn()
        if fig is not None:
            figures[name] = fig

    captions = {n: _DEFAULT_CAPTIONS.get(n, "") for n, _ in _FIGURE_ORDER}
    insights = _build_insights(headline, per_fold, per_horizon, per_level, bv, pvalues)
    return FigureBundle(figures=figures, captions=captions, insights=insights)
