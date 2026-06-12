"""Export M5 CV artifacts into a static JSON bundle for the Vue dashboard."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from m5.evaluation import WRMSSEComponents, compute_components, wrmsse
from m5.hierarchy import M5_LEVELS_SPEC, TOTAL_COL, TOTAL_VALUE
from m5.metrics import aggregate_series_metrics, naive_scale, per_series_metrics
from m5.scoring import ScoringInputs, fva_scores, per_fold_scores, per_horizon_scores

KEY_COLS = ("unique_id", "ds", "cutoff", "y")
STATIC_COLS = ("item_id", "dept_id", "cat_id", "store_id", "state_id")
SEGMENT_COLS = ("state_id", "store_id", "cat_id", "dept_id")
EPS = 1e-9


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_cv_files(artifacts_dir: Path, names: list[str] | None) -> tuple[pd.DataFrame, list[str], list[str]]:
    paths = (
        [artifacts_dir / f"cv_{name}.parquet" for name in names]
        if names
        else sorted(artifacts_dir.glob("cv_*.parquet"))
    )
    paths = [p for p in paths if p.is_file()]
    if not paths:
        raise FileNotFoundError(f"No CV artifacts found under {artifacts_dir}")

    merged: pd.DataFrame | None = None
    models: list[str] = []
    used: list[str] = []
    for path in paths:
        source = path.stem.removeprefix("cv_")
        df = pd.read_parquet(path)
        missing = set(KEY_COLS) - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        df["ds"] = pd.to_datetime(df["ds"])
        df["cutoff"] = pd.to_datetime(df["cutoff"])
        rename: dict[str, str] = {}
        for col in df.columns:
            if col in KEY_COLS:
                continue
            name = f"{source}_{col}"
            rename[col] = name
            models.append(name)
        if not rename:
            continue
        df = df.rename(columns=rename)
        keep = [*KEY_COLS, *rename.values()]
        if merged is None:
            merged = df[keep].copy()
        else:
            merged = merged.merge(
                df[["unique_id", "ds", "cutoff", *rename.values()]],
                on=["unique_id", "ds", "cutoff"],
                how="inner",
            )
        used.append(path.name)

    if merged is None or merged.empty:
        raise ValueError("Merged CV frame is empty. Check that artifacts share CV keys.")
    return merged, models, used


def statics_from_long(long_df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in STATIC_COLS if c in long_df.columns]
    return long_df[["unique_id", *present]].drop_duplicates("unique_id")


def weighted_metrics(
    truth: pd.DataFrame,
    forecast: pd.DataFrame,
    components: WRMSSEComponents,
    model_col: str,
    *,
    scales: pd.Series | None = None,
) -> dict[str, float]:
    renamed = forecast.rename(columns={model_col: "y_hat"})
    out: dict[str, float] = {"wrmsse": wrmsse(truth, renamed, components, forecast_col="y_hat")}
    ps = per_series_metrics(truth, renamed, forecast_col="y_hat", scales=scales)
    agg = aggregate_series_metrics(ps, weights=components.weights)
    out.update({str(k): float(v) for k, v in agg.to_dict().items()})

    merged = truth.merge(renamed[["unique_id", "ds", "y_hat"]], on=["unique_id", "ds"], how="inner")
    err = merged["y_hat"] - merged["y"]
    actual_abs = float(merged["y"].abs().sum())
    actual = float(merged["y"].sum())
    out["wmape"] = float(err.abs().sum() / max(actual_abs, EPS))
    out["bias_pct"] = float(err.sum() / max(abs(actual), EPS))
    return out


def headline(inp: ScoringInputs) -> pd.DataFrame:
    truth = inp.cv_df[["unique_id", "ds", "y"]]
    scales = naive_scale(inp.train, season_length=1)
    rows = []
    for model in inp.models:
        rows.append(
            {"model": model, **weighted_metrics(truth, inp.cv_df, inp.components, model, scales=scales)}
        )
    return pd.DataFrame(rows).sort_values("wrmsse", kind="stable").reset_index(drop=True)


def segment_scores(inp: ScoringInputs) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    statics = inp.statics.drop_duplicates("unique_id")
    scales = naive_scale(inp.train, season_length=1)
    for segment_col in SEGMENT_COLS:
        if segment_col not in statics.columns:
            continue
        df = inp.cv_df.merge(statics[["unique_id", segment_col]], on="unique_id", how="left")
        for segment, group in df.groupby(segment_col, observed=True):
            ids = pd.Index(group["unique_id"].unique())
            common = inp.components.weights.index.intersection(ids).intersection(inp.components.scales.index)
            if len(common) == 0:
                continue
            weights = inp.components.weights.loc[common]
            total = float(weights.sum())
            if total <= 0:
                continue
            components = WRMSSEComponents(weights=weights / total, scales=inp.components.scales.loc[common])
            truth = group[["unique_id", "ds", "y"]]
            for model in inp.models:
                metrics = weighted_metrics(truth, group, components, model, scales=scales)
                rows.append(
                    {
                        "model": model,
                        "segment_axis": segment_col,
                        "segment": str(segment),
                        "n_series": len(common),
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def level_components(level_train: pd.DataFrame) -> WRMSSEComponents:
    df = level_train.sort_values(["unique_id", "ds"])
    last_28 = (
        df.groupby("unique_id", observed=True).tail(28).groupby("unique_id", observed=True)["_rev"].sum()
    )
    weights = (
        (last_28 / last_28.sum()).rename("weight") if float(last_28.sum()) > 0 else last_28.rename("weight")
    )
    diffs = df.groupby("unique_id", observed=True)["y"].diff()
    scales = diffs.pow(2).groupby(df["unique_id"], observed=True).mean().rename("scale")
    scales = scales.replace({0.0: np.nan}).dropna()
    common = weights.index.intersection(scales.index)
    return WRMSSEComponents(weights=weights.loc[common], scales=scales.loc[common])


def join_keys(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    parts = [df[col].astype(str).reset_index(drop=True) for col in cols]
    out = parts[0]
    for part in parts[1:]:
        out = out + "/" + part
    return out


def level_scores(inp: ScoringInputs) -> pd.DataFrame:
    needed = {col for level in M5_LEVELS_SPEC for col in level} - {TOTAL_COL}
    statics = inp.statics.drop_duplicates("unique_id")

    cv = inp.cv_df.merge(statics[["unique_id", *sorted(needed)]], on="unique_id", how="left")
    cv[TOTAL_COL] = TOTAL_VALUE

    keep = ["unique_id", "ds", "y", *([c for c in ("sell_price",) if c in inp.train.columns])]
    train = inp.train[keep].merge(statics[["unique_id", *sorted(needed)]], on="unique_id", how="left")
    train[TOTAL_COL] = TOTAL_VALUE
    train["_rev"] = (
        train["y"] * train["sell_price"].fillna(0) if "sell_price" in train.columns else train["y"]
    )

    rows: list[dict[str, Any]] = []
    for idx, level_spec in enumerate(M5_LEVELS_SPEC):
        level_name = "/".join(c for c in level_spec if c != TOTAL_COL) or TOTAL_COL
        agg_train = (
            train.groupby([*level_spec, "ds"], observed=True)
            .agg(y=("y", "sum"), _rev=("_rev", "sum"))
            .reset_index()
        )
        agg_train["unique_id"] = join_keys(agg_train, level_spec)
        components = level_components(agg_train)

        agg_dict: dict[str, tuple[str, str]] = {"y": ("y", "sum")}
        for model in inp.models:
            agg_dict[model] = (model, "sum")
        agg_cv = cv.groupby([*level_spec, "ds"], observed=True).agg(**agg_dict).reset_index()
        agg_cv["unique_id"] = join_keys(agg_cv, level_spec)
        truth = agg_cv[["unique_id", "ds", "y"]]
        scales = naive_scale(agg_train[["unique_id", "ds", "y"]], season_length=1)
        for model in inp.models:
            metrics = weighted_metrics(truth, agg_cv, components, model, scales=scales)
            rows.append(
                {
                    "model": model,
                    "level": level_name,
                    "level_idx": idx,
                    "n_series": len(components.weights),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def cumulative_error(cv_df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    df = cv_df.copy()
    df["h"] = (df["ds"] - df["cutoff"]).dt.days.astype(int)
    rows: list[dict[str, Any]] = []
    for model in models:
        by_h = (
            df.assign(error=df[model] - df["y"], abs_error=(df[model] - df["y"]).abs(), actual=df["y"].abs())
            .groupby("h", observed=True)[["error", "abs_error", "actual"]]
            .sum()
            .sort_index()
            .cumsum()
            .reset_index()
        )
        for row in by_h.to_dict("records"):
            actual = float(row["actual"])
            rows.append(
                {
                    "model": model,
                    "h": int(row["h"]),
                    "cum_error": float(row["error"]),
                    "cum_abs_error": float(row["abs_error"]),
                    "cum_actual": actual,
                    "cum_error_pct": float(row["error"] / max(actual, EPS)),
                }
            )
    return pd.DataFrame(rows)


def sanitize(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [{k: sanitize(v) for k, v in row.items()} for row in df.to_dict("records")]


def choose_baseline(models: list[str]) -> str | None:
    preferred = ["SeasonalNaive", "Naive"]
    for token in preferred:
        matches = [m for m in models if m.endswith(token)]
        if matches:
            return matches[0]
    return models[0] if models else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--long-path", type=Path, default=repo_root() / "data" / "processed" / "long.parquet")
    parser.add_argument("--artifacts-dir", type=Path, default=repo_root() / "artifacts")
    parser.add_argument(
        "--out", type=Path, default=repo_root() / "frontend" / "public" / "data" / "accuracy-dashboard.json"
    )
    parser.add_argument("--models", nargs="*", help="Artifact names without cv_ prefix, e.g. stats lgbm hier")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    long_df = pd.read_parquet(args.long_path)
    long_df["ds"] = pd.to_datetime(long_df["ds"])
    cv_df, models, files = load_cv_files(args.artifacts_dir, args.models)
    train = long_df[long_df["ds"] < cv_df["ds"].min()].copy()
    if train.empty:
        raise ValueError("Training slice before first CV date is empty.")

    inp = ScoringInputs(
        cv_df=cv_df,
        train=train,
        statics=statics_from_long(long_df),
        components=compute_components(train),
        models=models,
    )
    baseline = choose_baseline(models)
    fva = fva_scores(inp, baseline=baseline, metric="mae") if baseline else pd.DataFrame()

    payload = {
        "generated_at": datetime.now(UTC).strftime("%B %d, %Y at %H:%M UTC"),
        "source": {
            "mode": "generated",
            "cv_files": files,
            "n_series": int(cv_df["unique_id"].nunique()),
            "n_rows": len(cv_df),
            "cutoffs": [ts.isoformat() for ts in sorted(pd.to_datetime(cv_df["cutoff"]).unique())],
        },
        "headline": records(headline(inp)),
        "levels": records(level_scores(inp)),
        "segments": records(segment_scores(inp)),
        "horizon": records(per_horizon_scores(inp)),
        "folds": records(per_fold_scores(inp)),
        "cumulative_error": records(cumulative_error(cv_df, models)),
        "fva": records(fva),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} with {len(models)} model columns from {len(files)} CV files.")


if __name__ == "__main__":
    main()
