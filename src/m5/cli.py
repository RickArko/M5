"""Typer CLI: ``m5 download | prep | cv | forecast | score``."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import typer

from m5.config import SETTINGS, set_global_seed
from m5.logging import logger

app = typer.Typer(add_completion=False, help="M5 forecasting toolkit.")


_CV_KEY_COLS = ("unique_id", "ds", "cutoff", "y")


def _load_cv_files(model_names: list[str], artifacts_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Read ``cv_<m>.parquet`` for each name and merge on (id, ds, cutoff).

    Returns the merged wide frame plus the list of forecast columns recovered
    across all inputs (the column-level "models" the report will score).
    """
    if not model_names:
        raise typer.BadParameter("Pass at least one --model.")
    frames: list[tuple[str, pd.DataFrame]] = []
    for m in model_names:
        path = artifacts_dir / f"cv_{m}.parquet"
        if not path.exists():
            raise typer.BadParameter(f"CV artifact not found: {path}")
        df = pd.read_parquet(path)
        df["ds"] = pd.to_datetime(df["ds"])
        df["cutoff"] = pd.to_datetime(df["cutoff"])
        frames.append((m, df))

    base = frames[0][1][list(_CV_KEY_COLS)].copy()
    forecast_cols: list[str] = []
    for _, df in frames:
        for c in df.columns:
            if c in _CV_KEY_COLS or c in forecast_cols:
                continue
            forecast_cols.append(c)
    merged = base
    for _, df in frames:
        cols = [c for c in df.columns if c not in _CV_KEY_COLS]
        merged = merged.merge(
            df[["unique_id", "ds", "cutoff", *cols]],
            on=["unique_id", "ds", "cutoff"],
            how="inner",
        )
    if merged.empty:
        raise typer.BadParameter(
            "Merged CV frame is empty — the CV files don't share (unique_id, ds, cutoff) keys."
        )
    return merged, forecast_cols


@app.command()
def download() -> None:
    """Download the M5 raw dataset via ``datasetsforecast``."""
    from datasetsforecast.m5 import M5

    SETTINGS.ensure_dirs()
    logger.info(f"Downloading M5 → {SETTINGS.data_dir}")
    M5.load(directory=str(SETTINGS.data_dir))
    logger.info("Done.")


@app.command()
def prep(
    last_n_days: int = typer.Option(SETTINGS.last_n_days, help="Trailing window of training data."),
    n_series: int = typer.Option(SETTINGS.n_series, help="Subsample N series (-1 = all)."),
    out: Path = typer.Option(None, help="Output parquet path (default: data/processed/long.parquet)."),
) -> None:
    """Build the long-format training frame and write it to parquet."""
    from m5.data import build_long_frame, load_calendar, load_prices, load_sales, reduce_mem_usage

    set_global_seed()
    SETTINGS.ensure_dirs()
    t0 = time.time()
    cal = load_calendar(SETTINGS.raw_dir)
    prices = load_prices(SETTINGS.raw_dir)
    sales = load_sales(SETTINGS.raw_dir, prices)
    long = build_long_frame(
        sales,
        cal,
        prices,
        last_n_days=last_n_days,
        n_series=n_series if n_series > 0 else None,
    )
    long = reduce_mem_usage(long)

    out_path = out or (SETTINGS.processed_dir / "long.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    long.to_parquet(out_path, index=False)
    logger.info(f"Wrote {out_path} ({len(long):,d} rows) in {time.time() - t0:.1f}s.")


@app.command()
def cv(
    model: str = typer.Argument("stats", help="One of: stats, lgbm, hier."),
    horizon: int = typer.Option(SETTINGS.horizon),
    n_windows: int = typer.Option(SETTINGS.n_windows),
    long_path: Path = typer.Option(None, help="Path to processed long parquet."),
) -> None:
    """Run reproducible rolling-origin cross-validation."""
    from m5.cv import hier_cv, lgbm_cv, stats_cv
    from m5.evaluation import compute_components, wrmsse_for_models

    long_path = long_path or SETTINGS.processed_dir / "long.parquet"
    df = pd.read_parquet(long_path)

    if model == "stats":
        cv_df = stats_cv(df, h=horizon, n_windows=n_windows)
    elif model == "lgbm":
        cv_df = lgbm_cv(df, h=horizon, n_windows=n_windows)
    elif model == "hier":
        cv_df = hier_cv(df, h=horizon, n_windows=n_windows)
    else:
        raise typer.BadParameter(f"Unknown model: {model!r}. Use 'stats', 'lgbm', or 'hier'.")
    components = compute_components(df[df["ds"] < cv_df["ds"].min()])
    truth = cv_df.rename(columns={"y": "y"})[["unique_id", "ds", "y"]]
    scores = wrmsse_for_models(truth, cv_df, components)
    logger.info(f"WRMSSE by model:\n{scores.to_string()}")

    out = SETTINGS.artifacts_dir / f"cv_{model}.parquet"
    cv_df.to_parquet(out, index=False)
    logger.info(f"Wrote {out}")


@app.command("cv-recipe")
def cv_recipe(
    recipe_path: Path = typer.Argument(..., help="Path to a YAML recipe (e.g. configs/m5/lgbm.yaml)."),
    horizon: int | None = typer.Option(None, help="Override recipe.task.horizon."),
    n_windows: int | None = typer.Option(None, help="Override recipe.cv.n_windows."),
    long_path: Path = typer.Option(None, help="Path to processed long parquet."),
) -> None:
    """Recipe-driven CV — loads YAML and dispatches on ``model.kind``.

    Adding a new task: ``configs/<task>/<model>.yaml`` + a per-task data prep
    that produces a long-frame parquet with the columns named in ``task.*``.
    """
    from m5.cv import cv_from_recipe
    from m5.evaluation import compute_components, wrmsse_for_models
    from m5.recipes import Recipe

    recipe = Recipe.from_yaml(recipe_path)
    long_path = long_path or SETTINGS.processed_dir / "long.parquet"
    logger.info(f"cv-recipe[{recipe_path.name}]: loading {long_path}")
    df = pd.read_parquet(long_path)

    cv_df = cv_from_recipe(recipe, df, h=horizon, n_windows=n_windows)
    components = compute_components(df[df["ds"] < cv_df["ds"].min()])
    truth = cv_df[["unique_id", "ds", "y"]]
    scores = wrmsse_for_models(truth, cv_df, components)
    logger.info(f"WRMSSE by model:\n{scores.to_string()}")

    out = SETTINGS.artifacts_dir / f"cv_{recipe_path.stem}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv_df.to_parquet(out, index=False)
    logger.info(f"Wrote {out}")


@app.command()
def forecast(
    model: str = typer.Argument("stats", help="One of: stats, lgbm, hier."),
    horizon: int = typer.Option(SETTINGS.horizon),
    long_path: Path = typer.Option(None),
) -> None:
    """Train on all available data and emit a future forecast."""
    from m5.models.hierarchical import fit_predict_hier
    from m5.models.lgbm import fit_predict_lgbm
    from m5.models.stats import fit_predict_stats

    long_path = long_path or SETTINGS.processed_dir / "long.parquet"
    logger.info(f"forecast {model}: loading {long_path}")
    df = pd.read_parquet(long_path)
    logger.info(f"forecast {model}: loaded {len(df):,d} rows, {df['unique_id'].nunique():,d} series")

    if model == "stats":
        out_df = fit_predict_stats(df, horizon=horizon)
    elif model == "lgbm":
        out_df = fit_predict_lgbm(df, horizon=horizon)
    elif model == "hier":
        out_df = fit_predict_hier(df, horizon=horizon)
    else:
        raise typer.BadParameter(f"Unknown model: {model!r}.")

    out = SETTINGS.forecasts_dir / f"forecast_{model}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False)
    logger.info(f"Wrote {out} ({len(out_df):,d} rows).")


@app.command()
def score(
    models: list[str] = typer.Option(
        ...,
        "--model",
        "-m",
        help="Artifact base name (reads artifacts/cv_<name>.parquet). Repeat for multiple.",
    ),
    out: Path = typer.Option(Path("reports"), help="Output directory for figures + report."),
    long_path: Path = typer.Option(None, help="Path to processed long parquet."),
    run_id: str = typer.Option("latest", help="Run id stamped into the report header."),
    bootstrap_iter: int = typer.Option(1000, help="Bootstrap resamples for significance matrix."),
    formats: str = typer.Option("png,svg,pdf", help="Figure formats to save (comma-separated)."),
    no_report: bool = typer.Option(False, help="Skip markdown/HTML stitching; only write figures + metrics."),
) -> None:
    """Score CV artifacts; emit metrics, figures, and a Markdown + HTML report."""
    from m5.evaluation import compute_components
    from m5.reporting import build_all_figures, render_report, save_figure
    from m5.reporting.report import RunMetadata
    from m5.scoring import (
        ScoringInputs,
        bias_variance_decomposition,
        error_concentration,
        headline_scores,
        paired_bootstrap_pvalues,
        per_fold_scores,
        per_horizon_scores,
        per_level_scores,
        per_segment_scores,
        residuals_long,
    )

    set_global_seed()
    long_path = long_path or SETTINGS.processed_dir / "long.parquet"
    if not long_path.exists():
        raise typer.BadParameter(f"Training long-frame not found at {long_path}; run `m5 prep` first.")

    logger.info(f"score: loading {long_path}")
    train = pd.read_parquet(long_path)
    train["ds"] = pd.to_datetime(train["ds"])

    cv_df, forecast_cols = _load_cv_files(models, SETTINGS.artifacts_dir)
    logger.info(
        f"score: merged {len(models)} CV file(s) → {len(cv_df):,d} rows × "
        f"{len(forecast_cols)} forecast columns: {forecast_cols}"
    )

    static_cols = [c for c in ("item_id", "dept_id", "cat_id", "store_id", "state_id") if c in train.columns]
    statics = train.drop_duplicates("unique_id")[["unique_id", *static_cols]].reset_index(drop=True)

    train_pre_cv = train[train["ds"] < cv_df["ds"].min()]
    if train_pre_cv.empty:
        raise typer.BadParameter(
            "Training frame has no rows before the first CV cutoff — wrong long.parquet for these CV files?"
        )
    components = compute_components(train_pre_cv)

    inp = ScoringInputs(
        cv_df=cv_df,
        train=train_pre_cv,
        statics=statics,
        components=components,
        models=forecast_cols,
    )

    logger.info("score: computing metrics …")
    headline = headline_scores(inp)
    per_fold = per_fold_scores(inp)
    per_horizon = per_horizon_scores(inp)
    per_level = per_level_scores(inp) if static_cols else pd.DataFrame()
    segment_frames = {
        cut: per_segment_scores(inp, cut)
        for cut in ("cat_id", "dept_id", "store_id", "state_id")
        if cut in static_cols
    }
    bv = bias_variance_decomposition(inp)
    pvalues = (
        paired_bootstrap_pvalues(inp, n_iter=bootstrap_iter, seed=SETTINGS.seed)
        if len(forecast_cols) > 1
        else None
    )
    residuals = residuals_long(inp)
    error_curves = error_concentration(inp)

    out.mkdir(parents=True, exist_ok=True)
    metrics_dir = out / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    headline.to_csv(metrics_dir / "headline.csv", index=False)
    headline.to_parquet(metrics_dir / "headline.parquet", index=False)
    per_fold.to_parquet(metrics_dir / "per_fold.parquet", index=False)
    per_horizon.to_parquet(metrics_dir / "per_horizon.parquet", index=False)
    if not per_level.empty:
        per_level.to_parquet(metrics_dir / "per_level.parquet", index=False)
    for cut, df in segment_frames.items():
        if not df.empty:
            df.to_parquet(metrics_dir / f"per_segment_{cut}.parquet", index=False)
    bv.to_parquet(metrics_dir / "bias_variance.parquet", index=False)
    if pvalues is not None:
        pvalues.to_parquet(metrics_dir / "pvalues.parquet")
    error_curves.to_parquet(metrics_dir / "error_concentration.parquet", index=False)
    logger.info(f"score: wrote metrics → {metrics_dir}")

    logger.info("score: building figures …")
    bundle = build_all_figures(
        headline=headline,
        per_fold=per_fold,
        per_horizon=per_horizon,
        per_level=per_level,
        segment_frames=segment_frames,
        bv=bv,
        pvalues=pvalues,
        residuals=residuals,
        error_curves=error_curves,
        cv_df=cv_df,
        train=train,
        models=forecast_cols,
    )

    fig_dir = out / "figures"
    fmt_tuple = tuple(f.strip() for f in formats.split(",") if f.strip())
    metadata_props = {
        "M5RunId": run_id,
        "M5Seed": str(SETTINGS.seed),
        "M5Models": ",".join(forecast_cols),
    }
    for name, fig in bundle.figures.items():
        save_figure(fig, name, out_dir=fig_dir, formats=fmt_tuple, metadata=metadata_props)
    logger.info(f"score: wrote {len(bundle.figures)} figure(s) → {fig_dir}")

    if no_report:
        logger.info("score: --no-report set; skipping markdown/html.")
        return

    metadata = RunMetadata.autodiscover(
        run_id=run_id,
        seed=SETTINGS.seed,
        horizon=int(per_horizon["h"].max()) if not per_horizon.empty else SETTINGS.horizon,
        n_windows=int(cv_df["cutoff"].nunique()),
        models=forecast_cols,
        n_series=int(cv_df["unique_id"].nunique()),
    )
    extras: list[tuple[str, pd.DataFrame]] = []
    if not per_level.empty:
        extras.append(("Per-level WRMSSE", per_level.drop(columns=["level_idx"], errors="ignore")))
    for cut, df in segment_frames.items():
        if not df.empty:
            extras.append((f"Per-segment WRMSSE ({cut})", df))
    paths = render_report(
        bundle,
        metadata=metadata,
        headline=headline,
        out_dir=out,
        extra_tables=extras,
    )
    logger.info(f"score: wrote {paths['md']} and {paths['html']}")


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
