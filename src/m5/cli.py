"""Typer CLI: ``m5 download | prep | cv | forecast | train | serve | score``."""

from __future__ import annotations

import time
from datetime import UTC
from pathlib import Path

import pandas as pd
import typer

from m5.config import REPO_ROOT, SETTINGS, set_global_seed
from m5.logging import logger

app = typer.Typer(add_completion=False, help="M5 forecasting toolkit.")


_CV_KEY_COLS = ("unique_id", "ds", "cutoff", "y")
_RAW_REQUIRED_FILES = ("calendar.csv", "sell_prices.csv", "sales_train_evaluation.csv")


def _missing_raw_files(raw_dir: Path) -> list[str]:
    """Return required raw CSVs that are absent or empty."""
    missing = []
    for name in _RAW_REQUIRED_FILES:
        path = raw_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(name)
    return missing


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
    merged = base
    for m, df in frames:
        # Rename forecast columns to avoid collisions across CV files.
        # e.g. LGBM from cv_lgbm -> lgbm_LGBM, LGBM from cv_store -> store_LGBM
        rename_map: dict[str, str] = {}
        for c in df.columns:
            if c in _CV_KEY_COLS:
                continue
            new_name = f"{m}_{c}"
            rename_map[c] = new_name
            forecast_cols.append(new_name)
        if rename_map:
            df = df.rename(columns=rename_map)
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
    from datasetsforecast.utils import download_file

    SETTINGS.ensure_dirs()
    missing = _missing_raw_files(SETTINGS.raw_dir)
    if not missing:
        logger.info(f"M5 raw CSVs already present under {SETTINGS.raw_dir}.")
        return

    logger.info(f"Missing M5 raw CSVs under {SETTINGS.raw_dir}: {', '.join(missing)}")
    logger.info(f"Downloading M5 → {SETTINGS.raw_dir}")
    download_file(directory=SETTINGS.raw_dir, source_url=M5.source_url, decompress=True)

    missing = _missing_raw_files(SETTINGS.raw_dir)
    if missing:
        raise RuntimeError(f"M5 download finished but required CSVs are still missing: {', '.join(missing)}")
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
    model: str = typer.Argument(
        "stats", help="One of: stats, lgbm, hier, segmented, store, store_cat, store_dept."
    ),
    horizon: int = typer.Option(SETTINGS.horizon),
    n_windows: int = typer.Option(SETTINGS.n_windows),
    long_path: Path = typer.Option(None, help="Path to processed long parquet."),
) -> None:
    """Run reproducible rolling-origin cross-validation."""
    from m5.cv import hier_cv, lgbm_cv, stats_cv
    from m5.evaluation import compute_components, wrmsse_for_models
    from m5.models.segmented import store_cat_cv, store_cv, store_dept_cv

    long_path = long_path or SETTINGS.processed_dir / "long.parquet"
    df = pd.read_parquet(long_path)

    if model == "stats":
        cv_df = stats_cv(df, h=horizon, n_windows=n_windows)
    elif model == "lgbm":
        cv_df = lgbm_cv(df, h=horizon, n_windows=n_windows)
    elif model == "hier":
        cv_df = hier_cv(df, h=horizon, n_windows=n_windows)
    elif model == "segmented" or model == "store":
        cv_df = store_cv(df, h=horizon, n_windows=n_windows)
    elif model == "store_cat":
        cv_df = store_cat_cv(df, h=horizon, n_windows=n_windows)
    elif model == "store_dept":
        cv_df = store_dept_cv(df, h=horizon, n_windows=n_windows)
    else:
        raise typer.BadParameter(
            f"Unknown model: {model!r}. Use 'stats', 'lgbm', 'hier', 'segmented', 'store', 'store_cat', or 'store_dept'."
        )
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
def train(
    horizon: int = typer.Option(
        SETTINGS.horizon,
        help="Horizon recorded in metadata. Inference can request any h up to M5_SERVE_MAX_HORIZON.",
    ),
    long_path: Path = typer.Option(
        None, help="Path to processed long parquet (default: data/processed/long.parquet)."
    ),
    out_dir: Path = typer.Option(
        None,
        help="Where to write the artifact. Default: artifacts/models/lgbm/<UTC-timestamp>/ "
        "(also updates artifacts/models/lgbm/latest symlink).",
    ),
    history_buffer_days: int = typer.Option(
        120,
        help="History tail (per series) bundled into the artifact for stateful inference.",
    ),
) -> None:
    """Fit the LightGBM model and persist a serving artifact.

    Writes ``model.joblib`` + ``metadata.json`` + ``history.parquet`` + ``statics.parquet``
    into a per-run directory. The FastAPI service (``python -m m5.serve``) loads from
    ``M5_SERVE_MODEL_DIR`` (defaults to the ``latest`` symlink updated by this command).
    """
    import json
    import subprocess
    from datetime import datetime
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    import joblib

    from m5.models.lgbm import DEFAULT_LAGS, DEFAULT_ROLLS, fit_lgbm

    set_global_seed()
    SETTINGS.ensure_dirs()

    long_path = long_path or SETTINGS.processed_dir / "long.parquet"
    if not long_path.exists():
        raise typer.BadParameter(f"Long-frame not found at {long_path} — run `m5 prep` first.")

    logger.info(f"train: loading {long_path}")
    df = pd.read_parquet(long_path)
    df["ds"] = pd.to_datetime(df["ds"])
    n_rows = len(df)
    n_series = int(df["unique_id"].nunique())
    training_cutoff = df["ds"].max()
    logger.info(f"train: {n_rows:,d} rows × {n_series:,d} series, cutoff={training_cutoff.date()}")

    fcst = fit_lgbm(df)

    # Decide artifact directory.
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    base_dir = SETTINGS.artifacts_dir / "models" / "lgbm"
    run_dir = out_dir if out_dir is not None else (base_dir / ts)
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1) Model.
    model_path = run_dir / "model.joblib"
    joblib.dump(fcst, model_path, compress=3)
    logger.info(f"train: wrote {model_path}")

    # 2) Trailing history per series (for stateful inference).
    max_lag = max(DEFAULT_LAGS) if DEFAULT_LAGS else 0
    max_roll = max(DEFAULT_ROLLS) if DEFAULT_ROLLS else 0
    needed_days = max(max_lag + max_roll, history_buffer_days)
    cutoff = training_cutoff - pd.Timedelta(days=needed_days)
    history = (
        df[df["ds"] >= cutoff][["unique_id", "ds", "y"]]
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)
    )
    history.to_parquet(run_dir / "history.parquet", index=False)
    logger.info(f"train: wrote history.parquet ({len(history):,d} rows, last {needed_days} days)")

    # 3) Static features per series.
    static_cols = ["unique_id"] + [
        c for c in ("item_id", "dept_id", "cat_id", "store_id", "state_id") if c in df.columns
    ]
    statics = df.drop_duplicates("unique_id")[static_cols].reset_index(drop=True)
    statics.to_parquet(run_dir / "statics.parquet", index=False)
    logger.info(f"train: wrote statics.parquet ({len(statics):,d} series)")

    # 4) Metadata.
    def _safe_version(pkg: str) -> str:
        try:
            return _pkg_version(pkg)
        except PackageNotFoundError:
            return "unknown"

    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        git_sha = "unknown"

    metadata = {
        "model_kind": "lgbm",
        "framework": "mlforecast",
        "framework_version": _safe_version("mlforecast"),
        "lightgbm_version": _safe_version("lightgbm"),
        "trained_at": ts,
        "git_sha": git_sha,
        "training_cutoff": training_cutoff.strftime("%Y-%m-%d"),
        "freq": "D",
        "horizon_default": int(horizon),
        "lags": list(DEFAULT_LAGS),
        "rolling_windows": list(DEFAULT_ROLLS),
        "n_series": n_series,
        "n_rows": int(n_rows),
        # Lower bound for stateless predict — clients must send at least this many rows
        # per series. mlforecast needs the full lag window to compute features.
        "min_history_required": int(max(max_lag + max_roll - 1, max_lag)),
        "static_features": [c for c in static_cols if c != "unique_id"],
        "seed": SETTINGS.seed,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    logger.info("train: wrote metadata.json")

    # 5) Update `latest` symlink — only when writing to the default location.
    if out_dir is None:
        latest = base_dir / "latest"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(ts)  # relative target — survives if base_dir is moved
        logger.info(f"train: latest -> {ts}")

    logger.info(f"train: artifact ready at {run_dir}")


@app.command()
def serve(
    host: str = typer.Option(None, help="Bind host. Default: M5_SERVE_HOST or 0.0.0.0."),
    port: int = typer.Option(None, help="Bind port. Default: M5_SERVE_PORT or 8000."),
    reload: bool = typer.Option(False, "--reload", help="Hot-reload on code changes (dev only)."),
    workers: int = typer.Option(None, help="Uvicorn worker count. Ignored when --reload is set."),
) -> None:
    """Run the FastAPI prediction service (uvicorn).

    Loads the model artifact pointed to by ``M5_SERVE_MODEL_DIR`` (default:
    ``artifacts/models/lgbm/latest``). Run ``m5 train`` first to produce one.
    """
    import uvicorn

    from m5.serve.config import ServeSettings

    s = ServeSettings()
    uvicorn.run(
        "m5.serve.app:create_app",
        factory=True,
        host=host or s.host,
        port=port or s.port,
        reload=reload,
        workers=(workers or s.workers) if not reload else 1,
        log_config=None,
    )


@app.command()
def forecast(
    model: str = typer.Argument(
        "stats", help="One of: stats, lgbm, hier, segmented, store, store_cat, store_dept."
    ),
    horizon: int = typer.Option(SETTINGS.horizon),
    long_path: Path = typer.Option(None),
) -> None:
    """Train on all available data and emit a future forecast."""
    from m5.models.hierarchical import fit_predict_hier
    from m5.models.lgbm import fit_predict_lgbm
    from m5.models.segmented import fit_predict_store, fit_predict_store_cat, fit_predict_store_dept
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
    elif model in ("segmented", "store"):
        out_df = fit_predict_store(df, horizon=horizon)
    elif model == "store_cat":
        out_df = fit_predict_store_cat(df, horizon=horizon)
    elif model == "store_dept":
        out_df = fit_predict_store_dept(df, horizon=horizon)
    else:
        raise typer.BadParameter(f"Unknown model: {model!r}.")

    out = SETTINGS.forecasts_dir / f"forecast_{model}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False)
    logger.info(f"Wrote {out} ({len(out_df):,d} rows).")


@app.command()
def ensemble(
    models: list[str] = typer.Option(
        ...,
        "--model",
        "-m",
        help="Artifact base name (reads artifacts/cv_<name>.parquet). Repeat for multiple.",
    ),
    weights: list[float] | None = typer.Option(
        None,
        "--weight",
        "-w",
        help="Weight per model. If omitted, equal weights are used. Must match number of models.",
    ),
    out_name: str = typer.Option(
        "ensemble", help="Output artifact base name (writes artifacts/cv_<out_name>.parquet)."
    ),
    artifacts_dir: Path = typer.Option(None, help="Directory containing cv_*.parquet files."),
) -> None:
    """Average multiple CV artifacts into a single ensemble forecast.

    Reads ``cv_<model>.parquet`` for each ``--model``, aligns them on
    ``(unique_id, ds, cutoff)``, computes a weighted average of the forecast
    columns, and writes ``cv_<out_name>.parquet`` with a single
    ``<out_name>`` column. The resulting artifact can be scored alongside
    the individual models via ``m5 score --model <out_name>``.
    """
    ad = artifacts_dir or SETTINGS.artifacts_dir
    merged, forecast_cols = _load_cv_files(models, ad)

    if weights is None:
        weights = [1.0 / len(forecast_cols)] * len(forecast_cols)
    elif len(weights) != len(forecast_cols):
        raise typer.BadParameter(
            f"Number of weights ({len(weights)}) must match number of forecast columns ({len(forecast_cols)})."
        )
    total = sum(weights)
    if total == 0:
        raise typer.BadParameter("Weights must not sum to zero.")
    weights = [w / total for w in weights]

    logger.info(f"ensemble: averaging {len(forecast_cols)} columns with weights {weights}")
    merged[out_name] = sum(merged[col] * w for col, w in zip(forecast_cols, weights, strict=True))

    # Keep only the key columns + the ensemble column
    out_cols = ["unique_id", "ds", "cutoff", "y", out_name]
    out_df = merged[out_cols].copy()
    out_path = ad / f"cv_{out_name}.parquet"
    out_df.to_parquet(out_path, index=False)
    logger.info(f"ensemble: wrote {out_path} ({len(out_df):,d} rows)")


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
    fva_baseline: str = typer.Option(
        "SeasonalNaive",
        "--fva-baseline",
        help="Forecast-column name to use as the FVA benchmark. Skipped if column not present.",
    ),
    fva_metric: str = typer.Option(
        "mae",
        "--fva-metric",
        help="FVA basis: mae | rmse | smape | wrmsse. Vandeput recommends mae.",
    ),
    fva_chain_arg: str = typer.Option(
        "",
        "--fva-chain",
        help="Optional ordered chain (comma-separated forecast columns) for the waterfall figure.",
    ),
) -> None:
    """Score CV artifacts; emit metrics, figures, and a Markdown + HTML report."""
    from m5.evaluation import compute_components
    from m5.reporting import build_all_figures, render_report, save_figure
    from m5.reporting.report import RunMetadata
    from m5.scoring import (
        ScoringInputs,
        bias_variance_decomposition,
        error_concentration,
        fva_chain,
        fva_per_fold,
        fva_scores,
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

    fva_star_df: pd.DataFrame | None = None
    fva_chain_df: pd.DataFrame | None = None
    fva_per_fold_df: pd.DataFrame | None = None
    if fva_baseline and fva_baseline in forecast_cols:
        logger.info(f"score: computing FVA vs {fva_baseline} ({fva_metric.upper()})")
        fva_star_df = fva_scores(inp, baseline=fva_baseline, metric=fva_metric)
        fva_per_fold_df = fva_per_fold(inp, baseline=fva_baseline, metric=fva_metric)
    elif fva_baseline:
        logger.warning(
            f"score: FVA baseline {fva_baseline!r} not in forecast columns {forecast_cols} — skipping FVA."
        )

    if fva_chain_arg:
        chain_steps = [s.strip() for s in fva_chain_arg.split(",") if s.strip()]
        missing = [s for s in chain_steps if s not in forecast_cols]
        if missing:
            logger.warning(f"score: chain steps not found {missing} — skipping waterfall.")
        elif len(chain_steps) >= 2:
            fva_chain_df = fva_chain(inp, chain=chain_steps, metric=fva_metric)

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
    if fva_star_df is not None:
        fva_star_df.to_parquet(metrics_dir / "fva_star.parquet", index=False)
        fva_star_df.to_csv(metrics_dir / "fva_star.csv", index=False)
    if fva_per_fold_df is not None:
        fva_per_fold_df.to_parquet(metrics_dir / "fva_per_fold.parquet", index=False)
    if fva_chain_df is not None:
        fva_chain_df.to_parquet(metrics_dir / "fva_chain.parquet", index=False)
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
        fva_star=fva_star_df,
        fva_chain_df=fva_chain_df,
        fva_per_fold_df=fva_per_fold_df,
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
    if fva_star_df is not None and not fva_star_df.empty:
        extras.append((f"FVA vs {fva_baseline} ({fva_metric.upper()})", fva_star_df))
    if fva_chain_df is not None and not fva_chain_df.empty:
        extras.append(("FVA chain", fva_chain_df.drop(columns=["is_baseline"], errors="ignore")))
    paths = render_report(
        bundle,
        metadata=metadata,
        headline=headline,
        out_dir=out,
        extra_tables=extras,
    )
    logger.info(f"score: wrote {paths['md']} and {paths['html']}")


@app.command()
def viz(
    model_dir: Path = typer.Option(
        None,
        help="Fitted artifact directory (default: artifacts/models/lgbm/latest).",
    ),
    long_path: Path = typer.Option(None, help="Path to the processed long parquet."),
    out_dir: Path = typer.Option(None, help="Output directory (default: assets/)."),
    horizon: int = typer.Option(28, help="Forecast horizon to visualise."),
    n_windows: int = typer.Option(3, help="Rolling-origin CV windows to embed in the HTML."),
    train_context: int = typer.Option(84, help="Trailing training-context days drawn before the cutoff."),
    gif: bool = typer.Option(
        True,
        "--gif/--no-gif",
        help="Also render assets/pipeline.gif (universal renderer support; ~50-100x larger than the SVG).",
    ),
    gif_fps: int = typer.Option(12, help="Frame rate for the GIF."),
    gif_duration: float = typer.Option(12.0, help="GIF loop length, seconds."),
) -> None:
    """Render the M5 pipeline visualisation (animated SVG + interactive D3 HTML + GIF).

    Loads the fitted serving artifact, picks a hero series, runs ``n_windows``
    rolling-origin predictions, and writes:

    - ``pipeline.svg``  — animated SVG (SMIL); plays in GitHub README and
      modern browsers; non-SMIL viewers (VSCode preview etc.) see the static
      final-frame composition.
    - ``pipeline.html`` — standalone D3.js page; scrub through CV windows,
      hover for per-day truth/forecast/baseline values.
    - ``pipeline.gif``  — animated GIF; the universally-rendered fallback.
    """
    from m5.viz import render_pipeline_viz

    set_global_seed()
    md = model_dir or REPO_ROOT / "artifacts" / "models" / "lgbm" / "latest"
    lp = long_path or SETTINGS.processed_dir / "long.parquet"
    od = out_dir or REPO_ROOT / "assets"
    render_pipeline_viz(
        model_dir=md,
        long_path=lp,
        out_dir=od,
        horizon=horizon,
        n_windows=n_windows,
        train_context=train_context,
        gif=gif,
        gif_fps=gif_fps,
        gif_duration=gif_duration,
    )


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
