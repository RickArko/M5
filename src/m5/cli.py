"""Typer CLI: ``m5 download | prep | cv | forecast | score``."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import typer

from m5.config import SETTINGS, set_global_seed
from m5.logging import logger

app = typer.Typer(add_completion=False, help="M5 forecasting toolkit.")


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
    model: str = typer.Argument("stats", help="One of: stats, lgbm."),
    horizon: int = typer.Option(SETTINGS.horizon),
    n_windows: int = typer.Option(SETTINGS.n_windows),
    long_path: Path = typer.Option(None, help="Path to processed long parquet."),
) -> None:
    """Run reproducible rolling-origin cross-validation."""
    from m5.cv import lgbm_cv, stats_cv
    from m5.evaluation import compute_components, wrmsse_for_models

    long_path = long_path or SETTINGS.processed_dir / "long.parquet"
    df = pd.read_parquet(long_path)

    runner = {"stats": stats_cv, "lgbm": lgbm_cv}.get(model)
    if runner is None:
        raise typer.BadParameter(f"Unknown model: {model!r}. Use 'stats' or 'lgbm'.")

    cv_df = runner(df, h=horizon, n_windows=n_windows)
    components = compute_components(df[df["ds"] < cv_df["ds"].min()])
    truth = cv_df.rename(columns={"y": "y"})[["unique_id", "ds", "y"]]
    scores = wrmsse_for_models(truth, cv_df, components)
    logger.info(f"WRMSSE by model:\n{scores.to_string()}")

    out = SETTINGS.artifacts_dir / f"cv_{model}.parquet"
    cv_df.to_parquet(out, index=False)
    logger.info(f"Wrote {out}")


@app.command()
def forecast(
    model: str = typer.Argument("stats"),
    horizon: int = typer.Option(SETTINGS.horizon),
    long_path: Path = typer.Option(None),
) -> None:
    """Train on all available data and emit a future forecast."""
    from m5.models.lgbm import fit_predict_lgbm
    from m5.models.stats import fit_predict_stats

    long_path = long_path or SETTINGS.processed_dir / "long.parquet"
    df = pd.read_parquet(long_path)

    runner = {"stats": fit_predict_stats, "lgbm": fit_predict_lgbm}.get(model)
    if runner is None:
        raise typer.BadParameter(f"Unknown model: {model!r}.")
    out_df = runner(df, horizon=horizon)

    out = SETTINGS.forecasts_dir / f"forecast_{model}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False)
    logger.info(f"Wrote {out} ({len(out_df):,d} rows).")


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
