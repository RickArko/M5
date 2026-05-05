"""Reproducible cross-validation using Nixtla's rolling-origin CV.

Both ``StatsForecast`` and ``MLForecast`` expose ``.cross_validation(...)``
with the same semantics: walk forward in steps of size ``step_size`` for
``n_windows`` windows of length ``h``. We always seed first.
"""

from __future__ import annotations

import pandas as pd

from m5.config import SETTINGS, set_global_seed
from m5.hierarchy import build_hierarchy, extract_bottom
from m5.logging import logger
from m5.models.hierarchical import build_hier_base_forecaster, build_hier_reconcilers
from m5.models.lgbm import build_lgbm_forecaster, encode_static_categoricals
from m5.models.stats import build_stats_forecaster


def stats_cv(
    df: pd.DataFrame,
    *,
    h: int = SETTINGS.horizon,
    n_windows: int = SETTINGS.n_windows,
    step_size: int | None = None,
    season_length: int = 7,
) -> pd.DataFrame:
    """Run rolling-origin CV with the statistical model bundle."""
    set_global_seed()
    sf = build_stats_forecaster(season_length=season_length)
    logger.info(f"stats_cv: h={h} n_windows={n_windows} step={step_size or h}")
    return sf.cross_validation(
        df=df[["unique_id", "ds", "y"]],
        h=h,
        n_windows=n_windows,
        step_size=step_size or h,
    )


def lgbm_cv(
    df: pd.DataFrame,
    *,
    h: int = SETTINGS.horizon,
    n_windows: int = SETTINGS.n_windows,
    step_size: int | None = None,
    static_cols: tuple[str, ...] = ("item_id", "dept_id", "cat_id", "store_id", "state_id"),
) -> pd.DataFrame:
    """Run rolling-origin CV with the LightGBM global model."""
    set_global_seed()
    fcst = build_lgbm_forecaster()
    statics_present = [c for c in static_cols if c in df.columns]
    df = encode_static_categoricals(df, statics_present)
    logger.info(f"lgbm_cv: h={h} n_windows={n_windows} step={step_size or h}")
    keep = ["unique_id", "ds", "y", *statics_present]
    keep += [c for c in ("snap", "is_event", "price_norm", "price_change_pct") if c in df.columns]
    return fcst.cross_validation(
        df=df[keep],
        h=h,
        n_windows=n_windows,
        step_size=step_size or h,
        static_features=statics_present,
    )


def hier_cv(
    df: pd.DataFrame,
    *,
    h: int = SETTINGS.horizon,
    n_windows: int = SETTINGS.n_windows,
    step_size: int | None = None,
    season_length: int = 7,
    bottom_only: bool = True,
) -> pd.DataFrame:
    """Rolling-origin CV with the hierarchical pipeline.

    Aggregates to all 12 M5 levels, runs Theta cross-validation at every
    level, then reconciles each cutoff with BottomUp / TopDown / MinTrace
    (OLS + shrinkage). With ``bottom_only=True`` (default) the result is
    sliced back to item × store and ids restored, so ``wrmsse_for_models``
    consumes it directly alongside ``stats_cv`` and ``lgbm_cv`` outputs.
    """
    from hierarchicalforecast.core import HierarchicalReconciliation

    set_global_seed()
    hier = build_hierarchy(df)
    sf = build_hier_base_forecaster(season_length=season_length)
    logger.info(
        f"hier_cv: h={h} n_windows={n_windows} step={step_size or h} "
        f"levels={len(hier.tags)} series={hier.Y_df['unique_id'].nunique()}"
    )
    cv_df = sf.cross_validation(
        df=hier.Y_df[["unique_id", "ds", "y"]],
        h=h,
        n_windows=n_windows,
        step_size=step_size or h,
        fitted=True,
    )
    fitted = sf.cross_validation_fitted_values()

    # reconcile() rejects non-numeric forecast columns and assumes Y_df is
    # uniquely keyed on (unique_id, ds). cross_validation_fitted_values
    # repeats earlier dates across cutoffs, so dedupe to the latest cutoff.
    truth = cv_df[["unique_id", "ds", "cutoff", "y"]]
    Y_hat = cv_df.drop(columns=["cutoff", "y"])
    fitted = (
        fitted.sort_values(["unique_id", "ds", "cutoff"])
        .drop_duplicates(subset=["unique_id", "ds"], keep="last")
        .drop(columns=["cutoff"])
    )

    hrec = HierarchicalReconciliation(reconcilers=build_hier_reconcilers())
    reconciled = hrec.reconcile(
        Y_hat_df=Y_hat,
        Y_df=fitted,
        S_df=hier.S_df,
        tags=hier.tags,
    )
    reconciled = reconciled.merge(truth, on=["unique_id", "ds"], how="left")

    if bottom_only:
        return extract_bottom(reconciled, hier)
    return reconciled
