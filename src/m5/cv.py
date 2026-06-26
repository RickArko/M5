"""Reproducible cross-validation using Nixtla's rolling-origin CV.

Both ``StatsForecast`` and ``MLForecast`` expose ``.cross_validation(...)``
with the same semantics: walk forward in steps of size ``step_size`` for
``n_windows`` windows of length ``h``. We always seed first.

Two entry points:

* ``stats_cv`` / ``lgbm_cv`` / ``hier_cv`` — back-compat per-model runners.
* :func:`cv_from_recipe` — recipe-driven dispatcher; static and dynamic
  feature columns come from :class:`~m5.recipes.TaskRecipe`, so adding a new
  task is "drop a YAML."
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from m5.config import SETTINGS, set_global_seed
from m5.hierarchy import build_hierarchy, extract_bottom
from m5.logging import logger
from m5.models.hierarchical import build_hier_base_forecaster, build_hier_reconcilers
from m5.models.lgbm import build_lgbm_forecaster, encode_static_categoricals
from m5.models.stats import build_stats_forecaster
from m5.recipes import (
    HierRecipe,
    LGBMRecipe,
    Recipe,
    StatsRecipe,
    build_lgbm_from_recipe,
    build_stats_from_recipe,
)


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
    dynamic_candidates = [
        "snap",
        "is_event",
        "price_norm",
        "price_change_pct",
        "days_to_next_event",
        "days_since_last_event",
        "week_of_month",
        "price_mean",
        "price_min",
        "price_max",
        "price_rank_in_store",
        "days_since_release",
        "cat_id_mean_dow",
        "dept_id_mean_dow",
        "store_id_mean_dow",
        "state_id_mean_dow",
        "store_cat_mean_dow",
    ]
    keep += [c for c in dynamic_candidates if c in df.columns]
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
    level, then reconciles each cutoff with grouped-compatible BottomUp /
    MinTrace methods. With ``bottom_only=True`` (default) the result is
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


def toto_cv(
    df: pd.DataFrame,
    *,
    h: int = SETTINGS.horizon,
    n_windows: int = SETTINGS.n_windows,
    step_size: int | None = None,
    model_name: str = "Datadog/Toto-2.0-22m",
    context_length: int = 512,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Rolling-origin CV with the TOTO zero-shot foundation model.

    Because TOTO is a pre-trained foundation model (no training step), each
    CV window slices the trailing *context_length* days as input and forecasts
    forward *h* steps.  Results follow the same Nixtla CV format as
    :func:`stats_cv` / :func:`lgbm_cv`.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.  Defaults to the 22M-parameter variant.
    context_length:
        Lookback window (days) fed to the model per series.
    batch_size:
        Number of series per inference batch.  Reduce on GPU OOM.
    """
    from m5.models.toto import toto_cv as _toto_cv

    set_global_seed()
    logger.info(
        f"toto_cv: h={h} n_windows={n_windows} step={step_size or h} "
        f"model={model_name} context={context_length}"
    )
    return _toto_cv(
        df,
        h=h,
        n_windows=n_windows,
        step_size=step_size,
        model_name=model_name,
        context_length=context_length,
        batch_size=batch_size,
    )


# ----------------------------------------------------------------------
# Recipe-driven dispatcher (Phase 3)
# ----------------------------------------------------------------------
def cv_from_recipe(
    recipe: Recipe,
    df: pd.DataFrame,
    *,
    h: int | None = None,
    n_windows: int | None = None,
    step_size: int | None = None,
) -> pd.DataFrame:
    """Run rolling-origin CV using a Recipe as the source of truth.

    Static and dynamic feature columns come from ``recipe.task.static_cols``
    and ``recipe.task.dynamic_cols`` (LGBM only — stats and hier are
    univariate). Knobs not specified here fall back to the recipe's CV block,
    then to :data:`SETTINGS`.

    A new task is added by dropping ``configs/<task>/<model>.yaml`` and
    calling this with the loaded recipe.
    """
    set_global_seed()

    h_eff = h if h is not None else recipe.task.horizon
    nw_eff = n_windows if n_windows is not None else recipe.cv.n_windows
    ss_eff = step_size if step_size is not None else (recipe.cv.step_size or h_eff)

    id_col = recipe.task.id_col
    time_col = recipe.task.time_col
    target_col = recipe.task.target_col
    base_cols = [id_col, time_col, target_col]

    if isinstance(recipe.model, LGBMRecipe):
        fcst = build_lgbm_from_recipe(recipe)
        statics_present = [c for c in recipe.task.static_cols if c in df.columns]
        df = encode_static_categoricals(df, statics_present)
        dynamics_present = [c for c in recipe.task.dynamic_cols if c in df.columns]
        keep = base_cols + statics_present + dynamics_present
        logger.info(
            f"cv_from_recipe[lgbm]: h={h_eff} n_windows={nw_eff} "
            f"statics={len(statics_present)} dynamics={len(dynamics_present)}"
        )
        return fcst.cross_validation(
            df=df[keep],
            h=h_eff,
            n_windows=nw_eff,
            step_size=ss_eff,
            static_features=statics_present,
        )

    if isinstance(recipe.model, StatsRecipe):
        sf = build_stats_from_recipe(recipe)
        logger.info(f"cv_from_recipe[stats]: h={h_eff} n_windows={nw_eff} models={len(recipe.model.models)}")
        return sf.cross_validation(df=df[base_cols], h=h_eff, n_windows=nw_eff, step_size=ss_eff)

    if isinstance(recipe.model, HierRecipe):
        # Hier delegates to the existing hier_cv, which already pulls its base
        # learner + reconciler list from the YAML via the rerouted builders.
        season = getattr(recipe.model.base_model, "season_length", 7)
        return hier_cv(df, h=h_eff, n_windows=nw_eff, step_size=ss_eff, season_length=season)

    raise ValueError(f"cv_from_recipe: unsupported recipe.model.kind={recipe.model.kind!r}")


def bayesian_cv(
    df: pd.DataFrame,
    *,
    h: int = SETTINGS.horizon,
    n_windows: int = SETTINGS.n_windows,
    step_size: int | None = None,
    n_series: int = 20,
    series_ids: list[str] | None = None,
    draws: int = 300,
    tune: int = 300,
    chains: int = 2,
    quiet: bool = True,
    likelihood: Literal["negbin", "zinb"] = "negbin",
) -> pd.DataFrame:
    """Rolling-origin CV with the Bayesian count GLM (optional ``bayesian`` group)."""
    from m5.models.bayesian import bayesian_cv as _bayesian_cv

    return _bayesian_cv(
        df,
        h=h,
        n_windows=n_windows,
        step_size=step_size,
        n_series=n_series,
        series_ids=series_ids,
        draws=draws,
        tune=tune,
        chains=chains,
        quiet=quiet,
        likelihood=likelihood,
    )


def bayesian_routed_cv(
    df: pd.DataFrame,
    *,
    h: int = SETTINGS.horizon,
    n_windows: int = SETTINGS.n_windows,
    step_size: int | None = None,
    n_series: int = 20,
    series_ids: list[str] | None = None,
    pool_col: str = "dept_id",
    min_hier_series: int = 2,
    draws: int = 300,
    tune: int = 300,
    chains: int = 2,
    quiet: bool = True,
) -> pd.DataFrame:
    """ADI/CV²-routed Bayesian CV (optional ``bayesian`` group)."""
    from m5.models.bayesian import bayesian_routed_cv as _bayesian_routed_cv

    return _bayesian_routed_cv(
        df,
        h=h,
        n_windows=n_windows,
        step_size=step_size,
        n_series=n_series,
        series_ids=series_ids,
        pool_col=pool_col,
        min_hier_series=min_hier_series,
        draws=draws,
        tune=tune,
        chains=chains,
        quiet=quiet,
    )
