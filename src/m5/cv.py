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
