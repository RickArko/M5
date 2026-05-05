"""Hierarchical forecasting via Nixtla ``hierarchicalforecast``.

Pipeline: aggregate the bottom-level long frame to all 12 M5 levels, fit a
statistical base model at every level, then reconcile via four standard
methods (BottomUp, TopDown forecast-proportions, MinTrace OLS, MinTrace
shrinkage). Each reconciler appears as its own column in the output, mirroring
how :mod:`m5.models.stats` returns Theta / AutoETS / SeasonalNaive side-by-side.

The base model is `statsforecast` with Theta — fast, deterministic, and a
strong M5 baseline. LightGBM-as-base would need per-level feature handling
(price/snap don't aggregate) and is left for a future extension.
"""

from __future__ import annotations

import pandas as pd
from hierarchicalforecast.core import HierarchicalReconciliation
from statsforecast import StatsForecast

from m5.config import SETTINGS
from m5.hierarchy import build_hierarchy, extract_bottom
from m5.recipes import (
    HIER_RECIPE_PATH,
    Recipe,
    build_hier_base_from_recipe,
    build_hier_reconcilers_from_recipe,
)

DEFAULT_FREQ = "D"
DEFAULT_SEASON = 7
BASE_MODEL_NAME = "Theta"


def _load_hier_recipe() -> Recipe:
    return Recipe.from_yaml(HIER_RECIPE_PATH)


def build_hier_reconcilers() -> list:
    """Default reconciler bundle from ``configs/m5/hier.yaml``: BU + TD + MinT(OLS) + MinT(shrink)."""
    return build_hier_reconcilers_from_recipe(_load_hier_recipe())


def build_hier_base_forecaster(
    *,
    season_length: int = DEFAULT_SEASON,
    freq: str = DEFAULT_FREQ,
    n_jobs: int = -1,
) -> StatsForecast:
    """Theta base learner used at every aggregation level. Loaded from configs/m5/hier.yaml."""
    recipe = _load_hier_recipe()
    assert recipe.model.kind == "hier"
    new_base = recipe.model.base_model.model_copy(update={"season_length": season_length})
    new_model = recipe.model.model_copy(update={"base_model": new_base})
    new_task = recipe.task.model_copy(update={"freq": freq})
    recipe = recipe.model_copy(update={"task": new_task, "model": new_model})
    return build_hier_base_from_recipe(recipe, n_jobs=n_jobs)


def fit_predict_hier(
    df: pd.DataFrame,
    *,
    horizon: int = SETTINGS.horizon,
    season_length: int = DEFAULT_SEASON,
    bottom_only: bool = True,
) -> pd.DataFrame:
    """Aggregate, fit Theta at every level, reconcile, return forecasts.

    Args:
        df: Bottom-level long frame (``unique_id, ds, y`` + static cols).
        horizon: Forecast horizon in days.
        season_length: Weekly seasonality on daily retail data.
        bottom_only: If True, slice the reconciled frame to the bottom level
            and remap unique_ids back to ``{item_id}_{store_id}`` so the
            existing :func:`m5.evaluation.wrmsse_for_models` can score it.

    Returns:
        Long frame with one row per (unique_id, ds) and one column per
        reconciler (``Theta/BottomUp``, ``Theta/TopDown_forecast_proportions``,
        ``Theta/MinTrace_ols``, ``Theta/MinTrace_mint_shrink``). The base
        ``Theta`` column is also retained for diagnostics.
    """
    hier = build_hierarchy(df)
    sf = build_hier_base_forecaster(season_length=season_length)
    Y_hat = sf.forecast(
        df=hier.Y_df[["unique_id", "ds", "y"]],
        h=horizon,
        fitted=True,
    )
    Y_fitted = sf.forecast_fitted_values()

    hrec = HierarchicalReconciliation(reconcilers=build_hier_reconcilers())
    reconciled = hrec.reconcile(
        Y_hat_df=Y_hat,
        Y_df=Y_fitted,
        S_df=hier.S_df,
        tags=hier.tags,
    )

    if bottom_only:
        return extract_bottom(reconciled, hier)
    return reconciled
