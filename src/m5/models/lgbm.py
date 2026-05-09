"""LightGBM global model via Nixtla ``mlforecast``.

We keep the feature menu deliberately small: lags 7/14/28 + 7-day rolling mean,
date features, snap flag, single event flag, normalised price. No mega-blender.

The canonical configuration lives in ``configs/m5/lgbm.yaml`` and is loaded
through :mod:`m5.recipes`. The functions below are thin back-compat wrappers
so existing callers (CLI, notebooks, tests) keep working unchanged.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
from mlforecast import MLForecast

from m5.config import SETTINGS
from m5.features import build_feature_frame
from m5.logging import logger
from m5.recipes import (
    LGBM_RECIPE_PATH,
    LagSpec,
    Recipe,
    build_lgbm_from_recipe,
)


def _load_lgbm_recipe() -> Recipe:
    return Recipe.from_yaml(LGBM_RECIPE_PATH)


def _lgbm_recipe_default_lags() -> tuple[int, ...]:
    r = _load_lgbm_recipe()
    assert r.model.kind == "lgbm"
    return tuple(r.model.lags.lags)


def _lgbm_recipe_default_rolls() -> tuple[int, ...]:
    r = _load_lgbm_recipe()
    assert r.model.kind == "lgbm"
    return tuple(r.model.lags.rolling_means_lagged.get(1, []))


# Module-level constants kept for back-compat; sourced from the YAML at import time
# so a recipe edit is a single-place change.
DEFAULT_LAGS: tuple[int, ...] = _lgbm_recipe_default_lags()
DEFAULT_ROLLS: tuple[int, ...] = _lgbm_recipe_default_rolls()


def encode_static_categoricals(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """LightGBM rejects ``object`` dtype — coerce static id columns to ``category``."""
    out = df.copy()
    for c in cols:
        if c in out.columns and out[c].dtype == "object":
            out[c] = out[c].astype("category")
    return out


def lgbm_params(seed: int = SETTINGS.seed) -> dict[str, Any]:
    """Canonical LightGBM hyperparams (Tweedie). Loaded from configs/m5/lgbm.yaml."""
    recipe = _load_lgbm_recipe()
    assert recipe.model.kind == "lgbm"
    return {**recipe.model.params, "seed": seed}


def build_lgbm_forecaster(
    *,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    rolling_windows: tuple[int, ...] = DEFAULT_ROLLS,
    freq: str = "D",
    n_jobs: int = -1,
    seed: int = SETTINGS.seed,
) -> MLForecast:
    """Construct an MLForecast with LightGBM and minimal date features.

    Defaults come from ``configs/m5/lgbm.yaml``. Pass kwargs to override per-call.
    """
    recipe = _load_lgbm_recipe()
    assert recipe.model.kind == "lgbm"
    new_lag_spec = LagSpec(
        lags=list(lags),
        rolling_means_lagged={1: list(rolling_windows)},
        differences=list(recipe.model.lags.differences),
    )
    new_model = recipe.model.model_copy(update={"lags": new_lag_spec})
    new_task = recipe.task.model_copy(update={"freq": freq})
    recipe = recipe.model_copy(update={"task": new_task, "model": new_model})
    return build_lgbm_from_recipe(recipe, seed=seed, n_jobs=n_jobs)


def fit_lgbm(
    df: pd.DataFrame,
    *,
    static_cols: tuple[str, ...] = ("item_id", "dept_id", "cat_id", "store_id", "state_id"),
    use_dynamic_features: bool = False,
) -> MLForecast:
    """Fit LightGBM on a long-frame and return the fitted MLForecast.

    Mirrors the train half of :func:`fit_predict_lgbm` without predicting —
    use this when you need to persist the trained model (e.g. ``m5 train``
    for the FastAPI service) instead of consuming predictions inline.

    The returned MLForecast carries enough state (training tail per series +
    LightGBM Booster) for ``.predict(h)`` and ``.predict(h, new_df=...)``
    to work directly against it.
    """
    n_rows, n_series = len(df), df["unique_id"].nunique()
    logger.info(f"fit_lgbm: features → {n_series:,d} series × {n_rows // max(n_series, 1):,d} rows each")
    t0 = time.time()
    df = build_feature_frame(df.copy())
    statics_present = [c for c in static_cols if c in df.columns]
    df = encode_static_categoricals(df, statics_present)

    dynamic_cols: list[str] = []
    if use_dynamic_features:
        dynamic_cols = [c for c in ("snap", "is_event", "price_norm", "price_change_pct") if c in df.columns]

    keep_cols = ["unique_id", "ds", "y", *statics_present, *dynamic_cols]
    fcst = build_lgbm_forecaster()
    logger.info(f"fit_lgbm: fitting LightGBM (dynamic={len(dynamic_cols)} cols)")
    fcst.fit(df[keep_cols], static_features=statics_present)
    logger.info(f"fit_lgbm: fit done in {time.time() - t0:.1f}s")
    return fcst


def fit_predict_lgbm(
    df: pd.DataFrame,
    *,
    horizon: int = SETTINGS.horizon,
    static_cols: tuple[str, ...] = ("item_id", "dept_id", "cat_id", "store_id", "state_id"),
    X_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """End-to-end fit+predict.

    If ``X_df`` (future calendar + prices) is provided, dynamic exogenous
    features are used; otherwise the model is trained on lags + date features
    + static categoricals only, so ``.predict(h)`` works without a future frame.
    """
    n_rows, n_series = len(df), df["unique_id"].nunique()
    logger.info(
        f"fit_predict_lgbm: features → {n_series:,d} series × {n_rows // max(n_series, 1):,d} rows each"
    )
    t0 = time.time()
    fcst = fit_lgbm(df, static_cols=static_cols, use_dynamic_features=X_df is not None)
    logger.info(f"fit_predict_lgbm: predicting (h={horizon})")
    out = fcst.predict(h=horizon, X_df=X_df) if X_df is not None else fcst.predict(h=horizon)
    logger.info(f"fit_predict_lgbm: total {time.time() - t0:.1f}s, {len(out):,d} forecast rows")
    return out
