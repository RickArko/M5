"""Segmented LightGBM models — store, store-category, store-department.

The winning M5 solution (YJ_STU) used 110 separate LightGBM models:
- 10 store-level models
- 30 store-category models
- 70 store-department models

Each model sees only a subset of series, allowing it to learn segment-specific
demand patterns. Predictions are concatenated back into a single long frame.
"""

from __future__ import annotations

import time

import pandas as pd
from mlforecast import MLForecast

from m5.config import SETTINGS
from m5.features import build_feature_frame
from m5.logging import logger
from m5.models.lgbm import build_lgbm_forecaster, encode_static_categoricals, fit_lgbm

# ------------------------------------------------------------------
# Segment definitions
# ------------------------------------------------------------------

SEGMENT_KEYS = {
    "store": ["store_id"],
    "store_cat": ["store_id", "cat_id"],
    "store_dept": ["store_id", "dept_id"],
}


def _segment_name(keys: list[str], values: tuple) -> str:
    return "_".join(str(v) for v in values)


def _iter_segments(df: pd.DataFrame, keys: list[str]):
    """Yield (segment_name, sub_df) for every unique key combination."""
    for values, sub in df.groupby(keys, observed=True):
        if not isinstance(values, tuple):
            values = (values,)
        yield _segment_name(keys, values), sub


# ------------------------------------------------------------------
# Fit / predict per segment
# ------------------------------------------------------------------


def _fit_segment(sub_df: pd.DataFrame, *, segment_name: str) -> MLForecast:
    """Fit a single LightGBM model on a segment."""
    n_series = sub_df["unique_id"].nunique()
    logger.info(f"  segment {segment_name}: {n_series:,d} series, {len(sub_df):,d} rows")
    t0 = time.time()
    fcst = fit_lgbm(sub_df, use_dynamic_features=True)
    logger.info(f"  segment {segment_name}: fit done in {time.time() - t0:.1f}s")
    return fcst


def _predict_segment(
    fcst: MLForecast,
    *,
    horizon: int,
    X_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Predict from a fitted segment model."""
    if X_df is not None:
        return fcst.predict(h=horizon, X_df=X_df)
    return fcst.predict(h=horizon)


# ------------------------------------------------------------------
# Top-level fit + predict
# ------------------------------------------------------------------


def fit_predict_segmented(
    df: pd.DataFrame,
    *,
    horizon: int = SETTINGS.horizon,
    segment_keys: list[str] | None = None,
    X_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Fit segmented LightGBM models and return concatenated forecasts.

    Args:
        df: Bottom-level long frame with ``unique_id, ds, y`` + statics.
        horizon: Forecast horizon.
        segment_keys: Which segmentation to use. If ``None``, defaults to
            ``["store_id"]`` (10 store-level models). Other sensible choices
            are ``["store_id", "cat_id"]`` (30 models) or
            ``["store_id", "dept_id"]`` (70 models).
        X_df: Future dynamic features (calendar / prices) for the forecast
            period. If provided, it must contain the same columns as the
            training frame's dynamic features.

    Returns:
        Long forecast frame with columns ``unique_id, ds, LGBM``.
    """
    if segment_keys is None:
        segment_keys = ["store_id"]

    # Pre-build features once on the full frame (cheaper than per-segment).
    df = build_feature_frame(df.copy())

    # Determine which statics are present
    static_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    statics_present = [c for c in static_cols if c in df.columns]

    # Identify dynamic features that were built
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
    dynamics_present = [c for c in dynamic_candidates if c in df.columns]

    keep_cols = ["unique_id", "ds", "y", *statics_present, *dynamics_present]
    df = encode_static_categoricals(df, statics_present)

    # If X_df is provided, also build features on it and restrict columns
    if X_df is not None:
        X_df = build_feature_frame(X_df.copy())
        X_df = encode_static_categoricals(X_df, statics_present)
        X_df = X_df[["unique_id", "ds", *statics_present, *dynamics_present]]

    logger.info(
        f"fit_predict_segmented: segment_by={segment_keys} h={horizon} "
        f"statics={len(statics_present)} dynamics={len(dynamics_present)}"
    )

    predictions: list[pd.DataFrame] = []
    t0 = time.time()

    for seg_name, sub in _iter_segments(df, segment_keys):
        sub_keep = sub[keep_cols].copy()
        fcst = _fit_segment(sub_keep, segment_name=seg_name)

        # Build X_df for this segment if global X_df was provided
        seg_X_df = None
        if X_df is not None:
            seg_mask = pd.Series(True, index=X_df.index)
            for k in segment_keys:
                seg_mask = seg_mask & (X_df[k].isin(sub[k].unique()))
            seg_X_df = X_df[seg_mask].copy() if seg_mask.any() else None

        pred = _predict_segment(fcst, horizon=horizon, X_df=seg_X_df)
        predictions.append(pred)

    out = pd.concat(predictions, ignore_index=True).sort_values(["unique_id", "ds"]).reset_index(drop=True)
    logger.info(f"fit_predict_segmented: total {time.time() - t0:.1f}s, {len(out):,d} forecast rows")
    return out


# ------------------------------------------------------------------
# Cross-validation for segmented models
# ------------------------------------------------------------------


def segmented_cv(
    df: pd.DataFrame,
    *,
    h: int = SETTINGS.horizon,
    n_windows: int = SETTINGS.n_windows,
    step_size: int | None = None,
    segment_keys: list[str] | None = None,
) -> pd.DataFrame:
    """Rolling-origin CV with segmented LightGBM models.

    Because ``mlforecast.cross_validation`` operates on a single dataframe,
    we run it **per segment** and concatenate the results. This mirrors the
    winning solution's approach of training separate models per segment.
    """
    from m5.config import set_global_seed

    set_global_seed()
    if segment_keys is None:
        segment_keys = ["store_id"]

    # Pre-build features
    df = build_feature_frame(df.copy())
    static_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    statics_present = [c for c in static_cols if c in df.columns]
    df = encode_static_categoricals(df, statics_present)

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
    dynamics_present = [c for c in dynamic_candidates if c in df.columns]
    keep_cols = ["unique_id", "ds", "y", *statics_present, *dynamics_present]

    logger.info(f"segmented_cv: segment_by={segment_keys} h={h} n_windows={n_windows} step={step_size or h}")

    cv_frames: list[pd.DataFrame] = []
    t0 = time.time()

    for seg_name, sub in _iter_segments(df, segment_keys):
        sub_keep = sub[keep_cols].copy()
        n_series = sub_keep["unique_id"].nunique()
        logger.info(f"  segment {seg_name}: CV on {n_series:,d} series")

        fcst = build_lgbm_forecaster()
        seg_cv = fcst.cross_validation(
            df=sub_keep,
            h=h,
            n_windows=n_windows,
            step_size=step_size or h,
            static_features=statics_present,
        )
        cv_frames.append(seg_cv)

    cv_df = (
        pd.concat(cv_frames, ignore_index=True)
        .sort_values(["unique_id", "ds", "cutoff"])
        .reset_index(drop=True)
    )
    logger.info(f"segmented_cv: total {time.time() - t0:.1f}s, {len(cv_df):,d} rows")
    return cv_df


# ------------------------------------------------------------------
# Convenience builders for the three canonical segmentations
# ------------------------------------------------------------------


def fit_predict_store(
    df: pd.DataFrame, *, horizon: int = SETTINGS.horizon, X_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """10 store-level models."""
    return fit_predict_segmented(df, horizon=horizon, segment_keys=["store_id"], X_df=X_df)


def fit_predict_store_cat(
    df: pd.DataFrame, *, horizon: int = SETTINGS.horizon, X_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """30 store-category models."""
    return fit_predict_segmented(df, horizon=horizon, segment_keys=["store_id", "cat_id"], X_df=X_df)


def fit_predict_store_dept(
    df: pd.DataFrame, *, horizon: int = SETTINGS.horizon, X_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """70 store-department models."""
    return fit_predict_segmented(df, horizon=horizon, segment_keys=["store_id", "dept_id"], X_df=X_df)


def store_cv(
    df: pd.DataFrame,
    *,
    h: int = SETTINGS.horizon,
    n_windows: int = SETTINGS.n_windows,
    step_size: int | None = None,
) -> pd.DataFrame:
    """Rolling-origin CV with 10 store-level models."""
    return segmented_cv(df, h=h, n_windows=n_windows, step_size=step_size, segment_keys=["store_id"])


def store_cat_cv(
    df: pd.DataFrame,
    *,
    h: int = SETTINGS.horizon,
    n_windows: int = SETTINGS.n_windows,
    step_size: int | None = None,
) -> pd.DataFrame:
    """Rolling-origin CV with 30 store-category models."""
    return segmented_cv(
        df, h=h, n_windows=n_windows, step_size=step_size, segment_keys=["store_id", "cat_id"]
    )


def store_dept_cv(
    df: pd.DataFrame,
    *,
    h: int = SETTINGS.horizon,
    n_windows: int = SETTINGS.n_windows,
    step_size: int | None = None,
) -> pd.DataFrame:
    """Rolling-origin CV with 70 store-department models."""
    return segmented_cv(
        df, h=h, n_windows=n_windows, step_size=step_size, segment_keys=["store_id", "dept_id"]
    )
