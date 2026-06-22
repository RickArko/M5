"""Minimal feature set — date features, snap, event flag, price normaliser.

All functions in this module are backend-agnostic — they accept and return
native DataFrames (pandas/polars).  The public API preserves backward
compatibility: when a pandas DataFrame is passed as input, a pandas
DataFrame is returned.
"""

from __future__ import annotations

import functools

import narwhals as nw
import numpy as np
import pandas as pd

from m5.backend import B, Float32, Int8, Int16, String, to_pandas


def _preserve_backend(func):
    """Decorator: wraps *func* so it accepts native DataFrames, converts to
    narwhals internally, and returns the same native type as the input."""

    @functools.wraps(func)
    def wrapper(df, *args, **kwargs):
        is_pandas = isinstance(df, pd.DataFrame)
        nw_df = B.from_native(df) if not isinstance(df, nw.DataFrame) else df
        result = func(nw_df, *args, **kwargs)
        return to_pandas(result) if is_pandas else result

    return wrapper


DATE_FEATURE_COLS = ("dayofweek", "day", "week", "month", "year", "is_weekend")
SNAP_COLS = ("snap_CA", "snap_TX", "snap_WI")
PRICE_COLS = ("sell_price", "price_norm", "price_change_pct")

# Sentinel for missing event distances.
_NO_EVENT: int = 999


# ── Date features ────────────────────────────────────────────────────


@_preserve_backend
def add_date_features(df, *, ds_col: str = "ds"):
    """Add dayofweek, day, week, month, year, is_weekend from *ds_col*."""
    wd = nw.col(ds_col).dt.weekday() - 1  # narwhals 1-7 → 0-6
    ordinal = nw.col(ds_col).dt.ordinal_day()
    return df.with_columns(
        wd.cast(Int8).alias("dayofweek"),
        nw.col(ds_col).dt.day().cast(Int8).alias("day"),
        ((ordinal - wd + 10) // 7).cast(Int8).alias("week"),
        nw.col(ds_col).dt.month().cast(Int8).alias("month"),
        nw.col(ds_col).dt.year().cast(Int16).alias("year"),
        (wd >= 5).cast(Int8).alias("is_weekend"),
    )


# ── Snap flag ─────────────────────────────────────────────────────────


@_preserve_backend
def add_snap_flag(df):
    """Per-row snap flag for the row's state — collapses 3 columns into 1."""
    if "state_id" not in df.columns:
        return df
    state_up = nw.col("state_id").cast(String).str.to_uppercase()
    expr = nw.lit(0, dtype=Int8)
    for s in ("CA", "TX", "WI"):
        col = f"snap_{s}"
        if col in df.columns:
            expr = nw.when(state_up == s).then(nw.col(col).cast(Int8)).otherwise(expr)
    return df.with_columns(expr.alias("snap"))


# ── Event flag ────────────────────────────────────────────────────────


@_preserve_backend
@_preserve_backend
def add_event_flag(df):
    """Single binary flag for ``any event today`` — drops sparse multi-hot encoding."""
    if "event_name_1" not in df.columns:
        return df
    return df.with_columns((nw.col("event_name_1").cast(String) != "none").cast(Int8).alias("is_event"))


# ── Price features ────────────────────────────────────────────────────


@_preserve_backend
def add_price_features(df):
    """Per-series price normalisation and week-over-week change."""
    if "sell_price" not in df.columns:
        return df
    sp = nw.col("sell_price")
    price_mean = sp.mean().over("unique_id")
    pct_chg = (sp - sp.shift(1).over("unique_id")) / sp.shift(1).over("unique_id")
    return df.with_columns(
        (sp / price_mean).cast(Float32).alias("price_norm"),
        pct_chg.fill_null(0).cast(Float32).alias("price_change_pct"),
    )


# ── Phase 2 — expanded features ───────────────────────────────────────


@_preserve_backend
def add_mean_encoding_features(df):
    """Historical mean-sales encodings by (group, dayofweek)."""
    if "y" not in df.columns:
        return df
    if "dayofweek" not in df.columns:
        df = add_date_features(df)
    agg_levels: list[str] = []
    for col in ("cat_id", "dept_id", "store_id", "state_id"):
        if col in df.columns:
            agg_levels.append(col)
    for level in agg_levels:
        name = f"{level}_mean_dow"
        if name in df.columns:
            continue
        encoding = (
            df.group_by(level, "dayofweek")
            .agg(nw.col("y").mean().alias(name))
            .with_columns(nw.col(name).cast(Float32))
        )
        df = df.join(encoding, on=[level, "dayofweek"], how="left")
    if "store_id" in df.columns and "cat_id" in df.columns:
        name = "store_cat_mean_dow"
        if name not in df.columns:
            encoding = (
                df.group_by("store_id", "cat_id", "dayofweek")
                .agg(nw.col("y").mean().alias(name))
                .with_columns(nw.col(name).cast(Float32))
            )
            df = df.join(encoding, on=["store_id", "cat_id", "dayofweek"], how="left")
    return df


@_preserve_backend
def add_calendar_features(df):
    """Calendar distance features — days to/from nearest event, week-of-month."""
    if "ds" not in df.columns:
        return df
    if "week_of_month" not in df.columns:
        df = df.with_columns(((nw.col("ds").dt.day() - 1) // 7 + 1).cast(Int8).alias("week_of_month"))
    has_event = "is_event" in df.columns or "event_name_1" in df.columns
    if has_event and ("days_to_next_event" not in df.columns or "days_since_last_event" not in df.columns):
        if "is_event" not in df.columns:
            df = add_event_flag(df)
        date_event = df.select("ds", "is_event").unique(subset=["ds"]).sort("ds").to_pandas()
        event_dates = date_event.loc[date_event["is_event"] == 1, "ds"]
        if not event_dates.empty:
            evt_vals = event_dates.sort_values().to_numpy()
            ds_vals = date_event["ds"].to_numpy()
            next_idx = np.searchsorted(evt_vals, ds_vals, side="right")
            clipped_next_idx = np.clip(next_idx, 0, len(evt_vals) - 1)
            next_evt = np.where(next_idx < len(evt_vals), evt_vals[clipped_next_idx], np.datetime64("NaT"))
            days_next = np.where(
                next_idx < len(evt_vals),
                (next_evt - ds_vals).astype("timedelta64[D]").astype(np.int16),
                np.int16(_NO_EVENT),
            )
            prev_idx = np.searchsorted(evt_vals, ds_vals, side="left") - 1
            clipped_prev_idx = np.clip(prev_idx, 0, len(evt_vals) - 1)
            prev_evt = np.where(prev_idx >= 0, evt_vals[clipped_prev_idx], np.datetime64("NaT"))
            days_prev = np.where(
                prev_idx >= 0,
                (ds_vals - prev_evt).astype("timedelta64[D]").astype(np.int16),
                np.int16(_NO_EVENT),
            )
            dist_df = B.from_dict(
                {
                    "ds": ds_vals,
                    "days_to_next_event": days_next,
                    "days_since_last_event": days_prev,
                }
            )
            df = df.join(dist_df, on="ds", how="left")
    return df


@_preserve_backend
def add_price_stats(df):
    """Per-series historical price min / max / mean, and per-store daily rank."""
    if "sell_price" not in df.columns:
        return df
    sp = nw.col("sell_price")
    df = df.with_columns(sp.mean().over("unique_id").cast(Float32).alias("price_mean"))
    df = df.with_columns(sp.min().over("unique_id").cast(Float32).alias("price_min"))
    df = df.with_columns(sp.max().over("unique_id").cast(Float32).alias("price_max"))
    if "store_id" in df.columns and "price_rank_in_store" not in df.columns:
        df = df.with_columns(
            sp.rank(method="average").over(["store_id", "ds"]).cast(Float32).alias("price_rank_in_store")
        )
    return df


@_preserve_backend
def add_release_features(df):
    """Days since the first non-zero sale for each series."""
    if "y" not in df.columns or "days_since_release" in df.columns:
        return df
    release = df.filter(nw.col("y") > 0).group_by("unique_id").agg(nw.col("ds").min().alias("release_date"))
    df = df.join(release, on="unique_id", how="left")
    duration_s = (nw.col("ds") - nw.col("release_date")).dt.total_seconds()
    df = df.with_columns((duration_s / 86400).cast(Int16).fill_null(-1).alias("days_since_release"))
    return df.drop(["release_date"])


# ── Pipeline ──────────────────────────────────────────────────────────


def build_feature_frame(df):
    """Apply the full minimal feature pipeline in place-friendly order.

    Accepts and returns ``pd.DataFrame`` (or any backend-native DataFrame).
    When the input is a pandas DataFrame, the output is also a pandas DataFrame
    (required for Nixtla downstream compatibility).
    """
    is_pandas = isinstance(df, pd.DataFrame)
    df = B.from_native(df) if not isinstance(df, nw.DataFrame) else df
    df = add_date_features(df)
    df = add_snap_flag(df)
    df = add_event_flag(df)
    df = add_price_features(df)
    df = add_mean_encoding_features(df)
    df = add_calendar_features(df)
    df = add_price_stats(df)
    df = add_release_features(df)
    return B.to_pandas(df) if is_pandas else df


# ── Static features ───────────────────────────────────────────────────


def static_features(df):
    """One row per series with category-level static features (ML-Forecast format).

    Accepts and returns ``pd.DataFrame`` (or any backend-native DataFrame).
    """
    is_pandas = isinstance(df, pd.DataFrame)
    df = B.from_native(df) if not isinstance(df, nw.DataFrame) else df
    cols = ["unique_id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    have = [c for c in cols if c in df.columns]
    result = df.unique(subset=["unique_id"]).select(have)
    return B.to_pandas(result) if is_pandas else result
