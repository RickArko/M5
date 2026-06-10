"""Minimal feature set — date features, snap, event flag, price normaliser.

Philosophy: keep the menu short. Lags/rolls are configured directly on the
MLForecast model (see :mod:`m5.models.lgbm`) so there's exactly one place
where temporal features are defined.

Phase 2 additions (aggregation encodings, calendar distances, price stats,
release date) follow the same rule: audit signal via CV diff before adding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DATE_FEATURE_COLS = ("dayofweek", "day", "week", "month", "year", "is_weekend")
SNAP_COLS = ("snap_CA", "snap_TX", "snap_WI")
PRICE_COLS = ("sell_price", "price_norm", "price_change_pct")


def add_date_features(df: pd.DataFrame, *, ds_col: str = "ds") -> pd.DataFrame:
    s = df[ds_col]
    df["dayofweek"] = s.dt.dayofweek.astype(np.int8)
    df["day"] = s.dt.day.astype(np.int8)
    df["week"] = s.dt.isocalendar().week.astype(np.int8)
    df["month"] = s.dt.month.astype(np.int8)
    df["year"] = s.dt.year.astype(np.int16)
    df["is_weekend"] = (s.dt.dayofweek >= 5).astype(np.int8)
    return df


def add_snap_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row snap flag for the row's state — collapses 3 columns into 1."""
    if "state_id" not in df.columns:
        return df
    state = df["state_id"].astype(str).str.upper()
    snap = np.zeros(len(df), dtype=np.int8)
    for s in ("CA", "TX", "WI"):
        col = f"snap_{s}"
        if col in df.columns:
            mask = state.eq(s).to_numpy()
            snap[mask] = df.loc[mask, col].astype(np.int8).to_numpy()
    df["snap"] = snap
    return df


def add_event_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Single binary flag for ``any event today`` — drops sparse multi-hot encoding."""
    if "event_name_1" in df.columns:
        df["is_event"] = (df["event_name_1"].astype(str) != "none").astype(np.int8)
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-series price normalisation and week-over-week change."""
    if "sell_price" not in df.columns:
        return df

    grp = df.groupby("unique_id", observed=True)["sell_price"]
    df["price_norm"] = (df["sell_price"] / grp.transform("mean")).astype(np.float32)
    df["price_change_pct"] = grp.pct_change(fill_method=None).fillna(0).astype(np.float32)
    return df


# ------------------------------------------------------------------
# Phase 2 — expanded features (mean encodings, calendar, price, release)
# ------------------------------------------------------------------


def add_mean_encoding_features(df: pd.DataFrame) -> pd.DataFrame:
    """Historical mean-sales encodings by (group, dayofweek).

    These are "static" in the sense that they are pre-computed from the
    full history and then merged back — they do not leak future information.
    """
    if "y" not in df.columns or "dayofweek" not in df.columns:
        return df

    # Ensure dayofweek exists
    if "dayofweek" not in df.columns:
        df = add_date_features(df)

    agg_levels = []
    if "cat_id" in df.columns:
        agg_levels.append("cat_id")
    if "dept_id" in df.columns:
        agg_levels.append("dept_id")
    if "store_id" in df.columns:
        agg_levels.append("store_id")
    if "state_id" in df.columns:
        agg_levels.append("state_id")

    for level in agg_levels:
        name = f"{level}_mean_dow"
        if name in df.columns:
            continue
        encoding = df.groupby([level, "dayofweek"], observed=True)["y"].mean().rename(name).astype(np.float32)
        df = df.merge(encoding, on=[level, "dayofweek"], how="left")

    # Two-way interaction: store + category
    if "store_id" in df.columns and "cat_id" in df.columns:
        name = "store_cat_mean_dow"
        if name not in df.columns:
            encoding = (
                df.groupby(["store_id", "cat_id", "dayofweek"], observed=True)["y"]
                .mean()
                .rename(name)
                .astype(np.float32)
            )
            df = df.merge(encoding, on=["store_id", "cat_id", "dayofweek"], how="left")

    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar distance features — days to/from nearest event, week-of-month."""
    if "ds" not in df.columns:
        return df

    s = df["ds"]

    # Week of month (1-5)
    if "week_of_month" not in df.columns:
        df["week_of_month"] = ((s.dt.day - 1) // 7 + 1).astype(np.int8)

    # Days to next / since last event — requires is_event or event_name_1
    has_event = "is_event" in df.columns or "event_name_1" in df.columns
    if has_event and ("days_to_next_event" not in df.columns or "days_since_last_event" not in df.columns):
        if "is_event" not in df.columns:
            df = add_event_flag(df)

        # Sort by date globally, compute event distances
        df_sorted = df[["ds", "is_event"]].drop_duplicates("ds").sort_values("ds")
        event_dates = df_sorted.loc[df_sorted["is_event"] == 1, "ds"]

        if not event_dates.empty:
            event_dates = event_dates.sort_values().reset_index(drop=True)

            # Vectorized nearest-event search
            ds_vals = df_sorted["ds"].to_numpy()
            evt_vals = event_dates.to_numpy()

            # next event
            next_idx = np.searchsorted(evt_vals, ds_vals, side="right")
            next_evt = np.where(
                next_idx < len(evt_vals),
                evt_vals[next_idx],
                np.datetime64("NaT"),
            )
            days_next = np.where(
                next_idx < len(evt_vals),
                (next_evt - ds_vals).astype("timedelta64[D]").astype(np.int16),
                np.int16(999),
            )

            # prev event
            prev_idx = np.searchsorted(evt_vals, ds_vals, side="left") - 1
            prev_evt = np.where(
                prev_idx >= 0,
                evt_vals[prev_idx],
                np.datetime64("NaT"),
            )
            days_prev = np.where(
                prev_idx >= 0,
                (ds_vals - prev_evt).astype("timedelta64[D]").astype(np.int16),
                np.int16(999),
            )

            df_sorted["days_to_next_event"] = days_next
            df_sorted["days_since_last_event"] = days_prev
            df = df.merge(
                df_sorted[["ds", "days_to_next_event", "days_since_last_event"]],
                on="ds",
                how="left",
            )

    return df


def add_price_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-series historical price min / max / mean, and per-store daily rank."""
    if "sell_price" not in df.columns:
        return df

    grp = df.groupby("unique_id", observed=True)["sell_price"]

    if "price_mean" not in df.columns:
        df["price_mean"] = grp.transform("mean").astype(np.float32)
    if "price_min" not in df.columns:
        df["price_min"] = grp.transform("min").astype(np.float32)
    if "price_max" not in df.columns:
        df["price_max"] = grp.transform("max").astype(np.float32)

    # Price rank within store per day
    if "store_id" in df.columns and "price_rank_in_store" not in df.columns:
        df["price_rank_in_store"] = (
            df.groupby(["store_id", "ds"], observed=True)["sell_price"]
            .rank(method="average", pct=True)
            .astype(np.float32)
        )

    return df


def add_release_features(df: pd.DataFrame) -> pd.DataFrame:
    """Days since the first non-zero sale for each series."""
    if "y" not in df.columns or "days_since_release" in df.columns:
        return df

    release = df[df["y"] > 0].groupby("unique_id", observed=True)["ds"].min().rename("release_date")
    df = df.merge(release, on="unique_id", how="left")
    df["days_since_release"] = (df["ds"] - df["release_date"]).dt.days.astype(np.int16)
    df = df.drop(columns=["release_date"])
    return df


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full minimal feature pipeline in place-friendly order."""
    df = add_date_features(df)
    df = add_snap_flag(df)
    df = add_event_flag(df)
    df = add_price_features(df)
    # Phase 2 expansions
    df = add_mean_encoding_features(df)
    df = add_calendar_features(df)
    df = add_price_stats(df)
    df = add_release_features(df)
    return df


def static_features(df: pd.DataFrame) -> pd.DataFrame:
    """One row per series with category-level static features (ML-Forecast format)."""
    cols = ["unique_id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    have = [c for c in cols if c in df.columns]
    return df.drop_duplicates("unique_id")[have].reset_index(drop=True)
