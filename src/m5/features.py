"""Minimal feature set — date features, snap, event flag, price normaliser.

Philosophy: keep the menu short. Lags/rolls are configured directly on the
MLForecast model (see :mod:`m5.models.lgbm`) so there's exactly one place
where temporal features are defined.
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


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full minimal feature pipeline in place-friendly order."""
    df = add_date_features(df)
    df = add_snap_flag(df)
    df = add_event_flag(df)
    df = add_price_features(df)
    return df


def static_features(df: pd.DataFrame) -> pd.DataFrame:
    """One row per series with category-level static features (ML-Forecast format)."""
    cols = ["unique_id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    have = [c for c in cols if c in df.columns]
    return df.drop_duplicates("unique_id")[have].reset_index(drop=True)
