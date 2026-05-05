"""Load and shape M5 raw CSVs into a Nixtla-compatible long frame.

Schema convention (Nixtla):
    unique_id : series id (item_id + "_" + store_id)
    ds        : datestamp (datetime64[ns])
    y         : target (float32)
    plus exogenous columns (calendar/price/snap features)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from m5.config import SETTINGS
from m5.logging import logger

ID_COLS = ["unique_id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
EVENT_COLS = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]


def _calendar_dtypes() -> dict[str, str | type]:
    return {
        "wm_yr_wk": np.uint16,
        "event_name_1": "category",
        "event_type_1": "category",
        "event_name_2": "category",
        "event_type_2": "category",
        "snap_CA": np.uint8,
        "snap_TX": np.uint8,
        "snap_WI": np.uint8,
    }


def load_calendar(raw_dir: Path) -> pd.DataFrame:
    dtypes = _calendar_dtypes()
    cal = pd.read_csv(
        raw_dir / "calendar.csv",
        dtype=dtypes,
        usecols=[*dtypes.keys(), "date"],
        parse_dates=["date"],
    )
    for col in EVENT_COLS:
        cal[col] = cal[col].cat.add_categories("none").fillna("none")
    cal["d"] = pd.Categorical([f"d_{i + 1}" for i in range(len(cal))])
    return cal


def load_prices(raw_dir: Path) -> pd.DataFrame:
    return pd.read_csv(
        raw_dir / "sell_prices.csv",
        dtype={
            "store_id": "category",
            "item_id": "category",
            "wm_yr_wk": np.uint16,
            "sell_price": np.float32,
        },
    )


def load_sales(raw_dir: Path, prices: pd.DataFrame, n_days: int = 1941) -> pd.DataFrame:
    """Load wide sales (one column per d_*) using the *evaluation* split."""
    dtypes: dict[str, object] = {
        "id": "category",
        "item_id": prices["item_id"].dtype,
        "dept_id": "category",
        "cat_id": "category",
        "store_id": "category",
        "state_id": "category",
    }
    dtypes.update({f"d_{i}": np.float32 for i in range(1, n_days + 1)})

    candidate = raw_dir / "sales_train_evaluation.csv"
    if not candidate.exists():
        candidate = raw_dir / "sales_train_validation.csv"
    sales = pd.read_csv(candidate, dtype=dtypes)

    sales["unique_id"] = pd.Categorical(
        sales["item_id"].astype(str) + "_" + sales["store_id"].astype(str)
    )
    return sales


def reduce_mem_usage(df: pd.DataFrame, *, verbose: bool = True) -> pd.DataFrame:
    """Down-cast numeric columns to the smallest dtype that fits."""
    int_kinds = (np.int8, np.int16, np.int32, np.int64)
    float_kinds = (np.float32, np.float64)
    start = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        kind = df[col].dtype.kind
        if kind == "i":
            cmin, cmax = df[col].min(), df[col].max()
            for t in int_kinds:
                if cmin >= np.iinfo(t).min and cmax <= np.iinfo(t).max:
                    df[col] = df[col].astype(t)
                    break
        elif kind == "f":
            cmin, cmax = df[col].min(), df[col].max()
            for t in float_kinds:
                if cmin >= np.finfo(t).min and cmax <= np.finfo(t).max:
                    df[col] = df[col].astype(t)
                    break
    end = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        logger.info(f"reduce_mem_usage: {start:,.1f} MB → {end:,.1f} MB ({100 * (start - end) / start:.1f}% drop)")
    return df


def build_long_frame(
    sales: pd.DataFrame,
    cal: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    last_n_days: int | None = None,
    n_series: int | None = None,
) -> pd.DataFrame:
    """Melt wide sales → Nixtla long frame, attach calendar + price features.

    Returns columns: ``unique_id, ds, y`` + static and time-varying covariates.
    """
    if n_series is not None and n_series > 0:
        kept = sales["unique_id"].drop_duplicates().sample(n=n_series, random_state=SETTINGS.seed)
        sales = sales[sales["unique_id"].isin(kept)]
        logger.info(f"Subsampled to {n_series:,d} series for fast iteration.")

    id_vars = ["unique_id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    long = sales.melt(id_vars=id_vars, var_name="d", value_name="y")

    long["d"] = long["d"].astype(cal["d"].dtype)
    long = long.merge(cal, on="d", how="left")
    long = long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    long = long.rename(columns={"date": "ds"})
    long = long.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    long = _drop_leading_zeros(long)
    if last_n_days is not None:
        cutoff = long["ds"].max() - pd.Timedelta(days=int(last_n_days))
        long = long[long["ds"] >= cutoff].reset_index(drop=True)
        logger.info(f"Kept last {last_n_days:,d} days → {len(long):,d} rows.")

    long["y"] = long["y"].astype(np.float32)
    return long


def _drop_leading_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """Remove leading zero observations within each series (item not yet stocked)."""
    has_started = df.groupby("unique_id", observed=True)["y"].transform(lambda s: s.gt(0).cummax())
    out = df[has_started.astype(bool)].reset_index(drop=True)
    logger.debug(f"Dropped {len(df) - len(out):,d} leading-zero rows.")
    return out


def split_train_horizon(df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/holdout split that mirrors the M5 evaluation window."""
    cutoff = df["ds"].max() - pd.Timedelta(days=horizon)
    train = df[df["ds"] <= cutoff].copy()
    holdout = df[df["ds"] > cutoff].copy()
    return train, holdout
