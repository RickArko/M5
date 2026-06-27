"""Load and shape M5 raw CSVs into a Nixtla-compatible long frame.

Schema convention (Nixtla):
    unique_id : series id (item_id + "_" + store_id)
    ds        : datestamp (datetime64[ns])
    y         : target (float32)
    plus exogenous columns (calendar/price/snap features)
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from m5.backend import B, nw, to_pandas
from m5.config import SETTINGS
from m5.logging import logger

ID_COLS = ["unique_id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
EVENT_COLS = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]

# Sale column range for melt.
_SALE_COL_PATTERN = "d_"


def _calendar_dtypes() -> dict[str, Any]:
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
    """Load M5 calendar CSV with typed dtypes. Returns pandas DataFrame."""
    dtypes = _calendar_dtypes()
    cal = pd.read_csv(
        raw_dir / "calendar.csv",
        dtype=dtypes,  # type: ignore[arg-type]
        usecols=[*dtypes.keys(), "date"],
        parse_dates=["date"],
    )
    for col in EVENT_COLS:
        cal[col] = cal[col].cat.add_categories("none").fillna("none")
    cal["d"] = pd.Categorical([f"d_{i + 1}" for i in range(len(cal))])
    return cal


def load_prices(raw_dir: Path) -> pd.DataFrame:
    """Load M5 sell_prices CSV with typed dtypes. Returns pandas DataFrame."""
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
    """Load wide sales (one column per d_*) using the *evaluation* split.

    Returns pandas DataFrame for backward compatibility with callers
    that expect pandas-specific dtypes (categorical, etc.).
    """
    dtypes: dict[str, Any] = {
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
    sales = pd.read_csv(candidate, dtype=dtypes)  # type: ignore[arg-type]

    sales["unique_id"] = pd.Categorical(sales["item_id"].astype(str) + "_" + sales["store_id"].astype(str))
    return sales


def reduce_mem_usage(df: pd.DataFrame, *, verbose: bool = True) -> pd.DataFrame:
    """Down-cast numeric columns to the smallest dtype that fits.

    Accepts and returns a pandas DataFrame (backward-compatible wrapper
    around :func:`m5.backend.shrink_dtypes`).
    """
    nw_df = B.from_native(df)
    nw_df = shrink_dtypes(nw_df, verbose=verbose)  # type: ignore[arg-type]
    return B.to_pandas(nw_df)


def shrink_dtypes(df: nw.DataFrame, *, verbose: bool = True) -> nw.DataFrame:
    """Down-cast numeric columns — backend-agnostic."""
    from m5.backend import shrink_dtypes as _shrink

    return _shrink(df, verbose=verbose)


def build_long_frame(
    sales: pd.DataFrame,
    cal: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    last_n_days: int | None = None,
    n_series: int | None = None,
) -> pd.DataFrame:
    """Melt wide sales \u2192 Nixtla long frame, attach calendar + price features.

    Returns a pandas DataFrame (for Nixtla downstream compatibility).
    Uses the narwhals backend internally for all processing.
    """
    # Subsample series if requested.
    if n_series is not None and n_series > 0:
        unique_ids = pd.Series(sales["unique_id"].unique())
        kept = unique_ids.sample(n=n_series, random_state=SETTINGS.seed)
        sales = sales[sales["unique_id"].isin(kept)].copy()
        logger.info(f"Subsampled to {n_series:,d} series for fast iteration.")

    # Convert inputs to narwhals for processing.
    sales_nw = B.from_native(sales)
    cal_nw = B.from_native(cal)
    prices_nw = B.from_native(prices)

    id_vars = ["unique_id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

    # Identify sale columns (d_1 .. d_N) — everything not in id_vars.
    all_cols = sales_nw.columns
    sale_cols = [c for c in all_cols if c not in {*id_vars, "id"}]

    # Melt: wide → long.
    long = sales_nw.unpivot(
        index=id_vars,
        on=sale_cols,
        variable_name="d",
        value_name="y",
    )

    # Cast d column to match calendar's dtype.
    cal_d_dtype = cal_nw.schema["d"]
    long = long.with_columns(nw.col("d").cast(cal_d_dtype))

    # Attach calendar and prices.
    long = long.join(cal_nw, on="d", how="left")  # type: ignore[arg-type]
    long = long.join(prices_nw, on=["store_id", "item_id", "wm_yr_wk"], how="left")  # type: ignore[arg-type]

    # Rename date → ds, sort.
    long = long.rename({"date": "ds"})
    long = long.sort(["unique_id", "ds"])

    # Drop leading zeros.
    long = _drop_leading_zeros_nw(long)  # type: ignore[arg-type]

    # Optional trailing-window trim.
    if last_n_days is not None and last_n_days > 0:
        max_ds = long.select(nw.col("ds").max()).item()
        cutoff = max_ds - timedelta(days=int(last_n_days))
        long = long.filter(nw.col("ds") >= cutoff)
        logger.info(f"Kept last {last_n_days:,d} days \u2192 {len(long):,d} rows.")

    # Cast y to float32.
    long = long.with_columns(nw.col("y").cast(nw.Float32))

    # Convert back to pandas for Nixtla downstream compatibility.
    return to_pandas(long)


def _drop_leading_zeros_nw(df: nw.DataFrame) -> nw.DataFrame:
    """Remove leading-zero observations — backend-agnostic version."""
    n_before = df.select(nw.len()).item()
    started = (nw.col("y") > 0).cum_max().over("unique_id")
    out = df.filter(started)
    n_after = out.select(nw.len()).item()
    logger.debug(f"Dropped {n_before - n_after:,d} leading-zero rows.")
    return out


def split_train_horizon(df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/holdout split that mirrors the M5 evaluation window."""
    nw_df = B.from_native(df)
    assert isinstance(nw_df, nw.DataFrame), "expected eager DataFrame"
    max_ds = nw_df.select(nw.col("ds").max()).item()
    cutoff = max_ds - timedelta(days=horizon)
    train = nw_df.filter(nw.col("ds") <= cutoff)
    holdout = nw_df.filter(nw.col("ds") > cutoff)
    return B.to_pandas(train), B.to_pandas(holdout)
