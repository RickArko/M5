"""Backwards-compatibility shim for the old ``src/process.py``.

The original EDA / forecast / scoring notebooks (`01_eda.ipynb` …
`05_mlforecast_lgbm.ipynb`) imported `from src.process import *`. This module
re-exports every name they used with the *old* schema preserved (`id` column
with `_evaluation` suffix, `date` time column) so the old notebooks keep
running without edits.

New code should import from :mod:`m5.data`, :mod:`m5.config`, etc. directly —
this shim is intentionally narrow and is not maintained for new features.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from m5.config import SETTINGS
from m5.data import load_calendar, load_prices, reduce_mem_usage
from m5.data import load_sales as _new_load_sales
from m5.logging import logger

# --- Old constants ---------------------------------------------------
id_col = "id"
time_col = "date"
id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
PATH_INPUT = SETTINGS.raw_dir


# --- Wide loaders (preserve old `id`/`_evaluation` column) -----------
def load_sales(input_path: Path, prices: pd.DataFrame) -> pd.DataFrame:
    """Wide sales loader with the old ``id`` column (`item_id_store_id_evaluation`)."""
    sales = _new_load_sales(input_path, prices)
    sales["id"] = pd.Categorical(
        sales["item_id"].astype(str) + "_" + sales["store_id"].astype(str) + "_evaluation"
    )
    if "unique_id" in sales.columns:
        sales = sales.drop(columns=["unique_id"])
    return sales


# --- Long-frame builders (old schema: `id`, `date`) -------------------
def filter_data(dflong: pd.DataFrame, last_n: int | None = None) -> pd.DataFrame:
    """Drop leading zeros and trim to last ``last_n`` days. Old `id`/`date` schema."""
    logger.info(f"Rows of input data: {dflong.shape[0]:,d}")
    dflong = dflong.sort_values(["id", "date"])
    dates = sorted(dflong["date"].unique())

    above_min_date = dflong["date"] >= dates[-last_n] if last_n is not None else True
    if last_n is not None:
        logger.info(f"Drop training data older than {last_n:,d} days old")

    without_leading_zeros = (
        dflong["y"].gt(0).groupby(dflong["id"], observed=True).transform("cummax")
    )
    keep_mask = without_leading_zeros & above_min_date
    dflong = dflong[keep_mask].reset_index(drop=True)
    logger.info(f"Rows of processed input data: {dflong.shape[0]:,d}")
    return dflong


def create_m5_fit_data(
    sales: pd.DataFrame,
    cal: pd.DataFrame,
    prices: pd.DataFrame,
    id_vars: list[str] | None = None,
    last_n: int | None = None,
) -> pd.DataFrame:
    """Old long-frame builder — preserves ``id`` and ``date`` columns."""
    if id_vars is None:
        id_vars = id_cols

    t0 = time.time()
    long = sales.melt(id_vars=id_vars, var_name="d", value_name="y")
    long["d"] = long["d"].astype(cal["d"].dtype)
    long = long.merge(cal, on=["d"])
    long = long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"])
    long = filter_data(long, last_n=last_n)
    logger.debug(
        f"Finished creating M5 fit data thru {long['date'].max()} in {time.time() - t0:,.1f}s"
    )
    return long


def create_future_features(
    last_date_train: pd.Timestamp,
    cal: pd.DataFrame,
    prices: pd.DataFrame,
    h: int,
) -> pd.DataFrame:
    """Future calendar + price frame for the forecast horizon (old schema)."""
    val_start = last_date_train
    val_end = last_date_train + pd.Timedelta(days=h)
    last_wmyrwk = cal[cal["date"] == last_date_train]["wm_yr_wk"].iloc[0]

    future_cal = cal[cal["date"].between(val_start, val_end)]
    future_prices = prices[prices["wm_yr_wk"] >= last_wmyrwk].copy()
    future_prices["id"] = (
        future_prices["item_id"].astype(str)
        + "_"
        + future_prices["store_id"].astype(str)
        + "_evaluation"
    )
    return future_prices.merge(future_cal, on="wm_yr_wk").drop(
        columns=["store_id", "item_id", "wm_yr_wk", "d"]
    )


def get_dfids(path_input: Path = PATH_INPUT) -> pd.DataFrame:
    """Return the (item, dept, cat, store, state) id table from `sales_test_evaluation.csv`."""
    truth = pd.read_csv(path_input / "sales_test_evaluation.csv")
    truth["id"] = truth["item_id"].astype(str) + "_" + truth["store_id"].astype(str) + "_evaluation"
    return truth[id_cols]


# Old absolute-path constant the EDA notebooks used. Now points at the
# unified `make prep` output.
TRAIN_PARQUET_PATH: Path = SETTINGS.processed_dir / "long.parquet"


def load_train_parquet(path: Path | str | None = None) -> pd.DataFrame:
    """Load the prepared training parquet and convert it to the **legacy** schema.

    The unified prep step (`make prep`) writes the Nixtla schema
    (``unique_id, ds, y``). The original notebooks expect the old schema
    (``id`` with ``_evaluation`` suffix, ``date`` time column). This loader
    bridges the two so the EDA notebooks keep running unchanged.

    Falls back to the legacy path ``data/train.snap.parquet`` if it exists, so
    repos that still have the old artifact also work.
    """
    if path is not None:
        p = Path(path)
    else:
        legacy = Path("data/train.snap.parquet")
        p = legacy if legacy.exists() else TRAIN_PARQUET_PATH

    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run `make prep` first to build the training parquet."
        )

    df = pd.read_parquet(p)
    if "unique_id" in df.columns:
        df = df.rename(columns={"unique_id": "id", "ds": "date"})
        # Restore the `_evaluation` suffix the old wide loader produced
        df["id"] = df["id"].astype(str) + "_evaluation"
    return df


__all__ = [
    "PATH_INPUT",
    "TRAIN_PARQUET_PATH",
    "create_future_features",
    "create_m5_fit_data",
    "filter_data",
    "get_dfids",
    "id_col",
    "id_cols",
    "load_calendar",
    "load_prices",
    "load_sales",
    "load_train_parquet",
    "reduce_mem_usage",
    "time_col",
]
