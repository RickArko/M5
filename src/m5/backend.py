"""Unified DataFrame backend — swap pandas/polars/PySpark via ``M5_BACKEND``.

Uses `narwhals`_ under the hood to provide a consistent Polars-style API
across pandas, Polars, and PySpark DataFrames.

Usage::

    from m5.backend import B, nw

    # B is the global Backend singleton configured by M5_BACKEND
    # nw is narwhals — use it for all DataFrame operations

    # Read data (backend-aware)
    df = B.read_parquet("data.parquet")
    df = B.from_native(pandas_df)          # wrap native → narwhals
    pdf = B.to_pandas(df)                  # always pandas (for Nixtla)
    native = B.to_native(df)               # back to original backend

    # All operations use pure narwhals API:
    df = df.with_columns(c=nw.col("a") + nw.col("b"))
    df = df.filter(nw.col("c") > 0)
    df = df.group_by("id").agg(nw.col("y").mean())
    df = df.join(other, on="id", how="left")

.. _narwhals: https://narwhals-dev.github.io/narwhals/
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import narwhals as nw

from m5.logging import logger

PANDAS = "pandas"
"""pandas backend (default)."""

POLARS = "polars"
"""Polars eager backend."""

SPARK = "pyspark"
"""PySpark (lazy) backend."""

_SUPPORTED_BACKENDS = frozenset({PANDAS, POLARS, SPARK})

# ── Narwhals dtype aliases (used for .cast()) ────────────────────────
Int8 = nw.Int8
Int16 = nw.Int16
Int32 = nw.Int32
Int64 = nw.Int64
UInt8 = nw.UInt8
UInt16 = nw.UInt16
UInt32 = nw.UInt32
UInt64 = nw.UInt64
Float32 = nw.Float32
Float64 = nw.Float64
Boolean = nw.Boolean
String = nw.String
Categorical = nw.Categorical
Datetime = nw.Datetime
Datetime = nw.Datetime
Date = nw.Date

# Map string dtype names (from config) to narwhals dtypes.
DTYPE_MAP: dict[str, nw.dtypes.DType] = {
    "int8": Int8,
    "int16": Int16,
    "int32": Int32,
    "int64": Int64,
    "uint8": UInt8,
    "uint16": UInt16,
    "uint32": UInt32,
    "float32": Float32,
    "float64": Float64,
    "bool": Boolean,
    "str": String,
    "category": Categorical,
}


def _resolve_backend_name(name: str | None) -> str:
    if name and name.lower() in _SUPPORTED_BACKENDS:
        return name.lower()
    raw = os.getenv("M5_BACKEND", PANDAS).lower()
    return raw if raw in _SUPPORTED_BACKENDS else PANDAS


# ── Backend class ─────────────────────────────────────────────────────


class Backend:
    """Unified DataFrame backend.

    Configure once per process via the ``M5_BACKEND`` environment variable
    (default: ``pandas``).  All DataFrame operations use the narwhals API
    directly — this class provides the I/O, conversion, and utility glue.

    Parameters
    ----------
    name:
        Backend name.  If ``None`` (default), read from ``M5_BACKEND``.
    """

    def __init__(self, name: str | None = None):
        self._name = _resolve_backend_name(name)
        # narwhals' ``backend`` param: ``None`` → pandas, ``"polars"`` → polars, etc.
        self._nw_backend: str | None = self._name if self._name != PANDAS else None
        logger.debug(f"Backend: {self._name}")

    # ── Properties ────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """Current backend name."""
        return self._name

    @property
    def is_pandas(self) -> bool:
        return self._name == PANDAS

    @property
    def is_polars(self) -> bool:
        return self._name == POLARS

    @property
    def is_spark(self) -> bool:
        return self._name == SPARK

    # ── I/O ───────────────────────────────────────────────────────

    def read_parquet(self, path: str | Path, **kwargs: Any) -> nw.DataFrame | nw.LazyFrame:
        """Read a parquet file using the configured backend."""
        return nw.read_parquet(str(path), backend=self._nw_backend, **kwargs)

    def write_parquet(self, df: nw.DataFrame | nw.LazyFrame, path: str | Path, **kwargs: Any) -> None:
        """Write a narwhals DataFrame to parquet."""
        df.write_parquet(str(path), **kwargs)

    def read_csv(self, path: str | Path, **kwargs: Any) -> nw.DataFrame | nw.LazyFrame:
        """Read a CSV file using the configured backend."""
        return nw.read_csv(str(path), backend=self._nw_backend, **kwargs)

    # ── Conversion ────────────────────────────────────────────────

    def from_native(self, df: Any, *, eager_only: bool = False) -> nw.DataFrame | nw.LazyFrame:
        """Wrap a native DataFrame (pandas/polars/pyspark) into narwhals."""
        return nw.from_native(df, eager_only=eager_only)

    def to_native(self, df: nw.DataFrame | nw.LazyFrame) -> Any:
        """Unwrap to the native DataFrame type."""
        return nw.to_native(df)

    def to_pandas(self, df: nw.DataFrame | nw.LazyFrame) -> Any:
        """Always return a pandas DataFrame — required for Nixtla interop."""
        if hasattr(df, "to_pandas"):
            return df.to_pandas()
        return nw.to_native(df)

    def from_dict(self, data: dict[str, Any], schema: dict[str, Any] | None = None) -> nw.DataFrame:
        """Create a narwhals DataFrame from a dict of columns.

        Note: when values are plain Python lists, the ``backend`` kwarg
        is required (narwhals >= 2.20).  We pass it automatically.
        """
        return nw.from_dict(data, schema=schema, backend=self._nw_backend)

    # ── Utilities ─────────────────────────────────────────────────

    def concat(
        self,
        items: list[nw.DataFrame | nw.LazyFrame],
        how: str = "vertical",
    ) -> nw.DataFrame | nw.LazyFrame:
        """Concatenate narwhals DataFrames."""
        return nw.concat(items, how=how)

    def collect(self, df: nw.LazyFrame) -> nw.DataFrame:
        """Materialize a LazyFrame into an eager DataFrame."""
        return df.collect()

    def memory_usage(self, df: nw.DataFrame) -> float:
        """Approximate memory usage in MB.

        Uses ``.estimated_size()`` for polars, converts to pandas for
        pandas-native ``.memory_usage(deep=True)``, and falls back to
        estimated_size for other backends.
        """
        if self.is_pandas:
            native = df.to_pandas()
            return float(native.memory_usage(deep=True).sum() / 1024**2)
        return float(df.estimated_size() / 1024**2)

    def set_global(self) -> None:
        """Set this instance as the module-level ``current`` backend.

        Useful for switching backends at runtime::

            polars_backend = Backend("polars")
            polars_backend.set_global()
            df = read_parquet("data.parquet")   # uses polars now
        """
        global current
        current = self


# ── Module-level singleton ────────────────────────────────────────────

current: Backend = Backend()
"""Global :class:`Backend` singleton.  Configured once at import time from
``M5_BACKEND``.  Re-assign or call ``.set_global()`` to switch at runtime."""

B: Backend = current
"""Alias for :data:`current` — shorter import."""

__all__ = [
    "PANDAS",
    "POLARS",
    "SPARK",
    "B",
    "Backend",
    "Boolean",
    "Categorical",
    "Datetime",
    "Float32",
    "Float64",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "String",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "current",
    "drop_leading_zeros",
    "from_native",
    "nw",
    "pct_change",
    "read_csv",
    "read_parquet",
    "shrink_dtypes",
    "tail",
    "to_native",
    "to_pandas",
    "write_parquet",
]


# ── Module-level convenience aliases ──────────────────────────────────


def read_parquet(path: str | Path, **kwargs: Any) -> nw.DataFrame | nw.LazyFrame:
    """Convenience: ``current.read_parquet(...)``."""
    return current.read_parquet(path, **kwargs)


def write_parquet(df: nw.DataFrame | nw.LazyFrame, path: str | Path, **kwargs: Any) -> None:
    """Convenience: ``current.write_parquet(...)``."""
    return current.write_parquet(df, path, **kwargs)


def read_csv(path: str | Path, **kwargs: Any) -> nw.DataFrame | nw.LazyFrame:
    """Convenience: ``current.read_csv(...)``."""
    return current.read_csv(path, **kwargs)


def from_native(df: Any, *, eager_only: bool = False) -> nw.DataFrame | nw.LazyFrame:
    """Convenience: ``current.from_native(...)``."""
    return current.from_native(df, eager_only=eager_only)


def to_native(df: nw.DataFrame | nw.LazyFrame) -> Any:
    """Convenience: ``current.to_native(...)``."""
    return current.to_native(df)


def to_pandas(df: nw.DataFrame | nw.LazyFrame) -> Any:
    """Convenience: always convert to pandas."""
    return current.to_pandas(df)


# ── Helper functions for operations narwhals doesn't support natively ──


def pct_change(
    df: nw.DataFrame,
    col: str,
    *,
    group_by: str | list[str] | None = None,
    order_by: str | list[str] | None = None,
) -> nw.DataFrame:
    """Percentage change, equivalent to pandas ``groupby().pct_change()``.

    Returns a new DataFrame with ``{col}`` replaced by the percentage
    change (null for the first row of each group, then ``(v - lag) / lag``).
    """
    c = nw.col(col)
    if group_by is not None:
        kw: dict[str, Any] = {}
        if order_by is not None:
            kw["order_by"] = order_by
        shifted = c.shift(1).over(group_by, **kw)
    else:
        shifted = c.shift(1)
    return df.with_columns(((c - shifted) / shifted).alias(col))


def tail(
    df: nw.DataFrame,
    n: int,
    *,
    group_by: str | list[str],
    order_by: str | list[str],
    keep: str = "last",
) -> nw.DataFrame:
    """Last *n* rows per group — like pandas ``groupby().tail(n)``.

    Uses a windowed cumulative count with the specified ordering, then
    filters to the last *n* rows within each group.

    Parameters
    ----------
    keep:
        ``"last"`` (default) — keep the *n* most recent rows per the
        ``order_by`` column.  ``"first"`` keeps the earliest *n* rows.
    """
    count_col = "_nw_tail_count"
    descending = keep == "last"
    if descending:
        sorted_df = df.sort(
            [
                *(group_by if isinstance(group_by, list) else [group_by]),
                *(order_by if isinstance(order_by, list) else [order_by]),
            ],
            descending=[False] * (1 if isinstance(group_by, str) else len(group_by))
            + [True] * (1 if isinstance(order_by, str) else len(order_by)),
        )
        counted = sorted_df.with_columns(
            nw.col(order_by[0] if isinstance(order_by, list) else order_by)
            .cum_count()
            .over(group_by)
            .alias(count_col)
        )
    else:
        counted = df.with_columns(
            nw.col(order_by[0] if isinstance(order_by, list) else order_by)
            .cum_count()
            .over(group_by, order_by=order_by)
            .alias(count_col)
        )
    return counted.filter(nw.col(count_col) <= n).drop([count_col])


def drop_leading_zeros(
    df: nw.DataFrame,
    *,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> nw.DataFrame:
    """Remove leading-zero observations within each series.

    Equivalent to::

        df.groupby(id_col)[target_col].transform(
            lambda s: s > 0).cummax().astype(bool)
        df = df[mask]
    """
    started = (nw.col(target_col) > 0).cum_max().over(id_col)
    return df.filter(started)


def shrink_dtypes(df: nw.DataFrame, *, verbose: bool = True) -> nw.DataFrame:
    """Down-cast numeric columns to the smallest dtype that fits.

    This is a narwhals-native version of pandas' ``reduce_mem_usage``.
    Only works correctly on eager DataFrames (pandas/polars).
    """
    if verbose:
        start_mb = current.memory_usage(df)

    schema = df.collect_schema()
    casts: dict[str, nw.dtypes.DType] = {}
    for col_name, dtype in schema.items():
        candidates: list = []
        if dtype == Int64:
            candidates = [Int32, Int16, Int8]
        elif dtype == Int32:
            candidates = [Int16, Int8]
        elif dtype == Int16:
            candidates = [Int8]
        elif dtype == UInt64:
            candidates = [UInt32, UInt16, UInt8]
        elif dtype == UInt32:
            candidates = [UInt16, UInt8]
        elif dtype == UInt16:
            candidates = [UInt8]
        elif dtype == Float64:
            candidates = [Float32]
        else:
            continue

        col_series = df.get_column(col_name)
        cmin = col_series.min()
        cmax = col_series.max()
        if cmin is None or cmax is None:
            continue
        cmin_f, cmax_f = float(cmin), float(cmax)
        for smaller_dtype in candidates:
            if _fits_in_dtype(cmin_f, cmax_f, smaller_dtype):
                casts[col_name] = smaller_dtype
                break

    if casts:
        df = df.with_columns(*[nw.col(c).cast(dt) for c, dt in casts.items()])

    if verbose:
        end_mb = current.memory_usage(df)
        drop_pct = 100.0 * (start_mb - end_mb) / start_mb if start_mb > 0 else 0.0
        logger.info(f"shrink_dtypes: {start_mb:,.1f} MB \u2192 {end_mb:,.1f} MB ({drop_pct:.1f}% drop)")
    return df


def _fits_in_dtype(cmin: float, cmax: float, dtype: nw.dtypes.DType) -> bool:
    """Check if a value range fits in a given narwhals numeric dtype."""
    import numpy as np

    # fmt: off
    if dtype is Int8:     lo, hi = int(np.iinfo(np.int8).min), int(np.iinfo(np.int8).max)       # noqa: E701
    elif dtype is Int16:  lo, hi = int(np.iinfo(np.int16).min), int(np.iinfo(np.int16).max)     # noqa: E701
    elif dtype is Int32:  lo, hi = int(np.iinfo(np.int32).min), int(np.iinfo(np.int32).max)     # noqa: E701
    elif dtype is UInt8:  lo, hi = 0, int(np.iinfo(np.uint8).max)                                # noqa: E701
    elif dtype is UInt16: lo, hi = 0, int(np.iinfo(np.uint16).max)                               # noqa: E701
    elif dtype is UInt32: lo, hi = 0, int(np.iinfo(np.uint32).max)                               # noqa: E701
    elif dtype is Float32: lo, hi = float(np.finfo(np.float32).min), float(np.finfo(np.float32).max)  # noqa: E701
    else:                 return False                                                          # noqa: E701
    # fmt: on
    return bool(cmin >= lo and cmax <= hi)


# ── Re-export narwhals at module level ────────────────────────────────

# Users can ``from m5.backend import nw`` and use the full narwhals API.
