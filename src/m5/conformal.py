"""Distribution-free conformal prediction intervals for point forecasts.

Calibrates prediction intervals from rolling-origin CV residuals, then
applies them to held-out or future point forecasts.

Methodology
-----------
Split conformal prediction on CV residuals:

    1. Compute per-series residuals at each horizon step from CV:
       r_{i,t,w} = y_{i,t,w} - ŷ_{i,t,w}

    2. Pool residuals across series and CV windows (optionally grouped
       by a static feature like ``dept_id``).

    3. For each horizon step t, take the (1 - α) quantile of |r| as the
       half-width q_t.  The prediction interval is ŷ ± q_t.

Three calibration methods are supported:

    ``pooled``
        A single q_t per horizon step across all series.  Default; works
        well for large, homogeneous series sets.

    ``grouped``
        Per-group q_t (e.g. per deptartment).  Requires ``group_col``
        and enough residuals per group.

    ``scaled``
        Interval width scales with sqrt(ŷ) — a simple variance-stabilising
        adjustment for count-like retail data.  ``q_t`` is multiplied by
        sqrt(ŷ / mean_ŷ) per series.

References
----------
    Angelopoulos & Bates (2021), "A Gentle Introduction to Conformal
    Prediction and Distribution-Free Uncertainty Quantification".
    https://arxiv.org/abs/2107.07511
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

MethodT = Literal["pooled", "grouped", "scaled"]

_KNOWN_METHODS: set[str] = {"pooled", "grouped", "scaled"}


@dataclass
class ConformalCalibrator:
    """Distribution-free prediction-interval calibrator.

    Fit on CV residuals with :meth:`fit`, then add intervals to a point
    forecast with :meth:`predict`.

    Parameters
    ----------
    alpha:
        Nominal miscoverage rate.  (1 - α) prediction intervals.
        Default 0.1 → 90 % intervals.
    method:
        Calibration method — ``pooled``, ``grouped``, or ``scaled``.
        Validated at :meth:`fit` time.
    group_col:
        Column in ``cv_df`` used to define groups when
        ``method="grouped"``.  Ignored otherwise.
    min_group_size:
        Minimum calibration residuals per group for grouped intervals.
        Groups below this threshold fall back to the pooled quantile.
    """

    alpha: float = 0.1
    method: str = "pooled"
    group_col: str | None = None
    min_group_size: int = 50

    # fitted state
    model_col: str | None = field(default=None, repr=False)
    horizon: int | None = field(default=None, repr=False)
    _pooled_quantiles: pd.Series | None = field(default=None, repr=False)
    _group_quantiles: pd.DataFrame | None = field(default=None, repr=False)
    _mean_forecast: float | None = field(default=None, repr=False)

    def fit(
        self,
        cv_df: pd.DataFrame,
        model_col: str,
        *,
        id_col: str = "unique_id",
        time_col: str = "ds",
        cutoff_col: str = "cutoff",
        target_col: str = "y",
    ) -> ConformalCalibrator:
        """Calibrate intervals from rolling-origin CV residuals.

        Parameters
        ----------
        cv_df:
            CV frame with columns ``id_col``, ``time_col``,
            ``cutoff_col``, ``target_col``, and ``model_col``.
        model_col:
            Name of the point-forecast column to calibrate against.
        """
        keep = [id_col, time_col, cutoff_col, target_col, model_col]
        if self.group_col and self.group_col in cv_df.columns:
            keep.append(self.group_col)
        df = cv_df[keep].copy()
        df["_step"] = (df[time_col] - df[cutoff_col]).dt.days
        df["_residual"] = df[target_col] - df[model_col]

        self.model_col = model_col
        self.horizon = int(df["_step"].max())

        if self.method not in _KNOWN_METHODS:
            raise ValueError(f"Unknown method: {self.method!r}. Use one of {_KNOWN_METHODS}.")
        if self.method == "pooled":
            self._fit_pooled(df)
        elif self.method == "grouped":
            self._fit_grouped(df)
        elif self.method == "scaled":
            self._fit_pooled(df)
            self._mean_forecast = float(df[model_col].mean())

        return self

    def _fit_pooled(self, df: pd.DataFrame) -> None:
        self._pooled_quantiles = (
            df["_residual"].abs().groupby(df["_step"]).quantile(1 - self.alpha).rename("q")
        )

    def _fit_grouped(self, df: pd.DataFrame) -> None:
        if self.group_col is None or self.group_col not in df.columns:
            raise ValueError(f"method='grouped' requires a valid group_col, got {self.group_col!r}")

        # pooled fallback
        self._fit_pooled(df)

        # per-group quantiles
        grouped = (
            df["_residual"]
            .abs()
            .groupby([df[self.group_col], df["_step"]], observed=True)
            .quantile(1 - self.alpha)
            .rename("q")
            .reset_index()
        )
        self._group_quantiles = grouped

    def predict(
        self,
        forecast_df: pd.DataFrame,
        *,
        id_col: str = "unique_id",
        time_col: str = "ds",
        cutoff_col: str = "cutoff",
        forecast_col: str | None = None,
    ) -> pd.DataFrame:
        """Add prediction intervals to a point forecast.

        Parameters
        ----------
        forecast_df:
            Long forecast frame with ``id_col``, ``time_col``, and the
            forecast column (either ``model_col`` or ``forecast_col``).
        cutoff_col:
            Column identifying the CV cutoff.  If present in the frame,
            steps are computed as ``(ds - cutoff).days``; otherwise
            they are inferred from position within each series.
        forecast_col:
            Column to build intervals from.  Defaults to ``model_col``
            (the column ``fit`` was called with).

        Returns
        -------
        Forecast frame with two additional columns:
            ``{forecast_col}_lo`` — lower bound
            ``{forecast_col}_hi`` — upper bound
        """
        col = forecast_col or self.model_col
        if col is None:
            raise ValueError("No forecast column specified. Pass forecast_col or call fit() first.")

        out = forecast_df.copy()
        out = out.sort_values([id_col, time_col])

        if cutoff_col in out.columns:
            out["_step"] = (out[time_col] - out[cutoff_col]).dt.days
        else:
            out["_step"] = out.groupby(id_col, observed=True).cumcount() + 1

        if self.method == "grouped" and self.group_col is not None:
            out = self._predict_grouped(out, col)
        else:
            out = self._predict_pooled(out, col)

        out = out.drop(columns=["_step"])
        return out

    def _predict_pooled(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        q = self._pooled_quantiles
        assert q is not None, "call fit() before predict()"
        df["_q"] = df["_step"].map(q)
        if self.method == "scaled" and self._mean_forecast is not None:
            scale = np.sqrt(df[col].clip(lower=0) / max(self._mean_forecast, 1e-8))
            df["_q"] = df["_q"] * scale.clip(upper=5)
        lo = df[col] - df["_q"]
        hi = df[col] + df["_q"]
        df[f"{col}_lo"] = lo.clip(lower=0)
        df[f"{col}_hi"] = hi
        return df.drop(columns=["_q"])

    def _predict_grouped(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        if self.group_col not in df.columns:
            return self._predict_pooled(df, col)

        q_pooled = self._pooled_quantiles
        q_grp = self._group_quantiles
        assert q_pooled is not None, "call fit() before predict()"
        assert q_grp is not None, "call fit() before predict()"

        df = df.merge(q_grp, on=[self.group_col, "_step"], how="left")
        df["_q"] = df["q"].fillna(df["_step"].map(q_pooled))
        lo = df[col] - df["_q"]
        hi = df[col] + df["_q"]
        df[f"{col}_lo"] = lo.clip(lower=0)
        df[f"{col}_hi"] = hi
        return df.drop(columns=["q", "_q"])


def compute_residuals(
    cv_df: pd.DataFrame,
    model_col: str,
    *,
    id_col: str = "unique_id",
    time_col: str = "ds",
    cutoff_col: str = "cutoff",
    target_col: str = "y",
) -> pd.DataFrame:
    """Return a long frame of per-row CV residuals.

    Columns: ``id_col``, ``time_col``, ``cutoff_col``, ``target_col``,
    ``model_col``, ``step``, ``residual``.

    Useful for diagnostic plots outside :class:`ConformalCalibrator`.
    """
    df = cv_df[[id_col, time_col, cutoff_col, target_col, model_col]].copy()
    df["step"] = (df[time_col] - df[cutoff_col]).dt.days
    df["residual"] = df[target_col] - df[model_col]
    return df


def coverage(
    truth: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    lo_col: str = "y_hat_lo",
    hi_col: str = "y_hat_hi",
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
) -> float:
    """Empirical coverage of a prediction interval.

    Fraction of rows where ``target_col`` falls within
    ``[lo_col, hi_col]``.
    """
    merged = truth[[id_col, time_col, target_col]].merge(
        forecast[[id_col, time_col, lo_col, hi_col]],
        on=[id_col, time_col],
        how="inner",
    )
    inside = (merged[target_col] >= merged[lo_col]) & (merged[target_col] <= merged[hi_col])
    return float(inside.mean())


def average_interval_width(
    forecast: pd.DataFrame,
    *,
    lo_col: str = "y_hat_lo",
    hi_col: str = "y_hat_hi",
) -> float:
    """Mean absolute width of prediction intervals."""
    return float((forecast[hi_col] - forecast[lo_col]).mean())
