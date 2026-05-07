"""Prediction endpoints — stateless (caller-provided history) and stateful (bundled)."""

from __future__ import annotations

import asyncio
import time

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status

from m5.serve.config import ServeSettings
from m5.serve.deps import get_model_handle, get_settings
from m5.serve.observability import PREDICT_LATENCY, PREDICT_SERIES
from m5.serve.schemas import (
    ForecastPoint,
    PredictByIdRequest,
    PredictRequest,
    PredictResponse,
)
from m5.serve.state import ModelHandle

router = APIRouter(prefix="/v1", tags=["predict"])


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict from caller-provided history (stateless)",
)
async def predict_stateless(
    body: PredictRequest,
    handle: ModelHandle = Depends(get_model_handle),
    settings: ServeSettings = Depends(get_settings),
) -> PredictResponse:
    _check_horizon(body.horizon, settings)
    if len(body.history) > settings.max_history_points:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"history size ({len(body.history):,d}) exceeds "
                f"server max ({settings.max_history_points:,d})."
            ),
        )

    history_df = pd.DataFrame(
        [{"unique_id": h.unique_id, "ds": pd.Timestamp(h.ds), "y": float(h.y)} for h in body.history]
    )
    n_series = int(history_df["unique_id"].nunique())
    _check_series_count(n_series, settings)

    # v1 contract: known series only. Cold-start would also need static features —
    # accepting that without statics produces silently-bad predictions.
    unknown = sorted(set(history_df["unique_id"].astype(str)) - handle.known_ids)
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"unknown unique_ids (cold-start not supported in v1): "
                f"{unknown[:5]}{'...' if len(unknown) > 5 else ''}"
            ),
        )

    # Per-series history must cover the lag window the model was fit with.
    min_required = handle.metadata.min_history_required
    counts = history_df.groupby("unique_id", observed=True)["ds"].count()
    short = counts[counts < min_required]
    if not short.empty:
        worst_id = str(short.idxmin())
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"insufficient history: {len(short)} series have < {min_required} rows "
                f"(worst: {worst_id} = {int(short.loc[worst_id])} rows)."
            ),
        )

    history_df = history_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    t0 = time.perf_counter()
    forecast_df = await asyncio.to_thread(handle.predict_with_history, history_df, body.horizon)
    elapsed = time.perf_counter() - t0
    PREDICT_LATENCY.labels("stateless").observe(elapsed)
    PREDICT_SERIES.labels("stateless").inc(len(forecast_df))

    return _to_response(
        forecast_df,
        horizon=body.horizon,
        n_series=n_series,
        elapsed=elapsed,
        version=handle.metadata.git_sha,
    )


@router.post(
    "/predict/by-id",
    response_model=PredictResponse,
    summary="Predict from bundled history (stateful)",
)
async def predict_stateful(
    body: PredictByIdRequest,
    handle: ModelHandle = Depends(get_model_handle),
    settings: ServeSettings = Depends(get_settings),
) -> PredictResponse:
    _check_horizon(body.horizon, settings)
    _check_series_count(len(body.unique_ids), settings)

    requested = list(dict.fromkeys(body.unique_ids))  # de-dup, preserve order
    unknown = [u for u in requested if u not in handle.known_ids]
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unknown unique_ids: {unknown[:5]}{'...' if len(unknown) > 5 else ''}",
        )

    t0 = time.perf_counter()
    forecast_df = await asyncio.to_thread(handle.predict_by_ids, requested, body.horizon)
    elapsed = time.perf_counter() - t0
    PREDICT_LATENCY.labels("stateful").observe(elapsed)
    PREDICT_SERIES.labels("stateful").inc(len(forecast_df))

    return _to_response(
        forecast_df,
        horizon=body.horizon,
        n_series=len(requested),
        elapsed=elapsed,
        version=handle.metadata.git_sha,
    )


# ----------------------------------------------------------- helpers


def _check_horizon(h: int, settings: ServeSettings) -> None:
    if h > settings.max_horizon:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"horizon ({h}) exceeds server max ({settings.max_horizon}).",
        )


def _check_series_count(n: int, settings: ServeSettings) -> None:
    if n > settings.max_series_per_request:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"series count ({n}) exceeds server max ({settings.max_series_per_request}).",
        )


def _to_response(
    forecast_df: pd.DataFrame,
    *,
    horizon: int,
    n_series: int,
    elapsed: float,
    version: str,
) -> PredictResponse:
    """mlforecast emits a wide frame with one column per model — surface it as ``y_hat``."""
    if forecast_df.empty:
        return PredictResponse(
            model_version=version,
            horizon=horizon,
            n_series=n_series,
            inference_ms=round(elapsed * 1000, 2),
            forecasts=[],
        )
    pred_cols = [c for c in forecast_df.columns if c not in ("unique_id", "ds")]
    if not pred_cols:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="forecast frame has no prediction columns",
        )
    # Prefer the canonical "LGBM" column; otherwise use the first non-key column
    # so the endpoint also serves recipe-driven models with custom aliases.
    col = "LGBM" if "LGBM" in pred_cols else pred_cols[0]

    # Vectorized conversion → list zip is both faster and easier to type
    # than constructing pd.Timestamp per row inside the comprehension.
    uids = forecast_df["unique_id"].astype(str).tolist()
    dss = pd.to_datetime(forecast_df["ds"]).dt.date.tolist()
    yhats = forecast_df[col].astype(float).tolist()
    points = [
        ForecastPoint(unique_id=uid, ds=ds, y_hat=yhat)
        for uid, ds, yhat in zip(uids, dss, yhats, strict=True)
    ]
    return PredictResponse(
        model_version=version,
        horizon=horizon,
        n_series=n_series,
        inference_ms=round(elapsed * 1000, 2),
        forecasts=points,
    )
