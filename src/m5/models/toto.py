"""Zero-shot forecasting with DataDog's TOTO 2.0 time series foundation model.

TOTO is a decoder-only transformer pre-trained on ~2 trillion time series
data points.  This module wraps the ``toto2`` package into the M5 model
interface so ``m5 cv toto`` and ``m5 forecast toto`` work identically to
the existing stats / lgbm runners.

References
----------
- GitHub: https://github.com/datadog/toto
- Paper:  https://arxiv.org/abs/2605.20119
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from m5.config import SETTINGS, set_global_seed
from m5.logging import logger

if TYPE_CHECKING:
    import torch

DEFAULT_MODEL_NAME = "Datadog/Toto-2.0-22m"
DEFAULT_CONTEXT_LENGTH = 512
DEFAULT_BATCH_SIZE = 32
DEFAULT_DECODE_BLOCK_SIZE: int | None = None


def _resolve_device() -> torch.device:
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_toto_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: torch.device | None = None,
) -> torch.nn.Module:
    """Load a TOTO 2.0 model from HuggingFace."""
    from toto2 import Toto2Model

    if device is None:
        device = _resolve_device()
    logger.info(f"build_toto_model: loading {model_name} on {device}")
    model = Toto2Model.from_pretrained(model_name)
    model = model.to(device).eval()
    return model


def _series_to_tensor(
    series: np.ndarray,
    context_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Convert a 1-D series to a ``(1, 1, context_length)`` tensor.

    Left-pads with zeros when the series is shorter than *context_length*.
    """
    import torch

    n = len(series)
    if n >= context_length:
        arr = series[-context_length:].astype(np.float32)
    else:
        arr = np.zeros(context_length, dtype=np.float32)
        arr[-n:] = series.astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)


def _forecast_batch(
    model: torch.nn.Module,
    batch_tensor: torch.Tensor,
    horizon: int,
    decode_block_size: int | None,
) -> np.ndarray:
    """Forecast a batch and return the median (0.5 quantile) predictions.

    Returns
    -------
    ndarray of shape ``(batch_size, horizon)``.
    """
    import torch

    batch_size = batch_tensor.shape[0]
    target_mask = torch.ones_like(batch_tensor, dtype=torch.bool)
    series_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=batch_tensor.device)

    quantiles = model.forecast(
        {"target": batch_tensor, "target_mask": target_mask, "series_ids": series_ids},
        horizon=horizon,
        decode_block_size=decode_block_size,
        has_missing_values=False,
    )
    median = quantiles[4].cpu().numpy()
    return median[:, 0, :]


def toto_forecast(
    df: pd.DataFrame,
    *,
    horizon: int = SETTINGS.horizon,
    model_name: str = DEFAULT_MODEL_NAME,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    decode_block_size: int | None = DEFAULT_DECODE_BLOCK_SIZE,
) -> pd.DataFrame:
    """Zero-shot forecast using TOTO.

    No training is performed — TOTO is a foundation model that forecasts
    directly from the provided context window.

    Parameters
    ----------
    df:
        Nixtla long frame with ``unique_id, ds, y``.
    horizon:
        Forecast horizon in days.
    model_name:
        HuggingFace model identifier or local path.
    context_length:
        Lookback window (days) fed to the model per series.
    batch_size:
        Number of series to infer per batch.  Reduce on OOM.
    decode_block_size:
        Decode block size for iterative generation.  ``None`` (default)
        uses a single forward pass, which is fast and accurate for short
        horizons (<=~100).

    Returns
    -------
    Nixtla-format forecast frame with columns ``unique_id, ds, TOTO``.
    """
    import torch

    set_global_seed()
    model = build_toto_model(model_name=model_name)
    device = next(model.parameters()).device

    sorted_df = df.sort_values(["unique_id", "ds"])
    cutoff = sorted_df["ds"].max()
    future_dates = pd.date_range(cutoff + pd.Timedelta(days=1), periods=horizon, freq="D")
    unique_ids = sorted_df["unique_id"].unique()
    n_series = len(unique_ids)

    logger.info(
        f"toto_forecast: {n_series:,d} series, horizon={horizon}, "
        f"context={context_length}, batch={batch_size}"
    )

    records: list[dict] = []
    log_every = max(1, n_series // batch_size // 10)

    for i in range(0, n_series, batch_size):
        batch_ids = unique_ids[i : i + batch_size]
        tensors = []
        for uid in batch_ids:
            series = sorted_df.loc[sorted_df["unique_id"] == uid, "y"].to_numpy()
            tensors.append(_series_to_tensor(series, context_length, device))
        batch_input = torch.cat(tensors, dim=0)
        preds = _forecast_batch(model, batch_input, horizon, decode_block_size)

        for j, uid in enumerate(batch_ids):
            for k in range(horizon):
                records.append({"unique_id": uid, "ds": future_dates[k], "TOTO": float(preds[j, k])})

        if (i // batch_size) % log_every == 0:
            logger.info(f"toto_forecast: {min(i + batch_size, n_series):,d}/{n_series:,d}")

    forecast_df = pd.DataFrame(records)
    if "unique_id" in df.columns:
        forecast_df["unique_id"] = forecast_df["unique_id"].astype(df["unique_id"].dtype)
    return forecast_df


def toto_cv(
    df: pd.DataFrame,
    *,
    h: int = SETTINGS.horizon,
    n_windows: int = SETTINGS.n_windows,
    step_size: int | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    decode_block_size: int | None = DEFAULT_DECODE_BLOCK_SIZE,
) -> pd.DataFrame:
    """Rolling-origin CV with the TOTO zero-shot model.

    Because TOTO requires no training, each CV window simply slices the
    appropriate context and forecasts forward.  Results are returned in the
    same Nixtla CV format used by ``stats_cv`` / ``lgbm_cv`` so they can be
    scored by ``m5 evaluation.wrmsse_for_models``.
    """
    import torch

    set_global_seed()
    step = step_size or h
    sorted_df = df.sort_values(["unique_id", "ds"])
    max_ds = sorted_df["ds"].max()
    unique_ids = sorted_df["unique_id"].unique()

    logger.info(f"toto_cv: h={h} n_windows={n_windows} step={step} context={context_length}")

    model = build_toto_model(model_name=model_name)
    device = next(model.parameters()).device

    all_folds: list[pd.DataFrame] = []
    log_every = max(1, len(unique_ids) // batch_size // 5)

    for w in range(n_windows):
        cutoff = max_ds - pd.Timedelta(days=(n_windows - 1 - w) * step + h)
        forecast_start = cutoff + pd.Timedelta(days=1)
        forecast_dates = pd.date_range(forecast_start, periods=h, freq="D")

        records: list[dict] = []
        for i in range(0, len(unique_ids), batch_size):
            batch_ids = unique_ids[i : i + batch_size]
            tensors = []
            for uid in batch_ids:
                series = sorted_df.loc[sorted_df["unique_id"] == uid, "y"].to_numpy()
                tensors.append(_series_to_tensor(series, context_length, device))
            batch_input = torch.cat(tensors, dim=0)
            preds = _forecast_batch(model, batch_input, h, decode_block_size)

            for j, uid in enumerate(batch_ids):
                for k, d in enumerate(forecast_dates):
                    records.append({"unique_id": uid, "ds": d, "cutoff": cutoff, "TOTO": float(preds[j, k])})

            if (i // batch_size) % log_every == 0:
                logger.info(
                    f"toto_cv window {w + 1}/{n_windows}: "
                    f"{min(i + batch_size, len(unique_ids)):,d}/{len(unique_ids):,d}"
                )

        fold_df = pd.DataFrame(records)
        fold_df = fold_df.merge(
            sorted_df[["unique_id", "ds", "y"]],
            on=["unique_id", "ds"],
            how="left",
        )
        all_folds.append(fold_df)

    cv_df = pd.concat(all_folds, ignore_index=True)
    if "unique_id" in df.columns:
        cv_df["unique_id"] = cv_df["unique_id"].astype(df["unique_id"].dtype)
    return cv_df
