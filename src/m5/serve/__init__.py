"""FastAPI service for the M5 LightGBM forecaster.

Public surface kept intentionally minimal — entry points are:

* :func:`m5.serve.app.create_app` — ASGI app factory (used by uvicorn ``--factory``)
* :class:`m5.serve.config.ServeSettings` — env-driven config (``M5_SERVE_*``)

Run locally::

    python -m m5.serve            # uses .env / M5_SERVE_* env vars
    uvicorn m5.serve.app:create_app --factory --host 0.0.0.0 --port 8000

The artifact contract (under ``M5_SERVE_MODEL_DIR``) is produced by ``m5 train``::

    metadata.json     # framework versions, training cutoff, lags, features, git SHA
    model.joblib      # the fitted MLForecast (joblib.dump)
    history.parquet   # trailing per-series history for stateful predict
    statics.parquet   # one row per series with the static covariates
"""

from __future__ import annotations

from m5.serve.app import create_app
from m5.serve.config import ServeSettings

__all__ = ["ServeSettings", "create_app"]
