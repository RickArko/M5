"""Loaded-model handle — owns the artifact lifecycle and prediction calls.

A single :class:`ModelHandle` is created during the FastAPI lifespan and reused
for every request. mlforecast's ``predict`` is sync/CPU-bound; predict methods
here are sync and the routes await them through ``asyncio.to_thread`` to keep
the event loop responsive.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from mlforecast import MLForecast

from m5.logging import logger


@dataclass(frozen=True)
class ModelMetadata:
    """Trained-model metadata — frozen snapshot loaded from metadata.json."""

    model_kind: str
    framework: str
    framework_version: str
    trained_at: str
    git_sha: str
    training_cutoff: str
    freq: str
    horizon_default: int
    lags: list[int]
    rolling_windows: list[int]
    n_series: int
    n_rows: int
    min_history_required: int
    static_features: list[str]
    seed: int


_REQUIRED_FILES = ("metadata.json", "model.joblib")


class ModelHandle:
    """Loads + serves a single MLForecast artifact.

    Construction raises on any missing required file; the FastAPI lifespan
    catches that and reports the service as not-ready via ``/readyz``.
    """

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        for name in _REQUIRED_FILES:
            if not (self.model_dir / name).exists():
                raise FileNotFoundError(f"Missing {name} under {self.model_dir}")

        # mlforecast.predict is not documented as thread-safe; mlforecast uses
        # internal pandas frames during prediction. Serialize predict calls.
        self._lock = threading.Lock()

        meta_raw = json.loads((self.model_dir / "metadata.json").read_text())
        self.metadata = ModelMetadata(
            model_kind=meta_raw["model_kind"],
            framework=meta_raw["framework"],
            framework_version=meta_raw["framework_version"],
            trained_at=meta_raw["trained_at"],
            git_sha=meta_raw["git_sha"],
            training_cutoff=meta_raw["training_cutoff"],
            freq=meta_raw["freq"],
            horizon_default=int(meta_raw["horizon_default"]),
            lags=list(meta_raw["lags"]),
            rolling_windows=list(meta_raw["rolling_windows"]),
            n_series=int(meta_raw["n_series"]),
            n_rows=int(meta_raw.get("n_rows", 0)),
            min_history_required=int(meta_raw["min_history_required"]),
            static_features=list(meta_raw["static_features"]),
            seed=int(meta_raw["seed"]),
        )

        logger.info(f"ModelHandle: loading {self.model_dir / 'model.joblib'}")
        self.fcst: MLForecast = joblib.load(self.model_dir / "model.joblib")

        statics_path = self.model_dir / "statics.parquet"
        self.statics: pd.DataFrame = (
            pd.read_parquet(statics_path) if statics_path.exists() else pd.DataFrame(columns=["unique_id"])
        )

        history_path = self.model_dir / "history.parquet"
        self.history: pd.DataFrame = (
            pd.read_parquet(history_path)
            if history_path.exists()
            else pd.DataFrame(columns=["unique_id", "ds", "y"])
        )
        if not self.history.empty:
            self.history["ds"] = pd.to_datetime(self.history["ds"])

        self.known_ids: frozenset[str] = (
            frozenset(self.statics["unique_id"].astype(str)) if not self.statics.empty else frozenset()
        )
        logger.info(
            f"ModelHandle: ready — {len(self.known_ids):,d} known series, "
            f"trained @ {self.metadata.training_cutoff}, version={self.metadata.git_sha}"
        )

    # -- Inference -----------------------------------------------------
    def predict_by_ids(self, ids: list[str], horizon: int) -> pd.DataFrame:
        """Stateful predict — uses the bundled training history."""
        with self._lock:
            return self.fcst.predict(h=horizon, ids=ids)

    def predict_with_history(self, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Stateless predict — caller-provided history overrides the bundled one.

        mlforecast's ``predict(new_df=...)`` requires every static feature the
        model was fit with to be present per row. Callers send only
        ``(unique_id, ds, y)``, so we hydrate the statics from the bundled
        ``statics.parquet`` for known series before forwarding.
        """
        history = self._attach_statics(history)
        with self._lock:
            return self.fcst.predict(h=horizon, new_df=history)

    def _attach_statics(self, history: pd.DataFrame) -> pd.DataFrame:
        """Merge bundled static features onto a caller-provided history frame.

        The merge is left-join on ``unique_id``; unknown ids would get NaN
        statics, but the route validates against ``known_ids`` first so this
        is a known-series-only path. Static columns are cast to ``category``
        to match the dtype LightGBM saw at fit time (parquet may round-trip
        them as ``object``).
        """
        if not self.metadata.static_features or self.statics.empty:
            return history
        cols = ["unique_id", *self.metadata.static_features]
        out = history.merge(self.statics[cols], on="unique_id", how="left")
        for c in self.metadata.static_features:
            if c in out.columns and out[c].dtype == "object":
                out[c] = out[c].astype("category")
        return out
