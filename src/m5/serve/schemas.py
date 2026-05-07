"""Pydantic v2 request/response models for the prediction API.

Validation goals:

* Reject malformed requests at the parser level (no manual checks in routes).
* Defensive caps on size live in :class:`ServeSettings` and are enforced in routes.
* Field shapes match the Nixtla long-frame schema (``unique_id``, ``ds``, ``y``).
"""

from __future__ import annotations

from datetime import date
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Strict(BaseModel):
    """Base for request models — forbids unknown keys, allows ``model_*`` field names."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class _Loose(BaseModel):
    """Base for response models — frozen, allows ``model_*`` field names."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())


# ---------------------------------------------------------------- Inputs


class HistoryPoint(_Strict):
    """A single (unique_id, ds, y) observation — the Nixtla long-frame row."""

    unique_id: Annotated[str, Field(min_length=1, max_length=128)]
    ds: date
    y: Annotated[float, Field(ge=0)]


class StaticFeatures(_Strict):
    """Per-series static covariates. Rejected by the route in v1 (cold-start out of scope)."""

    unique_id: Annotated[str, Field(min_length=1, max_length=128)]
    item_id: str | None = None
    dept_id: str | None = None
    cat_id: str | None = None
    store_id: str | None = None
    state_id: str | None = None


class PredictRequest(_Strict):
    """Stateless predict — caller sends recent history per series."""

    horizon: int = Field(..., ge=1, description="Number of days to forecast.")
    history: list[HistoryPoint] = Field(..., min_length=1)
    statics: list[StaticFeatures] | None = None
    level: list[int] | None = Field(
        default=None,
        description="Optional prediction-interval levels (e.g. [80, 95]). Each must be in (0, 100).",
    )

    @field_validator("level")
    @classmethod
    def _validate_levels(cls, v: list[int] | None) -> list[int] | None:
        if v is None:
            return v
        if not all(0 < x < 100 for x in v):
            raise ValueError("level entries must be in (0, 100)")
        return sorted(set(v))


class PredictByIdRequest(_Strict):
    """Stateful predict — uses bundled history; just pass ids + horizon."""

    horizon: int = Field(..., ge=1)
    unique_ids: list[Annotated[str, Field(min_length=1, max_length=128)]] = Field(..., min_length=1)
    level: list[int] | None = None

    @field_validator("level")
    @classmethod
    def _validate_levels(cls, v: list[int] | None) -> list[int] | None:
        if v is None:
            return v
        if not all(0 < x < 100 for x in v):
            raise ValueError("level entries must be in (0, 100)")
        return sorted(set(v))


# ---------------------------------------------------------------- Outputs


class ForecastPoint(_Loose):
    """A single forecasted point. ``y_hat`` is the LGBM mean prediction."""

    unique_id: str
    ds: date
    y_hat: float


class PredictResponse(_Loose):
    """Wrapper around a list of forecast points + run metadata."""

    model_version: str
    horizon: int
    n_series: int
    inference_ms: float
    forecasts: list[ForecastPoint]


class ModelInfoResponse(_Loose):
    """Trained model metadata, exposed at GET /v1/model."""

    model_kind: str
    framework: str
    framework_version: str
    trained_at: str
    git_sha: str
    training_cutoff: str
    horizon_default: int
    n_series: int
    lags: list[int]
    rolling_windows: list[int]
    min_history_required: int
    static_features: list[str]


class ProblemDetails(_Loose):
    """RFC 7807 problem details — the standard error body for the API."""

    type: str = "about:blank"
    title: str
    status: int
    detail: str | None = None
    instance: str | None = None
