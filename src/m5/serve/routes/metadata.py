"""``GET /v1/model`` — surface the trained-model metadata."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from m5.serve.deps import get_model_handle
from m5.serve.schemas import ModelInfoResponse
from m5.serve.state import ModelHandle

router = APIRouter(prefix="/v1", tags=["model"])


@router.get("/model", response_model=ModelInfoResponse, summary="Trained model metadata")
async def model_info(handle: ModelHandle = Depends(get_model_handle)) -> ModelInfoResponse:
    m = handle.metadata
    return ModelInfoResponse(
        model_kind=m.model_kind,
        framework=m.framework,
        framework_version=m.framework_version,
        trained_at=m.trained_at,
        git_sha=m.git_sha,
        training_cutoff=m.training_cutoff,
        horizon_default=m.horizon_default,
        n_series=m.n_series,
        lags=m.lags,
        rolling_windows=m.rolling_windows,
        min_history_required=m.min_history_required,
        static_features=m.static_features,
    )
