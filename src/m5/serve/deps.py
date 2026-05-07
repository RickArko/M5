"""FastAPI dependency providers.

Routes pull ``ServeSettings`` and ``ModelHandle`` via ``Depends(...)`` so they
are easily overrideable in tests.
"""

from __future__ import annotations

from fastapi import HTTPException, Request, status

from m5.serve.config import ServeSettings
from m5.serve.state import ModelHandle


def get_settings(request: Request) -> ServeSettings:
    return request.app.state.settings


def get_model_handle(request: Request) -> ModelHandle:
    handle: ModelHandle | None = getattr(request.app.state, "model_handle", None)
    if handle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded yet.",
        )
    return handle
