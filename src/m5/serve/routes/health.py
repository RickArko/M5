"""Liveness, readiness, and root identity endpoints.

These are intentionally unauthenticated — k8s probes hit them without credentials.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse, PlainTextResponse

from m5 import __version__ as PKG_VERSION

router = APIRouter(tags=["meta"])


@router.get("/", summary="Service identity")
async def root(request: Request) -> dict[str, str]:
    settings = request.app.state.settings
    return {"service": settings.service_name, "version": PKG_VERSION}


@router.get("/healthz", summary="Liveness probe")
async def healthz() -> PlainTextResponse:
    """Always 200 if the process is up — probe target for k8s liveness."""
    return PlainTextResponse("ok")


@router.get("/readyz", summary="Readiness probe")
async def readyz(request: Request) -> JSONResponse:
    """200 only after the model artifact has loaded successfully."""
    handle = getattr(request.app.state, "model_handle", None)
    if handle is None:
        return JSONResponse(
            {"ready": False, "reason": "model not loaded"},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    return JSONResponse({"ready": True, "version": handle.metadata.git_sha})
