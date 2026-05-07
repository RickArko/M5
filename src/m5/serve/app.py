"""FastAPI app factory + lifespan model loader.

The factory is the only public construct — uvicorn's ``--factory`` flag calls
it once per process; tests pass a custom :class:`ServeSettings` to point at a
toy artifact directory.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from m5 import __version__ as PKG_VERSION
from m5.logging import logger
from m5.serve.auth import make_api_key_dependency
from m5.serve.config import ServeSettings
from m5.serve.errors import install_handlers
from m5.serve.observability import RequestIdMiddleware, configure_logging
from m5.serve.routes import health, metadata, predict
from m5.serve.state import ModelHandle


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: ServeSettings = app.state.settings
    configure_logging(json_mode=settings.log_json, service_name=settings.service_name)
    logger.info(f"startup: loading model from {settings.model_dir}")
    try:
        app.state.model_handle = ModelHandle(settings.model_dir)
    except Exception as exc:
        # A missing/broken artifact must not crash the process — readiness will
        # fail and operators get visibility via /readyz + logs without a restart loop.
        logger.exception(f"startup: model load failed — service NOT READY ({exc})")
        app.state.model_handle = None
    yield
    logger.info("shutdown: releasing model handle")
    app.state.model_handle = None


def create_app(settings: ServeSettings | None = None) -> FastAPI:
    """Construct the FastAPI app. ``settings`` defaults to env-driven ServeSettings()."""
    settings = settings or ServeSettings()

    app = FastAPI(
        title="M5 Forecaster",
        version=PKG_VERSION,
        summary="LightGBM-backed daily forecaster trained on the M5 dataset.",
        description=(
            "Serves a fitted Nixtla MLForecast artifact via stateless and stateful "
            "predict endpoints. See `m5 train` for the artifact contract."
        ),
        lifespan=_lifespan,
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )
    app.state.settings = settings

    # Middleware (outermost first)
    app.add_middleware(RequestIdMiddleware)

    # Error handlers
    install_handlers(app)

    # Public meta routes (no auth) — health + readiness + identity
    app.include_router(health.router)

    # /metrics — Prometheus exposition. Lock down at the network layer (NetworkPolicy /
    # ingress) rather than via API key, since scrape clients rarely send custom headers.
    @app.get("/metrics", include_in_schema=False)
    async def metrics_endpoint() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # Authenticated routes (no-op when M5_SERVE_API_KEY is empty)
    auth_dep = make_api_key_dependency(settings)
    secured = APIRouter(dependencies=[Depends(auth_dep)])
    secured.include_router(metadata.router)
    secured.include_router(predict.router)
    app.include_router(secured)

    return app
