"""Request-ID middleware, JSON-logging toggle, and Prometheus metrics.

* ``X-Request-ID`` header is echoed back; missing header → server-generated UUIDv4.
* loguru records carry ``request_id`` via ``logger.contextualize``, so prod JSON
  logs include the id automatically.
* Prometheus counters/histograms cover request count, request latency, predict
  latency (just ``mlforecast.predict``, excluding validation), and series-points
  emitted. Scrape at ``GET /metrics``.
"""

from __future__ import annotations

import sys
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware

from m5.logging import logger as _logger

# ---------- Prometheus metrics ----------------------------------------
REQUESTS_TOTAL = Counter(
    "m5_requests_total",
    "Count of HTTP requests handled by the M5 forecaster service.",
    ["method", "path", "status"],
)
REQUEST_LATENCY = Histogram(
    "m5_request_latency_seconds",
    "End-to-end request latency (seconds) — middleware-observed.",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
PREDICT_LATENCY = Histogram(
    "m5_predict_latency_seconds",
    "Inference latency — mlforecast.predict only, excludes parsing / validation.",
    ["mode"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)
PREDICT_SERIES = Counter(
    "m5_predict_series_total",
    "Cumulative count of (series × horizon) forecast rows returned.",
    ["mode"],
)

REQUEST_ID_HEADER = "X-Request-ID"


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Attach an X-Request-ID to every request (echo upstream value if provided)."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        rid = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        request.state.request_id = rid
        with _logger.contextualize(request_id=rid):
            t0 = time.perf_counter()
            try:
                response = await call_next(request)
            except Exception:
                _logger.exception(
                    "Unhandled error in {method} {path}", method=request.method, path=request.url.path
                )
                raise
            elapsed = time.perf_counter() - t0
            response.headers[REQUEST_ID_HEADER] = rid
            REQUESTS_TOTAL.labels(request.method, request.url.path, str(response.status_code)).inc()
            REQUEST_LATENCY.labels(request.method, request.url.path).observe(elapsed)
            return response


def configure_logging(*, json_mode: bool, service_name: str) -> None:
    """Reconfigure loguru with either pretty (dev) or JSON (prod) sinks.

    Idempotent: safe to call multiple times — every call resets sinks via
    ``logger.remove()`` first.
    """
    _logger.remove()
    # Default extras so the format doesn't blow up before the request middleware fires.
    _logger.configure(extra={"request_id": "-", "service": service_name})

    if json_mode:
        # loguru's `serialize=True` emits one JSON object per record on stderr.
        _logger.add(
            sys.stderr,
            level="INFO",
            serialize=True,
            backtrace=False,
            diagnose=False,
        )
    else:
        _logger.add(
            sys.stderr,
            level="INFO",
            format=(
                "<green>{time:HH:mm:ss.SSS}</green> | "
                "<level>{level: <7}</level> | "
                "<cyan>{extra[request_id]:.8}</cyan> | "
                "<level>{message}</level>"
            ),
        )
