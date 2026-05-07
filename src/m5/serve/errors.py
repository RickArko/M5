"""Exception → ProblemDetails JSON mapper.

We translate every raised error into RFC 7807 ``application/problem+json`` so
clients have one stable error shape across validation, business-logic, and
unhandled paths. This also prevents leaking framework tracebacks in 500s.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette import status

from m5.logging import logger
from m5.serve.schemas import ProblemDetails

PROBLEM_MEDIA = "application/problem+json"


def _problem(
    *,
    status_code: int,
    title: str,
    detail: str | None = None,
    instance: str | None = None,
    type_: str = "about:blank",
) -> JSONResponse:
    body = ProblemDetails(
        type=type_,
        title=title,
        status=status_code,
        detail=detail,
        instance=instance,
    ).model_dump()
    return JSONResponse(body, status_code=status_code, media_type=PROBLEM_MEDIA)


def install_handlers(app: FastAPI) -> None:
    """Register HTTP / validation / catch-all handlers on the given app."""

    @app.exception_handler(HTTPException)
    async def _http_exc(request: Request, exc: HTTPException) -> JSONResponse:
        return _problem(
            status_code=exc.status_code,
            title=_default_title(exc.status_code),
            detail=str(exc.detail) if exc.detail else None,
            instance=str(request.url),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_exc(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Compact summary of the first few errors — enough to debug, doesn't dump 100 lines.
        errors = exc.errors()[:5]
        detail = "; ".join(f"{'/'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in errors)
        return _problem(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            title="Validation failed",
            detail=detail,
            instance=str(request.url),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error: {!r}", exc)
        return _problem(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            title="Internal server error",
            detail="An unexpected error occurred.",
            instance=str(request.url),
        )


def _default_title(code: int) -> str:
    if 400 <= code < 500:
        return "Client error"
    if 500 <= code < 600:
        return "Server error"
    return "HTTP error"
