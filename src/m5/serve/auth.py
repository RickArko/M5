"""Optional ``X-API-Key`` auth — opt-in via ``M5_SERVE_API_KEY``.

When the env var is empty/unset, auth is disabled and the dependency is a
no-op (intended for dev / behind-VPC deployments). When set, requests must
include ``X-API-Key: <secret>`` or get a 401.
"""

from __future__ import annotations

import hmac
from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import Header, HTTPException, status

from m5.serve.config import ServeSettings


def make_api_key_dependency(
    settings: ServeSettings,
) -> Callable[..., Coroutine[Any, Any, None]]:
    """Build an ``X-API-Key`` FastAPI dependency from settings.

    Returns a coroutine to be used with ``Depends(...)``. Constant-time string
    comparison (:func:`hmac.compare_digest`) defends against timing oracles.
    """
    expected = settings.api_key

    async def _dep(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
        if not expected:
            return  # auth disabled
        if x_api_key is None or not hmac.compare_digest(x_api_key.encode(), expected.encode()):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid X-API-Key.",
                headers={"WWW-Authenticate": "ApiKey"},
            )

    return _dep
