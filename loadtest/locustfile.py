"""Locust scenarios for the M5 prediction API.

Run locally against a running ``m5 serve`` instance:

    M5_LOADTEST_PAYLOAD=loadtest/payloads/unique_ids.txt \\
        uv run --group loadtest locust -f loadtest/locustfile.py --headless \\
        -u 20 -r 5 -H http://localhost:8000 --run-time 60s

Scenarios (weights):

* ``predict_single`` (80) — POST ``/v1/predict/by-id`` with 1 unique_id.
* ``predict_batch_10`` (15) — POST ``/v1/predict/by-id`` with 10 unique_ids.
* ``healthz`` (5) — GET ``/healthz``.

Environment:

* ``M5_LOADTEST_PAYLOAD`` (required) — path to a newline-delimited file of unique_ids,
  one per line. Generate via ``make loadtest-payload`` once.
* ``M5_LOADTEST_HORIZON`` (default ``28``) — forecast horizon per request.
* ``M5_SERVE_API_KEY`` (optional) — if set, sent as ``X-API-Key`` on every request.
* ``M5_LOADTEST_BATCH_N`` (default ``10``) — series-per-request for the batch task.

Failures: any non-2xx response is recorded as a Locust failure (RFC 7807 detail
captured if available).
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

from locust import HttpUser, between, events, task

# --- corpus loaded once at process start -----------------------------------

_PAYLOAD_PATH_ENV = "M5_LOADTEST_PAYLOAD"
_HORIZON_ENV = "M5_LOADTEST_HORIZON"
_API_KEY_ENV = "M5_SERVE_API_KEY"
_BATCH_N_ENV = "M5_LOADTEST_BATCH_N"

_UNIQUE_IDS: list[str] = []
_HORIZON = 28
_BATCH_N = 10


@events.init.add_listener
def _load_corpus(environment, **_kwargs):
    """Read unique_ids from the payload file once per worker process."""
    global _UNIQUE_IDS, _HORIZON, _BATCH_N

    payload_path = os.environ.get(_PAYLOAD_PATH_ENV)
    if not payload_path:
        raise RuntimeError(
            f"{_PAYLOAD_PATH_ENV} not set — point it at a newline-delimited "
            "unique_ids file (generate with `make loadtest-payload`)."
        )
    p = Path(payload_path)
    if not p.exists():
        raise RuntimeError(f"{_PAYLOAD_PATH_ENV}={payload_path} does not exist.")

    _UNIQUE_IDS = [line.strip() for line in p.read_text().splitlines() if line.strip()]
    if not _UNIQUE_IDS:
        raise RuntimeError(f"{p} is empty — no unique_ids to test against.")

    _HORIZON = int(os.environ.get(_HORIZON_ENV, "28"))
    _BATCH_N = int(os.environ.get(_BATCH_N_ENV, "10"))

    print(
        f"[loadtest] corpus loaded: {len(_UNIQUE_IDS):,d} unique_ids "
        f"from {p}; horizon={_HORIZON}; batch_n={_BATCH_N}"
    )


# --- user class ------------------------------------------------------------


class M5User(HttpUser):
    """Simulates an API consumer hitting the prediction endpoints."""

    wait_time = between(0.1, 0.5)
    abstract = False

    def on_start(self) -> None:
        api_key = os.environ.get(_API_KEY_ENV)
        if api_key:
            self.client.headers.update({"X-API-Key": api_key})

    @task(80)
    def predict_single(self) -> None:
        uid = random.choice(_UNIQUE_IDS)
        self._post_predict([uid], "predict_single")

    @task(15)
    def predict_batch_10(self) -> None:
        n = min(_BATCH_N, len(_UNIQUE_IDS))
        ids = random.sample(_UNIQUE_IDS, n)
        self._post_predict(ids, f"predict_batch_{n}")

    @task(5)
    def healthz(self) -> None:
        with self.client.get("/healthz", name="healthz", catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"healthz {resp.status_code}: {resp.text[:200]}")

    def _post_predict(self, unique_ids: list[str], name: str) -> None:
        body = {"horizon": _HORIZON, "unique_ids": unique_ids}
        with self.client.post(
            "/v1/predict/by-id",
            json=body,
            name=name,
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                detail = _extract_problem(resp.text)
                resp.failure(f"{resp.status_code}: {detail}")
                return
            try:
                payload = resp.json()
            except json.JSONDecodeError:
                resp.failure("response was not JSON")
                return
            expected = len(unique_ids) * _HORIZON
            got = len(payload.get("forecasts", []))
            if got != expected:
                resp.failure(f"forecast count mismatch: expected {expected}, got {got}")


# --- helpers ---------------------------------------------------------------


def _extract_problem(text: str) -> str:
    """Pull RFC 7807 ``detail`` out of a problem-details body; fall back to raw text."""
    try:
        body = json.loads(text)
    except json.JSONDecodeError:
        return text[:200]
    return str(body.get("detail") or body.get("title") or text)[:200]
