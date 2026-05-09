"""Integration: TestClient against a live FastAPI app with a freshly-trained model.

Trains the LightGBM model on a 3-series toy fixture (≈1-2s on CPU), writes a
full artifact directory to ``tmp_path``, and exercises every public endpoint.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from m5.models.lgbm import DEFAULT_LAGS, DEFAULT_ROLLS, fit_lgbm

# -----------------------------------------------------------------------------
# Module-scoped fixtures so the LGBM fit + TestClient lifespan run once per file.
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _toy_long() -> pd.DataFrame:
    """3 series × 200 days with weekly seasonality — mirrors tests/conftest.py::toy_long."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    rows = []
    for sid in ("FOODS_1_001_CA_1", "FOODS_1_002_CA_1", "HOUSEHOLD_1_001_TX_1"):
        seasonal = 5 + 3 * np.sin(np.arange(200) * 2 * np.pi / 7)
        trend = np.linspace(0, 4, 200)
        noise = rng.normal(0, 1, 200)
        y = np.clip(seasonal + trend + noise, 0, None).astype(np.float32)
        for d, v in zip(dates, y, strict=True):
            rows.append(
                {
                    "unique_id": sid,
                    "ds": d,
                    "y": float(v),
                    "item_id": "_".join(sid.split("_")[:3]),
                    "dept_id": "_".join(sid.split("_")[:2]),
                    "cat_id": sid.split("_")[0],
                    "store_id": "_".join(sid.split("_")[3:5]),
                    "state_id": sid.split("_")[3],
                    "sell_price": 1.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def model_artifact(tmp_path_factory: pytest.TempPathFactory, _toy_long: pd.DataFrame) -> Path:
    """Fit a real MLForecast on toy data and write a complete artifact directory."""
    out = tmp_path_factory.mktemp("model_artifact")

    fcst = fit_lgbm(_toy_long)
    joblib.dump(fcst, out / "model.joblib", compress=3)

    _toy_long[["unique_id", "ds", "y"]].to_parquet(out / "history.parquet", index=False)

    static_cols = ["unique_id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    _toy_long.drop_duplicates("unique_id")[static_cols].to_parquet(out / "statics.parquet", index=False)

    metadata = {
        "model_kind": "lgbm",
        "framework": "mlforecast",
        "framework_version": "test",
        "lightgbm_version": "test",
        "trained_at": "20260101T000000Z",
        "git_sha": "test",
        "training_cutoff": str(_toy_long["ds"].max().date()),
        "freq": "D",
        "horizon_default": 7,
        "lags": list(DEFAULT_LAGS),
        "rolling_windows": list(DEFAULT_ROLLS),
        "n_series": int(_toy_long["unique_id"].nunique()),
        "n_rows": len(_toy_long),
        "min_history_required": max(DEFAULT_LAGS) + max(DEFAULT_ROLLS) - 1,
        "static_features": ["item_id", "dept_id", "cat_id", "store_id", "state_id"],
        "seed": 42,
    }
    (out / "metadata.json").write_text(json.dumps(metadata))
    return out


@pytest.fixture(scope="module")
def client(model_artifact: Path) -> Iterator[TestClient]:
    """Live TestClient — `with` triggers lifespan so the model is loaded."""
    from m5.serve import ServeSettings, create_app

    settings = ServeSettings(
        model_dir=model_artifact,
        max_horizon=14,
        max_series_per_request=10,
    )
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def auth_client(model_artifact: Path) -> Iterator[TestClient]:
    """A second client with API-key auth enabled — for the auth-path tests."""
    from m5.serve import ServeSettings, create_app

    settings = ServeSettings(model_dir=model_artifact, api_key="s3cret")
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


# -----------------------------------------------------------------------------
# Health / metadata
# -----------------------------------------------------------------------------


def test_root(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["service"]
    assert body["version"]


def test_healthz(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.text == "ok"


def test_readyz_after_load(client: TestClient) -> None:
    r = client.get("/readyz")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    assert "version" in body


def test_metrics_exposes_prometheus_text(client: TestClient) -> None:
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "m5_requests_total" in r.text
    assert "m5_request_latency_seconds" in r.text


def test_model_info(client: TestClient) -> None:
    r = client.get("/v1/model")
    assert r.status_code == 200
    body = r.json()
    assert body["model_kind"] == "lgbm"
    assert body["n_series"] == 3
    assert body["lags"] == list(DEFAULT_LAGS)
    assert "item_id" in body["static_features"]


def test_request_id_echo(client: TestClient) -> None:
    r = client.get("/healthz", headers={"X-Request-ID": "req-abc-123"})
    assert r.headers["X-Request-ID"] == "req-abc-123"


def test_request_id_generated_when_missing(client: TestClient) -> None:
    r = client.get("/healthz")
    rid = r.headers.get("X-Request-ID")
    assert rid is not None and len(rid) >= 8


# -----------------------------------------------------------------------------
# Stateful predict (predict/by-id)
# -----------------------------------------------------------------------------


def test_predict_by_id_happy_path(client: TestClient) -> None:
    r = client.post(
        "/v1/predict/by-id",
        json={"horizon": 7, "unique_ids": ["FOODS_1_001_CA_1"]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_series"] == 1
    assert body["horizon"] == 7
    assert len(body["forecasts"]) == 7
    for fc in body["forecasts"]:
        assert fc["unique_id"] == "FOODS_1_001_CA_1"
        assert isinstance(fc["y_hat"], float)


def test_predict_by_id_unknown_id_returns_400(client: TestClient) -> None:
    r = client.post(
        "/v1/predict/by-id",
        json={"horizon": 7, "unique_ids": ["DOES_NOT_EXIST"]},
    )
    assert r.status_code == 400
    body = r.json()
    assert body["status"] == 400
    assert "unknown" in body["detail"]


def test_predict_by_id_oversize_horizon_returns_422(client: TestClient) -> None:
    r = client.post(
        "/v1/predict/by-id",
        json={"horizon": 9999, "unique_ids": ["FOODS_1_001_CA_1"]},
    )
    assert r.status_code == 422
    assert "horizon" in r.json()["detail"]


def test_predict_by_id_empty_ids_returns_422(client: TestClient) -> None:
    """Pydantic min_length=1 fires before the route runs."""
    r = client.post("/v1/predict/by-id", json={"horizon": 7, "unique_ids": []})
    assert r.status_code == 422


def test_predict_by_id_too_many_series_returns_422(client: TestClient) -> None:
    """Defensive cap on series-per-request — server config is 10 in this fixture."""
    r = client.post(
        "/v1/predict/by-id",
        json={"horizon": 1, "unique_ids": [f"id_{i}" for i in range(20)]},
    )
    # 422 fires from the route's max_series_per_request check, not pydantic.
    assert r.status_code == 422


# -----------------------------------------------------------------------------
# Stateless predict
# -----------------------------------------------------------------------------


def test_predict_stateless_happy_path(client: TestClient, _toy_long: pd.DataFrame) -> None:
    sid = "FOODS_1_001_CA_1"
    sub = _toy_long[_toy_long["unique_id"] == sid].sort_values("ds")
    history = [
        {"unique_id": row.unique_id, "ds": row.ds.date().isoformat(), "y": float(row.y)}
        for row in sub.itertuples(index=False)
    ]
    r = client.post("/v1/predict", json={"horizon": 7, "history": history})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_series"] == 1
    assert len(body["forecasts"]) == 7


def test_predict_stateless_short_history_400(client: TestClient) -> None:
    history = [{"unique_id": "FOODS_1_001_CA_1", "ds": "2020-01-01", "y": 1.0}]
    r = client.post("/v1/predict", json={"horizon": 7, "history": history})
    assert r.status_code == 400
    assert "insufficient history" in r.json()["detail"]


def test_predict_stateless_unknown_id_400(client: TestClient, _toy_long: pd.DataFrame) -> None:
    """Unknown unique_id is rejected even when history shape is valid."""
    base = _toy_long[_toy_long["unique_id"] == "FOODS_1_001_CA_1"].sort_values("ds")
    history = [
        {"unique_id": "UNKNOWN_SERIES", "ds": row.ds.date().isoformat(), "y": float(row.y)}
        for row in base.itertuples(index=False)
    ]
    r = client.post("/v1/predict", json={"horizon": 7, "history": history})
    assert r.status_code == 400
    assert "unknown" in r.json()["detail"]


# -----------------------------------------------------------------------------
# Auth
# -----------------------------------------------------------------------------


def test_auth_required_when_key_set(auth_client: TestClient) -> None:
    r = auth_client.post("/v1/predict/by-id", json={"horizon": 1, "unique_ids": ["FOODS_1_001_CA_1"]})
    assert r.status_code == 401


def test_auth_accepts_correct_key(auth_client: TestClient) -> None:
    r = auth_client.post(
        "/v1/predict/by-id",
        json={"horizon": 1, "unique_ids": ["FOODS_1_001_CA_1"]},
        headers={"X-API-Key": "s3cret"},
    )
    assert r.status_code == 200


def test_auth_does_not_protect_health_endpoints(auth_client: TestClient) -> None:
    """Probes are intentionally unauthenticated — k8s won't carry credentials."""
    assert auth_client.get("/healthz").status_code == 200
    assert auth_client.get("/readyz").status_code == 200
    assert auth_client.get("/metrics").status_code == 200
