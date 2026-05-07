"""Smoke: serve package imports and the app factory works without a model.

The factory must succeed even when ``model_dir`` is missing — the lifespan
absorbs the load failure and ``/readyz`` reports unready. This mirrors the
real-ops pattern where a sidecar / volume mount appears after the pod starts.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_serve_module_imports() -> None:
    import m5.serve  # noqa: F401
    from m5.serve import ServeSettings, create_app  # noqa: F401


def test_create_app_with_missing_model_dir(tmp_path: Path) -> None:
    from m5.serve import ServeSettings, create_app

    settings = ServeSettings(model_dir=tmp_path / "does-not-exist")
    app = create_app(settings)
    # FastAPI metadata
    assert app.title == "M5 Forecaster"
    # All four route families wired
    paths = {getattr(r, "path", None) for r in app.routes}
    for required in ("/healthz", "/readyz", "/metrics", "/v1/model", "/v1/predict", "/v1/predict/by-id"):
        assert required in paths, f"missing route: {required}"


def test_create_app_accepts_default_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calling create_app() with no args must not raise — env defaults are usable."""
    monkeypatch.setenv("M5_SERVE_MODEL_DIR", "/nonexistent")
    from m5.serve import create_app

    app = create_app()
    assert app is not None
