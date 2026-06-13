"""Service configuration. Env vars prefixed ``M5_SERVE_``.

Read once at app startup and stored on ``app.state.settings``. Mirrors the
env-var convention of :class:`m5.config.Settings` so the same ``.env`` drives
both training and serving.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from m5.config import REPO_ROOT


class ServeSettings(BaseSettings):
    """Runtime knobs for the FastAPI service."""

    # ``protected_namespaces=()`` lets us use ``model_*`` field names without
    # tripping pydantic's default warning (pydantic v2 reserves the ``model_``
    # prefix on BaseModel for its own descriptors).
    model_config = SettingsConfigDict(
        env_prefix="M5_SERVE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        protected_namespaces=(),
    )

    # --- Artifact ---------------------------------------------------
    model_dir: Path = Field(
        default_factory=lambda: REPO_ROOT / "artifacts" / "models" / "lgbm" / "latest",
        description="Directory containing model.joblib + metadata.json + history.parquet + statics.parquet.",
    )
    model_kind: str = Field(
        default="lgbm",
        description="Model kind: 'lgbm' (MLForecast artifact) or 'toto' (zero-shot TOTO — experimental).",
    )

    # --- Bind -------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=1, ge=1)

    # --- Request limits --------------------------------------------
    # All defensive caps — the validator stages reject oversize payloads
    # before any pandas / mlforecast work happens.
    max_horizon: int = Field(default=56, ge=1)
    max_series_per_request: int = Field(default=5000, ge=1)
    max_history_points: int = Field(default=2_000_000, ge=1)

    # --- Auth -------------------------------------------------------
    api_key: str = ""  # empty = no auth (dev mode)

    # --- Observability ---------------------------------------------
    log_json: bool = False
    service_name: str = "m5-forecaster"

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_key)
