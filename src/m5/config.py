"""Centralised paths, seeds, and run-time settings.

Settings are read from environment variables (with `.env` support) and
exposed as a frozen dataclass so every module gets the same view.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv(override=False)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    return int(raw) if raw is not None and raw != "" else default


def _env_path(key: str, default: Path) -> Path:
    raw = os.getenv(key)
    return Path(raw).expanduser() if raw else default


@dataclass(frozen=True)
class Settings:
    """Run-time configuration. Override via env vars (see `.env.example`)."""

    seed: int = field(default_factory=lambda: _env_int("M5_SEED", 42))
    horizon: int = field(default_factory=lambda: _env_int("M5_HORIZON", 28))
    n_windows: int = field(default_factory=lambda: _env_int("M5_N_WINDOWS", 3))
    last_n_days: int = field(default_factory=lambda: _env_int("M5_LAST_N_DAYS", 400))
    n_series: int = field(default_factory=lambda: _env_int("M5_N_SERIES", -1))
    data_dir: Path = field(default_factory=lambda: _env_path("DATA_DIR", REPO_ROOT / "data"))

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "m5" / "datasets"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def artifacts_dir(self) -> Path:
        return REPO_ROOT / "artifacts"

    @property
    def forecasts_dir(self) -> Path:
        return REPO_ROOT / "forecasts"

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.processed_dir, self.artifacts_dir, self.forecasts_dir):
            p.mkdir(parents=True, exist_ok=True)


SETTINGS = Settings()


def set_global_seed(seed: int | None = None) -> int:
    """Seed Python, NumPy, and (if importable) LightGBM/PyTorch for reproducibility."""
    s = SETTINGS.seed if seed is None else seed
    random.seed(s)
    np.random.seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)
    return s
