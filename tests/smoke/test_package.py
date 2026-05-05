"""Smoke: package metadata and Settings are sane."""

from __future__ import annotations

from pathlib import Path

from m5 import __version__
from m5.config import REPO_ROOT, SETTINGS, set_global_seed


def test_version_is_pep440_ish() -> None:
    assert __version__
    parts = __version__.split(".")
    assert all(p[0].isdigit() or p.startswith("0") for p in parts[:1])


def test_settings_paths_are_pathlike() -> None:
    assert isinstance(SETTINGS.data_dir, Path)
    assert isinstance(SETTINGS.raw_dir, Path)
    assert isinstance(SETTINGS.processed_dir, Path)
    assert SETTINGS.horizon == 28 or SETTINGS.horizon > 0
    assert SETTINGS.n_windows >= 1


def test_repo_root_points_at_pyproject() -> None:
    assert (REPO_ROOT / "pyproject.toml").exists()


def test_set_global_seed_returns_seed() -> None:
    seed = set_global_seed(123)
    assert seed == 123
