"""Unit: Settings dataclass + global seeding behave deterministically."""

from __future__ import annotations

import os
import random

import numpy as np

from m5.config import SETTINGS, Settings, set_global_seed


def test_settings_is_frozen() -> None:
    s = Settings()
    try:
        s.seed = 1  # type: ignore[misc]
    except Exception as exc:
        assert "frozen" in str(exc).lower() or isinstance(exc, AttributeError)
    else:
        raise AssertionError("Settings should be immutable")


def test_default_seed_is_42_unless_overridden(monkeypatch) -> None:
    monkeypatch.delenv("M5_SEED", raising=False)
    s = Settings()
    assert s.seed == 42


def test_env_overrides_take_effect(monkeypatch) -> None:
    monkeypatch.setenv("M5_SEED", "99")
    monkeypatch.setenv("M5_HORIZON", "7")
    s = Settings()
    assert s.seed == 99
    assert s.horizon == 7


def test_set_global_seed_seeds_python_and_numpy() -> None:
    set_global_seed(7)
    a_py, a_np = random.random(), np.random.rand()
    set_global_seed(7)
    b_py, b_np = random.random(), np.random.rand()
    assert a_py == b_py
    assert a_np == b_np
    assert os.environ["PYTHONHASHSEED"] == "7"


def test_artifacts_dir_under_repo_root() -> None:
    assert SETTINGS.artifacts_dir.name == "artifacts"
    assert SETTINGS.forecasts_dir.name == "forecasts"
