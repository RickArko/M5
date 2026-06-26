"""Smoke: every public surface must import without side-effects."""

from __future__ import annotations

import importlib

import pytest

PUBLIC_MODULES = [
    "m5",
    "m5.config",
    "m5.data",
    "m5.features",
    "m5.evaluation",
    "m5.cv",
    "m5.cli",
    "m5.logging",
    "m5.plots",
    "m5.models",
    "m5.models.stats",
    "m5.models.lgbm",
]


@pytest.mark.parametrize("module", PUBLIC_MODULES)
def test_module_importable(module: str) -> None:
    importlib.import_module(module)


def test_top_level_exports_version() -> None:
    import m5

    assert isinstance(m5.__version__, str)
    assert m5.__version__  # non-empty
