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
    "m5.legacy",
    "m5.models",
    "m5.models.stats",
    "m5.models.lgbm",
]


def test_legacy_shim_exposes_old_names() -> None:
    """The old EDA notebooks rely on these — guard against accidental rename."""
    from m5 import legacy

    expected = {
        "id_col",
        "time_col",
        "id_cols",
        "PATH_INPUT",
        "TRAIN_PARQUET_PATH",
        "load_calendar",
        "load_prices",
        "load_sales",
        "load_train_parquet",
        "filter_data",
        "create_m5_fit_data",
        "reduce_mem_usage",
        "create_future_features",
        "get_dfids",
    }
    missing = expected - set(dir(legacy))
    assert not missing, f"m5.legacy missing: {missing}"


@pytest.mark.parametrize("module", PUBLIC_MODULES)
def test_module_importable(module: str) -> None:
    importlib.import_module(module)


def test_top_level_exports_version() -> None:
    import m5

    assert isinstance(m5.__version__, str)
    assert m5.__version__  # non-empty
