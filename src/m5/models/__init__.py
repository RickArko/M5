"""Forecast models for the M5 task.

Imports are lazy so ``from m5.models.toto import ...`` does not pull in
LightGBM / statsforecast (OpenMP + Torch clash on Darwin otherwise).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "build_hier_base_forecaster",
    "build_hier_reconcilers",
    "build_lgbm_forecaster",
    "build_stats_forecaster",
    "fit_lgbm",
    "fit_predict_hier",
    "fit_predict_lgbm",
    "fit_predict_segmented",
    "fit_predict_stats",
    "fit_predict_store",
    "fit_predict_store_cat",
    "fit_predict_store_dept",
    "segmented_cv",
    "store_cat_cv",
    "store_cv",
    "store_dept_cv",
    "toto_cv",
    "toto_forecast",
]

_LAZY: dict[str, tuple[str, str]] = {
    "build_hier_base_forecaster": ("m5.models.hierarchical", "build_hier_base_forecaster"),
    "build_hier_reconcilers": ("m5.models.hierarchical", "build_hier_reconcilers"),
    "fit_predict_hier": ("m5.models.hierarchical", "fit_predict_hier"),
    "build_lgbm_forecaster": ("m5.models.lgbm", "build_lgbm_forecaster"),
    "fit_lgbm": ("m5.models.lgbm", "fit_lgbm"),
    "fit_predict_lgbm": ("m5.models.lgbm", "fit_predict_lgbm"),
    "fit_predict_segmented": ("m5.models.segmented", "fit_predict_segmented"),
    "fit_predict_store": ("m5.models.segmented", "fit_predict_store"),
    "fit_predict_store_cat": ("m5.models.segmented", "fit_predict_store_cat"),
    "fit_predict_store_dept": ("m5.models.segmented", "fit_predict_store_dept"),
    "segmented_cv": ("m5.models.segmented", "segmented_cv"),
    "store_cat_cv": ("m5.models.segmented", "store_cat_cv"),
    "store_cv": ("m5.models.segmented", "store_cv"),
    "store_dept_cv": ("m5.models.segmented", "store_dept_cv"),
    "build_stats_forecaster": ("m5.models.stats", "build_stats_forecaster"),
    "fit_predict_stats": ("m5.models.stats", "fit_predict_stats"),
    "toto_cv": ("m5.models.toto", "toto_cv"),
    "toto_forecast": ("m5.models.toto", "toto_forecast"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr = _LAZY[name]
    from importlib import import_module

    value = getattr(import_module(mod_name), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
