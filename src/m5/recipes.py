"""Declarative recipes — one YAML per (task, model) experiment.

A *recipe* is the single source of truth for every modelling decision: schema
(`task`), model family + hyperparams (`model`), and CV knobs (`cv`). It maps
1:1 to a YAML file under ``configs/`` and is loaded into a typed
:class:`Recipe` so the CLI and notebooks consume the same configuration.

Phase 1 surface (intentionally small, reversible):

* :class:`Recipe` — the top-level config.
* :func:`Recipe.from_yaml` — load a YAML file.
* :func:`build_forecaster` — construct an MLForecast / StatsForecast / hier
  pair from a recipe. Output matches the existing builders in
  ``m5.models.{lgbm,stats,hierarchical}`` byte-for-byte; equivalence is
  pinned by ``tests/unit/test_recipes.py``.

Runtime knobs (seed, paths) still come from :data:`m5.config.SETTINGS`. The
recipe captures *modelling* decisions; ``.env`` captures *run-time* knobs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from m5.config import SETTINGS

__all__ = [
    "CONFIGS_DIR",
    "HIER_RECIPE_PATH",
    "LGBM_RECIPE_PATH",
    "STATS_RECIPE_PATH",
    "CVRecipe",
    "HierForecaster",
    "HierRecipe",
    "LGBMRecipe",
    "LagSpec",
    "Recipe",
    "StatsModelSpec",
    "StatsRecipe",
    "TaskRecipe",
    "build_forecaster",
    "build_hier_base_from_recipe",
    "build_hier_reconcilers_from_recipe",
    "build_lgbm_from_recipe",
    "build_stats_from_recipe",
]


_REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = _REPO_ROOT / "configs"
LGBM_RECIPE_PATH = CONFIGS_DIR / "m5" / "lgbm.yaml"
STATS_RECIPE_PATH = CONFIGS_DIR / "m5" / "stats.yaml"
HIER_RECIPE_PATH = CONFIGS_DIR / "m5" / "hier.yaml"


# ----------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------
class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TaskRecipe(_Frozen):
    """Schema-level description of the prediction task."""

    name: str
    id_col: str = "unique_id"
    time_col: str = "ds"
    target_col: str = "y"
    freq: str = "D"
    horizon: int
    static_cols: list[str] = Field(default_factory=list)
    dynamic_cols: list[str] = Field(default_factory=list)
    drop_leading_zeros: bool = False


class LagSpec(_Frozen):
    """Lag and target-transform configuration for a global ML model."""

    lags: list[int] = Field(default_factory=list)
    # ``rolling_means_lagged[lag] = [windows]`` mirrors mlforecast's
    # ``lag_transforms={lag: [RollingMean(window_size=w) for w in windows]}``.
    rolling_means_lagged: dict[int, list[int]] = Field(default_factory=dict)
    differences: list[int] = Field(default_factory=list)


class LGBMRecipe(_Frozen):
    """LightGBM global model recipe (mlforecast)."""

    kind: Literal["lgbm"] = "lgbm"
    params: dict[str, Any] = Field(default_factory=dict)
    lags: LagSpec = Field(default_factory=LagSpec)
    date_features: list[str] = Field(default_factory=lambda: ["dayofweek", "day", "week", "month", "year"])


class StatsModelSpec(_Frozen):
    """One statsforecast model entry inside a stats bundle.

    Examples:
        ``StatsModelSpec(name="Theta", season_length=7, alias="Theta")``
        ``StatsModelSpec(name="AutoETS", season_length=7, model="ZNA", alias="AutoETS")``
    """

    model_config = ConfigDict(frozen=True, extra="allow")  # extras → constructor kwargs
    name: str
    alias: str | None = None


class StatsRecipe(_Frozen):
    """Statistical model bundle (statsforecast)."""

    kind: Literal["stats"] = "stats"
    models: list[StatsModelSpec]


ReconcilerName = Literal[
    "BU",
    # "TD_fp",  # Requires Strict Hierarchical Tree for nixtla
    "MinT_OLS",
    "MinT_WLS_struct",
    "MinT_WLS_var",
    "MinT_shrink",
    "ERM_closed",
    "ERM_reg_bu",
]


class HierRecipe(_Frozen):
    """Hierarchical reconciliation recipe."""

    kind: Literal["hier"] = "hier"
    base_model: StatsModelSpec
    reconcilers: list[ReconcilerName]


class CVRecipe(_Frozen):
    n_windows: int = 3
    step_size: int | None = None  # None ⇒ default to horizon


class Recipe(_Frozen):
    """Top-level config for one (task, model) experiment.

    Loaded from a YAML file under ``configs/``. The ``model`` field is a
    discriminated union keyed by ``kind`` (``lgbm`` | ``stats`` | ``hier``).
    """

    task: TaskRecipe
    model: LGBMRecipe | StatsRecipe | HierRecipe = Field(discriminator="kind")
    cv: CVRecipe = Field(default_factory=CVRecipe)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Recipe:
        """Parse a YAML file into a validated Recipe."""
        raw = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(raw)


# ----------------------------------------------------------------------
# Builders — recipe → forecaster
# ----------------------------------------------------------------------
def build_lgbm_from_recipe(
    recipe: Recipe,
    *,
    seed: int | None = None,
    n_jobs: int = -1,
) -> Any:
    """Build an MLForecast (LightGBM) from a recipe.

    Output matches :func:`m5.models.lgbm.build_lgbm_forecaster` exactly when
    ``configs/m5/lgbm.yaml`` is loaded. Seed is injected from
    :data:`SETTINGS.seed` so ``M5_SEED`` env override still works; pass
    ``seed=...`` to override per-call.
    """
    if not isinstance(recipe.model, LGBMRecipe):
        raise TypeError(f"build_lgbm_from_recipe: recipe.model.kind={recipe.model.kind!r}, expected 'lgbm'")

    import lightgbm as lgb
    from mlforecast import MLForecast
    from mlforecast.lag_transforms import RollingMean
    from mlforecast.target_transforms import Differences

    seed_val = SETTINGS.seed if seed is None else seed
    params = {**recipe.model.params, "seed": seed_val}

    lag_transforms: dict[int, list[Any]] = {
        lag: [RollingMean(window_size=w) for w in windows]
        for lag, windows in recipe.model.lags.rolling_means_lagged.items()
    }

    target_transforms: list[Any] | None = None
    if recipe.model.lags.differences:
        target_transforms = [Differences(recipe.model.lags.differences)]

    model = lgb.LGBMRegressor(**params, n_jobs=n_jobs)
    return MLForecast(
        models={"LGBM": model},
        freq=recipe.task.freq,
        lags=list(recipe.model.lags.lags),
        lag_transforms=lag_transforms,
        date_features=list(recipe.model.date_features),
        target_transforms=target_transforms,
        num_threads=n_jobs,
    )


_STATS_MODEL_REGISTRY = {
    "Theta": "statsforecast.models:Theta",
    "AutoETS": "statsforecast.models:AutoETS",
    "SeasonalNaive": "statsforecast.models:SeasonalNaive",
    "AutoARIMA": "statsforecast.models:AutoARIMA",
    "AutoCES": "statsforecast.models:AutoCES",
    "Naive": "statsforecast.models:Naive",
}


def _resolve_stats_class(name: str) -> Any:
    if name not in _STATS_MODEL_REGISTRY:
        raise ValueError(f"Unknown stats model {name!r}. Known: {sorted(_STATS_MODEL_REGISTRY)}")
    mod_path, cls_name = _STATS_MODEL_REGISTRY[name].split(":")
    import importlib

    return getattr(importlib.import_module(mod_path), cls_name)


def _instantiate_stats_model(spec: StatsModelSpec) -> Any:
    cls = _resolve_stats_class(spec.name)
    kwargs = spec.model_dump(exclude={"name"})
    if kwargs.get("alias") is None:
        kwargs.pop("alias", None)
    return cls(**kwargs)


def build_stats_from_recipe(recipe: Recipe, *, n_jobs: int = -1) -> Any:
    """Build a StatsForecast bundle from a recipe.

    Output matches :func:`m5.models.stats.build_stats_forecaster` exactly when
    ``configs/m5/stats.yaml`` is loaded.
    """
    if not isinstance(recipe.model, StatsRecipe):
        raise TypeError(f"build_stats_from_recipe: recipe.model.kind={recipe.model.kind!r}, expected 'stats'")
    from statsforecast import StatsForecast

    models = [_instantiate_stats_model(m) for m in recipe.model.models]
    return StatsForecast(models=models, freq=recipe.task.freq, n_jobs=n_jobs)


_RECONCILER_BUILDERS: dict[str, Any] = {
    "BU": lambda: _bottom_up(),
    "MinT_OLS": lambda: _mintrace("ols"),
    "MinT_WLS_struct": lambda: _mintrace("wls_struct"),
    "MinT_WLS_var": lambda: _mintrace("wls_var"),
    "MinT_shrink": lambda: _mintrace("mint_shrink"),
    "ERM_closed": lambda: _erm("closed"),
    "ERM_reg_bu": lambda: _erm("reg_bu"),
}


def _bottom_up() -> Any:
    from hierarchicalforecast.methods import BottomUp

    return BottomUp()


def _mintrace(method: str) -> Any:
    from hierarchicalforecast.methods import MinTrace

    return MinTrace(method=method)


def _erm(method: str, lambda_reg: float = 0.01) -> Any:
    from hierarchicalforecast.methods import ERM

    return ERM(method=method, lambda_reg=lambda_reg)


def build_hier_base_from_recipe(recipe: Recipe, *, n_jobs: int = -1) -> Any:
    """Build the base StatsForecast learner used at every hierarchy level."""
    if not isinstance(recipe.model, HierRecipe):
        raise TypeError(
            f"build_hier_base_from_recipe: recipe.model.kind={recipe.model.kind!r}, expected 'hier'"
        )
    from statsforecast import StatsForecast

    base = _instantiate_stats_model(recipe.model.base_model)
    return StatsForecast(models=[base], freq=recipe.task.freq, n_jobs=n_jobs)


def build_hier_reconcilers_from_recipe(recipe: Recipe) -> list[Any]:
    """Build the reconciler list for a hierarchical recipe."""
    if not isinstance(recipe.model, HierRecipe):
        raise TypeError(
            f"build_hier_reconcilers_from_recipe: recipe.model.kind={recipe.model.kind!r}, expected 'hier'"
        )
    return [_RECONCILER_BUILDERS[name]() for name in recipe.model.reconcilers]


@dataclass(frozen=True)
class HierForecaster:
    """Bundle returned by :func:`build_forecaster` for hierarchical recipes."""

    base: Any  # StatsForecast
    reconcilers: list[Any]


def build_forecaster(recipe: Recipe, *, seed: int | None = None, n_jobs: int = -1) -> Any:
    """Dispatch on ``recipe.model.kind`` and return the right forecaster.

    Returns:
        - ``MLForecast``           for ``kind='lgbm'``
        - ``StatsForecast``        for ``kind='stats'``
        - :class:`HierForecaster`  for ``kind='hier'`` (base + reconcilers)
    """
    kind = recipe.model.kind
    if kind == "lgbm":
        return build_lgbm_from_recipe(recipe, seed=seed, n_jobs=n_jobs)
    if kind == "stats":
        return build_stats_from_recipe(recipe, n_jobs=n_jobs)
    if kind == "hier":
        return HierForecaster(
            base=build_hier_base_from_recipe(recipe, n_jobs=n_jobs),
            reconcilers=build_hier_reconcilers_from_recipe(recipe),
        )
    raise ValueError(f"Unknown recipe.model.kind={kind!r}")
