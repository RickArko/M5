"""Recipe ↔ existing-builder equivalence tests.

The point of these tests is reversibility: as long as they pass, the YAML
recipes are *known* to produce the same forecasters as the hand-coded
``m5.models.{lgbm,stats,hierarchical}`` builders. Phase 2 will reroute the
existing builders through these recipes; if equivalence ever drifts, these
tests catch it before behaviour silently diverges.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from m5.config import SETTINGS
from m5.models.hierarchical import (
    build_hier_base_forecaster,
    build_hier_reconcilers,
)
from m5.models.lgbm import (
    DEFAULT_LAGS,
    DEFAULT_ROLLS,
    build_lgbm_forecaster,
    lgbm_params,
)
from m5.models.stats import build_stats_forecaster
from m5.recipes import (
    HierForecaster,
    HierRecipe,
    LGBMRecipe,
    Recipe,
    StatsRecipe,
    build_forecaster,
    build_hier_base_from_recipe,
    build_hier_reconcilers_from_recipe,
    build_lgbm_from_recipe,
    build_stats_from_recipe,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "configs" / "m5"


@pytest.fixture(scope="module")
def lgbm_recipe() -> Recipe:
    return Recipe.from_yaml(CONFIGS / "lgbm.yaml")


@pytest.fixture(scope="module")
def stats_recipe() -> Recipe:
    return Recipe.from_yaml(CONFIGS / "stats.yaml")


@pytest.fixture(scope="module")
def hier_recipe() -> Recipe:
    return Recipe.from_yaml(CONFIGS / "hier.yaml")


@pytest.fixture(scope="module")
def hier_experiments_recipe() -> Recipe:
    return Recipe.from_yaml(CONFIGS / "hier_experiments.yaml")


# ----------------------------------------------------------------------
# Schema / loading
# ----------------------------------------------------------------------
def test_lgbm_yaml_loads_as_lgbm_recipe(lgbm_recipe: Recipe) -> None:
    assert isinstance(lgbm_recipe.model, LGBMRecipe)
    assert lgbm_recipe.model.kind == "lgbm"
    assert lgbm_recipe.task.name == "m5"
    assert lgbm_recipe.task.horizon == SETTINGS.horizon


def test_stats_yaml_loads_as_stats_recipe(stats_recipe: Recipe) -> None:
    assert isinstance(stats_recipe.model, StatsRecipe)
    assert stats_recipe.model.kind == "stats"
    assert {m.name for m in stats_recipe.model.models} == {"Theta", "AutoETS", "SeasonalNaive"}


def test_hier_yaml_loads_as_hier_recipe(hier_recipe: Recipe) -> None:
    assert isinstance(hier_recipe.model, HierRecipe)
    assert hier_recipe.model.kind == "hier"
    assert hier_recipe.model.base_model.name == "Theta"
    assert hier_recipe.model.reconcilers == ["BU", "MinT_OLS", "MinT_shrink"]


def test_hier_experiments_yaml_loads_expanded_reconcilers(hier_experiments_recipe: Recipe) -> None:
    assert isinstance(hier_experiments_recipe.model, HierRecipe)
    assert hier_experiments_recipe.model.reconcilers == [
        "BU",
        "MinT_OLS",
        "MinT_WLS_struct",
        "MinT_WLS_var",
        "MinT_shrink",
        "ERM_closed",
        "ERM_reg_bu",
    ]


def test_recipe_is_frozen(lgbm_recipe: Recipe) -> None:
    with pytest.raises(Exception):  # pydantic v2 raises ValidationError on frozen mutation
        lgbm_recipe.task.horizon = 42  # type: ignore[misc]


# ----------------------------------------------------------------------
# LGBM equivalence
# ----------------------------------------------------------------------
def test_lgbm_recipe_lags_match_defaults(lgbm_recipe: Recipe) -> None:
    assert tuple(lgbm_recipe.model.lags.lags) == DEFAULT_LAGS
    assert tuple(lgbm_recipe.model.lags.rolling_means_lagged[1]) == DEFAULT_ROLLS


def test_lgbm_recipe_params_match_lgbm_params(lgbm_recipe: Recipe) -> None:
    """Recipe params + injected seed should equal the canonical lgbm_params dict."""
    recipe_params = {**lgbm_recipe.model.params, "seed": SETTINGS.seed}
    canonical = lgbm_params()
    assert recipe_params == canonical


def test_lgbm_recipe_builds_equivalent_forecaster(lgbm_recipe: Recipe) -> None:
    """Recipe-built MLForecast has the same model params + lags/rolls/freq as the builder."""
    a = build_lgbm_forecaster()
    b = build_lgbm_from_recipe(lgbm_recipe)

    assert a.models["LGBM"].get_params() == b.models["LGBM"].get_params()
    assert a.ts.freq == b.ts.freq
    assert list(a.ts.lags) == list(b.ts.lags)
    # lag_transforms: dict[int, list[Callable]] — compare keys + window sizes
    assert sorted(a.ts.lag_transforms.keys()) == sorted(b.ts.lag_transforms.keys())
    for lag in a.ts.lag_transforms:
        a_windows = [getattr(t, "window_size", None) for t in a.ts.lag_transforms[lag]]
        b_windows = [getattr(t, "window_size", None) for t in b.ts.lag_transforms[lag]]
        assert a_windows == b_windows
    assert list(a.ts.date_features) == list(b.ts.date_features)


# ----------------------------------------------------------------------
# Stats equivalence
# ----------------------------------------------------------------------
def test_stats_recipe_builds_equivalent_bundle(stats_recipe: Recipe) -> None:
    a = build_stats_forecaster()
    b = build_stats_from_recipe(stats_recipe)

    assert a.freq == b.freq
    a_aliases = [m.alias for m in a.models]
    b_aliases = [m.alias for m in b.models]
    assert a_aliases == b_aliases
    a_types = [type(m).__name__ for m in a.models]
    b_types = [type(m).__name__ for m in b.models]
    assert a_types == b_types
    # Verify season_length lined up
    a_seasons = [getattr(m, "season_length", None) for m in a.models]
    b_seasons = [getattr(m, "season_length", None) for m in b.models]
    assert a_seasons == b_seasons


# ----------------------------------------------------------------------
# Hier equivalence
# ----------------------------------------------------------------------
def test_hier_recipe_base_matches_builder(hier_recipe: Recipe) -> None:
    a = build_hier_base_forecaster()
    b = build_hier_base_from_recipe(hier_recipe)
    assert a.freq == b.freq
    assert [type(m).__name__ for m in a.models] == [type(m).__name__ for m in b.models]
    assert [m.alias for m in a.models] == [m.alias for m in b.models]


def test_hier_recipe_reconcilers_match_builder(hier_recipe: Recipe) -> None:
    a = build_hier_reconcilers()
    b = build_hier_reconcilers_from_recipe(hier_recipe)
    assert [type(r).__name__ for r in a] == [type(r).__name__ for r in b]
    # method= for MinTrace / ERM
    a_methods = [getattr(r, "method", None) for r in a]
    b_methods = [getattr(r, "method", None) for r in b]
    assert a_methods == b_methods


def test_hier_experiments_recipe_builds_all_reconcilers(hier_experiments_recipe: Recipe) -> None:
    reconcilers = build_hier_reconcilers_from_recipe(hier_experiments_recipe)
    assert [type(r).__name__ for r in reconcilers] == [
        "BottomUp",
        "MinTrace",
        "MinTrace",
        "MinTrace",
        "MinTrace",
        "ERM",
        "ERM",
    ]
    assert [getattr(r, "method", None) for r in reconcilers] == [
        None,
        "ols",
        "wls_struct",
        "wls_var",
        "mint_shrink",
        "closed",
        "reg_bu",
    ]


# ----------------------------------------------------------------------
# Dispatcher
# ----------------------------------------------------------------------
def test_build_forecaster_dispatches_lgbm(lgbm_recipe: Recipe) -> None:
    from mlforecast import MLForecast

    fcst = build_forecaster(lgbm_recipe)
    assert isinstance(fcst, MLForecast)
    assert "LGBM" in fcst.models


def test_build_forecaster_dispatches_stats(stats_recipe: Recipe) -> None:
    from statsforecast import StatsForecast

    fcst = build_forecaster(stats_recipe)
    assert isinstance(fcst, StatsForecast)


def test_build_forecaster_dispatches_hier(hier_recipe: Recipe) -> None:
    bundle = build_forecaster(hier_recipe)
    assert isinstance(bundle, HierForecaster)
    assert len(bundle.reconcilers) == 3


def test_build_forecaster_rejects_wrong_kind_for_lgbm_builder(stats_recipe: Recipe) -> None:
    with pytest.raises(TypeError, match="kind='stats'"):
        build_lgbm_from_recipe(stats_recipe)
