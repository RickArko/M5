"""Unit tests for the M5 hierarchy aggregation."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest


def _have(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


pytestmark = pytest.mark.skipif(
    not _have("hierarchicalforecast"), reason="hierarchicalforecast not installed"
)


def test_spec_has_twelve_levels() -> None:
    from m5.hierarchy import M5_LEVELS_SPEC

    assert len(M5_LEVELS_SPEC) == 12


def test_bottom_level_includes_every_referenced_column() -> None:
    """``aggregate`` requires the bottom spec to be a superset of every upper level."""
    from m5.hierarchy import M5_LEVELS_SPEC

    bottom = set(M5_LEVELS_SPEC[-1])
    for level in M5_LEVELS_SPEC[:-1]:
        assert set(level).issubset(bottom), f"upper level {level} leaks columns past bottom"


def test_build_hierarchy_returns_expected_shape(toy_long: pd.DataFrame) -> None:
    from m5.hierarchy import BOTTOM_LEVEL_KEY, build_hierarchy

    hier = build_hierarchy(toy_long)
    assert len(hier.tags) == 12
    assert len(hier.bottom_ids) == toy_long["unique_id"].nunique()
    # S_df rows = total series across all 12 levels; columns = bottom + unique_id label
    assert hier.S_df.shape[0] == sum(len(ids) for ids in hier.tags.values())
    assert BOTTOM_LEVEL_KEY in hier.tags


def test_bottom_sums_match_total_per_date(toy_long: pd.DataFrame) -> None:
    """Aggregation invariant: ``y`` at the Total level == sum of bottom-level ``y``."""
    from m5.hierarchy import build_hierarchy

    hier = build_hierarchy(toy_long)
    totals = hier.Y_df[hier.Y_df["unique_id"] == "Total"].set_index("ds")["y"]
    bottom = hier.Y_df[hier.Y_df["unique_id"].isin(hier.bottom_ids)].groupby("ds")["y"].sum()
    np.testing.assert_allclose(totals.to_numpy(), bottom.to_numpy(), rtol=1e-6)


def test_extract_bottom_round_trips_unique_ids(toy_long: pd.DataFrame) -> None:
    """The hierarchical id format is internal; downstream sees the project ids."""
    from m5.hierarchy import build_hierarchy, extract_bottom

    hier = build_hierarchy(toy_long)
    fake = hier.Y_df.assign(model=hier.Y_df["y"] * 2.0)
    out = extract_bottom(fake, hier)
    assert set(out["unique_id"]) == set(toy_long["unique_id"])


def test_build_hierarchy_rejects_missing_attribute_columns(toy_long: pd.DataFrame) -> None:
    from m5.hierarchy import build_hierarchy

    with pytest.raises(ValueError, match="Missing attribute columns"):
        build_hierarchy(toy_long.drop(columns=["state_id"]))
