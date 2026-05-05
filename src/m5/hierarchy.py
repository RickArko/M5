"""M5 12-level hierarchy via Nixtla ``hierarchicalforecast``.

The official M5 evaluation aggregates 30,490 item × store series into 12
overlapping levels (Total, state, store, category, department, plus their
cross-products, item, state × item, item × store). This module wraps
``hierarchicalforecast.utils.aggregate`` with the M5-specific spec and
provides helpers to round-trip the bottom level back to the project's
``{item_id}_{store_id}`` convention.

Notes:
    * ``aggregate`` requires that the *bottom* spec contain every column
      referenced by any upper level — so we list all six attribute columns at
      every grouping level, even though most are functionally implied
      (``store_id`` → ``state_id``, ``item_id`` → ``dept_id`` → ``cat_id``).
    * Series produced by ``aggregate`` use slash-joined unique_ids
      (``Total/CA/CA_1/FOODS/FOODS_1/FOODS_1_001``); :func:`extract_bottom`
      converts the bottom level back to ``FOODS_1_001_CA_1`` so existing
      WRMSSE scoring keeps working unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from hierarchicalforecast.utils import aggregate

TOTAL_COL = "Total"
TOTAL_VALUE = "Total"

# The 12 official M5 levels. Order matters: aggregate() treats the last entry
# as the bottom level, and every column referenced upstream must appear in it.
M5_LEVELS_SPEC: list[list[str]] = [
    [TOTAL_COL],
    [TOTAL_COL, "state_id"],
    [TOTAL_COL, "state_id", "store_id"],
    [TOTAL_COL, "cat_id"],
    [TOTAL_COL, "cat_id", "dept_id"],
    [TOTAL_COL, "state_id", "cat_id"],
    [TOTAL_COL, "state_id", "cat_id", "dept_id"],
    [TOTAL_COL, "state_id", "store_id", "cat_id"],
    [TOTAL_COL, "state_id", "store_id", "cat_id", "dept_id"],
    [TOTAL_COL, "cat_id", "dept_id", "item_id"],
    [TOTAL_COL, "state_id", "cat_id", "dept_id", "item_id"],
    [TOTAL_COL, "state_id", "store_id", "cat_id", "dept_id", "item_id"],
]

BOTTOM_LEVEL_KEY = "/".join(M5_LEVELS_SPEC[-1])


@dataclass(frozen=True)
class Hierarchy:
    """Result of aggregating a long frame to M5's 12 levels."""

    Y_df: pd.DataFrame  # unique_id, ds, y across all 12 levels
    S_df: pd.DataFrame  # summing matrix (n_total_series x n_bottom_series)
    tags: dict[str, np.ndarray]  # level_key -> array of unique_ids at that level
    bottom_id_map: pd.Series  # hier unique_id -> "{item_id}_{store_id}"

    @property
    def bottom_ids(self) -> np.ndarray:
        return self.tags[BOTTOM_LEVEL_KEY]


def build_hierarchy(
    df: pd.DataFrame,
    *,
    spec: list[list[str]] = M5_LEVELS_SPEC,
) -> Hierarchy:
    """Aggregate a Nixtla long frame to all 12 M5 levels.

    Args:
        df: Long frame with ``unique_id, ds, y`` plus the static attribute
            columns ``state_id, store_id, cat_id, dept_id, item_id``.
        spec: Hierarchy specification (defaults to the 12 official M5 levels).

    Returns:
        :class:`Hierarchy` bundling the aggregated series, summing matrix,
        per-level tags, and a map from hierarchical bottom-level unique_ids
        back to the project's ``{item_id}_{store_id}`` convention.
    """
    needed = {col for level in spec for col in level} - {TOTAL_COL}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing attribute columns required by spec: {sorted(missing)}")

    work = df.drop(columns=["unique_id"], errors="ignore").copy()
    work[TOTAL_COL] = TOTAL_VALUE

    keep_cols = ["ds", "y", TOTAL_COL, *sorted(needed)]
    Y_df, S_df, tags = aggregate(df=work[keep_cols], spec=spec)

    bottom_ids = tags[BOTTOM_LEVEL_KEY]
    parsed = pd.Series(bottom_ids, name="hier_id").str.split("/", expand=True)
    parsed.columns = spec[-1]
    bottom_id_map = pd.Series(
        (parsed["item_id"].astype(str) + "_" + parsed["store_id"].astype(str)).to_numpy(),
        index=pd.Index(bottom_ids, name="hier_id"),
        name="unique_id",
    )

    return Hierarchy(Y_df=Y_df, S_df=S_df, tags=tags, bottom_id_map=bottom_id_map)


def extract_bottom(
    reconciled: pd.DataFrame,
    hierarchy: Hierarchy,
    *,
    id_col: str = "unique_id",
) -> pd.DataFrame:
    """Slice a reconciled frame to the bottom level and restore original ids.

    Maps the slash-joined hierarchical unique_id back to the project's
    ``{item_id}_{store_id}`` convention so the result drops straight into
    :func:`m5.evaluation.wrmsse_for_models`.
    """
    mask = reconciled[id_col].isin(hierarchy.bottom_ids)
    out = reconciled.loc[mask].copy()
    out[id_col] = out[id_col].map(hierarchy.bottom_id_map)
    return out.reset_index(drop=True)
