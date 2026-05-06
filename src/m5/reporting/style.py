"""One source of truth for figure styling.

Call :func:`apply_style` once at the top of any rendering pass. The palette
is colour-blind-safe (Wong / Okabe-Ito) and :func:`model_color` deterministically
hashes a model name to a palette entry so the same model gets the same colour
across every figure in a run.
"""

from __future__ import annotations

import hashlib
from typing import Final

import matplotlib as mpl

# Wong/Okabe-Ito 8-colour palette — designed for colour-blind accessibility.
PALETTE: Final[tuple[str, ...]] = (
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#999999",  # neutral grey
)

# Reserve stable colours for the canonical model names so leaderboards always
# look the same. Anything not in this dict falls back to a hash-based pick.
MODEL_COLORS: Final[dict[str, str]] = {
    "SeasonalNaive": "#999999",
    "Theta": "#0072B2",
    "AutoETS": "#56B4E9",
    "LGBM": "#D55E00",
    "BU": "#009E73",
    "TopDown": "#CC79A7",
    "MinTrace_ols": "#E69F00",
    "MinTrace_shrink": "#F0E442",
}


def model_color(model: str) -> str:
    """Stable colour for a model name across every figure in the report."""
    if model in MODEL_COLORS:
        return MODEL_COLORS[model]
    digest = hashlib.md5(model.encode("utf-8")).digest()
    return PALETTE[digest[0] % len(PALETTE)]


def apply_style() -> None:
    """Idempotent: reset matplotlib rcParams to the report style."""
    mpl.rcParams.update(
        {
            "figure.figsize": (8.0, 5.0),
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#cccccc",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.title_fontsize": 9,
            "lines.linewidth": 1.6,
            "lines.markersize": 4,
            "axes.prop_cycle": mpl.cycler(color=PALETTE),  # type: ignore[attr-defined]
        }
    )
