"""Small matplotlib helpers used by the notebooks."""

from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter


def _thousands(x: float, _pos: int) -> str:
    return f"{x:,.0f}"


def _percentage(x: float, _pos: int) -> str:
    return f"{x * 100:.0f}%"


def format_yaxis_thousands(ax: Axes) -> None:
    ax.yaxis.set_major_formatter(FuncFormatter(_thousands))


def format_yaxis_percentage(ax: Axes) -> None:
    ax.yaxis.set_major_formatter(FuncFormatter(_percentage))
