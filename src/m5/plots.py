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


def configure_style(
    *,
    figsize: tuple[float, float] = (12, 5),
    dpi: int = 110,
    palette: str = "colorblind",
) -> None:
    """Apply the shared notebook plot style (idempotent).

    Sets default ``figure.figsize``, ``figure.dpi``, ``axes.grid`` on matplotlib
    and the seaborn palette if seaborn is importable. Call once at the top of a
    notebook so every figure has a consistent look without per-cell rcParam tweaks.
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.figsize": figsize,
            "figure.dpi": dpi,
            "axes.grid": True,
        }
    )
    try:
        import seaborn as sns
    except ImportError:
        return
    sns.set_palette(palette)
