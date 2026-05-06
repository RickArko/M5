"""Figure factory + report stitcher for the M5 evaluation framework.

Public surface kept tight on purpose — most callers only need
:func:`render_all` (build every figure + write metrics) and the two report
renderers.
"""

from __future__ import annotations

from m5.reporting.figures import build_all_figures
from m5.reporting.report import render_html, render_markdown, render_report
from m5.reporting.save import save_figure
from m5.reporting.style import MODEL_COLORS, PALETTE, apply_style, model_color

__all__ = [
    "MODEL_COLORS",
    "PALETTE",
    "apply_style",
    "build_all_figures",
    "model_color",
    "render_html",
    "render_markdown",
    "render_report",
    "save_figure",
]
