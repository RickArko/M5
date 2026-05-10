"""Visualisation tooling — animated SVG + interactive D3.js renderer.

Single entrypoint: :func:`render_pipeline_viz` reads a fitted artifact and
emits two self-contained files (``pipeline.svg`` + ``pipeline.html``) that
walk a viewer through fit -> predict -> score on one hero series.
"""

from __future__ import annotations

from m5.viz.pipeline import (
    VizPayload,
    build_payload,
    render_gif,
    render_html,
    render_pipeline_viz,
    render_svg,
)

__all__ = [
    "VizPayload",
    "build_payload",
    "render_gif",
    "render_html",
    "render_pipeline_viz",
    "render_svg",
]
