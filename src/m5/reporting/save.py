"""Multi-format figure saving with embedded provenance metadata.

PNG goes in the markdown report (renders on GitHub), SVG in the HTML report
(scales cleanly in browsers), PDF for shareable print copies. Each format has
its own metadata constraints — PNG/PDF accept arbitrary string keys, SVG
only accepts Dublin Core fields. This module handles the translation so
callers can pass one metadata dict and have provenance survive across all
three formats.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

DEFAULT_FORMATS: tuple[str, ...] = ("png", "svg", "pdf")

# matplotlib's SVG backend only accepts these (Dublin Core) keys. Anything
# else gets folded into Description so it's still discoverable.
_SVG_DC_KEYS = frozenset(
    {
        "Coverage",
        "Contributor",
        "Creator",
        "Date",
        "Description",
        "Format",
        "Identifier",
        "Language",
        "Publisher",
        "Relation",
        "Rights",
        "Source",
        "Subject",
        "Title",
        "Type",
        "Keywords",
    }
)


def _default_metadata() -> dict[str, str]:
    return {
        "Creator": "m5.reporting",
        "Date": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
    }


_PDF_KEYS = frozenset(
    {"Title", "Author", "Subject", "Keywords", "Creator", "Producer", "CreationDate", "ModDate", "Trapped"}
)


def _metadata_for_format(meta: Mapping[str, str], fmt: str) -> dict[str, Any]:
    if fmt == "svg":
        dc = {k: v for k, v in meta.items() if k in _SVG_DC_KEYS}
        extras = {k: v for k, v in meta.items() if k not in _SVG_DC_KEYS}
        if extras:
            existing = dc.get("Description", "").strip()
            packed = "; ".join(f"{k}={v}" for k, v in extras.items())
            dc["Description"] = (existing + " | " + packed).strip(" |") if existing else packed
        return dc
    if fmt == "pdf":
        translated: dict[str, Any] = dict(meta)
        if "Date" in translated and "CreationDate" not in translated:
            raw = translated.pop("Date")
            try:
                translated["CreationDate"] = _dt.datetime.fromisoformat(raw)
            except (TypeError, ValueError):
                translated["CreationDate"] = _dt.datetime.now(_dt.UTC)
        kept = {k: v for k, v in translated.items() if k in _PDF_KEYS}
        extras = {k: v for k, v in translated.items() if k not in _PDF_KEYS}
        if extras:
            existing = kept.get("Subject", "").strip() if isinstance(kept.get("Subject"), str) else ""
            packed = "; ".join(f"{k}={v}" for k, v in extras.items())
            kept["Subject"] = (existing + " | " + packed).strip(" |") if existing else packed
        return kept
    return dict(meta)


def save_figure(
    fig: Figure,
    name: str,
    *,
    out_dir: Path | str,
    formats: tuple[str, ...] = DEFAULT_FORMATS,
    dpi: int = 150,
    metadata: Mapping[str, str] | None = None,
    close: bool = True,
) -> dict[str, Path]:
    """Save ``fig`` as ``name.<fmt>`` for each requested format.

    Returns a mapping ``fmt → Path`` so the report stitcher can embed by
    relative path. Closes the figure after saving unless ``close=False``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {**_default_metadata(), **(dict(metadata) if metadata else {})}
    paths: dict[str, Path] = {}
    for fmt in formats:
        path = out_dir / f"{name}.{fmt}"
        kwargs: dict[str, Any] = {"dpi": dpi}
        if fmt in {"png", "pdf", "svg"}:
            kwargs["metadata"] = _metadata_for_format(meta, fmt)
        fig.savefig(path, **kwargs)
        paths[fmt] = path
    if close:
        plt.close(fig)
    return paths
