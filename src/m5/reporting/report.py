"""Stitch the figure bundle + metric tables into Markdown and HTML reports.

Both formats are generated from the same content blocks so they don't drift.
Markdown references PNG (renders on GitHub); HTML references SVG (vector).
Tables are rendered without :mod:`tabulate` to keep the dep tree small.
"""

from __future__ import annotations

import datetime as _dt
import html
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from m5.reporting.figures import _FIGURE_ORDER, FigureBundle

__all__ = ["RunMetadata", "render_html", "render_markdown", "render_report"]


@dataclass
class RunMetadata:
    """Provenance header rendered at the top of every report."""

    run_id: str
    timestamp: str = field(default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds"))
    git_sha: str = ""
    seed: int = 0
    horizon: int = 0
    n_windows: int = 0
    models: list[str] = field(default_factory=list)
    n_series: int = 0

    @classmethod
    def autodiscover(cls, **kwargs: object) -> RunMetadata:
        sha = ""
        try:
            sha = (
                subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
                .decode()
                .strip()
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        return cls(git_sha=sha, **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Tiny markdown table renderer (avoids the tabulate dependency)               #
# --------------------------------------------------------------------------- #
def _format_value(v: object) -> str:
    if isinstance(v, np.floating | float):
        if not np.isfinite(v):
            return "—"
        return f"{v:.4f}"
    if isinstance(v, np.integer | int):
        return f"{int(v):,d}"
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    return str(v)


def _df_to_md(df: pd.DataFrame, *, index: bool = False) -> str:
    cols = ([df.index.name or ""] if index else []) + list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for idx, row in df.iterrows():
        cells = [_format_value(idx)] if index else []
        cells += [_format_value(row[c]) for c in df.columns]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def _df_to_html(df: pd.DataFrame, *, index: bool = False) -> str:
    return df.to_html(
        index=index,
        float_format=lambda x: "—" if pd.isna(x) else f"{x:.4f}",
        classes="m5-table",
        border=0,
        escape=True,
    )


# --------------------------------------------------------------------------- #
# Markdown renderer                                                            #
# --------------------------------------------------------------------------- #
def render_markdown(
    bundle: FigureBundle,
    *,
    metadata: RunMetadata,
    headline: pd.DataFrame,
    figures_subdir: str = "figures",
    image_format: str = "png",
) -> str:
    lines: list[str] = []
    lines.append(f"# M5 Forecast Evaluation — {metadata.run_id}")
    lines.append("")
    lines.append(
        f"_Generated {metadata.timestamp} · git {metadata.git_sha or '—'} · "
        f"seed {metadata.seed} · horizon {metadata.horizon}d × "
        f"{metadata.n_windows} folds · {metadata.n_series:,d} series · "
        f"models: {', '.join(metadata.models) or '—'}_"
    )
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    if not headline.empty:
        lines.append(_df_to_md(headline.round(4)))
    else:
        lines.append("_no headline metrics_")
    lines.append("")
    for name, title in _FIGURE_ORDER:
        if name not in bundle.figures:
            continue
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"![{title}]({figures_subdir}/{name}.{image_format})")
        lines.append("")
        if bundle.captions.get(name):
            lines.append(f"_{bundle.captions[name]}_")
            lines.append("")
        if bundle.insights.get(name):
            lines.append(f"> {bundle.insights[name]}")
            lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# HTML renderer                                                                #
# --------------------------------------------------------------------------- #
_HTML_CSS = """
:root { color-scheme: light dark; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
       max-width: 980px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }
h1 { border-bottom: 2px solid #ddd; padding-bottom: .35rem; }
h2 { margin-top: 2.5rem; border-bottom: 1px solid #eee; padding-bottom: .25rem; }
.meta { color: #666; font-size: .85rem; margin: .5rem 0 1.5rem; }
.caption { color: #555; font-size: .85rem; margin-top: .25rem; font-style: italic; }
.insight { background: #fff8e1; border-left: 4px solid #f0c14b; padding: .55rem .75rem;
           margin: .75rem 0 1.5rem; font-size: .9rem; }
table.m5-table { border-collapse: collapse; margin: .5rem 0 1rem; font-size: .85rem; }
table.m5-table th, table.m5-table td { border-bottom: 1px solid #eee; padding: .35rem .65rem;
           text-align: right; }
table.m5-table th { background: #fafafa; font-weight: 600; }
table.m5-table td:first-child, table.m5-table th:first-child { text-align: left; }
img.figure { max-width: 100%; height: auto; display: block; margin: 1rem 0; }
@media (prefers-color-scheme: dark) {
  body { background: #1a1a1a; color: #eee; }
  h1, h2 { border-color: #333; }
  table.m5-table th { background: #222; }
  table.m5-table th, table.m5-table td { border-color: #333; }
  .insight { background: #2a2210; border-left-color: #d4a93a; }
}
"""


def render_html(
    bundle: FigureBundle,
    *,
    metadata: RunMetadata,
    headline: pd.DataFrame,
    figures_subdir: str = "figures",
    image_format: str = "svg",
) -> str:
    parts: list[str] = []
    parts.append("<!doctype html>")
    parts.append('<html lang="en">')
    parts.append("<head>")
    parts.append('<meta charset="utf-8">')
    parts.append(f"<title>M5 Evaluation — {html.escape(metadata.run_id)}</title>")
    parts.append(f"<style>{_HTML_CSS}</style>")
    parts.append("</head>")
    parts.append("<body>")
    parts.append(f"<h1>M5 Forecast Evaluation — {html.escape(metadata.run_id)}</h1>")
    parts.append('<div class="meta">')
    parts.append(
        f"Generated {html.escape(metadata.timestamp)} · git "
        f"{html.escape(metadata.git_sha or '—')} · seed {metadata.seed} · "
        f"horizon {metadata.horizon}d × {metadata.n_windows} folds · "
        f"{metadata.n_series:,d} series · models: "
        f"{html.escape(', '.join(metadata.models) or '—')}"
    )
    parts.append("</div>")
    parts.append("<h2>Headline</h2>")
    if not headline.empty:
        parts.append(_df_to_html(headline.round(4)))
    else:
        parts.append("<p><em>no headline metrics</em></p>")
    for name, title in _FIGURE_ORDER:
        if name not in bundle.figures:
            continue
        parts.append(f"<h2>{html.escape(title)}</h2>")
        parts.append(
            f'<img class="figure" src="{figures_subdir}/{name}.{image_format}" alt="{html.escape(title)}">'
        )
        if bundle.captions.get(name):
            parts.append(f'<div class="caption">{html.escape(bundle.captions[name])}</div>')
        if bundle.insights.get(name):
            parts.append(f'<div class="insight">{bundle.insights[name]}</div>')
    parts.append("</body>")
    parts.append("</html>")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Combined renderer                                                            #
# --------------------------------------------------------------------------- #
def render_report(
    bundle: FigureBundle,
    *,
    metadata: RunMetadata,
    headline: pd.DataFrame,
    out_dir: Path | str,
    figures_subdir: str = "figures",
    extra_tables: Iterable[tuple[str, pd.DataFrame]] = (),
) -> dict[str, Path]:
    """Write ``report.md`` + ``report.html`` next to the figures directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = render_markdown(bundle, metadata=metadata, headline=headline, figures_subdir=figures_subdir)
    if extra_tables:
        md_extras = ["\n## Appendix — supporting tables\n"]
        for name, df in extra_tables:
            md_extras.append(f"### {name}\n")
            md_extras.append(_df_to_md(df.round(4)) if not df.empty else "_empty_")
            md_extras.append("")
        md = md + "\n" + "\n".join(md_extras)
    html_doc = render_html(bundle, metadata=metadata, headline=headline, figures_subdir=figures_subdir)

    md_path = out_dir / "report.md"
    html_path = out_dir / "report.html"
    md_path.write_text(md, encoding="utf-8")
    html_path.write_text(html_doc, encoding="utf-8")
    return {"md": md_path, "html": html_path}
