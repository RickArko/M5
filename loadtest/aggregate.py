"""Aggregate per-tier Locust CSVs into one comparison report.

Phase 3 of ``docs/plans/api_loadtest.md``. Inputs sit under
``reports/loadtest/<sweep_ts>/`` with the layout the phase-2 orchestrator
writes::

    <alias>_stats.csv             # locust aggregate, per-endpoint rows
    <alias>_stats_history.csv     # locust time-series
    <alias>_meta.json             # tier config + realised wall_s/usd

Outputs (next to the inputs)::

    summary.md                    # comparison table + interpretation
    figures/01_latency_vs_rps.png
    figures/02_cost_per_mrps.png
    figures/03_users_vs_p99.png

Usage::

    uv run --group loadtest python -m loadtest.aggregate reports/loadtest/<ts>/
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import typer

app = typer.Typer(add_completion=False, help="Aggregate locust CSVs into a comparison report.")


# ---------------------------------------------------------------- data model


@dataclass(frozen=True)
class TierResult:
    """One row in the comparison table — per-tier headline."""

    alias: str
    machine_type: str
    hourly_usd: float
    max_users: int
    realised_wall_s: float
    realised_usd: float
    total_reqs: int
    failure_pct: float
    rps_avg: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    usd_per_mrps: float  # USD per million-requests-served-at-this-throughput


def _load_meta(meta_path: Path) -> dict[str, float | int | str]:
    return json.loads(meta_path.read_text())


def _aggregated_row(stats_csv: Path) -> pd.Series:
    """Pull the locust 'Aggregated' row (last row in *_stats.csv)."""
    df = pd.read_csv(stats_csv)
    aggr = df[df["Name"] == "Aggregated"]
    if aggr.empty:
        raise RuntimeError(f"{stats_csv}: no 'Aggregated' row found")
    return aggr.iloc[-1]


def _tier_result(meta: dict[str, float | int | str], stats_csv: Path) -> TierResult:
    aggr = _aggregated_row(stats_csv)
    total_reqs = int(aggr["Request Count"])
    failures = int(aggr["Failure Count"])
    rps_avg = float(aggr["Requests/s"])
    p50 = float(aggr["50%"])
    p95 = float(aggr["95%"])
    p99 = float(aggr["99%"])
    realised_usd = float(meta["realised_usd"])
    usd_per_mrps = realised_usd / max(total_reqs, 1) * 1_000_000
    return TierResult(
        alias=str(meta["alias"]),
        machine_type=str(meta["machine_type"]),
        hourly_usd=float(meta["hourly_usd"]),
        max_users=int(meta["max_users"]),
        realised_wall_s=float(meta["realised_wall_s"]),
        realised_usd=realised_usd,
        total_reqs=total_reqs,
        failure_pct=(failures / total_reqs * 100.0) if total_reqs else 0.0,
        rps_avg=rps_avg,
        p50_ms=p50,
        p95_ms=p95,
        p99_ms=p99,
        usd_per_mrps=usd_per_mrps,
    )


def _collect(sweep_dir: Path) -> list[TierResult]:
    metas = sorted(sweep_dir.glob("*_meta.json"))
    results: list[TierResult] = []
    for m in metas:
        alias = m.stem.replace("_meta", "")
        stats_csv = sweep_dir / f"{alias}_stats.csv"
        if not stats_csv.exists():
            typer.echo(f"  (warn) {stats_csv} missing — skipping {alias}", err=True)
            continue
        meta = _load_meta(m)
        try:
            results.append(_tier_result(meta, stats_csv))
        except Exception as e:
            typer.echo(f"  (warn) {alias}: {e}", err=True)
    if not results:
        raise RuntimeError(f"no usable tier results under {sweep_dir}")
    return results


# ---------------------------------------------------------------- figures


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _figure_latency_vs_rps(results: list[TierResult], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    aliases = [r.alias for r in results]
    x = range(len(aliases))
    ax.plot(x, [r.p50_ms for r in results], "o-", label="p50", linewidth=2)
    ax.plot(x, [r.p95_ms for r in results], "s-", label="p95", linewidth=2)
    ax.plot(x, [r.p99_ms for r in results], "^-", label="p99", linewidth=2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(aliases)
    ax.set_ylabel("Latency (ms)")
    ax.set_xlabel("Tier")
    ax.set_title("Latency by tier (lower is better)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_fig(fig, out)


def _figure_cost_per_mrps(results: list[TierResult], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    aliases = [r.alias for r in results]
    values = [r.usd_per_mrps for r in results]
    bars = ax.bar(aliases, values, color="#4c72b0")
    for bar, v in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v,
            f"${v:,.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("USD per million requests")
    ax.set_title("Cost per million requests by tier (lower is better)")
    ax.grid(True, axis="y", alpha=0.3)
    _save_fig(fig, out)


def _figure_users_vs_p99(sweep_dir: Path, results: list[TierResult], out: Path) -> None:
    """Plot p99 latency over time (proxy for users-vs-p99 since locust ramps linearly)."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for r in results:
        history = sweep_dir / f"{r.alias}_stats_history.csv"
        if not history.exists():
            continue
        df = pd.read_csv(history)
        # The 'Aggregated' rows in history give us the rolling p99.
        aggr = df[df["Name"] == "Aggregated"].copy()
        if aggr.empty or "User Count" not in aggr.columns or "99%" not in aggr.columns:
            continue
        ax.plot(
            aggr["User Count"],
            pd.to_numeric(aggr["99%"], errors="coerce"),
            label=f"{r.alias} ({r.machine_type})",
            linewidth=1.6,
            alpha=0.9,
        )
    ax.set_xlabel("Concurrent users")
    ax.set_ylabel("p99 latency (ms)")
    ax.set_title("p99 latency vs concurrent users — saturation knee per tier")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    _save_fig(fig, out)


# ---------------------------------------------------------------- summary

_SUMMARY_HEADER = """# M5 API load-test sweep — {sweep_id}

Sweep id: `{sweep_id}` · {n} tiers · generated {ts}

## Headline

| tier | machine_type | $/h | max_users | reqs | fail % | rps | p50 ms | p95 ms | p99 ms | $/MRPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
"""


def _summary_md(sweep_dir: Path, results: list[TierResult]) -> str:
    import datetime as _dt

    sweep_id = sweep_dir.name
    rows = []
    for r in results:
        rows.append(
            f"| {r.alias} | {r.machine_type} | ${r.hourly_usd:.3f} | {r.max_users} | "
            f"{r.total_reqs:,} | {r.failure_pct:.2f} | {r.rps_avg:.1f} | "
            f"{r.p50_ms:.0f} | {r.p95_ms:.0f} | {r.p99_ms:.0f} | ${r.usd_per_mrps:,.2f} |"
        )

    body = (
        _SUMMARY_HEADER.format(
            sweep_id=sweep_id,
            n=len(results),
            ts=_dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        + "\n".join(rows)
        + "\n\n"
    )

    winner_cost = min(results, key=lambda r: r.usd_per_mrps)
    winner_p99 = min(results, key=lambda r: r.p99_ms)
    body += (
        "## Interpretation\n\n"
        f"- **Cheapest per MRPS:** `{winner_cost.alias}` ({winner_cost.machine_type}) at "
        f"${winner_cost.usd_per_mrps:,.2f}/MRPS.\n"
        f"- **Lowest p99 latency:** `{winner_p99.alias}` ({winner_p99.machine_type}) at "
        f"{winner_p99.p99_ms:.0f} ms.\n"
        "- Tiers with `fail %` > 5 saturated during the hold phase — their "
        "$/MRPS numbers should be discounted (you'd actually need a bigger tier in prod).\n"
        "\n## Figures\n\n"
        "- ![Latency by tier](figures/01_latency_vs_rps.png)\n"
        "- ![Cost per million requests](figures/02_cost_per_mrps.png)\n"
        "- ![p99 vs concurrent users](figures/03_users_vs_p99.png)\n"
    )
    return body


# ---------------------------------------------------------------- CLI


@app.command()
def run(
    sweep_dir: Path = typer.Argument(..., help="Directory under reports/loadtest/."),
) -> None:
    """Build summary.md + 3 figures from the per-tier CSVs in ``sweep_dir``."""
    if not sweep_dir.exists():
        raise typer.BadParameter(f"{sweep_dir} does not exist.")
    typer.echo(f"aggregating {sweep_dir}…")
    results = _collect(sweep_dir)
    typer.echo(f"  loaded {len(results)} tier(s): {[r.alias for r in results]}")

    figures = sweep_dir / "figures"
    _figure_latency_vs_rps(results, figures / "01_latency_vs_rps.png")
    _figure_cost_per_mrps(results, figures / "02_cost_per_mrps.png")
    _figure_users_vs_p99(sweep_dir, results, figures / "03_users_vs_p99.png")

    summary_path = sweep_dir / "summary.md"
    summary_path.write_text(_summary_md(sweep_dir, results))
    typer.echo(f"  wrote {summary_path}")
    typer.echo(f"  wrote {figures}/01_latency_vs_rps.png, 02_cost_per_mrps.png, 03_users_vs_p99.png")


@app.callback(invoke_without_command=True)
def _main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


if __name__ == "__main__":
    app()
