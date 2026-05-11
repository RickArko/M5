"""GCP tier-sweep orchestrator for the M5 API load test.

Phase 2 of ``docs/plans/api_loadtest.md``. For each tier defined in
``loadtest/tiers.yaml``:

1. ``terraform apply`` with ``serve_machine_type=<tier>``, ``create_train=false``.
2. Poll ``/readyz`` until the model artifact has loaded (max 5 min).
3. Pre-warm with a brief stream of light traffic.
4. Run Locust headless against the tier with ramp/hold/cooldown phases.
5. ``terraform destroy -target=google_compute_instance.serve`` — **always**,
   even on failure (try/finally guard).
6. Record realised wall-time spend; abort the sweep if cumulative spend
   would breach ``max_total_spend_usd`` in tiers.yaml.

Usage::

    uv run --group loadtest python -m loadtest.sweep tier --alias cheap
    uv run --group loadtest python -m loadtest.sweep all
    uv run --group loadtest python -m loadtest.sweep all --dry-run    # plan only

Environment::

    GOOGLE_APPLICATION_CREDENTIALS  required for terraform
    M5_LOADTEST_PAYLOAD             optional; default loadtest/payloads/unique_ids.txt
    M5_SERVE_API_KEY                optional; forwarded to locust
    M5_LOADTEST_TF_DIR              optional; default cloud/terraform/gcp
"""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TIERS_PATH = REPO_ROOT / "loadtest" / "tiers.yaml"
DEFAULT_TF_DIR = REPO_ROOT / "cloud" / "terraform" / "gcp"
DEFAULT_PAYLOAD = REPO_ROOT / "loadtest" / "payloads" / "unique_ids.txt"
REPORTS_BASE = REPO_ROOT / "reports" / "loadtest"

app = typer.Typer(add_completion=False, help="M5 API load-test tier sweep.")


# ---------------------------------------------------------------- config


@dataclass(frozen=True)
class TierConfig:
    """A single tier read from tiers.yaml + sweep-level defaults."""

    alias: str
    machine_type: str
    hourly_usd: float
    max_users: int
    warm_s: int
    hold_s: int
    ramp_s: int
    cooldown_s: int
    horizon: int
    batch_n: int
    notes: str = ""

    @property
    def planned_max_seconds(self) -> int:
        """Upper bound on wall-time for cost-guardrail math (overshoots intentionally)."""
        # boot + ready (~120s) + warm + ramp + hold + cooldown + teardown (~60s)
        return 120 + self.warm_s + self.ramp_s + self.hold_s + self.cooldown_s + 60

    @property
    def planned_max_usd(self) -> float:
        return self.planned_max_seconds * self.hourly_usd / 3600.0


@dataclass(frozen=True)
class SweepConfig:
    sweep_id: str
    max_total_spend_usd: float
    tiers: list[TierConfig] = field(default_factory=list)


def load_tiers(path: Path = TIERS_PATH) -> SweepConfig:
    cfg: dict[str, Any] = yaml.safe_load(path.read_text())
    defaults = cfg.get("defaults", {})
    tiers = [
        TierConfig(
            alias=t["alias"],
            machine_type=t["machine_type"],
            hourly_usd=float(t["hourly_usd"]),
            max_users=int(t["max_users"]),
            warm_s=int(t.get("warm_s", defaults.get("warm_s", 30))),
            hold_s=int(t.get("hold_s", defaults.get("hold_s", 180))),
            ramp_s=int(t.get("ramp_s", defaults.get("ramp_s", 60))),
            cooldown_s=int(t.get("cooldown_s", defaults.get("cooldown_s", 30))),
            horizon=int(t.get("horizon", defaults.get("horizon", 28))),
            batch_n=int(t.get("batch_n", defaults.get("batch_n", 10))),
            notes=t.get("notes", ""),
        )
        for t in cfg["tiers"]
    ]
    return SweepConfig(
        sweep_id=cfg.get("sweep_id", "unnamed-sweep"),
        max_total_spend_usd=float(cfg.get("max_total_spend_usd", 5.0)),
        tiers=tiers,
    )


# ---------------------------------------------------------------- terraform


def _terraform(args: list[str], tf_dir: Path) -> subprocess.CompletedProcess[str]:
    """Run a terraform command in ``tf_dir`` with GOOGLE_APPLICATION_CREDENTIALS inherited."""
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS not set; terraform cannot auth.")
    cmd = ["terraform", "-chdir=" + str(tf_dir), *args]
    typer.echo(f"  $ {shlex.join(cmd)}")
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def terraform_apply_serve(tier: TierConfig, tf_dir: Path) -> str:
    """Apply terraform with create_serve=true; return the serve VM public IP."""
    _terraform(
        [
            "apply",
            "-auto-approve",
            "-var=create_train=false",
            "-var=create_serve=true",
            f"-var=serve_machine_type={tier.machine_type}",
        ],
        tf_dir,
    )
    out = _terraform(["output", "-raw", "serve_public_ip"], tf_dir).stdout.strip()
    if not out or out == "null":
        raise RuntimeError("terraform did not emit serve_public_ip — module misconfigured?")
    return out


def terraform_destroy_serve(tf_dir: Path) -> None:
    """Destroy the serve VM. Idempotent — fine if it's already gone."""
    try:
        _terraform(
            [
                "destroy",
                "-auto-approve",
                "-target=google_compute_instance.serve",
            ],
            tf_dir,
        )
    except subprocess.CalledProcessError as e:
        typer.echo(
            f"  (warn) terraform destroy returned non-zero: {e.returncode}; continuing teardown anyway.",
            err=True,
        )


# ---------------------------------------------------------------- http polling


def _http_get(url: str, timeout: float = 5.0) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return 0, ""


def wait_for_ready(ip: str, port: int = 8000, max_s: int = 300) -> None:
    """Poll /readyz until 200; fall back to /healthz if /readyz never goes ready."""
    url = f"http://{ip}:{port}/readyz"
    typer.echo(f"  waiting for {url} (max {max_s}s)…")
    started = time.monotonic()
    last_status = -1
    while time.monotonic() - started < max_s:
        status, _ = _http_get(url, timeout=3.0)
        if status == 200:
            typer.echo(f"  ready after {int(time.monotonic() - started)}s")
            return
        if status != last_status:
            typer.echo(f"    /readyz → {status} ({int(time.monotonic() - started)}s)")
            last_status = status
        time.sleep(5)
    raise RuntimeError(f"/readyz never reached 200 after {max_s}s")


def warm(ip: str, port: int, payload: Path, warm_s: int) -> None:
    """Hit a few endpoints to JIT / page-cache before measurement."""
    typer.echo(f"  warming for {warm_s}s…")
    end = time.monotonic() + warm_s
    ids = [line.strip() for line in payload.read_text().splitlines() if line.strip()][:10]
    while time.monotonic() < end:
        _http_get(f"http://{ip}:{port}/healthz", timeout=3.0)
        # Hit predict via curl is fine here — we don't need locust for warmup
        if ids:
            try:
                import json

                body = json.dumps({"horizon": 28, "unique_ids": ids[:1]}).encode()
                req = urllib.request.Request(
                    f"http://{ip}:{port}/v1/predict/by-id",
                    data=body,
                    headers={"Content-Type": "application/json"},
                )
                if api_key := os.environ.get("M5_SERVE_API_KEY"):
                    req.add_header("X-API-Key", api_key)
                with urllib.request.urlopen(req, timeout=5.0):
                    pass
            except Exception:
                pass
        time.sleep(1.0)


# ---------------------------------------------------------------- locust


def run_locust(
    *,
    tier: TierConfig,
    ip: str,
    port: int,
    payload: Path,
    out_prefix: Path,
) -> dict[str, float]:
    """Run locust headless and return realised wall-clock seconds.

    Locust writes CSVs to ``<out_prefix>_stats.csv`` + ``_stats_history.csv``
    + ``_failures.csv``; the HTML summary goes to ``<out_prefix>.html``.
    """
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "M5_LOADTEST_PAYLOAD": str(payload),
        "M5_LOADTEST_HORIZON": str(tier.horizon),
        "M5_LOADTEST_BATCH_N": str(tier.batch_n),
    }
    spawn_rate = max(1, tier.max_users // max(1, tier.ramp_s))
    cmd = [
        "locust",
        "-f",
        str(REPO_ROOT / "loadtest" / "locustfile.py"),
        "--headless",
        "-u",
        str(tier.max_users),
        "-r",
        str(spawn_rate),
        "-H",
        f"http://{ip}:{port}",
        "--run-time",
        f"{tier.hold_s + tier.cooldown_s}s",
        "--csv",
        str(out_prefix),
        "--html",
        f"{out_prefix}.html",
    ]
    typer.echo(f"  $ {shlex.join(cmd)}")
    start = time.monotonic()
    rc = subprocess.run(cmd, env=env, check=False).returncode
    wall_s = time.monotonic() - start
    if rc != 0:
        typer.echo(f"  (warn) locust returned {rc} after {wall_s:.0f}s — saved partial CSVs", err=True)
    return {"wall_seconds": wall_s, "locust_rc": float(rc)}


# ---------------------------------------------------------------- one tier


def run_tier(
    tier: TierConfig,
    *,
    tf_dir: Path,
    payload: Path,
    sweep_ts: str,
    serve_port: int = 8000,
) -> dict[str, float]:
    """Apply → ready → warm → locust → destroy. Returns realised cost dict."""
    out_prefix = REPORTS_BASE / sweep_ts / tier.alias
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    tier_started = time.monotonic()
    try:
        typer.echo(f"\n=== tier {tier.alias} ({tier.machine_type}) ===")
        ip = terraform_apply_serve(tier, tf_dir)
        typer.echo(f"  serve VM up at {ip}")
        # 900s: serve.sh's `docker compose up --build` on a fresh VM rebuilds
        # the full python/uvicorn/mlforecast image. On the slowest tier (e2-small)
        # that takes 5-8 min; 5 min is too tight, 15 min is safe-with-margin.
        wait_for_ready(ip, port=serve_port, max_s=900)
        warm(ip, port=serve_port, payload=payload, warm_s=tier.warm_s)
        stats = run_locust(tier=tier, ip=ip, port=serve_port, payload=payload, out_prefix=out_prefix)
    finally:
        # Always destroy — even on KeyboardInterrupt / locust crash / readiness timeout.
        typer.echo("  tearing down serve VM (trap)…")
        terraform_destroy_serve(tf_dir)
    wall_s = time.monotonic() - tier_started
    realised_usd = wall_s * tier.hourly_usd / 3600.0
    typer.echo(f"  tier {tier.alias}: wall={wall_s:.0f}s, ~${realised_usd:.4f}")
    result = {**stats, "realised_wall_s": wall_s, "realised_usd": realised_usd}

    # Persist per-tier metadata so phase-3 aggregate can compute $/MRPS without
    # re-running terraform or re-deriving cost from log scraping.
    import json

    meta_path = out_prefix.parent / f"{tier.alias}_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "alias": tier.alias,
                "machine_type": tier.machine_type,
                "hourly_usd": tier.hourly_usd,
                "max_users": tier.max_users,
                "horizon": tier.horizon,
                "batch_n": tier.batch_n,
                "warm_s": tier.warm_s,
                "ramp_s": tier.ramp_s,
                "hold_s": tier.hold_s,
                "cooldown_s": tier.cooldown_s,
                **{k: float(v) for k, v in result.items()},
            },
            indent=2,
        )
        + "\n"
    )
    return result


# ---------------------------------------------------------------- CLI


def _resolve_paths(*, payload: Path | None, tf_dir: Path | None) -> tuple[Path, Path]:
    payload = payload or Path(os.environ.get("M5_LOADTEST_PAYLOAD") or DEFAULT_PAYLOAD)
    if not payload.exists():
        raise typer.BadParameter(f"payload {payload} missing — run `make loadtest-payload` first.")
    tf_dir = tf_dir or Path(os.environ.get("M5_LOADTEST_TF_DIR") or DEFAULT_TF_DIR)
    if not (tf_dir / "main.tf").exists():
        raise typer.BadParameter(f"{tf_dir} is not a terraform module (no main.tf).")
    if not shutil.which("terraform"):
        raise typer.BadParameter("terraform not on PATH.")
    return payload, tf_dir


@contextmanager
def _signal_guard():
    """Make SIGINT/SIGTERM raise KeyboardInterrupt so try/finally fires."""

    def _raise(signum, frame):
        raise KeyboardInterrupt(f"caught signal {signum}")

    prev_int = signal.signal(signal.SIGINT, _raise)
    prev_term = signal.signal(signal.SIGTERM, _raise)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, prev_int)
        signal.signal(signal.SIGTERM, prev_term)


@app.command()
def tier(
    alias: str = typer.Option(..., "--alias", "-a", help="Tier alias from tiers.yaml."),
    payload: Path = typer.Option(None, help="Path to unique_ids corpus."),
    tf_dir: Path = typer.Option(None, help="Path to the terraform module."),
    serve_port: int = typer.Option(8000),
) -> None:
    """Run a single tier end-to-end."""
    payload, tf_dir = _resolve_paths(payload=payload, tf_dir=tf_dir)
    cfg = load_tiers()
    selected = next((t for t in cfg.tiers if t.alias == alias), None)
    if selected is None:
        raise typer.BadParameter(f"tier {alias!r} not in tiers.yaml; have: {[t.alias for t in cfg.tiers]}")
    sweep_ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    with _signal_guard():
        run_tier(selected, tf_dir=tf_dir, payload=payload, sweep_ts=sweep_ts, serve_port=serve_port)


@app.command()
def all(
    payload: Path = typer.Option(None),
    tf_dir: Path = typer.Option(None),
    serve_port: int = typer.Option(8000),
    dry_run: bool = typer.Option(False, help="Print the plan without applying terraform."),
) -> None:
    """Run the full sweep over every tier in tiers.yaml."""
    payload, tf_dir = _resolve_paths(payload=payload, tf_dir=tf_dir)
    cfg = load_tiers()
    sweep_ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    typer.echo(f"sweep_id={cfg.sweep_id}  ts={sweep_ts}  tiers={[t.alias for t in cfg.tiers]}")
    typer.echo(f"max_total_spend_usd guardrail: ${cfg.max_total_spend_usd:.2f}")
    planned = sum(t.planned_max_usd for t in cfg.tiers)
    typer.echo(f"planned max spend (all tiers, upper-bound): ${planned:.4f}")
    if planned > cfg.max_total_spend_usd:
        raise typer.BadParameter(
            f"planned max spend ${planned:.2f} would breach guardrail "
            f"${cfg.max_total_spend_usd:.2f}. Lower hold_s or drop a tier."
        )
    if dry_run:
        typer.echo("dry-run: not invoking terraform.")
        return

    cumulative_usd = 0.0
    results: list[dict[str, float]] = []
    with _signal_guard():
        for t in cfg.tiers:
            if cumulative_usd + t.planned_max_usd > cfg.max_total_spend_usd:
                typer.echo(
                    f"ABORT: cumulative=${cumulative_usd:.4f} + "
                    f"next-tier-max=${t.planned_max_usd:.4f} would breach "
                    f"${cfg.max_total_spend_usd:.2f}.",
                    err=True,
                )
                sys.exit(2)
            res = run_tier(t, tf_dir=tf_dir, payload=payload, sweep_ts=sweep_ts, serve_port=serve_port)
            cumulative_usd += res["realised_usd"]
            results.append({"alias": t.alias, **res})  # type: ignore[dict-item]

    typer.echo(f"\nSWEEP DONE — cumulative spend ${cumulative_usd:.4f}")
    typer.echo(f"reports at {REPORTS_BASE / sweep_ts}")
    typer.echo("\nnext: `python -m loadtest.aggregate " + str(REPORTS_BASE / sweep_ts) + "`")


@app.callback(invoke_without_command=True)
def _main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


if __name__ == "__main__":
    app()
