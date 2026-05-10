"""Animated SVG + interactive D3 page that walk through the M5 pipeline.

Reads a fitted serving artifact (``artifacts/models/lgbm/latest/``) and the
processed long-frame, picks a hero series, runs the model forward by ``h``
days, computes a Seasonal-Naive(7) baseline + per-window RMSE, and emits two
self-contained files:

* ``pipeline.svg``  — animated (SMIL) SVG, embeds inline in GitHub README.md.
* ``pipeline.html`` — standalone D3.js v7 page; loads from CDN, scrub-able.

Both consume the same :class:`VizPayload`, so the SVG matches what an
interactive viewer would render at its starting frame.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from html import escape
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from m5.config import REPO_ROOT
from m5.logging import logger

DEFAULT_MODEL_DIR = REPO_ROOT / "artifacts" / "models" / "lgbm" / "latest"
DEFAULT_LONG_PATH = REPO_ROOT / "data" / "processed" / "long.parquet"
DEFAULT_OUT_DIR = REPO_ROOT / "assets"


# --------------------------------------------------------------------- payload


@dataclass
class CVWindow:
    """One rolling-origin window — kept JSON-serialisable for embedding."""

    cutoff: str
    forecast: list[float]
    baseline: list[float]
    truth: list[float]
    rmse_lgbm: float
    rmse_baseline: float


@dataclass
class VizPayload:
    """Everything the renderers need — pure data, JSON-serialisable."""

    hero_id: str
    hero_label: str
    train_dates: list[str]
    train_y: list[float]
    windows: list[CVWindow] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @property
    def latest(self) -> CVWindow:
        return self.windows[-1]


# ------------------------------------------------------------------ data prep


def _pick_hero(history: pd.DataFrame, statics: pd.DataFrame, *, prefer_cat: str = "FOODS") -> str:
    """Pick a series with rich, non-flat history — prefer FOODS for visual interest."""
    counts = history[history["y"] > 0].groupby("unique_id", observed=True).size().sort_values(ascending=False)
    if "cat_id" in statics.columns:
        ids = statics.loc[statics["cat_id"].astype(str) == prefer_cat, "unique_id"].astype(str)
        pool = counts[counts.index.astype(str).isin(ids)]
        if not pool.empty:
            return str(pool.index[0])
    if counts.empty:
        raise ValueError("history.parquet has no positive-sales rows; cannot pick a hero series.")
    return str(counts.index[0])


def _seasonal_naive(train_y: np.ndarray, h: int, season: int = 7) -> np.ndarray:
    """Repeat the trailing ``season`` days forward to length ``h``."""
    if len(train_y) < season:
        return np.full(h, float(np.mean(train_y)) if len(train_y) else 0.0)
    last = train_y[-season:]
    reps = (h + season - 1) // season
    return np.tile(last, reps)[:h]


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def _attach_statics(history: pd.DataFrame, statics: pd.DataFrame, static_cols: list[str]) -> pd.DataFrame:
    """Merge static features onto ``history`` so ``predict(new_df=...)`` works."""
    if not static_cols or statics.empty:
        return history
    cols = ["unique_id", *static_cols]
    out = history.merge(statics[cols], on="unique_id", how="left")
    for c in static_cols:
        if c in out.columns and out[c].dtype == "object":
            out[c] = out[c].astype("category")
    return out


def _predict_window(
    fcst,
    *,
    history: pd.DataFrame,
    statics: pd.DataFrame,
    static_cols: list[str],
    hero: str,
    cutoff: pd.Timestamp,
    horizon: int,
) -> np.ndarray:
    """Forecast ``horizon`` days for ``hero`` starting from ``cutoff``+1.

    Slices the bundled history at ``cutoff``, hydrates statics, calls
    ``MLForecast.predict(new_df=...)``. Returns the LightGBM forecast as a
    1-D float array of length ``horizon``.
    """
    new_df = history[(history["unique_id"] == hero) & (history["ds"] <= cutoff)].copy()
    new_df = _attach_statics(new_df, statics, static_cols)
    pred = fcst.predict(h=horizon, new_df=new_df)
    pred_col = next(c for c in pred.columns if c not in ("unique_id", "ds"))
    pred = pred.sort_values("ds").reset_index(drop=True)
    return pred[pred_col].to_numpy(dtype=float)[:horizon]


def build_payload(
    *,
    model_dir: Path = DEFAULT_MODEL_DIR,
    long_path: Path = DEFAULT_LONG_PATH,
    horizon: int = 28,
    n_windows: int = 3,
    train_context: int = 84,
) -> VizPayload:
    """Load the fitted artifact, pick a hero, run rolling-origin predictions."""
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Fitted artifact not found at {model_dir}. Run `make train` (after `make prep`) to produce one."
        )

    metadata = json.loads((model_dir / "metadata.json").read_text())
    history = pd.read_parquet(model_dir / "history.parquet")
    statics = pd.read_parquet(model_dir / "statics.parquet")
    history["ds"] = pd.to_datetime(history["ds"])

    hero = _pick_hero(history, statics)
    hero_label = hero.replace("_", " · ")
    static_cols = list(metadata.get("static_features", []))
    logger.info(f"viz: hero series = {hero}")

    if not long_path.exists():
        raise FileNotFoundError(f"Long frame not found at {long_path} — run `make prep` first.")
    long = pd.read_parquet(long_path, columns=["unique_id", "ds", "y"])
    long["ds"] = pd.to_datetime(long["ds"])
    long_hero = long[long["unique_id"] == hero].sort_values("ds").reset_index(drop=True)

    cutoff_meta = pd.Timestamp(metadata["training_cutoff"])
    last_truth = long_hero["ds"].max()
    # Build windows that all sit inside the available truth so RMSE is meaningful.
    # If we have at least horizon truth days after the model's training_cutoff, anchor
    # the latest window there; otherwise back off into in-sample territory.
    anchor = (
        cutoff_meta
        if (long_hero["ds"] > cutoff_meta).sum() >= horizon
        else last_truth - pd.Timedelta(days=horizon)
    )
    cutoffs = [anchor - pd.Timedelta(days=horizon * k) for k in range(n_windows - 1, -1, -1)]

    fcst = joblib.load(model_dir / "model.joblib")
    windows: list[CVWindow] = []
    for cutoff in cutoffs:
        truth_window = long_hero[
            (long_hero["ds"] > cutoff) & (long_hero["ds"] <= cutoff + pd.Timedelta(days=horizon))
        ]
        truth_y = truth_window["y"].to_numpy(dtype=float)
        if len(truth_y) < horizon:
            logger.warning(f"viz: window @ {cutoff.date()} has only {len(truth_y)} truth days; skipping.")
            continue
        try:
            forecast_y = _predict_window(
                fcst,
                history=history,
                statics=statics,
                static_cols=static_cols,
                hero=hero,
                cutoff=cutoff,
                horizon=horizon,
            )
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(f"viz: predict failed @ {cutoff.date()}: {exc}; falling back to seasonal-naive.")
            train_so_far = long_hero[long_hero["ds"] <= cutoff]["y"].to_numpy(dtype=float)
            forecast_y = _seasonal_naive(train_so_far, horizon)
        train_so_far = long_hero[long_hero["ds"] <= cutoff]["y"].to_numpy(dtype=float)
        baseline_y = _seasonal_naive(train_so_far, horizon)
        windows.append(
            CVWindow(
                cutoff=cutoff.strftime("%Y-%m-%d"),
                forecast=[float(v) for v in forecast_y],
                baseline=[float(v) for v in baseline_y],
                truth=[float(v) for v in truth_y],
                rmse_lgbm=_rmse(forecast_y, truth_y),
                rmse_baseline=_rmse(baseline_y, truth_y),
            )
        )
    if not windows:
        raise RuntimeError("Could not build any CV windows — check truth coverage in long.parquet.")

    final_cutoff = pd.Timestamp(windows[-1].cutoff)
    train_tail = long_hero[long_hero["ds"] <= final_cutoff].tail(train_context)

    return VizPayload(
        hero_id=hero,
        hero_label=hero_label,
        train_dates=[d.strftime("%Y-%m-%d") for d in train_tail["ds"]],
        train_y=[float(v) for v in train_tail["y"]],
        windows=windows,
        metadata={
            "training_cutoff": metadata["training_cutoff"],
            "n_series": int(metadata["n_series"]),
            "n_rows": int(metadata.get("n_rows", 0)),
            "lags": list(metadata["lags"]),
            "rolling_windows": list(metadata["rolling_windows"]),
            "horizon": int(horizon),
            "n_windows": int(n_windows),
            "framework": metadata.get("framework", "mlforecast"),
            "framework_version": metadata.get("framework_version", ""),
            "lightgbm_version": metadata.get("lightgbm_version", ""),
            "git_sha": metadata.get("git_sha", ""),
        },
    )


# ----------------------------------------------------------------- SVG render

# Master loop length, all SMIL animations share this dur and use keyTimes
# normalised against it. Tweak if you want a faster/slower walkthrough.
_DUR_S = 14.0


def _kt(*ts: float) -> str:
    """Format a keyTimes string normalised to the master loop length."""
    return ";".join(f"{t / _DUR_S:.4f}".rstrip("0").rstrip(".") or "0" for t in ts)


def _vals(*vs: object) -> str:
    return ";".join(str(v) for v in vs)


def _polyline_points(xs: list[float], ys: list[float]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys, strict=True))


def render_svg(payload: VizPayload, *, width: int = 960, height: int = 540) -> str:
    """Animated SVG that walks fit -> CV split -> features -> predict -> score.

    Single 14-second loop, six phases. Uses SMIL (works in GitHub README,
    most browsers, and offline). Renderers that strip SMIL (some PDF
    converters, GH camo proxy on referenced URLs) get the final-frame
    static composition for free.
    """
    win = payload.latest
    train = np.asarray(payload.train_y, dtype=float)
    truth = np.asarray(win.truth, dtype=float)
    fc = np.asarray(win.forecast, dtype=float)
    base = np.asarray(win.baseline, dtype=float)

    # --- layout ---
    px_left, px_right = 60, 900
    py_top, py_bot = 95, 360
    chart_w = px_right - px_left
    chart_h = py_bot - py_top
    n_train = len(train)
    horizon = len(truth)
    n_total = n_train + horizon
    step_x = chart_w / max(n_total - 1, 1)
    cutoff_x = px_left + (n_train - 1) * step_x

    y_max = float(max(train.max() if n_train else 1.0, truth.max(), fc.max(), base.max(), 1.0))
    y_max *= 1.15

    def yp(v: float) -> float:
        return py_bot - (v / y_max) * chart_h

    train_xs = [px_left + i * step_x for i in range(n_train)]
    test_xs = [px_left + (n_train - 1 + i) * step_x for i in range(horizon)]

    train_pts = _polyline_points(train_xs, [yp(v) for v in train])
    truth_pts = _polyline_points(test_xs, [yp(v) for v in truth])
    fc_pts = _polyline_points(test_xs, [yp(v) for v in fc])
    base_pts = _polyline_points(test_xs, [yp(v) for v in base])

    # error bars: vertical segments between forecast and truth at each test day
    err_bars = "".join(
        f'<line x1="{x:.2f}" y1="{yp(t):.2f}" x2="{x:.2f}" y2="{yp(f):.2f}" class="err"/>'
        for x, t, f in zip(test_xs, truth, fc, strict=True)
    )

    # --- leaderboard bars (RMSE compared) ---
    lb_x = px_left + 80
    lb_y = 415
    lb_w = 320
    lb_h = 90
    rmse_max = max(win.rmse_lgbm, win.rmse_baseline, 1e-9) * 1.15
    bar_lgbm_w = (win.rmse_lgbm / rmse_max) * lb_w
    bar_base_w = (win.rmse_baseline / rmse_max) * lb_w
    lift_pct = (
        (win.rmse_baseline - win.rmse_lgbm) / win.rmse_baseline * 100.0 if win.rmse_baseline > 1e-9 else 0.0
    )

    # --- feature pills (lags / rolls / price / snap / event) ---
    lags = payload.metadata.get("lags", [7, 14, 28])
    rolls = payload.metadata.get("rolling_windows", [7, 28])
    pill_labels = [f"lag {x}" for x in lags] + [f"roll {x}" for x in rolls] + ["price", "snap", "is_event"]
    pill_y = py_top - 12
    pill_xs: list[float] = []
    pill_x = cutoff_x + 12
    for _ in pill_labels:
        pill_xs.append(pill_x)
        pill_x += 56
    pills_svg = "".join(
        f"""<g class="pill" opacity="1">
              <rect x="{x:.1f}" y="{pill_y - 16:.1f}" width="50" height="18" rx="9" />
              <text x="{x + 25:.1f}" y="{pill_y - 3:.1f}" text-anchor="middle">{escape(label)}</text>
              <animate attributeName="opacity"
                       values="0;0;0;1;1;1;1"
                       keyTimes="{_kt(0, 4.2, 4.5 + i * 0.06, 5.0 + i * 0.06, 9.5, _DUR_S - 0.05, _DUR_S)}"
                       dur="{_DUR_S}s" repeatCount="indefinite"/>
            </g>"""
        for i, (x, label) in enumerate(zip(pill_xs, pill_labels, strict=True))
    )

    # --- caption sequence ---
    captions = [
        ("setup", "30,490 series · pick one", 0.0, 1.8),
        ("ctx", f"Training context — last {n_train} days of {payload.hero_label}", 1.8, 3.0),
        ("split", f"Rolling-origin CV: hold out h={horizon} days", 3.0, 4.5),
        ("feat", f"Build features at the cutoff: {len(pill_labels)} columns", 4.5, 6.5),
        ("fit", "Fit LightGBM (Tweedie) — global model across all series", 6.5, 8.0),
        ("pred", f"Predict {horizon} days from {win.cutoff}", 8.0, 10.0),
        (
            "score",
            f"Score vs truth — RMSE LGBM {win.rmse_lgbm:.2f} vs Naive {win.rmse_baseline:.2f}",
            10.0,
            14.0,
        ),
    ]

    # The final ("score") caption stays visible at the end so non-SMIL
    # renderers (VSCode markdown preview etc.) show a meaningful frame.
    def _caption_attrs(idx: int) -> tuple[str, str]:
        if idx == len(captions) - 1:
            return ("1", "0;1;1;1")
        return ("0", "0;1;1;0")

    caption_svg = "".join(
        f"""<text x="60" y="68" class="caption" opacity="{_caption_attrs(idx)[0]}">{escape(text)}
              <animate attributeName="opacity"
                       values="{_caption_attrs(idx)[1]}"
                       keyTimes="{_kt(0, t0 + 0.2, t1 - 0.2, t1)}"
                       dur="{_DUR_S}s" repeatCount="indefinite"/>
            </text>"""
        for idx, (_, text, t0, t1) in enumerate(captions)
    )

    # --- SMIL helpers for the chart layers ---
    # Animation values stay at opacity=1 at the end of the loop so the static
    # final-frame attribute is also the SMIL final-frame value — non-SMIL
    # renderers (VSCode preview, thumbnail tools) display the fully-composed
    # pipeline; SMIL viewers see the same composition built up over time.
    def reveal(start: float, span: float = 1.5) -> str:
        """Animate opacity 0 → 1 across [start, start+span]; hold visible."""
        return (
            f'<animate attributeName="opacity" '
            f'values="0;0;1;1;1" '
            f'keyTimes="{_kt(0, start, start + span, _DUR_S - 0.05, _DUR_S)}" '
            f'dur="{_DUR_S}s" repeatCount="indefinite"/>'
        )

    # Static axis ticks: every 14 days
    tick_lines = []
    for i in range(0, n_total, 14):
        x = px_left + i * step_x
        tick_lines.append(f'<line x1="{x:.2f}" y1="{py_bot}" x2="{x:.2f}" y2="{py_bot + 4}" class="tick"/>')
    # Y axis grid: 4 horizontal lines
    grid = "".join(
        f'<line x1="{px_left}" y1="{py_top + i * (chart_h / 4):.2f}" '
        f'x2="{px_right}" y2="{py_top + i * (chart_h / 4):.2f}" class="grid"/>'
        for i in range(1, 4)
    )

    lift_label = f"-{lift_pct:.1f}% RMSE" if lift_pct >= 0 else f"+{abs(lift_pct):.1f}% RMSE"

    meta_line = (
        f"{payload.metadata['n_series']:,} series  ·  "
        f"trained @ {payload.metadata['training_cutoff']}  ·  "
        f"{payload.metadata['framework']} {payload.metadata.get('framework_version', '')}"
    )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img"
     aria-label="M5 forecasting pipeline animation: training context, rolling-origin CV split, feature build, LightGBM prediction, scoring against held-out truth.">
  <title>M5 pipeline — fit, predict, score (animated)</title>
  <desc>Animated SVG visualising the M5 forecasting pipeline on a single hero
  series ({escape(payload.hero_label)}): trailing training context draws in,
  the cross-validation cutoff slides into place, feature pills appear at the
  cutoff, the LightGBM forecast is drawn alongside a Seasonal-Naive baseline,
  truth overlays with error bars, and a final RMSE comparison settles in.</desc>
  <style><![CDATA[
    .panel  {{ fill: #0d1117; }}
    .border {{ fill: none; stroke: #30363d; stroke-width: 1; }}
    .title  {{ fill: #f0f6fc; font: 600 18px ui-sans-serif, system-ui, sans-serif; }}
    .caption{{ fill: #8b949e; font: 13px ui-monospace, SFMono-Regular, monospace; }}
    .meta   {{ fill: #6e7681; font: 11px ui-monospace, SFMono-Regular, monospace; }}
    .grid   {{ stroke: #21262d; stroke-width: 1; stroke-dasharray: 2 4; }}
    .tick   {{ stroke: #484f58; stroke-width: 1; }}
    .axis   {{ stroke: #30363d; stroke-width: 1; }}
    .train  {{ stroke: #c9d1d9; stroke-width: 1.6; fill: none; }}
    .truth  {{ stroke: #58a6ff; stroke-width: 2.0; fill: none; stroke-linecap: round; }}
    .fc     {{ stroke: #f778ba; stroke-width: 2.2; fill: none; stroke-linecap: round; }}
    .base   {{ stroke: #d29922; stroke-width: 1.6; fill: none; stroke-dasharray: 4 3; }}
    .err    {{ stroke: #f85149; stroke-width: 1.4; opacity: 0.55; }}
    .train-mask {{ fill: #58a6ff; opacity: 0.04; }}
    .test-mask  {{ fill: #f778ba; opacity: 0.07; }}
    .cutoff {{ stroke: #f778ba; stroke-width: 1.6; stroke-dasharray: 4 4; }}
    .legend-sw {{ stroke-width: 3; stroke-linecap: round; }}
    .legend-tx {{ fill: #c9d1d9; font: 12px ui-monospace, SFMono-Regular, monospace; }}
    .pill rect {{ fill: #161b22; stroke: #30363d; stroke-width: 1; }}
    .pill text {{ fill: #8b949e; font: 600 9.5px ui-monospace, SFMono-Regular, monospace; letter-spacing: .02em; }}
    .lb-bg  {{ fill: #161b22; stroke: #30363d; stroke-width: 1; }}
    .lb-bar-base {{ fill: #d29922; }}
    .lb-bar-lgbm {{ fill: #f778ba; }}
    .lb-label {{ fill: #c9d1d9; font: 12px ui-monospace, SFMono-Regular, monospace; }}
    .lb-num {{ fill: #f0f6fc; font: 600 13px ui-monospace, SFMono-Regular, monospace; }}
    .lift-ok {{ fill: #3fb950; font: 700 16px ui-monospace, SFMono-Regular, monospace; }}
    .lift-bad{{ fill: #f85149; font: 700 16px ui-monospace, SFMono-Regular, monospace; }}
  ]]></style>
  <defs>
    <!-- Static rect width = final reveal width so non-SMIL renderers show
         the fully-revealed lines. SMIL viewers see the animation override. -->
    <clipPath id="reveal-train">
      <rect x="{px_left - 2}" y="{py_top - 4}" height="{chart_h + 8}" width="{cutoff_x - px_left + 2:.1f}">
        <animate attributeName="width"
          values="0;0;{cutoff_x - px_left + 2:.1f};{cutoff_x - px_left + 2:.1f};{cutoff_x - px_left + 2:.1f}"
          keyTimes="{_kt(0, 1.8, 3.0, _DUR_S - 0.05, _DUR_S)}"
          dur="{_DUR_S}s" repeatCount="indefinite"/>
      </rect>
    </clipPath>
    <clipPath id="reveal-fc">
      <rect x="{cutoff_x:.2f}" y="{py_top - 4}" height="{chart_h + 8}" width="{px_right - cutoff_x + 2:.1f}">
        <animate attributeName="width"
          values="0;0;{px_right - cutoff_x + 2:.1f};{px_right - cutoff_x + 2:.1f};{px_right - cutoff_x + 2:.1f}"
          keyTimes="{_kt(0, 6.5, 8.0, _DUR_S - 0.05, _DUR_S)}"
          dur="{_DUR_S}s" repeatCount="indefinite"/>
      </rect>
    </clipPath>
    <clipPath id="reveal-truth">
      <rect x="{cutoff_x:.2f}" y="{py_top - 4}" height="{chart_h + 8}" width="{px_right - cutoff_x + 2:.1f}">
        <animate attributeName="width"
          values="0;0;{px_right - cutoff_x + 2:.1f};{px_right - cutoff_x + 2:.1f};{px_right - cutoff_x + 2:.1f}"
          keyTimes="{_kt(0, 8.5, 9.7, _DUR_S - 0.05, _DUR_S)}"
          dur="{_DUR_S}s" repeatCount="indefinite"/>
      </rect>
    </clipPath>
  </defs>

  <rect class="panel" width="{width}" height="{height}" rx="10"/>
  <rect class="border" x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="10"/>

  <text class="title" x="60" y="42">M5 pipeline · fit → predict → score</text>
  <text class="meta" x="60" y="528">{escape(meta_line)}</text>
  <text class="meta" x="{width - 60}" y="528" text-anchor="end">hero · {escape(payload.hero_label)}</text>

  {caption_svg}

  <g>
    {grid}
    <line x1="{px_left}" y1="{py_top}" x2="{px_left}" y2="{py_bot}" class="axis"/>
    <line x1="{px_left}" y1="{py_bot}" x2="{px_right}" y2="{py_bot}" class="axis"/>
    {"".join(tick_lines)}
  </g>

  <g class="train-mask" opacity="1">
    <rect x="{px_left}" y="{py_top}" width="{cutoff_x - px_left:.2f}" height="{chart_h}"/>
    {reveal(3.0, 0.5)}
  </g>
  <g class="test-mask" opacity="1">
    <rect x="{cutoff_x:.2f}" y="{py_top}" width="{px_right - cutoff_x:.2f}" height="{chart_h}"/>
    {reveal(3.2, 0.5)}
  </g>

  <g clip-path="url(#reveal-train)">
    <polyline class="train" points="{train_pts}"/>
  </g>

  <g opacity="1">
    <line class="cutoff" x1="{cutoff_x:.2f}" y1="{py_top - 6}" x2="{cutoff_x:.2f}" y2="{py_bot + 6}"/>
    <text class="caption" x="{cutoff_x:.2f}" y="{py_top - 24}" text-anchor="middle">cutoff · {escape(win.cutoff)}</text>
    {reveal(3.0, 0.6)}
  </g>

  {pills_svg}

  <g clip-path="url(#reveal-fc)">
    <polyline class="base" points="{base_pts}"/>
    <polyline class="fc" points="{fc_pts}"/>
  </g>

  <g clip-path="url(#reveal-truth)">
    <polyline class="truth" points="{truth_pts}"/>
    {err_bars}
  </g>

  <!-- Legend (top-right of plot) -->
  <g transform="translate({px_right - 200},{py_top + 4})">
    <line class="legend-sw" x1="0" y1="0" x2="14" y2="0" stroke="#c9d1d9"/>
    <text class="legend-tx" x="20" y="4">train</text>
    <line class="legend-sw" x1="62" y1="0" x2="76" y2="0" stroke="#58a6ff"/>
    <text class="legend-tx" x="82" y="4">truth</text>
    <line class="legend-sw" x1="124" y1="0" x2="138" y2="0" stroke="#f778ba"/>
    <text class="legend-tx" x="144" y="4">LGBM</text>
  </g>

  <!-- Leaderboard panel -->
  <g transform="translate({lb_x},{lb_y})" opacity="1">
    {reveal(10.0, 0.7)}
    <rect class="lb-bg" x="-12" y="-12" width="{lb_w + 24}" height="{lb_h + 24}" rx="6"/>
    <text class="lb-label" x="0" y="6">RMSE on the held-out window</text>

    <text class="lb-label" x="0" y="32">Seasonal Naive</text>
    <rect class="lb-bar-base" x="120" y="20" rx="3"
          height="14" width="{bar_base_w:.1f}">
      <animate attributeName="width" values="0;0;{bar_base_w:.1f};{bar_base_w:.1f};{bar_base_w:.1f}"
               keyTimes="{_kt(0, 10.4, 11.4, _DUR_S - 0.05, _DUR_S)}"
               dur="{_DUR_S}s" repeatCount="indefinite"/>
    </rect>
    <text class="lb-num" x="{lb_w + 6}" y="32">{win.rmse_baseline:.2f}</text>

    <text class="lb-label" x="0" y="62">LightGBM</text>
    <rect class="lb-bar-lgbm" x="120" y="50" rx="3"
          height="14" width="{bar_lgbm_w:.1f}">
      <animate attributeName="width" values="0;0;{bar_lgbm_w:.1f};{bar_lgbm_w:.1f};{bar_lgbm_w:.1f}"
               keyTimes="{_kt(0, 10.6, 11.6, _DUR_S - 0.05, _DUR_S)}"
               dur="{_DUR_S}s" repeatCount="indefinite"/>
    </rect>
    <text class="lb-num" x="{lb_w + 6}" y="62">{win.rmse_lgbm:.2f}</text>

    <text class="{"lift-ok" if lift_pct >= 0 else "lift-bad"}" x="0" y="92">{escape(lift_label)}</text>
  </g>
</svg>
"""


# --------------------------------------------------------------- HTML render


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>M5 pipeline — fit · predict · score</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    :root {
      --bg: #0d1117; --panel: #161b22; --border: #30363d; --fg: #e6edf3;
      --muted: #8b949e; --train: #c9d1d9; --truth: #58a6ff; --fc: #f778ba;
      --base: #d29922; --err: #f85149; --ok: #3fb950;
    }
    @media (prefers-color-scheme: light) {
      :root {
        --bg: #ffffff; --panel: #f6f8fa; --border: #d0d7de; --fg: #1f2328;
        --muted: #656d76; --train: #57606a; --truth: #0969da; --fc: #bf3989;
        --base: #9a6700; --err: #cf222e; --ok: #1a7f37;
      }
    }
    body { background: var(--bg); color: var(--fg); margin: 0;
           font: 14px ui-sans-serif, system-ui, -apple-system, sans-serif; }
    main { max-width: 1100px; margin: 0 auto; padding: 24px; }
    h1 { margin: 0 0 4px; font-size: 22px; font-weight: 600; }
    p.lead { color: var(--muted); margin: 0 0 18px; }
    .panel { background: var(--panel); border: 1px solid var(--border);
             border-radius: 8px; padding: 14px 18px; }
    .controls { display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
                margin: 14px 0; }
    .controls button { background: var(--panel); color: var(--fg);
                       border: 1px solid var(--border); border-radius: 6px;
                       padding: 6px 14px; cursor: pointer; font: inherit; }
    .controls button:hover { border-color: var(--fc); }
    .controls .group { display: flex; align-items: center; gap: 8px; }
    .controls label { color: var(--muted); }
    .meta-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;
                margin-top: 14px; }
    .meta-row .cell { background: var(--panel); border: 1px solid var(--border);
                      border-radius: 6px; padding: 8px 12px; }
    .meta-row .cell .k { color: var(--muted); font-size: 11px;
                         text-transform: uppercase; letter-spacing: .05em; }
    .meta-row .cell .v { font-size: 16px; font-weight: 600; margin-top: 2px; }
    svg .axis path, svg .axis line { stroke: var(--border); }
    svg .axis text { fill: var(--muted); }
    .legend { display: flex; gap: 16px; flex-wrap: wrap; margin: 8px 0; }
    .legend span { display: inline-flex; align-items: center; gap: 6px;
                   color: var(--muted); font-family: ui-monospace, SFMono-Regular, monospace;
                   font-size: 12px; }
    .legend i { width: 14px; height: 0; border-top: 3px solid var(--train); }
    .tooltip { position: absolute; pointer-events: none; background: var(--panel);
               border: 1px solid var(--border); border-radius: 6px; padding: 6px 10px;
               font-family: ui-monospace, SFMono-Regular, monospace; font-size: 12px;
               opacity: 0; transition: opacity 80ms; }
    a { color: var(--truth); }
    code { background: var(--panel); padding: 1px 6px; border-radius: 4px;
           border: 1px solid var(--border); font-size: 12px; }
  </style>
</head>
<body>
<main>
  <h1>M5 pipeline · fit · predict · score</h1>
  <p class="lead" id="lead"></p>

  <div class="legend">
    <span><i style="border-color:var(--train)"></i> training context</span>
    <span><i style="border-color:var(--truth)"></i> held-out truth</span>
    <span><i style="border-color:var(--fc)"></i> LightGBM forecast</span>
    <span><i style="border-color:var(--base);border-style:dashed"></i> Seasonal Naive(7) baseline</span>
  </div>

  <div class="panel">
    <svg id="chart" viewBox="0 0 1040 460" preserveAspectRatio="xMidYMid meet"
         style="width:100%;height:auto;display:block"></svg>
  </div>

  <div class="controls">
    <div class="group">
      <button id="play">▶ Play</button>
      <button id="reset">↺ Restart</button>
    </div>
    <div class="group">
      <label for="window">CV window</label>
      <select id="window"></select>
    </div>
    <div class="group">
      <label>phase</label>
      <input id="scrub" type="range" min="0" max="1000" value="0" style="width:240px"/>
      <span id="phase-name" style="color:var(--muted)"></span>
    </div>
  </div>

  <div class="meta-row">
    <div class="cell"><div class="k">hero series</div><div class="v" id="m-hero"></div></div>
    <div class="cell"><div class="k">cutoff</div><div class="v" id="m-cutoff"></div></div>
    <div class="cell"><div class="k">RMSE · LGBM</div><div class="v" id="m-lgbm" style="color:var(--fc)"></div></div>
    <div class="cell"><div class="k">RMSE · Seasonal Naive</div><div class="v" id="m-base" style="color:var(--base)"></div></div>
  </div>

  <p class="lead" style="margin-top:18px">
    Static animated preview lives at
    <code>assets/pipeline.svg</code>. Regenerate with
    <code>uv run m5 viz</code> after <code>make train</code>.
  </p>
</main>
<div class="tooltip" id="tip"></div>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const PAYLOAD = __PAYLOAD__;

const W = 1040, H = 460;
const M = { top: 30, right: 30, bottom: 36, left: 50 };
const innerW = W - M.left - M.right;
const innerH = H - M.top - M.bottom;

const svg = d3.select("#chart");
const root = svg.append("g").attr("transform", `translate(${M.left},${M.top})`);

document.getElementById("lead").textContent =
  `${PAYLOAD.metadata.n_series.toLocaleString()} series · ` +
  `LightGBM (Tweedie) trained @ ${PAYLOAD.metadata.training_cutoff} · ` +
  `lags ${PAYLOAD.metadata.lags.join("/")} · h=${PAYLOAD.metadata.horizon}`;

document.getElementById("m-hero").textContent = PAYLOAD.hero_label;

// Window picker
const wsel = document.getElementById("window");
PAYLOAD.windows.forEach((w, i) => {
  const opt = document.createElement("option");
  opt.value = i; opt.textContent = `${w.cutoff}  (RMSE LGBM ${w.rmse_lgbm.toFixed(2)})`;
  wsel.appendChild(opt);
});
wsel.value = PAYLOAD.windows.length - 1;

const PHASES = [
  { name: "context",  start: 0.00, end: 0.18, caption: "Trailing training context" },
  { name: "split",    start: 0.18, end: 0.32, caption: "Rolling-origin split (h=28)" },
  { name: "features", start: 0.32, end: 0.50, caption: "Build features at the cutoff" },
  { name: "predict",  start: 0.50, end: 0.72, caption: "Predict 28 days ahead" },
  { name: "score",    start: 0.72, end: 1.00, caption: "Score against held-out truth" },
];

function ease(t) { return t < 0 ? 0 : t > 1 ? 1 : t * t * (3 - 2 * t); }
function frac(p, ph) { return ease((p - ph.start) / (ph.end - ph.start)); }

function buildSeries(winIdx) {
  const w = PAYLOAD.windows[winIdx];
  const train = PAYLOAD.train_y.map((y, i) => ({ i, y, kind: "train" }));
  const offset = train.length;
  const test = w.truth.map((y, i) => ({ i: offset + i, y, kind: "truth" }));
  const fc = w.forecast.map((y, i) => ({ i: offset + i, y, kind: "fc" }));
  const base = w.baseline.map((y, i) => ({ i: offset + i, y, kind: "base" }));
  const yMax = Math.max(...train.map(d => d.y), ...test.map(d => d.y),
                         ...fc.map(d => d.y), ...base.map(d => d.y), 1) * 1.15;
  return { w, train, test, fc, base, offset, total: offset + w.truth.length, yMax };
}

let state = buildSeries(+wsel.value);
let phaseProgress = 0;     // 0..1 across phases
let playing = false;
let raf = null;
let t0 = 0;
const PERIOD = 14000;      // ms per loop

const x = d3.scaleLinear();
const y = d3.scaleLinear();

function rescale(s) {
  x.domain([0, s.total - 1]).range([0, innerW]);
  y.domain([0, s.yMax]).range([innerH, 0]);
}

function setupAxes() {
  root.selectAll(".axis").remove();
  rescale(state);
  const xAxis = d3.axisBottom(x)
    .ticks(8)
    .tickFormat(i => {
      const idx = Math.round(i);
      if (idx < state.offset) {
        return PAYLOAD.train_dates[idx] ? PAYLOAD.train_dates[idx].slice(5) : "";
      }
      const k = idx - state.offset;
      return state.w.cutoff && k >= 0 ? `+${k}d` : "";
    });
  const yAxis = d3.axisLeft(y).ticks(5);
  root.append("g").attr("class","axis").attr("transform",`translate(0,${innerH})`).call(xAxis);
  root.append("g").attr("class","axis").call(yAxis);
}

const lineGen = d3.line().x(d => x(d.i)).y(d => y(d.y)).curve(d3.curveMonotoneX);

function ensure(sel, tag, cls) {
  let el = sel.select("." + cls);
  if (el.empty()) el = sel.append(tag).attr("class", cls);
  return el;
}

function draw() {
  const ph = PHASES.findLast ? PHASES.findLast(p => phaseProgress >= p.start) || PHASES[0]
                              : (function() { let last = PHASES[0]; for (const p of PHASES) if (phaseProgress >= p.start) last = p; return last; })();
  document.getElementById("phase-name").textContent = ph.caption;

  // Train mask (left) + test mask (right)
  const cutX = x(state.offset - 1);
  const trainMaskOpacity = phaseProgress > 0.18 ? 0.04 : 0;
  const testMaskOpacity  = phaseProgress > 0.18 ? 0.07 : 0;
  ensure(root, "rect", "mask-train")
    .attr("x", 0).attr("y", 0)
    .attr("width", cutX).attr("height", innerH)
    .attr("fill", "var(--truth)").attr("opacity", trainMaskOpacity);
  ensure(root, "rect", "mask-test")
    .attr("x", cutX).attr("y", 0)
    .attr("width", Math.max(0, innerW - cutX)).attr("height", innerH)
    .attr("fill", "var(--fc)").attr("opacity", testMaskOpacity);

  // Cutoff line
  ensure(root, "line", "cutoff")
    .attr("x1", cutX).attr("x2", cutX).attr("y1", -6).attr("y2", innerH + 6)
    .attr("stroke", "var(--fc)").attr("stroke-width", 1.5)
    .attr("stroke-dasharray", "4 4")
    .attr("opacity", phaseProgress > 0.18 ? 1 : 0);

  // Train line — reveal by clipping width
  const trainCtx = state.train;
  const trainFrac = ease(Math.min(1, phaseProgress / 0.18));
  ensure(root, "path", "train")
    .datum(trainCtx)
    .attr("fill", "none").attr("stroke", "var(--train)").attr("stroke-width", 1.6)
    .attr("d", lineGen)
    .attr("stroke-dasharray", null);
  // Mask via clip rect
  let trainClip = root.select("#clip-train");
  if (trainClip.empty()) {
    trainClip = root.append("clipPath").attr("id", "clip-train").append("rect")
      .attr("y", -6).attr("height", innerH + 12);
  } else {
    trainClip = trainClip.select("rect");
  }
  trainClip.attr("x", 0).attr("width", cutX * trainFrac);
  root.select(".train").attr("clip-path", "url(#clip-train)");

  // Forecast + baseline — reveal between predict phase
  const fcFrac = ease(Math.max(0, Math.min(1, (phaseProgress - 0.50) / 0.22)));
  ensure(root, "path", "fc")
    .datum(state.fc)
    .attr("fill", "none").attr("stroke", "var(--fc)").attr("stroke-width", 2.2)
    .attr("stroke-linecap", "round")
    .attr("d", lineGen);
  ensure(root, "path", "base")
    .datum(state.base)
    .attr("fill", "none").attr("stroke", "var(--base)").attr("stroke-width", 1.6)
    .attr("stroke-dasharray", "4 3")
    .attr("d", lineGen);
  let fcClip = root.select("#clip-fc");
  if (fcClip.empty()) {
    fcClip = root.append("clipPath").attr("id", "clip-fc").append("rect")
      .attr("y", -6).attr("height", innerH + 12);
  } else { fcClip = fcClip.select("rect"); }
  fcClip.attr("x", cutX).attr("width", (innerW - cutX) * fcFrac);
  root.select(".fc").attr("clip-path", "url(#clip-fc)");
  root.select(".base").attr("clip-path", "url(#clip-fc)");

  // Truth + error bars — score phase
  const truthFrac = ease(Math.max(0, Math.min(1, (phaseProgress - 0.72) / 0.20)));
  ensure(root, "path", "truth")
    .datum(state.test)
    .attr("fill", "none").attr("stroke", "var(--truth)").attr("stroke-width", 2)
    .attr("stroke-linecap", "round")
    .attr("d", lineGen);
  let truthClip = root.select("#clip-truth");
  if (truthClip.empty()) {
    truthClip = root.append("clipPath").attr("id", "clip-truth").append("rect")
      .attr("y", -6).attr("height", innerH + 12);
  } else { truthClip = truthClip.select("rect"); }
  truthClip.attr("x", cutX).attr("width", (innerW - cutX) * truthFrac);
  root.select(".truth").attr("clip-path", "url(#clip-truth)");

  // Error bars — vertical lines from truth to forecast at each test step
  const bars = root.selectAll(".err").data(state.test, d => d.i);
  bars.exit().remove();
  bars.enter().append("line").attr("class", "err")
        .attr("stroke", "var(--err)").attr("stroke-width", 1.4).attr("opacity", 0.55)
      .merge(bars)
        .attr("x1", d => x(d.i)).attr("x2", d => x(d.i))
        .attr("y1", d => y(d.y))
        .attr("y2", (_, i) => y(state.fc[i].y))
        .attr("opacity", truthFrac > 0.4 ? 0.55 * truthFrac : 0);

  // Hover-friendly invisible rectangle for tooltip
  ensure(root, "rect", "hover")
    .attr("x", 0).attr("y", 0).attr("width", innerW).attr("height", innerH)
    .attr("fill", "transparent")
    .on("mousemove", function(ev) {
      const [mx] = d3.pointer(ev, this);
      const i = Math.max(0, Math.min(state.total - 1, Math.round(x.invert(mx))));
      const off = state.offset;
      const tip = document.getElementById("tip");
      let html;
      if (i < off) {
        html = `<b>train</b> · ${PAYLOAD.train_dates[i]}<br/>y = ${state.train[i].y.toFixed(2)}`;
      } else {
        const k = i - off;
        html = `<b>+${k}d</b> after cutoff<br/>` +
               `truth = ${state.test[k].y.toFixed(2)}<br/>` +
               `LGBM = ${state.fc[k].y.toFixed(2)}<br/>` +
               `Naive = ${state.base[k].y.toFixed(2)}`;
      }
      tip.innerHTML = html;
      tip.style.opacity = 1;
      tip.style.left = (ev.pageX + 14) + "px";
      tip.style.top = (ev.pageY - 10) + "px";
    })
    .on("mouseleave", () => { document.getElementById("tip").style.opacity = 0; });
}

function updateMetrics() {
  document.getElementById("m-cutoff").textContent = state.w.cutoff;
  document.getElementById("m-lgbm").textContent = state.w.rmse_lgbm.toFixed(2);
  document.getElementById("m-base").textContent = state.w.rmse_baseline.toFixed(2);
}

function tick(now) {
  if (!playing) return;
  if (!t0) t0 = now;
  phaseProgress = ((now - t0) % PERIOD) / PERIOD;
  document.getElementById("scrub").value = Math.round(phaseProgress * 1000);
  draw();
  raf = requestAnimationFrame(tick);
}

function setPlaying(p) {
  playing = p;
  document.getElementById("play").textContent = p ? "❚❚ Pause" : "▶ Play";
  if (p) { t0 = 0; raf = requestAnimationFrame(tick); }
  else if (raf) { cancelAnimationFrame(raf); raf = null; }
}

document.getElementById("play").addEventListener("click", () => setPlaying(!playing));
document.getElementById("reset").addEventListener("click", () => {
  phaseProgress = 0; t0 = 0;
  document.getElementById("scrub").value = 0;
  draw();
});
document.getElementById("scrub").addEventListener("input", (e) => {
  setPlaying(false);
  phaseProgress = e.target.value / 1000;
  draw();
});
wsel.addEventListener("change", () => {
  state = buildSeries(+wsel.value);
  setupAxes();
  updateMetrics();
  draw();
});

setupAxes();
updateMetrics();
draw();
setPlaying(true);
</script>
</body>
</html>
"""


def render_html(payload: VizPayload) -> str:
    """Return a self-contained HTML page with embedded payload + D3 v7 from CDN."""
    return _HTML_TEMPLATE.replace("__PAYLOAD__", payload.to_json())


# ----------------------------------------------------------------- GIF render


def _ease(t: float) -> float:
    """Smoothstep — mirrors the easing the D3 page uses."""
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return t * t * (3 - 2 * t)


def render_gif(
    payload: VizPayload,
    out_path: Path,
    *,
    fps: int = 12,
    duration: float = 12.0,
    width: int = 720,
    height: int = 405,
) -> Path:
    """Render an animated GIF using matplotlib + Pillow.

    Same payload + phase order as :func:`render_svg`, but encoded to GIF so it
    plays in *every* renderer (VSCode preview, README on web, image viewers,
    PDF exports). File size is roughly 50-100x the SVG, but it is universal.
    """
    # Heavy imports stay inside the function so importing m5.viz stays cheap.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    win = payload.latest
    train = np.asarray(payload.train_y, dtype=float)
    truth = np.asarray(win.truth, dtype=float)
    fc = np.asarray(win.forecast, dtype=float)
    base = np.asarray(win.baseline, dtype=float)
    n_train = len(train)
    h = len(truth)
    n_total = n_train + h

    bg, panel = "#0d1117", "#161b22"
    cl_train = "#c9d1d9"
    cl_truth = "#58a6ff"
    cl_fc = "#f778ba"
    cl_base = "#d29922"
    cl_err = "#f85149"
    cl_text = "#f0f6fc"
    cl_muted = "#8b949e"
    cl_grid = "#21262d"
    cl_axis = "#30363d"

    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor=bg)
    ax = fig.add_axes((0.07, 0.27, 0.88, 0.60))
    ax.set_facecolor(bg)
    ax.set_xlim(-0.5, n_total - 0.5)
    y_max = float(max(train.max() if n_train else 1.0, truth.max(), fc.max(), base.max(), 1.0)) * 1.15
    ax.set_ylim(0, y_max)
    ax.tick_params(colors=cl_muted, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(cl_axis)
    ax.grid(True, color=cl_grid, alpha=0.6, linestyle="--", linewidth=0.8)
    ax.set_xlabel("days from cutoff", color=cl_muted, fontsize=9)
    xticks = list(range(0, n_total + 1, 14))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(i - n_train) for i in xticks])

    x_train = np.arange(n_train)
    x_test = np.arange(n_train, n_total)

    train_mask = ax.axvspan(-0.5, n_train - 0.5, color=cl_truth, alpha=0.04, visible=False)
    test_mask = ax.axvspan(n_train - 0.5, n_total - 0.5, color=cl_fc, alpha=0.07, visible=False)
    cutoff_line = ax.axvline(n_train - 0.5, color=cl_fc, linestyle="--", linewidth=1.4, visible=False)

    (line_train,) = ax.plot([], [], color=cl_train, linewidth=1.6, label="train")
    (line_base,) = ax.plot([], [], color=cl_base, linewidth=1.4, linestyle="--", label="Seasonal Naive")
    (line_fc,) = ax.plot([], [], color=cl_fc, linewidth=2.2, label="LightGBM")
    (line_truth,) = ax.plot([], [], color=cl_truth, linewidth=2.0, label="truth")
    err_collection = ax.vlines([], [], [], colors=cl_err, alpha=0.55, linewidth=1.4)

    legend = ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=8,
        labelcolor=cl_train,
        ncol=4,
    )
    for txt in legend.get_texts():
        txt.set_color(cl_train)

    fig.text(
        0.07,
        0.93,
        "M5 pipeline · fit → predict → score",
        color=cl_text,
        fontsize=13,
        weight=600,
    )
    caption = fig.text(0.07, 0.89, "", color=cl_muted, fontsize=9, family="monospace")
    cutoff_label = ax.text(
        n_train - 0.5,
        y_max * 1.02,
        f"cutoff · {win.cutoff}",
        color=cl_muted,
        fontsize=8,
        ha="center",
        visible=False,
    )

    rmse_max = max(win.rmse_lgbm, win.rmse_baseline, 1e-9) * 1.15
    lb_x0, lb_y0, lb_w, lb_h = 0.07, 0.06, 0.50, 0.14
    lb_bg = mpatches.FancyBboxPatch(
        (lb_x0 - 0.01, lb_y0 - 0.02),
        lb_w + 0.02,
        lb_h + 0.04,
        transform=fig.transFigure,
        facecolor=panel,
        edgecolor=cl_axis,
        linewidth=1,
        boxstyle="round,pad=0.005",
    )
    fig.add_artist(lb_bg)
    fig.text(
        lb_x0,
        lb_y0 + lb_h - 0.005,
        "RMSE on held-out window",
        color=cl_muted,
        fontsize=8.5,
        family="monospace",
    )
    bar_axes_w = lb_w - 0.18
    bar_max = win.rmse_baseline if rmse_max == 0 else rmse_max
    base_bar = mpatches.Rectangle(
        (lb_x0 + 0.16, lb_y0 + 0.07),
        0.0,
        0.025,
        transform=fig.transFigure,
        facecolor=cl_base,
    )
    lgbm_bar = mpatches.Rectangle(
        (lb_x0 + 0.16, lb_y0 + 0.025),
        0.0,
        0.025,
        transform=fig.transFigure,
        facecolor=cl_fc,
    )
    fig.add_artist(base_bar)
    fig.add_artist(lgbm_bar)
    fig.text(lb_x0, lb_y0 + 0.085, "Seasonal Naive", color=cl_text, fontsize=8.5, family="monospace")
    fig.text(lb_x0, lb_y0 + 0.040, "LightGBM", color=cl_text, fontsize=8.5, family="monospace")
    base_num = fig.text(
        lb_x0 + lb_w - 0.005,
        lb_y0 + 0.085,
        "",
        color=cl_text,
        fontsize=9,
        family="monospace",
        weight=600,
        ha="right",
    )
    lgbm_num = fig.text(
        lb_x0 + lb_w - 0.005,
        lb_y0 + 0.040,
        "",
        color=cl_text,
        fontsize=9,
        family="monospace",
        weight=600,
        ha="right",
    )
    lift_pct = (
        (win.rmse_baseline - win.rmse_lgbm) / win.rmse_baseline * 100.0 if win.rmse_baseline > 1e-9 else 0.0
    )
    lift_text = f"-{lift_pct:.1f}% RMSE" if lift_pct >= 0 else f"+{abs(lift_pct):.1f}% RMSE"
    lift_color = "#3fb950" if lift_pct >= 0 else cl_err
    lift_label = fig.text(
        lb_x0 + lb_w + 0.04, lb_y0 + 0.06, "", color=lift_color, fontsize=14, weight=700, family="monospace"
    )

    captions_seq = [
        (0.00, 0.13, f"Training context — last {n_train} days of {payload.hero_label}"),
        (0.13, 0.30, f"Rolling-origin CV: hold out h={h} days"),
        (0.30, 0.50, "Build features at the cutoff (lags / rolls / price / snap)"),
        (0.50, 0.72, f"Predict {h} days from {win.cutoff}"),
        (0.72, 1.00, f"Score vs truth — RMSE LGBM {win.rmse_lgbm:.2f} vs Naive {win.rmse_baseline:.2f}"),
    ]

    def caption_for(p: float) -> str:
        for t0, t1, txt in captions_seq:
            if t0 <= p <= t1:
                return txt
        return captions_seq[-1][2]

    n_frames = max(round(fps * duration), 4)

    def update(frame: int):
        p = frame / (n_frames - 1) if n_frames > 1 else 1.0
        caption.set_text(caption_for(p))

        # Phase 1: train context draws in (0 .. 0.18)
        train_frac = _ease(min(1.0, p / 0.18))
        n_show = max(round(n_train * train_frac), 0)
        line_train.set_data(x_train[:n_show], train[:n_show])

        # Phase 2: cutoff + masks (>= 0.18)
        if p > 0.18:
            train_mask.set_visible(True)
            test_mask.set_visible(True)
            cutoff_line.set_visible(True)
            cutoff_label.set_visible(True)

        # Phase 3: predict (0.50 .. 0.72)
        fc_frac = _ease(max(0.0, min(1.0, (p - 0.50) / 0.22)))
        n_fc = max(round(h * fc_frac), 0)
        line_fc.set_data(x_test[:n_fc], fc[:n_fc])
        line_base.set_data(x_test[:n_fc], base[:n_fc])

        # Phase 4: truth + error bars (0.72 .. 0.92)
        truth_frac = _ease(max(0.0, min(1.0, (p - 0.72) / 0.20)))
        n_truth = max(round(h * truth_frac), 0)
        line_truth.set_data(x_test[:n_truth], truth[:n_truth])
        if n_truth > 0:
            segs = [[(x_test[i], truth[i]), (x_test[i], fc[i])] for i in range(n_truth)]
            err_collection.set_segments(segs)
        else:
            err_collection.set_segments([])

        # Phase 5: leaderboard bars + numbers (0.85 .. 1.0)
        lb_frac = _ease(max(0.0, min(1.0, (p - 0.85) / 0.12)))
        base_bar.set_width(bar_axes_w * (win.rmse_baseline / bar_max) * lb_frac)
        lgbm_bar.set_width(bar_axes_w * (win.rmse_lgbm / bar_max) * lb_frac)
        base_num.set_text(f"{win.rmse_baseline * lb_frac:.2f}" if lb_frac > 0.05 else "")
        lgbm_num.set_text(f"{win.rmse_lgbm * lb_frac:.2f}" if lb_frac > 0.05 else "")
        lift_label.set_text(lift_text if lb_frac > 0.6 else "")

        return [
            line_train,
            line_fc,
            line_base,
            line_truth,
            err_collection,
            cutoff_line,
            train_mask,
            test_mask,
            cutoff_label,
            base_bar,
            lgbm_bar,
            base_num,
            lgbm_num,
            lift_label,
            caption,
        ]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=int(1000 / fps), blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    anim.save(str(out_path), writer=writer)
    plt.close(fig)
    return out_path


# ----------------------------------------------------------- top-level driver


def render_pipeline_viz(
    *,
    model_dir: Path = DEFAULT_MODEL_DIR,
    long_path: Path = DEFAULT_LONG_PATH,
    out_dir: Path = DEFAULT_OUT_DIR,
    horizon: int = 28,
    n_windows: int = 3,
    train_context: int = 84,
    gif: bool = True,
    gif_fps: int = 12,
    gif_duration: float = 12.0,
) -> dict[str, Path]:
    """Build payload + write ``pipeline.svg``, ``pipeline.html``, and (optionally)
    ``pipeline.gif`` under ``out_dir``.

    The GIF is opt-out because matplotlib + PillowWriter takes a few seconds and
    inflates the output ~50-100x vs the SVG. Set ``gif=False`` for the fast path
    (SVG + HTML only).
    """
    payload = build_payload(
        model_dir=model_dir,
        long_path=long_path,
        horizon=horizon,
        n_windows=n_windows,
        train_context=train_context,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = out_dir / "pipeline.svg"
    html_path = out_dir / "pipeline.html"
    svg_path.write_text(render_svg(payload))
    html_path.write_text(render_html(payload))
    paths: dict[str, Path] = {"svg": svg_path, "html": html_path}
    logger.info(
        f"viz: hero={payload.hero_label}  "
        f"RMSE LGBM={payload.latest.rmse_lgbm:.2f}  Naive={payload.latest.rmse_baseline:.2f}"
    )
    logger.info(f"viz: wrote {svg_path} ({svg_path.stat().st_size / 1024:.1f} KB)")
    logger.info(f"viz: wrote {html_path} ({html_path.stat().st_size / 1024:.1f} KB)")
    if gif:
        gif_path = out_dir / "pipeline.gif"
        render_gif(payload, gif_path, fps=gif_fps, duration=gif_duration)
        logger.info(f"viz: wrote {gif_path} ({gif_path.stat().st_size / 1024:.1f} KB)")
        paths["gif"] = gif_path
    return paths
