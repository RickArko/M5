"""Unit tests for ``m5.viz`` — payload helpers + SVG/HTML renderers."""

from __future__ import annotations

import json
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from m5.viz.pipeline import (
    CVWindow,
    VizPayload,
    _rmse,
    _seasonal_naive,
    render_html,
    render_svg,
)

# ---- helpers ---------------------------------------------------------------


def _payload(n_train: int = 84, h: int = 28, n_windows: int = 3) -> VizPayload:
    rng = np.random.default_rng(42)
    base = 5 + 3 * np.sin(np.arange(n_train + h * n_windows) * 2 * np.pi / 7)
    series = np.clip(base + rng.normal(0, 0.5, base.shape), 0, None)
    train_y = series[:n_train].astype(float).tolist()
    train_dates = [f"2024-01-{i + 1:02d}" for i in range(n_train)]
    windows = []
    for k in range(n_windows):
        s = n_train + k * h
        truth = series[s : s + h].astype(float)
        forecast = truth + rng.normal(0, 0.7, h)
        baseline = _seasonal_naive(np.asarray(train_y), h)
        windows.append(
            CVWindow(
                cutoff=f"2024-03-{k + 1:02d}",
                forecast=forecast.tolist(),
                baseline=baseline.tolist(),
                truth=truth.tolist(),
                rmse_lgbm=_rmse(forecast, truth),
                rmse_baseline=_rmse(baseline, truth),
            )
        )
    return VizPayload(
        hero_id="FOODS_3_001_CA_1",
        hero_label="FOODS 3 001 · CA 1",
        train_dates=train_dates,
        train_y=train_y,
        windows=windows,
        metadata={
            "training_cutoff": "2024-03-01",
            "n_series": 30490,
            "n_rows": 12_000_000,
            "lags": [7, 14, 28],
            "rolling_windows": [7, 28],
            "horizon": h,
            "n_windows": n_windows,
            "framework": "mlforecast",
            "framework_version": "1.0.0",
            "lightgbm_version": "4.6.0",
            "git_sha": "deadbeef",
        },
    )


# ---- pure helpers ----------------------------------------------------------


def test_seasonal_naive_repeats_last_week() -> None:
    train = np.arange(14, dtype=float)
    out = _seasonal_naive(train, h=14, season=7)
    expected = np.tile(train[-7:], 2)
    np.testing.assert_array_equal(out, expected)


def test_seasonal_naive_short_history_uses_mean() -> None:
    train = np.array([3.0, 5.0])
    out = _seasonal_naive(train, h=5, season=7)
    assert out.shape == (5,)
    assert np.allclose(out, 4.0)


def test_seasonal_naive_empty_history_safe() -> None:
    out = _seasonal_naive(np.array([], dtype=float), h=3, season=7)
    assert out.shape == (3,)
    assert np.all(out == 0.0)


def test_rmse_perfect_is_zero() -> None:
    a = np.array([1.0, 2.0, 3.0])
    assert _rmse(a, a) == 0.0


def test_rmse_known_value() -> None:
    a = np.array([1.0, 2.0])
    b = np.array([2.0, 4.0])
    assert _rmse(a, b) == pytest.approx(np.sqrt((1 + 4) / 2))


# ---- SVG renderer ----------------------------------------------------------


def test_render_svg_is_well_formed() -> None:
    payload = _payload()
    svg = render_svg(payload)
    # parses as XML
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    assert "viewBox" in root.attrib


def test_render_svg_embeds_hero_label_and_metrics() -> None:
    payload = _payload()
    svg = render_svg(payload)
    assert payload.hero_label in svg
    # latest-window RMSE numbers should appear
    assert f"{payload.latest.rmse_lgbm:.2f}" in svg
    assert f"{payload.latest.rmse_baseline:.2f}" in svg


def test_render_svg_has_smil_animations() -> None:
    payload = _payload()
    svg = render_svg(payload)
    # at least one animate per phase: training reveal, forecast reveal,
    # truth reveal, leaderboard bar, captions, pills
    assert svg.count("<animate") >= 6
    assert 'repeatCount="indefinite"' in svg


def test_render_svg_uses_polylines_for_each_series() -> None:
    payload = _payload()
    svg = render_svg(payload)
    # train, truth, forecast, baseline polylines
    assert svg.count('class="train"') >= 1
    assert svg.count('class="truth"') >= 1
    assert svg.count('class="fc"') >= 1
    assert svg.count('class="base"') >= 1


# ---- HTML renderer ---------------------------------------------------------


def test_render_html_embeds_payload_json() -> None:
    payload = _payload()
    html = render_html(payload)
    # placeholder must be replaced
    assert "__PAYLOAD__" not in html
    # the JSON literal should round-trip
    start = html.index("const PAYLOAD = ")
    end = html.index(";", start)
    embedded = html[start + len("const PAYLOAD = ") : end]
    parsed = json.loads(embedded)
    assert parsed["hero_id"] == payload.hero_id
    assert len(parsed["windows"]) == len(payload.windows)
    assert parsed["windows"][-1]["rmse_lgbm"] == pytest.approx(payload.latest.rmse_lgbm)


def test_render_html_loads_d3_v7() -> None:
    html = render_html(_payload())
    assert "d3.v7" in html


def test_render_html_has_play_pause_and_scrub_controls() -> None:
    html = render_html(_payload())
    assert 'id="play"' in html
    assert 'id="reset"' in html
    assert 'id="scrub"' in html
    assert 'id="window"' in html


# ---- payload helpers -------------------------------------------------------


def test_payload_to_json_round_trips() -> None:
    payload = _payload(n_train=20, h=7, n_windows=2)
    parsed = json.loads(payload.to_json())
    assert parsed["hero_id"] == payload.hero_id
    assert len(parsed["train_y"]) == 20
    assert len(parsed["windows"][0]["forecast"]) == 7
    assert payload.latest is payload.windows[-1]


# ---- SVG static fallback ---------------------------------------------------


def test_render_svg_static_frame_is_composed_not_blank() -> None:
    """Non-SMIL renderers (VSCode preview etc.) must see useful content.

    All elements that would be visible at the end of the animation should
    have static attribute values matching the final composed state, so a
    viewer that strips SMIL still sees the train line + forecast + truth +
    leaderboard etc., not an empty panel.
    """
    payload = _payload()
    svg = render_svg(payload)
    # Pills appear as opacity=1 statically (final state visible)
    assert 'class="pill" opacity="1"' in svg
    # Mask groups visible statically
    assert 'class="train-mask" opacity="1"' in svg
    assert 'class="test-mask" opacity="1"' in svg
    # Animation values end at 1, not 0 (no fade-out)
    assert 'values="0;0;1;1;1"' in svg
    # No initial width="0" on clip rects or leaderboard bars (static = final)
    assert 'width="0"' not in svg


# ---- GIF renderer (smoke; small payload to keep it under ~3s) ---------------


def test_render_gif_writes_a_valid_gif(tmp_path) -> None:
    """Smoke test the matplotlib GIF pipeline on a tiny payload."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")
    from PIL import Image

    from m5.viz.pipeline import render_gif

    payload = _payload(n_train=14, h=7, n_windows=1)
    out = tmp_path / "pipeline.gif"
    render_gif(payload, out, fps=6, duration=2.0, width=320, height=180)
    assert out.exists()
    assert out.stat().st_size > 1024  # > 1 KB
    with Image.open(out) as img:
        assert img.format == "GIF"
        # Aspect-ratio check rather than exact pixel match — matplotlib's
        # `savefig.dpi` default differs across platforms (e.g. 150 on the GH
        # runner vs the figure's 100). render_gif pins dpi at save time, but
        # we keep the assertion tolerant in case a future matplotlib changes
        # the contract.
        w, h = img.size
        assert w >= 320 and h >= 180
        assert abs(w / h - 320 / 180) < 0.01
        # Multi-frame
        assert getattr(img, "n_frames", 1) > 1
