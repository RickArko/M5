from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display; required when running headless / in CI

import pandas as pd

from m5.evaluation import compute_components
from m5.reporting import build_all_figures, render_report, save_figure
from m5.reporting.report import RunMetadata
from m5.scoring import (
    ScoringInputs,
    bias_variance_decomposition,
    error_concentration,
    fva_chain,
    fva_per_fold,
    fva_scores,
    headline_scores,
    paired_bootstrap_pvalues,
    per_fold_scores,
    per_horizon_scores,
    per_level_scores,
    per_segment_scores,
    residuals_long,
)


def _build_inputs(toy_cv: pd.DataFrame, toy_train_for_cv: pd.DataFrame) -> ScoringInputs:
    statics = toy_train_for_cv.drop_duplicates("unique_id")[
        ["unique_id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    ].reset_index(drop=True)
    return ScoringInputs(
        cv_df=toy_cv,
        train=toy_train_for_cv,
        statics=statics,
        components=compute_components(toy_train_for_cv),
        models=["Perfect", "Biased", "Naive"],
    )


def test_full_render_produces_expected_artifacts(
    toy_cv: pd.DataFrame, toy_train_for_cv: pd.DataFrame, tmp_path: Path
) -> None:
    inp = _build_inputs(toy_cv, toy_train_for_cv)

    headline = headline_scores(inp)
    per_fold = per_fold_scores(inp)
    per_horizon = per_horizon_scores(inp)
    per_level = per_level_scores(inp)
    segment_frames = {
        "cat_id": per_segment_scores(inp, "cat_id"),
        "state_id": per_segment_scores(inp, "state_id"),
    }
    bv = bias_variance_decomposition(inp)
    pvalues = paired_bootstrap_pvalues(inp, n_iter=50, seed=1)
    residuals = residuals_long(inp)
    error_curves = error_concentration(inp)
    fva_star = fva_scores(inp, baseline="Biased", metric="mae")
    fva_pf = fva_per_fold(inp, baseline="Biased", metric="mae")
    fva_chain_df = fva_chain(inp, chain=["Naive", "Biased", "Perfect"], metric="mae")

    bundle = build_all_figures(
        headline=headline,
        per_fold=per_fold,
        per_horizon=per_horizon,
        per_level=per_level,
        segment_frames=segment_frames,
        bv=bv,
        pvalues=pvalues,
        residuals=residuals,
        error_curves=error_curves,
        cv_df=toy_cv,
        train=toy_train_for_cv,
        models=inp.models,
        fva_star=fva_star,
        fva_chain_df=fva_chain_df,
        fva_per_fold_df=fva_pf,
    )

    fig_dir = tmp_path / "figures"
    saved_paths: dict[str, dict[str, Path]] = {}
    for name, fig in bundle.figures.items():
        saved_paths[name] = save_figure(fig, name, out_dir=fig_dir, formats=("png", "svg", "pdf"))
        for fmt, p in saved_paths[name].items():
            assert p.exists(), f"missing {fmt} for {name}"
            assert p.stat().st_size > 0

    # The 12-figure menu includes some that may legitimately skip (calibration
    # without PIs, per_segment with no panels, etc.). Require at least the core 8.
    must_have = {
        "01_leaderboard",
        "03_per_fold_stability",
        "04_per_horizon_curve",
        "06_bias_variance",
        "07_significance_matrix",
        "08_residuals_time",
        "09_residual_histogram",
        "11_forecast_examples",
        "13_fva_star",
        "14_fva_waterfall",
        "15_fva_per_fold",
    }
    assert must_have.issubset(set(bundle.figures.keys())), (
        f"missing figures: {must_have - set(bundle.figures.keys())}"
    )

    metadata = RunMetadata.autodiscover(
        run_id="test", seed=42, horizon=14, n_windows=2, models=inp.models, n_series=3
    )
    paths = render_report(bundle, metadata=metadata, headline=headline, out_dir=tmp_path)
    md = paths["md"].read_text()
    html_doc = paths["html"].read_text()
    assert "M5 Forecast Evaluation" in md
    assert "Headline" in md
    assert "Perfect" in md  # leaderboard table includes the perfect model
    assert "<html" in html_doc
    assert "Perfect" in html_doc
    # Figures referenced in markdown use png; html uses svg by default.
    assert "figures/01_leaderboard.png" in md
    assert "figures/01_leaderboard.svg" in html_doc


def test_save_figure_embeds_metadata(
    toy_cv: pd.DataFrame, toy_train_for_cv: pd.DataFrame, tmp_path: Path
) -> None:
    inp = _build_inputs(toy_cv, toy_train_for_cv)
    bundle = build_all_figures(
        headline=headline_scores(inp),
        per_fold=per_fold_scores(inp),
        per_horizon=per_horizon_scores(inp),
        per_level=pd.DataFrame(),
        segment_frames={},
        bv=bias_variance_decomposition(inp),
        pvalues=None,
        residuals=residuals_long(inp),
        error_curves=error_concentration(inp),
        cv_df=toy_cv,
        train=toy_train_for_cv,
        models=inp.models,
    )
    fig = bundle.figures["01_leaderboard"]
    paths = save_figure(fig, "smoke", out_dir=tmp_path, formats=("svg",), metadata={"M5RunId": "smoke-run"})
    text = paths["svg"].read_text()
    assert "smoke-run" in text
