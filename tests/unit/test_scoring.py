from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from m5.evaluation import compute_components
from m5.scoring import (
    ScoringInputs,
    bias_variance_decomposition,
    discover_models,
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


def _make_inputs(toy_cv: pd.DataFrame, toy_train_for_cv: pd.DataFrame) -> ScoringInputs:
    statics = toy_train_for_cv.drop_duplicates("unique_id")[
        ["unique_id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    ].reset_index(drop=True)
    components = compute_components(toy_train_for_cv)
    return ScoringInputs(
        cv_df=toy_cv,
        train=toy_train_for_cv,
        statics=statics,
        components=components,
        models=["Perfect", "Biased", "Naive"],
    )


def test_discover_models_excludes_protected(toy_cv: pd.DataFrame) -> None:
    cols = discover_models(
        toy_cv, exclude=("item_id", "dept_id", "cat_id", "store_id", "state_id", "sell_price")
    )
    assert set(cols) == {"Perfect", "Biased", "Naive"}


def test_headline_scores_ranks_perfect_first(toy_cv: pd.DataFrame, toy_train_for_cv: pd.DataFrame) -> None:
    inp = _make_inputs(toy_cv, toy_train_for_cv)
    h = headline_scores(inp)
    assert next(iter(h["model"])) == "Perfect"
    assert h.loc[h["model"] == "Perfect", "wrmsse"].iloc[0] == 0.0
    assert (h.loc[h["model"] == "Biased", "wrmsse"].iloc[0]) > 0
    assert {"rmse", "mae", "smape", "bias", "mase"}.issubset(h.columns)
    assert h.loc[h["model"] == "Biased", "bias"].iloc[0] > 0


def test_per_fold_scores_one_row_per_model_per_cutoff(
    toy_cv: pd.DataFrame, toy_train_for_cv: pd.DataFrame
) -> None:
    inp = _make_inputs(toy_cv, toy_train_for_cv)
    pf = per_fold_scores(inp)
    assert len(pf) == toy_cv["cutoff"].nunique() * 3
    assert (pf.loc[pf["model"] == "Perfect", "wrmsse"] == 0).all()


def test_per_horizon_increases_for_biased(toy_cv: pd.DataFrame, toy_train_for_cv: pd.DataFrame) -> None:
    inp = _make_inputs(toy_cv, toy_train_for_cv)
    ph = per_horizon_scores(inp)
    assert (ph.loc[ph["model"] == "Perfect", "wrmsse"] == 0).all()
    biased = ph[ph["model"] == "Biased"].sort_values("h")
    assert len(biased) > 0  # at least one horizon row


def test_per_segment_segments_are_subset_of_statics(
    toy_cv: pd.DataFrame, toy_train_for_cv: pd.DataFrame
) -> None:
    inp = _make_inputs(toy_cv, toy_train_for_cv)
    seg = per_segment_scores(inp, "cat_id")
    assert not seg.empty
    assert set(seg["segment"]).issubset({"FOODS", "HOUSEHOLD"})
    assert (seg.loc[seg["model"] == "Perfect", "wrmsse"] == 0).all()


def test_per_level_covers_all_12_levels(toy_cv: pd.DataFrame, toy_train_for_cv: pd.DataFrame) -> None:
    inp = _make_inputs(toy_cv, toy_train_for_cv)
    pl = per_level_scores(inp)
    # Some levels collapse on a 3-series toy (e.g. one state has one store), but
    # we should still see Perfect scoring 0 wherever it scores at all.
    assert not pl.empty
    assert (pl.loc[pl["model"] == "Perfect", "wrmsse"] < 1e-5).all()
    assert pl["level_idx"].nunique() >= 3


def test_bias_variance_recovers_constant_bias(toy_cv: pd.DataFrame, toy_train_for_cv: pd.DataFrame) -> None:
    inp = _make_inputs(toy_cv, toy_train_for_cv)
    bv = bias_variance_decomposition(inp)
    perfect = bv[bv["model"] == "Perfect"].iloc[0]
    biased = bv[bv["model"] == "Biased"].iloc[0]
    assert perfect["mse"] == 0
    assert biased["bias"] == pytest.approx(1.5, abs=1e-6)
    assert biased["bias_sq"] == pytest.approx(1.5**2, abs=1e-6)
    # Residuals are constant (= +1.5) → variance ≈ 0
    assert biased["variance"] < 1e-9


def test_error_concentration_skips_perfect(toy_cv: pd.DataFrame, toy_train_for_cv: pd.DataFrame) -> None:
    inp = _make_inputs(toy_cv, toy_train_for_cv)
    ec = error_concentration(inp)
    # Perfect contributes zero error so it's skipped (would be a flat undefined curve).
    assert "Perfect" not in set(ec["model"])
    assert {"Biased", "Naive"}.issubset(set(ec["model"]))
    # Cumulative weight ends at 1.0 for each present model.
    for m in ("Biased", "Naive"):
        last = ec[ec["model"] == m].sort_values("rank").iloc[-1]
        assert last["cum_weight"] == pytest.approx(1.0)
        assert last["cum_error_share"] == pytest.approx(1.0)


def test_paired_bootstrap_is_reproducible(toy_cv: pd.DataFrame, toy_train_for_cv: pd.DataFrame) -> None:
    inp = _make_inputs(toy_cv, toy_train_for_cv)
    p1 = paired_bootstrap_pvalues(inp, n_iter=200, seed=7)
    p2 = paired_bootstrap_pvalues(inp, n_iter=200, seed=7)
    assert np.array_equal(p1.values, p2.values)
    # Diagonal is the trivial "A no better than A" → 1.0 by construction.
    assert (np.diag(p1.values) == 1.0).all()
    # Perfect should beat Biased: P(Perfect ≥ Biased) is small.
    assert p1.loc["Perfect", "Biased"] < 0.05
    assert p1.loc["Biased", "Perfect"] > 0.95


def test_residuals_long_has_one_row_per_model_per_obs(
    toy_cv: pd.DataFrame, toy_train_for_cv: pd.DataFrame
) -> None:
    inp = _make_inputs(toy_cv, toy_train_for_cv)
    res = residuals_long(inp)
    assert set(res["model"]) == {"Perfect", "Biased", "Naive"}
    assert (res.loc[res["model"] == "Perfect", "residual"] == 0).all()
    assert np.allclose(res.loc[res["model"] == "Biased", "residual"], 1.5)


# ---------------------------------------------------------------------------
# Forecast Value Added — Vandeput, 2021
# ---------------------------------------------------------------------------
def test_fva_perfect_beats_baseline(toy_cv: pd.DataFrame, toy_train_for_cv: pd.DataFrame) -> None:
    inp = _make_inputs(toy_cv, toy_train_for_cv)
    fva = fva_scores(inp, baseline="Biased", metric="mae")
    # Baseline is excluded from the result; only Perfect and Naive remain.
    assert set(fva["model"]) == {"Perfect", "Naive"}
    perfect = fva[fva["model"] == "Perfect"].iloc[0]
    assert perfect["fva_abs"] > 0
    assert perfect["fva_pct"] > 0
    # Bias is exactly 1.5 → MAE(Biased) = 1.5; Perfect MAE = 0 → FVA = 1.5.
    assert perfect["fva_abs"] == pytest.approx(1.5, abs=1e-6)


def test_fva_destroying_value_is_negative(toy_cv: pd.DataFrame, toy_train_for_cv: pd.DataFrame) -> None:
    inp = _make_inputs(toy_cv, toy_train_for_cv)
    fva = fva_scores(inp, baseline="Perfect", metric="mae")
    biased = fva[fva["model"] == "Biased"].iloc[0]
    assert biased["fva_abs"] < 0  # Biased destroys value vs Perfect


def test_fva_baseline_must_exist(toy_cv: pd.DataFrame, toy_train_for_cv: pd.DataFrame) -> None:
    inp = _make_inputs(toy_cv, toy_train_for_cv)
    with pytest.raises(ValueError, match=r"not in inp\.models"):
        fva_scores(inp, baseline="DoesNotExist")


def test_fva_metric_validated(toy_cv: pd.DataFrame, toy_train_for_cv: pd.DataFrame) -> None:
    inp = _make_inputs(toy_cv, toy_train_for_cv)
    with pytest.raises(ValueError, match="Unknown FVA metric"):
        fva_scores(inp, baseline="Biased", metric="bogus")


def test_fva_chain_first_row_is_baseline(toy_cv: pd.DataFrame, toy_train_for_cv: pd.DataFrame) -> None:
    inp = _make_inputs(toy_cv, toy_train_for_cv)
    chain = fva_chain(inp, chain=["Naive", "Biased", "Perfect"], metric="mae")
    assert len(chain) == 3
    assert chain.iloc[0]["step"] == "Naive"
    assert chain.iloc[0]["fva_abs"] == 0.0
    assert chain.iloc[0]["is_baseline"]
    # Perfect is the last step and has zero error → its FVA equals previous error.
    assert chain.iloc[-1]["error_step"] == 0.0
    assert chain.iloc[-1]["fva_abs"] > 0


def test_fva_per_fold_one_row_per_model_per_cutoff(
    toy_cv: pd.DataFrame, toy_train_for_cv: pd.DataFrame
) -> None:
    inp = _make_inputs(toy_cv, toy_train_for_cv)
    pf = fva_per_fold(inp, baseline="Biased", metric="mae")
    # 2 folds x 2 non-baseline models = 4 rows.
    assert len(pf) == toy_cv["cutoff"].nunique() * 2
    # Perfect adds value across every fold.
    assert (pf.loc[pf["model"] == "Perfect", "fva_abs"] > 0).all()
