"""Integration tests for Bayesian NegBin CV (optional pymc group)."""

from __future__ import annotations

import pytest

pytest.importorskip("pymc")


@pytest.mark.integration
@pytest.mark.slow
def test_bayesian_cv_toy_panel(toy_with_calendar) -> None:
    from m5.models.bayesian import bayesian_cv

    cv = bayesian_cv(
        toy_with_calendar,
        h=14,
        n_windows=1,
        n_series=2,
        draws=40,
        tune=40,
        chains=2,
    )

    assert list(cv.columns) == ["unique_id", "ds", "cutoff", "y", "Bayes-NegBin"]
    assert cv["unique_id"].nunique() == 2
    assert len(cv) == 2 * 14
    assert cv["Bayes-NegBin"].notna().all()
    assert (cv["Bayes-NegBin"] >= 0).all()


@pytest.mark.integration
@pytest.mark.slow
def test_bayesian_cv_zinb_toy_panel(toy_with_calendar) -> None:
    from m5.models.bayesian import bayesian_cv

    cv = bayesian_cv(
        toy_with_calendar,
        h=14,
        n_windows=1,
        n_series=2,
        draws=40,
        tune=40,
        chains=2,
        likelihood="zinb",
    )

    assert list(cv.columns) == ["unique_id", "ds", "cutoff", "y", "Bayes-ZINB"]
    assert len(cv) == 2 * 14
    assert (cv["Bayes-ZINB"] >= 0).all()


@pytest.mark.integration
@pytest.mark.slow
def test_fit_bayes_hier_zinb_toy_panel(toy_with_calendar) -> None:
    from m5.models.bayesian import (
        extract_posterior,
        fit_bayes_hier_zinb,
        posterior_mean_forecast_hier_zinb,
    )

    panel = toy_with_calendar[
        toy_with_calendar["unique_id"].isin(toy_with_calendar["unique_id"].unique()[:2])
    ]
    idata, enc = fit_bayes_hier_zinb(panel, draws=40, tune=40, chains=2, quiet=True)
    uid = next(iter(enc.series_map))
    future = panel.loc[panel["unique_id"].astype(str) == uid].tail(14)
    post = extract_posterior(idata)
    y_hat = posterior_mean_forecast_hier_zinb(post, future, series_idx=enc.series_map[uid])
    assert len(y_hat) == 14
    assert (y_hat >= 0).all()


@pytest.mark.integration
@pytest.mark.slow
def test_bayesian_routed_cv_toy_panel(toy_with_calendar) -> None:
    from m5.models.bayesian import bayesian_routed_cv

    cv = bayesian_routed_cv(
        toy_with_calendar,
        h=14,
        n_windows=1,
        n_series=3,
        draws=40,
        tune=40,
        chains=2,
        min_hier_series=2,
    )
    assert "Bayes-Routed" in cv.columns
    assert "demand_class" in cv.columns
    assert "route" in cv.columns
    assert len(cv) == 3 * 14
