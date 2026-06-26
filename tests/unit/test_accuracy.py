"""Unit tests for accuracy_for_models."""

from __future__ import annotations

import pandas as pd
import pytest

from m5.evaluation import accuracy_for_models, wrmsse_for_models


@pytest.mark.unit
def test_accuracy_for_models_orders_perfect_first(toy_cv: pd.DataFrame) -> None:
    truth = toy_cv[["unique_id", "ds", "y"]]
    acc = accuracy_for_models(truth, toy_cv, model_cols=["Perfect", "Biased", "Naive"])

    assert acc.loc["Perfect", "MAE"] == 0.0
    assert acc.loc["Perfect", "RMSE"] == 0.0
    assert acc.loc["Biased", "MAE"] == pytest.approx(1.5)
    assert list(acc.index) == sorted(acc.index, key=lambda m: acc.loc[m, "MAE"])


@pytest.mark.unit
def test_accuracy_for_models_matches_wrmsse_ranking(
    toy_cv: pd.DataFrame, toy_train_for_cv: pd.DataFrame
) -> None:
    from m5.evaluation import compute_components

    truth = toy_cv[["unique_id", "ds", "y"]]
    components = compute_components(toy_train_for_cv)
    wr = wrmsse_for_models(truth, toy_cv, components, model_cols=["Perfect", "Biased", "Naive"])
    acc = accuracy_for_models(truth, toy_cv, model_cols=["Perfect", "Biased", "Naive"])

    assert wr.idxmin() == acc["MAE"].idxmin() == "Perfect"
