"""Unit: Pydantic v2 validators reject malformed requests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from m5.serve.schemas import (
    HistoryPoint,
    ModelInfoResponse,
    PredictByIdRequest,
    PredictRequest,
    ProblemDetails,
)


def test_history_point_rejects_negative_y() -> None:
    with pytest.raises(ValidationError):
        HistoryPoint(unique_id="X", ds="2024-01-01", y=-1.0)


def test_history_point_rejects_blank_id() -> None:
    with pytest.raises(ValidationError):
        HistoryPoint(unique_id="", ds="2024-01-01", y=1.0)


def test_history_point_parses_iso_date() -> None:
    p = HistoryPoint(unique_id="X", ds="2024-01-01", y=1.0)
    assert p.ds.isoformat() == "2024-01-01"


def test_predict_request_requires_history() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(horizon=7, history=[])


def test_predict_request_horizon_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(horizon=0, history=[HistoryPoint(unique_id="X", ds="2024-01-01", y=1.0)])


def test_predict_request_levels_must_be_in_open_interval() -> None:
    base = {"horizon": 1, "history": [{"unique_id": "X", "ds": "2024-01-01", "y": 1.0}]}
    for bad in ([0], [100], [-5], [50, 100]):
        with pytest.raises(ValidationError):
            PredictRequest(**base, level=bad)


def test_predict_request_levels_dedup_and_sort() -> None:
    req = PredictRequest(
        horizon=1,
        history=[HistoryPoint(unique_id="X", ds="2024-01-01", y=1.0)],
        level=[95, 80, 80],
    )
    assert req.level == [80, 95]


def test_predict_request_extra_keys_forbidden() -> None:
    """`extra="forbid"` keeps schemas honest — unknown payload keys are a client bug."""
    with pytest.raises(ValidationError):
        PredictRequest.model_validate(
            {
                "horizon": 1,
                "history": [{"unique_id": "X", "ds": "2024-01-01", "y": 1.0}],
                "rogue": "field",
            }
        )


def test_predict_by_id_requires_at_least_one_id() -> None:
    with pytest.raises(ValidationError):
        PredictByIdRequest(horizon=7, unique_ids=[])


def test_predict_by_id_dedup_is_route_responsibility() -> None:
    """Schema accepts duplicates; the route layer de-duplicates so the response shape is stable."""
    req = PredictByIdRequest(horizon=7, unique_ids=["A", "A", "B"])
    assert req.unique_ids == ["A", "A", "B"]


def test_problem_details_serializes_with_default_type() -> None:
    p = ProblemDetails(title="oops", status=400)
    body = p.model_dump()
    assert body["type"] == "about:blank"
    assert body["status"] == 400


def test_model_info_response_rejects_unknown_fields_via_loose_base() -> None:
    """Response models are frozen — mutation raises."""
    info = ModelInfoResponse(
        model_kind="lgbm",
        framework="mlforecast",
        framework_version="1.0.0",
        trained_at="20250101T000000Z",
        git_sha="abc",
        training_cutoff="2024-01-01",
        horizon_default=28,
        n_series=10,
        lags=[7, 14, 28],
        rolling_windows=[7, 28],
        min_history_required=56,
        static_features=["item_id"],
    )
    with pytest.raises(ValidationError):
        info.model_kind = "stats"  # type: ignore[misc]
