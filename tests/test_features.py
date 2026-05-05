from __future__ import annotations

import pandas as pd

from m5.features import add_date_features, add_event_flag, add_price_features, add_snap_flag, build_feature_frame


def test_date_features_attach_expected_cols(toy_long: pd.DataFrame) -> None:
    out = add_date_features(toy_long.copy())
    for col in ("dayofweek", "day", "week", "month", "year", "is_weekend"):
        assert col in out.columns
    assert out["is_weekend"].isin([0, 1]).all()
    assert out["dayofweek"].between(0, 6).all()


def test_snap_flag_uses_state_specific_column(toy_long: pd.DataFrame) -> None:
    df = toy_long.copy()
    df["snap_CA"] = 1
    df["snap_TX"] = 0
    df["snap_WI"] = 0
    out = add_snap_flag(df)
    ca_mask = out["state_id"].eq("CA")
    tx_mask = out["state_id"].eq("TX")
    assert (out.loc[ca_mask, "snap"] == 1).all()
    assert (out.loc[tx_mask, "snap"] == 0).all()


def test_event_flag_default_zero(toy_long: pd.DataFrame) -> None:
    df = toy_long.copy()
    df["event_name_1"] = "none"
    out = add_event_flag(df)
    assert (out["is_event"] == 0).all()


def test_price_features_norm_around_one(toy_long: pd.DataFrame) -> None:
    out = add_price_features(toy_long.copy())
    assert "price_norm" in out.columns
    assert "price_change_pct" in out.columns
    means = out.groupby("unique_id")["price_norm"].mean()
    for v in means:
        assert abs(v - 1.0) < 1e-5


def test_build_feature_frame_idempotent_columns(toy_long: pd.DataFrame) -> None:
    df = toy_long.copy()
    df["snap_CA"] = 0
    df["snap_TX"] = 0
    df["snap_WI"] = 0
    df["event_name_1"] = "none"
    out = build_feature_frame(df)
    expected = {"dayofweek", "snap", "is_event", "price_norm"}
    assert expected.issubset(out.columns)
