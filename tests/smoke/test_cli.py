"""Smoke: every CLI subcommand renders ``--help`` without exploding."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from m5.cli import app


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def test_root_help(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "M5 forecasting toolkit" in result.stdout


@pytest.mark.parametrize("subcommand", ["download", "prep", "cv", "cv-recipe", "forecast"])
def test_subcommand_help(runner: CliRunner, subcommand: str) -> None:
    result = runner.invoke(app, [subcommand, "--help"])
    assert result.exit_code == 0
    assert subcommand in result.stdout.lower() or "Usage" in result.stdout


def test_cv_rejects_unknown_model(runner: CliRunner, tmp_path) -> None:
    """Validate input handling without touching real data."""
    fake = tmp_path / "long.parquet"
    import pandas as pd

    pd.DataFrame({"unique_id": [], "ds": [], "y": []}).to_parquet(fake)
    result = runner.invoke(app, ["cv", "bogus", "--long-path", str(fake)])
    assert result.exit_code != 0
