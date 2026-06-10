"""Generate a side-by-side comparison table of model WRMSSE scores.

Reads the merged headline CSV from a scoring run and emits a markdown table
with absolute scores and lift vs. a chosen baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_headline(report_dir: Path) -> pd.DataFrame:
    path = report_dir / "metrics" / "headline.csv"
    if not path.exists():
        alt = report_dir / "headline.csv"
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(f"No headline.csv found under {report_dir}")
    return pd.read_csv(path)


def format_table(df: pd.DataFrame, baseline_model: str) -> str:
    """Return a markdown table with WRMSSE and lift vs baseline."""
    if "model" not in df.columns or "wrmsse" not in df.columns:
        raise ValueError("headline.csv must contain 'model' and 'wrmsse' columns")

    base = df.loc[df["model"] == baseline_model, "wrmsse"]
    if base.empty:
        raise ValueError(
            f"Baseline model '{baseline_model}' not found in headline. Known: {df['model'].tolist()}"
        )
    base_wr = float(base.iloc[0])

    df = df.sort_values("wrmsse").reset_index(drop=True)
    df["lift"] = ((base_wr - df["wrmsse"]) / base_wr * 100).round(2)
    df["wrmsse"] = df["wrmsse"].round(6)

    lines = [
        "| Model | WRMSSE | Lift vs Baseline |",
        "|-------|--------|------------------|",
    ]
    for _, row in df.iterrows():
        sign = "+" if row["lift"] > 0 else ""
        lines.append(f"| {row['model']} | {row['wrmsse']:.6f} | {sign}{row['lift']:.2f}% |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare model WRMSSE scores")
    parser.add_argument("report_dir", type=Path, help="Directory containing metrics/headline.csv")
    parser.add_argument("--baseline", default="lgbm", help="Model to use as baseline for lift calculation")
    args = parser.parse_args()

    df = load_headline(args.report_dir)
    table = format_table(df, args.baseline)
    print(table)


if __name__ == "__main__":
    main()
