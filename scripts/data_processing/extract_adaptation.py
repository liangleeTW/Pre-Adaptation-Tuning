"""Extract adaptation-phase errors from raw group data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RAW_DIR = Path("raw")
OUT_DIR = Path("data/derived")


def load_group(path: Path, group: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"ID": "subject"})
    df["group"] = group
    return df


def trial_columns(start: int, end: int) -> list[str]:
    return [str(i) for i in range(start, end + 1)]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trial_cols = trial_columns(51, 150)

    groups = {
        "EC": RAW_DIR / "EC_main.csv",
        "EO+": RAW_DIR / "EO+_main.csv",
        "EO-": RAW_DIR / "EO-_main.csv",
    }

    frames = []
    for group, path in groups.items():
        if not path.exists():
            raise FileNotFoundError(path)
        df = load_group(path, group)
        missing = [c for c in trial_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing trial columns in {path.name}: {missing[:5]}")
        frames.append(df[["subject", "group"] + trial_cols])

    wide = pd.concat(frames, ignore_index=True)
    long_rows = []
    for _, row in wide.iterrows():
        for idx, col in enumerate(trial_cols):
            long_rows.append(
                {
                    "subject": row["subject"],
                    "group": row["group"],
                    "trial": idx + 1,
                    "error": float(row[col]),
                }
            )

    trials_df = pd.DataFrame(long_rows)
    trials_path = OUT_DIR / "adaptation_trials.csv"
    trials_df.to_csv(trials_path, index=False)

    early = trials_df[trials_df["trial"] <= 10]
    late = trials_df[trials_df["trial"] > (trials_df["trial"].max() - 10)]
    summary = (
        trials_df.groupby(["subject", "group"])["error"]
        .agg(mean_error="mean", sd_error="std")
        .reset_index()
    )
    summary["early_mean_error"] = (
        early.groupby(["subject", "group"])["error"].mean().values
    )
    summary["late_mean_error"] = (
        late.groupby(["subject", "group"])["error"].mean().values
    )
    summary["adapt_gain"] = summary["early_mean_error"] - summary["late_mean_error"]

    summary_path = OUT_DIR / "adaptation_subjects.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Wrote {trials_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
