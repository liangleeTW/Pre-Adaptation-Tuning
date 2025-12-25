"""Extract proprioceptive pre/post1 summaries from raw group CSVs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RAW_DIR = Path("raw")
OUT_DIR = Path("data/derived")


def trial_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col.isdigit():
            cols.append(col)
    return cols


def load_group(path: Path, group: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["modality"] == "proprioceptive"]
    df = df[df["session"].isin(["pre", "post1"])].copy()
    df["group"] = group
    return df


def summarize_trials(df: pd.DataFrame) -> pd.DataFrame:
    cols = trial_columns(df)
    if not cols:
        raise ValueError("No trial columns found (expected numeric column names).")
    values = df[cols].astype(float)
    df = df.copy()
    df["trial_mean"] = values.mean(axis=1)
    df["trial_var"] = values.var(axis=1, ddof=1)
    df["trial_sd"] = values.std(axis=1, ddof=1)
    df["precision"] = 1.0 / df["trial_var"]
    df["log_precision"] = np.where(df["precision"] > 0, np.log(df["precision"]), np.nan)
    return df


def build_delta(df: pd.DataFrame) -> pd.DataFrame:
    pre = df[df["session"] == "pre"][["ID", "group", "precision", "log_precision"]].copy()
    post = df[df["session"] == "post1"][["ID", "group", "precision", "log_precision"]].copy()
    pre = pre.rename(
        columns={
            "precision": "precision_pre",
            "log_precision": "log_precision_pre",
        }
    )
    post = post.rename(
        columns={
            "precision": "precision_post1",
            "log_precision": "log_precision_post1",
        }
    )
    merged = pre.merge(post, on=["ID", "group"], how="inner")
    merged["delta_pi"] = merged["precision_post1"] - merged["precision_pre"]
    merged["delta_log_pi"] = merged["log_precision_post1"] - merged["log_precision_pre"]
    return merged


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    groups = {
        "EC": RAW_DIR / "EC_prepost.csv",
        "EO+": RAW_DIR / "EO+_prepost.csv",
        "EO-": RAW_DIR / "EO-_prepost.csv",
    }

    frames = []
    for group, path in groups.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}.")
        frames.append(load_group(path, group))

    raw = pd.concat(frames, ignore_index=True)
    summary = summarize_trials(raw)

    summary_cols = [
        "ID",
        "group",
        "session",
        "modality",
        "trial_mean",
        "trial_sd",
        "trial_var",
        "precision",
        "log_precision",
    ]
    summary_path = OUT_DIR / "proprio_prepost_summary.csv"
    summary[summary_cols].to_csv(summary_path, index=False)

    delta = build_delta(summary)
    delta_path = OUT_DIR / "proprio_delta_pi.csv"
    delta.to_csv(delta_path, index=False)

    print(f"Wrote {summary_path}")
    print(f"Wrote {delta_path}")


if __name__ == "__main__":
    main()
