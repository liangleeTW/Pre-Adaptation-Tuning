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


def build_openloop_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Extract openloop reaching variance at post1 for R_obs baseline.

    Openloop reaching = reaching without visual feedback (eyes open, no prism).
    This measures pure motor execution variability, ideal for R_obs.
    """
    # Filter for openloop modality at post1
    openloop = raw_df[(raw_df["modality"] == "openloop") & (raw_df["session"] == "post1")].copy()

    if openloop.empty:
        raise ValueError("No openloop data found at post1 session.")

    # Get trial columns and compute variance
    cols = trial_columns(openloop)
    if not cols:
        raise ValueError("No trial columns found for openloop data.")

    values = openloop[cols].astype(float)
    openloop["openloop_var_post1"] = values.var(axis=1, ddof=1)
    openloop["openloop_sd_post1"] = values.std(axis=1, ddof=1)
    openloop["openloop_mean_post1"] = values.mean(axis=1)

    return openloop[["ID", "group", "openloop_var_post1", "openloop_sd_post1", "openloop_mean_post1"]]


def build_visual_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Extract visual localization variance at post1.

    Visual test = localization of visual targets (cursor position judgment).
    This measures visual encoding/processing noise.
    """
    # Filter for visual modality at post1
    visual = raw_df[(raw_df["modality"] == "visual") & (raw_df["session"] == "post1")].copy()

    if visual.empty:
        raise ValueError("No visual data found at post1 session.")

    # Get trial columns and compute variance
    cols = trial_columns(visual)
    if not cols:
        raise ValueError("No trial columns found for visual data.")

    values = visual[cols].astype(float)
    visual["visual_var_post1"] = values.var(axis=1, ddof=1)
    visual["visual_sd_post1"] = values.std(axis=1, ddof=1)
    visual["visual_mean_post1"] = values.mean(axis=1)

    return visual[["ID", "group", "visual_var_post1", "visual_sd_post1", "visual_mean_post1"]]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    groups = {
        "EC": RAW_DIR / "EC_prepost.csv",
        "EO+": RAW_DIR / "EO+_prepost.csv",
        "EO-": RAW_DIR / "EO-_prepost.csv",
    }

    frames = []
    raw_frames = []  # Keep raw data for openloop extraction
    for group, path in groups.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}.")
        raw_df = pd.read_csv(path)
        raw_df["group"] = group
        raw_frames.append(raw_df)
        frames.append(load_group(path, group))

    raw = pd.concat(frames, ignore_index=True)
    raw_all = pd.concat(raw_frames, ignore_index=True)  # Full raw data including openloop
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

    # Build proprioceptive delta
    delta = build_delta(summary)

    # Extract openloop variance at post1 (motor execution noise)
    openloop = build_openloop_summary(raw_all)

    # Extract visual variance at post1 (visual encoding noise)
    visual = build_visual_summary(raw_all)

    # Merge openloop and visual with delta
    delta = delta.merge(openloop, on=["ID", "group"], how="left")
    delta = delta.merge(visual, on=["ID", "group"], how="left")

    delta_path = OUT_DIR / "proprio_delta_pi.csv"
    delta.to_csv(delta_path, index=False)

    print(f"Wrote {summary_path}")
    print(f"Wrote {delta_path}")
    print(f"  Added openloop_var_post1 for {openloop.shape[0]} subjects")
    print(f"  Added visual_var_post1 for {visual.shape[0]} subjects")


if __name__ == "__main__":
    main()
