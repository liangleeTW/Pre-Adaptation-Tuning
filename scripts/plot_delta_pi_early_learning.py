"""Scatter plot of Δπ vs early learning rate (slope) by group.

Definition used here:
    - Early window: first EARLY_TRIALS trials (default 10)
    - Early learning rate: slope of error vs. trial within that window
      (more negative = faster early learning).
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


EARLY_TRIALS = 10


def compute_early_slope(trials: pd.DataFrame, subject: str) -> float | None:
    """Return slope of error vs trial for the subject in the early window."""
    sub = trials[(trials["subject"] == subject) & (trials["trial"] <= EARLY_TRIALS)]
    if len(sub) < 2:
        return None
    slope, _ = np.polyfit(sub["trial"], sub["error"], 1)
    return float(slope)


def load_subject_metrics() -> pd.DataFrame:
    """Load Δπ, baseline R, and early learning slope per subject."""
    trials = pd.read_csv("data/derived/adaptation_trials.csv")
    delta_df = pd.read_csv("data/derived/proprio_delta_pi.csv")

    # Baseline R_post1
    delta_df = delta_df[delta_df["precision_post1"] > 0].copy()
    delta_df["r_post1"] = 1.0 / delta_df["precision_post1"]

    slopes: list[Tuple[str, float]] = []
    for subject in trials["subject"].unique():
        slope = compute_early_slope(trials, subject)
        if slope is not None:
            slopes.append((subject, slope))

    slope_df = pd.DataFrame(slopes, columns=["subject", "early_slope"])

    merged = delta_df.rename(columns={"ID": "subject"}).merge(
        slope_df, on="subject", how="inner"
    )
    return merged


def plot_delta_vs_slope(df: pd.DataFrame) -> None:
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.3)

    colors = {"EC": "#1f77b4", "EO+": "#ff7f0e", "EO-": "#2ca02c"}
    markers = {"EC": "o", "EO+": "s", "EO-": "^"}

    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    for group in ["EC", "EO+", "EO-"]:
        gdata = df[df["group"] == group]
        ax.scatter(
            gdata["delta_pi"],
            gdata["early_slope"],
            c=colors[group],
            marker=markers[group],
            s=95,
            alpha=0.7,
            edgecolors="white",
            linewidth=1.4,
            label=group,
        )
        if len(gdata) >= 3:
            sns.regplot(
                x="delta_pi",
                y="early_slope",
                data=gdata,
                scatter=False,
                ci=None,
                color=colors[group],
                ax=ax,
                line_kws={"lw": 2.3, "alpha": 0.8},
            )

    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.axvline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xlabel("Δπ (Post1 − Pre precision)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Early learning rate (slope of error vs trial)", fontsize=13, fontweight="bold")
    ax.set_title(f"Δπ vs Early Learning Rate (first {EARLY_TRIALS} trials)", fontsize=15, fontweight="bold")
    ax.legend(title="Group", fontsize=10, title_fontsize=11, frameon=True, fancybox=True)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "delta_pi_vs_early_learning.png"
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main() -> None:
    df = load_subject_metrics()
    print(f"Using EARLY_TRIALS = {EARLY_TRIALS}")
    print(f"N subjects with slopes: {len(df)}")
    plot_delta_vs_slope(df)


if __name__ == "__main__":
    main()
