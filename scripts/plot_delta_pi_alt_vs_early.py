"""Scatter plots of alternative Δ definitions vs early learning slope.

Metrics:
- Δlogπ = log_precision_post1 - log_precision_pre   (precision = 1/var)
- Δlogσ = log(sd_post1) - log(sd_pre)               (σ = sqrt(var))

Early learning slope = slope of error vs trial over first 10 adaptation trials.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


EARLY_TRIALS = 10


def compute_early_slopes(trials: pd.DataFrame) -> pd.DataFrame:
    """Return subject-level early slopes."""
    slopes: List[Tuple[str, float]] = []
    for subject in trials["subject"].unique():
        sub = trials[(trials["subject"] == subject) & (trials["trial"] <= EARLY_TRIALS)]
        if len(sub) < 2:
            continue
        slope, _ = np.polyfit(sub["trial"], sub["error"], 1)
        slopes.append((subject, float(slope)))
    return pd.DataFrame(slopes, columns=["subject", "early_slope"])


def compute_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    """Compute Δlogπ and Δlogσ per subject."""
    pre = summary[summary["session"] == "pre"].copy()
    post = summary[summary["session"] == "post1"].copy()

    pre = pre.rename(
        columns={
            "precision": "precision_pre",
            "log_precision": "log_precision_pre",
            "trial_sd": "sd_pre",
        }
    )
    post = post.rename(
        columns={
            "precision": "precision_post1",
            "log_precision": "log_precision_post1",
            "trial_sd": "sd_post1",
        }
    )

    merged = pre.merge(
        post[["ID", "group", "precision_post1", "log_precision_post1", "sd_post1"]],
        on=["ID", "group"],
        how="inner",
    )
    merged["delta_log_pi"] = merged["log_precision_post1"] - merged["log_precision_pre"]
    merged["log_sigma_pre"] = np.log(merged["sd_pre"])
    merged["log_sigma_post1"] = np.log(merged["sd_post1"])
    merged["delta_log_sigma"] = merged["log_sigma_post1"] - merged["log_sigma_pre"]
    return merged[
        [
            "ID",
            "group",
            "delta_log_pi",
            "delta_log_sigma",
            "precision_post1",
            "precision_pre",
        ]
    ].rename(columns={"ID": "subject"})


def plot(df: pd.DataFrame) -> None:
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.25)

    colors = {"EC": "#1f77b4", "EO+": "#ff7f0e", "EO-": "#2ca02c"}
    markers = {"EC": "o", "EO+": "s", "EO-": "^"}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    def scatter(ax, x_col, title):
        for g in ["EC", "EO+", "EO-"]:
            gdata = df[df["group"] == g]
            ax.scatter(
                gdata[x_col],
                gdata["early_slope"],
                c=colors[g],
                marker=markers[g],
                s=95,
                alpha=0.7,
                edgecolors="white",
                linewidth=1.4,
                label=g,
            )
            if len(gdata) >= 3:
                sns.regplot(
                    x=x_col,
                    y="early_slope",
                    data=gdata,
                    scatter=False,
                    ci=None,
                    color=colors[g],
                    ax=ax,
                    line_kws={"lw": 2.3, "alpha": 0.8},
                )
        ax.axhline(0, color="gray", ls=":", lw=1)
        ax.axvline(0, color="gray", ls=":", lw=1, alpha=0.6)
        ax.set_xlabel(x_col, fontsize=13, fontweight="bold")
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.legend(title="Group", fontsize=10, title_fontsize=11, frameon=True, fancybox=True)
        ax.grid(True, alpha=0.3)

    scatter(axes[0], "delta_log_pi", "Δlogπ vs Early Slope")
    scatter(axes[1], "delta_log_sigma", "Δlogσ vs Early Slope")

    axes[0].set_ylabel("Early learning slope (error vs trial, first 10)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("")

    fig.suptitle("Alternative Δ definitions vs Early Learning Rate", fontsize=17, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "delta_pi_alternatives_vs_early_slope.png"
    pdf = png.with_suffix(".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def main() -> None:
    trials = pd.read_csv("data/derived/adaptation_trials.csv")
    summary = pd.read_csv("data/derived/proprio_prepost_summary.csv")
    slopes = compute_early_slopes(trials)
    deltas = compute_deltas(summary)
    merged = deltas.merge(slopes, on="subject", how="inner")
    print(f"N subjects with slopes and deltas: {len(merged)}")
    plot(merged)


if __name__ == "__main__":
    main()
