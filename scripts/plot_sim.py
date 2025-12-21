"""Plot basic simulation outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot prism adaptation simulations.")
    parser.add_argument("--outdir", type=str, default="data/sim")
    parser.add_argument("--max-traces", type=int, default=12)
    return parser.parse_args()


def plot_mean_trajectory(trials: pd.DataFrame, fig_path: Path) -> None:
    grouped = trials.groupby("trial")["error"]
    mean = grouped.mean()
    sem = grouped.sem()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(mean.index, mean.values, color="#1b6f8a", lw=2, label="Mean error")
    ax.fill_between(mean.index, mean - sem, mean + sem, color="#1b6f8a", alpha=0.2)
    ax.axhline(0, color="#444444", lw=1, ls="--")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Error (cm)")
    ax.set_title("Mean Error Trajectory")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


def plot_subject_traces(trials: pd.DataFrame, fig_path: Path, max_traces: int) -> None:
    subjects = trials["subject"].unique()[:max_traces]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for subject in subjects:
        sub = trials[trials["subject"] == subject]
        ax.plot(sub["trial"], sub["error"], alpha=0.6, lw=1)
    ax.axhline(0, color="#444444", lw=1, ls="--")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Error (cm)")
    ax.set_title(f"Individual Trajectories (n={len(subjects)})")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


def plot_precision_mapping(subjects: pd.DataFrame, fig_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(subjects["delta_pi"], subjects["r_measure"], alpha=0.7, color="#5b2c83")
    ax.set_xlabel("Delta precision (Δπ)")
    ax.set_ylabel("R measure")
    ax.set_title("Precision Change vs Measurement Noise")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    trials_path = outdir / "sim_trials.csv"
    subjects_path = outdir / "sim_subjects.csv"

    if not trials_path.exists() or not subjects_path.exists():
        raise FileNotFoundError("Missing sim_trials.csv or sim_subjects.csv in outdir.")

    trials = pd.read_csv(trials_path)
    subjects = pd.read_csv(subjects_path)

    fig_dir = outdir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_mean_trajectory(trials, fig_dir / "mean_error.png")
    plot_subject_traces(trials, fig_dir / "trajectories.png", args.max_traces)
    plot_precision_mapping(subjects, fig_dir / "precision_mapping.png")

    print(f"Wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
