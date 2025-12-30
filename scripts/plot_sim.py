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
    parser.add_argument("--early-trials", type=int, default=10)
    parser.add_argument("--late-trials", type=int, default=10)
    return parser.parse_args()


def shade_windows(ax, n_trials: int, early_trials: int, late_trials: int) -> None:
    if early_trials > 0:
        ax.axvspan(1, early_trials, color="#cfcfcf", alpha=0.2)
    if late_trials > 0:
        ax.axvspan(n_trials - late_trials + 1, n_trials, color="#cfcfcf", alpha=0.2)


def plot_mean_trajectory(
    trials: pd.DataFrame,
    fig_path: Path,
    early_trials: int,
    late_trials: int,
) -> None:
    grouped_obs = trials.groupby("trial")["error"]
    mean_obs = grouped_obs.mean()
    sem_obs = grouped_obs.sem()

    mean_true = None
    sem_true = None
    if "error_true" in trials.columns:
        grouped_true = trials.groupby("trial")["error_true"]
        mean_true = grouped_true.mean()
        sem_true = grouped_true.sem()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(mean_obs.index, mean_obs.values, color="#1b6f8a", lw=2, label="Observed mean")
    ax.fill_between(
        mean_obs.index,
        mean_obs - sem_obs,
        mean_obs + sem_obs,
        color="#1b6f8a",
        alpha=0.2,
    )
    if mean_true is not None:
        ax.plot(
            mean_true.index,
            mean_true.values,
            color="#444444",
            lw=1.8,
            ls="--",
            label="Bias-free mean",
        )
        ax.fill_between(
            mean_true.index,
            mean_true - sem_true,
            mean_true + sem_true,
            color="#444444",
            alpha=0.15,
        )
    ax.axhline(0, color="#444444", lw=1, ls="--")
    if "plateau_b" in trials.columns:
        plateau_b = float(trials["plateau_b"].iloc[0])
        ax.axhline(plateau_b, color="#7a2d2d", lw=1, ls=":")
    shade_windows(ax, int(trials["trial"].max()), early_trials, late_trials)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Error (cm)")
    ax.set_title("Mean Error Trajectory (Observed vs Bias-Free)")
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


def plot_plateau_offset(
    trials: pd.DataFrame,
    fig_path: Path,
    early_trials: int,
    late_trials: int,
) -> None:
    if "error_true" not in trials.columns:
        return
    trials = trials.copy()
    trials["offset"] = trials["error"] - trials["error_true"]
    grouped = trials.groupby("trial")["offset"]
    mean = grouped.mean()
    sem = grouped.sem()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(mean.index, mean.values, color="#7a2d2d", lw=2, label="Observed - bias-free")
    ax.fill_between(mean.index, mean - sem, mean + sem, color="#7a2d2d", alpha=0.2)
    if "plateau_b" in trials.columns:
        plateau_b = float(trials["plateau_b"].iloc[0])
        ax.axhline(plateau_b, color="#444444", lw=1, ls="--")
    shade_windows(ax, int(trials["trial"].max()), early_trials, late_trials)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Offset (cm)")
    ax.set_title("Plateau Offset (Observed - Bias-Free)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


def plot_group_trajectories(
    trials: pd.DataFrame,
    fig_path: Path,
    early_trials: int,
    late_trials: int,
) -> None:
    if "group" not in trials.columns:
        return
    groups = sorted(trials["group"].dropna().unique().tolist())
    if not groups:
        return
    colors = plt.get_cmap("tab10").colors

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for idx, group in enumerate(groups):
        sub = trials[trials["group"] == group]
        grouped = sub.groupby("trial")["error"]
        mean = grouped.mean()
        sem = grouped.sem()
        color = colors[idx % len(colors)]
        ax.plot(mean.index, mean.values, color=color, lw=2, label=str(group))
        ax.fill_between(mean.index, mean - sem, mean + sem, color=color, alpha=0.15)

    ax.axhline(0, color="#444444", lw=1, ls="--")
    shade_windows(ax, int(trials["trial"].max()), early_trials, late_trials)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Error (cm)")
    ax.set_title("Mean Error Trajectories by Group")
    ax.legend(frameon=False)
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

    plot_mean_trajectory(trials, fig_dir / "mean_error.png", args.early_trials, args.late_trials)
    plot_subject_traces(trials, fig_dir / "trajectories.png", args.max_traces)
    plot_precision_mapping(subjects, fig_dir / "precision_mapping.png")
    plot_plateau_offset(trials, fig_dir / "plateau_offset.png", args.early_trials, args.late_trials)
    plot_group_trajectories(trials, fig_dir / "group_trajectories.png", args.early_trials, args.late_trials)

    print(f"Wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
