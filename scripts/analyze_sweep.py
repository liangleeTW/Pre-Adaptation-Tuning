"""Summarize sweep outputs and generate quick-look plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze simulation sweep outputs.")
    parser.add_argument("--sweep-dir", type=str, default="data/sim_sweep")
    parser.add_argument("--early-trials", type=int, default=10)
    parser.add_argument("--late-trials", type=int, default=10)
    return parser.parse_args()


def compute_run_summary(run_dir: Path, early_trials: int, late_trials: int) -> dict[str, float]:
    trials = pd.read_csv(run_dir / "sim_trials.csv")
    subjects = pd.read_csv(run_dir / "sim_subjects.csv")

    mean_by_trial = trials.groupby("trial")["error"].mean().reset_index()

    early = mean_by_trial[mean_by_trial["trial"] <= early_trials]
    late = mean_by_trial[mean_by_trial["trial"] > (mean_by_trial["trial"].max() - late_trials)]

    early_mean = float(early["error"].mean())
    late_mean = float(late["error"].mean())
    adapt_gain = early_mean - late_mean

    slope, _ = np.polyfit(early["trial"], early["error"], 1)
    early_slope = float(slope)

    k_early = trials[trials["trial"] <= early_trials]["kalman_gain"].mean()

    corr_r_delta = np.corrcoef(subjects["r_post1"], subjects["delta_pi"])[0, 1]
    corr_delta_r = np.corrcoef(subjects["delta_pi"], subjects["r_measure"])[0, 1]

    return {
        "early_mean_error": early_mean,
        "late_mean_error": late_mean,
        "adapt_gain": adapt_gain,
        "early_slope": early_slope,
        "mean_k_early": float(k_early),
        "corr_rpost1_delta": float(corr_r_delta),
        "corr_delta_rmeasure": float(corr_delta_r),
    }


def plot_strength_effects(summary: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    m0 = summary[summary["model"] == "M0"]["early_slope"].mean()
    ax.axhline(m0, color="#444444", ls="--", lw=1, label="M0 baseline")

    for model, strength_col, label, color in [
        ("M1", "beta", "M1 beta", "#1b6f8a"),
        ("M2", "lam", "M2 lambda", "#7a2d2d"),
    ]:
        sub = summary[summary["model"] == model]
        grouped = sub.groupby(strength_col)["early_slope"]
        x = grouped.mean().index.values
        y = grouped.mean().values
        yerr = grouped.sem().values
        ax.errorbar(x, y, yerr=yerr, marker="o", lw=2, color=color, label=label)

    ax.set_xlabel("Modulation strength")
    ax.set_ylabel("Early slope (mean error vs trial)")
    ax.set_title("Early Learning Slope vs Modulation Strength")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "early_slope_vs_strength.png", dpi=160)
    plt.close(fig)


def plot_plateau_effects(summary: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    grouped = summary.groupby(["model", "plateau_frac"])["late_mean_error"]
    for model, color in [("M0", "#1b6f8a"), ("M1", "#5b2c83"), ("M2", "#7a2d2d")]:
        sub = grouped.get_group((model, 0.0)) if (model, 0.0) in grouped.groups else None
        model_df = summary[summary["model"] == model]
        means = model_df.groupby("plateau_frac")["late_mean_error"].mean()
        sems = model_df.groupby("plateau_frac")["late_mean_error"].sem()
        ax.errorbar(
            means.index.values,
            means.values,
            yerr=sems.values,
            marker="o",
            lw=2,
            color=color,
            label=model,
        )

    ax.set_xlabel("Plateau fraction (|m|)")
    ax.set_ylabel("Late mean error")
    ax.set_title("Plateau Confound: Late Error vs Plateau")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "late_error_vs_plateau.png", dpi=160)
    plt.close(fig)


def plot_rho_realization(summary: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(summary["rho"], summary["corr_rpost1_delta"], alpha=0.6, color="#1b6f8a")
    ax.axhline(0, color="#444444", lw=1, ls="--")
    ax.set_xlabel("Target rho")
    ax.set_ylabel("Realized corr(r_post1, delta_pi)")
    ax.set_title("Collinearity Check")
    fig.tight_layout()
    fig.savefig(outdir / "rho_realization.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    sweep_dir = Path(args.sweep_dir)
    index_path = sweep_dir / "index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing {index_path}.")

    index = pd.read_csv(index_path)

    summaries = []
    for _, row in index.iterrows():
        run_dir = Path(row["output_dir"])
        summary = compute_run_summary(run_dir, args.early_trials, args.late_trials)
        summary.update(
            {
                "run_id": row["run_id"],
                "run_name": row["run_name"],
                "model": row["model"],
                "beta": row["beta"],
                "lam": row["lam"],
                "delta_pi_sd": row["delta_pi_sd"],
                "rho": row["rho"],
                "plateau_frac": row["plateau_frac"],
                "seed": row["seed"],
            }
        )
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_path = sweep_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    fig_dir = sweep_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_strength_effects(summary_df, fig_dir)
    plot_plateau_effects(summary_df, fig_dir)
    plot_rho_realization(summary_df, fig_dir)

    print(f"Wrote {summary_path} and figures to {fig_dir}")


if __name__ == "__main__":
    main()
