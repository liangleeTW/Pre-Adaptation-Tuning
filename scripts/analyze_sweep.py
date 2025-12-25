"""Summarize sweep outputs and generate quick-look plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sanitize_label(label: str) -> str:
    if not label:
        return "group"
    label = label.replace("+", "plus").replace("-", "minus")
    cleaned = "".join(c if c.isalnum() else "_" for c in label).strip("_")
    return cleaned.lower() or "group"


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

    early_mean_error = float(early["error"].mean())
    late_mean = float(late["error"].mean())
    adapt_gain = early_mean_error - late_mean

    slope, _ = np.polyfit(early["trial"], early["error"], 1)
    early_slope = float(slope)

    k_early = trials[trials["trial"] <= early_trials]["kalman_gain"].mean()

    corr_r_delta = np.corrcoef(subjects["r_post1"], subjects["delta_pi"])[0, 1]
    corr_delta_r = np.corrcoef(subjects["delta_pi"], subjects["r_measure"])[0, 1]

    early_subject = trials[trials["trial"] <= early_trials]
    early_by_subject = early_subject.groupby("subject")["error"].mean().reset_index()
    early_by_subject["abs_error"] = early_by_subject["error"].abs()
    merged = subjects.merge(early_by_subject, on="subject", how="inner")
    corr_delta_early_abs = np.corrcoef(merged["delta_pi"], merged["abs_error"])[0, 1]
    corr_delta_early = np.corrcoef(merged["delta_pi"], merged["error"])[0, 1]
    pos_mask = merged["delta_pi"] > 0
    neg_mask = merged["delta_pi"] < 0
    pos_mean = float(merged.loc[pos_mask, "error"].mean()) if pos_mask.any() else float("nan")
    neg_mean = float(merged.loc[neg_mask, "error"].mean()) if neg_mask.any() else float("nan")
    posneg_diff = pos_mean - neg_mean if pos_mask.any() and neg_mask.any() else float("nan")

    group_metrics: dict[str, float] = {}
    if "group" in trials.columns and "group" in subjects.columns:
        groups = sorted(trials["group"].dropna().unique().tolist())
        for group in groups:
            safe = sanitize_label(str(group))
            group_trials = trials[trials["group"] == group]
            group_mean_by_trial = group_trials.groupby("trial")["error"].mean().reset_index()
            group_early = group_mean_by_trial[group_mean_by_trial["trial"] <= early_trials]
            if len(group_early) >= 2:
                group_slope, _ = np.polyfit(group_early["trial"], group_early["error"], 1)
                group_metrics[f"group_{safe}_early_slope"] = float(group_slope)
            group_metrics[f"group_{safe}_early_mean_error"] = float(
                group_trials[group_trials["trial"] <= early_trials]["error"].mean()
            )
            group_subjects = subjects[subjects["group"] == group]
            group_metrics[f"group_{safe}_delta_pi_mean"] = float(group_subjects["delta_pi"].mean())
            group_metrics[f"group_{safe}_delta_pi_sd"] = float(group_subjects["delta_pi"].std(ddof=1))

    return {
        "early_mean_error": early_mean_error,
        "late_mean_error": late_mean,
        "adapt_gain": adapt_gain,
        "early_slope": early_slope,
        "mean_k_early": float(k_early),
        "corr_rpost1_delta": float(corr_r_delta),
        "corr_delta_rmeasure": float(corr_delta_r),
        "corr_delta_early_abs_error": float(corr_delta_early_abs),
        "corr_delta_early_error": float(corr_delta_early),
        "early_error_pos_delta": pos_mean,
        "early_error_neg_delta": neg_mean,
        "early_error_posneg_diff": posneg_diff,
        **group_metrics,
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
    ax.set_ylim(0.85, 0.95)
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
    low = min(summary["rho"].min(), summary["corr_rpost1_delta"].min())
    high = max(summary["rho"].max(), summary["corr_rpost1_delta"].max())
    ax.plot([low, high], [low, high], color="#444444", lw=1, ls="--")
    ax.set_xlabel("Target rho")
    ax.set_ylabel("Realized corr(r_post1, delta_pi)")
    ax.set_title("Collinearity Check")
    fig.tight_layout()
    fig.savefig(outdir / "rho_realization.png", dpi=160)
    plt.close(fig)


def plot_delta_pi_vs_early_error(summary: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for model, strength_col, label, color in [
        ("M1", "beta", "M1 beta", "#1b6f8a"),
        ("M2", "lam", "M2 lambda", "#7a2d2d"),
    ]:
        sub = summary[summary["model"] == model]
        grouped = sub.groupby(strength_col)["corr_delta_early_abs_error"]
        x = grouped.mean().index.values
        y = grouped.mean().values
        yerr = grouped.sem().values
        ax.errorbar(x, y, yerr=yerr, marker="o", lw=2, color=color, label=label)

    ax.axhline(0, color="#444444", lw=1, ls="--")
    ax.set_xlabel("Modulation strength")
    ax.set_ylabel("corr(Δπ, early |error|)")
    ax.set_title("Δπ vs Early |Error| (Correlation)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "delta_pi_vs_early_abs_error.png", dpi=160)
    plt.close(fig)


def plot_group_strength_effects(summary: pd.DataFrame, outdir: Path) -> None:
    group_cols = [c for c in summary.columns if c.startswith("group_") and c.endswith("_early_slope")]
    if not group_cols:
        return

    for col in group_cols:
        group_label = col[len("group_") : -len("_early_slope")]
        fig, ax = plt.subplots(figsize=(7.5, 4.5))

        for model, strength_col, label, color in [
            ("M1", "beta", "M1 beta", "#1b6f8a"),
            ("M2", "lam", "M2 lambda", "#7a2d2d"),
        ]:
            sub = summary[summary["model"] == model]
            grouped = sub.groupby(strength_col)[col]
            x = grouped.mean().index.values
            y = grouped.mean().values
            yerr = grouped.sem().values
            ax.errorbar(x, y, yerr=yerr, marker="o", lw=2, color=color, label=label)

        ax.set_xlabel("Modulation strength")
        ax.set_ylabel("Early slope (mean error vs trial)")
        ax.set_title(f"Early Slope vs Strength ({group_label})")
        ax.set_ylim(0.85, 0.95)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(outdir / f"early_slope_vs_strength_{group_label}.png", dpi=160)
        plt.close(fig)


def plot_group_comparison(summary: pd.DataFrame, outdir: Path) -> None:
    group_cols = [c for c in summary.columns if c.startswith("group_") and c.endswith("_early_slope")]
    if not group_cols:
        return

    def label_from_col(col: str) -> str:
        return col[len("group_") : -len("_early_slope")]

    def plot_model(model: str, strength_col: str, title: str, filename: str) -> None:
        sub = summary[summary["model"] == model]
        if sub.empty:
            return

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for col in group_cols:
            grouped = sub.groupby(strength_col)[col]
            x = grouped.mean().index.values
            y = grouped.mean().values
            yerr = grouped.sem().values
            ax.errorbar(x, y, yerr=yerr, marker="o", lw=2, label=label_from_col(col))

        ax.set_xlabel("Modulation strength")
        ax.set_ylabel("Early slope (mean error vs trial)")
        ax.set_title(title)
        ax.set_ylim(0.85, 0.95)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(outdir / filename, dpi=160)
        plt.close(fig)

    plot_model("M2", "lam", "Early Slope vs Lambda (Group Comparison)", "early_slope_vs_lambda_groups.png")
    plot_model("M1", "beta", "Early Slope vs Beta (Group Comparison)", "early_slope_vs_beta_groups.png")


def plot_group_combined_strengths(summary: pd.DataFrame, outdir: Path) -> None:
    group_cols = [c for c in summary.columns if c.startswith("group_") and c.endswith("_early_slope")]
    if not group_cols:
        return

    def label_from_col(col: str) -> str:
        return col[len("group_") : -len("_early_slope")]

    sub_m2 = summary[summary["model"] == "M2"]
    sub_m1 = summary[summary["model"] == "M1"]
    if sub_m2.empty and sub_m1.empty:
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = plt.get_cmap("tab10").colors

    for idx, col in enumerate(group_cols):
        group_label = label_from_col(col)
        color = colors[idx % len(colors)]

        if not sub_m2.empty:
            grouped = sub_m2.groupby("lam")[col]
            x = grouped.mean().index.values
            y = grouped.mean().values
            yerr = grouped.sem().values
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                lw=2,
                color=color,
                label=group_label,
            )

        if not sub_m1.empty:
            grouped = sub_m1.groupby("beta")[col]
            x = grouped.mean().index.values
            y = grouped.mean().values
            yerr = grouped.sem().values
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker="s",
                lw=2,
                ls="--",
                color=color,
                label="_nolegend_",
            )

    ax.set_xlabel("Modulation strength")
    ax.set_ylabel("Early slope (mean error vs trial)")
    ax.set_title("Early Slope vs Strength (Groups: M2 solid, M1 dashed)")
    ax.set_ylim(0.85, 0.95)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "early_slope_vs_strength_groups_combined.png", dpi=160)
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
    plot_delta_pi_vs_early_error(summary_df, fig_dir)
    plot_group_strength_effects(summary_df, fig_dir)
    plot_group_comparison(summary_df, fig_dir)
    plot_group_combined_strengths(summary_df, fig_dir)

    print(f"Wrote {summary_path} and figures to {fig_dir}")


if __name__ == "__main__":
    main()
