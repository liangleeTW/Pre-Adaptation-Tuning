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
    parser.add_argument("--policy-delta-pi-sd", type=float, default=None)
    parser.add_argument("--policy-rho", type=float, default=None)
    parser.add_argument("--policy-plateau-frac", type=float, default=None)
    parser.add_argument("--policy-seed", type=int, default=None)
    parser.add_argument("--policy-bins", type=int, default=12)
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
            group_metrics[f"group_{safe}_r_measure_mean"] = float(group_subjects["r_measure"].mean())
            group_metrics[f"group_{safe}_k_early"] = float(
                group_trials[group_trials["trial"] <= early_trials]["kalman_gain"].mean()
            )

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


def plot_plateau_residuals(summary: pd.DataFrame, outdir: Path) -> None:
    df = summary.copy()
    if "plateau_b" not in df.columns or "late_mean_error" not in df.columns:
        return
    df["late_residual"] = df["late_mean_error"] - df["plateau_b"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    grouped = df.groupby(["model", "plateau_b"])["late_residual"]
    for model, color in [("M0", "#1b6f8a"), ("M1", "#5b2c83"), ("M2", "#7a2d2d")]:
        model_df = df[df["model"] == model]
        means = model_df.groupby("plateau_b")["late_residual"].mean()
        sems = model_df.groupby("plateau_b")["late_residual"].sem()
        ax.errorbar(
            means.index.values,
            means.values,
            yerr=sems.values,
            marker="o",
            lw=2,
            color=color,
            label=model,
        )

    ax.axhline(0, color="#444444", lw=1, ls="--")
    ax.set_xlabel("Plateau bias b")
    ax.set_ylabel("Late error residual (late mean - b)")
    ax.set_title("Plateau Residual Check")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "late_error_residual_vs_plateau.png", dpi=160)
    plt.close(fig)


def plot_group_late_error(trials: pd.DataFrame, outdir: Path, late_trials: int) -> None:
    if "group" not in trials.columns:
        return
    max_trial = trials["trial"].max()
    late = trials[trials["trial"] > (max_trial - late_trials)]
    grouped = late.groupby(["group", "model"])["error"]
    means = grouped.mean().reset_index()
    sems = grouped.sem().reset_index().rename(columns={"error": "sem"})
    merged = means.merge(sems, on=["group", "model"], how="left")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = {"EC": "#1b6f8a", "EO+": "#7a2d2d", "EO-": "#5b2c83"}
    for model in sorted(merged["model"].unique()):
        sub = merged[merged["model"] == model]
        x = np.arange(len(sub["group"]))
        ax.errorbar(
            x + {"M0": -0.2, "M1": 0.0, "M2": 0.2}[model],
            sub["error"],
            yerr=sub["sem"],
            fmt="o",
            label=model,
            color="#444444",
        )
    ax.set_xticks(np.arange(len(sorted(merged["group"].unique()))))
    ax.set_xticklabels(sorted(merged["group"].unique()))
    ax.set_xlabel("Group")
    ax.set_ylabel("Late mean error")
    ax.set_title("Late Error by Group and Model")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "late_error_by_group.png", dpi=160)
    plt.close(fig)


def plot_rho_realization(summary: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(summary["rho"], summary["corr_rpost1_delta"], alpha=0.6, color="#1b6f8a")
    low = min(summary["rho"].min(), summary["corr_rpost1_delta"].min())
    high = max(summary["rho"].max(), summary["corr_rpost1_delta"].max())
    ax.plot([low, high], [low, high], color="#444444", lw=1, ls="--")
    ax.set_xlabel("Target rho")
    ax.set_ylabel("Realized corr(r_post1, delta_pi)")
    ax.set_title("Collinearity Check (Scatter)")
    fig.tight_layout()
    fig.savefig(outdir / "rho_realization_scatter.png", dpi=160)
    plt.close(fig)


def plot_rho_realization_box(summary: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    groups = []
    labels = []
    for rho in sorted(summary["rho"].unique()):
        groups.append(summary.loc[summary["rho"] == rho, "corr_rpost1_delta"].values)
        labels.append(f"{rho:.2f}")
    ax.boxplot(groups, labels=labels)
    ax.axhline(0, color="#444444", lw=1, ls="--")
    ax.set_xlabel("Target rho")
    ax.set_ylabel("Realized corr(r_post1, delta_pi)")
    ax.set_title("Collinearity Check (Boxplot)")
    fig.tight_layout()
    fig.savefig(outdir / "rho_realization_boxplot.png", dpi=160)
    plt.close(fig)


def plot_rho_sample_scatter(trials: pd.DataFrame, outdir: Path) -> None:
    if "r_post1" not in trials.columns or "delta_pi" not in trials.columns:
        return
    sample = trials.drop_duplicates("subject")
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.scatter(sample["r_post1"], sample["delta_pi"], alpha=0.6, color="#1b6f8a")
    ax.set_xlabel("r_post1")
    ax.set_ylabel("delta_pi")
    ax.set_title("Sample Run: r_post1 vs delta_pi")
    fig.tight_layout()
    fig.savefig(outdir / "rho_sample_scatter.png", dpi=160)
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
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "early_slope_vs_strength_groups_combined.png", dpi=160)
    plt.close(fig)


def parse_group_values(value: str) -> list[float]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [float(x) for x in value.split(",") if x.strip()]


def parse_group_labels(value: str) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def build_group_long(summary: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    group_cols = [c for c in summary.columns if c.startswith("group_") and c.endswith("_delta_pi_mean")]
    meta = index[["run_id", "group_labels", "group_lams"]].copy()
    merged = summary.merge(meta, on="run_id", how="left")
    rows = []
    for _, row in merged.iterrows():
        labels = parse_group_labels(row.get("group_labels", ""))
        lams = parse_group_values(row.get("group_lams", ""))
        lam_map = (
            {sanitize_label(label): lams[idx] for idx, label in enumerate(labels)}
            if labels and lams
            else {}
        )
        for col in group_cols:
            group_label = col[len("group_") : -len("_delta_pi_mean")]
            rows.append(
                {
                    "run_id": row["run_id"],
                    "model": row["model"],
                    "beta": row["beta"],
                    "lam": row["lam"],
                    "group": group_label,
                    "delta_pi_mean": row[col],
                    "r_measure_mean": row.get(f"group_{group_label}_r_measure_mean", np.nan),
                    "k_early": row.get(f"group_{group_label}_k_early", np.nan),
                    "lam_group": lam_map.get(group_label, np.nan),
                }
            )
    return pd.DataFrame(rows)


def plot_r_mapping_regimes(summary: pd.DataFrame, index: pd.DataFrame, outdir: Path) -> None:
    long_df = build_group_long(summary, index)
    if long_df.empty:
        return

    m2 = long_df[long_df["model"] == "M2"].copy()
    if m2.empty:
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    group_colors = {"ec": "#1b6f8a", "eoplus": "#7a2d2d", "eominus": "#5b2c83"}
    sign_markers = {1: "o", 0: "s", -1: "^"}
    sign_labels = {1: "lambda > 0", 0: "lambda = 0", -1: "lambda < 0"}

    m2 = m2[m2["lam_group"].notna()]
    if m2.empty:
        return

    m2["lam_sign"] = np.sign(m2["lam_group"]).astype(int)
    grouped = m2.groupby(["group", "lam_sign"])[["delta_pi_mean", "r_measure_mean"]].mean().reset_index()

    for _, row in grouped.iterrows():
        color = group_colors.get(row["group"], "#444444")
        marker = sign_markers.get(row["lam_sign"], "o")
        ax.scatter(
            row["delta_pi_mean"],
            row["r_measure_mean"],
            color=color,
            marker=marker,
            s=70,
            edgecolor="none",
        )
        ax.annotate(
            row["group"],
            (row["delta_pi_mean"], row["r_measure_mean"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=9,
            color=color,
        )

    ax.axvline(0, color="#444444", lw=1, ls="--")
    ax.set_xlabel("Mean Δπ (group)")
    ax.set_ylabel("Mean R measure (group)")
    ax.set_title("Measurement Noise vs Δπ (M2 Regimes)")
    group_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=group_colors.get(label, "#444444"), markersize=8, label=label)
        for label in ["ec", "eoplus", "eominus"]
        if label in grouped["group"].unique().tolist()
    ]
    sign_handles = [
        plt.Line2D([0], [0], marker=sign_markers[key], color="#444444", linestyle="None", markersize=8, label=sign_labels[key])
        for key in [-1, 0, 1]
        if key in grouped["lam_sign"].unique().tolist()
    ]
    ax.legend(handles=group_handles + sign_handles, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "r_mapping_regimes_m2.png", dpi=160)
    plt.close(fig)


def plot_policy_family(
    trials: pd.DataFrame,
    outdir: Path,
    early_trials: int,
    model: str,
    strength_col: str,
    filename: str,
    title: str,
    bins: int,
) -> None:
    if trials.empty:
        return
    sub = trials[trials["model"] == model]
    if sub.empty or strength_col not in sub.columns:
        return

    early = sub[sub["trial"] <= early_trials]
    k_by_subject = (
        early.groupby(["subject", strength_col])["kalman_gain"]
        .mean()
        .reset_index()
        .rename(columns={"kalman_gain": "k_early"})
    )
    delta_by_subject = (
        early.drop_duplicates(["subject", strength_col])[
            ["subject", strength_col, "delta_pi"]
        ]
        .rename(columns={"delta_pi": "delta_pi"})
    )
    merged = k_by_subject.merge(delta_by_subject, on=["subject", strength_col], how="inner")
    if merged.empty:
        return

    bins = max(4, int(bins))
    edges = np.linspace(merged["delta_pi"].min(), merged["delta_pi"].max(), bins + 1)
    merged["bin"] = pd.cut(merged["delta_pi"], edges, include_lowest=True)
    grouped = (
        merged.groupby([strength_col, "bin"])
        .agg(delta_pi=("delta_pi", "mean"), k_early=("k_early", "mean"))
        .reset_index()
    )

    strengths = sorted(grouped[strength_col].unique().tolist())
    if not strengths:
        return

    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(vmin=min(strengths), vmax=max(strengths))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for strength in strengths:
        rows = grouped[grouped[strength_col] == strength].sort_values("delta_pi")
        color = cmap(norm(strength))
        ax.plot(
            rows["delta_pi"],
            rows["k_early"],
            marker="o",
            lw=2,
            color=color,
            label=f"{strength_col}={strength:g}",
        )

    ax.axvline(0, color="#444444", lw=1, ls="--")
    ax.set_xlabel("Δπ (binned)")
    ax.set_ylabel("Mean early Kalman gain")
    ax.set_title(title)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / filename, dpi=160)
    plt.close(fig)


def filter_policy_index(index: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    policy = index.copy()
    if args.policy_delta_pi_sd is not None:
        policy = policy[policy["delta_pi_sd"] == args.policy_delta_pi_sd]
    if args.policy_rho is not None:
        policy = policy[policy["rho"] == args.policy_rho]
    if args.policy_plateau_frac is not None:
        policy = policy[policy["plateau_frac"] == args.policy_plateau_frac]
    if args.policy_seed is not None:
        policy = policy[policy["seed"] == args.policy_seed]
    return policy


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

    trials_all = []
    for _, row in index.iterrows():
        run_dir = Path(row["output_dir"])
        trials_df = pd.read_csv(run_dir / "sim_trials.csv")
        trials_df["run_id"] = row["run_id"]
        trials_all.append(trials_df)
    trials_df = pd.concat(trials_all, ignore_index=True)

    fig_dir = sweep_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_strength_effects(summary_df, fig_dir)
    plot_plateau_effects(summary_df, fig_dir)
    plot_plateau_residuals(summary_df, fig_dir)
    plot_group_late_error(trials_df, fig_dir, args.late_trials)
    plot_rho_realization(summary_df, fig_dir)
    plot_rho_realization_box(summary_df, fig_dir)
    plot_rho_sample_scatter(trials_df, fig_dir)
    plot_delta_pi_vs_early_error(summary_df, fig_dir)
    plot_group_strength_effects(summary_df, fig_dir)
    plot_group_comparison(summary_df, fig_dir)
    plot_group_combined_strengths(summary_df, fig_dir)
    policy_index = filter_policy_index(index, args)
    policy_run_ids = set(policy_index["run_id"].tolist())
    policy_summary = summary_df[summary_df["run_id"].isin(policy_run_ids)]
    policy_trials = trials_df[trials_df["run_id"].isin(policy_run_ids)]
    if not policy_summary.empty:
        plot_r_mapping_regimes(policy_summary, policy_index, fig_dir)
    if not policy_trials.empty:
        plot_policy_family(
            policy_trials,
            fig_dir,
            args.early_trials,
            model="M2",
            strength_col="lam",
            filename="policy_family_lambda.png",
            title="Policy Family Across Lambda (Mean Early Gain)",
            bins=args.policy_bins,
        )
        plot_policy_family(
            policy_trials,
            fig_dir,
            args.early_trials,
            model="M1",
            strength_col="beta",
            filename="policy_family_beta.png",
            title="Policy Family Across Beta (Mean Early Gain)",
            bins=args.policy_bins,
        )

    print(f"Wrote {summary_path} and figures to {fig_dir}")


if __name__ == "__main__":
    main()
