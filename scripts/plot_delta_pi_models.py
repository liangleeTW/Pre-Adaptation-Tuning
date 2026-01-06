"""Scatter plots of Δπ (or Δlogπ) vs. R for M1 and M2 by group."""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(metric: str = "pi") -> pd.DataFrame:
    """Return subject-level Δ, baseline R, and group labels."""
    delta_df = pd.read_csv("data/derived/proprio_delta_pi.csv")
    delta_df = delta_df[delta_df["precision_post1"] > 0].copy()
    delta_df["r_post1"] = 1.0 / delta_df["precision_post1"]
    delta_col = "delta_pi" if metric == "pi" else "delta_log_pi"
    return delta_df[["ID", "group", delta_col, "r_post1"]].rename(columns={"ID": "subject", delta_col: "delta"})


def load_params(fit_path: str = "data/derived/real_fit_numpyro_optimized.csv") -> tuple[dict, dict]:
    """Return group-specific parameters for M1 (beta) and M2 (lambda)."""
    fit_results = pd.read_csv(fit_path)

    m1 = fit_results[fit_results["model"] == "M1"].iloc[0]
    m2 = fit_results[fit_results["model"] == "M2"].iloc[0]

    betas = {"EC": m1["beta_EC"], "EO+": m1["beta_EO+"], "EO-": m1["beta_EO-"]}
    lams = {"EC": m2["lam_EC"], "EO+": m2["lam_EO+"], "EO-": m2["lam_EO-"]}
    return betas, lams


def predict_r_m1(row: pd.Series, betas: dict) -> float:
    return row["r_post1"] + betas[row["group"]] * row["delta"]


def predict_r_m2(row: pd.Series, lams: dict) -> float:
    lam = float(lams[row["group"]])
    lam = min(1.0, max(-1.0, lam))
    scale = 1.0 - lam * np.tanh(row["delta"])
    return row["r_post1"] * scale


def gain_series(
    r: float,
    n_trials: int = 200,
    A: float = 1.0,
    Q: float = 1e-4,
    p0: float = 1.0,
) -> np.ndarray:
    """Return Kalman gains across trials for constant R."""
    gains = np.zeros(n_trials)
    p_prev = p0
    for t in range(n_trials):
        p_pred = A * p_prev * A + Q
        k_t = p_pred / (p_pred + r)
        p_t = (1.0 - k_t) * p_pred
        gains[t] = k_t
        p_prev = p_t
    return gains


def gain_stat(r: float, stat: str = "steady") -> float:
    """Summarize gain for a given R."""
    g = gain_series(r)
    stat = stat.lower()
    if stat == "steady":
        return float(g[-1])
    if stat == "first":
        return float(g[0])
    if stat == "mean10":
        return float(g[:10].mean())
    raise ValueError("stat must be steady|first|mean10")


def plot_delta_pi_vs_r(
    data: pd.DataFrame,
    betas: dict,
    lams: dict,
    metric_label: str,
    save_pdf: bool,
    y_var: str,
    k_stat: str,
) -> None:
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.35)

    colors = {"EC": "#1f77b4", "EO+": "#ff7f0e", "EO-": "#2ca02c"}
    markers = {"EC": "o", "EO+": "s", "EO-": "^"}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    x_min, x_max = data["delta"].min(), data["delta"].max()
    x_range = np.linspace(x_min, x_max, 150)

    def scatter_panel(ax, r_col: str, title: str, param_label: str):
        for group in ["EC", "EO+", "EO-"]:
            gdata = data[data["group"] == group]
            ax.scatter(
                gdata["delta"],
                gdata[r_col],
                c=colors[group],
                marker=markers[group],
                s=95,
                alpha=0.65,
                edgecolors="white",
                linewidth=1.4,
                label=group,
            )

            median_r0 = gdata["r_post1"].median()
            if r_col == "R_M1":
                beta = betas[group]
                y_line = median_r0 + beta * x_range
                slope = f"β={beta:.2f}"
            else:
                lam = lams[group]
                y_line = median_r0 * (1.0 - lam * np.tanh(x_range))
                slope = f"λ={lam:.2f}"

            ax.plot(x_range, y_line, color=colors[group], linewidth=3, alpha=0.8, label=f"{group} {slope}")

        ax.set_xlabel("Δπ (Post1 − Pre precision)", fontsize=13, fontweight="bold")
        ax.set_ylabel("R (measurement noise)", fontsize=13, fontweight="bold")
        ax.set_title(title, fontsize=15, fontweight="bold", pad=14)
        ax.axvline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
        ax.legend(title=param_label, fontsize=10, title_fontsize=11, frameon=True, fancybox=True)
        ax.grid(True, alpha=0.3)

    if y_var.lower() == "r":
        y_label = "R (measurement noise)"
    else:
        y_label = f"K (Kalman gain, {k_stat})"
    axes[0].set_ylabel(y_label, fontsize=13, fontweight="bold")
    axes[1].set_ylabel("")

    scatter_panel(axes[0], f"{y_var.upper()}_M1", f"M1: {y_var.upper()} vs Δ{metric_label}", "Group params")
    scatter_panel(axes[1], f"{y_var.upper()}_M2", f"M2: {y_var.upper()} vs Δ{metric_label}", "Group params")

    fig.suptitle(f"Δ{metric_label} vs. {y_var.upper()} by Group (M1 vs M2)", fontsize=17, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"delta_{metric_label}_vs_{y_var.upper()}_m1_m2.png"
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {png_path}")
    if save_pdf:
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved: {pdf_path}")


def main(
    metric: str = "pi",
    fit_path: str = "data/derived/real_fit_numpyro_optimized.csv",
    save_pdf: bool = True,
    y_var: str = "R",
    k_stat: str = "steady",
) -> None:
    y_var = y_var.upper()
    if y_var not in {"R", "K"}:
        raise ValueError("y_var must be R or K")
    k_stat = k_stat.lower()
    if k_stat not in {"steady", "first", "mean10"}:
        raise ValueError("k_stat must be steady|first|mean10")
    data = load_data(metric=metric)
    betas, lams = load_params(fit_path=fit_path)
    data["R_M1"] = data.apply(predict_r_m1, axis=1, betas=betas)
    data["R_M2"] = data.apply(predict_r_m2, axis=1, lams=lams)
    if y_var == "K":
        data["K_M1"] = data["R_M1"].apply(lambda r: gain_stat(r, stat=k_stat))
        data["K_M2"] = data["R_M2"].apply(lambda r: gain_stat(r, stat=k_stat))
    metric_label = "π" if metric == "pi" else "logπ"
    plot_delta_pi_vs_r(data, betas, lams, metric_label, save_pdf=save_pdf, y_var=y_var, k_stat=k_stat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Δ vs R for M1/M2.")
    parser.add_argument("--metric", choices=["pi", "logpi"], default="pi", help="Use Δπ or Δlogπ.")
    parser.add_argument("--fit-path", default="data/derived/real_fit_numpyro_optimized.csv", help="Path to fit results.")
    parser.add_argument("--no-pdf", action="store_true", help="Do not save PDF output.")
    parser.add_argument("--y", choices=["R", "K"], default="R", help="Plot R (noise) or K (Kalman gain).")
    parser.add_argument("--k-stat", choices=["steady", "first", "mean10"], default="steady", help="Gain summary when y=K.")
    args = parser.parse_args()
    main(metric=args.metric, fit_path=args.fit_path, save_pdf=not args.no_pdf, y_var=args.y, k_stat=args.k_stat)
