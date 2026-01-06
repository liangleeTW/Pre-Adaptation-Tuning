"""Scatter plot of Δ (precision change) vs Kalman gain K (M1/M2).

Designed for small K values: y-axis auto-scales tightly to the data range.
K is computed from fitted R via the scalar Kalman recursion (A=1, Q=1e-4).
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def gain_series(r: float, n_trials: int = 200, A: float = 1.0, Q: float = 1e-4, p0: float = 1.0) -> np.ndarray:
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
    g = gain_series(r)
    stat = stat.lower()
    if stat == "steady":
        return float(g[-1])
    if stat == "first":
        return float(g[0])
    if stat == "mean10":
        return float(g[:10].mean())
    raise ValueError("stat must be steady|first|mean10")


def load_data(metric: str) -> pd.DataFrame:
    df = pd.read_csv("data/derived/proprio_delta_pi.csv")
    df = df[df["precision_post1"] > 0].copy()
    df["r_post1"] = 1.0 / df["precision_post1"]
    delta_col = "delta_pi" if metric == "pi" else "delta_log_pi"
    return df[["ID", "group", delta_col, "r_post1"]].rename(columns={"ID": "subject", delta_col: "delta"})


def load_params(fit_path: str):
    fit = pd.read_csv(fit_path)
    m1 = fit[fit["model"] == "M1"].iloc[0]
    m2 = fit[fit["model"] == "M2"].iloc[0]
    betas = {"EC": m1["beta_EC"], "EO+": m1["beta_EO+"], "EO-": m1["beta_EO-"]}
    lams = {"EC": m2["lam_EC"], "EO+": m2["lam_EO+"], "EO-": m2["lam_EO-"]}
    return betas, lams


def compute_R(df: pd.DataFrame, betas, lams) -> pd.DataFrame:
    out = df.copy()
    out["R_M1"] = out.apply(lambda r: r["r_post1"] + betas[r["group"]] * r["delta"], axis=1)
    out["R_M2"] = out.apply(
        lambda r: r["r_post1"] * (1.0 - lams[r["group"]] * np.tanh(r["delta"])), axis=1
    )
    return out


def plot(df: pd.DataFrame, metric_label: str, k_stat: str, save_pdf: bool):
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.3)

    colors = {"EC": "#1f77b4", "EO+": "#ff7f0e", "EO-": "#2ca02c"}
    markers = {"EC": "o", "EO+": "s", "EO-": "^"}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    def scatter(ax, y_col, title):
        for g in ["EC", "EO+", "EO-"]:
            gdata = df[df["group"] == g]
            ax.scatter(
                gdata["delta"],
                gdata[y_col],
                c=colors[g],
                marker=markers[g],
                s=90,
                alpha=0.75,
                edgecolors="white",
                linewidth=1.3,
                label=g,
            )
        ax.axvline(0, color="gray", ls=":", lw=1)
        ax.set_xlabel(f"Δ{metric_label}", fontsize=13, fontweight="bold")
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Group", fontsize=10, title_fontsize=11, frameon=True, fancybox=True)

    scatter(axes[0], "K_M1", f"M1: K ({k_stat}) vs Δ{metric_label}")
    scatter(axes[1], "K_M2", f"M2: K ({k_stat}) vs Δ{metric_label}")

    # Tight y-scale
    y_min = min(df["K_M1"].min(), df["K_M2"].min())
    y_max = max(df["K_M1"].max(), df["K_M2"].max())
    pad = 0.1 * (y_max - y_min) if y_max > y_min else 0.001
    axes[0].set_ylim(y_min - pad, y_max + pad)
    axes[0].set_ylabel(f"Kalman gain K ({k_stat})", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("")

    fig.suptitle(f"Δ{metric_label} vs Kalman Gain by Group", fontsize=17, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"delta_{metric_label}_vs_K_{k_stat}_m1_m2.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    print(f"Saved: {png}")
    if save_pdf:
        fig.savefig(png.with_suffix(".pdf"), bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser(description="Plot Δ vs Kalman gain K for M1/M2.")
    ap.add_argument("--metric", choices=["pi", "logpi"], default="pi")
    ap.add_argument("--fit-path", default="data/derived/real_fit_numpyro_optimized.csv")
    ap.add_argument("--k-stat", choices=["steady", "first", "mean10"], default="steady")
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()

    df = load_data(args.metric)
    betas, lams = load_params(args.fit_path)
    df = compute_R(df, betas, lams)
    df["K_M1"] = df["R_M1"].apply(lambda r: gain_stat(r, args.k_stat))
    df["K_M2"] = df["R_M2"].apply(lambda r: gain_stat(r, args.k_stat))
    metric_label = "π" if args.metric == "pi" else "logπ"
    plot(df, metric_label, args.k_stat, save_pdf=not args.no_pdf)


if __name__ == "__main__":
    main()
