"""Plot estimated vs true parameters from recovery outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot recovery scatter plots.")
    parser.add_argument("--sweep-dir", type=str, default="data/sim_sweep")
    parser.add_argument("--infile", type=str, default="recovery.csv")
    parser.add_argument("--outdir", type=str, default="recovery_figures")
    return parser.parse_args()


def scatter(ax, x, y, title, xlab, ylab):
    ax.scatter(x, y, alpha=0.6, color="#1b6f8a", edgecolor="none")
    low = min(x.min(), y.min())
    high = max(x.max(), y.max())
    ax.plot([low, high], [low, high], color="#444444", lw=1, ls="--")
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)


def main() -> None:
    args = parse_args()
    sweep_dir = Path(args.sweep_dir)
    df = pd.read_csv(sweep_dir / args.infile)

    outdir = sweep_dir / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    m1 = df[df["model"] == "M1"]
    if "beta_all" in m1.columns and not m1.empty:
        fig, ax = plt.subplots(figsize=(5.5, 5.0))
        scatter(
            ax,
            m1["true_beta"],
            m1["beta_all"],
            "Recovery: M1 (beta)",
            "True beta",
            "Estimated beta",
        )
        fig.tight_layout()
        fig.savefig(outdir / "recovery_beta_scatter.png", dpi=160)
        plt.close(fig)

    m2 = df[df["model"] == "M2"]
    if "lam_all" in m2.columns and not m2.empty:
        fig, ax = plt.subplots(figsize=(5.5, 5.0))
        scatter(
            ax,
            m2["true_lam"],
            m2["lam_all"],
            "Recovery: M2 (lambda)",
            "True lambda",
            "Estimated lambda",
        )
        fig.tight_layout()
        fig.savefig(outdir / "recovery_lambda_scatter.png", dpi=160)
        plt.close(fig)

    print(f"Wrote plots to {outdir}")


if __name__ == "__main__":
    main()
