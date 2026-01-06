"""Plot group Δπ distributions and print sweep args derived from data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = Path("data/derived/proprio_delta_pi.csv")
OUT_DIR = Path("data/derived/figures")
ARGS_DIR = Path("data/derived")


def plot_hist(df: pd.DataFrame, value_col: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = {"EC": "#1b6f8a", "EO+": "#7a2d2d", "EO-": "#5b2c83"}
    for group in sorted(df["group"].unique()):
        sub = df[df["group"] == group]
        ax.hist(
            sub[value_col],
            bins=20,
            alpha=0.45,
            density=True,
            label=f"{group} (n={len(sub)})",
            color=colors.get(group, None),
        )
    ax.axvline(0, color="#444444", lw=1, ls="--")
    ax.set_xlabel(f"{value_col} (post1 - pre)")
    ax.set_ylabel("Density")
    ax.set_title(f"Group {value_col} distributions")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def format_list(values: list[float]) -> str:
    return ",".join(f"{v:.3f}" for v in values)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    plot_hist(df, "delta_log_pi", OUT_DIR / "delta_log_pi_groups.png")
    plot_hist(df, "delta_pi", OUT_DIR / "delta_pi_groups.png")

    summary = df.groupby("group")[["delta_pi", "delta_log_pi"]].agg(["mean", "std", "count"])
    group_order = sorted(df["group"].unique())
    weights = [summary.loc[g, ("delta_pi", "count")] for g in group_order]
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()

    means_pi = [summary.loc[g, ("delta_pi", "mean")] for g in group_order]
    sds_pi = [summary.loc[g, ("delta_pi", "std")] for g in group_order]
    means_log = [summary.loc[g, ("delta_log_pi", "mean")] for g in group_order]
    sds_log = [summary.loc[g, ("delta_log_pi", "std")] for g in group_order]

    args_pi = (
        "  --group-labels "
        + ",".join(group_order)
        + " --group-delta-pi-means "
        + format_list(means_pi)
        + " --group-delta-pi-sds "
        + format_list(sds_pi)
        + " --group-weights "
        + format_list(weights.tolist())
    )
    args_log = (
        "  --group-labels "
        + ",".join(group_order)
        + " --group-delta-pi-means "
        + format_list(means_log)
        + " --group-delta-pi-sds "
        + format_list(sds_log)
        + " --group-weights "
        + format_list(weights.tolist())
    )

    print("Derived sweep args (delta_pi scale):")
    print(args_pi)
    print("Derived sweep args (delta_log_pi scale):")
    print(args_log)

    ARGS_DIR.mkdir(parents=True, exist_ok=True)
    args_path = ARGS_DIR / "sweep_args.txt"
    args_path.write_text(
        "\n".join(
            [
                "Derived sweep args (delta_pi scale):",
                args_pi,
                "Derived sweep args (delta_log_pi scale):",
                args_log,
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
