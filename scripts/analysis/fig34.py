"""
fig34.py — Reproduces Fig. 3 (β_R) and Fig. 4 (β_cog) of the paper from the
shipped M3 posterior (data/posteriors/m3_posterior.nc).

Outputs:
  figures/fig3.png  — β_R figure
  figures/fig4.png  — β_cog figure

Panel layout (same for both figures):
  A (top-left):  Noise modulation ratio vs Δlogπ
  B (top-right): Posterior distributions + summary statistics
  C (bottom):    Posterior CDF (full width)
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]

tol_bright = [
    '#4477AA',  # blue
    '#EE6677',  # red
    '#228833',  # green
    '#CCBB44',  # yellow
    '#66CCEE',  # cyan
    '#AA3377',  # purple
    '#BBBBBB',  # grey
]

COLORS = {"EC": tol_bright[1], "EO+": tol_bright[0], "EO-": tol_bright[2]}
GROUPS = ["EC", "EO+", "EO-"]


def load_data() -> tuple[dict, dict, pd.DataFrame]:
    idata = az.from_netcdf(
        ROOT / "data/posteriors/m3_posterior.nc"
    )
    bs = idata.posterior["beta_state"].values.reshape(-1, 3)
    bo = idata.posterior["beta_obs"].values.reshape(-1, 3)
    beta_R   = {g: bs[:, i] for i, g in enumerate(GROUPS)}
    beta_cog = {g: bo[:, i] for i, g in enumerate(GROUPS)}
    df_delta = pd.read_csv(ROOT / "data/proprio_delta_pi.csv")
    return beta_R, beta_cog, df_delta


# ── Panel A: noise modulation ─────────────────────────────────────────────────

def _draw_noise_panel(
    ax,
    samples: dict,
    df_delta: pd.DataFrame,
    ylabel: str,
) -> None:
    x_data_min = df_delta["delta_log_pi"].min()
    x_data_max = df_delta["delta_log_pi"].max()
    x_smooth = np.linspace(x_data_min - 0.1, x_data_max + 0.1, 300)

    rng = np.random.default_rng(42)
    y_max_data = 0.0

    for group in GROUPS:
        s = samples[group]
        color = COLORS[group]
        beta_med = np.median(s)

        idx = rng.choice(len(s), size=min(1000, len(s)), replace=False)
        y_mat = np.exp(np.outer(s[idx], x_smooth))
        y_lo = np.percentile(y_mat,  5, axis=0)
        y_hi = np.percentile(y_mat, 95, axis=0)
        y_max_data = max(y_max_data, y_hi.max())

        ax.fill_between(x_smooth, y_lo, y_hi, alpha=0.20, color=color, zorder=2)
        ax.plot(x_smooth, np.exp(beta_med * x_smooth),
                color=color, lw=2.5, label=group, zorder=3)

    # Data dots
    for group in GROUPS:
        df_grp = df_delta[df_delta["group"] == group]
        x_pts = df_grp["delta_log_pi"].values
        beta_med = np.median(samples[group])
        ax.scatter(x_pts, np.exp(beta_med * x_pts),
                   color=COLORS[group], s=35, alpha=0.35,
                   edgecolors='none', zorder=4)

    ax.axhline(1.0, color='black', ls='--', lw=1.0, alpha=0.6, zorder=1)
    ax.axvline(0.0, color='black', ls='--', lw=1.0, alpha=0.6, zorder=1)

    ax.set_xlim(x_data_min - 0.15, x_data_max + 0.15)
    ax.set_ylim(0, y_max_data * 1.05)
    ax.set_xlabel(
        r"Change in proprioceptive precision ($\Delta\log\pi$)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)


# ── Panel B: posterior distributions + forest plot ────────────────────────────

def _draw_posterior_panel(
    ax,
    samples: dict,
    xlabel: str,
    xlim: tuple | None,
) -> None:
    max_density = 0.0
    for group in GROUPS:
        s = samples[group]
        kde_x = np.linspace(s.min() - 0.5, s.max() + 0.5, 200)
        kde = stats.gaussian_kde(s)
        ax.fill_between(kde_x, kde(kde_x), alpha=0.3,
                        color=COLORS[group], label=group)
        ax.plot(kde_x, kde(kde_x), color=COLORS[group], lw=2)
        max_density = max(max_density, kde(kde_x).max())

    ax.axvline(0, color="black", ls="--", lw=1.0, alpha=0.6)
    ax.set_ylim(0, max_density * 1.38)  # pin bottom to 0, just above forest plot at 1.3
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Density", fontsize=11)

    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        cur = ax.get_xlim()
        ax.set_xlim(cur[0], cur[1] + 2.5)

    # Forest plot overlay above the density curves
    y_positions = [max_density * 1.3, max_density * 1.2, max_density * 1.1]
    for i, group in enumerate(GROUPS):
        s = samples[group]
        median  = np.median(s)
        hdi_low, hdi_high = np.percentile(s, [5, 95])
        ax.errorbar(
            median, y_positions[i],
            xerr=[[median - hdi_low], [hdi_high - median]],
            fmt="o", color=COLORS[group], capsize=5, capthick=2,
            markersize=6, alpha=0.9, zorder=10,
        )
        ax.text(
            hdi_high + 0.08, y_positions[i],
            f"{median:.2f} [{hdi_low:.2f}, {hdi_high:.2f}]",
            va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="none", alpha=0.8),
        )



# ── Panel C: posterior CDF ────────────────────────────────────────────────────

def _draw_cdf_panel(
    ax,
    samples: dict,
    xlabel: str,
    ylabel: str,
    ax2_xlabel: str,
    regular_ticks: np.ndarray,
) -> None:
    all_s = np.concatenate([samples[g] for g in GROUPS])
    theta_range = np.linspace(all_s.min() - 0.5, all_s.max() + 0.5, 500)
    medians = {}

    for group in GROUPS:
        s = samples[group]
        cdf = np.array([np.mean(s < theta) for theta in theta_range])
        ax.plot(theta_range, cdf, color=COLORS[group], lw=2.5, label=group)
        medians[group] = np.median(s)

    ax.axhline(0.90, color='gray', ls='--', alpha=0.5, lw=1.5)
    ax.axhline(0.50, color='gray', ls=':', alpha=0.5, lw=1.5)
    ax.axvline(0.0, color='black', ls='--', lw=1.0, alpha=0.6, zorder=1)

    for group in GROUPS:
        median_val = medians[group]
        ax.axvline(median_val, color=COLORS[group], ls='--', alpha=0.6, lw=1)
        ax.plot(median_val, 0.5, 'o', color=COLORS[group], markersize=8,
                markeredgecolor='white', markeredgewidth=1.5, zorder=5)

    median_values = np.array([medians[g] for g in GROUPS])
    theta_ticks = np.sort(np.concatenate([regular_ticks, median_values]))
    ax.set_xticks(theta_ticks)
    ax.set_xlim(theta_ticks.min(), theta_ticks.max())
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_ylim(-0.02, 1.05)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 0.90, 1.0])

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(theta_ticks)
    ax2.set_xticklabels([f'{np.exp(t):.2f}' for t in theta_ticks])
    ax2.set_xlabel(ax2_xlabel, fontsize=13)


# ── Composite figure ──────────────────────────────────────────────────────────

def make_figure(
    samples: dict,
    df_delta: pd.DataFrame,
    noise_ylabel: str,
    post_xlabel: str,
    post_xlim: tuple | None,
    cdf_xlabel: str,
    cdf_ylabel: str,
    cdf_ax2_xlabel: str,
    cdf_regular_ticks: np.ndarray,
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    _draw_noise_panel(ax_a, samples, df_delta, noise_ylabel)
    _draw_posterior_panel(ax_b, samples, post_xlabel, post_xlim)
    _draw_cdf_panel(ax_c, samples, cdf_xlabel, cdf_ylabel,
                    cdf_ax2_xlabel, cdf_regular_ticks)

    ax_a.set_title("A", fontsize=14, fontweight="bold", loc="left")
    ax_b.set_title("B", fontsize=14, fontweight="bold", loc="left")
    ax_c.set_title("C", fontsize=14, fontweight="bold", loc="left")

    # Single shared legend centered between the two rows
    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 0.52), ncol=3,
               fontsize=11, framealpha=1.0, edgecolor='#CCCCCC')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main() -> None:
    beta_R, beta_cog, df_delta = load_data()
    out = ROOT / "figures"

    # fig3: β_R
    make_figure(
        samples=beta_R,
        df_delta=df_delta,
        noise_ylabel=r"State noise relative to baseline ($R \,/\, R_0$)",
        post_xlabel=r"$\beta_R$",
        post_xlim=(-0.5, 3),
        cdf_xlabel=r"Threshold $\theta_{\beta_R}$",
        cdf_ylabel=r"$P(\beta_R < \theta_{\beta_R} \mid data)$",
        cdf_ax2_xlabel=r"$\exp(\theta_{\beta_R})$",
        cdf_regular_ticks=np.array([-0.5, 0.0, 1.5, 2.0]),
        output_path=out / "fig3.png",
    )

    # fig4: β_cog
    make_figure(
        samples=beta_cog,
        df_delta=df_delta,
        noise_ylabel=(r"State noise relative to baseline "
                      r"($V_{\mathrm{cog}} \,/\, V_{\mathrm{cog},0}$)"),
        post_xlabel=r"$\beta_{cog}$",
        post_xlim=None,
        cdf_xlabel=r"Threshold $\theta_{\beta_{cog}}$",
        cdf_ylabel=r"$P(\beta_{cog} < \theta_{\beta_{cog}} \mid data)$",
        cdf_ax2_xlabel=r"$\exp(\theta_{\beta_{cog}})$",
        cdf_regular_ticks=np.array([-3.0, -2.5, -2.0, -1.0, 0.5, 1.0, 1.5]),
        output_path=out / "fig4.png",
    )


if __name__ == "__main__":
    main()
