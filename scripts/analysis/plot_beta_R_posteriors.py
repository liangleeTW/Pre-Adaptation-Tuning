"""
Replot β_R posteriors for paper with updated styling.

This script generates the beta_R posteriors figure with:
- Updated color scheme
- β_R notation instead of beta_state
- Modified panel layouts and styling
"""

from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy import stats

# Tol colorblind-friendly palette
# https://sronpersonalpages.nl/~pault/
tol_bright = [
    '#4477AA',  # blue
    '#EE6677',  # red
    '#228833',  # green
    '#CCBB44',  # yellow
    '#66CCEE',  # cyan
    '#AA3377',  # purple
    '#BBBBBB',  # grey
]


def load_posterior(nc_path: Path) -> az.InferenceData:
    """Load posterior samples from NetCDF file."""
    if not nc_path.exists():
        raise FileNotFoundError(f"Posterior file not found: {nc_path}")
    return az.from_netcdf(nc_path)


def extract_beta_state(idata: az.InferenceData, group_labels: list[str]) -> dict:
    """Extract β_R samples for each group."""
    # Shape: (chains, draws, groups)
    beta_state = idata.posterior["beta_state"].values

    # Flatten chains and draws: (n_samples, groups)
    n_chains, n_draws, n_groups = beta_state.shape
    beta_flat = beta_state.reshape(-1, n_groups)

    return {
        "samples": beta_flat,
        "n_samples": beta_flat.shape[0],
        "group_labels": group_labels,
        "by_group": {g: beta_flat[:, i] for i, g in enumerate(group_labels)}
    }


def compute_ordering_probability(beta_data: dict) -> dict:
    """Compute probability of different group orderings."""
    samples = beta_data["by_group"]
    n = beta_data["n_samples"]

    # Expected ordering based on theory: EC > EO+ > EO-
    ec, eo_plus, eo_minus = samples["EC"], samples["EO+"], samples["EO-"]

    orderings = {
        "EC > EO+ > EO-": np.mean((ec > eo_plus) & (eo_plus > eo_minus)),
        "EC > EO- > EO+": np.mean((ec > eo_minus) & (eo_minus > eo_plus)),
        "EO+ > EC > EO-": np.mean((eo_plus > ec) & (ec > eo_minus)),
        "EO+ > EO- > EC": np.mean((eo_plus > eo_minus) & (eo_minus > ec)),
        "EO- > EC > EO+": np.mean((eo_minus > ec) & (ec > eo_plus)),
        "EO- > EO+ > EC": np.mean((eo_minus > eo_plus) & (eo_plus > ec)),
    }

    return orderings


def plot_posteriors(beta_data: dict, output_path: Path) -> None:
    """Plot posterior distributions and comparisons with paper styling."""
    samples = beta_data["by_group"]
    groups = beta_data["group_labels"]

    # Create figure with custom layout
    # Top row: A (left) and B (right)
    # Bottom row: C (full width CDF plot)
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3, height_ratios=[1, 1])
    ax_a = fig.add_subplot(gs[0, 0])  # Panel A: top left
    ax_b = fig.add_subplot(gs[0, 1])  # Panel B: top right
    ax_c = fig.add_subplot(gs[1, :])  # Panel C: bottom row, full width (CDF)

    # Tol colorblind-friendly color scheme
    colors = {
        "EC": tol_bright[1],   # red
        "EO+": tol_bright[0],  # blue
        "EO-": tol_bright[2]   # green
    }

    diff_colors = {
        "EC-EO+": tol_bright[5],  # purple
        "EC-EO-": tol_bright[4],  # cyan
        "EO+-EO-": tol_bright[3]  # yellow
    }

    # Panel A: Posterior densities with forest plot overlay
    ax = ax_a

    # Plot density distributions
    max_density = 0
    for group in groups:
        s = samples[group]
        kde_x = np.linspace(s.min() - 0.5, s.max() + 0.5, 200)
        kde = stats.gaussian_kde(s)
        ax.fill_between(kde_x, kde(kde_x), alpha=0.3, color=colors[group], label=group)
        ax.plot(kde_x, kde(kde_x), color=colors[group], lw=2)
        max_density = max(max_density, kde(kde_x).max())

    ax.axvline(0, color="black", ls="-", lw=1, alpha=0.5)
    ax.set_xlabel(r"$\beta_R$", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(r"A. Posterior Distributions and Summary Statistics of $\beta_R$", fontsize=14, fontweight="bold")
    ax.set_xlim(-0.5, 3)

    # Plot forest plot overlay at the top with minimal spacing
    # Position from top: use 85%, 90%, 95% of max_density
    # y_positions = [max_density * 0.95, max_density * 0.90, max_density * 0.85]
    y_positions = [max_density * 1.3, max_density * 1.2, max_density * 1.1]

    for i, group in enumerate(groups):
        s = samples[group]
        median = np.median(s)
        hdi_low, hdi_high = np.percentile(s, [2.5, 97.5])
        y_pos = y_positions[i]

        # Plot error bars
        ax.errorbar(median, y_pos, xerr=[[median - hdi_low], [hdi_high - median]],
                    fmt="o", color=colors[group], capsize=5, capthick=2, markersize=6,
                    alpha=0.9, zorder=10)

        # Add text annotations
        ax.text(hdi_high + 0.08, y_pos, f"{median:.2f} [{hdi_low:.2f}, {hdi_high:.2f}]",
                va="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.3",
                facecolor="white", edgecolor="none", alpha=0.8))

    # Add legend for density curves
    ax.legend(loc="upper right", fontsize=10)

    # Panel B: Pairwise differences (no text in distribution)
    ax = ax_b

    comparisons = [("EC", "EO+"), ("EC", "EO-"), ("EO+", "EO-")]
    # Keys for color dictionary lookup
    color_keys = ["EC-EO+", "EC-EO-", "EO+-EO-"]
    # Display labels with aligned minus signs
    display_labels = [
        r"EC$\;$-$\;$EO+",
        r"EC$\;$-$\;$EO-",
        r"EO+$\,$-$\;$EO-"
    ]
    # display_labels = [
    #     r"EC$\;\;\;\;$-$\;$EO+",
    #     r"EC$\;\;\;\;$-$\;$EO-",
    #     r"EO+$\,$-$\;$EO-"
    # ]
    # display_labels = [
    #     r" EC$\;\;\;$-$\;$EO+",
    #     r" EC$\;\;\;$-$\;$EO-",
    #     r"EO+$\,$-$\;$EO-"
    # ]

    for i, (g1, g2) in enumerate(comparisons):
        diff = samples[g1] - samples[g2]
        kde_x = np.linspace(diff.min() - 0.3, diff.max() + 0.3, 200)
        kde = stats.gaussian_kde(diff)
        color_key = color_keys[i]  # Use for color lookup
        display_label = display_labels[i]  # Use for legend
        ax.fill_between(kde_x, kde(kde_x), alpha=0.3, color=diff_colors[color_key],
                       label=display_label)
        ax.plot(kde_x, kde(kde_x), color=diff_colors[color_key], lw=2)

        # Removed: probability annotation text

    ax.axvline(0, color="black", ls="-", lw=1, alpha=0.5)
    ax.set_xlabel(r"Difference in $\beta_R$", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(r"B. Posterior Distributions of Pairwise Differences", fontsize=14, fontweight="bold")

    # Expand x-axis slightly to prevent legend overlap
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0] - 0.2, xlim[1] + 0.2)
    ax.legend(loc="upper right")

    # # Panel D: Ordering probabilities (COMMENTED OUT)
    # ax = ax_d
    #
    # orderings = compute_ordering_probability(beta_data)
    #
    # # Only keep the top 3 orderings
    # sorted_orderings = sorted(orderings.items(), key=lambda x: -x[1])[:3]
    #
    # labels = [o[0] for o in sorted_orderings]
    # probs = [o[1] for o in sorted_orderings]
    #
    # # Vertical bars (bar instead of barh), all same color (no red)
    # bars = ax.bar(range(len(labels)), probs, color=tol_bright[0], alpha=0.7)
    #
    # ax.set_xticks(range(len(labels)))
    # ax.set_xticklabels(labels, rotation=45, ha="right")
    # ax.set_ylabel("Posterior Probability", fontsize=12)
    # ax.set_title("C. Probability of Group Orderings", fontsize=14, fontweight="bold")
    # ax.set_ylim(0, 1)
    #
    # # Add percentage labels
    # for i, prob in enumerate(probs):
    #     ax.text(i, prob + 0.02, f"{prob:.1%}", ha="center", fontsize=10)

    # Panel C: CDF plot (expanded to full width at bottom)
    ax = ax_c

    # Create threshold range
    all_samples = np.concatenate([samples[g] for g in groups])
    theta_range = np.linspace(all_samples.min() - 0.5, all_samples.max() + 0.5, 500)

    # Store median values for vertical lines
    medians = {}

    # Plot CDF for each group
    for group in groups:
        s = samples[group]
        # Compute CDF: P(beta_R < threshold)
        cdf = np.array([np.mean(s < theta) for theta in theta_range])
        ax.plot(theta_range, cdf, color=colors[group], lw=2.5, label=group)

        # Find median (where CDF = 0.5)
        medians[group] = np.median(s)

    # Add reference lines
    ax.axhline(0.95, color='gray', ls='--', alpha=0.5, lw=1.5)
    ax.axhline(0.50, color='gray', ls=':', alpha=0.5, lw=1.5)

    # Add vertical lines at median and circles at y=0.5 intercepts
    for group in groups:
        median_val = medians[group]
        ax.axvline(median_val, color=colors[group], ls='--', alpha=0.6, lw=1)
        ax.plot(median_val, 0.5, 'o', color=colors[group], markersize=8,
                markeredgecolor='white', markeredgewidth=1.5, zorder=5)

    # Combine regular ticks with median values
    regular_ticks = np.array([-0.5, 0.0,1.5, 2.0])
    median_values = np.array([medians[g] for g in groups])
    theta_ticks = np.sort(np.concatenate([regular_ticks, median_values]))
    ax.set_xticks(theta_ticks)
    ax.set_xlim(theta_ticks.min(), theta_ticks.max())  # Match tick range exactly

    # Format bottom x-axis to 2 decimal places
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Styling
    ax.set_xlabel(r"Threshold $\theta_{\beta_R}$", fontsize=13)
    ax.set_ylabel(r"$P(\beta_R < \theta_{\beta_R} \mid data)$", fontsize=13)
    ax.set_title(r"C. Posterior CDF of $\beta_R$ Across Groups", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_ylim(-0.02, 1.05)

    # Add custom y-axis ticks to show 0.5 and 0.95 reference lines
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 0.95, 1.0])

    # Add top x-axis with exp(theta)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    # Use the same tick positions as bottom axis
    exp_ticks = np.exp(theta_ticks)
    ax2.set_xticks(theta_ticks)
    ax2.set_xticklabels([f'{exp_val:.2f}' for exp_val in exp_ticks])
    ax2.set_xlabel(r"$\exp(\theta_{\beta_R})$", fontsize=13)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot β_R posteriors for paper")
    parser.add_argument("--posterior-path", type=Path,
                       default=Path("data/derived/posteriors/m3_dd_posterior.nc"),
                       help="Path to posterior NetCDF file")
    parser.add_argument("--output-path", type=Path,
                       default=Path("data/derived/bayesian_analysis/m3_dd_beta_state/beta_R_posteriors.png"),
                       help="Output path for the figure")
    args = parser.parse_args()

    # Create output directory
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load posterior
    print(f"Loading posterior from: {args.posterior_path}")
    idata = load_posterior(args.posterior_path)

    # Group labels (alphabetically sorted as in the model)
    group_labels = ["EC", "EO+", "EO-"]

    # Extract β_R samples
    beta_data = extract_beta_state(idata, group_labels)
    print(f"Loaded {beta_data['n_samples']} posterior samples")

    # Generate visualization
    print("\nGenerating visualization...")
    plot_posteriors(beta_data, args.output_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
