"""
Replot β_cog posteriors for paper with updated styling.

This script generates the beta_cog posteriors figure with:
- Updated color scheme
- β_cog notation instead of beta_obs
- Modified panel layouts and styling
- CDF plot in Panel C
"""

from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
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


def extract_beta_obs(idata: az.InferenceData, group_labels: list[str]) -> dict:
    """Extract β_cog samples for each group."""
    beta_obs = idata.posterior["beta_obs"].values
    n_chains, n_draws, n_groups = beta_obs.shape
    beta_flat = beta_obs.reshape(-1, n_groups)

    return {
        "samples": beta_flat,
        "n_samples": beta_flat.shape[0],
        "group_labels": group_labels,
        "by_group": {g: beta_flat[:, i] for i, g in enumerate(group_labels)}
    }


def plot_posteriors(beta_data: dict, output_path: Path) -> None:
    """Plot posterior distributions and comparisons with paper styling."""
    samples = beta_data["by_group"]
    groups = beta_data["group_labels"]

    # Create figure with custom layout: 2x2 grid where bottom row spans full width
    fig = plt.figure(figsize=(12.5, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
    ax_a = fig.add_subplot(gs[0, 0])  # Panel A: top left
    ax_b = fig.add_subplot(gs[0, 1])  # Panel B: top right
    ax_c = fig.add_subplot(gs[1, :])  # Panel C: bottom row, full width

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
    ax.set_xlabel(r"$\beta_{cog}$", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(r"A. Posterior Distributions and Summary Statistics of $\beta_{cog}$", fontsize=14, fontweight="bold")

    # Expand x-axis slightly to prevent legend overlap
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0], xlim[1] + 2.5)

    # Plot forest plot overlay at the top with minimal spacing
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

    ax.legend(loc="upper right")

    # Removed: interpretation annotations

    # # Panel B: Forest plot (COMMENTED OUT)
    # ax = ax_b
    # y_pos = np.arange(len(groups))
    #
    # for i, group in enumerate(groups):
    #     s = samples[group]
    #     median = np.median(s)
    #     hdi_low, hdi_high = np.percentile(s, [2.5, 97.5])
    #
    #     ax.errorbar(median, i, xerr=[[median - hdi_low], [hdi_high - median]],
    #                fmt="o", color=colors[group], capsize=5, capthick=2, markersize=10)
    #     ax.text(hdi_high + 0.1, i, f"{median:.2f} [{hdi_low:.2f}, {hdi_high:.2f}]",
    #            va="center", fontsize=10)
    #
    # ax.axvline(0, color="black", ls="-", lw=1, alpha=0.5)
    # ax.set_yticks(y_pos)
    # ax.set_yticklabels(groups)
    # ax.set_xlabel(r"$\beta_{cog}$", fontsize=12)
    # ax.set_title(r"B. Forest Plot (Median + 95% HDI)", fontsize=14, fontweight="bold")

    # Panel B: Pairwise differences (moved from Panel D)
    ax = ax_b

    comparisons = [("EC", "EO+"), ("EC", "EO-"), ("EO+", "EO-")]
    comparison_labels = ["EC-EO+", "EC-EO-", "EO+-EO-"]

    for i, (g1, g2) in enumerate(comparisons):
        diff = samples[g1] - samples[g2]
        kde_x = np.linspace(diff.min() - 0.3, diff.max() + 0.3, 200)
        kde = stats.gaussian_kde(diff)
        label = comparison_labels[i]
        ax.fill_between(kde_x, kde(kde_x), alpha=0.3, color=diff_colors[label],
                       label=f"{g1} - {g2}")
        ax.plot(kde_x, kde(kde_x), color=diff_colors[label], lw=2)

    ax.axvline(0, color="black", ls="-", lw=1, alpha=0.5)
    ax.set_xlabel(r"Difference in $\beta_{cog}$", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(r"B. Pairwise Differences", fontsize=14, fontweight="bold")

    # Expand x-axis slightly to prevent legend overlap
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0] - 0.3, xlim[1] + 0.3)
    ax.legend(loc="upper right")

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
        # Compute CDF: P(beta_cog < threshold)
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
        # Add median value text at the intersection                                                                                          
        # ax.text(median_val, 0.55, f'{median_val:.2f}',                                                                                        
        #         ha='center', va='top', fontsize=8,                                                                                        
        #         color=colors[group], fontweight='bold',                                                                                      
        #         bbox=dict(boxstyle="round,pad=0.3", facecolor="white",                                                                       
        #                 edgecolor=colors[group], alpha=0.8))                                                                        
  

    # Set custom tick positions for bottom axis (theta values)
    theta_ticks = np.array([-3.0, -2.5, -2.0, -1.0, 0.5, 1.0, 1.5])
    # ax.set_xticks(theta_ticks)
    # ax.set_xlim(theta_ticks.min(), theta_ticks.max())  # Match tick range exactly                                                            
    
    # Combine regular ticks with median values                                                                                               
    regular_ticks = np.array([-3.0, -2.5, -2.0, -1.0, 0.5, 1.0, 1.5])
    median_values = np.array([medians[g] for g in groups])
    theta_ticks = np.sort(np.concatenate([regular_ticks, median_values]))
    ax.set_xticks(theta_ticks)
    # ax.set_xlim(theta_range.min(), theta_range.max()) 
    ax.set_xlim(theta_ticks.min(), theta_ticks.max())  # Match tick range exactly
    # Format bottom x-axis to 2 decimal places                                                                                               
    from matplotlib.ticker import FormatStrFormatter                                                                                         
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Enable grid after setting ticks
    # ax.grid(True, alpha=0.3)

    # Styling
    ax.set_xlabel(r"Threshold $\theta_{\beta_{cog}}$", fontsize=13)
    ax.set_ylabel(r"$P(\beta_{cog} < \theta_{\beta_{cog}} \mid data)$", fontsize=13)
    ax.set_title(r"C. Posterior CDF of $\beta_{cog}$ Across Groups", fontsize=13, fontweight="bold")
    # ax.legend(loc="lower right")
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
    ax2.set_xlabel(r"$\exp(\theta_{\beta_{cog}})$", fontsize=13)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot β_cog posteriors for paper")
    parser.add_argument("--posterior-path", type=Path,
                       default=Path("data/derived/posteriors/m3_dd_posterior.nc"),
                       help="Path to posterior NetCDF file")
    parser.add_argument("--output-path", type=Path,
                       default=Path("data/derived/bayesian_analysis/m3_dd_beta_obs/beta_cog_posteriors.png"),
                       help="Output path for the figure")
    args = parser.parse_args()

    # Create output directory
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load posterior
    print(f"Loading posterior from: {args.posterior_path}")
    idata = load_posterior(args.posterior_path)

    # Group labels
    group_labels = ["EC", "EO+", "EO-"]

    # Extract β_cog samples
    beta_data = extract_beta_obs(idata, group_labels)
    print(f"Loaded {beta_data['n_samples']} posterior samples")

    # Generate visualization
    print("\nGenerating visualization...")
    plot_posteriors(beta_data, args.output_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
