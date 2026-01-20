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

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Updated color scheme
    colors = {
        "EC": "#0072B2",      # Blue
        "EO+": "#D55E00",     # Vermillion
        "EO-": "#009E73"      # Bluish Green
    }

    diff_colors = {
        "EC-EO+": "#CC79A7",  # Reddish Purple
        "EC-EO-": "#56B4E9",  # Sky Blue
        "EO+-EO-": "#E69F00"  # Orange
    }

    # Panel A: Posterior densities (no dashlines, no text annotations)
    ax = axes[0, 0]
    for group in groups:
        s = samples[group]
        kde_x = np.linspace(s.min() - 0.5, s.max() + 0.5, 200)
        kde = stats.gaussian_kde(s)
        ax.fill_between(kde_x, kde(kde_x), alpha=0.3, color=colors[group], label=group)
        ax.plot(kde_x, kde(kde_x), color=colors[group], lw=2)
        # Removed: ax.axvline(np.median(s), color=colors[group], ls="--", alpha=0.7)

    ax.axvline(0, color="black", ls="-", lw=1.5)
    ax.set_xlabel(r"$\beta_{cog}$", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(r"A. Posterior Distributions of $\beta_{cog}$", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")

    # Removed: interpretation annotations

    # Panel B: Forest plot
    ax = axes[0, 1]
    y_pos = np.arange(len(groups))

    for i, group in enumerate(groups):
        s = samples[group]
        median = np.median(s)
        hdi_low, hdi_high = np.percentile(s, [2.5, 97.5])

        ax.errorbar(median, i, xerr=[[median - hdi_low], [hdi_high - median]],
                   fmt="o", color=colors[group], capsize=5, capthick=2, markersize=10)
        ax.text(hdi_high + 0.1, i, f"{median:.2f} [{hdi_low:.2f}, {hdi_high:.2f}]",
               va="center", fontsize=10)

    ax.axvline(0, color="black", ls="-", lw=1.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(groups)
    ax.set_xlabel(r"$\beta_{cog}$", fontsize=12)
    ax.set_title(r"B. Forest Plot (Median + 95% HDI)", fontsize=14, fontweight="bold")

    # Panel C: CDF plot (like threshold_curves_flipped.png)
    ax = axes[1, 0]

    # Create threshold range
    all_samples = np.concatenate([samples[g] for g in groups])
    theta_range = np.linspace(all_samples.min() - 0.5, all_samples.max() + 0.5, 500)

    # Plot CDF for each group
    for group in groups:
        s = samples[group]
        # Compute CDF: P(beta_cog < threshold)
        cdf = np.array([np.mean(s < theta) for theta in theta_range])
        ax.plot(theta_range, cdf, color=colors[group], lw=2.5, label=group)

    # Add reference lines
    ax.axhline(0.95, color='gray', ls='--', alpha=0.5, lw=1.5)
    ax.axhline(0.50, color='gray', ls=':', alpha=0.5, lw=1.5)

    # Set custom tick positions for bottom axis (theta values)
    theta_ticks = np.array([-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_xticks(theta_ticks)
    ax.set_xlim(theta_range.min(), theta_range.max())

    # Enable grid after setting ticks
    ax.grid(True, alpha=0.3)

    # Styling
    ax.set_xlabel(r"Threshold $\theta_{\beta_{cog}}$", fontsize=12)
    ax.set_ylabel(r"$P(\beta_{cog} < \theta_{\beta_{cog}} \mid data)$", fontsize=12)
    ax.set_title(r"C. Posterior CDF of $\beta_{cog}$ Across Groups", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_ylim(-0.02, 1.05)

    # Add top x-axis with exp(theta)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    # Use the same tick positions as bottom axis
    exp_ticks = np.exp(theta_ticks)
    ax2.set_xticks(theta_ticks)
    ax2.set_xticklabels([f'{exp_val:.2f}' for exp_val in exp_ticks])
    ax2.set_xlabel(r"$\exp(\theta_{\beta_{cog}})$", fontsize=12)

    # Panel D: Pairwise differences
    ax = axes[1, 1]

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

    ax.axvline(0, color="black", ls="-", lw=1.5)
    ax.set_xlabel(r"Difference in $\beta_{cog}$", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(r"D. Pairwise Differences", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")

    plt.tight_layout()
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
