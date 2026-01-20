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
from scipy import stats


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

    # Panel A: Posterior densities (no vertical dash lines)
    ax = axes[0, 0]
    for group in groups:
        s = samples[group]
        kde_x = np.linspace(s.min() - 0.5, s.max() + 0.5, 200)
        kde = stats.gaussian_kde(s)
        ax.fill_between(kde_x, kde(kde_x), alpha=0.3, color=colors[group], label=group)
        ax.plot(kde_x, kde(kde_x), color=colors[group], lw=2)
        # Removed: ax.axvline(np.median(s), color=colors[group], ls="--", alpha=0.7)

    ax.axvline(0, color="black", ls="-", lw=1, alpha=0.5)
    ax.set_xlabel(r"$\beta_R$", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(r"A. Posterior Distributions of $\beta_R$", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")

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

    ax.axvline(0, color="black", ls="-", lw=1, alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(groups)
    ax.set_xlabel(r"$\beta_R$", fontsize=12)
    ax.set_title(r"B. Forest Plot (Median + 95% HDI)", fontsize=14, fontweight="bold")
    ax.set_xlim(-0.5, 2.5)

    # Panel C: Pairwise differences (no text in distribution)
    ax = axes[1, 0]

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

        # Removed: probability annotation text

    ax.axvline(0, color="black", ls="-", lw=1.5)
    ax.set_xlabel(r"Difference in $\beta_R$", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(r"C. Posterior Distributions of Pairwise Differences", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")

    # Panel D: Ordering probabilities (vertical bars, only 3 groups, same color)
    ax = axes[1, 1]

    orderings = compute_ordering_probability(beta_data)

    # Only keep the top 3 orderings
    sorted_orderings = sorted(orderings.items(), key=lambda x: -x[1])[:3]

    labels = [o[0] for o in sorted_orderings]
    probs = [o[1] for o in sorted_orderings]

    # Vertical bars (bar instead of barh), all same color (no red)
    bars = ax.bar(range(len(labels)), probs, color="#0072B2", alpha=0.7)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Posterior Probability", fontsize=12)
    ax.set_title("D. Probability of Group Orderings", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1)

    # Add percentage labels
    for i, prob in enumerate(probs):
        ax.text(i, prob + 0.02, f"{prob:.1%}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot β_R posteriors for paper")
    parser.add_argument("--posterior-path", type=Path,
                       default=Path("data/derived/posteriors/m-twor-sensory_posterior.nc"),
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
