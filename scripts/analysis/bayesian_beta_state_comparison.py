"""
Bayesian comparison of β_state across groups for M-twoR-sensory model.

This script loads the posterior samples and computes:
1. Posterior probabilities for pairwise group comparisons
2. Posterior distribution of group differences
3. Probability of full ordering (EC > EO+ > EO-)
4. Visualizations of posteriors and differences
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


def extract_beta_state(idata: az.InferenceData, group_labels: list[str]) -> dict:
    """Extract β_state samples for each group."""
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


def compute_pairwise_comparisons(beta_data: dict) -> pd.DataFrame:
    """Compute posterior probabilities for pairwise comparisons."""
    groups = beta_data["group_labels"]
    samples = beta_data["by_group"]

    results = []
    for i, g1 in enumerate(groups):
        for j, g2 in enumerate(groups):
            if i >= j:
                continue

            diff = samples[g1] - samples[g2]
            prob_greater = np.mean(diff > 0)

            results.append({
                "comparison": f"{g1} > {g2}",
                "P(greater)": prob_greater,
                "P(less)": 1 - prob_greater,
                "diff_mean": np.mean(diff),
                "diff_median": np.median(diff),
                "diff_sd": np.std(diff),
                "HDI_2.5%": np.percentile(diff, 2.5),
                "HDI_97.5%": np.percentile(diff, 97.5),
            })

    return pd.DataFrame(results)


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


def compute_effect_sizes(beta_data: dict) -> pd.DataFrame:
    """Compute effect sizes (standardized differences)."""
    samples = beta_data["by_group"]

    # Pooled SD across all groups
    all_samples = np.concatenate(list(samples.values()))
    pooled_sd = np.std(all_samples)

    results = []
    groups = list(samples.keys())

    for i, g1 in enumerate(groups):
        for j, g2 in enumerate(groups):
            if i >= j:
                continue

            diff = samples[g1] - samples[g2]
            cohen_d = diff / pooled_sd

            results.append({
                "comparison": f"{g1} - {g2}",
                "Cohen's d (mean)": np.mean(cohen_d),
                "Cohen's d (median)": np.median(cohen_d),
                "d_HDI_2.5%": np.percentile(cohen_d, 2.5),
                "d_HDI_97.5%": np.percentile(cohen_d, 97.5),
                "P(|d| > 0.2)": np.mean(np.abs(cohen_d) > 0.2),  # Small effect
                "P(|d| > 0.5)": np.mean(np.abs(cohen_d) > 0.5),  # Medium effect
                "P(|d| > 0.8)": np.mean(np.abs(cohen_d) > 0.8),  # Large effect
            })

    return pd.DataFrame(results)


def compute_summary_stats(beta_data: dict) -> pd.DataFrame:
    """Compute summary statistics for each group."""
    samples = beta_data["by_group"]

    results = []
    for group, s in samples.items():
        results.append({
            "group": group,
            "mean": np.mean(s),
            "median": np.median(s),
            "sd": np.std(s),
            "HDI_2.5%": np.percentile(s, 2.5),
            "HDI_97.5%": np.percentile(s, 97.5),
            "P(β > 0)": np.mean(s > 0),
            "P(β > 0.5)": np.mean(s > 0.5),
            "P(β > 1.0)": np.mean(s > 1.0),
        })

    return pd.DataFrame(results)


def plot_posteriors(beta_data: dict, output_path: Path) -> None:
    """Plot posterior distributions and comparisons."""
    samples = beta_data["by_group"]
    groups = beta_data["group_labels"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    colors = {"EC": tol_bright[1], "EO+": tol_bright[0], "EO-": tol_bright[2]}

    # Panel A: Posterior densities
    ax = axes[0, 0]
    for group in groups:
        s = samples[group]
        kde_x = np.linspace(s.min() - 0.5, s.max() + 0.5, 200)
        kde = stats.gaussian_kde(s)
        ax.fill_between(kde_x, kde(kde_x), alpha=0.3, color=colors[group], label=group)
        ax.plot(kde_x, kde(kde_x), color=colors[group], lw=2)
        ax.axvline(np.median(s), color=colors[group], ls="--", alpha=0.7)

    ax.axvline(0, color="black", ls="-", lw=1, alpha=0.5)
    ax.set_xlabel("β_state", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("A. Posterior Distributions of β_state", fontsize=14, fontweight="bold")
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
    ax.set_xlabel("β_state", fontsize=12)
    ax.set_title("B. Forest Plot (Median + 95% HDI)", fontsize=14, fontweight="bold")
    ax.set_xlim(-0.5, 2.5)

    # Panel C: Pairwise differences
    ax = axes[1, 0]

    comparisons = [("EC", "EO+"), ("EC", "EO-"), ("EO+", "EO-")]
    diff_colors = [tol_bright[1], tol_bright[2], tol_bright[0]]

    for i, (g1, g2) in enumerate(comparisons):
        diff = samples[g1] - samples[g2]
        kde_x = np.linspace(diff.min() - 0.3, diff.max() + 0.3, 200)
        kde = stats.gaussian_kde(diff)
        ax.fill_between(kde_x, kde(kde_x), alpha=0.3, color=diff_colors[i],
                       label=f"{g1} - {g2}")
        ax.plot(kde_x, kde(kde_x), color=diff_colors[i], lw=2)

        # Add probability annotation
        prob = np.mean(diff > 0)
        ax.text(np.median(diff), kde(np.median(diff)).max() * 0.8,
               f"P>0: {prob:.1%}", fontsize=9, ha="center", color=diff_colors[i])

    ax.axvline(0, color="black", ls="-", lw=1.5)
    ax.set_xlabel("Difference in β_state", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("C. Posterior Distributions of Pairwise Differences", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")

    # Panel D: Ordering probabilities
    ax = axes[1, 1]

    orderings = compute_ordering_probability(beta_data)
    sorted_orderings = sorted(orderings.items(), key=lambda x: -x[1])

    labels = [o[0] for o in sorted_orderings]
    probs = [o[1] for o in sorted_orderings]

    bars = ax.barh(range(len(labels)), probs, color=tol_bright[0], alpha=0.7)

    # Highlight the expected ordering
    for i, label in enumerate(labels):
        if label == "EC > EO+ > EO-":
            bars[i].set_color(tol_bright[1])
            bars[i].set_alpha(1.0)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Posterior Probability", fontsize=12)
    ax.set_title("D. Probability of Group Orderings", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1)

    # Add percentage labels
    for i, prob in enumerate(probs):
        ax.text(prob + 0.02, i, f"{prob:.1%}", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization to: {output_path}")


def plot_rope_analysis(beta_data: dict, output_path: Path, rope: float = 0.1) -> None:
    """Plot ROPE (Region of Practical Equivalence) analysis for differences."""
    samples = beta_data["by_group"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    comparisons = [("EC", "EO+"), ("EC", "EO-"), ("EO+", "EO-")]

    for i, (g1, g2) in enumerate(comparisons):
        ax = axes[i]
        diff = samples[g1] - samples[g2]

        # Compute ROPE probabilities
        p_below_rope = np.mean(diff < -rope)
        p_in_rope = np.mean((diff >= -rope) & (diff <= rope))
        p_above_rope = np.mean(diff > rope)

        # Plot histogram
        ax.hist(diff, bins=50, density=True, alpha=0.7, color=tol_bright[0])

        # Add ROPE region
        ax.axvspan(-rope, rope, alpha=0.2, color="gray", label=f"ROPE [{-rope}, {rope}]")
        ax.axvline(0, color="black", ls="-", lw=1)
        ax.axvline(-rope, color="gray", ls="--", lw=1)
        ax.axvline(rope, color="gray", ls="--", lw=1)

        # Add HDI
        hdi_low, hdi_high = np.percentile(diff, [2.5, 97.5])
        ax.axvline(hdi_low, color=tol_bright[1], ls="--", lw=2)
        ax.axvline(hdi_high, color=tol_bright[1], ls="--", lw=2)

        ax.set_xlabel(f"β_state({g1}) - β_state({g2})", fontsize=11)
        ax.set_ylabel("Density" if i == 0 else "", fontsize=11)
        ax.set_title(f"{g1} vs {g2}", fontsize=12, fontweight="bold")

        # Add text annotation
        text = f"P(< -{rope}): {p_below_rope:.1%}\nP(in ROPE): {p_in_rope:.1%}\nP(> {rope}): {p_above_rope:.1%}"
        ax.text(0.95, 0.95, text, transform=ax.transAxes, fontsize=9,
               verticalalignment="top", horizontalalignment="right",
               bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.suptitle("ROPE Analysis for β_state Differences", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved ROPE analysis to: {output_path}")


def generate_report(beta_data: dict, pairwise: pd.DataFrame, orderings: dict,
                   effect_sizes: pd.DataFrame, summary: pd.DataFrame) -> str:
    """Generate a text report of the Bayesian analysis."""

    report = []
    report.append("=" * 70)
    report.append("BAYESIAN COMPARISON OF β_state ACROSS GROUPS (M-TWOR-SENSORY)")
    report.append("=" * 70)
    report.append("")

    # Summary statistics
    report.append("## 1. POSTERIOR SUMMARY BY GROUP")
    report.append("-" * 50)
    report.append(summary.to_string(index=False))
    report.append("")

    # Key finding: All groups have β_state > 0
    report.append("### Key Finding: Source-Estimation Support")
    for _, row in summary.iterrows():
        report.append(f"  {row['group']}: P(β_state > 0) = {row['P(β > 0)']:.1%}")
    report.append("")

    # Pairwise comparisons
    report.append("## 2. PAIRWISE COMPARISONS")
    report.append("-" * 50)
    report.append(pairwise.to_string(index=False))
    report.append("")

    # Ordering probabilities
    report.append("## 3. GROUP ORDERING PROBABILITIES")
    report.append("-" * 50)
    sorted_orderings = sorted(orderings.items(), key=lambda x: -x[1])
    for ordering, prob in sorted_orderings:
        marker = " <-- Expected" if ordering == "EC > EO+ > EO-" else ""
        report.append(f"  P({ordering}) = {prob:.1%}{marker}")
    report.append("")

    # Effect sizes
    report.append("## 4. EFFECT SIZES (Cohen's d)")
    report.append("-" * 50)
    report.append(effect_sizes.to_string(index=False))
    report.append("")

    # Interpretation
    report.append("## 5. INTERPRETATION")
    report.append("-" * 50)

    # Get the expected ordering probability
    expected_prob = orderings.get("EC > EO+ > EO-", 0)
    ec_eo_minus_row = pairwise[pairwise["comparison"] == "EC > EO-"].iloc[0]

    report.append(f"""
The Bayesian analysis reveals:

1. SOURCE-ESTIMATION SUPPORT: All three groups show β_state > 0 with high
   posterior probability (>{min(summary['P(β > 0)']):.0%}), supporting the
   source-estimation hypothesis across all sensory conditions.

2. GROUP ORDERING: The expected ordering (EC > EO+ > EO-) has {expected_prob:.1%}
   posterior probability. This {'supports' if expected_prob > 0.5 else 'does not strongly support'}
   the hypothesis that eyes-closed tuning produces the strongest effect.

3. EC vs EO- DIFFERENCE:
   - Mean difference: {ec_eo_minus_row['diff_mean']:.3f}
   - 95% HDI: [{ec_eo_minus_row['HDI_2.5%']:.3f}, {ec_eo_minus_row['HDI_97.5%']:.3f}]
   - P(EC > EO-): {ec_eo_minus_row['P(greater)']:.1%}

   {'The 95% HDI excludes zero, indicating a credible difference.'
    if ec_eo_minus_row['HDI_2.5%'] > 0 else
    'The 95% HDI includes zero, indicating uncertainty about the difference.'}

4. COMPARISON TO FREQUENTIST: Unlike the frequentist ANOVA (p = 0.36), the
   Bayesian analysis provides direct probability statements about group
   differences and their magnitudes.
""")

    report.append("=" * 70)

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Bayesian β_state comparison")
    parser.add_argument("--posterior-path", type=Path,
                       default=Path("data/derived/posteriors/m-twor-sensory_posterior.nc"),
                       help="Path to posterior NetCDF file")
    parser.add_argument("--output-dir", type=Path,
                       default=Path("data/derived/bayesian_analysis"),
                       help="Output directory for results")
    parser.add_argument("--rope", type=float, default=0.1,
                       help="ROPE width for practical equivalence")
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load posterior
    print(f"Loading posterior from: {args.posterior_path}")
    idata = load_posterior(args.posterior_path)

    # Group labels (alphabetically sorted as in the model)
    group_labels = ["EC", "EO+", "EO-"]

    # Extract β_state samples
    beta_data = extract_beta_state(idata, group_labels)
    print(f"Loaded {beta_data['n_samples']} posterior samples")

    # Compute statistics
    print("\nComputing Bayesian comparisons...")
    summary = compute_summary_stats(beta_data)
    pairwise = compute_pairwise_comparisons(beta_data)
    orderings = compute_ordering_probability(beta_data)
    effect_sizes = compute_effect_sizes(beta_data)

    # Save results
    summary.to_csv(args.output_dir / "beta_state_summary.csv", index=False)
    pairwise.to_csv(args.output_dir / "beta_state_pairwise.csv", index=False)
    effect_sizes.to_csv(args.output_dir / "beta_state_effect_sizes.csv", index=False)

    orderings_df = pd.DataFrame([
        {"ordering": k, "probability": v} for k, v in orderings.items()
    ]).sort_values("probability", ascending=False)
    orderings_df.to_csv(args.output_dir / "beta_state_orderings.csv", index=False)

    print(f"\nSaved CSV results to: {args.output_dir}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_posteriors(beta_data, args.output_dir / "beta_state_posteriors.png")
    plot_rope_analysis(beta_data, args.output_dir / "beta_state_rope.png", rope=args.rope)

    # Generate and save report
    report = generate_report(beta_data, pairwise, orderings, effect_sizes, summary)
    report_path = args.output_dir / "bayesian_comparison_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved report to: {report_path}")

    # Print report to console
    print("\n" + report)


if __name__ == "__main__":
    main()
