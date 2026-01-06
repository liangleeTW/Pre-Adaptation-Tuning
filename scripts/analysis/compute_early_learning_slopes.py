"""
Compute early learning slopes from trial data and validate relationship with Δlogπ.
Phase 4.3 of analysis plan.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_context("paper", font_scale=1.2)
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


def compute_learning_slope(errors: np.ndarray, early_window: tuple[int, int] = (0, 15),
                           late_window: tuple[int, int] = (10, 25)) -> float:
    """
    Compute learning slope as difference between early and late error magnitudes.

    Args:
        errors: Array of trial-by-trial errors for one subject
        early_window: Trials to average for early phase (0-indexed)
        late_window: Trials to average for late phase

    Returns:
        Slope: mean(early) - mean(late) [positive = adaptation occurred]
    """
    if len(errors) < late_window[1]:
        return np.nan

    early_mean = np.nanmean(errors[early_window[0]:early_window[1]])
    late_mean = np.nanmean(errors[late_window[0]:late_window[1]])

    return early_mean - late_mean


def compute_all_slopes(trials: pd.DataFrame, delta: pd.DataFrame) -> pd.DataFrame:
    """Compute learning slopes for all subjects."""
    slopes_data = []

    for subject in trials['subject'].unique():
        subj_trials = trials[trials['subject'] == subject].sort_values('trial')
        errors = subj_trials['error'].values

        if len(errors) < 25:
            continue

        slope = compute_learning_slope(errors)

        # Get subject metadata
        subj_delta = delta[delta['ID'] == subject]
        if subj_delta.empty:
            continue

        subj_delta = subj_delta.iloc[0]

        slopes_data.append({
            'subject': subject,
            'group': subj_delta['group'],
            'slope': slope,
            'delta_log_pi': subj_delta['delta_log_pi'],
            'r_post1': subj_delta['r_post1'],
            'precision_post1': subj_delta['precision_post1'],
            'mean_error_early': np.nanmean(errors[0:15]),
            'mean_error_late': np.nanmean(errors[40:60]),  # Plateau estimate
        })

    return pd.DataFrame(slopes_data)


def plot_slope_vs_deltalogpi(slopes: pd.DataFrame, out_path: Path):
    """Plot empirical learning slope vs Δlogπ."""
    groups = ['EC', 'EO+', 'EO-']
    group_colors = {'EC': '#e74c3c', 'EO+': '#3498db', 'EO-': '#2ecc71'}

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot scatter per group
    for group in groups:
        group_data = slopes[slopes['group'] == group]
        ax.scatter(group_data['delta_log_pi'], group_data['slope'],
                  alpha=0.6, s=80, color=group_colors[group],
                  label=f'{group} (N={len(group_data)})',
                  edgecolors='black', linewidth=0.8)

        # Fit regression line
        mask = group_data['delta_log_pi'].notna() & group_data['slope'].notna()
        if mask.sum() > 2:
            x = group_data[mask]['delta_log_pi'].values
            y = group_data[mask]['slope'].values
            slope_fit, intercept, r_value, p_value, stderr = stats.linregress(x, y)

            x_range = np.linspace(x.min() - 0.1, x.max() + 0.1, 100)
            y_pred = slope_fit * x_range + intercept

            ax.plot(x_range, y_pred, color=group_colors[group],
                   linewidth=2.5, alpha=0.7, linestyle='--')

            # Add annotation
            ax.text(0.05, 0.95 - 0.08 * groups.index(group),
                   f'{group}: r={r_value:.2f}, p={p_value:.3f}',
                   transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor=group_colors[group],
                            alpha=0.2))

    # Overall correlation (pooled)
    mask = slopes['delta_log_pi'].notna() & slopes['slope'].notna()
    if mask.sum() > 2:
        x_all = slopes[mask]['delta_log_pi'].values
        y_all = slopes[mask]['slope'].values
        r_all, p_all = stats.pearsonr(x_all, y_all)

        ax.text(0.05, 0.70, f'Overall: r={r_all:.2f}, p={p_all:.3f}',
               transform=ax.transAxes, fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Δlog(π) [log precision change]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Learning Slope [mean(error 1-15) - mean(error 10-25)]',
                 fontsize=12, fontweight='bold')
    ax.set_title('Empirical Validation: Does Higher Δlog(π) → Slower Learning?',
                fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3)

    # Add interpretation box
    if r_all < 0:
        interpretation = "NEGATIVE correlation:\nHigher Δlogπ → SMALLER slope → SLOWER learning ✓"
        box_color = 'lightcoral'
    else:
        interpretation = "POSITIVE correlation:\nHigher Δlogπ → LARGER slope → FASTER learning ✗"
        box_color = 'lightgreen'

    ax.text(0.98, 0.05, interpretation,
           transform=ax.transAxes, fontsize=10,
           ha='right', va='bottom',
           bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.6))

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def plot_slope_distribution_by_group(slopes: pd.DataFrame, out_path: Path):
    """Box plot of learning slopes by group."""
    groups = ['EC', 'EO+', 'EO-']
    group_colors = {'EC': '#e74c3c', 'EO+': '#3498db', 'EO-': '#2ecc71'}

    fig, ax = plt.subplots(figsize=(8, 6))

    # Violin plot
    parts = ax.violinplot(
        [slopes[slopes['group'] == g]['slope'].dropna().values for g in groups],
        positions=range(len(groups)),
        widths=0.7,
        showmeans=True,
        showmedians=True
    )

    # Color violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(list(group_colors.values())[i])
        pc.set_alpha(0.6)

    # Overlay scatter
    for i, group in enumerate(groups):
        group_data = slopes[slopes['group'] == group]['slope'].dropna()
        x_jitter = np.random.normal(i, 0.04, size=len(group_data))
        ax.scatter(x_jitter, group_data, alpha=0.5, s=40,
                  color=group_colors[group], edgecolors='black', linewidth=0.5)

        # Add mean label
        mean_val = group_data.mean()
        ax.text(i, mean_val + 0.5, f'μ={mean_val:.1f}',
               ha='center', fontsize=10, fontweight='bold')

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=11, fontweight='bold')
    ax.set_ylabel('Learning Slope [mean(error 1-15) - mean(error 10-25)]',
                 fontsize=12, fontweight='bold')
    ax.set_title('Learning Slope Distribution by Group', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Statistical test (ANOVA)
    from scipy.stats import f_oneway
    group_data = [slopes[slopes['group'] == g]['slope'].dropna().values for g in groups]
    f_stat, p_val = f_oneway(*group_data)

    ax.text(0.5, 0.95, f'ANOVA: F={f_stat:.2f}, p={p_val:.3f}',
           transform=ax.transAxes, ha='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compute early learning slopes")
    parser.add_argument("--trials", type=Path,
                       default=Path("data/derived/adaptation_trials.csv"))
    parser.add_argument("--delta", type=Path,
                       default=Path("data/derived/proprio_delta_pi.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    parser.add_argument("--out-csv", type=Path,
                       default=Path("data/derived/learning_slopes.csv"))

    args = parser.parse_args()

    print("Loading data...")
    trials = pd.read_csv(args.trials)
    delta = pd.read_csv(args.delta)

    # Add delta_log_pi if not present
    if 'delta_log_pi' not in delta.columns:
        delta['delta_log_pi'] = np.log(delta['precision_post1']) - np.log(delta['precision_pre'])
    delta['r_post1'] = 1.0 / delta['precision_post1']

    print(f"Computing learning slopes for {trials['subject'].nunique()} subjects...")
    slopes = compute_all_slopes(trials, delta)

    print(f"Computed slopes for {len(slopes)} subjects")
    print("\nSummary by group:")
    print(slopes.groupby('group')['slope'].describe())

    # Save slopes
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    slopes.to_csv(args.out_csv, index=False)
    print(f"\nSaved slopes to: {args.out_csv}")

    # Plot 1: Slope vs Δlogπ
    print("\nGenerating plot: Slope vs Δlogπ...")
    plot_slope_vs_deltalogpi(slopes, args.out_dir / "empirical_slope_vs_deltalogpi.png")

    # Plot 2: Slope distribution by group
    print("Generating plot: Slope distribution...")
    plot_slope_distribution_by_group(slopes, args.out_dir / "slope_distribution_by_group.png")

    # Correlation test
    print("\n=== Correlation Analysis ===")
    mask = slopes['delta_log_pi'].notna() & slopes['slope'].notna()
    r, p = stats.pearsonr(slopes[mask]['delta_log_pi'], slopes[mask]['slope'])
    print(f"Overall correlation: r = {r:.3f}, p = {p:.4f}")

    if r < 0:
        print("✓ NEGATIVE correlation: Higher Δlogπ → SLOWER learning (consistent with positive β)")
    else:
        print("✗ POSITIVE correlation: Higher Δlogπ → FASTER learning (INCONSISTENT with model)")

    print("\n✅ Early learning slope analysis complete!")


if __name__ == "__main__":
    main()
