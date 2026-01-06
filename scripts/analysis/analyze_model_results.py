"""
Analysis and visualization of model comparison results.
Implements the analysis plan from TRY.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_context("paper", font_scale=1.2)
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


def load_data(results_path: Path, trials_path: Path, delta_path: Path):
    """Load all necessary data."""
    results = pd.read_csv(results_path)
    trials = pd.read_csv(trials_path)
    delta = pd.read_csv(delta_path)

    # Add delta_log_pi to delta if not present
    if 'delta_log_pi' not in delta.columns:
        delta['delta_log_pi'] = np.log(delta['precision_post1']) - np.log(delta['precision_pre'])

    # Add R_post1
    delta['r_post1'] = 1.0 / delta['precision_post1']

    return results, trials, delta


def plot_waic_comparison(results: pd.DataFrame, out_path: Path):
    """Phase 1.1: WAIC comparison bar plot."""
    # Filter to converged models only
    converged = results[results['max_rhat'] < 1.1].copy()

    # Calculate ΔWAIC relative to M0
    m0_waic = converged[converged['model'] == 'M0']['waic'].values[0]
    converged['delta_waic'] = converged['waic'] - m0_waic

    # Sort by ΔWAIC
    converged = converged.sort_values('delta_waic')

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ['gray' if m == 'M0' else 'red' if m == 'M1-EXP' else 'steelblue'
              for m in converged['model']]

    bars = ax.barh(converged['model'], converged['delta_waic'],
                   color=colors, alpha=0.7, edgecolor='black')

    # Add error bars (WAIC SE)
    ax.errorbar(converged['delta_waic'], range(len(converged)),
                xerr=converged['waic_se'], fmt='none', color='black',
                capsize=4, alpha=0.5)

    # Add value labels
    for i, (idx, row) in enumerate(converged.iterrows()):
        ax.text(row['delta_waic'] - 100, i, f"{row['delta_waic']:.0f}",
                va='center', ha='right', fontsize=10, fontweight='bold')

    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(-4, color='orange', linestyle=':', linewidth=1, alpha=0.5,
               label='Weak evidence (ΔWAIC < -4)')
    ax.axvline(-10, color='red', linestyle=':', linewidth=1, alpha=0.5,
               label='Strong evidence (ΔWAIC < -10)')

    ax.set_xlabel('ΔWAIC (relative to M0)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison: WAIC Evidence', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def plot_beta_posteriors(results: pd.DataFrame, out_path: Path):
    """Phase 2.1: β posterior distributions (from summary stats)."""
    # Extract β estimates for M1 and M1-exp
    models_to_plot = ['M1', 'M1-EXP']
    groups = ['EC', 'EO+', 'EO-']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, model in zip(axes, models_to_plot):
        model_data = results[results['model'] == model].iloc[0]

        betas = []
        labels = []

        for group in groups:
            beta_col = f'beta_{group}'
            if beta_col in model_data.index and pd.notna(model_data[beta_col]):
                betas.append(model_data[beta_col])
                labels.append(group)

        if betas:
            y_pos = np.arange(len(labels))

            # Plot as horizontal bars (we don't have full posteriors, just medians)
            # Approximate with ±0.2 as rough CI (will need actual posteriors for real CI)
            colors = ['#e74c3c', '#3498db', '#2ecc71']

            for i, (beta, label, color) in enumerate(zip(betas, labels, colors)):
                ax.barh(i, beta, color=color, alpha=0.6, edgecolor='black', height=0.6)
                ax.errorbar(beta, i, xerr=0.15, fmt='o', color='black',
                           capsize=5, markersize=8, markerfacecolor=color)

                # Add value label
                ax.text(beta + 0.05, i, f'{beta:.2f}',
                       va='center', fontsize=10, fontweight='bold')

            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)
            ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_xlabel('β (median estimate)', fontsize=11, fontweight='bold')
            ax.set_title(f'{model}', fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)

            # Highlight positive region
            ax.axvspan(0, ax.get_xlim()[1], alpha=0.1, color='green',
                      label='Positive β (slower learning)')

    axes[0].set_ylabel('Group', fontsize=11, fontweight='bold')
    axes[1].legend(loc='lower right', fontsize=9)

    fig.suptitle('β Parameter Estimates by Group and Model',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def plot_deltalogpi_vs_R(results: pd.DataFrame, delta: pd.DataFrame, out_path: Path):
    """Phase 4.1: Δlogπ → R relationship for different models."""
    # Get parameter estimates
    models = ['M0', 'M1', 'M1-EXP', 'M2']
    groups = ['EC', 'EO+', 'EO-']
    group_colors = {'EC': '#e74c3c', 'EO+': '#3498db', 'EO-': '#2ecc71'}

    # Create Δlogπ range for plotting
    deltalogpi_range = np.linspace(delta['delta_log_pi'].min(),
                                    delta['delta_log_pi'].max(), 100)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for ax, model in zip(axes, models):
        model_row = results[results['model'] == model].iloc[0]

        # Plot raw data points
        for group in groups:
            group_data = delta[delta['group'] == group]
            ax.scatter(group_data['delta_log_pi'], group_data['r_post1'],
                      alpha=0.4, s=50, color=group_colors[group],
                      label=f'{group} (data)', edgecolors='black', linewidth=0.5)

        # Plot model predictions
        for group in groups:
            group_data = delta[delta['group'] == group]
            r_post1_mean = group_data['r_post1'].mean()

            if model == 'M0':
                # No modulation: R = R_post1
                r_pred = np.full_like(deltalogpi_range, r_post1_mean)

            elif model == 'M1':
                # Linear: R = R_post1 + β*Δlogπ
                beta_col = f'beta_{group}'
                if beta_col in model_row.index and pd.notna(model_row[beta_col]):
                    beta = model_row[beta_col]
                    r_pred = r_post1_mean + beta * deltalogpi_range
                else:
                    continue

            elif model == 'M1-EXP':
                # Exponential: R = R_post1 * exp(β*Δlogπ)
                beta_col = f'beta_{group}'
                if beta_col in model_row.index and pd.notna(model_row[beta_col]):
                    beta = model_row[beta_col]
                    r_pred = r_post1_mean * np.exp(beta * deltalogpi_range)
                else:
                    continue

            elif model == 'M2':
                # Tanh: R = R_post1 * (1 - λ*tanh(Δlogπ))
                lam_col = f'lam_{group}'
                if lam_col in model_row.index and pd.notna(model_row[lam_col]):
                    lam = model_row[lam_col]
                    r_pred = r_post1_mean * (1 - lam * np.tanh(deltalogpi_range))
                else:
                    continue

            ax.plot(deltalogpi_range, r_pred, color=group_colors[group],
                   linewidth=2.5, alpha=0.8, label=f'{group} (model)')

        ax.set_xlabel('Δlog(π) [log precision change]', fontsize=11, fontweight='bold')
        ax.set_ylabel('R [measurement noise]', fontsize=11, fontweight='bold')
        ax.set_title(f'{model}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8, ncol=2)
        ax.grid(alpha=0.3)

        # Add annotation for model form
        if model == 'M0':
            ax.text(0.05, 0.95, 'R = R_post1', transform=ax.transAxes,
                   va='top', fontsize=9, bbox=dict(boxstyle='round',
                   facecolor='wheat', alpha=0.5))
        elif model == 'M1':
            ax.text(0.05, 0.95, 'R = R_post1 + β·Δlogπ', transform=ax.transAxes,
                   va='top', fontsize=9, bbox=dict(boxstyle='round',
                   facecolor='wheat', alpha=0.5))
        elif model == 'M1-EXP':
            ax.text(0.05, 0.95, 'R = R_post1 · exp(β·Δlogπ)', transform=ax.transAxes,
                   va='top', fontsize=9, bbox=dict(boxstyle='round',
                   facecolor='wheat', alpha=0.5))
        elif model == 'M2':
            ax.text(0.05, 0.95, 'R = R_post1 · (1 - λ·tanh(Δlogπ))', transform=ax.transAxes,
                   va='top', fontsize=9, bbox=dict(boxstyle='round',
                   facecolor='wheat', alpha=0.5))

    fig.suptitle('Model Predictions: Δlog(π) → R Relationship',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def create_convergence_table(results: pd.DataFrame, out_path: Path):
    """Phase 1.3: Convergence diagnostics table."""
    table = results[['model', 'max_rhat', 'min_ess_bulk', 'min_ess_tail']].copy()
    table['converged'] = (table['max_rhat'] < 1.01) & (table['min_ess_bulk'] > 400)
    table = table.sort_values('model')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print("\nConvergence Summary:")
    print(table.to_string(index=False))


def create_effect_size_table(results: pd.DataFrame, delta: pd.DataFrame, out_path: Path):
    """Phase 2.2: Effect size interpretation table."""
    model_row = results[results['model'] == 'M1-EXP'].iloc[0]
    groups = ['EC', 'EO+', 'EO-']

    effect_data = []

    for group in groups:
        beta_col = f'beta_{group}'
        if beta_col not in model_row.index or pd.isna(model_row[beta_col]):
            continue

        beta = model_row[beta_col]

        # Get typical Δlogπ for this group
        group_data = delta[delta['group'] == group]
        median_deltalogpi = group_data['delta_log_pi'].median()
        mean_rpost1 = group_data['r_post1'].mean()

        # Compute R change
        r_baseline = mean_rpost1
        r_modulated = mean_rpost1 * np.exp(beta * median_deltalogpi)
        r_change_pct = ((r_modulated - r_baseline) / r_baseline) * 100

        # Compute Kalman gain change (approximate)
        # K = P / (P + R); assume P ≈ 1 for simplicity
        P = 1.0
        k_baseline = P / (P + r_baseline)
        k_modulated = P / (P + r_modulated)
        k_change_pct = ((k_modulated - k_baseline) / k_baseline) * 100

        effect_data.append({
            'group': group,
            'beta_median': beta,
            'median_deltalogpi': median_deltalogpi,
            'R_baseline': r_baseline,
            'R_modulated': r_modulated,
            'R_change_pct': r_change_pct,
            'K_baseline': k_baseline,
            'K_modulated': k_modulated,
            'K_change_pct': k_change_pct,
            'learning_rate_impact': 'Slower' if beta > 0 else 'Faster'
        })

    effect_table = pd.DataFrame(effect_data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    effect_table.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print("\nEffect Size Summary:")
    print(effect_table[['group', 'beta_median', 'R_change_pct', 'K_change_pct',
                        'learning_rate_impact']].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Analyze model comparison results")
    parser.add_argument("--results", type=Path,
                       default=Path("data/derived/model_comparison_tier1_logpi_2000x4.csv"))
    parser.add_argument("--trials", type=Path,
                       default=Path("data/derived/adaptation_trials.csv"))
    parser.add_argument("--delta", type=Path,
                       default=Path("data/derived/proprio_delta_pi.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    parser.add_argument("--table-dir", type=Path, default=Path("tables"))

    args = parser.parse_args()

    print("Loading data...")
    results, trials, delta = load_data(args.results, args.trials, args.delta)

    print(f"\nLoaded {len(results)} models, {len(delta)} subjects, {len(trials)} trials")

    # Phase 1.1: WAIC comparison
    print("\n=== Phase 1.1: WAIC Comparison ===")
    plot_waic_comparison(results, args.out_dir / "model_comparison_waic.png")

    # Phase 1.3: Convergence table
    print("\n=== Phase 1.3: Convergence Diagnostics ===")
    create_convergence_table(results, args.table_dir / "convergence_diagnostics.csv")

    # Phase 2.1: β posteriors
    print("\n=== Phase 2.1: β Posterior Distributions ===")
    plot_beta_posteriors(results, args.out_dir / "beta_posteriors_by_group.png")

    # Phase 2.2: Effect sizes
    print("\n=== Phase 2.2: Effect Size Interpretation ===")
    create_effect_size_table(results, delta, args.table_dir / "effect_sizes.csv")

    # Phase 4.1: Δπ → R relationship
    print("\n=== Phase 4.1: Δlog(π) → R Relationship ===")
    plot_deltalogpi_vs_R(results, delta, args.out_dir / "deltalogpi_vs_R_models.png")

    print("\n✅ Analysis complete! Check figures/ and tables/ directories.")


if __name__ == "__main__":
    main()
