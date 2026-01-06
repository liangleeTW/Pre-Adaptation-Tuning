"""
Posterior predictive checks: Compare model predictions to observed adaptation curves.
Phase 3.1 of analysis plan.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prism_sim.model import ModelParams

sns.set_context("paper", font_scale=1.1)
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


def kalman_predict(errors_init: np.ndarray, r: float, m: float, A: float, Q: float, b: float) -> np.ndarray:
    """
    Run Kalman filter forward to predict error sequence.

    Args:
        errors_init: Initial errors (first few trials) for warm-start (optional)
        r: Measurement noise
        m: Perturbation magnitude
        A: State transition
        Q: Process noise
        b: Plateau offset

    Returns:
        predictions: Predicted errors for all trials
    """
    n_trials = 100
    predictions = np.zeros(n_trials)

    # Initialize
    x = 0.0  # State estimate
    p = 1.0  # State variance

    for t in range(n_trials):
        # Predict
        x_pred = A * x
        p_pred = A * p * A + Q

        # Predicted observation
        y_pred = -x_pred + (m + b)
        predictions[t] = y_pred

        # Update (use actual observation if available, else use prediction)
        if t < len(errors_init):
            y_obs = errors_init[t]
        else:
            y_obs = y_pred  # Self-predict

        # Kalman update
        s = p_pred + r
        v = y_obs - y_pred
        k = -p_pred / s  # h = -1
        x = x_pred + k * v
        p = (1.0 + k) * p_pred

    return predictions


def compute_model_predictions(trials: pd.DataFrame, delta: pd.DataFrame,
                              results: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Compute predictions for all subjects under a specific model."""
    model_row = results[results['model'] == model_name].iloc[0]

    predictions_list = []

    for subject_id in trials['subject'].unique():
        # Get subject data
        subj_trials = trials[trials['subject'] == subject_id].sort_values('trial')
        subj_delta = delta[delta['ID'] == subject_id]

        if subj_delta.empty:
            continue

        subj_delta = subj_delta.iloc[0]
        group = subj_delta['group']

        # Get model parameters
        r_post1 = subj_delta['r_post1']
        delta_logpi = subj_delta['delta_log_pi']

        # Compute R based on model
        if model_name == 'M0':
            r = r_post1
        elif model_name == 'M1':
            beta_col = f'beta_{group}'
            if beta_col in model_row.index and pd.notna(model_row[beta_col]):
                beta = model_row[beta_col]
                r = r_post1 + beta * delta_logpi
            else:
                continue
        elif model_name == 'M1-EXP':
            beta_col = f'beta_{group}'
            if beta_col in model_row.index and pd.notna(model_row[beta_col]):
                beta = model_row[beta_col]
                r = r_post1 * np.exp(beta * delta_logpi)
            else:
                continue
        elif model_name == 'M2':
            lam_col = f'lam_{group}'
            if lam_col in model_row.index and pd.notna(model_row[lam_col]):
                lam = model_row[lam_col]
                r = r_post1 * (1 - lam * np.tanh(delta_logpi))
            else:
                continue
        else:
            continue

        # Get plateau
        b_col = f'b_{group}'
        if b_col in model_row.index and pd.notna(model_row[b_col]):
            b = model_row[b_col]
        else:
            b = model_row.get('b', 0.0)

        # Get Kalman parameters
        params = ModelParams()

        # Generate predictions
        errors_obs = subj_trials['error'].values
        predictions = kalman_predict(errors_obs[:10], r, params.m, params.A, params.Q, b)

        for t, (pred, obs) in enumerate(zip(predictions, errors_obs)):
            predictions_list.append({
                'subject': subject_id,
                'group': group,
                'trial': t + 1,
                'error_obs': obs,
                'error_pred': pred,
                'residual': obs - pred,
                'model': model_name,
                'R': r,
                'b': b
            })

    return pd.DataFrame(predictions_list)


def plot_group_adaptation_curves(predictions: pd.DataFrame, out_path: Path):
    """Plot observed vs predicted curves by group and model."""
    groups = ['EC', 'EO+', 'EO-']
    models = predictions['model'].unique()

    n_models = len(models)
    n_groups = len(groups)

    fig, axes = plt.subplots(n_groups, 1, figsize=(12, 10), sharex=True)

    group_colors = {'EC': '#e74c3c', 'EO+': '#3498db', 'EO-': '#2ecc71'}
    model_styles = {
        'M0': {'linestyle': '--', 'alpha': 0.5, 'color': 'gray', 'label': 'M0'},
        'M1': {'linestyle': '-.', 'alpha': 0.7, 'color': 'blue', 'label': 'M1'},
        'M1-EXP': {'linestyle': '-', 'alpha': 0.9, 'color': 'red', 'linewidth': 2.5, 'label': 'M1-exp'},
        'M2': {'linestyle': ':', 'alpha': 0.6, 'color': 'purple', 'label': 'M2'}
    }

    for ax, group in zip(axes, groups):
        group_data = predictions[predictions['group'] == group]

        # Plot observed data (mean ± SE)
        obs_mean = group_data.groupby('trial')['error_obs'].mean()
        obs_se = group_data.groupby('trial')['error_obs'].sem()
        trials = obs_mean.index

        ax.plot(trials, obs_mean, 'o-', color='black', linewidth=2,
               markersize=3, alpha=0.8, label='Observed', zorder=10)
        ax.fill_between(trials, obs_mean - obs_se, obs_mean + obs_se,
                        color='black', alpha=0.1)

        # Plot model predictions
        for model in models:
            model_data = group_data[group_data['model'] == model]
            pred_mean = model_data.groupby('trial')['error_pred'].mean()

            style = model_styles.get(model, {})
            ax.plot(trials, pred_mean, **style, zorder=5)

        ax.set_ylabel(f'{group}\nError (°)', fontsize=11, fontweight='bold')
        ax.set_title(f'Group {group} (N={group_data["subject"].nunique()})',
                    fontsize=12, fontweight='bold', loc='left')
        ax.grid(alpha=0.3)
        ax.legend(loc='upper right', fontsize=9, ncol=2)

        # Add horizontal line at plateau
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)

    axes[-1].set_xlabel('Trial', fontsize=12, fontweight='bold')
    fig.suptitle('Posterior Predictive Checks: Observed vs Predicted Adaptation Curves',
                fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def plot_residuals_by_trial(predictions: pd.DataFrame, out_path: Path):
    """Plot residuals vs trial number."""
    models = predictions['model'].unique()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, model in zip(axes, models):
        model_data = predictions[predictions['model'] == model]

        # Scatter plot with transparency
        ax.scatter(model_data['trial'], model_data['residual'],
                  alpha=0.1, s=10, color='steelblue')

        # Mean residual by trial
        residual_mean = model_data.groupby('trial')['residual'].mean()
        residual_se = model_data.groupby('trial')['residual'].sem()
        trials = residual_mean.index

        ax.plot(trials, residual_mean, 'o-', color='red', linewidth=2,
               markersize=4, label='Mean residual')
        ax.fill_between(trials, residual_mean - 2*residual_se, residual_mean + 2*residual_se,
                        color='red', alpha=0.2)

        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Trial', fontsize=11, fontweight='bold')
        ax.set_ylabel('Residual (obs - pred)', fontsize=11, fontweight='bold')
        ax.set_title(f'{model}', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(loc='best', fontsize=9)

        # Add RMSE annotation
        rmse = np.sqrt((model_data['residual']**2).mean())
        ax.text(0.05, 0.95, f'RMSE = {rmse:.2f}°',
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               va='top')

    fig.suptitle('Residual Diagnostics by Model', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def compute_fit_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute fit quality metrics for each model."""
    metrics = []

    for model in predictions['model'].unique():
        model_data = predictions[predictions['model'] == model]

        # Compute metrics
        residuals = model_data['residual'].values
        obs = model_data['error_obs'].values
        pred = model_data['error_pred'].values

        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))

        # R² (proportion of variance explained)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((obs - np.mean(obs))**2)
        r2 = 1 - (ss_res / ss_tot)

        # Correlation
        r = np.corrcoef(obs, pred)[0, 1]

        metrics.append({
            'model': model,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'correlation': r,
            'n_predictions': len(residuals)
        })

    return pd.DataFrame(metrics)


def main():
    parser = argparse.ArgumentParser(description="Posterior predictive checks")
    parser.add_argument("--trials", type=Path,
                       default=Path("data/derived/adaptation_trials.csv"))
    parser.add_argument("--delta", type=Path,
                       default=Path("data/derived/proprio_delta_pi.csv"))
    parser.add_argument("--results", type=Path,
                       default=Path("data/derived/model_comparison_tier1_logpi_2000x4.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    parser.add_argument("--out-csv", type=Path,
                       default=Path("data/derived/model_predictions.csv"))

    args = parser.parse_args()

    print("Loading data...")
    trials = pd.read_csv(args.trials)
    delta = pd.read_csv(args.delta)
    results = pd.read_csv(args.results)

    # Add delta_log_pi if needed
    if 'delta_log_pi' not in delta.columns:
        delta['delta_log_pi'] = np.log(delta['precision_post1']) - np.log(delta['precision_pre'])
    delta['r_post1'] = 1.0 / delta['precision_post1']

    # Filter to converged models
    converged_models = results[results['max_rhat'] < 1.1]['model'].tolist()
    print(f"Converged models: {converged_models}")

    # Compute predictions for each model
    all_predictions = []
    for model in converged_models:
        if model in ['M-Q', 'M-HYBRID']:
            continue  # Skip non-converged
        print(f"\nComputing predictions for {model}...")
        preds = compute_model_predictions(trials, delta, results, model)
        all_predictions.append(preds)

    predictions = pd.concat(all_predictions, ignore_index=True)

    # Save predictions
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.out_csv, index=False)
    print(f"\nSaved predictions to: {args.out_csv}")

    # Compute fit metrics
    print("\n=== Fit Quality Metrics ===")
    fit_metrics = compute_fit_metrics(predictions)
    print(fit_metrics.to_string(index=False))
    fit_metrics.to_csv(args.out_csv.parent / "model_fit_metrics.csv", index=False)

    # Generate plots
    print("\nGenerating plots...")
    plot_group_adaptation_curves(predictions, args.out_dir / "ppc_adaptation_curves_by_group.png")
    plot_residuals_by_trial(predictions, args.out_dir / "residuals_by_trial.png")

    print("\n✅ Posterior predictive checks complete!")


if __name__ == "__main__":
    main()
