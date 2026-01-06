"""
Posterior predictive checks for M-obs models.
Validate that separating observation noise fixes the negative R² problem.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("paper", font_scale=1.1)
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


def kalman_predict_obs(errors_obs: np.ndarray, r_state: float, r_obs: float,
                        m: float, A: float, Q: float, b: float) -> np.ndarray:
    """
    One-step-ahead predictions using Kalman filter.

    Args:
        errors_obs: All observed errors (use for updating filter)
        r_state: State uncertainty (affects learning)
        r_obs: Observation noise (doesn't affect learning)
        m, A, Q, b: Kalman parameters

    Returns:
        predictions: One-step-ahead predicted errors for each trial
    """
    n_trials = len(errors_obs)
    predictions = np.zeros(n_trials)

    # Initialize
    x = 0.0
    p = 1.0

    for t in range(n_trials):
        # Predict
        x_pred = A * x
        p_pred = A * p * A + Q

        # Predicted observation (this is our one-step-ahead forecast)
        y_pred = -x_pred + (m + b)
        predictions[t] = y_pred

        # Update using ACTUAL observation (this is key for one-step-ahead)
        y_obs = errors_obs[t]

        # Kalman update (uses R_state only)
        s_kalman = p_pred + r_state
        v = y_obs - y_pred
        k = -p_pred / s_kalman
        x = x_pred + k * v
        p = (1.0 + k) * p_pred

    return predictions


def compute_model_predictions_obs(trials: pd.DataFrame, delta: pd.DataFrame,
                                   results: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Compute predictions for M-obs models."""
    model_row = results[results['model'] == model_name].iloc[0]

    predictions_list = []

    for subject_id in trials['subject'].unique():
        subj_trials = trials[trials['subject'] == subject_id].sort_values('trial')
        subj_delta = delta[delta['ID'] == subject_id]

        if subj_delta.empty:
            continue

        subj_delta = subj_delta.iloc[0]
        group = subj_delta['group']

        # Get subject-specific parameters
        r_post1 = subj_delta['r_post1']
        delta_logpi = subj_delta['delta_log_pi']

        # Get plateau
        b_col = f'b_{group}'
        b = model_row[b_col]

        # Compute R_state and R_obs based on model
        if model_name == 'M-obs-fixed':
            r_state = r_post1
            r_obs = model_row['r_obs']

        elif model_name == 'M-obs':
            r_state = r_post1
            beta_obs_col = f'beta_obs_{group}'
            r_obs_base_col = f'r_obs_base_{group}'
            beta_obs = model_row[beta_obs_col]
            r_obs_base = model_row[r_obs_base_col]
            r_obs = r_obs_base * np.exp(beta_obs * delta_logpi)

        elif model_name == 'M-twoR':
            # Both modulated
            beta_state_col = f'beta_state_{group}'
            beta_obs_col = f'beta_obs_{group}'
            r_obs_base_col = f'r_obs_base_{group}'

            beta_state = model_row[beta_state_col]
            beta_obs = model_row[beta_obs_col]
            r_obs_base = model_row[r_obs_base_col]

            r_state = r_post1 * np.exp(beta_state * delta_logpi)
            r_obs = r_obs_base * np.exp(beta_obs * delta_logpi)
        else:
            continue

        # Kalman parameters (fixed)
        m = -12.1
        A = 1.0
        Q = 1e-4

        # Generate one-step-ahead predictions
        errors_obs = subj_trials['error'].values
        predictions = kalman_predict_obs(errors_obs, r_state, r_obs, m, A, Q, b)

        for t, (pred, obs) in enumerate(zip(predictions, errors_obs)):
            predictions_list.append({
                'subject': subject_id,
                'group': group,
                'trial': t + 1,
                'error_obs': obs,
                'error_pred': pred,
                'residual': obs - pred,
                'model': model_name,
                'R_state': r_state,
                'R_obs': r_obs,
                'b': b
            })

    return pd.DataFrame(predictions_list)


def plot_group_adaptation_curves_obs(predictions: pd.DataFrame, out_path: Path):
    """Plot observed vs predicted curves for M-obs models."""
    groups = ['EC', 'EO+', 'EO-']
    models = predictions['model'].unique()

    fig, axes = plt.subplots(len(groups), 1, figsize=(12, 10), sharex=True)

    group_colors = {'EC': '#e74c3c', 'EO+': '#3498db', 'EO-': '#2ecc71'}
    model_styles = {
        'M-obs-fixed': {'linestyle': '--', 'alpha': 0.6, 'color': 'gray', 'label': 'M-obs-fixed'},
        'M-obs': {'linestyle': '-.', 'alpha': 0.8, 'color': 'blue', 'linewidth': 2, 'label': 'M-obs'},
        'M-twoR': {'linestyle': '-', 'alpha': 0.9, 'color': 'red', 'linewidth': 2.5, 'label': 'M-twoR (best)'}
    }

    for ax, group in zip(axes, groups):
        group_data = predictions[predictions['group'] == group]

        # Plot observed data
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
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)

    axes[-1].set_xlabel('Trial', fontsize=12, fontweight='bold')
    fig.suptitle('M-obs Models: Posterior Predictive Checks',
                fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def compute_fit_metrics_obs(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute fit quality metrics."""
    metrics = []

    for model in predictions['model'].unique():
        model_data = predictions[predictions['model'] == model]

        residuals = model_data['residual'].values
        obs = model_data['error_obs'].values
        pred = model_data['error_pred'].values

        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))

        # R²
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((obs - np.mean(obs))**2)
        r2 = 1 - (ss_res / ss_tot)

        # Correlation
        r = np.corrcoef(obs, pred)[0, 1]

        # Late-trial residual bias (trials 40-60)
        late_trials = model_data[model_data['trial'].between(40, 60)]
        late_residual_mean = late_trials['residual'].mean()
        late_residual_std = late_trials['residual'].std()

        metrics.append({
            'model': model,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'correlation': r,
            'late_residual_mean': late_residual_mean,
            'late_residual_std': late_residual_std,
            'n_predictions': len(residuals)
        })

    return pd.DataFrame(metrics)


def main():
    parser = argparse.ArgumentParser(description="PPC for M-obs models")
    parser.add_argument("--trials", type=Path,
                       default=Path("data/derived/adaptation_trials.csv"))
    parser.add_argument("--delta", type=Path,
                       default=Path("data/derived/proprio_delta_pi.csv"))
    parser.add_argument("--results", type=Path,
                       default=Path("data/derived/m_obs_results.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    parser.add_argument("--out-csv", type=Path,
                       default=Path("data/derived/m_obs_predictions.csv"))

    args = parser.parse_args()

    print("Loading data...")
    trials = pd.read_csv(args.trials)
    delta = pd.read_csv(args.delta)
    results = pd.read_csv(args.results)

    # Add delta_log_pi if needed
    if 'delta_log_pi' not in delta.columns:
        delta['delta_log_pi'] = np.log(delta['precision_post1']) - np.log(delta['precision_pre'])
    delta['r_post1'] = 1.0 / delta['precision_post1']

    # Compute predictions for each model
    all_predictions = []
    for model in results['model'].unique():
        print(f"\nComputing predictions for {model}...")
        preds = compute_model_predictions_obs(trials, delta, results, model)
        all_predictions.append(preds)

    predictions = pd.concat(all_predictions, ignore_index=True)

    # Save predictions
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.out_csv, index=False)
    print(f"\nSaved predictions to: {args.out_csv}")

    # Compute fit metrics
    print("\n=== Fit Quality Metrics ===")
    fit_metrics = compute_fit_metrics_obs(predictions)
    print(fit_metrics.to_string(index=False))
    fit_metrics.to_csv(args.out_csv.parent / "m_obs_fit_metrics.csv", index=False)

    # Generate plots
    print("\nGenerating plots...")
    plot_group_adaptation_curves_obs(predictions, args.out_dir / "ppc_m_obs_adaptation_curves.png")

    print("\n✅ M-obs posterior predictive checks complete!")

    # Print comparison to old models
    print("\n=== Comparison to Previous Models ===")
    print("Previous best (M1-exp):")
    print("  WAIC = 44,425")
    print("  R² = -1.05 (catastrophic)")
    print("\nCurrent best (M-twoR):")
    print(f"  WAIC = {results[results['model']=='M-twoR']['waic'].iloc[0]:.1f}")
    print(f"  R² = {fit_metrics[fit_metrics['model']=='M-twoR']['R2'].iloc[0]:.3f}")
    print(f"  ΔWAIC = {results[results['model']=='M-twoR']['waic'].iloc[0] - 44425:.1f}")


if __name__ == "__main__":
    main()
