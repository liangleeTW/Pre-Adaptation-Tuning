"""
Posterior Predictive Checks for Final Three Models.

This script:
1. Loads fitted model results from fit_real_data.py
2. Generates one-step-ahead predictions using posterior mean parameters
3. Computes fit metrics (R², RMSE, correlation, bias)
4. Creates visualization comparing observed vs predicted learning curves
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


# ============================================================================
# Kalman Prediction Functions
# ============================================================================

def kalman_predict_standard(errors_obs: np.ndarray, r_measure: float,
                            m: float, A: float, Q: float, b: float) -> np.ndarray:
    """
    One-step-ahead predictions for M1-Coupling (standard Kalman filter).

    At each trial t:
    1. Predict y_t using state x_{t-1}
    2. Update state using observed y_t
    3. Repeat
    """
    n_trials = len(errors_obs)
    predictions = np.zeros(n_trials)

    x = 0.0
    p = 1.0

    for t in range(n_trials):
        # Predict
        x_pred = A * x
        p_pred = A * p * A + Q

        # One-step-ahead forecast
        y_pred = -x_pred + (m + b)
        predictions[t] = y_pred

        # Update using actual observation
        y_obs = errors_obs[t]
        s = p_pred + r_measure
        v = y_obs - y_pred
        k = -p_pred / s
        x = x_pred + k * v
        p = (1.0 + k) * p_pred

    return predictions


def kalman_predict_separated(errors_obs: np.ndarray, r_state: float, r_obs: float,
                             m: float, A: float, Q: float, b: float) -> np.ndarray:
    """
    One-step-ahead predictions for M2-Dissociation and M3-DD.

    Key: Uses r_state for Kalman gain, but total variance = r_state + r_obs
    """
    n_trials = len(errors_obs)
    predictions = np.zeros(n_trials)

    x = 0.0
    p = 1.0

    for t in range(n_trials):
        # Predict
        x_pred = A * x
        p_pred = A * p * A + Q

        # One-step-ahead forecast
        y_pred = -x_pred + (m + b)
        predictions[t] = y_pred

        # Update using actual observation
        y_obs = errors_obs[t]

        # Kalman gain uses r_state only
        s_kalman = p_pred + r_state
        v = y_obs - y_pred
        k = -p_pred / s_kalman
        x = x_pred + k * v
        p = (1.0 + k) * p_pred

    return predictions


# ============================================================================
# Prediction Generation
# ============================================================================

def compute_predictions(trials: pd.DataFrame, delta: pd.DataFrame,
                       results: pd.DataFrame, model_name: str,
                       m: float = -12.1, A: float = 1.0, Q: float = 1e-4) -> pd.DataFrame:
    """
    Generate one-step-ahead predictions for a fitted model.

    Args:
        trials: Trial-level data (subject, trial, error)
        delta: Subject-level proprioceptive tuning data
        results: Model fit results from fit_real_data.py
        model_name: One of 'M1-Coupling', 'M2-Dissociation', 'M3-DD'
    """
    model_row = results[results['model'] == model_name].iloc[0]
    predictions_list = []

    for subject_id in trials['subject'].unique():
        subj_trials = trials[trials['subject'] == subject_id].sort_values('trial')
        subj_delta = delta[delta['ID'] == subject_id]

        if subj_delta.empty:
            continue

        subj_delta = subj_delta.iloc[0]
        group = subj_delta['group']

        # Get subject parameters
        r_post1 = subj_delta['r_post1']
        delta_logpi = subj_delta['delta_log_pi']

        # Get plateau (assumes group-specific b)
        b_col = f'b_{group}' if f'b_{group}' in model_row.index else 'b'
        b = model_row[b_col] if pd.notna(model_row.get(b_col)) else model_row.get('b', 0)

        # ====================================================================
        # Model-specific parameter extraction
        # ====================================================================

        if model_name == 'M1-Coupling':
            # Single R parameter
            beta_col = f'beta_{group}'
            beta = model_row[beta_col]
            r_measure = r_post1 * np.exp(beta * delta_logpi)

            # Use standard Kalman
            errors_obs = subj_trials['error'].values
            predictions = kalman_predict_standard(errors_obs, r_measure, m, A, Q, b)

            # Store parameters
            r_state = r_measure
            r_obs = 0.0  # Not separated in M1

        elif model_name == 'M2-Dissociation':
            # Separated R_state and R_obs
            beta_state_col = f'beta_state_{group}'
            beta_obs_col = f'beta_obs_{group}'
            r_obs_base_col = f'r_obs_base_{group}'

            beta_state = model_row[beta_state_col]
            beta_obs = model_row[beta_obs_col]
            r_obs_base = model_row[r_obs_base_col]

            r_state = r_post1 * np.exp(beta_state * delta_logpi)
            r_obs = r_obs_base * np.exp(beta_obs * delta_logpi)

            # Use separated Kalman
            errors_obs = subj_trials['error'].values
            predictions = kalman_predict_separated(errors_obs, r_state, r_obs, m, A, Q, b)

        elif model_name == 'M3-DD':
            # Decomposed R_obs
            beta_state_col = f'beta_state_{group}'
            beta_obs_col = f'beta_obs_{group}'
            r_cognitive_col = f'r_cognitive_{group}'

            beta_state = model_row[beta_state_col]
            beta_obs = model_row[beta_obs_col]
            r_cognitive = model_row[r_cognitive_col]

            r_state = r_post1 * np.exp(beta_state * delta_logpi)

            # R_obs = sensory (fixed) + cognitive (modulated)
            r_motor = subj_delta['openloop_var_post1']
            r_visual = subj_delta['visual_var_post1']
            r_obs = r_motor + r_visual + r_cognitive * np.exp(beta_obs * delta_logpi)

            # Use separated Kalman
            errors_obs = subj_trials['error'].values
            predictions = kalman_predict_separated(errors_obs, r_state, r_obs, m, A, Q, b)

        else:
            raise ValueError(f"Unknown model: {model_name}")

        # ====================================================================
        # Store predictions
        # ====================================================================

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


# ============================================================================
# Metrics and Visualization
# ============================================================================

def compute_fit_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Compute fit quality metrics for each model.

    Metrics:
    - R²: Variance explained (1 = perfect, 0 = as good as mean, <0 = worse than mean)
    - RMSE: Root mean square error (lower is better)
    - Correlation: Pearson r between observed and predicted
    - Late bias: Mean residual in trials 70-100 (should be near zero)
    """
    metrics = []

    for model in predictions['model'].unique():
        model_data = predictions[predictions['model'] == model]

        residuals = model_data['residual'].values
        obs = model_data['error_obs'].values
        pred = model_data['error_pred'].values

        # RMSE
        rmse = np.sqrt(np.mean(residuals**2))

        # MAE
        mae = np.mean(np.abs(residuals))

        # R²
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((obs - np.mean(obs))**2)
        r2 = 1 - (ss_res / ss_tot)

        # Correlation
        r = np.corrcoef(obs, pred)[0, 1]

        # Late-trial bias (trials 70-100)
        late_trials = model_data[model_data['trial'].between(70, 100)]
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


def plot_learning_curves(predictions: pd.DataFrame, out_path: Path):
    """
    Plot observed vs predicted learning curves by group and model.
    """
    groups = ['EC', 'EO+', 'EO-']
    models = predictions['model'].unique()

    fig, axes = plt.subplots(len(groups), 1, figsize=(12, 10), sharex=True)

    group_colors = {'EC': '#e74c3c', 'EO+': '#3498db', 'EO-': '#2ecc71'}
    model_styles = {
        'M1-Coupling': {'linestyle': '--', 'alpha': 0.6, 'linewidth': 2, 'color': 'gray'},
        'M2-Dissociation': {'linestyle': '-', 'alpha': 0.9, 'linewidth': 2.5, 'color': 'red'},
        'M3-DD': {'linestyle': '-.', 'alpha': 0.8, 'linewidth': 2, 'color': 'blue'}
    }

    for ax, group in zip(axes, groups):
        group_data = predictions[predictions['group'] == group]

        # Plot observed data
        obs_mean = group_data.groupby('trial')['error_obs'].mean()
        obs_se = group_data.groupby('trial')['error_obs'].sem()
        trials = obs_mean.index

        ax.plot(trials, obs_mean, 'o-', color='black', linewidth=2,
               markersize=2, alpha=0.8, label='Observed', zorder=10)
        ax.fill_between(trials, obs_mean - obs_se, obs_mean + obs_se,
                        color='black', alpha=0.1)

        # Plot model predictions
        for model in models:
            if model not in group_data['model'].values:
                continue

            model_data = group_data[group_data['model'] == model]
            pred_mean = model_data.groupby('trial')['error_pred'].mean()

            style = model_styles.get(model, {})
            ax.plot(trials, pred_mean, label=model, **style, zorder=5)

        ax.set_ylabel(f'{group}\nError (°)', fontsize=11, fontweight='bold')
        ax.set_title(f'Group {group} (N={group_data["subject"].nunique()})',
                    fontsize=12, fontweight='bold', loc='left')
        ax.grid(alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)

    axes[-1].set_xlabel('Trial', fontsize=12, fontweight='bold')
    fig.suptitle('Posterior Predictive Checks: Observed vs Predicted Learning Curves',
                fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def plot_r2_comparison(metrics: pd.DataFrame, out_path: Path):
    """Bar plot comparing R² across models."""
    fig, ax = plt.subplots(figsize=(8, 6))

    models = metrics['model'].values
    r2_values = metrics['R2'].values

    colors = ['gray' if r2 < 0 else 'red' if 'M2' in m else 'blue' for m, r2 in zip(models, r2_values)]

    bars = ax.bar(range(len(models)), r2_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    # Add value labels
    for i, (v, m) in enumerate(zip(r2_values, models)):
        ax.text(i, v + (0.01 if v > 0 else -0.05), f'{v:.3f}',
               ha='center', fontweight='bold', fontsize=11)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel('R² (One-Step-Ahead)', fontsize=12, fontweight='bold')
    ax.set_title('Predictive Accuracy: Variance Explained', fontsize=13, fontweight='bold')
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.grid(axis='y', alpha=0.3)

    # Add interpretation
    ax.text(0.5, 0.95, 'R² > 0: Better than mean\nR² < 0: Worse than mean',
           transform=ax.transAxes, ha='center', va='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="PPC for final three models")
    parser.add_argument("--trials", type=Path,
                       default=Path("data/raw/adaptation_trials.csv"))
    parser.add_argument("--delta", type=Path,
                       default=Path("data/raw/proprio_delta_pi.csv"))
    parser.add_argument("--results", type=Path,
                       default=Path("data/derived/model_comparison_final.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    parser.add_argument("--models", type=str,
                       default="M1-Coupling,M2-Dissociation,M3-DD",
                       help="Comma-separated list of models")

    args = parser.parse_args()

    print("="*70)
    print("Posterior Predictive Checks for Final Models")
    print("="*70)

    # Load data
    print("\nLoading data...")
    trials = pd.read_csv(args.trials)
    delta = pd.read_csv(args.delta)
    results = pd.read_csv(args.results)

    # Ensure required columns
    if 'delta_log_pi' not in delta.columns:
        delta['delta_log_pi'] = np.log(delta['precision_post1']) - np.log(delta['precision_pre'])
    delta['r_post1'] = 1.0 / delta['precision_post1']

    print(f"Loaded {len(trials['subject'].unique())} subjects, {len(trials)} trials")

    # Parse models
    models = [m.strip() for m in args.models.split(",")]
    print(f"\nGenerating predictions for: {', '.join(models)}")

    # Compute predictions for each model
    all_predictions = []
    for model in models:
        if model not in results['model'].values:
            print(f"⚠️  {model} not found in results, skipping")
            continue

        print(f"\n  Computing predictions for {model}...")
        preds = compute_predictions(trials, delta, results, model)
        all_predictions.append(preds)

    if not all_predictions:
        print("\n❌ No valid predictions generated!")
        return

    predictions = pd.concat(all_predictions, ignore_index=True)

    # Save predictions
    out_csv = args.out_dir.parent / "data/derived/final_model_predictions.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_csv, index=False)
    print(f"\nSaved predictions to: {out_csv}")

    # Compute fit metrics
    print("\n" + "="*70)
    print("Fit Quality Metrics")
    print("="*70)
    fit_metrics = compute_fit_metrics(predictions)
    print(fit_metrics.to_string(index=False))

    fit_metrics_path = args.out_dir.parent / "data/derived/final_model_fit_metrics.csv"
    fit_metrics.to_csv(fit_metrics_path, index=False)
    print(f"\nSaved metrics to: {fit_metrics_path}")

    # Generate plots
    print("\n" + "="*70)
    print("Generating Plots")
    print("="*70)

    plot_learning_curves(predictions, args.out_dir / "ppc_learning_curves.png")
    plot_r2_comparison(fit_metrics, args.out_dir / "ppc_r2_comparison.png")

    print("\n" + "="*70)
    print("✅ Posterior Predictive Checks Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
