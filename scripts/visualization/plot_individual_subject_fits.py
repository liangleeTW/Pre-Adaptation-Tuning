#!/usr/bin/env python3
"""
Plot individual subject fits showing heterogeneity across Δlog π values.

Selects 4 representative subjects (low, medium-low, medium-high, high Δπ)
and shows:
- Panel A: Observed errors
- Panel B: State trajectory
- Panel C: Predictions vs observed

This demonstrates how M2-dual captures individual differences.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
slopes = pd.read_csv('data/derived/state_based_slopes.csv')
predictions = pd.read_csv('data/derived/m_obs_predictions.csv')
trials = pd.read_csv('data/derived/adaptation_trials.csv')

# Select 4 subjects spanning Δπ range
slopes_sorted = slopes.sort_values('delta_log_pi')
n = len(slopes_sorted)
selected_indices = [int(n*0.1), int(n*0.35), int(n*0.65), int(n*0.9)]
selected_subjects = slopes_sorted.iloc[selected_indices]['subject'].values

# Create figure
fig, axes = plt.subplots(4, 3, figsize=(15, 12))
fig.suptitle('Individual Subject Fits: M2-dual Captures Heterogeneity Across Δlog π',
             fontsize=14, fontweight='bold', y=0.995)

for idx, subject in enumerate(selected_subjects):
    # Get subject data
    subj_info = slopes[slopes['subject'] == subject].iloc[0]
    delta_pi = subj_info['delta_log_pi']
    group = subj_info['group']

    # Get predictions
    subj_pred = predictions[predictions['subject'] == subject].copy()
    subj_trials = trials[trials['subject'] == subject].copy()

    if len(subj_pred) == 0 or len(subj_trials) == 0:
        continue

    # Merge trials with predictions
    subj_data = subj_trials.merge(subj_pred, on=['subject', 'trial'], how='inner')

    # Panel A: Observed errors
    ax = axes[idx, 0]
    ax.plot(subj_data['trial'], subj_data['reach_error'], 'o-',
            color='gray', alpha=0.6, markersize=3, label='Observed')
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_ylabel('Reach Error (°)', fontsize=10)
    ax.set_title(f'{subject}\nΔlog π = {delta_pi:.2f} ({group})', fontsize=9)
    ax.grid(True, alpha=0.3)
    if idx == 3:
        ax.set_xlabel('Trial', fontsize=10)

    # Panel B: State trajectory
    ax = axes[idx, 1]
    ax.plot(subj_pred['trial'], subj_pred['state_mean'], '-',
            color='blue', linewidth=2, label='State estimate')
    ax.fill_between(subj_pred['trial'],
                     subj_pred['state_mean'] - subj_pred['state_std'],
                     subj_pred['state_mean'] + subj_pred['state_std'],
                     alpha=0.3, color='blue')
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_title('Internal State ($x_t$)', fontsize=9)
    ax.grid(True, alpha=0.3)
    if idx == 3:
        ax.set_xlabel('Trial', fontsize=10)

    # Panel C: Predictions vs observed
    ax = axes[idx, 2]
    ax.plot(subj_data['trial'], subj_data['reach_error'], 'o',
            color='gray', alpha=0.5, markersize=3, label='Observed')
    ax.plot(subj_pred['trial'], subj_pred['pred_mean'], '-',
            color='red', linewidth=2, label='M2-dual prediction')
    ax.fill_between(subj_pred['trial'],
                     subj_pred['pred_mean'] - subj_pred['pred_std'],
                     subj_pred['pred_mean'] + subj_pred['pred_std'],
                     alpha=0.2, color='red')
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_title('Model Fit', fontsize=9)
    ax.grid(True, alpha=0.3)
    if idx == 3:
        ax.set_xlabel('Trial', fontsize=10)
    if idx == 0:
        ax.legend(fontsize=8, loc='upper right')

# Column titles
axes[0, 0].text(0.5, 1.15, 'Observed Errors', ha='center', va='bottom',
                transform=axes[0, 0].transAxes, fontsize=11, fontweight='bold')
axes[0, 1].text(0.5, 1.15, 'State Trajectory', ha='center', va='bottom',
                transform=axes[0, 1].transAxes, fontsize=11, fontweight='bold')
axes[0, 2].text(0.5, 1.15, 'Predictions vs Observed', ha='center', va='bottom',
                transform=axes[0, 2].transAxes, fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/individual_subject_fits.png', dpi=300, bbox_inches='tight')
print("✅ Created: figures/individual_subject_fits.png")
print(f"   Selected subjects: {', '.join(selected_subjects)}")
print(f"   Δπ range: {slopes_sorted.iloc[selected_indices[0]]['delta_log_pi']:.2f} to {slopes_sorted.iloc[selected_indices[-1]]['delta_log_pi']:.2f}")
