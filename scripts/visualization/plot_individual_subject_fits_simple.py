#!/usr/bin/env python3
"""
Plot individual subject fits - simplified version using available data.
Shows observed vs predicted errors for subjects spanning Δπ range.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
slopes = pd.read_csv('data/derived/state_based_slopes.csv')
predictions = pd.read_csv('data/derived/m_obs_predictions.csv')

# Select 4 subjects spanning Δπ range
slopes_sorted = slopes.sort_values('delta_log_pi')
n = len(slopes_sorted)
selected_indices = [int(n*0.1), int(n*0.35), int(n*0.65), int(n*0.9)]
selected_subjects = slopes_sorted.iloc[selected_indices]['subject'].values

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Individual Subject Fits: M2-dual Captures Heterogeneity Across Δlog π',
             fontsize=14, fontweight='bold')

axes = axes.flatten()

for idx, subject in enumerate(selected_subjects):
    # Get subject info
    subj_info = slopes[slopes['subject'] == subject].iloc[0]
    delta_pi = subj_info['delta_log_pi']
    group = subj_info['group']
    state_slope = subj_info['state_slope']

    # Get predictions for this subject
    subj_pred = predictions[predictions['subject'] == subject].copy()

    if len(subj_pred) == 0:
        continue

    ax = axes[idx]

    # Plot observed vs predicted
    ax.plot(subj_pred['trial'], subj_pred['error_obs'], 'o',
            color='gray', alpha=0.5, markersize=4, label='Observed errors')
    ax.plot(subj_pred['trial'], subj_pred['error_pred'], '-',
            color='red', linewidth=2, label='M2-dual prediction')
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)

    # Calculate R² for this subject
    residuals = subj_pred['error_obs'] - subj_pred['error_pred']
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((subj_pred['error_obs'] - subj_pred['error_obs'].mean())**2)
    r2 = 1 - (ss_res / ss_tot)

    ax.set_xlabel('Trial', fontsize=10)
    ax.set_ylabel('Reach Error (°)', fontsize=10)
    ax.set_title(f'{subject} ({group})\nΔlog π = {delta_pi:.2f}, State slope = {state_slope:.2f}\nR² = {r2:.3f}',
                 fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper right')

plt.tight_layout()
plt.savefig('figures/individual_subject_fits.png', dpi=300, bbox_inches='tight')
print("✅ Created: figures/individual_subject_fits.png")
print(f"   Selected subjects: {', '.join(selected_subjects)}")
delta_pis = [slopes[slopes['subject'] == s]['delta_log_pi'].values[0] for s in selected_subjects]
print(f"   Δπ range: {min(delta_pis):.2f} to {max(delta_pis):.2f}")
