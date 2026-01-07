#!/usr/bin/env python3
"""
Plot Kalman gain evolution comparing high vs low Δlog π subjects.

Shows how β_state > 0 produces smaller gains (slower learning) for
subjects with higher proprioceptive precision.

Creates 3 panels:
- Panel A: Gain evolution for high vs low Δπ subjects
- Panel B: Average gain vs Δπ across all subjects
- Panel C: State covariance P_t evolution (shows convergence)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load data
slopes = pd.read_csv('data/derived/state_based_slopes.csv')
predictions = pd.read_csv('data/derived/m_obs_predictions.csv')
results = pd.read_csv('data/derived/m_obs_results.csv')

# Merge to get beta values
data = slopes.merge(results, on='subject', how='inner')

# Select high and low Δπ subjects (top and bottom quartiles)
q1 = data['delta_log_pi'].quantile(0.25)
q3 = data['delta_log_pi'].quantile(0.75)

low_deltalogpi = data[data['delta_log_pi'] <= q1]
high_deltalogpi = data[data['delta_log_pi'] >= q3]

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Kalman Gain Evolution: β_state > 0 Produces Slower Learning',
             fontsize=14, fontweight='bold')

# Panel A: Gain evolution for representative subjects
ax = axes[0]

# Select 2-3 subjects from each group
n_examples = 3
low_subjects = low_deltalogpi.sample(n=min(n_examples, len(low_deltalogpi)))['subject'].values
high_subjects = high_deltalogpi.sample(n=min(n_examples, len(high_deltalogpi)))['subject'].values

for subject in low_subjects:
    subj_pred = predictions[predictions['subject'] == subject]
    if 'gain' in subj_pred.columns and len(subj_pred) > 0:
        ax.plot(subj_pred['trial'], np.abs(subj_pred['gain']),
                color='blue', alpha=0.6, linewidth=1.5)

for subject in high_subjects:
    subj_pred = predictions[predictions['subject'] == subject]
    if 'gain' in subj_pred.columns and len(subj_pred) > 0:
        ax.plot(subj_pred['trial'], np.abs(subj_pred['gain']),
                color='red', alpha=0.6, linewidth=1.5)

# Create custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='blue', linewidth=2, label=f'Low Δlog π (< {q1:.2f})'),
    Line2D([0], [0], color='red', linewidth=2, label=f'High Δlog π (> {q3:.2f})')
]
ax.legend(handles=legend_elements, fontsize=10, loc='upper right')
ax.set_xlabel('Trial', fontsize=11)
ax.set_ylabel('|Kalman Gain|', fontsize=11)
ax.set_title('(A) Gain Evolution by Δlog π', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)

# Panel B: Average gain vs Δπ
ax = axes[1]

# Compute average gain for each subject (trials 1-30, early learning)
avg_gains = []
deltalogpi_vals = []

for subject in data['subject'].unique():
    subj_pred = predictions[predictions['subject'] == subject]
    if 'gain' in subj_pred.columns and len(subj_pred) >= 30:
        early_gain = np.abs(subj_pred[subj_pred['trial'] <= 30]['gain']).mean()
        delta_pi = data[data['subject'] == subject]['delta_log_pi'].values[0]
        avg_gains.append(early_gain)
        deltalogpi_vals.append(delta_pi)

avg_gains = np.array(avg_gains)
deltalogpi_vals = np.array(deltalogpi_vals)

# Scatter plot
colors = ['red' if d >= q3 else 'blue' if d <= q1 else 'gray'
          for d in deltalogpi_vals]
ax.scatter(deltalogpi_vals, avg_gains, c=colors, alpha=0.6, s=50)

# Fit line
if len(deltalogpi_vals) > 0:
    slope, intercept, r_value, p_value, std_err = stats.linregress(deltalogpi_vals, avg_gains)
    x_fit = np.linspace(deltalogpi_vals.min(), deltalogpi_vals.max(), 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, 'k--', linewidth=2, alpha=0.7)
    ax.text(0.05, 0.95, f'r = {r_value:.3f}\np = {p_value:.3f}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.set_xlabel('Δlog π', fontsize=11)
ax.set_ylabel('Mean |Gain| (Trials 1-30)', fontsize=11)
ax.set_title('(B) Higher Δπ → Smaller Gain', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Panel C: State covariance P_t evolution
ax = axes[2]

# Plot P_t for representative subjects
for subject in low_subjects:
    subj_pred = predictions[predictions['subject'] == subject]
    if 'P' in subj_pred.columns and len(subj_pred) > 0:
        ax.plot(subj_pred['trial'], subj_pred['P'],
                color='blue', alpha=0.6, linewidth=1.5)

for subject in high_subjects:
    subj_pred = predictions[predictions['subject'] == subject]
    if 'P' in subj_pred.columns and len(subj_pred) > 0:
        ax.plot(subj_pred['trial'], subj_pred['P'],
                color='red', alpha=0.6, linewidth=1.5)

ax.set_xlabel('Trial', fontsize=11)
ax.set_ylabel('State Covariance $P_t$', fontsize=11)
ax.set_title('(C) $P_t$ Convergence', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()
plt.savefig('figures/kalman_gain_evolution.png', dpi=300, bbox_inches='tight')
print("✅ Created: figures/kalman_gain_evolution.png")
print(f"   Low Δπ quartile: < {q1:.2f}")
print(f"   High Δπ quartile: > {q3:.2f}")
if len(deltalogpi_vals) > 0:
    print(f"   Gain vs Δπ correlation: r = {r_value:.3f}, p = {p_value:.3f}")
