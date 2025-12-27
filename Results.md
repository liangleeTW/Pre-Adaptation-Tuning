# Results (Simulation Phase)

This document summarizes the results obtained so far and maps them to the
questions and steps in `Plan.md`. Unless otherwise noted, the results below
come from the latest sweep outputs in `data/sim_sweep_20251226_122357/` using
the current simulation code (plateau implemented as observational bias; state
update uses bias-free error).

## 1. Scope and Core Questions (Plan.md Section 1)

Primary research questions:
1) Does proprioceptive reliability at onset influence subsequent error
   correction?
2) If so, is that influence linear (M1) or bounded/saturating (M2)?

At this stage we are only evaluating these questions in simulated data to
validate expected signatures and confounds. Real-data fitting is not yet done.

## 2. Pre-Model Checks (Plan.md Section 2)

Status: not yet run on real data.

Planned checks:
- Independence between onset reliability and tuning direction.
- Phenomenological mapping between early learning and precision metrics.

Next step: run these analyses once empirical data are available.

## 3. Simulation Results (Plan.md Section 3)

All metrics are derived from `data/sim_sweep_20251226_122357/summary.csv` and
per-run summaries using the early window (trials 1-10) and late window (last 10
trials), with calibration based on empirical adaptation errors.

### 3.1 Simulation Settings (Plan.md 3.1, 3.4)

We simulate group-structured tuning using empirical proprioceptive data from
`raw/` to set realistic ranges for \(\Delta\pi\), and we calibrate the
adaptation error scale using empirical adaptation-phase errors (trials 51-150).

Settings:
- Models: M0, M1, M2.
- M2 \(\lambda\): \(-0.8,-0.5,-0.2,0,0.2,0.5,0.8\).
- M1 \(\beta\): \(0,0.2,0.5,0.8\).
- Trials per subject: 100.
- Subjects per run: 60 (multinomially allocated to groups by observed weights).
- Calibrated perturbation scale: \(m\) set to mean early adaptation error.
- Plateau bias: \(b\) set to mean late adaptation error (plus a zero condition).
- State update uses bias-free error \(e_t\); observed error adds \(b\).

Plots:
- `data/derived/figures/delta_pi_groups.png`
- `data/derived/figures/delta_log_pi_groups.png`

Interpretation:
Group \(\Delta\pi\) distributions are positive and overlapping, which limits
group separation and makes modulation effects subtle. Calibration aligns the
simulation’s error scale and plateau with observed behavior.

### 3.2 Early Learning vs Modulation Strength (Aggregate)

Plot:
- `data/sim_sweep_20251226_122357/figures/early_slope_vs_strength.png`

Axis interpretation:
- X-axis: modulation strength (\(\beta\) for M1, \(\lambda\) for M2).
- Y-axis: early slope of mean error vs trial (trials 1-10); more negative
  slopes indicate faster early correction.

Cognitive interpretation:
With calibration, early slopes are negative (errors decline). Modulation effects
are small at the aggregate level, consistent with modest \(\Delta\pi\)
separation. M2 shows a weak trend with \(\lambda\); M1 is nearly flat.

### 3.3 Early Learning vs Modulation Strength (Group-Specific)

Plots:
- `data/sim_sweep_20251226_122357/figures/early_slope_vs_strength_ec.png`
- `data/sim_sweep_20251226_122357/figures/early_slope_vs_strength_eoplus.png`
- `data/sim_sweep_20251226_122357/figures/early_slope_vs_strength_eominus.png`
- `data/sim_sweep_20251226_122357/figures/early_slope_vs_strength_groups_combined.png`
- `data/sim_sweep_20251226_122357/figures/early_slope_vs_lambda_groups.png`
- `data/sim_sweep_20251226_122357/figures/early_slope_vs_beta_groups.png`

Axis interpretation:
- X-axis: modulation strength (same scale for M1/M2).
- Y-axis: early slope (trials 1-10) per group.

Cognitive interpretation:
EO+ shows the clearest M1 vs M2 divergence, making it the most diagnostic group.
EO- is relatively flat across strengths. EC shows modest sensitivity but no
strong directional shift. In the combined plot, solid (M2) and dashed (M1)
diverge most clearly for EO+.

Hypothesis match:
Group patterns remain consistent with a bounded modulation account, but effect
sizes are small under realistic \(\Delta\pi\) overlap.

### 3.4 Delta Precision vs Early Error (Mechanistic Signature)

Plot:
- `data/sim_sweep_20251226_122357/figures/delta_pi_vs_early_abs_error.png`

Axis interpretation:
- X-axis: modulation strength (\(\beta\) or \(\lambda\)).
- Y-axis: corr(\(\Delta\pi\), early \(|error|\)).

Cognitive interpretation:
M1 yields positive correlations (larger \(\Delta\pi\) -> larger early error),
while M2 yields negative correlations (larger \(\Delta\pi\) -> smaller early
error). This remains the key sign-diagnostic contrast for model recovery.

### 3.5 Plateau Confound and Late Error

Plots:
- `data/sim_sweep_20251226_122357/figures/late_error_vs_plateau.png`
- `data/sim_sweep_20251226_122357/figures/late_error_residual_vs_plateau.png`
- `data/sim_sweep_20251226_122357/figures/late_error_by_group.png`

Axis interpretation:
- Late error vs plateau: X-axis is \(b\) (or fraction); Y-axis is late mean error.
- Residual plot: Y-axis is late mean error minus \(b\).
- Group plot: late mean error by group and model.

Cognitive interpretation:
The plateau confound is visible and matches calibration (late errors shift by
approximately \(b\)). The residual plot checks that late error is explained by
the plateau term rather than by continued learning. Group-level late errors are
now visible and can be compared to empirical late errors.

### 3.6 Collinearity Check

Plots:
- `data/sim_sweep_20251226_122357/figures/rho_realization_scatter.png`
- `data/sim_sweep_20251226_122357/figures/rho_realization_boxplot.png`
- `data/sim_sweep_20251226_122357/figures/rho_sample_scatter.png`

Axis interpretation:
- Scatter/boxplot: target \(\rho(R^{post1}_P, \Delta\pi)\) vs realized correlation.
- Sample scatter: empirical relationship between \(R^{post1}_P\) and \(\Delta\pi\).

Cognitive interpretation:
Collinearity limits identifiability of \(\beta\) and \(\lambda\). The boxplot
shows dispersion around target values, while the sample scatter makes the
overlap intuitive.

### 3.7 Recovery Fitting (Simulation Only)

Outputs:
- `data/sim_sweep_20251226_122357/recovery.csv`
- `data/sim_sweep_20251226_122357/recovery_summary.csv`
- `data/sim_sweep_20251226_122357/recovery_figures/recovery_beta_scatter.png`
- `data/sim_sweep_20251226_122357/recovery_figures/recovery_lambda_scatter.png`

Summary interpretation:
Recovery improves with calibration but remains imperfect. M2 shows stronger
correlation between true and estimated \(\lambda\) than M1 does for \(\beta\),
indicating better identifiability for the bounded modulation model under
realistic tuning overlap. Biases remain substantial, motivating further
recovery checks and potential model refinements before real-data fitting.

## 4. Real-Data Fitting (Plan.md Section 4)

Status: not started. The simulation phase now includes recovery fitting on
simulated data; real-data fitting awaits integration of adaptation-phase error
trials.

## 5. Visualization Outputs

Plots are referenced within each subsection above.

Interactive notebook:
- `notebooks/sweep_analysis.ipynb`

## 6. Next Steps (Aligned to Plan.md)

1) Run recovery with group-specific modulation and assess identifiability.
2) Calibrate \(Q\) and \(R\) using adaptation error dispersion.
3) Run pre-model checks on empirical data (Plan.md 2.1, 2.2) and integrate
   adaptation-phase error trials for real-data fitting.
