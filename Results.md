# Results (Simulation Phase)

This document summarizes the results obtained so far and maps them to the
questions and steps in `Plan.md`. All results below come from the most recent
sweep outputs in `data/sim_sweep/` using the current simulation code (plateau
implemented as observational bias; state update uses bias-free error).

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

All metrics are derived from `data/sim_sweep/summary.csv` and additional
per-run summaries using the early window (trials 1-10) and late window (last 10
trials), matching the sweep defaults.

### 3.1 Early Learning vs Modulation Strength (Plan.md 3.4, 3.5)

We summarize early learning using the slope of mean error over trials 1-10.
The baseline (M0) early slope is ~0.898. Increasing modulation strength
reduces the early slope (slower correction on average):

M1 (beta):
- beta 0.0: 0.898
- beta 0.2: 0.893
- beta 0.5: 0.878
- beta 0.8: 0.862

M2 (lambda):
- lambda 0.0: 0.898
- lambda 0.2: 0.897
- lambda 0.5: 0.890
- lambda 0.8: 0.876

Interpretation: stronger modulation lowers average early learning rate when
aggregating across mixed positive/negative delta_pi. The effect is modest
because opposing delta_pi directions partially cancel.

### 3.2 Delta Precision vs Early Error (Plan.md 2.2 proxy in simulation)

To check whether M1 and M2 produce opposite directional signatures, we computed
correlations between delta_pi and early absolute error (mean |error| over
trials 1-10) for each run:

M1 (beta), corr(delta_pi, early |error|):
- beta 0.0: 0.166
- beta 0.2: 0.589
- beta 0.5: 0.811
- beta 0.8: 0.878

M2 (lambda), corr(delta_pi, early |error|):
- lambda 0.0: 0.166
- lambda 0.2: -0.197
- lambda 0.5: -0.596
- lambda 0.8: -0.785

Interpretation:
- M1 produces a positive association: higher delta_pi increases R, slowing
  learning and inflating early |error|.
- M2 produces a negative association: higher delta_pi reduces R, increasing
  learning and shrinking early |error|.
- The nonzero baseline at strength 0 reflects collinearity between r_post1 and
  delta_pi (by design in the sweep).

This is the intended mechanistic contrast for model recovery.

### 3.3 Plateau Confound (Plan.md 3.4, 3.5)

With the plateau implemented as observational bias (b), late errors should
deviate from zero even if the state converges. The sweep confirms this:

Late mean error (trial 91-100, across models):
- plateau 0.00: about -0.146 cm (near zero)
- plateau 0.15: about +1.67 cm
- plateau effect size: +1.82 cm (0.15 * |m|, consistent across models)

Interpretation: the confound is now visible in the output, allowing us to test
whether fitting models without a plateau term can spuriously attribute the
residual error to precision modulation.

### 3.4 Collinearity Stress (Plan.md 3.4)

We compared target rho to realized corr(r_post1, delta_pi):
- target 0.0 -> realized about -0.089
- target 0.3 -> realized about 0.206
- target 0.6 -> realized about 0.528

Interpretation: the realized correlation is weaker than the target because
r_post1 is lognormal and the sample size is finite. This is acceptable for now
but should be tightened if we need strict collinearity control.

## 4. Real-Data Fitting (Plan.md Section 4)

Status: not started. The simulation phase now provides clear qualitative
signatures to validate recovery when the fitting pipeline is added.

## 5. Visualization Outputs

Quick-look plots:
- `data/sim_sweep/figures/early_slope_vs_strength.png`
- `data/sim_sweep/figures/late_error_vs_plateau.png`
- `data/sim_sweep/figures/rho_realization.png`

Interactive notebook:
- `notebooks/sweep_analysis.ipynb`

## 6. Next Steps (Aligned to Plan.md)

1) Add recovery metrics (bias/RMSE of lambda/beta) once the fitting stage
   exists (Plan.md 3.5).
2) Implement model fitting on simulated data (Plan.md 3.1, 4.2).
3) Run pre-model checks on empirical data when available (Plan.md 2.1, 2.2).
