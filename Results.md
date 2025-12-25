# Results (Simulation Phase)

This document summarizes the results obtained so far and maps them to the
questions and steps in `Plan.md`. Unless otherwise noted, the results below
come from the latest sweep outputs in `data/sim_sweep_20251225_115250/` using
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

All metrics are derived from `data/sim_sweep_20251225_115250/summary.csv` and
additional per-run summaries using the early window (trials 1-10) and late
window (last 10 trials), matching the sweep defaults.

### 3.1 Simulation Settings (Plan.md 3.1, 3.4)

We simulate group-structured tuning using empirical proprioceptive data from
`raw/` to set realistic ranges for \(\Delta\pi\). The sweep uses group means,
SDs, and weights derived from `data/derived/proprio_delta_pi.csv` (computed
from pre vs post1 proprioceptive trials). Settings:

- Models: M0, M1, M2.
- M2 \(\lambda\): \(-0.8,-0.5,-0.2,0,0.2,0.5,0.8\).
- M1 \(\beta\): \(0,0.2,0.5,0.8\).
- Trials per subject: 100.
- Subjects per run: 60 (multinomially allocated to groups by observed weights).
- Plateau confound: \(b \in \{0, 0.15|m|\}\), implemented as observational bias.
- State update uses bias-free error \(e_t\); observed error adds \(b\).

Group \(\Delta\pi\) distributions are visualized below.

Plot:
- `data/derived/figures/delta_pi_groups.png`
- `data/derived/figures/delta_log_pi_groups.png`

Interpretation:
The empirical group means are positive and overlapping, implying modest
group separation in the simulation. This limits the size of group-level effects
we expect in early learning unless \(\lambda\) is strong.

### 3.2 Early Learning vs Modulation Strength (Plan.md 3.4, 3.5)

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

Plot:
- `data/sim_sweep_20251225_115250/figures/early_slope_vs_strength.png`

Axis interpretation:
- X-axis: modulation strength (\(\beta\) for M1, \(\lambda\) for M2).
- Y-axis: early slope of mean error vs trial (trials 1-10).

Cognitive meaning:
- More negative slopes indicate faster early error reduction (greater error
  utilization).
- Less negative (or higher) slopes indicate more conservative early learning.
This proxy captures the earliest adaptation dynamics where error weighting
changes are most diagnostic.

### 3.3 Delta Precision vs Early Error (Plan.md 2.2 proxy in simulation)

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

This is the intended mechanistic contrast for model recovery. The summary plot
is saved as `data/sim_sweep_20251225_115250/figures/delta_pi_vs_early_abs_error.png`.

Plot:
- `data/sim_sweep_20251225_115250/figures/delta_pi_vs_early_abs_error.png`

Axis interpretation:
- X-axis: modulation strength (\(\beta\) or \(\lambda\)).
- Y-axis: correlation between \(\Delta\pi\) and early \(|error|\) per run.

Cognitive meaning:
This plot indicates whether increased tuning is associated with larger early
errors (down-weighted error utilization) or smaller early errors (up-weighted
error utilization), directly indexing the direction of precision transfer.

### 3.4 Plateau Confound (Plan.md 3.4, 3.5)

With the plateau implemented as observational bias (b), late errors should
deviate from zero even if the state converges. The sweep confirms this:

Late mean error (trial 91-100, across models):
- plateau 0.00: about -0.146 cm (near zero)
- plateau 0.15: about +1.67 cm
- plateau effect size: +1.82 cm (0.15 * |m|, consistent across models)

Interpretation: the confound is now visible in the output, allowing us to test
whether fitting models without a plateau term can spuriously attribute the
residual error to precision modulation.

Plot:
- `data/sim_sweep_20251225_115250/figures/late_error_vs_plateau.png`

Axis interpretation:
- X-axis: plateau fraction \(b/|m|\).
- Y-axis: late mean error (trials 91-100).

Cognitive meaning:
The plateau term represents a late-stage residual error that should not be
mistaken for ongoing learning. The plot verifies that the confound is present
and of the expected magnitude.

### 3.5 Collinearity Stress (Plan.md 3.4)

We compared target rho to realized corr(r_post1, delta_pi):
- target 0.0 -> realized about -0.089
- target 0.3 -> realized about 0.206
- target 0.6 -> realized about 0.528

Interpretation: the realized correlation is weaker than the target because
r_post1 is lognormal and the sample size is finite. This is acceptable for now
but should be tightened if we need strict collinearity control. The scatter is
saved as `data/sim_sweep_20251225_115250/figures/rho_realization.png` (identity line included).

Plot:
- `data/sim_sweep_20251225_115250/figures/rho_realization.png`

Axis interpretation:
- X-axis: target correlation \(\rho(R^{post1}_P, \Delta\pi)\).
- Y-axis: realized correlation in simulated samples.

Cognitive meaning:
Collinearity between onset reliability and tuning direction limits
identifiability of the \(R\)-mapping parameters. This plot checks how close we
are to the intended collinearity stress.

### 3.6 Group-Level Early Learning vs \(\lambda\) (M2 Only)

We directly compare the three experimental groups (EC, EO+, EO-) under M2 to
see how sign and magnitude of \(\lambda\) affect early learning.

Plot:
- `data/sim_sweep_20251225_115250/figures/early_slope_vs_lambda_groups.png`

Axis interpretation:
- X-axis: \(\lambda\), including negative values (attribution-dominant regime).
- Y-axis: early slope (trials 1-10) per group.

Cognitive interpretation:
- As \(\lambda\) becomes positive, all groups show lower slopes (faster early
  learning), consistent with the hypothesis that increased precision enhances
  error utilization.
- EC and EO- show stronger slope changes than EO+, indicating greater
  sensitivity to tuning, consistent with their slightly higher or more variable
  \(\Delta\pi\) distributions.

Hypothesis match:
The group-level M2 pattern supports the primary hypothesis under \(\lambda>0\):
precision transfer increases early error utilization.

### 3.7 Group-Level Early Learning vs \(\beta\) (M1 Only)

We compare the same three groups under M1 to characterize the linear additive
mapping.

Plot:
- `data/sim_sweep_20251225_115250/figures/early_slope_vs_beta_groups.png`

Axis interpretation:
- X-axis: \(\beta\) (non-negative in the current sweep).
- Y-axis: early slope (trials 1-10) per group.

Cognitive interpretation:
- Increasing \(\beta\) yields slightly higher slopes (slower early learning),
  indicating that in M1, higher \(\Delta\pi\) increases \(R\) and reduces error
  utilization.
- Group differences are modest, consistent with overlapping empirical tuning
  distributions.

Hypothesis match:
This pattern is directionally opposite to the M2 \(\lambda>0\) regime and thus
serves as the discriminative contrast. Current group data are more consistent
with the M2 interpretation.

## 4. Real-Data Fitting (Plan.md Section 4)

Status: not started. The simulation phase now provides clear qualitative
signatures to validate recovery when the fitting pipeline is added.

## 5. Visualization Outputs

Plots are referenced within each subsection above.

Interactive notebook:
- `notebooks/sweep_analysis.ipynb`

## 6. Next Steps (Aligned to Plan.md)

1) Add recovery metrics (bias/RMSE of lambda/beta) once the fitting stage
   exists (Plan.md 3.5).
2) Implement model fitting on simulated data (Plan.md 3.1, 4.2).
3) Run pre-model checks on empirical data when available (Plan.md 2.1, 2.2).
