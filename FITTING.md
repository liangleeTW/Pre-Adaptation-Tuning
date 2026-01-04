# Fitting Quickstart

This doc covers the real-data fitting phase (see Plan.md §4 for rationale and hypotheses about \(\lambda\)).

## Install deps

```bash
uv sync
```

## Data inputs

Derived files already in `data/derived/`:
- `adaptation_trials.csv` — long-format adaptation errors (columns: subject, group, trial, error).
- `proprio_delta_pi.csv` — pre/post1 proprioceptive precision and \(\Delta\pi\), \(\Delta\log\pi\).

If you need to regenerate from raw:
```bash
uv run scripts/extract_adaptation.py
uv run scripts/extract_prepost.py
```

## Fit via maximum likelihood (group-specific, plateau on)

Models: M0, M1, M2; group-specific modulation; shared plateau \(b\) is mandatory.

```bash
uv run python scripts/fit_real_data.py --models M0,M1,M2 --metric pi --out-path data/derived/real_fit_results.csv
```

Outputs:
- `data/derived/real_fit_results.csv` (NLL, AIC/BIC, group-specific \(\beta\)/\(\lambda\), plateau \(b\)).

Interpretation: use AIC/BIC for model comparison; inspect signs/magnitudes of \(\lambda\) (M2) or \(\beta\) (M1) per group to adjudicate the \(\lambda>0\) vs \(\lambda<0\) hypotheses (Plan.md §1.2, §4).

## Fit via PyMC (Bayesian, group-specific, plateau on)

Models: M0, M1, M2; group-specific modulation; shared plateau \(b\).

```bash
uv run python scripts/fit_real_data_pymc.py \
  --models M0,M1,M2 \
  --draws 2000 --tune 2000 --chains 4 --target-accept 0.9 \
  --out-path data/derived/real_fit_pymc_plateau.csv
```

Outputs:
- `data/derived/real_fit_pymc_plateau.csv` (WAIC/LOO, posterior medians of \(\beta\)/\(\lambda\), plateau \(b\)).

Recommended checks:
- Model evidence: WAIC/LOO (higher elpd / lower deviance is better).
- \(\Pr(\lambda_g>0)\) per group for M2 to test reliability vs source-estimation routes.
- Early-trial predictive ordering by group to see if it matches the \(\lambda>0\) (warm) vs \(\lambda<0\) (cool) schematic in Plan.md §1.2.

To focus only on \(\lambda\):
```bash
uv run python scripts/fit_real_data_pymc.py --models M2 --draws 2000 --tune 2000 --chains 4 --target-accept 0.9 --out-path data/derived/real_fit_pymc_M2.csv
```

To use \(\Delta\log\pi\) instead of \(\Delta\pi\):
```bash
... --metric logpi
```

## Fit via NumPyro (JAX backend, faster compile)

Models: M0, M1, M2; group-specific modulation; plateau on. Uses vectorized Kalman likelihood in JAX.

Full run (shared plateau):
```bash
poetry run python scripts/fit_real_data_numpyro.py \
  --models M0,M1,M2 \
  --draws 2000 --tune 2000 --chains 4 --target-accept 0.9 \
  --out-path data/derived/real_fit_numpyro_optimized.csv
```

Options:
- `--plateau-group-specific` for group-specific \(b\).
- `--metric logpi` to use \(\Delta\log\pi\).
- `--max-subjects N` for quick smoke tests.

Outputs:
- `data/derived/real_fit_numpyro_optimized.csv` (WAIC/LOO, posterior medians of \(\beta\)/\(\lambda\), \(b\), \(\Pr(\lambda>0)\)).

Notes:
- JAX runs on CPU by default; chains parallelize across available cores. Initial JIT can take time; afterward sampling is fast.
- If Pareto \(k\) warnings appear, increase draws/tune or raise `--target-accept` (e.g., 0.95).

## What to report (cognitive-facing)

- Model evidence: WAIC/LOO (Bayes) and AIC/BIC (MLE).
- Parameter posteriors: medians + 95% CI for \(\lambda_g\) (M2) or \(\beta_g\) (M1); \(\Pr(\lambda_g>0)\).
- Plateau \(b\): posterior median to show residual asymptote; shared across groups in the current code.
- Group ordering: posterior predictive early gains per group vs the sign-specific predictions (Plan.md §1.2).

## Visual checks (not yet scripted)

After fitting, add:
- Posterior predictive trajectories vs observed trials by group.
- Early-gain scatter vs \(\Delta\pi\) per group.
- Residuals/late-plateau overlays to validate the fitted \(b\).
