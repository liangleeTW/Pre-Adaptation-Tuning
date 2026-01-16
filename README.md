# Pre-Adaptation Tuning: Dissociating Learning from Variability in Sensorimotor Adaptation

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency-poetry-blue)](https://python-poetry.org/)

Analysis code for investigating how proprioceptive tuning independently modulates learning rate and execution variability in visuomotor adaptation.

**Paper Models:**
- **M1-Coupling**: Single observation noise parameter (baseline)
- **M2-Dissociation**: Separate state and observation noise (best fit)
- **M3-DD**: Decomposed observation noise into sensory + cognitive components (mechanistic interpretation)

---

## Table of Contents

1. [Setup](#setup)
2. [Data Structure](#data-structure)
3. [Analysis Pipeline](#analysis-pipeline)
4. [Verification](#verification)
5. [Troubleshooting](#troubleshooting)
6. [Contributors](#contributors)

---

## Setup

### Prerequisites

- macOS or Linux
- Python 3.12
- [pyenv](https://github.com/pyenv/pyenv) (recommended)
- [Poetry](https://python-poetry.org/) for dependency management

### Installation

```bash
# 1. Install pyenv (if not already installed)
brew install pyenv

# 2. Install Python 3.12
pyenv install 3.12.0
pyenv local 3.12.0

# 3. Install Poetry
brew install poetry

# 4. Clone repository and navigate to project directory
cd Pre-Adaptation-Tuning

# 5. Configure Poetry to create virtualenv in project
poetry config virtualenvs.in-project true

# 6. Install dependencies
poetry install --sync

# Verify installation
poetry run python --version  # Should show Python 3.12.x
```

**Key dependencies:**
- `numpyro` (Bayesian MCMC sampling)
- `jax` (Accelerated numerical computing)
- `arviz` (Bayesian diagnostics)
- `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`

---

## Data Structure

```
Pre-Adaptation-Tuning/
├── data/
│   ├── raw/
│   │   ├── adaptation_trials.csv                   # INPUT
│   │   └── proprio_delta_pi.csv                    # INPUT
│   └── derived/
│       ├── model_comparison_final.csv              # Output from Step 1
│       ├── final_model_predictions.csv             # Output from Step 2
│       ├── final_model_fit_metrics.csv             # Output from Step 2
│       ├── posteriors/                             # Output from Step 1
│       │   ├── m1_coupling_posterior.nc
│       │   ├── m2_dissociation_posterior.nc
│       │   └── m3_dd_posterior.nc
│       └── bayesian_analysis/                      # Output from Step 3
│           ├── m2_dissociation_beta_state/
│           └── m3_dd_beta_state/
├── scripts/
│   ├── modeling/
│   │   └── fit_real_data.py
│   ├── analysis/
│   │   ├── ppc_final_models.py
│   │   ├── run_bayesian_comparisons.py
│   │   ├── bayesian_beta_state_comparison.py
│   │   └── bayesian_beta_obs_comparison.py
│   └── visualization/
├── figures/
└── Results.md
```

---

## Analysis Pipeline

### Step 1: Model Fitting

**Command:**
```bash
poetry run python scripts/modeling/fit_real_data.py --plateau-group-specific
```

**Inputs:**
- `data/raw/adaptation_trials.csv`
- `data/raw/proprio_delta_pi.csv`

**Outputs:**
- `data/derived/model_comparison_final.csv`
- `data/derived/posteriors/m1_coupling_posterior.nc`
- `data/derived/posteriors/m2_dissociation_posterior.nc`
- `data/derived/posteriors/m3_dd_posterior.nc`

---

### Step 2: Posterior Predictive Checks

**Command:**
```bash
poetry run python scripts/analysis/ppc_final_models.py
```

**Inputs:**
- `data/raw/adaptation_trials.csv`
- `data/raw/proprio_delta_pi.csv`
- `data/derived/model_comparison_final.csv`

**Outputs:**
- `data/derived/final_model_predictions.csv`
- `data/derived/final_model_fit_metrics.csv`
- `figures/ppc_learning_curves.png`
- `figures/ppc_r2_comparison.png`

---

### Step 3: Bayesian Group Comparisons

**Command:**
```bash
poetry run python scripts/analysis/run_bayesian_comparisons.py
# When prompted, choose option 2 (M3-DD) or 3 (Both M2 and M3)
```

**Inputs:**
- `data/derived/posteriors/m2_dissociation_posterior.nc`
- `data/derived/posteriors/m3_dd_posterior.nc`

**Outputs (for each model and parameter):**
- `data/derived/bayesian_analysis/*/beta_state_summary.csv`
- `data/derived/bayesian_analysis/*/beta_state_pairwise.csv`
- `data/derived/bayesian_analysis/*/beta_state_orderings.csv`
- `data/derived/bayesian_analysis/*/beta_state_effect_sizes.csv`
- `data/derived/bayesian_analysis/*/beta_state_posteriors.png`
- `data/derived/bayesian_analysis/*/beta_state_rope.png`
- `data/derived/bayesian_analysis/*/bayesian_comparison_report.txt`

---

### Step 4: Publication Figures

Figures are automatically generated in Steps 2-3:
- `figures/ppc_learning_curves.png`
- `figures/ppc_r2_comparison.png`
- `data/derived/bayesian_analysis/*/beta_state_posteriors.png`
- `data/derived/bayesian_analysis/*/beta_state_rope.png`

---

## Verification

### Check Model Convergence

```bash
poetry run python -c "
import arviz as az
idata = az.from_netcdf('data/derived/posteriors/m2_dissociation_posterior.nc')
summary = az.summary(idata, var_names=['beta_state', 'beta_obs'], round_to=3)
print(summary[['mean', 'r_hat', 'ess_bulk']])
"
```

---

### Verify Model Comparison

```bash
poetry run python -c "
import pandas as pd
df = pd.read_csv('data/derived/model_comparison_final.csv')
print('\\nModel Comparison Results:')
print(df[['model', 'waic', 'loo', 'n_subjects']].to_string(index=False))
"
```

---

### Verify PPC Results

```bash
poetry run python -c "
import pandas as pd
df = pd.read_csv('data/derived/final_model_fit_metrics.csv')
print('\\nPPC Fit Quality:')
print(df[['model', 'R2', 'RMSE', 'late_residual_mean']].to_string(index=False))
"
```

---

### Verify Bayesian Comparisons

```bash
poetry run python -c "
import pandas as pd

# β_state summary
df = pd.read_csv('data/derived/bayesian_analysis/m2_dissociation_beta_state/beta_state_summary.csv')
print('\\nβ_state Summary:')
print(df[['group', 'mean', 'P(β > 0)']].to_string(index=False))

# β_state pairwise
df = pd.read_csv('data/derived/bayesian_analysis/m2_dissociation_beta_state/beta_state_pairwise.csv')
print('\\nβ_state Pairwise:')
print(df[['comparison', 'P(greater)', 'diff_mean']].to_string(index=False))
"
```

---

## Troubleshooting

### MCMC Convergence Issues

```bash
poetry run python scripts/modeling/fit_real_data.py --plateau-group-specific --draws 2000 --tune 2000
```

---

### JAX/NumPyro Errors

```bash
pip uninstall jax jaxlib
pip install --upgrade jax jaxlib
```

---

### Memory Issues

```bash
# Reduce parallelization
export XLA_FLAGS='--xla_force_host_platform_device_count=4'

# Or fit models sequentially
poetry run python scripts/modeling/fit_real_data.py --plateau-group-specific --models M1-Coupling
poetry run python scripts/modeling/fit_real_data.py --plateau-group-specific --models M2-Dissociation
poetry run python scripts/modeling/fit_real_data.py --plateau-group-specific --models M3-DD
```

---

## Contributors

<a href="https://github.com/sizluluEZ"><img src="https://avatars.githubusercontent.com/u/132829530?v=4" width="50px;" alt="sizluluEZ"/></a>
<a href="https://github.com/liangleeTW"><img src="https://avatars.githubusercontent.com/u/52850586?v=4" width="50px;" alt="liangleeTW"/></a>
