# Pre-Adaptation-Tuning

Analysis code for proprioceptive tuning effects on visuomotor adaptation.

---

## Setup

### 1. Install Python 3.12 with pyenv

```bash
# Install pyenv (if not already installed)
brew install pyenv

# Install Python 3.12
pyenv install 3.12

# Set local version for this project
pyenv local 3.12
```

### 2. Install Poetry

```bash
# Install Poetry via Homebrew
brew install poetry
```

### 3. Install Dependencies

```bash
# Activate virtual environment
poetry init --python "^3.12" -q  # skip this if poetry.lock already exists
poetry env use $(pyenv which python)
```

---

## Run Main Analysis

### 1. Extract Data from Raw Files

```bash
poetry run python scripts/data_processing/extract_adaptation.py
poetry run python scripts/data_processing/extract_prepost.py
```

### 2. Fit Models to Real Data

**Fit baseline model (M1-exp):**
```bash
poetry run python scripts/modeling/fit_real_data_numpyro.py
```

**Fit primary model (M2-dual):**
```bash
poetry run python scripts/modeling/fit_real_data_m_obs.py --plateau-group-specific
```

This produces:
- `data/derived/m_obs_results.csv` - Parameter estimates (β_state, β_obs)
- `data/derived/m_obs_predictions.csv` - Trial-by-trial predictions
- `data/derived/m_obs_fit_metrics.csv` - Fit quality metrics

### 3. Validate Model Results

```bash
poetry run python scripts/analysis/validate_m_obs_mechanisms.py
```

This generates:
- `figures/main_comparison_m1exp_vs_mtwoR.png` - Main 8-panel figure
- `figures/state_slopes_vs_deltalogpi.png` - Mechanistic validation
- `figures/group_comparisons.png` - Group-level effects
- `data/derived/state_based_slopes.csv` - State trajectory slopes

---

## Generate Figures

**All visualizations:**
```bash
poetry run python scripts/visualization/plot_delta_pi_models.py
poetry run python scripts/visualization/plot_delta_pi_early_learning.py
poetry run python scripts/visualization/plot_delta_vs_gain.py
```

Figures are saved to `figures/`.

---

## Key Output Files

After running the pipeline, find results in:
- `data/derived/m_obs_results.csv` - Model parameter estimates
- `data/derived/state_based_slopes.csv` - Learning slope validation
- `figures/main_comparison_m1exp_vs_mtwoR.png` - Main figure
- `VALIDATION_SUMMARY.md` - Complete validation report

---

## Documentation


---

## Contributors

<a href="https://github.com/sizluluEZ"><img src="https://avatars.githubusercontent.com/u/132829530?v=4" width="50px;" alt="sizluluEZ"/></a>
<a href="https://github.com/liangleeTW"><img src="https://avatars.githubusercontent.com/u/52850586?v=4" width="50px;" alt="liangleeTW"/></a>


## Citation


## License

See `LICENSE` file.
