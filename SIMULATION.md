# Simulation Quickstart

This doc covers the minimal simulation workflow (no Bayesian fitting yet).

## Install deps

```bash
uv sync
```

## Run a baseline simulation

```bash
uv run ./scripts/run_sim.py --model M0
```

Outputs:
- `data/sim/sim_trials.csv`
- `data/sim/sim_subjects.csv`

Note: `sim_trials.csv` includes `error` (observed, with plateau bias) and
`error_true` (bias-free) to isolate plateau confounds.

## Make plots

```bash
uv run ./scripts/plot_sim.py
```

Outputs:
- `data/sim/figures/mean_error.png`
- `data/sim/figures/trajectories.png`
- `data/sim/figures/precision_mapping.png`

## Jupyter notebook (interactive)

Install Jupyter if needed:

```bash
uv add --dev jupyter
```

Launch:

```bash
uv run jupyter notebook
```

Notebook:
- `notebooks/sim_plots.ipynb`

## Parameter sweep (default grid)

Defaults are chosen to match the Plan.md simulation section:
- Models: M0, M1, M2
- M2 lambda: -0.8, -0.5, -0.2, 0, 0.2, 0.5, 0.8
- M1 beta: 0, 0.2, 0.5, 0.8
- Delta precision SD: 0.5, 1.0, 1.6
- Collinearity rho: 0, 0.3, 0.6
- Plateau fraction: 0, 0.15
- Seeds: 0, 1

Run the full sweep:

```bash
uv run ./scripts/run_sweep.py
```

Alias:

```bash
uv run ./scripts/run_sims.py
```

Outputs:
- `data/sim_sweep/index.csv`
- `data/sim_sweep/run_####_*/sim_trials.csv`
- `data/sim_sweep/run_####_*/sim_subjects.csv`

Customize ranges with flags, for example:

```bash
uv run ./scripts/run_sweep.py --models M2 --lams 0,0.5 --delta-pi-sds 1.0 --seeds 0
```

## Group-structured tuning (EO+/EO-/EC)

You can impose group-specific \(\Delta\pi\) distributions to mirror the tuning phase:

```bash
uv run ./scripts/run_sweep.py \
  --group-labels EO+,EO-,EC \
  --group-delta-pi-means 0.3,0.0,-0.3 \
  --group-delta-pi-sds 0.6,0.6,0.6 \
  --group-weights 0.33,0.33,0.34
```

`--delta-pi-sds` acts as a global scale multiplier on the group SDs, allowing
additional sweeps over tuning dispersion.

Use `--delta-pi-metric logpi` to interpret the tuning variable as Δlog precision
instead of Δprecision. This only changes interpretation; the simulation uses the
provided values directly.

## Group-specific modulation grids (lambda/beta)

You can sweep multiple *group-specific* \(\lambda\) (M2) or \(\beta\) (M1) sets.
This is useful for testing whether groups follow different policy regimes.

The grid format is a semicolon-separated list of group-specific parameter sets
in the same order as `--group-labels`.

Example (M2, group-specific lambda grid):

```bash
uv run ./scripts/run_sweep.py \
  --models M2 \
  --lams 0.0 \
  --group-labels EC,EO+,EO- \
  --group-delta-pi-means 0.198,0.096,0.062 \
  --group-delta-pi-sds 0.377,0.237,0.248 \
  --group-weights 0.362,0.333,0.304 \
  --group-lams-grid "0.8,0.0,-0.8;0.6,0.0,-0.6;0.4,0.0,-0.4;0.2,0.0,-0.2" \
  --delta-pi-sds 1.5 \
  --rhos 0.0 \
  --plateau-fracs 0.0 \
  --n-seeds 10 \
  --outdir data/sim_sweep_group_lam_grid
```

Example (M1, group-specific beta grid):

```bash
uv run ./scripts/run_sweep.py \
  --models M1 \
  --betas 0.0 \
  --group-labels EC,EO+,EO- \
  --group-delta-pi-means 0.198,0.096,0.062 \
  --group-delta-pi-sds 0.377,0.237,0.248 \
  --group-weights 0.362,0.333,0.304 \
  --group-betas-grid "0.5,0.0,-0.5;0.3,0.0,-0.3;0.2,0.0,-0.2" \
  --delta-pi-sds 1.5 \
  --rhos 0.0 \
  --plateau-fracs 0.0 \
  --n-seeds 10 \
  --outdir data/sim_sweep_group_beta_grid
```

Notes:
- `--lams 0.0` / `--betas 0.0` keeps the global grid at a single value while the
  group-specific grid does the sweep.
- `--n-seeds N` auto-generates seeds `0..N-1`.

## One-command sweep from data

If you have run `scripts/plot_delta_pi_groups.py`, you can launch a sweep using
the derived group parameters:

```bash
uv run python scripts/run_sweep_from_data.py --metric pi
```

Add any extra sweep args at the end, e.g.:

```bash
uv run python scripts/run_sweep_from_data.py --metric pi --lams -0.5,0,0.5 --seeds 0
```

By default, this command will auto-load calibration values from
`data/derived/adaptation_calibration.csv` and pass:
- `--m` set to the mean early adaptation error
- `--plateau-bs` set to `0.0` and the mean late adaptation error

Disable calibration with:

```bash
uv run python scripts/run_sweep_from_data.py --metric pi --no-calibration
```

## Group-specific modulation (beta/lambda)

You can assign different \(\beta\) or \(\lambda\) per group at generation time:

```bash
uv run ./scripts/run_sweep.py \
  --group-labels EC,EO+,EO- \
  --group-delta-pi-means 0.198,0.096,0.062 \
  --group-delta-pi-sds 0.377,0.237,0.248 \
  --group-betas 0.2,0.1,0.0
```

For M2, use `--group-lams` instead of `--group-betas`.

## Parameter recovery (simulation fitting)

Fit each run to recover \(\beta\) or \(\lambda\) from simulated data:

```bash
uv run python scripts/fit_sim_recovery.py --sweep-dir data/sim_sweep_20251225_115250
```

Summarize recovery metrics (bias/RMSE/sign match):

```bash
uv run python scripts/analyze_recovery.py --sweep-dir data/sim_sweep_20251225_115250
```

## Sweep summary + plots

After the sweep finishes:

```bash
uv run ./scripts/analyze_sweep.py
```

Outputs:
- `data/sim_sweep/summary.csv`
- `data/sim_sweep/figures/early_slope_vs_strength.png`
- `data/sim_sweep/figures/late_error_vs_plateau.png`
- `data/sim_sweep/figures/rho_realization.png`

Notebook:
- `notebooks/sweep_analysis.ipynb`

## Example sweeps

Linear modulation:

```bash
PYTHONPATH=src python scripts/run_sim.py --model M1 --beta 0.2 --delta-pi-sd 0.8
```

Bounded modulation:

```bash
PYTHONPATH=src python scripts/run_sim.py --model M2 --lam 0.5 --delta-pi-sd 1.5
```

Plateau confound:

```bash
PYTHONPATH=src python scripts/run_sim.py --plateau-frac 0.15
```
