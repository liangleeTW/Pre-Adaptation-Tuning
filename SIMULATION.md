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
