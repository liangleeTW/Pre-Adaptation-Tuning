# Pre-Adaptation-Tuning

Simulation and analysis code for precision transfer in prism adaptation.

## Quickstart (macOS)

```bash
brew install uv
uv python install 3.12
uv sync
uv run ./scripts/run_sim.py
uv run ./scripts/plot_sim.py
```

Key outputs:
- `data/sim/sim_trials.csv`
- `data/sim/sim_subjects.csv`
- `data/sim/figures/`

## Repository map

- `Plan.md` — research plan and modeling assumptions
- `Results.md` — current simulation findings aligned to the plan
- `SIMULATION.md` — how to run simulations and sweeps
- `src/prism_sim/` — core simulation logic (models, state-space)
- `scripts/` — CLI entry points (run_sim, run_sweep, analyze_sweep)
- `notebooks/` — interactive plotting and sweep exploration
- `data/` — generated outputs (not versioned)

This repo uses `pyenv` to install Python 3.12 and `uv` to manage the virtual
environment. Follow these steps after cloning.

## 1) Install uv (recommended)

macOS (Homebrew):

```bash
brew install uv
```

Install Python 3.12 via uv:

```bash
uv python install 3.12
```

## 2) Optional: pyenv (if you already use it)

If you prefer pyenv for global version management, you can use it instead of
`uv python install`:

```bash
brew install pyenv
pyenv install 3.12
pyenv local 3.12
```

## 3) Create and use the virtual environment

```bash
uv venv --python 3.12
source .venv/bin/activate
```

Install dependencies (via `pyproject.toml`):

```bash
uv pip install -r requirements.txt
# or
uv sync
```

## 4) Common commands

Check interpreter:

```bash
python --version
```

Deactivate:

```bash
deactivate
```

## How to run the sweep

```bash
uv run ./scripts/run_sims.py
uv run ./scripts/analyze_sweep.py
```

## How to explore interactively

```bash
uv run jupyter notebook
```

Open:
- `notebooks/sim_plots.ipynb`
- `notebooks/sweep_analysis.ipynb`

## Data and outputs

Generated outputs in `data/` are not tracked by git. To reproduce figures and
tables, run the simulations and analysis scripts listed above. If we decide to
share a curated dataset later, we can add a small `data/example/` folder.
