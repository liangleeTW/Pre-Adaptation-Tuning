# Pre-Adaptation Tuning

Code for *"Prior Sensory Tuning Orients Error Processing During Sensorimotor Adaptation."*
Implements the three Bayesian state-space models (M1 Coupling, M2 Dissociation,
M3 Dissociation + Decomposition) and the parameter / model recovery pipeline.

[Link to paper — TBD]

## Setup

Requires Python 3.12 and [Poetry](https://python-poetry.org/).

```bash
poetry config virtualenvs.in-project true
poetry env use 3.12
poetry install
```

## Reproduce paper figures

The repository ships fitted posteriors and recovery results — see
[`data/README.md`](data/README.md). The figure script runs end-to-end on
shipped data:

```bash
poetry run python scripts/analysis/fig34.py
```

Writes `fig3.png` (β_R) and `fig4.png` (β_cog) to `figures/`.

## Re-fit from raw data

Raw and trial-level CSVs are not redistributed (see [`data/README.md`](data/README.md)
for the expected schemas). With those files in place, the full pipeline is:

### State-space models (M1 / M2 / M3)

Requires `data/adaptation_trials.csv`.

```bash
# M1 (Coupling)
poetry run python scripts/modeling/fit_m1.py --plateau-group-specific --save-posterior

# M2 (Dissociation) and M3 (Dissociation + Decomposition)
poetry run python scripts/modeling/fit_m2_m3.py --plateau-group-specific --save-posterior

# Parameter recovery and model recovery → Table 2
poetry run python scripts/recovery/parameter_recovery.py
poetry run python scripts/recovery/model_recovery.py
```

### Localisation HLM (Fig. 2)

Heteroscedastic complete-pooling Bayesian HLM in Stan via `cmdstanpy`.
Requires `scripts/complete_pooling/_preprocessed/pa_long_trials.csv` and a
working CmdStan installation (`cmdstanpy.install_cmdstan()` once).

```bash
cd scripts/complete_pooling
poetry run jupyter nbconvert --execute --inplace completepooling_pipe.ipynb
poetry run jupyter nbconvert --execute --inplace completepooling_vis.ipynb
```

`completepooling_pipe.ipynb` fits the CP1–CP8 family and writes posteriors to
`./results/CP{L}/posterior.nc` plus `CP_model_comparison_loo.csv`.
`completepooling_vis.ipynb` reads CP4 / CP7 / CP8 posteriors and renders the
μ and σ panels behind Fig. 2.

## Contributors

<a href="https://github.com/sizluluEZ"><img src="https://avatars.githubusercontent.com/u/132829530?v=4" width="50px;" alt="sizluluEZ"/></a>
<a href="https://github.com/liangleeTW"><img src="https://avatars.githubusercontent.com/u/52850586?v=4" width="50px;" alt="liangleeTW"/></a>

## License

See [`LICENSE`](LICENSE).
