# Data

This folder ships fitted artifacts only. Raw and trial-level participant data are not redistributed.

## Files

| Path | Description |
|---|---|
| `proprio_delta_pi.csv` | Per-subject (n = 72) summary: pre/post proprioceptive precision, Δlogπ, and post-tuning open-loop and visual variances. Required by all model and recovery scripts. |
| `m2_m3_results.csv` | Fit summary for M2 and M3: WAIC, LOO, p_loo, max R̂, ESS, and per-group posterior medians of β_R, β_cog, V_0, V_cog. Backs Table 1 (M2 / M3 rows) and the parameter values reported in the Results. |
| `posteriors/m2_posterior.nc` | M2 (Dissociation) posterior — ArviZ NetCDF. |
| `posteriors/m3_posterior.nc` | M3 (Dissociation + Decomposition) posterior. Read by `scripts/analysis/fig34.py`. |
| `recovery/param_recovery_m3.csv` | Generating-vs-recovered parameters for M3 → Table 2. |
| `recovery/param_recovery_m2onm3.csv` | Same, for M2 fit to M3-generated data → Table 2. |
| `recovery/model_recovery_results.csv` | Per-simulation WAIC / LOO outcome of M2-vs-M3 model recovery. |

The M1 (Coupling) posterior is not shipped — running `scripts/modeling/fit_m1.py --save-posterior` writes it to `posteriors/m1_posterior.nc`.

## Input schemas (not included)

### Long-format trial table — `_preprocessed/pa_long_trials.csv`

Required by `scripts/complete_pooling/completepooling_pipe.ipynb`. One row per trial.

| Column | Meaning |
|---|---|
| `ID` | Subject identifier (e.g. `EC_002`) |
| `subject_id` | Integer subject index |
| `session` | `pre`, `post1`, … (categorical; first level is the model baseline) |
| `modality` | `visual`, `proprioceptive`, or `openloop` |
| `group` | `EC`, `EO-`, or `EO+` |
| `trial` | Trial index within the (subject × session × modality) cell |
| `y` | Signed horizontal localisation error in cm (rightward > 0) |

### Raw per-context CSVs

The data-processing pipeline expects six CSV files — two per visual context (`EC`, `EO+`, `EO-`):

#### `<ctx>_main.csv` — arm-reaching task

| Column | Meaning |
|---|---|
| *(unnamed index)* | Row index |
| `ID` | Subject identifier (e.g. `EC_002`) |
| `1` … `150` | Signed horizontal endpoint error in cm (rightward > 0) for each of 150 trials = 50 pre-adaptation tuning trials + 100 prism-adaptation trials |

#### `<ctx>_prepost.csv` — localisation tests

| Column | Meaning |
|---|---|
| *(unnamed index)* | Row index |
| `ID` | Subject identifier |
| `session` | `pre` or `post` |
| `modality` | `visual` (V), `proprioceptive` (P), or `openloop` (VP) |
| `1` … `40` | Signed horizontal localisation error in cm (8 targets × 5 cycles) |

Each subject contributes one row to `<ctx>_main.csv` and six rows to `<ctx>_prepost.csv` (3 modalities × 2 sessions).
