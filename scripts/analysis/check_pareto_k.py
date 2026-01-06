"""Identify subjects with high Pareto k (influential observations)."""
import numpy as np
import pandas as pd
import arviz as az
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[0]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.fit_real_data_numpyro import *

# Load data and run a quick fit
args = parse_args()
args.models = "M1"  # Just check M1 (the winning model)
args.draws = 500
args.tune = 500
args.chains = 2

trials = pd.read_csv(args.trials_path)
subjects_full = prepare_subject_table(args.delta_path, args.metric, trials)
group_labels = sorted(subjects_full["group"].unique())

print("Running M1 to extract Pareto k diagnostics...")
print("(This will take ~30 seconds)\n")

# Run fitting
from scripts.fit_real_data_numpyro import run
result = run("M1", trials, subjects_full, args, group_labels)

# The run function doesn't return idata, so we need to modify the script
# For now, let me show you how to do it manually
print("To extract Pareto k values, you need to save the InferenceData object.")
print("\nAdd this to the run() function in fit_real_data_numpyro.py after line 210:")
print("""
    # Save InferenceData for diagnostics
    if args.save_idata:
        import pickle
        with open(args.idata_path, 'wb') as f:
            pickle.dump(idata, f)
""")

print("\nThen you can load it and extract Pareto k:")
print("""
import pickle
import arviz as az

with open('idata.pkl', 'rb') as f:
    idata = pickle.load(f)

# Compute LOO with pointwise Pareto k
loo_result = az.loo(idata, var_name='log_likelihood', pointwise=True)
pareto_k = loo_result.pareto_k.values

# Find subjects with high Pareto k
high_k_subjects = np.where(pareto_k > 0.7)[0]
print(f"Subjects with Pareto k > 0.7: {high_k_subjects}")
print(f"Their Pareto k values: {pareto_k[high_k_subjects]}")

# Create a summary
subject_diagnostics = pd.DataFrame({
    'subject_index': range(len(pareto_k)),
    'pareto_k': pareto_k
})
subject_diagnostics = subject_diagnostics.sort_values('pareto_k', ascending=False)
print(subject_diagnostics.head(10))
""")
