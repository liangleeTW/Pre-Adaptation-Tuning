"""
Unified model fitting for final three models.

This file consolidates the three final models used in the paper:
- M1-Coupling: Single observation noise parameter
- M2-Dissociation: Separate R_state and R_obs parameters
- M3-DD: Decomposed R_obs into sensory and cognitive components

Based on fit_real_data_numpyro.py and fit_real_data_m_obs.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

import arviz as az
import jax
import jax.numpy as jnp
from jax import random, jit, vmap, lax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

# Set JAX to use all CPU cores
import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
# Enable float64 for numerical stability
jax.config.update('jax_enable_x64', True)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ============================================================================
# Utility Functions
# ============================================================================

def require_columns(df: pd.DataFrame, cols: Iterable[str], path: Path) -> None:
    """Check that required columns exist in DataFrame."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")


def prepare_subject_table(delta_path: Path, metric: str, trials: pd.DataFrame) -> pd.DataFrame:
    """Prepare subject-level data with proprioceptive tuning metrics.

    Includes openloop and visual variance for M3-DD model if available.
    """
    delta_df = pd.read_csv(delta_path)
    require_columns(
        delta_df,
        ["ID", "group", "precision_post1", "delta_pi", "delta_log_pi"],
        delta_path,
    )
    delta_df = delta_df[delta_df["precision_post1"] > 0].copy()
    delta_df["r_post1"] = 1.0 / delta_df["precision_post1"]
    delta_col = "delta_pi" if metric == "pi" else "delta_log_pi"

    trial_subjects = trials[["subject", "group"]].drop_duplicates()
    merged = trial_subjects.merge(
        delta_df,
        left_on=["subject", "group"],
        right_on=["ID", "group"],
        how="inner",
    )
    merged = merged.dropna(subset=["r_post1", delta_col])

    # Include openloop and visual variance if available (for M3-DD)
    keep_cols = ["subject", "group", "r_post1", delta_col]
    if "openloop_var_post1" in delta_df.columns:
        keep_cols.append("openloop_var_post1")
        merged = merged.dropna(subset=["openloop_var_post1"])
    if "visual_var_post1" in delta_df.columns:
        keep_cols.append("visual_var_post1")
        merged = merged.dropna(subset=["visual_var_post1"])

    merged = merged[keep_cols].rename(columns={delta_col: "delta_pi"})
    if merged.empty:
        raise ValueError("No overlapping subjects between trials and delta file.")
    return merged


def build_error_matrix(trials: pd.DataFrame, subjects: pd.DataFrame) -> np.ndarray:
    """Build matrix of errors (subjects x trials)."""
    pivot = trials.pivot(index="subject", columns="trial", values="error")
    ordered = pivot.loc[subjects["subject"]]
    return ordered.to_numpy(dtype=float)


# ============================================================================
# Kalman Filter Implementations
# ============================================================================

@jit
def kalman_loglik(errors: jnp.ndarray, r_measure: float, m: float,
                  A: float, Q: float, b: float) -> float:
    """
    Standard Kalman filter for M1-Coupling.

    Single observation noise parameter that affects both learning and variability.
    """
    def kalman_step(carry, y):
        x, p, ll = carry
        x_pred = A * x
        p_pred = A * p * A + Q
        y_pred = -x_pred + (m + b)
        s = p_pred + r_measure
        v = y - y_pred
        ll_update = -0.5 * (jnp.log(2.0 * jnp.pi * s) + (v * v) / s)
        k = -p_pred / s
        x_new = x_pred + k * v
        p_new = (1.0 + k) * p_pred
        ll_new = ll + ll_update
        return (x_new, p_new, ll_new), None

    init_state = (0.0, 1.0, 0.0)  # (x, p, ll)
    final_state, _ = lax.scan(kalman_step, init_state, errors)
    return final_state[2]


@jit
def kalman_loglik_obs(errors: jnp.ndarray, r_state: float, r_obs: float,
                      m: float, A: float, Q: float, b: float) -> float:
    """
    Kalman filter with separated state and observation noise for M2/M3.

    Key difference: Separates state uncertainty (R_state) from observation noise (R_obs).
    - R_state affects Kalman gain (learning)
    - R_obs affects only trial-to-trial variability
    """
    def kalman_step(carry, y):
        x, p, ll = carry

        # Predict
        x_pred = A * x
        p_pred = A * p * A + Q
        y_pred = -x_pred + (m + b)

        # Innovation variance for Kalman gain (state uncertainty only)
        s_kalman = p_pred + r_state

        # Innovation variance for likelihood (state + observation noise)
        s_total = p_pred + r_state + r_obs

        # Innovation
        v = y - y_pred

        # Log-likelihood (uses total variance)
        ll_update = -0.5 * (jnp.log(2.0 * jnp.pi * s_total) + (v * v) / s_total)

        # Kalman gain (uses state uncertainty only)
        k = -p_pred / s_kalman

        # Update
        x_new = x_pred + k * v
        p_new = (1.0 + k) * p_pred
        ll_new = ll + ll_update

        return (x_new, p_new, ll_new), None

    init_state = (0.0, 1.0, 0.0)  # (x, p, ll)
    final_state, _ = lax.scan(kalman_step, init_state, errors)
    return final_state[2]


# ============================================================================
# Model Definitions
# ============================================================================

def model_numpyro(model_name: str, errors: jnp.ndarray, group_idx: jnp.ndarray,
                  n_groups: int, r_post1: jnp.ndarray, delta_pi: jnp.ndarray,
                  m: float, A: float, Q: float, plateau_group_specific: bool,
                  r_openloop: jnp.ndarray = None, r_visual: jnp.ndarray = None):
    """
    Unified model for all three final models.

    Models:
    - M1-Coupling: S = R = R₀ * exp(β * Δlog π)
    - M2-Dissociation: S = R + V, both modulated by Δlog π
    - M3-DD: S = R + (V_motor + V_visual + V_cog * exp(β * Δlog π))
    """
    # Plateau parameter
    if plateau_group_specific:
        b = numpyro.sample("b", dist.Normal(0.0, 30.0).expand((n_groups,)))
        b_subj = b[group_idx]
    else:
        b = numpyro.sample("b", dist.Normal(0.0, 30.0))
        b_subj = jnp.repeat(b, len(group_idx))

    model_name = model_name.upper()

    # ========================================================================
    # M1-Coupling: Single observation noise parameter
    # ========================================================================
    if model_name in {"M1-COUPLING", "M1COUPLING", "M1"}:
        beta = numpyro.sample("beta", dist.Normal(0.0, 1.0).expand((n_groups,)))
        r_measure = r_post1 * jnp.exp(beta[group_idx] * delta_pi)

        # Use standard Kalman filter
        vectorized_kalman = vmap(kalman_loglik, in_axes=(0, 0, None, None, None, 0))
        logliks = vectorized_kalman(errors, r_measure, m, A, Q, b_subj)

    # ========================================================================
    # M2-Dissociation: Separate R_state and R_obs, both modulated
    # ========================================================================
    elif model_name in {"M2-DISSOCIATION", "M2DISSOCIATION", "M2"}:
        beta_state = numpyro.sample("beta_state", dist.Normal(0.0, 1.0).expand((n_groups,)))
        beta_obs = numpyro.sample("beta_obs", dist.Normal(0.0, 1.0).expand((n_groups,)))

        # R_state modulated (affects learning)
        r_state = r_post1 * jnp.exp(beta_state[group_idx] * delta_pi)

        # R_obs modulated (affects variability)
        r_obs_base = numpyro.sample("r_obs_base", dist.HalfNormal(2.0).expand((n_groups,)))
        r_obs_subj = r_obs_base[group_idx] * jnp.exp(beta_obs[group_idx] * delta_pi)

        # Use separated Kalman filter
        vectorized_kalman = vmap(kalman_loglik_obs, in_axes=(0, 0, 0, None, None, None, 0))
        logliks = vectorized_kalman(errors, r_state, r_obs_subj, m, A, Q, b_subj)

    # ========================================================================
    # M3-DD: Decomposed R_obs into sensory (fixed) + cognitive (modulated)
    # ========================================================================
    elif model_name in {"M3-DD", "M3DD", "M3"}:
        if r_openloop is None or r_visual is None:
            raise ValueError("M3-DD requires openloop_var_post1 and visual_var_post1 data")

        beta_state = numpyro.sample("beta_state", dist.Normal(0.0, 1.0).expand((n_groups,)))
        beta_obs = numpyro.sample("beta_obs", dist.Normal(0.0, 1.0).expand((n_groups,)))

        # R_state modulated (affects learning)
        r_state = r_post1 * jnp.exp(beta_state[group_idx] * delta_pi)

        # R_obs decomposed:
        # - Sensory components (fixed): motor execution + visual encoding
        r_sensory = r_openloop + r_visual

        # - Cognitive component (modulated): attention, strategy, etc.
        r_cognitive = numpyro.sample("r_cognitive", dist.HalfNormal(2.0).expand((n_groups,)))
        r_cognitive_modulated = r_cognitive[group_idx] * jnp.exp(beta_obs[group_idx] * delta_pi)

        # Total observation noise
        r_obs_subj = r_sensory + r_cognitive_modulated

        # Use separated Kalman filter
        vectorized_kalman = vmap(kalman_loglik_obs, in_axes=(0, 0, 0, None, None, None, 0))
        logliks = vectorized_kalman(errors, r_state, r_obs_subj, m, A, Q, b_subj)

    else:
        raise ValueError(f"Unknown model: {model_name}. "
                        f"Valid models: M1-Coupling, M2-Dissociation, M3-DD")

    # Add to model
    numpyro.factor("obs", jnp.sum(logliks))
    numpyro.deterministic("log_likelihood", logliks)


# ============================================================================
# Fitting and Analysis
# ============================================================================

def run(model_name: str, trials: pd.DataFrame, subjects: pd.DataFrame,
        args: argparse.Namespace, group_labels: list[str]) -> tuple[dict, az.InferenceData]:
    """
    Run MCMC fitting for a single model.

    Returns:
        results: Dictionary with WAIC, LOO, and other metrics
        idata: ArviZ InferenceData object with posterior samples
    """
    print(f"\n{'='*70}")
    print(f"Fitting {model_name}")
    print(f"{'='*70}")

    # Prepare data
    errors = build_error_matrix(trials, subjects)
    n_subj, n_trials = errors.shape
    print(f"Data: {n_subj} subjects × {n_trials} trials")

    # Group indices
    group_map = {g: i for i, g in enumerate(group_labels)}
    group_idx = jnp.array([group_map[g] for g in subjects["group"]])
    n_groups = len(group_labels)

    # Subject-level parameters
    r_post1 = jnp.array(subjects["r_post1"].values)
    delta_pi = jnp.array(subjects["delta_pi"].values)

    # Get openloop and visual variance if available (for M3-DD)
    r_openloop = None
    r_visual = None
    if "openloop_var_post1" in subjects.columns:
        r_openloop = jnp.array(subjects["openloop_var_post1"].values)
    if "visual_var_post1" in subjects.columns:
        r_visual = jnp.array(subjects["visual_var_post1"].values)

    print(f"Groups: {group_labels} (n={n_groups})")
    print(f"Delta metric: {args.metric}")
    print(f"Prism displacement m = {args.m}°")

    # Run MCMC
    print(f"\nRunning MCMC: {args.draws} draws × {args.chains} chains "
          f"(+ {args.tune} warmup)")

    rng_key = random.PRNGKey(args.random_seed)
    kernel = NUTS(model_numpyro, target_accept_prob=args.target_accept)
    mcmc = MCMC(
        kernel,
        num_warmup=args.tune,
        num_samples=args.draws,
        num_chains=args.chains,
        progress_bar=True,
    )

    mcmc.run(
        rng_key,
        model_name=model_name,
        errors=errors,
        group_idx=group_idx,
        n_groups=n_groups,
        r_post1=r_post1,
        delta_pi=delta_pi,
        m=args.m,
        A=args.A,
        Q=args.Q,
        plateau_group_specific=args.plateau_group_specific,
        r_openloop=r_openloop,
        r_visual=r_visual,
    )

    # Get samples
    samples = mcmc.get_samples()
    print("\nConvergence diagnostics:")
    mcmc.print_summary()

    # Convert to ArviZ
    idata = az.from_numpyro(mcmc)

    # Compute WAIC and LOO
    print("\nComputing WAIC and LOO...")
    waic_result = az.waic(idata, scale="deviance")
    loo_result = az.loo(idata, scale="deviance")

    waic = float(waic_result.elpd_waic * -2)
    loo = float(loo_result.elpd_loo * -2)

    print(f"WAIC = {waic:.1f}")
    print(f"LOO = {loo:.1f}")

    # Extract parameter estimates
    results = {
        "model": model_name,
        "waic": waic,
        "loo": loo,
        "n_subjects": n_subj,
        "n_trials": n_trials,
        "n_groups": n_groups,
    }

    # Add group-specific parameters
    for group in group_labels:
        g_idx = group_map[group]

        if "beta" in samples:
            results[f"beta_{group}"] = float(samples["beta"][..., g_idx].mean())
        if "beta_state" in samples:
            results[f"beta_state_{group}"] = float(samples["beta_state"][..., g_idx].mean())
        if "beta_obs" in samples:
            results[f"beta_obs_{group}"] = float(samples["beta_obs"][..., g_idx].mean())
        if "r_obs_base" in samples:
            results[f"r_obs_base_{group}"] = float(samples["r_obs_base"][..., g_idx].mean())
        if "r_cognitive" in samples:
            results[f"r_cognitive_{group}"] = float(samples["r_cognitive"][..., g_idx].mean())
        if "b" in samples and args.plateau_group_specific:
            results[f"b_{group}"] = float(samples["b"][..., g_idx].mean())

    if "b" in samples and not args.plateau_group_specific:
        results["b"] = float(samples["b"].mean())

    return results, idata


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Fit final three models: M1-Coupling, M2-Dissociation, M3-DD"
    )

    # Data paths
    p.add_argument("--trials-path", type=Path,
                  default=Path("data/raw/adaptation_trials.csv"))
    p.add_argument("--delta-path", type=Path,
                  default=Path("data/raw/proprio_delta_pi.csv"))

    # Model selection
    p.add_argument("--models", type=str,
                  default="M1-Coupling,M2-Dissociation,M3-DD",
                  help="Comma-separated list of models to fit")
    p.add_argument("--metric", choices=["pi", "logpi"], default="logpi",
                  help="Use delta_pi or delta_log_pi")

    # MCMC settings
    p.add_argument("--draws", type=int, default=1000,
                  help="Posterior samples per chain")
    p.add_argument("--tune", type=int, default=1000,
                  help="Warmup iterations")
    p.add_argument("--chains", type=int, default=4,
                  help="Number of parallel chains")
    p.add_argument("--target-accept", type=float, default=0.85,
                  help="NUTS target acceptance probability")
    p.add_argument("--random-seed", type=int, default=0)

    # Model parameters
    p.add_argument("--m", type=float, default=-12.1,
                  help="Prism displacement (degrees)")
    p.add_argument("--A", type=float, default=1.0,
                  help="State transition matrix")
    p.add_argument("--Q", type=float, default=1e-4,
                  help="Process noise")
    p.add_argument("--plateau-group-specific", action="store_true",
                  help="Estimate separate plateau (b) for each group")

    # Output paths
    p.add_argument("--out-path", type=Path,
                  default=Path("data/derived/model_comparison_final.csv"))
    p.add_argument("--posterior-dir", type=Path,
                  default=Path("data/derived/posteriors"))

    args = p.parse_args()

    # Load data
    print("Loading data...")
    trials = pd.read_csv(args.trials_path)
    require_columns(trials, ["subject", "group", "trial", "error"], args.trials_path)

    subjects = prepare_subject_table(args.delta_path, args.metric, trials)
    group_labels = sorted(subjects["group"].unique())

    print(f"Loaded {len(subjects)} subjects in {len(group_labels)} groups: {group_labels}")

    # Parse models to fit
    models = [m.strip() for m in args.models.split(",")]
    print(f"\nFitting {len(models)} models: {', '.join(models)}")

    # Fit each model
    all_results = []
    for model_name in models:
        try:
            results, idata = run(model_name, trials, subjects, args, group_labels)
            all_results.append(results)

            # Save posterior
            if args.posterior_dir:
                args.posterior_dir.mkdir(parents=True, exist_ok=True)
                nc_path = args.posterior_dir / f"{model_name.lower().replace('-', '_')}_posterior.nc"
                idata.to_netcdf(nc_path)
                print(f"Saved posterior: {nc_path}")

        except Exception as e:
            print(f"\n❌ Error fitting {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        args.out_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(args.out_path, index=False)
        print(f"\n{'='*70}")
        print(f"Saved results to: {args.out_path}")
        print(f"{'='*70}\n")
        print(results_df.to_string(index=False))
    else:
        print("\n⚠️  No models successfully fit.")


if __name__ == "__main__":
    main()
