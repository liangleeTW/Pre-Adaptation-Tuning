"""
Fit M-obs model: Kalman filter with explicit observation noise.

Key difference from current models:
- Separates state uncertainty (P) from observation noise (R_obs)
- Allows fitting both early learning AND late-trial oscillations
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

# Set JAX config
import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
jax.config.update('jax_enable_x64', True)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def require_columns(df: pd.DataFrame, cols: Iterable[str], path: Path) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")


def prepare_subject_table(delta_path: Path, metric: str, trials: pd.DataFrame) -> pd.DataFrame:
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

    # Include openloop and visual variance if available (for decomposed models)
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
    pivot = trials.pivot(index="subject", columns="trial", values="error")
    ordered = pivot.loc[subjects["subject"]]
    return ordered.to_numpy(dtype=float)


@jit
def kalman_loglik_obs(errors: jnp.ndarray, r_state: float, r_obs: float,
                      m: float, A: float, Q: float, b: float) -> float:
    """
    Kalman filter with explicit observation noise.

    Key difference: Separates state uncertainty (R_state) from observation noise (R_obs).

    State: x_{t+1} = A*x_t + w_t, w_t ~ N(0, Q)
    Observation: y_t = -x_t + m + b + ε_t, ε_t ~ N(0, R_obs)

    Kalman gain computed using R_state (affects learning)
    Likelihood computed using R_state + R_obs (total variance)
    """

    def kalman_step(carry, y):
        x, p, ll = carry

        # Predict
        x_pred = A * x
        p_pred = A * p * A + Q

        # Predicted observation
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


def model_numpyro_obs(model_name: str, errors: jnp.ndarray, group_idx: jnp.ndarray,
                      n_groups: int, r_post1: jnp.ndarray, delta_pi: jnp.ndarray,
                      m: float, A: float, Q: float, plateau_group_specific: bool,
                      r_openloop: jnp.ndarray = None, r_visual: jnp.ndarray = None):
    """M-obs model variants.

    Models:
    - M-OBS-FIXED: Fixed R_obs, R_state = r_post1 (proprioceptive variance)
    - M-OBS: R_obs modulated by Δπ, R_state = r_post1
    - M-TWOR: Both R_state and R_obs modulated, R_obs_base is free parameter
    - M-TWOR-OPENLOOP: Both modulated, R_obs_base = openloop variance (empirical)
    - M-TWOR-SENSORY: R_obs = openloop + visual + R_cognitive (decomposed)
    """

    # Plateau
    if plateau_group_specific:
        b = numpyro.sample("b", dist.Normal(0.0, 30.0).expand((n_groups,)))
        b_subj = b[group_idx]
    else:
        b = numpyro.sample("b", dist.Normal(0.0, 30.0))
        b_subj = jnp.repeat(b, len(group_idx))

    model_name = model_name.upper()

    if model_name == "M-OBS-FIXED":
        # M-obs with fixed R_obs (not modulated)
        # This is baseline to test if adding observation noise helps fit

        r_state = r_post1  # Use baseline proprioceptive noise for state
        r_obs = numpyro.sample("r_obs", dist.HalfNormal(2.0))  # Shared observation noise
        r_obs_subj = jnp.repeat(r_obs, len(group_idx))

    elif model_name == "M-OBS":
        # M-obs with Δπ modulating R_obs
        # Tests: Does Δπ affect trial-to-trial execution variability?

        r_state = r_post1  # State uncertainty fixed at baseline
        beta_obs = numpyro.sample("beta_obs", dist.Normal(0.0, 1.0).expand((n_groups,)))

        # R_obs modulated by Δπ
        r_obs_base = numpyro.sample("r_obs_base", dist.HalfNormal(2.0).expand((n_groups,)))
        r_obs_subj = r_obs_base[group_idx] * jnp.exp(beta_obs[group_idx] * delta_pi)

    elif model_name == "M-TWOR":
        # M-twoR: Both R_state and R_obs modulated
        # Tests: Which noise does Δπ affect? Learning or variability?

        beta_state = numpyro.sample("beta_state", dist.Normal(0.0, 1.0).expand((n_groups,)))
        beta_obs = numpyro.sample("beta_obs", dist.Normal(0.0, 1.0).expand((n_groups,)))

        # R_state modulated (affects learning)
        r_state = r_post1 * jnp.exp(beta_state[group_idx] * delta_pi)

        # R_obs modulated (affects variability)
        r_obs_base = numpyro.sample("r_obs_base", dist.HalfNormal(2.0).expand((n_groups,)))
        r_obs_subj = r_obs_base[group_idx] * jnp.exp(beta_obs[group_idx] * delta_pi)

    elif model_name == "M-TWOR-OPENLOOP":
        # M-twoR with empirical R_obs baseline from openloop reaching variance
        # Key difference: R_obs_base is NOT a free parameter, but measured from
        # openloop reaching (no visual feedback) at post1.
        #
        # Cognitive rationale:
        # - Openloop reaching variance = motor execution noise without vision
        # - This is exactly what R_obs should capture: trial-to-trial scatter
        #   that doesn't affect learning (Kalman gain)
        # - Grounding R_obs in empirical data provides mechanistic validation

        if r_openloop is None:
            raise ValueError("M-TWOR-OPENLOOP requires openloop_var_post1 data")

        beta_state = numpyro.sample("beta_state", dist.Normal(0.0, 1.0).expand((n_groups,)))
        beta_obs = numpyro.sample("beta_obs", dist.Normal(0.0, 1.0).expand((n_groups,)))

        # R_state modulated (affects learning) - same as M-TWOR
        r_state = r_post1 * jnp.exp(beta_state[group_idx] * delta_pi)

        # R_obs modulated with EMPIRICAL baseline (openloop variance)
        # No r_obs_base parameter - using measured openloop variance instead
        r_obs_subj = r_openloop * jnp.exp(beta_obs[group_idx] * delta_pi)

    elif model_name == "M-TWOR-SENSORY":
        # M-twoR with decomposed R_obs into sensory and cognitive components
        #
        # R_obs = R_motor + R_visual + R_cognitive × exp(β_obs × Δlog π)
        #
        # Where:
        # - R_motor = openloop_var_post1 (motor execution noise, fixed)
        # - R_visual = visual_var_post1 (visual encoding noise, fixed)
        # - R_cognitive = free parameter (attention, strategy, unmeasured factors)
        #
        # Cognitive rationale:
        # - Sensory noise (motor + visual) should NOT be modulated by Δπ
        # - Only cognitive factors (attention, strategy) are modulated
        # - This tests whether proprioceptive tuning specifically affects
        #   cognitive aspects of trial-to-trial variability

        if r_openloop is None or r_visual is None:
            raise ValueError("M-TWOR-SENSORY requires openloop_var_post1 and visual_var_post1 data")

        beta_state = numpyro.sample("beta_state", dist.Normal(0.0, 1.0).expand((n_groups,)))
        beta_obs = numpyro.sample("beta_obs", dist.Normal(0.0, 1.0).expand((n_groups,)))

        # R_state modulated (affects learning) - same as M-TWOR
        r_state = r_post1 * jnp.exp(beta_state[group_idx] * delta_pi)

        # R_cognitive: free parameter for unmeasured cognitive factors
        r_cognitive = numpyro.sample("r_cognitive", dist.HalfNormal(2.0).expand((n_groups,)))

        # R_obs decomposed: sensory (fixed) + cognitive (modulated)
        # Sensory components: motor execution + visual encoding
        r_sensory = r_openloop + r_visual  # Fixed, empirical

        # Cognitive component modulated by Δπ
        r_cognitive_modulated = r_cognitive[group_idx] * jnp.exp(beta_obs[group_idx] * delta_pi)

        # Total observation noise
        r_obs_subj = r_sensory + r_cognitive_modulated

    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Vectorized likelihood
    vectorized_kalman = vmap(kalman_loglik_obs, in_axes=(0, 0, 0, None, None, None, 0))
    logliks = vectorized_kalman(errors, r_state, r_obs_subj, m, A, Q, b_subj)

    # Add to model
    numpyro.factor("obs", jnp.sum(logliks))
    numpyro.deterministic("log_likelihood", logliks)


def run(model_name: str, trials: pd.DataFrame, subjects: pd.DataFrame,
        args: argparse.Namespace, group_labels: list[str]) -> dict:
    """Fit model and return summary."""

    errors = build_error_matrix(trials, subjects)
    group_to_idx = {g: i for i, g in enumerate(group_labels)}
    group_idx = subjects["group"].map(group_to_idx).to_numpy()
    r_post1 = subjects["r_post1"].to_numpy()
    delta_pi = subjects["delta_pi"].to_numpy()

    # Get openloop variance if available (for M-TWOR-OPENLOOP and M-TWOR-SENSORY)
    r_openloop = None
    if "openloop_var_post1" in subjects.columns:
        r_openloop = subjects["openloop_var_post1"].to_numpy()

    # Get visual variance if available (for M-TWOR-SENSORY)
    r_visual = None
    if "visual_var_post1" in subjects.columns:
        r_visual = subjects["visual_var_post1"].to_numpy()

    kernel = NUTS(
        model_numpyro_obs,
        dense_mass=False,
        target_accept_prob=args.target_accept,
        max_tree_depth=10,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=args.tune,
        num_samples=args.draws,
        num_chains=args.chains,
        chain_method='parallel' if args.chains > 1 else 'sequential',
        progress_bar=True,
    )

    rng_key = random.PRNGKey(args.random_seed)
    mcmc.run(
        rng_key,
        model_name=model_name,
        errors=jnp.asarray(errors),
        group_idx=jnp.asarray(group_idx),
        n_groups=len(group_labels),
        r_post1=jnp.asarray(r_post1),
        delta_pi=jnp.asarray(delta_pi),
        m=args.m,
        A=args.A,
        Q=args.Q,
        plateau_group_specific=args.plateau_group_specific,
        r_openloop=jnp.asarray(r_openloop) if r_openloop is not None else None,
        r_visual=jnp.asarray(r_visual) if r_visual is not None else None,
    )

    # Convert to InferenceData
    coords = {"subject": np.arange(len(subjects))}
    dims = {"log_likelihood": ["subject"]}
    idata = az.from_numpyro(mcmc, coords=coords, dims=dims)

    # Convergence diagnostics
    convergence_summary = az.summary(idata, var_names=["b", "beta_state", "beta_obs", "r_obs", "r_obs_base", "r_cognitive"], filter_vars="like")
    max_rhat = convergence_summary["r_hat"].max() if "r_hat" in convergence_summary else np.nan
    min_ess_bulk = convergence_summary["ess_bulk"].min() if "ess_bulk" in convergence_summary else np.nan
    min_ess_tail = convergence_summary["ess_tail"].min() if "ess_tail" in convergence_summary else np.nan

    if max_rhat > 1.01:
        print(f"  ⚠️  WARNING: Max Rhat = {max_rhat:.3f} (should be < 1.01)")
    if min_ess_bulk < 400:
        print(f"  ⚠️  WARNING: Min ESS_bulk = {min_ess_bulk:.0f} (should be > 400)")

    # WAIC/LOO
    try:
        if 'log_likelihood' in idata.posterior:
            import xarray as xr
            ll_data = idata.posterior['log_likelihood']
            if hasattr(idata, 'log_likelihood') and 'obs' in idata.log_likelihood:
                idata.log_likelihood = idata.log_likelihood.drop_vars('obs')
            if hasattr(idata, 'log_likelihood'):
                idata.log_likelihood['log_likelihood'] = ll_data
            else:
                idata.add_groups({'log_likelihood': xr.Dataset({'log_likelihood': ll_data})})

        waic_res = az.waic(idata, var_name='log_likelihood')
        loo_res = az.loo(idata, var_name='log_likelihood')
        waic = float(waic_res.elpd_waic * -2)
        waic_se = float(waic_res.se * 2)
        waic_p = float(waic_res.p_waic)
        loo = float(loo_res.elpd_loo * -2)
        loo_se = float(loo_res.se * 2)
        loo_p = float(loo_res.p_loo)
    except Exception as e:
        print(f"  Warning: WAIC/LOO computation failed: {e}")
        waic = waic_se = waic_p = loo = loo_se = loo_p = np.nan

    post = mcmc.get_samples()

    summary = {
        "model": model_name,
        "metric": args.metric,
        "n_subjects": len(subjects),
        "n_trials": errors.shape[1],
        "waic": waic,
        "waic_se": waic_se,
        "waic_p": waic_p,
        "loo": loo,
        "loo_se": loo_se,
        "loo_p": loo_p,
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess_bulk,
        "min_ess_tail": min_ess_tail,
    }

    # Extract parameters
    if "beta_state" in post:
        beta_state_med = np.median(np.array(post["beta_state"]), axis=0)
        for g, val in enumerate(beta_state_med):
            summary[f"beta_state_{group_labels[g]}"] = float(val)

    if "beta_obs" in post:
        beta_obs_med = np.median(np.array(post["beta_obs"]), axis=0)
        for g, val in enumerate(beta_obs_med):
            summary[f"beta_obs_{group_labels[g]}"] = float(val)

    if "r_obs" in post:
        r_obs_med = np.median(np.array(post["r_obs"]))
        summary["r_obs"] = float(r_obs_med)

    if "r_obs_base" in post:
        r_obs_base_med = np.median(np.array(post["r_obs_base"]), axis=0)
        for g, val in enumerate(r_obs_base_med):
            summary[f"r_obs_base_{group_labels[g]}"] = float(val)

    if "r_cognitive" in post:
        r_cognitive_med = np.median(np.array(post["r_cognitive"]), axis=0)
        for g, val in enumerate(r_cognitive_med):
            summary[f"r_cognitive_{group_labels[g]}"] = float(val)

    b_med = np.median(np.array(post["b"]), axis=0)
    if np.ndim(b_med) == 0:
        summary["b"] = float(b_med)
    else:
        for g, val in enumerate(b_med):
            summary[f"b_{group_labels[g]}"] = float(val)

    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit M-obs models")
    p.add_argument("--trials-path", type=Path, default=Path("data/derived/adaptation_trials.csv"))
    p.add_argument("--delta-path", type=Path, default=Path("data/derived/proprio_delta_pi.csv"))
    p.add_argument("--metric", choices=["pi", "logpi"], default="logpi")
    p.add_argument("--models", type=str, default="M-obs-fixed,M-obs,M-twoR,M-twoR-openloop,M-twoR-sensory")
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--tune", type=int, default=1000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--target-accept", type=float, default=0.85)
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--m", type=float, default=-12.1)
    p.add_argument("--A", type=float, default=1.0)
    p.add_argument("--Q", type=float, default=1e-4)
    p.add_argument("--max-subjects", type=int, default=None)
    p.add_argument("--plateau-group-specific", action="store_true")
    p.add_argument("--out-path", type=Path, default=Path("data/derived/m_obs_results.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"JAX devices: {jax.devices()}")
    print(f"Number of chains: {args.chains}")
    print(f"Warmup: {args.tune}, Samples: {args.draws}")

    trials = pd.read_csv(args.trials_path)
    require_columns(trials, ["subject", "group", "trial", "error"], args.trials_path)
    subjects = prepare_subject_table(args.delta_path, args.metric, trials)

    if args.max_subjects is not None:
        subjects = subjects.head(args.max_subjects)
        print(f"Limited to {len(subjects)} subjects for testing")

    group_labels = sorted(subjects["group"].unique())
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    rows = []
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Fitting {model_name}...")
        print(f"{'='*60}")
        res = run(model_name, trials, subjects, args, group_labels)
        rows.append(res)
        print(f"\nFinished {model_name}:")
        print(f"  WAIC = {res['waic']:.1f}")
        print(f"  LOO = {res['loo']:.1f}")
        print(f"  Rhat = {res['max_rhat']:.3f}")
        print(f"  ESS_bulk = {res['min_ess_bulk']:.0f}")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_path, index=False)
    print(f"\n{'='*60}")
    print(f"Wrote {args.out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
