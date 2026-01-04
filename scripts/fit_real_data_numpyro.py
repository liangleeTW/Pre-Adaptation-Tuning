"""NumPyro-based fitting of real adaptation data (M0–M2, group-specific, plateau optional)."""

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
from jax import random, jit, vmap
from jax import lax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, log_likelihood

# Set JAX to use all CPU cores on M2
import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
# Enable float64 for numerical stability
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

    keep_cols = ["subject", "group", "r_post1", delta_col]
    merged = merged[keep_cols].rename(columns={delta_col: "delta_pi"})
    if merged.empty:
        raise ValueError("No overlapping subjects between trials and delta file.")
    return merged


def build_error_matrix(trials: pd.DataFrame, subjects: pd.DataFrame) -> np.ndarray:
    pivot = trials.pivot(index="subject", columns="trial", values="error")
    ordered = pivot.loc[subjects["subject"]]
    return ordered.to_numpy(dtype=float)


@jit
def kalman_loglik(errors: jnp.ndarray, r_measure: float, m: float, A: float, Q: float, b: float) -> float:
    """Scalar-state Kalman log-likelihood using lax.scan for speed."""

    def kalman_step(carry, y):
        x, p, ll = carry
        x_pred = A * x
        p_pred = A * p * A + Q
        y_pred = -x_pred + (m + b)
        s = p_pred + r_measure
        v = y - y_pred
        ll_update = -0.5 * (jnp.log(2.0 * jnp.pi * s) + (v * v) / s)
        # FIXED: Kalman gain with correct sign (h = -1 in observation model)
        k = -p_pred / s
        x_new = x_pred + k * v  # Clearer: update from prediction
        p_new = (1.0 + k) * p_pred  # Equivalent to (1 - k*h)*p since k is negative
        ll_new = ll + ll_update
        return (x_new, p_new, ll_new), None

    init_state = (0.0, 1.0, 0.0)  # (x, p, ll)
    final_state, _ = lax.scan(kalman_step, init_state, errors)
    return final_state[2]  # return final log-likelihood


def model_numpyro(model_name: str, errors: jnp.ndarray, group_idx: jnp.ndarray, n_groups: int, r_post1: jnp.ndarray, delta_pi: jnp.ndarray, m: float, A: float, Q: float, plateau_group_specific: bool):
    if plateau_group_specific:
        b = numpyro.sample("b", dist.Normal(0.0, 30.0).expand((n_groups,)))
        b_subj = b[group_idx]
    else:
        b = numpyro.sample("b", dist.Normal(0.0, 30.0))
        b_subj = jnp.repeat(b, len(group_idx))

    if model_name == "M0":
        beta = None
        lam = None
        r_measure = r_post1
    elif model_name == "M1":
        beta = numpyro.sample("beta", dist.Normal(0.0, 1.0).expand((n_groups,)))
        r_measure = r_post1 + beta[group_idx] * delta_pi
        lam = None
    else:  # M2
        lam = numpyro.sample("lam", dist.Normal(0.0, 0.5).expand((n_groups,)))
        lam_t = jnp.tanh(lam)
        r_measure = r_post1 * (1.0 - lam_t[group_idx] * jnp.tanh(delta_pi))
        beta = None

    # Vectorized likelihood computation across subjects (fast)
    vectorized_kalman = vmap(kalman_loglik, in_axes=(0, 0, None, None, None, 0))
    logliks = vectorized_kalman(errors, r_measure, m, A, Q, b_subj)

    # Add total log-likelihood to model
    numpyro.factor("obs", jnp.sum(logliks))

    # Store per-subject log-likelihoods for WAIC/LOO computation
    # This will be used by ArviZ for model comparison
    numpyro.deterministic("log_likelihood", logliks)


def run(model_name: str, trials: pd.DataFrame, subjects: pd.DataFrame, args: argparse.Namespace, group_labels: list[str]) -> dict[str, float]:
    errors = build_error_matrix(trials, subjects)
    group_to_idx = {g: i for i, g in enumerate(group_labels)}
    group_idx = subjects["group"].map(group_to_idx).to_numpy()
    r_post1 = subjects["r_post1"].to_numpy()
    delta_pi = subjects["delta_pi"].to_numpy()

    kernel = NUTS(
        model_numpyro,
        dense_mass=False,  # Diagonal mass matrix is faster
        target_accept_prob=args.target_accept,
        max_tree_depth=10,  # Limit tree depth to prevent long trajectories
    )
    mcmc = MCMC(
        kernel,
        num_warmup=args.tune,
        num_samples=args.draws,
        num_chains=args.chains,
        chain_method='parallel' if args.chains > 1 else 'sequential',  # Parallel chains
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
    )

    # Convert to InferenceData with proper log_likelihood structure
    # Add coordinates for subjects
    coords = {"subject": np.arange(len(subjects))}
    dims = {"log_likelihood": ["subject"]}

    idata = az.from_numpyro(
        mcmc,
        coords=coords,
        dims=dims,
    )

    # Compute convergence diagnostics
    convergence_summary = az.summary(idata, var_names=["b", "beta", "lam"], filter_vars="like")
    max_rhat = convergence_summary["r_hat"].max() if "r_hat" in convergence_summary else np.nan
    min_ess_bulk = convergence_summary["ess_bulk"].min() if "ess_bulk" in convergence_summary else np.nan
    min_ess_tail = convergence_summary["ess_tail"].min() if "ess_tail" in convergence_summary else np.nan

    # Print convergence warnings if needed
    if max_rhat > 1.01:
        print(f"  ⚠️  WARNING: Max Rhat = {max_rhat:.3f} (should be < 1.01)")
    if min_ess_bulk < 400:
        print(f"  ⚠️  WARNING: Min ESS_bulk = {min_ess_bulk:.0f} (should be > 400)")

    try:
        # Move log_likelihood from posterior to log_likelihood group for WAIC/LOO
        # Create a proper log_likelihood group from the posterior deterministic
        if 'log_likelihood' in idata.posterior:
            import xarray as xr
            # Extract log_likelihood from posterior
            ll_data = idata.posterior['log_likelihood']
            # Remove 'obs' from log_likelihood group if it exists (from factor)
            if hasattr(idata, 'log_likelihood') and 'obs' in idata.log_likelihood:
                idata.log_likelihood = idata.log_likelihood.drop_vars('obs')
            # Add or replace with the proper per-subject log-likelihoods
            if hasattr(idata, 'log_likelihood'):
                idata.log_likelihood['log_likelihood'] = ll_data
            else:
                idata.add_groups({'log_likelihood': xr.Dataset({'log_likelihood': ll_data})})

        # Compute WAIC/LOO using the log_likelihood variable
        waic_res = az.waic(idata, var_name='log_likelihood')
        loo_res = az.loo(idata, var_name='log_likelihood')
        # ArviZ returns ELPDData objects - convert ELPD to deviance scale (*-2)
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
        "group_specific": True,
        "fit_plateau": True,
        "plateau_group_specific": args.plateau_group_specific,
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
    if "beta" in post:
        beta_med = np.median(np.array(post["beta"]), axis=0)
        for g, val in enumerate(beta_med):
            summary[f"beta_{group_labels[g]}"] = float(val)

    # FIXED: Report tanh-transformed lambda (the actual parameter in the model)
    if "lam" in post:
        lam_raw = np.array(post["lam"])  # Raw unconstrained parameter
        lam_transformed = np.tanh(lam_raw)  # Constrained to [-1, 1]
        lam_med = np.median(lam_transformed, axis=0)
        for g, val in enumerate(lam_med):
            summary[f"lam_{group_labels[g]}"] = float(val)
        # Also report saturation indicators
        for g in range(lam_transformed.shape[-1]):
            summary[f"lam_saturated_{group_labels[g]}"] = float((np.abs(lam_transformed[:, g]) > 0.95).mean())

    b_med = np.median(np.array(post["b"]), axis=0)
    if np.ndim(b_med) == 0:
        summary["b"] = float(b_med)
    else:
        for g, val in enumerate(b_med):
            summary[f"b_{group_labels[g]}"] = float(val)

    # Sign prob for lam if present (use transformed values)
    if "lam" in post:
        lam_raw = np.array(post["lam"])
        lam_transformed = np.tanh(lam_raw)
        for g in range(lam_transformed.shape[-1]):
            summary[f"Pr_lam_gt0_{group_labels[g]}"] = float((lam_transformed[:, g] > 0).mean())

    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NumPyro fit with group-specific modulation and plateau.")
    p.add_argument("--trials-path", type=Path, default=Path("data/derived/adaptation_trials.csv"))
    p.add_argument("--delta-path", type=Path, default=Path("data/derived/proprio_delta_pi.csv"))
    p.add_argument("--metric", choices=["pi", "logpi"], default="pi")
    p.add_argument("--models", type=str, default="M0,M1,M2")
    p.add_argument("--draws", type=int, default=500, help="Posterior samples per chain (default: 500)")
    p.add_argument("--tune", type=int, default=500, help="Warmup iterations (default: 500)")
    p.add_argument("--chains", type=int, default=4, help="Number of parallel chains (default: 4)")
    p.add_argument("--target-accept", type=float, default=0.85, help="NUTS target acceptance (default: 0.85)")
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--m", type=float, default=-12.1)
    p.add_argument("--A", type=float, default=1.0)
    p.add_argument("--Q", type=float, default=1e-4)
    p.add_argument("--max-subjects", type=int, default=None)
    p.add_argument("--plateau-group-specific", action="store_true")
    p.add_argument("--out-path", type=Path, default=Path("data/derived/real_fit_numpyro.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Print JAX configuration
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX default backend: {jax.default_backend()}")
    print(f"Number of chains: {args.chains}")
    print(f"Warmup: {args.tune}, Samples: {args.draws}")

    trials = pd.read_csv(args.trials_path)
    require_columns(trials, ["subject", "group", "trial", "error"], args.trials_path)
    subjects = prepare_subject_table(args.delta_path, args.metric, trials)
    if args.max_subjects is not None:
        subjects = subjects.head(args.max_subjects)
        print(f"Limited to {len(subjects)} subjects for testing")
    group_labels = sorted(subjects["group"].unique())

    models = [m.strip().upper() for m in args.models.split(",") if m.strip()]

    rows = []
    for model_name in models:
        res = run(model_name, trials, subjects, args, group_labels)
        rows.append(res)
        print(f"Finished {model_name}: WAIC={res['waic']}, LOO={res['loo']}")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_path, index=False)
    print(f"Wrote {args.out_path}")


if __name__ == "__main__":
    main()
