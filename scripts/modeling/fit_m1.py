"""
Fit the paper's M1 (Coupling) state-space model.

M1 collapses the total endpoint variance into a single update-relevant noise
term, S = R, modulated by proprioceptive precision change Δlogπ:

    state:        x_{t+1} = A x_t + w_t,        w_t ~ N(0, Q)
    observation:  y_t     = m - x_t + b + ε_t,  ε_t ~ N(0, R)
    R = R_0 · exp(β_R · Δlogπ)

with A = 1 and Q = 1e-4 fixed, and R_0 = per-subject post-tuning proprioceptive
variance (`r_post1`). The same R drives both the Kalman gain and the
endpoint-error likelihood.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import arviz as az
import jax
import jax.numpy as jnp
from jax import jit, lax, random, vmap
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
jax.config.update('jax_enable_x64', True)


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
    merged = (
        trial_subjects
        .merge(delta_df, left_on=["subject", "group"],
               right_on=["ID", "group"], how="inner")
        .dropna(subset=["r_post1", delta_col])
    )
    return (
        merged[["subject", "group", "r_post1", delta_col]]
        .rename(columns={delta_col: "delta_pi"})
        .reset_index(drop=True)
    )


def build_error_matrix(trials: pd.DataFrame, subjects: pd.DataFrame) -> np.ndarray:
    pivot = trials.pivot(index="subject", columns="trial", values="error")
    return pivot.loc[subjects["subject"]].to_numpy(dtype=float)


@jit
def kalman_loglik(errors: jnp.ndarray, r: float, m: float, A: float, Q: float, b: float) -> float:
    """Single-noise Kalman log-likelihood (S = R) used by paper M1."""
    def step(carry, y):
        x, p, ll = carry
        x_pred = A * x
        p_pred = A * p * A + Q
        y_pred = -x_pred + (m + b)
        s = p_pred + r
        v = y - y_pred
        ll_update = -0.5 * (jnp.log(2.0 * jnp.pi * s) + (v * v) / s)
        k = -p_pred / s
        x_new = x_pred + k * v
        p_new = (1.0 + k) * p_pred
        return (x_new, p_new, ll + ll_update), None

    init = (0.0, 1.0, 0.0)
    final, _ = lax.scan(step, init, errors)
    return final[2]


def model(errors, group_idx, n_groups, r_post1, delta_pi, m, A, Q, plateau_group_specific):
    """NumPyro model for paper M1: R = R_0 · exp(β_R · Δlogπ)."""
    if plateau_group_specific:
        b = numpyro.sample("b", dist.Normal(0.0, 30.0).expand((n_groups,)))
        b_subj = b[group_idx]
    else:
        b = numpyro.sample("b", dist.Normal(0.0, 30.0))
        b_subj = jnp.repeat(b, len(group_idx))

    beta_R = numpyro.sample("beta_R", dist.Normal(0.0, 1.0).expand((n_groups,)))
    r = r_post1 * jnp.exp(beta_R[group_idx] * delta_pi)

    vectorized = vmap(kalman_loglik, in_axes=(0, 0, None, None, None, 0))
    logliks = vectorized(errors, r, m, A, Q, b_subj)
    numpyro.factor("obs", jnp.sum(logliks))
    numpyro.deterministic("log_likelihood", logliks)


def run(trials, subjects, args, group_labels) -> tuple[dict, az.InferenceData]:
    errors = build_error_matrix(trials, subjects)
    group_to_idx = {g: i for i, g in enumerate(group_labels)}
    group_idx = subjects["group"].map(group_to_idx).to_numpy()
    r_post1 = subjects["r_post1"].to_numpy()
    delta_pi = subjects["delta_pi"].to_numpy()

    kernel = NUTS(model, dense_mass=False,
                  target_accept_prob=args.target_accept, max_tree_depth=10)
    mcmc = MCMC(kernel, num_warmup=args.tune, num_samples=args.draws,
                num_chains=args.chains,
                chain_method='parallel' if args.chains > 1 else 'sequential',
                progress_bar=True)

    rng_key = random.PRNGKey(args.random_seed)
    mcmc.run(rng_key,
             errors=jnp.asarray(errors),
             group_idx=jnp.asarray(group_idx),
             n_groups=len(group_labels),
             r_post1=jnp.asarray(r_post1),
             delta_pi=jnp.asarray(delta_pi),
             m=args.m, A=args.A, Q=args.Q,
             plateau_group_specific=args.plateau_group_specific)

    coords = {"subject": np.arange(len(subjects))}
    dims = {"log_likelihood": ["subject"]}
    idata = az.from_numpyro(mcmc, coords=coords, dims=dims)

    diag = az.summary(idata, filter_vars="like", var_names=["b", "beta_R"])
    max_rhat = diag["r_hat"].max() if "r_hat" in diag else np.nan
    min_ess_bulk = diag["ess_bulk"].min() if "ess_bulk" in diag else np.nan
    min_ess_tail = diag["ess_tail"].min() if "ess_tail" in diag else np.nan
    if max_rhat > 1.01:
        print(f"  ⚠️  WARNING: Max R̂ = {max_rhat:.3f} (should be < 1.01)")
    if min_ess_bulk < 400:
        print(f"  ⚠️  WARNING: Min ESS_bulk = {min_ess_bulk:.0f} (should be > 400)")

    try:
        import xarray as xr
        ll_data = idata.posterior["log_likelihood"]
        if hasattr(idata, "log_likelihood") and "obs" in idata.log_likelihood:
            idata.log_likelihood = idata.log_likelihood.drop_vars("obs")
        if hasattr(idata, "log_likelihood"):
            idata.log_likelihood["log_likelihood"] = ll_data
        else:
            idata.add_groups({"log_likelihood": xr.Dataset({"log_likelihood": ll_data})})
        waic_res = az.waic(idata, var_name="log_likelihood")
        loo_res = az.loo(idata, var_name="log_likelihood")
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
        "model": "M1",
        "metric": args.metric,
        "n_subjects": len(subjects),
        "n_trials": errors.shape[1],
        "waic": waic, "waic_se": waic_se, "waic_p": waic_p,
        "loo": loo, "loo_se": loo_se, "loo_p": loo_p,
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess_bulk,
        "min_ess_tail": min_ess_tail,
    }
    beta_R_med = np.median(np.array(post["beta_R"]), axis=0)
    for g, val in enumerate(beta_R_med):
        summary[f"beta_R_{group_labels[g]}"] = float(val)

    b_med = np.median(np.array(post["b"]), axis=0)
    if np.ndim(b_med) == 0:
        summary["b"] = float(b_med)
    else:
        for g, val in enumerate(b_med):
            summary[f"b_{group_labels[g]}"] = float(val)

    return summary, idata


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit paper M1 (Coupling) state-space model.")
    p.add_argument("--trials-path", type=Path, default=Path("data/adaptation_trials.csv"))
    p.add_argument("--delta-path", type=Path, default=Path("data/proprio_delta_pi.csv"))
    p.add_argument("--metric", choices=["pi", "logpi"], default="logpi")
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
    p.add_argument("--out-path", type=Path, default=Path("data/m1_results.csv"))
    p.add_argument("--save-posterior", action="store_true",
                   help="Save full posterior as data/posteriors/m1_posterior.nc")
    p.add_argument("--posterior-dir", type=Path, default=Path("data/posteriors"))
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
        print(f"Limited to {len(subjects)} subjects")

    group_labels = sorted(subjects["group"].unique())
    print(f"\n{'='*60}\nFitting M1\n{'='*60}")
    res, idata = run(trials, subjects, args, group_labels)
    print(f"\nFinished M1: WAIC={res['waic']:.1f}  LOO={res['loo']:.1f}  "
          f"R̂={res['max_rhat']:.3f}  ESS_bulk={res['min_ess_bulk']:.0f}")

    if args.save_posterior:
        args.posterior_dir.mkdir(parents=True, exist_ok=True)
        nc_path = args.posterior_dir / "m1_posterior.nc"
        idata.to_netcdf(nc_path)
        print(f"  Saved posterior → {nc_path}")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([res]).to_csv(args.out_path, index=False)
    print(f"Wrote {args.out_path}")


if __name__ == "__main__":
    main()
