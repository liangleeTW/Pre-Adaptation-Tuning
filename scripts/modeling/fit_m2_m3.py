"""
Fit the paper's M2 (Dissociation) and M3 (Dissociation + Decomposition)
state-space models to per-trial endpoint errors.

Both models share the same Kalman likelihood:

    state:        x_{t+1} = A x_t + w_t,        w_t ~ N(0, Q)
    observation:  y_t     = m - x_t + b + ε_t,  ε_t ~ N(0, S)

with A = 1 and Q = 1e-4 held fixed. They differ in how the total endpoint
variance S decomposes into an update-relevant component R (drives the Kalman
gain) and a residual component V (contributes to endpoint variability only):

    M2:  R = R_0 · exp(β_R · Δlogπ)
         V = V_0 · exp(β_cog · Δlogπ)

    M3:  R = R_0 · exp(β_R · Δlogπ)
         V = V_motor + V_visual + V_cog · exp(β_cog · Δlogπ)

R_0 is the per-subject post-tuning proprioceptive variance (`r_post1`).
V_motor and V_visual in M3 are the per-subject post-tuning open-loop
(reach-only) and visual localisation variances.
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

# Set JAX config
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
jax.config.update('jax_enable_x64', True)


# Internal sample-site names match the variables stored in the shipped
# posteriors (m2_posterior.nc, m3_posterior.nc):
#   beta_state   ↔ paper β_R     (modulates update-relevant noise R)
#   beta_obs     ↔ paper β_cog   (modulates residual / cognitive noise)
#   r_obs_base   ↔ paper V_0     (M2 only)
#   r_cognitive  ↔ paper V_cog   (M3 only)


def require_columns(df: pd.DataFrame, cols: Iterable[str], path: Path) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")


def prepare_subject_table(delta_path: Path, metric: str, trials: pd.DataFrame) -> pd.DataFrame:
    delta_df = pd.read_csv(delta_path)
    require_columns(
        delta_df,
        ["ID", "group", "precision_post1", "delta_pi", "delta_log_pi",
         "openloop_var_post1", "visual_var_post1"],
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
        .dropna(subset=["r_post1", delta_col,
                        "openloop_var_post1", "visual_var_post1"])
    )

    keep_cols = ["subject", "group", "r_post1", delta_col,
                 "openloop_var_post1", "visual_var_post1"]
    merged = merged[keep_cols].rename(columns={delta_col: "delta_pi"})
    if merged.empty:
        raise ValueError("No overlapping subjects between trials and delta file.")
    return merged


def build_error_matrix(trials: pd.DataFrame, subjects: pd.DataFrame) -> np.ndarray:
    pivot = trials.pivot(index="subject", columns="trial", values="error")
    ordered = pivot.loc[subjects["subject"]]
    return ordered.to_numpy(dtype=float)


@jit
def kalman_loglik(errors: jnp.ndarray, r_state: float, r_obs: float,
                  m: float, A: float, Q: float, b: float) -> float:
    """Per-subject Kalman log-likelihood with separate R (state) and V (obs)."""
    def step(carry, y):
        x, p, ll = carry
        # Predict
        x_pred = A * x
        p_pred = A * p * A + Q
        # Predicted observation
        y_pred = -x_pred + (m + b)
        # Innovation variance for the Kalman gain (update-relevant only)
        s_kalman = p_pred + r_state
        # Innovation variance for the likelihood (state + residual)
        s_total = p_pred + r_state + r_obs
        v = y - y_pred
        ll_update = -0.5 * (jnp.log(2.0 * jnp.pi * s_total) + (v * v) / s_total)
        # Kalman gain
        k = -p_pred / s_kalman
        x_new = x_pred + k * v
        p_new = (1.0 + k) * p_pred
        return (x_new, p_new, ll + ll_update), None

    init = (0.0, 1.0, 0.0)
    final, _ = lax.scan(step, init, errors)
    return final[2]


def model(model_name: str, errors: jnp.ndarray, group_idx: jnp.ndarray,
          n_groups: int, r_post1: jnp.ndarray, delta_pi: jnp.ndarray,
          r_openloop: jnp.ndarray, r_visual: jnp.ndarray,
          m: float, A: float, Q: float, plateau_group_specific: bool):
    """NumPyro model for paper M2 or M3."""
    if plateau_group_specific:
        b = numpyro.sample("b", dist.Normal(0.0, 30.0).expand((n_groups,)))
        b_subj = b[group_idx]
    else:
        b = numpyro.sample("b", dist.Normal(0.0, 30.0))
        b_subj = jnp.repeat(b, len(group_idx))

    name = model_name.upper()

    # Update-relevant noise R = R_0 · exp(β_R · Δlogπ) — common to M2 and M3.
    beta_state = numpyro.sample("beta_state", dist.Normal(0.0, 1.0).expand((n_groups,)))
    r_state = r_post1 * jnp.exp(beta_state[group_idx] * delta_pi)

    beta_obs = numpyro.sample("beta_obs", dist.Normal(0.0, 1.0).expand((n_groups,)))

    if name == "M2":
        # V = V_0 · exp(β_cog · Δlogπ),  V_0 free per group.
        r_obs_base = numpyro.sample("r_obs_base", dist.HalfNormal(2.0).expand((n_groups,)))
        r_obs_subj = r_obs_base[group_idx] * jnp.exp(beta_obs[group_idx] * delta_pi)

    elif name == "M3":
        # V = V_motor + V_visual + V_cog · exp(β_cog · Δlogπ).
        # V_motor / V_visual are fixed to per-subject empirical post-tuning
        # open-loop and visual localisation variances.
        r_cognitive = numpyro.sample("r_cognitive", dist.HalfNormal(2.0).expand((n_groups,)))
        r_obs_subj = (
            r_openloop + r_visual
            + r_cognitive[group_idx] * jnp.exp(beta_obs[group_idx] * delta_pi)
        )

    else:
        raise ValueError(f"Unknown model: {model_name!r} (expected 'M2' or 'M3')")

    # Vectorised likelihood across subjects
    vectorized = vmap(kalman_loglik, in_axes=(0, 0, 0, None, None, None, 0))
    logliks = vectorized(errors, r_state, r_obs_subj, m, A, Q, b_subj)
    numpyro.factor("obs", jnp.sum(logliks))
    numpyro.deterministic("log_likelihood", logliks)


def run(model_name: str, trials: pd.DataFrame, subjects: pd.DataFrame,
        args: argparse.Namespace, group_labels: list[str]) -> tuple[dict, az.InferenceData]:
    errors = build_error_matrix(trials, subjects)
    group_to_idx = {g: i for i, g in enumerate(group_labels)}
    group_idx = subjects["group"].map(group_to_idx).to_numpy()
    r_post1 = subjects["r_post1"].to_numpy()
    delta_pi = subjects["delta_pi"].to_numpy()
    r_openloop = subjects["openloop_var_post1"].to_numpy()
    r_visual = subjects["visual_var_post1"].to_numpy()

    kernel = NUTS(model, dense_mass=False,
                  target_accept_prob=args.target_accept, max_tree_depth=10)
    mcmc = MCMC(kernel, num_warmup=args.tune, num_samples=args.draws,
                num_chains=args.chains,
                chain_method='parallel' if args.chains > 1 else 'sequential',
                progress_bar=True)

    rng_key = random.PRNGKey(args.random_seed)
    mcmc.run(
        rng_key,
        model_name=model_name,
        errors=jnp.asarray(errors),
        group_idx=jnp.asarray(group_idx),
        n_groups=len(group_labels),
        r_post1=jnp.asarray(r_post1),
        delta_pi=jnp.asarray(delta_pi),
        r_openloop=jnp.asarray(r_openloop),
        r_visual=jnp.asarray(r_visual),
        m=args.m, A=args.A, Q=args.Q,
        plateau_group_specific=args.plateau_group_specific,
    )

    coords = {"subject": np.arange(len(subjects))}
    dims = {"log_likelihood": ["subject"]}
    idata = az.from_numpyro(mcmc, coords=coords, dims=dims)

    # Convergence diagnostics
    diag = az.summary(idata, filter_vars="like",
                      var_names=["b", "beta_state", "beta_obs",
                                 "r_obs_base", "r_cognitive"])
    max_rhat = diag["r_hat"].max() if "r_hat" in diag else np.nan
    min_ess_bulk = diag["ess_bulk"].min() if "ess_bulk" in diag else np.nan
    min_ess_tail = diag["ess_tail"].min() if "ess_tail" in diag else np.nan
    if max_rhat > 1.01:
        print(f"  ⚠️  WARNING: Max R̂ = {max_rhat:.3f} (should be < 1.01)")
    if min_ess_bulk < 400:
        print(f"  ⚠️  WARNING: Min ESS_bulk = {min_ess_bulk:.0f} (should be > 400)")

    # WAIC / LOO from the deterministic log_likelihood
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
        "model": model_name,
        "metric": args.metric,
        "n_subjects": len(subjects),
        "n_trials": errors.shape[1],
        "waic": waic, "waic_se": waic_se, "waic_p": waic_p,
        "loo": loo, "loo_se": loo_se, "loo_p": loo_p,
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess_bulk,
        "min_ess_tail": min_ess_tail,
    }

    # Internal sample-site → paper symbol used in the output CSV.
    paper_label = {"beta_state": "beta_R", "beta_obs": "beta_cog",
                   "r_obs_base": "V0", "r_cognitive": "V_cog"}
    for var, label in paper_label.items():
        if var in post:
            med = np.median(np.array(post[var]), axis=0)
            for g, val in enumerate(med):
                summary[f"{label}_{group_labels[g]}"] = float(val)

    b_med = np.median(np.array(post["b"]), axis=0)
    if np.ndim(b_med) == 0:
        summary["b"] = float(b_med)
    else:
        for g, val in enumerate(b_med):
            summary[f"b_{group_labels[g]}"] = float(val)

    return summary, idata


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit paper M2 and M3 state-space models")
    p.add_argument("--trials-path", type=Path, default=Path("data/adaptation_trials.csv"))
    p.add_argument("--delta-path", type=Path, default=Path("data/proprio_delta_pi.csv"))
    p.add_argument("--metric", choices=["pi", "logpi"], default="logpi")
    p.add_argument("--models", type=str, default="M2,M3")
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
    p.add_argument("--out-path", type=Path, default=Path("data/m2_m3_results.csv"))
    p.add_argument("--exclude-subjects", type=str, default=None,
                   help="Comma-separated list of subject IDs to exclude")
    p.add_argument("--save-posterior", action="store_true",
                   help="Save full posterior as NetCDF (data/posteriors/<model>_posterior.nc)")
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

    if args.exclude_subjects:
        exclude = [s.strip() for s in args.exclude_subjects.split(",")]
        n_before = len(subjects)
        subjects = subjects[~subjects["subject"].isin(exclude)]
        print(f"Excluded {n_before - len(subjects)} subjects: {exclude}")

    if args.max_subjects is not None:
        subjects = subjects.head(args.max_subjects)
        print(f"Limited to {len(subjects)} subjects")

    print("\nSubjects per group:")
    for g in sorted(subjects["group"].unique()):
        print(f"  {g}: {len(subjects[subjects['group'] == g])}")

    group_labels = sorted(subjects["group"].unique())
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    if args.save_posterior:
        args.posterior_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nPosteriors will be saved to: {args.posterior_dir}")

    rows = []
    for model_name in models:
        print(f"\n{'='*60}\nFitting {model_name}\n{'='*60}")
        res, idata = run(model_name, trials, subjects, args, group_labels)
        rows.append(res)
        print(f"\nFinished {model_name}: WAIC={res['waic']:.1f}  LOO={res['loo']:.1f}  "
              f"R̂={res['max_rhat']:.3f}  ESS_bulk={res['min_ess_bulk']:.0f}")
        if args.save_posterior and idata is not None:
            nc_path = args.posterior_dir / f"{model_name.lower()}_posterior.nc"
            idata.to_netcdf(nc_path)
            print(f"  Saved posterior → {nc_path}")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_path, index=False)
    print(f"\n{'='*60}\nWrote {args.out_path}\n{'='*60}")


if __name__ == "__main__":
    main()
