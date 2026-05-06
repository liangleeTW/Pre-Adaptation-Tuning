"""
Pillar 2 — Parameter Recovery
===============================
Simulate data from M3's posterior, refit M3 (and M2) to each synthetic dataset,
compare recovered vs true parameters.

Shows:
  - M3's β_state is unbiased and identifiable
  - M2 recovers similar β_state (main result is robust) but inflates R_obs_base
    by absorbing the empirical sensory baseline — an overfitting artifact

Runtime: ~1-2 hrs for --n-sims 200 (recommended). Run overnight.
Command:
    poetry run python scripts/recovery/parameter_recovery.py \
        --n-sims 200 --draws 500 --warmup 500 --chains 2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax, jit, vmap
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

# ── constants ─────────────────────────────────────────────────────────────────
M_TRUE   = -12.1
A        = 1.0
Q        = 1e-4
N_TRIALS = 100
SEED     = 42
GROUP_ORDER = ["EC", "EO+", "EO-"]

POSTERIOR_DIR = ROOT / "data" / "posteriors"
TRIALS_PATH   = ROOT / "data" / "adaptation_trials.csv"
DELTA_PATH    = ROOT / "data" / "proprio_delta_pi.csv"
OUT_DIR       = ROOT / "data" / "recovery"
FIG_DIR       = ROOT / "figures" / "recovery"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ── Kalman simulator (matches fitting code exactly) ───────────────────────────
@jit
def simulate_one_subject(key, r_state: float, r_obs: float, b: float) -> jnp.ndarray:
    """
    Generative model matching fit_m2_m3.py:
      y_t = -x_t + M_TRUE + b + ε_t,  ε_t ~ N(0, r_state + r_obs)
      Kalman gain uses r_state only.
    """
    def step(carry, _):
        x, p, rng = carry
        rng, k1, k2 = jax.random.split(rng, 3)

        x_pred = A * x
        p_pred = A * p * A + Q
        s_kalman = p_pred + r_state

        # True state gets tiny process noise
        proc  = jax.random.normal(k1) * jnp.sqrt(Q)
        x_true = x_pred + proc

        # Total observation noise
        total_noise = jax.random.normal(k2) * jnp.sqrt(r_state + r_obs)
        y_t = -x_true + M_TRUE + b + total_noise

        v = y_t - (-x_pred + M_TRUE + b)
        k = -p_pred / s_kalman
        x_new = x_pred + k * v
        p_new = (1.0 + k) * p_pred

        return (x_new, p_new, rng), y_t

    _, errors = lax.scan(step, (0.0, 1.0, key), None, length=N_TRIALS)
    return errors


# ── JAX Kalman log-likelihood (same as fitting code) ─────────────────────────
@jit
def kalman_loglik_obs(errors, r_state, r_obs, m, A_, Q_, b):
    def step(carry, y):
        x, p, ll = carry
        x_pred = A_ * x
        p_pred = A_ * p * A_ + Q_
        y_pred = -x_pred + (m + b)
        s_kalman = p_pred + r_state
        s_total  = p_pred + r_state + r_obs
        v = y - y_pred
        ll_update = -0.5 * (jnp.log(2.0 * jnp.pi * s_total) + v * v / s_total)
        k = -p_pred / s_kalman
        x_new = x_pred + k * v
        p_new = (1.0 + k) * p_pred
        return (x_new, p_new, ll + ll_update), None
    (_, _, ll), _ = lax.scan(step, (0.0, 1.0, 0.0), errors)
    return ll


# ── NUMPYro model definitions ─────────────────────────────────────────────────
def numpyro_m3(errors, group_idx, n_groups, r_post1, delta_pi,
               r_openloop, r_visual, m, A_, Q_):
    """M3-decomp (M-TWOR-SENSORY)."""
    b          = numpyro.sample("b",          dist.Normal(0.0, 30.0).expand((n_groups,)))
    beta_state = numpyro.sample("beta_state", dist.Normal(0.0, 1.0).expand((n_groups,)))
    beta_obs   = numpyro.sample("beta_obs",   dist.Normal(0.0, 1.0).expand((n_groups,)))
    r_cognitive= numpyro.sample("r_cognitive",dist.HalfNormal(2.0).expand((n_groups,)))

    r_state = r_post1 * jnp.exp(beta_state[group_idx] * delta_pi)
    r_obs   = (r_openloop + r_visual) + r_cognitive[group_idx] * jnp.exp(beta_obs[group_idx] * delta_pi)

    logliks = vmap(kalman_loglik_obs, in_axes=(0, 0, 0, None, None, None, 0))(
        errors, r_state, r_obs, m, A_, Q_, b[group_idx]
    )
    numpyro.factor("obs", jnp.sum(logliks))
    numpyro.deterministic("log_likelihood", logliks)


def numpyro_m2(errors, group_idx, n_groups, r_post1, delta_pi,
               m, A_, Q_):
    """M2-dual (M-TWOR)."""
    b          = numpyro.sample("b",          dist.Normal(0.0, 30.0).expand((n_groups,)))
    beta_state = numpyro.sample("beta_state", dist.Normal(0.0, 1.0).expand((n_groups,)))
    beta_obs   = numpyro.sample("beta_obs",   dist.Normal(0.0, 1.0).expand((n_groups,)))
    r_obs_base = numpyro.sample("r_obs_base", dist.HalfNormal(2.0).expand((n_groups,)))

    r_state = r_post1 * jnp.exp(beta_state[group_idx] * delta_pi)
    r_obs   = r_obs_base[group_idx] * jnp.exp(beta_obs[group_idx] * delta_pi)

    logliks = vmap(kalman_loglik_obs, in_axes=(0, 0, 0, None, None, None, 0))(
        errors, r_state, r_obs, m, A_, Q_, b[group_idx]
    )
    numpyro.factor("obs", jnp.sum(logliks))
    numpyro.deterministic("log_likelihood", logliks)


# ── data loading ──────────────────────────────────────────────────────────────
def load_data():
    trials = pd.read_csv(TRIALS_PATH)
    delta  = pd.read_csv(DELTA_PATH)
    delta  = delta[delta["precision_post1"] > 0].copy()
    delta["r_post1"] = 1.0 / delta["precision_post1"]

    merged = (
        trials[["subject", "group"]].drop_duplicates()
        .merge(delta, left_on=["subject", "group"], right_on=["ID", "group"], how="inner")
        .dropna(subset=["r_post1", "delta_log_pi", "openloop_var_post1", "visual_var_post1"])
    )
    subjects = merged[
        ["subject", "group", "r_post1", "delta_log_pi", "openloop_var_post1", "visual_var_post1"]
    ].rename(columns={"delta_log_pi": "delta_pi"}).reset_index(drop=True)
    return subjects


def load_posterior_samples(n_sims: int, rng: np.random.Generator) -> list[dict]:
    """Draw n_sims parameter sets from M3's real posterior."""
    nc_path = POSTERIOR_DIR / "m3_posterior.nc"
    idata   = az.from_netcdf(str(nc_path))
    stacked = idata.posterior.stack(sample=("chain", "draw"))
    n_total = stacked.sizes["sample"]
    idx     = rng.choice(n_total, size=min(n_sims, n_total), replace=False)

    draws = []
    for i in idx:
        d = {}
        for var in stacked.data_vars:
            arr = stacked[var].values
            d[var] = arr[..., i]
        draws.append(d)
    return draws


# ── fitting helper ────────────────────────────────────────────────────────────
def fit_model(model_fn, errors_jnp, group_idx_jnp, n_groups, r_post1_jnp,
              delta_pi_jnp, extra_kwargs, draws, warmup, chains, rng_seed) -> dict:
    kernel = NUTS(model_fn, dense_mass=False, target_accept_prob=0.85, max_tree_depth=10)
    mcmc   = MCMC(kernel, num_warmup=warmup, num_samples=draws,
                  num_chains=chains, chain_method="parallel", progress_bar=False)
    mcmc.run(
        jax.random.PRNGKey(rng_seed),
        errors=errors_jnp, group_idx=group_idx_jnp, n_groups=n_groups,
        r_post1=r_post1_jnp, delta_pi=delta_pi_jnp,
        m=M_TRUE, A_=A, Q_=Q,
        **extra_kwargs,
    )

    import xarray as xr
    coords = {"subject": np.arange(errors_jnp.shape[0])}
    dims   = {"log_likelihood": ["subject"]}
    idata  = az.from_numpyro(mcmc, coords=coords, dims=dims)

    # Move log_likelihood deterministic into log_likelihood group for WAIC
    if "log_likelihood" in idata.posterior:
        ll_data = idata.posterior["log_likelihood"]
        if hasattr(idata, "log_likelihood"):
            idata.log_likelihood["log_likelihood"] = ll_data
        else:
            idata.add_groups({"log_likelihood": xr.Dataset({"log_likelihood": ll_data})})

    try:
        waic_res = az.waic(idata, var_name="log_likelihood")
        waic     = float(waic_res.elpd_waic * -2)
        p_waic   = float(waic_res.p_waic)
    except Exception:
        waic = p_waic = float("nan")

    post = mcmc.get_samples()
    result = {"waic": waic, "p_waic": p_waic}
    for key in ["beta_state", "beta_obs", "r_obs_base", "r_cognitive"]:
        if key in post:
            result[key] = np.median(np.array(post[key]), axis=0)  # (n_groups,)
    return result


# ── main recovery loop ────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sims",  type=int, default=200)
    ap.add_argument("--draws",   type=int, default=500)
    ap.add_argument("--warmup",  type=int, default=500)
    ap.add_argument("--chains",  type=int, default=2)
    args = ap.parse_args()

    print(f"Parameter recovery: {args.n_sims} simulations, "
          f"{args.warmup} warmup + {args.draws} draws, {args.chains} chains")
    print(f"JAX devices: {jax.devices()}")

    subjects = load_data()
    n_subj   = len(subjects)
    group_labels = sorted(subjects["group"].unique())
    g2i      = {g: i for i, g in enumerate(group_labels)}
    gi       = subjects["group"].map(g2i).to_numpy()
    r_post1  = subjects["r_post1"].to_numpy()
    delta_pi = subjects["delta_pi"].to_numpy()
    r_openloop = subjects["openloop_var_post1"].to_numpy()
    r_visual   = subjects["visual_var_post1"].to_numpy()

    rng    = np.random.default_rng(SEED)
    draws  = load_posterior_samples(args.n_sims, rng)

    rows_m3, rows_m2 = [], []
    sim_key = jax.random.PRNGKey(SEED + 1)

    for sim_i, params in enumerate(draws):
        print(f"\n[Sim {sim_i+1}/{args.n_sims}]")

        # True M3 parameters for this draw
        beta_state_true = params["beta_state"]  # (n_groups,)
        beta_obs_true   = params["beta_obs"]
        r_cog_true      = params["r_cognitive"]
        b_arr           = params["b"]
        b_true          = b_arr if b_arr.ndim == 1 else np.full(len(group_labels), b_arr)

        # Compute per-subject true r_state and r_obs
        r_state_true = r_post1 * np.exp(beta_state_true[gi] * delta_pi)
        r_obs_true   = (r_openloop + r_visual) + r_cog_true[gi] * np.exp(beta_obs_true[gi] * delta_pi)
        r_state_true = np.clip(r_state_true, 1e-8, None)
        r_obs_true   = np.clip(r_obs_true,   1e-8, None)

        # Simulate synthetic dataset
        sim_key, subkey = jax.random.split(sim_key)
        subkeys = jax.random.split(subkey, n_subj)
        errors  = np.stack([
            np.array(simulate_one_subject(subkeys[i], float(r_state_true[i]),
                                          float(r_obs_true[i]), float(b_true[gi[i]])))
            for i in range(n_subj)
        ])  # (n_subj, N_TRIALS)

        errors_jnp   = jnp.asarray(errors)
        gi_jnp       = jnp.asarray(gi)
        r_post1_jnp  = jnp.asarray(r_post1)
        dp_jnp       = jnp.asarray(delta_pi)
        rol_jnp      = jnp.asarray(r_openloop)
        rvis_jnp     = jnp.asarray(r_visual)

        # ── Fit M3 ────────────────────────────────────────────────────────────
        print(f"  Fitting M3...")
        rec_m3 = fit_model(
            numpyro_m3, errors_jnp, gi_jnp, len(group_labels),
            r_post1_jnp, dp_jnp,
            extra_kwargs=dict(r_openloop=rol_jnp, r_visual=rvis_jnp),
            draws=args.draws, warmup=args.warmup, chains=args.chains,
            rng_seed=sim_i * 2
        )

        # ── Fit M2 ────────────────────────────────────────────────────────────
        print(f"  Fitting M2...")
        rec_m2 = fit_model(
            numpyro_m2, errors_jnp, gi_jnp, len(group_labels),
            r_post1_jnp, dp_jnp,
            extra_kwargs={},
            draws=args.draws, warmup=args.warmup, chains=args.chains,
            rng_seed=sim_i * 2 + 1
        )

        # Record results
        for g, grp in enumerate(group_labels):
            rows_m3.append({
                "sim": sim_i, "group": grp,
                "true_beta_state": float(beta_state_true[g]),
                "rec_beta_state":  float(rec_m3.get("beta_state", [np.nan]*len(group_labels))[g]),
                "true_beta_obs":   float(beta_obs_true[g]),
                "rec_beta_obs":    float(rec_m3.get("beta_obs", [np.nan]*len(group_labels))[g]),
                "true_r_cognitive":float(r_cog_true[g]),
                "rec_r_cognitive": float(rec_m3.get("r_cognitive", [np.nan]*len(group_labels))[g]),
                "waic": rec_m3["waic"],
            })
            rows_m2.append({
                "sim": sim_i, "group": grp,
                "true_beta_state": float(beta_state_true[g]),
                "rec_beta_state":  float(rec_m2.get("beta_state", [np.nan]*len(group_labels))[g]),
                "true_beta_obs":   float(beta_obs_true[g]),
                "rec_beta_obs":    float(rec_m2.get("beta_obs", [np.nan]*len(group_labels))[g]),
                # True sensory baseline for comparison (r_cog is not directly in M2)
                "true_r_sensory_base": float(
                    (r_openloop[gi == g] + r_visual[gi == g]).mean()
                ),
                "rec_r_obs_base":  float(rec_m2.get("r_obs_base", [np.nan]*len(group_labels))[g]),
                "waic": rec_m2["waic"],
            })

        # Save checkpoints after every 10 sims
        if (sim_i + 1) % 10 == 0 or sim_i == 0:
            pd.DataFrame(rows_m3).to_csv(OUT_DIR / "param_recovery_m3.csv", index=False)
            pd.DataFrame(rows_m2).to_csv(OUT_DIR / "param_recovery_m2onm3.csv", index=False)
            print(f"  Checkpoint saved ({sim_i+1} sims done)")

    df_m3 = pd.DataFrame(rows_m3)
    df_m2 = pd.DataFrame(rows_m2)
    df_m3.to_csv(OUT_DIR / "param_recovery_m3.csv", index=False)
    df_m2.to_csv(OUT_DIR / "param_recovery_m2onm3.csv", index=False)
    print(f"\nResults saved to {OUT_DIR}")

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n=== PARAMETER RECOVERY SUMMARY ===")
    for grp in GROUP_ORDER:
        sub = df_m3[df_m3["group"] == grp].dropna(subset=["rec_beta_state"])
        true_bs = sub["true_beta_state"].values
        rec_bs  = sub["rec_beta_state"].values
        if len(true_bs) < 3:
            continue
        r, p = stats.pearsonr(true_bs, rec_bs)
        bias = float(np.mean(rec_bs - true_bs))
        rmse = float(np.sqrt(np.mean((rec_bs - true_bs) ** 2)))
        print(f"  M3 β_state [{grp}]: r={r:.3f}  bias={bias:+.3f}  RMSE={rmse:.3f}  (n={len(true_bs)})")

    print()
    for grp in GROUP_ORDER:
        m3_sub = df_m3[df_m3["group"] == grp].dropna(subset=["rec_beta_state"])
        m2_sub = df_m2[df_m2["group"] == grp].dropna(subset=["rec_beta_state"])
        m3_bs = m3_sub["rec_beta_state"].values
        m2_bs = m2_sub["rec_beta_state"].values
        if len(m3_bs) > 0 and len(m2_bs) > 0:
            print(f"  β_state recovered [{grp}]: M3 median={np.median(m3_bs):.3f}  "
                  f"M2 median={np.median(m2_bs):.3f}  "
                  f"(true: {m3_sub['true_beta_state'].median():.3f})")

    print()
    print("=== R_obs_base INFLATION IN M2 ===")
    for grp in GROUP_ORDER:
        sub = df_m2[df_m2["group"] == grp].dropna(subset=["rec_r_obs_base"])
        if len(sub) == 0:
            continue
        true_sens = sub["true_r_sensory_base"].mean()
        rec_base  = sub["rec_r_obs_base"].mean()
        print(f"  {grp}: true sensory base={true_sens:.3f}  M2 R_obs_base={rec_base:.3f}  "
              f"ratio={rec_base/true_sens:.2f}x")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_recovery(df_m3, df_m2, group_labels)


def plot_recovery(df_m3: pd.DataFrame, df_m2: pd.DataFrame, group_labels: list):
    GROUP_COLORS = {"EC": "#2196F3", "EO+": "#FF9800", "EO-": "#4CAF50"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.subplots_adjust(wspace=0.38)

    # ── Panel A: M3 β_state recovery scatter ─────────────────────────────────
    ax = axes[0]
    for grp in GROUP_ORDER:
        sub = df_m3[df_m3["group"] == grp].dropna(subset=["true_beta_state", "rec_beta_state"])
        ax.scatter(sub["true_beta_state"], sub["rec_beta_state"],
                   color=GROUP_COLORS.get(grp, "gray"), alpha=0.6, s=30, label=grp)
    lims = [df_m3["true_beta_state"].min() - 0.1, df_m3["true_beta_state"].max() + 0.1]
    ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("True β_state"); ax.set_ylabel("Recovered β_state")
    ax.set_title("M3 Parameter Recovery\nβ_state", fontweight="bold")
    ax.legend(fontsize=8)

    # ── Panel B: M2 β_state recovery scatter (fitting M3-generated data) ─────
    ax = axes[1]
    for grp in GROUP_ORDER:
        m3s = df_m3[df_m3["group"] == grp].dropna(subset=["true_beta_state"])
        m2s = df_m2[df_m2["group"] == grp].dropna(subset=["rec_beta_state"])
        n   = min(len(m3s), len(m2s))
        if n == 0:
            continue
        ax.scatter(m3s["true_beta_state"].values[:n], m2s["rec_beta_state"].values[:n],
                   color=GROUP_COLORS.get(grp, "gray"), alpha=0.6, s=30, label=grp)
    ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("True β_state (M3 generative)"); ax.set_ylabel("Recovered β_state (M2 fit)")
    ax.set_title("M2 fit to M3-generated data\nβ_state", fontweight="bold")
    ax.legend(fontsize=8)

    # ── Panel C: R_obs_base inflation ─────────────────────────────────────────
    ax = axes[2]
    for j, grp in enumerate(GROUP_ORDER):
        sub = df_m2[df_m2["group"] == grp].dropna(subset=["rec_r_obs_base"])
        if len(sub) == 0:
            continue
        true_base = sub["true_r_sensory_base"].values
        rec_base  = sub["rec_r_obs_base"].values
        ax.scatter(true_base, rec_base,
                   color=GROUP_COLORS.get(grp, "gray"), alpha=0.6, s=30, label=grp)
    lo = min(df_m2["true_r_sensory_base"].min(), df_m2["rec_r_obs_base"].dropna().min()) - 0.1
    hi = max(df_m2["true_r_sensory_base"].max(), df_m2["rec_r_obs_base"].dropna().max()) + 0.5
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, linewidth=1, label="identity")
    ax.set_xlabel("True sensory baseline (r_motor + r_visual)")
    ax.set_ylabel("M2 recovered R_obs_base")
    ax.set_title("M2 R_obs_base Inflation\n(should match identity if unbiased)",
                 fontweight="bold")
    ax.legend(fontsize=8)

    fig.suptitle("Parameter Recovery: M3-generated data", fontsize=13, fontweight="bold")
    out = FIG_DIR / "param_recovery_beta_state.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
