"""
Pillar 1 — Model Recovery
==========================
Simulate data from M3's posterior. Fit both M2 and M3 to each synthetic dataset.
Show that WAIC cannot reliably distinguish M2 from M3 at N=72 — and that the
observed real-data ΔWAIC (~61) is consistent with M2-wins-even-when-M3-is-true.

Also simulate from M2 and fit both, to show the confusion is asymmetric
(M3-generated data is harder for WAIC to identify than M2-generated data).

Runtime: ~2-4 hrs for --n-sims 100 (recommended). Run overnight.
Command:
    poetry run python scripts/recovery/model_recovery.py \
        --n-sims 100 --draws 500 --warmup 500 --chains 2
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

M_TRUE   = -12.1
A        = 1.0
Q        = 1e-4
N_TRIALS = 100
SEED     = 99
GROUP_ORDER = ["EC", "EO+", "EO-"]

POSTERIOR_DIR = ROOT / "data" / "posteriors"
TRIALS_PATH   = ROOT / "data" / "adaptation_trials.csv"
DELTA_PATH    = ROOT / "data" / "proprio_delta_pi.csv"
OUT_DIR       = ROOT / "data" / "recovery"
FIG_DIR       = ROOT / "figures" / "recovery"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Real-data WAIC values for reference line on plots
REAL_WAIC_M2 = 34974.934
REAL_WAIC_M3 = 35036.417
REAL_DELTA   = REAL_WAIC_M3 - REAL_WAIC_M2   # ~61, positive means M2 wins


# ── Kalman simulator (matches fitting code exactly) ───────────────────────────
@jit
def simulate_one_subject(key, r_state: float, r_obs: float, b: float) -> jnp.ndarray:
    def step(carry, _):
        x, p, rng = carry
        rng, k1, k2 = jax.random.split(rng, 3)
        x_pred = A * x
        p_pred = A * p * A + Q
        s_kalman = p_pred + r_state
        proc  = jax.random.normal(k1) * jnp.sqrt(Q)
        x_true = x_pred + proc
        total_noise = jax.random.normal(k2) * jnp.sqrt(r_state + r_obs)
        y_t = -x_true + M_TRUE + b + total_noise
        v = y_t - (-x_pred + M_TRUE + b)
        k = -p_pred / s_kalman
        x_new = x_pred + k * v
        p_new = (1.0 + k) * p_pred
        return (x_new, p_new, rng), y_t
    _, errors = lax.scan(step, (0.0, 1.0, key), None, length=N_TRIALS)
    return errors


# ── Kalman log-likelihood ─────────────────────────────────────────────────────
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


# ── NUMPYro models ────────────────────────────────────────────────────────────
def numpyro_m3(errors, group_idx, n_groups, r_post1, delta_pi,
               r_openloop, r_visual, m, A_, Q_):
    b          = numpyro.sample("b",          dist.Normal(0.0, 30.0).expand((n_groups,)))
    beta_state = numpyro.sample("beta_state", dist.Normal(0.0, 1.0).expand((n_groups,)))
    beta_obs   = numpyro.sample("beta_obs",   dist.Normal(0.0, 1.0).expand((n_groups,)))
    r_cognitive= numpyro.sample("r_cognitive",dist.HalfNormal(2.0).expand((n_groups,)))
    r_state = r_post1 * jnp.exp(beta_state[group_idx] * delta_pi)
    r_obs   = (r_openloop + r_visual) + r_cognitive[group_idx] * jnp.exp(beta_obs[group_idx] * delta_pi)
    logliks = vmap(kalman_loglik_obs, in_axes=(0, 0, 0, None, None, None, 0))(
        errors, r_state, r_obs, m, A_, Q_, b[group_idx])
    numpyro.factor("obs", jnp.sum(logliks))
    numpyro.deterministic("log_likelihood", logliks)


def numpyro_m2(errors, group_idx, n_groups, r_post1, delta_pi,
               m, A_, Q_):
    b          = numpyro.sample("b",          dist.Normal(0.0, 30.0).expand((n_groups,)))
    beta_state = numpyro.sample("beta_state", dist.Normal(0.0, 1.0).expand((n_groups,)))
    beta_obs   = numpyro.sample("beta_obs",   dist.Normal(0.0, 1.0).expand((n_groups,)))
    r_obs_base = numpyro.sample("r_obs_base", dist.HalfNormal(2.0).expand((n_groups,)))
    r_state = r_post1 * jnp.exp(beta_state[group_idx] * delta_pi)
    r_obs   = r_obs_base[group_idx] * jnp.exp(beta_obs[group_idx] * delta_pi)
    logliks = vmap(kalman_loglik_obs, in_axes=(0, 0, 0, None, None, None, 0))(
        errors, r_state, r_obs, m, A_, Q_, b[group_idx])
    numpyro.factor("obs", jnp.sum(logliks))
    numpyro.deterministic("log_likelihood", logliks)


# ── data loading ──────────────────────────────────────────────────────────────
def load_subjects():
    trials = pd.read_csv(TRIALS_PATH)
    delta  = pd.read_csv(DELTA_PATH)
    delta  = delta[delta["precision_post1"] > 0].copy()
    delta["r_post1"] = 1.0 / delta["precision_post1"]
    merged = (
        trials[["subject", "group"]].drop_duplicates()
        .merge(delta, left_on=["subject", "group"], right_on=["ID", "group"], how="inner")
        .dropna(subset=["r_post1", "delta_log_pi", "openloop_var_post1", "visual_var_post1"])
    )
    return merged[
        ["subject", "group", "r_post1", "delta_log_pi", "openloop_var_post1", "visual_var_post1"]
    ].rename(columns={"delta_log_pi": "delta_pi"}).reset_index(drop=True)


def load_posterior_draws(model_name: str, n: int, rng: np.random.Generator) -> list[dict]:
    idata   = az.from_netcdf(str(POSTERIOR_DIR / f"{model_name}_posterior.nc"))
    stacked = idata.posterior.stack(sample=("chain", "draw"))
    n_total = stacked.sizes["sample"]
    idx     = rng.choice(n_total, size=min(n, n_total), replace=False)
    draws   = []
    for i in idx:
        d = {var: stacked[var].values[..., i] for var in stacked.data_vars}
        draws.append(d)
    return draws


# ── fitting helper ────────────────────────────────────────────────────────────
def compute_waic(mcmc, n_subj):
    import xarray as xr
    coords = {"subject": np.arange(n_subj)}
    dims   = {"log_likelihood": ["subject"]}
    idata  = az.from_numpyro(mcmc, coords=coords, dims=dims)
    if "log_likelihood" in idata.posterior:
        ll_data = idata.posterior["log_likelihood"]
        if hasattr(idata, "log_likelihood"):
            idata.log_likelihood["log_likelihood"] = ll_data
        else:
            idata.add_groups({"log_likelihood": xr.Dataset({"log_likelihood": ll_data})})
    try:
        res  = az.waic(idata, var_name="log_likelihood")
        return float(res.elpd_waic * -2)
    except Exception:
        return float("nan")


def fit_and_waic(model_fn, rng_seed, draws, warmup, chains, **model_kwargs) -> float:
    kernel = NUTS(model_fn, dense_mass=False, target_accept_prob=0.85, max_tree_depth=10)
    mcmc   = MCMC(kernel, num_warmup=warmup, num_samples=draws,
                  num_chains=chains, chain_method="parallel", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(rng_seed), **model_kwargs)
    n_subj = model_kwargs["errors"].shape[0]
    return compute_waic(mcmc, n_subj)


# ── simulate from one draw ────────────────────────────────────────────────────
def simulate_from_m3(params, subjects, gi, rng_seed):
    n_subj     = len(subjects)
    group_labels = sorted(subjects["group"].unique())
    r_post1    = subjects["r_post1"].to_numpy()
    delta_pi   = subjects["delta_pi"].to_numpy()
    r_openloop = subjects["openloop_var_post1"].to_numpy()
    r_visual   = subjects["visual_var_post1"].to_numpy()

    beta_state = params["beta_state"]
    beta_obs   = params["beta_obs"]
    r_cog      = params["r_cognitive"]
    b_arr      = params["b"]
    b_subj     = b_arr[gi] if b_arr.ndim == 1 else np.full(n_subj, b_arr)

    r_state = np.clip(r_post1 * np.exp(beta_state[gi] * delta_pi), 1e-8, None)
    r_obs   = np.clip((r_openloop + r_visual) + r_cog[gi] * np.exp(beta_obs[gi] * delta_pi), 1e-8, None)

    key     = jax.random.PRNGKey(rng_seed)
    subkeys = jax.random.split(key, n_subj)
    errors  = np.stack([
        np.array(simulate_one_subject(subkeys[i], float(r_state[i]),
                                      float(r_obs[i]), float(b_subj[i])))
        for i in range(n_subj)
    ])
    return errors


def simulate_from_m2(params, subjects, gi, rng_seed):
    n_subj     = len(subjects)
    r_post1    = subjects["r_post1"].to_numpy()
    delta_pi   = subjects["delta_pi"].to_numpy()

    beta_state = params["beta_state"]
    beta_obs   = params["beta_obs"]
    r_obs_base = params["r_obs_base"]
    b_arr      = params["b"]
    b_subj     = b_arr[gi] if b_arr.ndim == 1 else np.full(n_subj, b_arr)

    r_state = np.clip(r_post1 * np.exp(beta_state[gi] * delta_pi), 1e-8, None)
    r_obs   = np.clip(r_obs_base[gi] * np.exp(beta_obs[gi] * delta_pi), 1e-8, None)

    key     = jax.random.PRNGKey(rng_seed)
    subkeys = jax.random.split(key, n_subj)
    errors  = np.stack([
        np.array(simulate_one_subject(subkeys[i], float(r_state[i]),
                                      float(r_obs[i]), float(b_subj[i])))
        for i in range(n_subj)
    ])
    return errors


# ── plotting ──────────────────────────────────────────────────────────────────
def plot_results(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.38)

    # ── Panel A: ΔWAIC when true model is M3 ─────────────────────────────────
    ax = axes[0]
    d_m3 = df[df["true_model"] == "M3"]["delta_waic"].dropna()
    ax.hist(d_m3, bins=25, color="#d62728", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="k", linestyle="--", linewidth=1.5, label="ΔWAIC = 0 (tie)")
    ax.axvline(REAL_DELTA, color="purple", linestyle="-", linewidth=2,
               label=f"Real data ΔWAIC = {REAL_DELTA:.1f}")
    pct_m2_wins = float((d_m3 > 0).mean() * 100)
    ax.text(0.97, 0.97,
            f"M2 wins: {pct_m2_wins:.1f}%\n(n={len(d_m3)} sims)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax.set_xlabel("ΔWAIC (M3 − M2)  [positive = M2 wins]")
    ax.set_ylabel("Count")
    ax.set_title("True model = M3\nHow often does M2 win WAIC?", fontweight="bold")
    ax.legend(fontsize=8)

    # ── Panel B: ΔWAIC when true model is M2 ─────────────────────────────────
    ax = axes[1]
    d_m2 = df[df["true_model"] == "M2"]["delta_waic"].dropna()
    ax.hist(d_m2, bins=25, color="#1f77b4", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="k", linestyle="--", linewidth=1.5, label="ΔWAIC = 0 (tie)")
    ax.axvline(REAL_DELTA, color="purple", linestyle="-", linewidth=2,
               label=f"Real data ΔWAIC = {REAL_DELTA:.1f}")
    pct_m2_wins2 = float((d_m2 > 0).mean() * 100)
    ax.text(0.97, 0.97,
            f"M2 wins: {pct_m2_wins2:.1f}%\n(n={len(d_m2)} sims)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax.set_xlabel("ΔWAIC (M3 − M2)  [positive = M2 wins]")
    ax.set_ylabel("Count")
    ax.set_title("True model = M2\nHow often does M2 win WAIC?", fontweight="bold")
    ax.legend(fontsize=8)

    # ── Panel C: Confusion matrix heatmap ─────────────────────────────────────
    ax = axes[2]
    # Confusion: rows = true model, cols = WAIC winner
    m3_m2_wins  = float((d_m3 > 0).mean())
    m3_m3_wins  = 1.0 - m3_m2_wins
    m2_m2_wins2 = float((d_m2 > 0).mean())
    m2_m3_wins  = 1.0 - m2_m2_wins2

    cm = np.array([[m3_m3_wins, m3_m2_wins],
                   [m2_m3_wins, m2_m2_wins2]])
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["M3 selected", "M2 selected"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["True: M3", "True: M2"])
    ax.set_title("Model Recovery Confusion Matrix\n(WAIC)", fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                    fontsize=14, color="white" if cm[i,j] > 0.5 else "black",
                    fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Model Recovery: Can WAIC Distinguish M2 from M3?",
                 fontsize=13, fontweight="bold")
    out = FIG_DIR / "model_recovery_confusion.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)

    # ── ΔWAIC distribution with real-data overlay ─────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.hist(d_m3, bins=25, color="#d62728", alpha=0.65, label="True model = M3", density=True)
    ax2.hist(d_m2, bins=25, color="#1f77b4", alpha=0.65, label="True model = M2", density=True)
    ax2.axvline(0, color="k", linestyle="--", linewidth=1.5)
    ax2.axvline(REAL_DELTA, color="purple", linestyle="-", linewidth=2.5,
                label=f"Observed real-data ΔWAIC = {REAL_DELTA:.1f}")
    ax2.set_xlabel("ΔWAIC (WAIC_M3 − WAIC_M2)  [positive = M2 wins]", fontsize=11)
    ax2.set_ylabel("Density")
    ax2.set_title("WAIC Difference Distribution vs Real-Data Observation", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    out2 = FIG_DIR / "model_recovery_waic_dist.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out2}")
    plt.close(fig2)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sims",  type=int, default=100,
                    help="Number of synthetic datasets per true model (total = 2x this)")
    ap.add_argument("--draws",   type=int, default=500)
    ap.add_argument("--warmup",  type=int, default=500)
    ap.add_argument("--chains",  type=int, default=2)
    args = ap.parse_args()

    print(f"Model recovery: {args.n_sims} sims × 2 true models")
    print(f"  MCMC: {args.warmup} warmup + {args.draws} draws, {args.chains} chains")
    print(f"JAX devices: {jax.devices()}")
    print(f"Real-data ΔWAIC (M3-M2) = {REAL_DELTA:.1f}  [positive = M2 wins]")

    subjects = load_subjects()
    n_subj   = len(subjects)
    group_labels = sorted(subjects["group"].unique())
    g2i      = {g: i for i, g in enumerate(group_labels)}
    gi       = subjects["group"].map(g2i).to_numpy()

    r_post1_arr  = jnp.asarray(subjects["r_post1"].to_numpy())
    delta_pi_arr = jnp.asarray(subjects["delta_pi"].to_numpy())
    rol_arr      = jnp.asarray(subjects["openloop_var_post1"].to_numpy())
    rvis_arr     = jnp.asarray(subjects["visual_var_post1"].to_numpy())
    gi_arr       = jnp.asarray(gi)
    n_groups     = len(group_labels)

    rng = np.random.default_rng(SEED)

    print(f"\nLoading M3 posterior ({args.n_sims} draws)...")
    m3_draws = load_posterior_draws("m3", args.n_sims, rng)
    print(f"Loading M2 posterior ({args.n_sims} draws)...")
    m2_draws = load_posterior_draws("m2", args.n_sims, rng)

    rows = []

    # ── Section 1: True model = M3 ────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("Section 1: True model = M3, fit M2 and M3")
    print(f"{'='*55}")

    for sim_i, params in enumerate(m3_draws):
        print(f"\n  [M3→ Sim {sim_i+1}/{args.n_sims}]")
        errors = simulate_from_m3(params, subjects, gi, rng_seed=SEED + sim_i)
        errors_jnp = jnp.asarray(errors)

        print(f"    Fitting M3...")
        waic_m3 = fit_and_waic(
            numpyro_m3, rng_seed=sim_i * 10,
            draws=args.draws, warmup=args.warmup, chains=args.chains,
            errors=errors_jnp, group_idx=gi_arr, n_groups=n_groups,
            r_post1=r_post1_arr, delta_pi=delta_pi_arr,
            r_openloop=rol_arr, r_visual=rvis_arr,
            m=M_TRUE, A_=A, Q_=Q,
        )

        print(f"    Fitting M2...")
        waic_m2 = fit_and_waic(
            numpyro_m2, rng_seed=sim_i * 10 + 1,
            draws=args.draws, warmup=args.warmup, chains=args.chains,
            errors=errors_jnp, group_idx=gi_arr, n_groups=n_groups,
            r_post1=r_post1_arr, delta_pi=delta_pi_arr,
            m=M_TRUE, A_=A, Q_=Q,
        )

        delta = waic_m3 - waic_m2   # positive = M2 wins
        winner = "M2" if delta > 0 else "M3"
        print(f"    WAIC M3={waic_m3:.1f}  M2={waic_m2:.1f}  Δ={delta:+.1f}  → {winner} wins")

        rows.append(dict(true_model="M3", sim=sim_i,
                         waic_m3=waic_m3, waic_m2=waic_m2, delta_waic=delta, winner=winner))

        if (sim_i + 1) % 5 == 0 or sim_i == 0:
            pd.DataFrame(rows).to_csv(OUT_DIR / "model_recovery_results.csv", index=False)
            n_m3_done = sim_i + 1
            pct = (np.array([r["delta_waic"] for r in rows if r["true_model"] == "M3"]) > 0).mean()
            print(f"    Checkpoint: {n_m3_done} M3-sims done, M2 wins {pct*100:.1f}% so far")

    # ── Section 2: True model = M2 ────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("Section 2: True model = M2, fit M2 and M3")
    print(f"{'='*55}")

    for sim_i, params in enumerate(m2_draws):
        print(f"\n  [M2→ Sim {sim_i+1}/{args.n_sims}]")
        errors = simulate_from_m2(params, subjects, gi, rng_seed=SEED + 1000 + sim_i)
        errors_jnp = jnp.asarray(errors)

        print(f"    Fitting M3...")
        waic_m3 = fit_and_waic(
            numpyro_m3, rng_seed=sim_i * 10 + 5000,
            draws=args.draws, warmup=args.warmup, chains=args.chains,
            errors=errors_jnp, group_idx=gi_arr, n_groups=n_groups,
            r_post1=r_post1_arr, delta_pi=delta_pi_arr,
            r_openloop=rol_arr, r_visual=rvis_arr,
            m=M_TRUE, A_=A, Q_=Q,
        )

        print(f"    Fitting M2...")
        waic_m2 = fit_and_waic(
            numpyro_m2, rng_seed=sim_i * 10 + 5001,
            draws=args.draws, warmup=args.warmup, chains=args.chains,
            errors=errors_jnp, group_idx=gi_arr, n_groups=n_groups,
            r_post1=r_post1_arr, delta_pi=delta_pi_arr,
            m=M_TRUE, A_=A, Q_=Q,
        )

        delta = waic_m3 - waic_m2
        winner = "M2" if delta > 0 else "M3"
        print(f"    WAIC M3={waic_m3:.1f}  M2={waic_m2:.1f}  Δ={delta:+.1f}  → {winner} wins")

        rows.append(dict(true_model="M2", sim=sim_i,
                         waic_m3=waic_m3, waic_m2=waic_m2, delta_waic=delta, winner=winner))

        if (sim_i + 1) % 5 == 0 or sim_i == 0:
            pd.DataFrame(rows).to_csv(OUT_DIR / "model_recovery_results.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "model_recovery_results.csv", index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("FINAL SUMMARY")
    print(f"{'='*55}")

    d_m3 = df[df["true_model"] == "M3"]["delta_waic"].dropna()
    d_m2 = df[df["true_model"] == "M2"]["delta_waic"].dropna()

    if len(d_m3) > 0:
        pct_m2_wins_when_m3_true = float((d_m3 > 0).mean() * 100)
        pct_rd_in_dist = float((d_m3 >= REAL_DELTA).mean() * 100)
        print(f"\nWhen TRUE model = M3 (n={len(d_m3)}):")
        print(f"  M2 wins WAIC: {pct_m2_wins_when_m3_true:.1f}%")
        print(f"  Median ΔWAIC: {d_m3.median():.1f}")
        print(f"  Real-data ΔWAIC ({REAL_DELTA:.1f}) in upper {pct_rd_in_dist:.1f}% of distribution")
        print(f"  → Real-data gap is {'consistent' if pct_rd_in_dist > 5 else 'unusually large'} with M3 being true")

    if len(d_m2) > 0:
        pct_m2_wins_when_m2_true = float((d_m2 > 0).mean() * 100)
        print(f"\nWhen TRUE model = M2 (n={len(d_m2)}):")
        print(f"  M2 wins WAIC: {pct_m2_wins_when_m2_true:.1f}%")
        print(f"  Median ΔWAIC: {d_m2.median():.1f}")

    print(f"\nReal-data ΔWAIC: M3 − M2 = {REAL_DELTA:.1f} (positive = M2 wins)")

    if len(d_m3) > 0 and len(d_m2) > 0:
        plot_results(df)

    print(f"\nOutputs:")
    print(f"  {OUT_DIR}/model_recovery_results.csv")
    print(f"  {FIG_DIR}/model_recovery_confusion.png")
    print(f"  {FIG_DIR}/model_recovery_waic_dist.png")


if __name__ == "__main__":
    main()
