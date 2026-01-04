"""PyMC-based Bayesian fit of real adaptation data (M0–M2) with WAIC/LOO."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pytensor import scan

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prism_sim.model import compute_r

# Optional JAX backend
try:
    from pymc.sampling_jax import sample_numpyro_nuts
except ImportError:
    sample_numpyro_nuts = None

try:
    import pytensor
    pytensor.config.cxx = "/usr/bin/clang++"
    pytensor.config.optimizer_excluding = "local_useless_inc_subtensor"
except Exception:
    pass

SAMPLING_JAX_AVAILABLE = False
try:
    from pymc.sampling_jax import sample_numpyro_nuts
    SAMPLING_JAX_AVAILABLE = True
except Exception:
    sample_numpyro_nuts = None

def parse_models(text: str) -> list[str]:
    return [m.strip().upper() for m in text.split(",") if m.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyMC fit with WAIC/LOO.")
    parser.add_argument(
        "--trials-path",
        type=Path,
        default=Path("data/derived/adaptation_trials.csv"),
        help="Long-format adaptation errors.",
    )
    parser.add_argument(
        "--delta-path",
        type=Path,
        default=Path("data/derived/proprio_delta_pi.csv"),
        help="Proprioceptive delta file (pre/post1 precision).",
    )
    parser.add_argument(
        "--metric",
        choices=["pi", "logpi"],
        default="pi",
        help="Use delta_pi or delta_log_pi.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="M0,M1,M2",
        help="Comma-separated list of models to fit.",
    )
    parser.add_argument("--draws", type=int, default=1000, help="Posterior draws.")
    parser.add_argument("--tune", type=int, default=1000, help="Tuning iterations.")
    parser.add_argument("--target-accept", type=float, default=0.9, help="NUTS target accept.")
    parser.add_argument("--chains", type=int, default=2, help="Number of chains.")
    parser.add_argument("--random-seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--m", type=float, default=-12.1, help="Prism shift constant.")
    parser.add_argument("--A", type=float, default=1.0, help="State transition A.")
    parser.add_argument("--Q", type=float, default=1e-4, help="Process noise Q.")
    parser.add_argument("--max-subjects", type=int, default=None, help="Limit number of subjects (for testing).")
    parser.add_argument(
        "--use-jax",
        action="store_true",
        help="Use JAX/NumPyro NUTS sampler instead of PyTensor backend.",
    )
    parser.add_argument(
        "--plateau-group-specific",
        action="store_true",
        help="Use group-specific plateau b instead of shared b.",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("data/derived/real_fit_pymc.csv"),
        help="Where to write WAIC/LOO summary.",
    )
    # Hard-coded choices: plateau included; modulation and plateau are group-specific.
    return parser.parse_args()


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
    delta_df = delta_df.copy()
    delta_df = delta_df[delta_df["precision_post1"] > 0]
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


def kalman_loglik_sym(
    errors_np: np.ndarray,
    r_measure: pt.TensorVariable,
    m: float,
    A: float,
    Q: float,
    b: pt.TensorVariable,
) -> pt.TensorVariable:
    """Symbolic Kalman log-likelihood for one subject (kept in the graph)."""

    y_data = pt.as_tensor_variable(errors_np.astype(np.float64))

    A_t = pt.as_tensor_variable(np.float64(A))
    Q_t = pt.as_tensor_variable(np.float64(Q))
    m_t = pt.as_tensor_variable(np.float64(m))

    def kalman_step(y_t, x_prev, p_prev, ll_prev, r_m, A_v, Q_v, m_v, b_v):
        x_pred = A_v * x_prev
        p_pred = A_v * p_prev * A_v + Q_v
        y_pred = -x_pred + (m_v + b_v)
        s = p_pred + r_m  # h=-1 => h^2=1
        v = y_t - y_pred
        ll_inc = -0.5 * (pt.log(2.0 * np.pi * s) + (v * v) / s)
        k = p_pred / s
        x_new = A_v * x_prev + k * v
        p_new = (1.0 - k) * p_pred
        ll_new = ll_prev + ll_inc
        return x_new, p_new, ll_new

    x0 = pt.as_tensor_variable(np.float64(0.0))
    p0 = pt.as_tensor_variable(np.float64(1.0))
    ll0 = pt.as_tensor_variable(np.float64(0.0))

    outputs, _ = scan(
        fn=kalman_step,
        sequences=[y_data],
        outputs_info=[x0, p0, ll0],
        non_sequences=[r_measure, A_t, Q_t, m_t, b],
        n_steps=y_data.shape[0],
    )
    ll_seq = outputs[2]
    return ll_seq[-1]


def run_model(
    model_name: str,
    errors: np.ndarray,
    group_idx: np.ndarray,
    n_groups: int,
    group_labels: list[str],
    r_post1: np.ndarray,
    delta_pi: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, float]:
    n_subjects, n_trials = errors.shape
    with pm.Model() as model:
        beta = None
        lam = None
        # Group-specific modulation only
        if model_name == "M1":
            beta = pm.Normal("beta", 0.0, 1.0, shape=n_groups)
        elif model_name == "M2":
            lam_raw = pm.Normal("lam_raw", 0.0, 0.5, shape=n_groups)
            lam = pm.Deterministic("lam", pm.math.tanh(lam_raw))

        # Plateau mandatory; optionally group-specific
        if args.plateau_group_specific:
            b = pm.Normal("b", 0.0, 30.0, shape=n_groups)
        else:
            b = pm.Normal("b", 0.0, 30.0)

        r_post1_t = pm.Data("r_post1", r_post1)
        delta_t = pm.Data("delta_pi", delta_pi)
        group_idx_t = pm.Data("group_idx", group_idx)

        if model_name == "M0":
            r_measure = pm.math.maximum(r_post1_t, 1e-8)
        elif model_name == "M1":
            beta_subj = beta[group_idx_t]
            r_measure = pm.math.maximum(r_post1_t + beta_subj * delta_t, 1e-8)
        else:
            lam_subj = lam[group_idx_t]
            r_measure = pm.math.maximum(
                r_post1_t * (1.0 - lam_subj * pm.math.tanh(delta_t)), 1e-8
            )

        # Compute per-subject log-likelihoods symbolically (kept small by per-subject scan)
        logps = []
        for i in range(n_subjects):
            b_val = b[group_idx_t[i]] if args.plateau_group_specific else b
            ll_i = kalman_loglik_sym(
                errors[i],
                r_measure[i],
                args.m,
                args.A,
                args.Q,
                b_val,
            )
            logps.append(ll_i)

        logp_subjects = pt.stack(logps)
        pm.Deterministic("log_likelihood", logp_subjects)
        # Add each subject's log-likelihood separately to avoid a giant Add node
        for i, ll_i in enumerate(logps):
            pm.Potential(f"loglik_subj_{i}", ll_i)

        idata_kwargs = {"log_likelihood": {"y": logp_subjects}}

        if args.use_jax:
            if not SAMPLING_JAX_AVAILABLE:
                raise RuntimeError("sample_numpyro_nuts not available. Install a PyMC version with JAX support.")
            idata = sample_numpyro_nuts(
                draws=args.draws,
                tune=args.tune,
                chains=args.chains,
                target_accept=args.target_accept,
                random_seed=args.random_seed,
                postprocessing_backend="cpu",
                idata_kwargs=idata_kwargs,
            )
        else:
            idata = pm.sample(
                draws=args.draws,
                tune=args.tune,
                chains=args.chains,
                target_accept=args.target_accept,
                random_seed=args.random_seed,
                compute_convergence_checks=False,
                progressbar=True,
                return_inferencedata=True,
                idata_kwargs=idata_kwargs,
            )

        waic = pm.waic(idata, scale="deviance")
        try:
            loo = pm.loo(idata, scale="deviance")
        except (IndexError, ValueError) as e:
            # LOO requires multiple draws for importance sampling
            print(f"Warning: LOO computation failed (need more draws): {e}")
            loo = None

    summary = {
        "model": model_name,
        "group_specific": True,
        "fit_plateau": True,
        "plateau_group_specific": args.plateau_group_specific,
        "metric": args.metric,
        "n_subjects": n_subjects,
        "n_trials": n_trials,
        "waic": float(waic.elpd_waic * -2),  # Convert to deviance scale
        "waic_se": float(waic.se * 2),
        "waic_p": float(waic.p_waic),
        "loo": float(loo.elpd_loo * -2) if loo is not None else None,  # Convert to deviance scale
        "loo_se": float(loo.se * 2) if loo is not None else None,
        "loo_p": float(loo.p_loo) if loo is not None else None,
    }

    if beta is not None:
        beta_post = idata.posterior["beta"].stack(sample=("chain", "draw")).median(dim="sample").values
        if beta_post.ndim == 0:
            summary["beta_all"] = float(beta_post)
        else:
            for g, val in enumerate(beta_post):
                summary[f"beta_{group_labels[g]}"] = float(val)
    if lam is not None:
        lam_post = idata.posterior["lam"].stack(sample=("chain", "draw")).median(dim="sample").values
        if lam_post.ndim == 0:
            summary["lam_all"] = float(lam_post)
        else:
            for g, val in enumerate(lam_post):
                summary[f"lam_{group_labels[g]}"] = float(val)
    b_post = idata.posterior["b"].stack(sample=("chain", "draw")).median(dim="sample").values
    if b_post.ndim == 0:
        summary["b"] = float(b_post)
    else:
        for g, val in enumerate(b_post):
            summary[f"b_{group_labels[g]}"] = float(val)

    return summary


def main() -> None:
    args = parse_args()
    if not args.trials_path.exists():
        raise FileNotFoundError(args.trials_path)
    if not args.delta_path.exists():
        raise FileNotFoundError(args.delta_path)

    trials = pd.read_csv(args.trials_path)
    require_columns(trials, ["subject", "group", "trial", "error"], args.trials_path)
    subjects = prepare_subject_table(args.delta_path, args.metric, trials)

    # Limit subjects for testing if requested
    if args.max_subjects is not None:
        subjects = subjects.head(args.max_subjects)
        print(f"Limited to {len(subjects)} subjects for testing")

    errors = build_error_matrix(trials, subjects)

    # Encode groups as integers
    group_labels = sorted(subjects["group"].unique())
    group_to_idx = {g: i for i, g in enumerate(group_labels)}
    group_idx = subjects["group"].map(group_to_idx).to_numpy()
    r_post1 = subjects["r_post1"].to_numpy()
    delta_pi = subjects["delta_pi"].to_numpy()

    models = parse_models(args.models)
    fit_flags = [True]  # only group-specific modulation
    rows: list[dict[str, float]] = []
    for model_name in models:
        for flag in fit_flags:
            res = run_model(
                model_name=model_name,
                errors=errors,
                group_idx=group_idx,
                n_groups=len(group_labels),
                group_labels=group_labels,
                r_post1=r_post1,
                delta_pi=delta_pi,
                args=args,
            )
            rows.append(res)
            loo_str = f"{res['loo']:.1f}" if res['loo'] is not None else "N/A"
            print(
                f"Finished {model_name} | group_specific={flag} | "
                f"WAIC={res['waic']:.1f} | LOO={loo_str}"
            )

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_path, index=False)
    print(f"Wrote {args.out_path}")


if __name__ == "__main__":
    main()
