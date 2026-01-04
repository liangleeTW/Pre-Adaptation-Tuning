"""Approximate Bayesian fit of real adaptation data with WAIC/LOO."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.stats as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prism_sim.model import ModelParams, compute_r


def parse_models(text: str) -> list[str]:
    return [m.strip().upper() for m in text.split(",") if m.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bayesian fit with WAIC/LOO.")
    parser.add_argument(
        "--trials-path",
        type=Path,
        default=Path("data/derived/adaptation_trials.csv"),
        help="Path to long-format adaptation errors.",
    )
    parser.add_argument(
        "--delta-path",
        type=Path,
        default=Path("data/derived/proprio_delta_pi.csv"),
        help="Path to proprioceptive delta file (pre/post1 precision).",
    )
    parser.add_argument(
        "--metric",
        choices=["pi", "logpi"],
        default="pi",
        help="Use delta_pi or delta_log_pi to modulate R.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="M0,M1,M2",
        help="Comma-separated list of models to fit.",
    )
    parser.add_argument("--fit-plateau", action="store_true", help="Estimate plateau b.")
    parser.add_argument(
        "--plateau-group-specific",
        action="store_true",
        help="Allow group-specific plateau parameters.",
    )
    parser.add_argument("--draws", type=int, default=1500, help="Kept posterior draws.")
    parser.add_argument("--burn", type=int, default=1500, help="Burn-in iterations.")
    parser.add_argument("--thin", type=int, default=2, help="Thinning interval.")
    parser.add_argument("--step-size", type=float, default=0.15, help="RW std dev.")
    parser.add_argument("--m", type=float, default=-12.1, help="Prism shift constant.")
    parser.add_argument("--A", type=float, default=1.0, help="State transition A.")
    parser.add_argument("--Q", type=float, default=1e-4, help="Process noise Q.")
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("data/derived/real_fit_bayes.csv"),
        help="Where to write fit summary CSV.",
    )
    parser.add_argument(
        "--pooled",
        action="store_true",
        help="Fit pooled parameters (group-agnostic). Default: on.",
    )
    parser.add_argument(
        "--no-pooled",
        dest="pooled",
        action="store_false",
        help="Disable pooled fits.",
    )
    parser.add_argument(
        "--group-specific",
        action="store_true",
        help="Fit group-specific modulation parameters. Default: on.",
    )
    parser.add_argument(
        "--no-group-specific",
        dest="group_specific",
        action="store_false",
        help="Disable group-specific fits.",
    )
    parser.set_defaults(pooled=True, group_specific=True)
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


def loglik_pointwise(errors: np.ndarray, params: ModelParams, r_measure: float) -> np.ndarray:
    """Per-observation log-likelihood for one subject."""
    a = params.A
    q = params.Q
    h = -1.0
    c = params.m + params.b

    x = 0.0
    p = 1.0

    ll = np.zeros_like(errors, dtype=float)
    for i, y in enumerate(errors):
        x_pred = a * x
        p_pred = a * p * a + q
        y_pred = h * x_pred + c
        s = h * p_pred * h + r_measure

        v = y - y_pred
        ll[i] = -0.5 * (math.log(2.0 * math.pi * s) + (v * v) / s)

        k = p_pred * h / s
        x = x_pred + k * v
        p = (1.0 - k * h) * p_pred
    return ll


def log_prior_beta(value: float) -> float:
    return st.norm.logpdf(value, loc=0.0, scale=1.0)


def log_prior_lam(z_value: float, scale: float = 0.5) -> tuple[float, float]:
    lam = math.tanh(z_value)
    log_jac = math.log(1.0 - lam * lam + 1e-12)
    return st.norm.logpdf(lam, loc=0.0, scale=scale) + log_jac, lam


def log_prior_b(value: float) -> float:
    return st.norm.logpdf(value, loc=0.0, scale=30.0)


def run_mcmc(
    model: str,
    subjects: pd.DataFrame,
    trials: pd.DataFrame,
    errors_by_subject: dict[str, np.ndarray],
    base_params: ModelParams,
    group_specific: bool,
    fit_plateau: bool,
    plateau_group_specific: bool,
    draws: int,
    burn: int,
    thin: int,
    step_size: float,
    rng: np.random.Generator,
):
    group_order = sorted(subjects["group"].unique().tolist()) if group_specific else ["all"]
    mod_count = len(group_order) if model in {"M1", "M2"} else 0
    b_count = 0
    if fit_plateau:
        b_count = len(group_order) if plateau_group_specific else 1

    dim = mod_count + b_count
    theta = np.zeros(dim, dtype=float)

    kept_params: list[dict[str, float]] = []
    kept_loglik: list[np.ndarray] = []

    total_iters = burn + draws * thin
    accept = 0

    def unpack(theta_vec: np.ndarray):
        idx = 0
        if model == "M1":
            betas = theta_vec[idx : idx + mod_count]
            lams = None
        elif model == "M2":
            betas = None
            lams = theta_vec[idx : idx + mod_count]
        else:
            betas = None
            lams = None
        idx += mod_count
        if fit_plateau:
            if plateau_group_specific:
                bs = theta_vec[idx : idx + b_count]
            else:
                bs = np.array([theta_vec[idx]])
        else:
            bs = np.array([base_params.b])
        return betas, lams, bs

    def log_posterior(theta_vec: np.ndarray):
        betas, lams_raw, bs_raw = unpack(theta_vec)

        # Priors
        lp = 0.0
        if betas is not None:
            lp += sum(log_prior_beta(float(b)) for b in betas)
        lam_values = None
        if lams_raw is not None:
            lam_values = []
            for z in lams_raw:
                lp_z, lam_val = log_prior_lam(float(z))
                lam_values.append(lam_val)
                lp += lp_z
        if fit_plateau:
            lp += sum(log_prior_b(float(b)) for b in bs_raw)

        # Likelihood
        group_betas = {}
        group_lams = {}
        for i, group in enumerate(group_order):
            if model == "M1":
                group_betas[group] = float(betas[i])
            elif model == "M2":
                group_lams[group] = float(lam_values[i])

        if fit_plateau and plateau_group_specific:
            group_bs = {group: float(bs_raw[i]) for i, group in enumerate(group_order)}
        else:
            shared_b = float(bs_raw[0]) if fit_plateau else base_params.b
            group_bs = {group: shared_b for group in group_order}

        total_loglik = 0.0
        pointwise_list = []
        for _, row in subjects.iterrows():
            subj = row["subject"]
            group = row["group"] if group_specific else "all"
            r_post1 = float(row["r_post1"])
            delta_pi = float(row["delta_pi"])
            beta = group_betas.get(group, 0.0)
            lam = group_lams.get(group, 0.0)
            b_val = group_bs[group]

            params = ModelParams(A=base_params.A, Q=base_params.Q, m=base_params.m, b=b_val)
            r_measure = compute_r(model, r_post1, delta_pi, beta, lam)
            ll = loglik_pointwise(errors_by_subject[subj], params, r_measure)
            pointwise_list.append(ll)
            total_loglik += float(ll.sum())

        pointwise = np.concatenate(pointwise_list)
        return lp + total_loglik, total_loglik, pointwise, group_betas, group_lams, group_bs

    current_lp, current_ll, current_pointwise, _, _, _ = log_posterior(theta)
    for it in range(total_iters):
        proposal = theta + rng.normal(scale=step_size, size=dim)
        prop_lp, prop_ll, prop_pointwise, _, _, _ = log_posterior(proposal)
        log_alpha = prop_lp - current_lp
        if math.log(rng.uniform()) < log_alpha:
            theta = proposal
            current_lp = prop_lp
            current_ll = prop_ll
            current_pointwise = prop_pointwise
            accept += 1

        if it >= burn and (it - burn) % thin == 0:
            _, _, _, gb, gl, gbs = log_posterior(theta)
            kept_params.append(
                {
                    "logpost": current_lp,
                    "loglik": current_ll,
                    "betas": gb,
                    "lams": gl,
                    "bs": gbs,
                }
            )
            kept_loglik.append(current_pointwise.copy())

    accept_rate = accept / total_iters
    return kept_params, np.vstack(kept_loglik), accept_rate, group_order


def compute_waic(loglik: np.ndarray) -> dict[str, float]:
    # loglik shape: (draws, n_obs)
    lppd = float(np.sum(np.log(np.mean(np.exp(loglik), axis=0))))
    p_waic = float(np.sum(np.var(loglik, axis=0, ddof=1)))
    waic = -2.0 * (lppd - p_waic)
    return {"waic": waic, "lppd": lppd, "p_waic": p_waic}


def main() -> None:
    args = parse_args()
    if not args.trials_path.exists():
        raise FileNotFoundError(args.trials_path)
    if not args.delta_path.exists():
        raise FileNotFoundError(args.delta_path)

    trials = pd.read_csv(args.trials_path)
    require_columns(trials, ["subject", "group", "trial", "error"], args.trials_path)
    subjects = prepare_subject_table(args.delta_path, args.metric, trials)

    errors_by_subject = {
        subj: trials[trials["subject"] == subj]["error"].values for subj in trials["subject"].unique()
    }
    base_params = ModelParams(A=args.A, Q=args.Q, m=args.m, b=0.0)

    models = parse_models(args.models)
    fit_flags = []
    if args.pooled:
        fit_flags.append(False)
    if args.group_specific:
        fit_flags.append(True)
    if not fit_flags:
        raise ValueError("At least one of pooled or group-specific fits must be enabled.")

    rng = np.random.default_rng(0)
    rows = []

    for model in models:
        for flag in fit_flags:
            params, loglik, acc_rate, group_order = run_mcmc(
                model=model,
                subjects=subjects,
                trials=trials,
                errors_by_subject=errors_by_subject,
                base_params=base_params,
                group_specific=flag,
                fit_plateau=args.fit_plateau,
                plateau_group_specific=args.plateau_group_specific,
                draws=args.draws,
                burn=args.burn,
                thin=args.thin,
                step_size=args.step_size,
                rng=rng,
            )

            try:
                import arviz as az

                loglik_reshaped = loglik[np.newaxis, :, :]  # chain x draw x obs
                dummy_posterior = np.zeros((1, loglik.shape[0], 1))
                idata = az.from_dict(
                    posterior={"theta": dummy_posterior},
                    log_likelihood={"y": loglik_reshaped},
                )
                loo_res = az.loo(idata, pointwise=False)
                waic_res = az.waic(idata, pointwise=False)
                loo_elpd = float(loo_res.elpd_loo)
                loo_se = float(loo_res.se)
                loo_p = float(loo_res.p_loo)
                waic_val = float(waic_res.waic)
                waic_p = float(waic_res.p_waic)
                lppd = float(waic_p - waic_val / 2.0)
            except ImportError:
                waic_stats = compute_waic(loglik)
                loo_elpd = math.nan
                loo_se = math.nan
                loo_p = math.nan
                waic_val = waic_stats["waic"]
                waic_p = waic_stats["p_waic"]
                lppd = waic_stats["lppd"]

            med_params = {}
            betas_all = []
            lams_all = []
            bs_all = []
            for sample in params:
                if sample["betas"]:
                    betas_all.append(sample["betas"])
                if sample["lams"]:
                    lams_all.append(sample["lams"])
                if sample["bs"]:
                    bs_all.append(sample["bs"])

            def median_param(dicts: list[dict[str, float]], key: str, default: float = math.nan):
                if not dicts:
                    return default
                values = [d.get(key, default) for d in dicts if key in d]
                return float(np.median(values)) if values else default

            for group in group_order:
                med_params[f"beta_{group}"] = median_param(betas_all, group)
                med_params[f"lam_{group}"] = median_param(lams_all, group)
                med_params[f"b_{group}"] = median_param(bs_all, group)

            row = {
                "model": model,
                "group_specific": flag,
                "fit_plateau": args.fit_plateau,
                "plateau_group_specific": args.plateau_group_specific,
                "metric": args.metric,
                "n_subjects": len(subjects),
                "n_trials": len(trials),
                "draws": loglik.shape[0],
                "accept_rate": acc_rate,
                "waic": waic_val,
                "waic_p": waic_p,
                "lppd": lppd,
                "loo_elpd": loo_elpd,
                "loo_se": loo_se,
                "loo_p": loo_p,
            }
            row.update(med_params)
            rows.append(row)

            print(
                f"[{model} | group_specific={flag}] acc={acc_rate:.2f} "
                f"WAIC={waic_val:.1f} LOO elpd={loo_elpd:.1f}"
            )

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_path, index=False)
    print(f"Wrote {args.out_path}")


if __name__ == "__main__":
    main()
