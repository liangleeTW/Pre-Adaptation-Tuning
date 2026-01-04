"""Fit real adaptation data under M0–M2 using maximum likelihood."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prism_sim.fit import loglik_subject, stack_errors
from prism_sim.model import ModelParams, compute_r


def parse_models(text: str) -> list[str]:
    return [m.strip().upper() for m in text.split(",") if m.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit real adaptation data (M0–M2).")
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
    # Plateau is required and group-specific modulation only.
    parser.add_argument(
        "--plateau-bounds",
        type=str,
        default="-25,25",
        help="Bounds for plateau b (min,max).",
    )
    parser.add_argument(
        "--b-init",
        type=float,
        default=None,
        help="Initial plateau guess; defaults to mean of last 10 trials.",
    )
    parser.add_argument("--m", type=float, default=-12.1, help="Prism shift constant.")
    parser.add_argument("--A", type=float, default=1.0, help="State transition A.")
    parser.add_argument("--Q", type=float, default=1e-4, help="Process noise Q.")
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("data/derived/real_fit_results.csv"),
        help="Where to write fit summary CSV.",
    )
    # Hard-coded choices: plateau included; modulation is group-specific only.
    return parser.parse_args()


def require_columns(df: pd.DataFrame, cols: Iterable[str], path: Path) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")


def estimate_plateau(trials: pd.DataFrame, tail: int = 10) -> float:
    max_trial = trials["trial"].max()
    late = trials[trials["trial"] > (max_trial - tail)]
    return float(late["error"].mean())


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


def negative_loglik(
    params_vec: np.ndarray,
    subjects: pd.DataFrame,
    errors_by_subject: dict[str, np.ndarray],
    model: str,
    base_params: ModelParams,
    group_order: list[str],
) -> float:
    idx = 0
    if model == "M0":
        mod_params: dict[str, float] = {}
        mod_count = 0
    else:
        mod_count = len(group_order)
        values = params_vec[idx : idx + mod_count]
        idx += mod_count
        mod_params = {group: values[i] for i, group in enumerate(group_order)}

    plateau = float(params_vec[idx])

    params = ModelParams(A=base_params.A, Q=base_params.Q, m=base_params.m, b=plateau)
    total = 0.0
    for _, row in subjects.iterrows():
        subject_id = row["subject"]
        r_post1 = float(row["r_post1"])
        delta_pi = float(row["delta_pi"])
        group = row["group"]

        if model == "M0":
            beta = 0.0
            lam = 0.0
        elif model == "M1":
            beta = float(mod_params[group])
            lam = 0.0
        else:
            beta = 0.0
            lam = float(mod_params[group])

        r_measure = compute_r(model, r_post1, delta_pi, beta, lam)
        total -= loglik_subject(errors_by_subject[subject_id], params, r_measure)
    return total


def fit_single(
    model: str,
    subjects: pd.DataFrame,
    trials: pd.DataFrame,
    errors_by_subject: dict[str, np.ndarray],
    base_params: ModelParams,
    plateau_bounds: tuple[float, float],
) -> dict[str, float]:
    group_order = sorted(subjects["group"].unique().tolist())
    if model == "M0":
        mod_count = 0
        bounds: list[tuple[float, float]] = []
        x0: list[float] = []
    elif model == "M1":
        mod_count = len(group_order)
        bounds = [(-2.0, 2.0)] * mod_count
        x0 = [0.0] * mod_count
    elif model == "M2":
        mod_count = len(group_order)
        bounds = [(-0.95, 0.95)] * mod_count
        x0 = [0.0] * mod_count
    else:
        raise ValueError(f"Unknown model '{model}'.")

    bounds.append(plateau_bounds)
    x0.append(base_params.b)

    opt = minimize(
        negative_loglik,
        np.array(x0, dtype=float),
        args=(
            subjects,
            errors_by_subject,
            model,
            base_params,
            group_order,
        ),
        method="L-BFGS-B",
        bounds=bounds,
    )
    nll = float(opt.fun)
    params_vec = opt.x

    result = {
        "model": model,
        "group_specific": True,
        "fit_plateau": True,
        "n_params": mod_count + 1,
        "n_subjects": len(subjects),
        "n_trials": len(trials),
        "nll": float(nll),
    }

    idx = 0
    if model in {"M1", "M2"}:
        for i, group in enumerate(group_order):
            key = f"{'beta' if model == 'M1' else 'lam'}_{group}"
            result[key] = float(params_vec[idx + i])
        idx += mod_count
    result["plateau_b"] = float(params_vec[idx])

    # Information criteria
    n_obs = len(trials)
    k = result["n_params"]
    result["aic"] = 2 * k + 2 * nll
    result["bic"] = math.log(n_obs) * k + 2 * nll
    return result


def main() -> None:
    args = parse_args()
    if not args.trials_path.exists():
        raise FileNotFoundError(args.trials_path)
    if not args.delta_path.exists():
        raise FileNotFoundError(args.delta_path)

    trials = pd.read_csv(args.trials_path)
    require_columns(trials, ["subject", "group", "trial", "error"], args.trials_path)
    subjects = prepare_subject_table(args.delta_path, args.metric, trials)
    errors_by_subject = stack_errors(trials, subject_col="subject")

    plateau_bounds = tuple(float(x) for x in args.plateau_bounds.split(","))
    if len(plateau_bounds) != 2:
        raise ValueError("plateau-bounds must be 'min,max'.")

    b_init = args.b_init
    if b_init is None:
        b_init = estimate_plateau(trials)

    base_params = ModelParams(A=args.A, Q=args.Q, m=args.m, b=b_init)

    models = parse_models(args.models)
    results: list[dict[str, float]] = []

    fit_flags = [True]  # only group-specific fits
    for model in models:
        for flag in fit_flags:
            res = fit_single(
                model,
                subjects,
                trials,
                errors_by_subject,
                base_params,
                plateau_bounds=plateau_bounds,
            )
            res["metric"] = args.metric
            results.append(res)
            print(
                f"Fitted {model} | group_specific={flag} | nll={res['nll']:.2f} | "
                f"params={res['n_params']}"
            )

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(args.out_path, index=False)
    print(f"Wrote {args.out_path}")


if __name__ == "__main__":
    main()
