"""Fit simulated runs to recover beta/lambda parameters."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prism_sim.fit import loglik_subject, stack_errors
from prism_sim.model import ModelParams, compute_r


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit simulated sweep runs for recovery.")
    parser.add_argument("--sweep-dir", type=str, default="data/sim_sweep")
    parser.add_argument("--fit-model", type=str, default="auto", choices=["auto", "M0", "M1", "M2"])
    parser.add_argument("--group-specific", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--run-ids", type=str, default="")
    parser.add_argument("--out-name", type=str, default="recovery.csv")
    return parser.parse_args()


def parse_run_ids(value: str) -> set[int]:
    if not value:
        return set()
    return {int(x) for x in value.split(",") if x.strip()}


def negative_loglik(
    params_vec: np.ndarray,
    subjects: pd.DataFrame,
    errors_by_subject: dict[int, np.ndarray],
    model: str,
    base_params: ModelParams,
    group_specific: bool,
    group_order: list[str],
) -> float:
    total = 0.0
    if model == "M0":
        param_map = {}
    elif model == "M1":
        param_map = {
            group: params_vec[i] for i, group in enumerate(group_order)
        } if group_specific else {"all": params_vec[0]}
    elif model == "M2":
        param_map = {
            group: params_vec[i] for i, group in enumerate(group_order)
        } if group_specific else {"all": params_vec[0]}
    else:
        raise ValueError(f"Unknown model '{model}'.")

    for _, row in subjects.iterrows():
        subject_id = int(row["subject"])
        r_post1 = float(row["r_post1"])
        delta_pi = float(row["delta_pi"])
        group = row["group"] if group_specific else "all"

        if model == "M0":
            beta = 0.0
            lam = 0.0
        elif model == "M1":
            beta = float(param_map[group])
            lam = 0.0
        else:
            beta = 0.0
            lam = float(param_map[group])

        r_measure = compute_r(model, r_post1, delta_pi, beta, lam)
        total -= loglik_subject(errors_by_subject[subject_id], base_params, r_measure)

    return total


def fit_run(run_dir: Path, fit_model: str, group_specific: bool) -> dict[str, float]:
    trials = pd.read_csv(run_dir / "sim_trials.csv")
    subjects = pd.read_csv(run_dir / "sim_subjects.csv")

    model = fit_model
    if model == "auto":
        model = str(subjects["model"].iloc[0])

    params = ModelParams(b=float(subjects["plateau_b"].iloc[0]))
    errors_by_subject = stack_errors(trials)

    if group_specific:
        group_order = sorted(subjects["group"].unique().tolist())
    else:
        group_order = ["all"]

    if model == "M0":
        n_params = 0
        result = {"nll": negative_loglik(np.zeros(0), subjects, errors_by_subject, model, params, group_specific, group_order)}
    else:
        n_params = len(group_order)
        x0 = np.zeros(n_params)
        if model == "M2":
            bounds = [(-0.95, 0.95)] * n_params
        else:
            bounds = [(-2.0, 2.0)] * n_params
        opt = minimize(
            negative_loglik,
            x0,
            args=(subjects, errors_by_subject, model, params, group_specific, group_order),
            method="L-BFGS-B",
            bounds=bounds,
        )
        result = {"nll": float(opt.fun)}
        for i, group in enumerate(group_order):
            key = f"{'beta' if model == 'M1' else 'lam'}_{group}"
            result[key] = float(opt.x[i])

    result["model"] = model
    result["group_specific"] = group_specific
    result["n_params"] = n_params
    return result


def main() -> None:
    args = parse_args()
    sweep_dir = Path(args.sweep_dir)
    index = pd.read_csv(sweep_dir / "index.csv")

    run_ids = parse_run_ids(args.run_ids)
    rows = []

    for _, row in index.iterrows():
        run_id = int(row["run_id"])
        if run_ids and run_id not in run_ids:
            continue
        run_dir = Path(row["output_dir"])
        fit = fit_run(run_dir, args.fit_model, args.group_specific)
        fit.update(
            {
                "run_id": run_id,
                "run_name": row["run_name"],
                "true_model": row["model"],
                "true_beta": row["beta"],
                "true_lam": row["lam"],
                "group_labels": row.get("group_labels", ""),
                "delta_pi_metric": row.get("delta_pi_metric", ""),
            }
        )
        rows.append(fit)
        if args.max_runs and len(rows) >= args.max_runs:
            break

    out_path = sweep_dir / args.out_name
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
