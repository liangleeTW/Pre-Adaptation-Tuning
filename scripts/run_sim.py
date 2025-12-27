"""Run a minimal simulation sweep and write CSV outputs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prism_sim.model import ModelParams
from prism_sim.simulate import SubjectConfig, simulate_cohort, sample_correlated_normals


def parse_float_list(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def parse_str_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prism adaptation simulations.")
    parser.add_argument("--n-subjects", type=int, default=60)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--model", type=str, default="M0", choices=["M0", "M1", "M2"])
    parser.add_argument("--beta", type=float, default=0.0, help="M1 linear modulation")
    parser.add_argument("--lam", type=float, default=0.0, help="M2 tanh modulation")
    parser.add_argument("--delta-pi-sd", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=0.0)
    parser.add_argument("--r-mean", type=float, default=1.0, help="Mean of R_post1 (log scale)")
    parser.add_argument("--r-log-sd", type=float, default=0.3, help="SD of log(R_post1)")
    parser.add_argument("--plateau-frac", type=float, default=0.0, help="b as fraction of |m|")
    parser.add_argument("--plateau-b", type=float, default=None, help="Absolute plateau bias")
    parser.add_argument("--m", type=float, default=None, help="Perturbation size (signed)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="data/sim")
    parser.add_argument(
        "--delta-pi-metric",
        type=str,
        default="pi",
        choices=["pi", "logpi"],
        help="Interpretation of delta_pi (raw precision or log precision).",
    )
    parser.add_argument("--group-labels", type=str, default="")
    parser.add_argument("--group-delta-pi-means", type=str, default="")
    parser.add_argument("--group-delta-pi-sds", type=str, default="")
    parser.add_argument("--group-weights", type=str, default="")
    parser.add_argument("--group-betas", type=str, default="")
    parser.add_argument("--group-lams", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rng = np.random.default_rng(args.seed)
    z_r, z_delta = sample_correlated_normals(args.n_subjects, args.rho, rng)
    log_r = math.log(args.r_mean) + args.r_log_sd * z_r
    r_post1 = np.exp(log_r)

    group_labels = parse_str_list(args.group_labels)
    group_means = parse_float_list(args.group_delta_pi_means)
    group_sds = parse_float_list(args.group_delta_pi_sds)
    group_weights = parse_float_list(args.group_weights)
    group_betas = parse_float_list(args.group_betas)
    group_lams = parse_float_list(args.group_lams)

    if group_labels:
        if not (len(group_labels) == len(group_means) == len(group_sds)):
            raise ValueError("Group labels, means, and sds must have matching lengths.")
        if group_weights and len(group_weights) != len(group_labels):
            raise ValueError("Group weights must match number of group labels.")
        if group_betas and len(group_betas) != len(group_labels):
            raise ValueError("Group betas must match number of group labels.")
        if group_lams and len(group_lams) != len(group_labels):
            raise ValueError("Group lams must match number of group labels.")
    elif group_betas or group_lams:
        raise ValueError("Group betas/lams require group labels.")
        weights = np.array(group_weights) if group_weights else np.ones(len(group_labels))
        weights = weights / weights.sum()
        group_sizes = rng.multinomial(args.n_subjects, weights)
        group_assignments = []
        for label, size in zip(group_labels, group_sizes):
            group_assignments.extend([label] * size)
        rng.shuffle(group_assignments)
        delta_pi = np.zeros(args.n_subjects)
        group_index = {label: i for i, label in enumerate(group_labels)}
        for i, label in enumerate(group_assignments):
            idx = group_index[label]
            delta_pi[i] = group_means[idx] + (group_sds[idx] * args.delta_pi_sd) * z_delta[i]
        groups = group_assignments
    else:
        delta_pi = args.delta_pi_sd * z_delta
        groups = ["all"] * args.n_subjects

    m_value = args.m if args.m is not None else ModelParams().m
    if args.plateau_b is not None:
        b_value = args.plateau_b
    else:
        b_value = args.plateau_frac * abs(m_value)
    params = ModelParams(m=m_value, b=b_value)

    subjects = [
        SubjectConfig(
            subject_id=i,
            r_post1=float(r_post1[i]),
            delta_pi=float(delta_pi[i]),
            model=args.model,
            beta=group_betas[group_labels.index(groups[i])]
            if group_betas
            else args.beta,
            lam=group_lams[group_labels.index(groups[i])]
            if group_lams
            else args.lam,
            group=groups[i],
        )
        for i in range(args.n_subjects)
    ]

    outputs, r_values = simulate_cohort(subjects, params, args.n_trials, seed=args.seed)

    rows = []
    for cfg, out, r_val in zip(subjects, outputs, r_values):
        for t in range(args.n_trials):
            rows.append(
                {
                    "subject": cfg.subject_id,
                    "trial": t + 1,
                    "error": out["e_obs"][t],
                    "error_true": out["e_true"][t],
                    "state": out["x"][t],
                    "kalman_gain": out["k"][t],
                    "state_var": out["p"][t],
                    "r_post1": cfg.r_post1,
                    "delta_pi": cfg.delta_pi,
                    "r_measure": r_val,
                    "model": cfg.model,
                    "beta": cfg.beta,
                    "lam": cfg.lam,
                    "plateau_b": params.b,
                    "m": params.m,
                    "group": cfg.group,
                    "delta_pi_metric": args.delta_pi_metric,
                }
            )

    trials_df = pd.DataFrame(rows)
    subjects_df = pd.DataFrame(
        {
            "subject": [s.subject_id for s in subjects],
            "r_post1": r_post1,
            "delta_pi": delta_pi,
            "r_measure": r_values,
            "model": args.model,
            "beta": [s.beta for s in subjects],
            "lam": [s.lam for s in subjects],
            "plateau_b": params.b,
            "m": params.m,
            "group": groups,
            "delta_pi_metric": args.delta_pi_metric,
        }
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    trials_path = outdir / "sim_trials.csv"
    subjects_path = outdir / "sim_subjects.csv"

    trials_df.to_csv(trials_path, index=False)
    subjects_df.to_csv(subjects_path, index=False)

    print(f"Wrote {trials_path} and {subjects_path}")


if __name__ == "__main__":
    main()
