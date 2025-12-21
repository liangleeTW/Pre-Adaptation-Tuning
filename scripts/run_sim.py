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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="data/sim")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rng = np.random.default_rng(args.seed)
    z_r, z_delta = sample_correlated_normals(args.n_subjects, args.rho, rng)
    log_r = math.log(args.r_mean) + args.r_log_sd * z_r
    r_post1 = np.exp(log_r)
    delta_pi = args.delta_pi_sd * z_delta

    params = ModelParams(b=args.plateau_frac * abs(ModelParams().m))

    subjects = [
        SubjectConfig(
            subject_id=i,
            r_post1=float(r_post1[i]),
            delta_pi=float(delta_pi[i]),
            model=args.model,
            beta=args.beta,
            lam=args.lam,
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
            "beta": args.beta,
            "lam": args.lam,
            "plateau_b": params.b,
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
