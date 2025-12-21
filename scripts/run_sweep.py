"""Run a parameter sweep over simulation settings."""

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


def parse_int_list(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def format_float(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simulation parameter sweep.")
    parser.add_argument("--n-subjects", type=int, default=60)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--models", type=str, default="M0,M1,M2")
    parser.add_argument("--betas", type=str, default="0.0,0.2,0.5,0.8")
    parser.add_argument("--lams", type=str, default="0.0,0.2,0.5,0.8")
    parser.add_argument("--delta-pi-sds", type=str, default="0.5,1.0,1.6")
    parser.add_argument("--rhos", type=str, default="0.0,0.3,0.6")
    parser.add_argument("--plateau-fracs", type=str, default="0.0,0.15")
    parser.add_argument("--seeds", type=str, default="0,1")
    parser.add_argument("--r-mean", type=float, default=1.0)
    parser.add_argument("--r-log-sd", type=float, default=0.3)
    parser.add_argument("--outdir", type=str, default="data/sim_sweep")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    models = [m.strip().upper() for m in args.models.split(",") if m.strip()]
    betas = parse_float_list(args.betas)
    lams = parse_float_list(args.lams)
    delta_pi_sds = parse_float_list(args.delta_pi_sds)
    rhos = parse_float_list(args.rhos)
    plateau_fracs = parse_float_list(args.plateau_fracs)
    seeds = parse_int_list(args.seeds)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict[str, object]] = []
    run_id = 0

    for model in models:
        if model == "M0":
            model_betas = [0.0]
            model_lams = [0.0]
        elif model == "M1":
            model_betas = betas
            model_lams = [0.0]
        elif model == "M2":
            model_betas = [0.0]
            model_lams = lams
        else:
            raise ValueError(f"Unknown model '{model}'. Use M0, M1, or M2.")

        for beta in model_betas:
            for lam in model_lams:
                for delta_sd in delta_pi_sds:
                    for rho in rhos:
                        for plateau in plateau_fracs:
                            for seed in seeds:
                                rng = np.random.default_rng(seed)
                                z_r, z_delta = sample_correlated_normals(
                                    args.n_subjects, rho, rng
                                )
                                log_r = math.log(args.r_mean) + args.r_log_sd * z_r
                                r_post1 = np.exp(log_r)
                                delta_pi = delta_sd * z_delta

                                params = ModelParams(
                                    b=plateau * abs(ModelParams().m),
                                )

                                subjects = [
                                    SubjectConfig(
                                        subject_id=i,
                                        r_post1=float(r_post1[i]),
                                        delta_pi=float(delta_pi[i]),
                                        model=model,
                                        beta=beta,
                                        lam=lam,
                                    )
                                    for i in range(args.n_subjects)
                                ]

                                outputs, r_values = simulate_cohort(
                                    subjects, params, args.n_trials, seed=seed
                                )

                                run_id += 1
                                run_name = (
                                    f"{model}_b{format_float(beta)}_l{format_float(lam)}"
                                    f"_d{format_float(delta_sd)}_r{format_float(rho)}"
                                    f"_p{format_float(plateau)}_s{seed}"
                                )
                                run_dir = outdir / f"run_{run_id:04d}_{run_name}"
                                run_dir.mkdir(parents=True, exist_ok=True)

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
                                        "model": model,
                                        "beta": beta,
                                        "lam": lam,
                                        "plateau_b": params.b,
                                    }
                                )

                                trials_df.to_csv(run_dir / "sim_trials.csv", index=False)
                                subjects_df.to_csv(run_dir / "sim_subjects.csv", index=False)

                                index_rows.append(
                                    {
                                        "run_id": run_id,
                                        "run_name": run_name,
                                        "model": model,
                                        "beta": beta,
                                        "lam": lam,
                                        "delta_pi_sd": delta_sd,
                                        "rho": rho,
                                        "plateau_frac": plateau,
                                        "seed": seed,
                                        "n_subjects": args.n_subjects,
                                        "n_trials": args.n_trials,
                                        "r_mean": args.r_mean,
                                        "r_log_sd": args.r_log_sd,
                                        "output_dir": str(run_dir),
                                    }
                                )

    index_df = pd.DataFrame(index_rows)
    index_df.to_csv(outdir / "index.csv", index=False)
    print(f"Wrote {len(index_rows)} runs to {outdir}")


if __name__ == "__main__":
    main()
