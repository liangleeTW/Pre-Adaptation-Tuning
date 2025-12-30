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


def parse_grid_list(value: str) -> list[list[float]]:
    if not value:
        return []
    sets = []
    for chunk in value.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        sets.append(parse_float_list(chunk))
    return sets


def format_float(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simulation parameter sweep.")
    parser.add_argument("--n-subjects", type=int, default=60)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--models", type=str, default="M0,M1,M2")
    parser.add_argument("--betas", type=str, default="0.0,0.2,0.5,0.8")
    parser.add_argument("--lams", type=str, default="-0.8,-0.5,-0.2,0.0,0.2,0.5,0.8")
    parser.add_argument("--delta-pi-sds", type=str, default="0.5,1.0,1.6")
    parser.add_argument("--rhos", type=str, default="0.0,0.3,0.6")
    parser.add_argument("--plateau-fracs", type=str, default="0.0,0.15")
    parser.add_argument("--plateau-bs", type=str, default="")
    parser.add_argument("--m", type=float, default=None)
    parser.add_argument("--seeds", type=str, default="0,1")
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--r-mean", type=float, default=1.0)
    parser.add_argument("--r-log-sd", type=float, default=0.3)
    parser.add_argument("--outdir", type=str, default="data/sim_sweep")
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
    parser.add_argument("--group-betas-grid", type=str, default="")
    parser.add_argument("--group-lams-grid", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    models = [m.strip().upper() for m in args.models.split(",") if m.strip()]
    betas = parse_float_list(args.betas)
    lams = parse_float_list(args.lams)
    delta_pi_sds = parse_float_list(args.delta_pi_sds)
    rhos = parse_float_list(args.rhos)
    plateau_fracs = parse_float_list(args.plateau_fracs)
    plateau_bs = parse_float_list(args.plateau_bs)
    seeds = parse_int_list(args.seeds)
    if args.n_seeds is not None:
        seeds = list(range(int(args.n_seeds)))

    group_labels = [g.strip() for g in args.group_labels.split(",") if g.strip()]
    group_means = parse_float_list(args.group_delta_pi_means)
    group_sds = parse_float_list(args.group_delta_pi_sds)
    group_weights = parse_float_list(args.group_weights)
    group_betas = parse_float_list(args.group_betas)
    group_lams = parse_float_list(args.group_lams)
    group_betas_grid = parse_grid_list(args.group_betas_grid)
    group_lams_grid = parse_grid_list(args.group_lams_grid)

    if group_labels:
        if not (len(group_labels) == len(group_means) == len(group_sds)):
            raise ValueError("Group labels, means, and sds must have matching lengths.")
        if group_weights and len(group_weights) != len(group_labels):
            raise ValueError("Group weights must match number of group labels.")
        if group_betas and len(group_betas) != len(group_labels):
            raise ValueError("Group betas must match number of group labels.")
        if group_lams and len(group_lams) != len(group_labels):
            raise ValueError("Group lams must match number of group labels.")
        if group_betas_grid and group_betas:
            raise ValueError("Use either group betas or group betas grid, not both.")
        if group_lams_grid and group_lams:
            raise ValueError("Use either group lams or group lams grid, not both.")
        for grid in group_betas_grid:
            if len(grid) != len(group_labels):
                raise ValueError("Each group betas grid entry must match number of group labels.")
        for grid in group_lams_grid:
            if len(grid) != len(group_labels):
                raise ValueError("Each group lams grid entry must match number of group labels.")
    elif group_betas or group_lams or group_betas_grid or group_lams_grid:
        raise ValueError("Group betas/lams require group labels.")

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
                if model == "M1":
                    beta_sets = group_betas_grid or ([group_betas] if group_betas else [None])
                    lam_sets = [None]
                elif model == "M2":
                    beta_sets = [None]
                    lam_sets = group_lams_grid or ([group_lams] if group_lams else [None])
                else:
                    beta_sets = [None]
                    lam_sets = [None]

                for group_betas_run in beta_sets:
                    for group_lams_run in lam_sets:
                        for delta_sd in delta_pi_sds:
                            for rho in rhos:
                                plateau_values = plateau_bs or [
                                    frac * abs(args.m if args.m is not None else ModelParams().m)
                                    for frac in plateau_fracs
                                ]
                                for plateau_b in plateau_values:
                                    for seed in seeds:
                                        rng = np.random.default_rng(seed)
                                        z_r, z_delta = sample_correlated_normals(
                                            args.n_subjects, rho, rng
                                        )
                                        log_r = math.log(args.r_mean) + args.r_log_sd * z_r
                                        r_post1 = np.exp(log_r)

                                        if group_labels:
                                            weights = (
                                                np.array(group_weights)
                                                if group_weights
                                                else np.ones(len(group_labels))
                                            )
                                            weights = weights / weights.sum()
                                            group_sizes = rng.multinomial(args.n_subjects, weights)
                                            group_assignments = []
                                            for label, size in zip(group_labels, group_sizes):
                                                group_assignments.extend([label] * size)
                                            rng.shuffle(group_assignments)
                                            group_index = {label: i for i, label in enumerate(group_labels)}
                                            delta_pi = np.zeros(args.n_subjects)
                                            for i, label in enumerate(group_assignments):
                                                idx = group_index[label]
                                                delta_pi[i] = group_means[idx] + (group_sds[idx] * delta_sd) * z_delta[i]
                                            groups = group_assignments
                                        else:
                                            delta_pi = delta_sd * z_delta
                                            groups = ["all"] * args.n_subjects

                                        params = ModelParams(
                                            m=args.m if args.m is not None else ModelParams().m,
                                            b=plateau_b,
                                        )

                                        subjects = [
                                            SubjectConfig(
                                                subject_id=i,
                                                r_post1=float(r_post1[i]),
                                                delta_pi=float(delta_pi[i]),
                                                model=model,
                                                beta=group_betas_run[group_labels.index(groups[i])]
                                                if group_betas_run
                                                else beta,
                                                lam=group_lams_run[group_labels.index(groups[i])]
                                                if group_lams_run
                                                else lam,
                                                group=groups[i],
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
                                            f"_p{format_float(plateau_b / abs(params.m) if abs(params.m) > 0 else 0.0)}_s{seed}"
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
                                                "group": groups,
                                                "model": model,
                                                "beta": [s.beta for s in subjects],
                                                "lam": [s.lam for s in subjects],
                                                "plateau_b": params.b,
                                                "m": params.m,
                                                "delta_pi_metric": args.delta_pi_metric,
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
                                                "plateau_frac": plateau_b / abs(params.m) if abs(params.m) > 0 else 0.0,
                                                "plateau_b": plateau_b,
                                                "seed": seed,
                                                "n_subjects": args.n_subjects,
                                                "n_trials": args.n_trials,
                                                "r_mean": args.r_mean,
                                                "r_log_sd": args.r_log_sd,
                                                "m": params.m,
                                                "group_labels": ",".join(group_labels) if group_labels else "",
                                                "group_delta_pi_means": ",".join(
                                                    f"{v:.3f}" for v in group_means
                                                )
                                                if group_labels
                                                else "",
                                                "group_delta_pi_sds": ",".join(
                                                    f"{v:.3f}" for v in group_sds
                                                )
                                                if group_labels
                                                else "",
                                                "group_weights": ",".join(
                                                    f"{v:.3f}" for v in group_weights
                                                )
                                                if group_weights
                                                else "",
                                                "group_betas": ",".join(
                                                    f"{v:.3f}" for v in group_betas_run
                                                )
                                                if group_betas_run
                                                else "",
                                                "group_lams": ",".join(
                                                    f"{v:.3f}" for v in group_lams_run
                                                )
                                                if group_lams_run
                                                else "",
                                                "delta_pi_metric": args.delta_pi_metric,
                                                "output_dir": str(run_dir),
                                            }
                                        )

    index_df = pd.DataFrame(index_rows)
    index_df.to_csv(outdir / "index.csv", index=False)
    print(f"Wrote {len(index_rows)} runs to {outdir}")


if __name__ == "__main__":
    main()
