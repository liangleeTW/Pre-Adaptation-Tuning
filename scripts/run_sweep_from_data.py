"""Run sweep using group args derived from proprioceptive data."""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd


ARGS_PATH = Path("data/derived/sweep_args.txt")
CALIB_PATH = Path("data/derived/adaptation_calibration.csv")
ADAPT_PATH = Path("data/derived/adaptation_trials.csv")
RUN_SWEEP = Path("scripts/run_sweep.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sweep with derived group args.")
    parser.add_argument("--metric", choices=["pi", "logpi"], default="pi")
    parser.add_argument("--no-calibration", action="store_true", help="Disable auto calibration.")
    parser.add_argument("--dry-run", action="store_true", help="Print command only.")
    parser.add_argument("extra_args", nargs="*", help="Extra args passed to run_sweep.py")
    return parser.parse_args()


def load_args(metric: str) -> str:
    if not ARGS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {ARGS_PATH}. Run scripts/plot_delta_pi_groups.py first."
        )
    lines = [line.strip() for line in ARGS_PATH.read_text(encoding="utf-8").splitlines()]
    label = "delta_pi scale" if metric == "pi" else "delta_log_pi scale"
    for i, line in enumerate(lines):
        if label in line and i + 1 < len(lines):
            return lines[i + 1]
    raise ValueError(f"Could not find args for metric '{metric}'.")


def main() -> None:
    args = parse_args()
    group_args = load_args(args.metric)
    cmd = ["uv", "run", str(RUN_SWEEP), "--delta-pi-metric", args.metric]
    cmd += group_args.split()
    if not args.no_calibration and CALIB_PATH.exists():
        df = pd.read_csv(CALIB_PATH)
        if "group" in df.columns and "early_mean" in df.columns and "late_mean" in df.columns:
            early = df["early_mean"].mean()
            late = df["late_mean"].mean()
            if "--m" not in args.extra_args:
                cmd += ["--m", f"{early:.3f}"]
            if "--plateau-bs" not in args.extra_args and "--plateau-fracs" not in args.extra_args:
                cmd += ["--plateau-bs", f"0.0,{late:.3f}"]
    elif not args.no_calibration and ADAPT_PATH.exists():
        df = pd.read_csv(ADAPT_PATH)
        if {"group", "trial", "error"}.issubset(df.columns):
            max_trial = df["trial"].max()
            early = df[df["trial"] <= 10]
            late = df[df["trial"] > (max_trial - 10)]
            summary = (
                df.groupby("group")["error"]
                .agg(mean_error="mean", sd_error="std")
                .reset_index()
            )
            summary["early_mean"] = early.groupby("group")["error"].mean().values
            summary["late_mean"] = late.groupby("group")["error"].mean().values
            CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
            summary.to_csv(CALIB_PATH, index=False)
            early_mean = summary["early_mean"].mean()
            late_mean = summary["late_mean"].mean()
            if "--m" not in args.extra_args:
                cmd += ["--m", f"{early_mean:.3f}"]
            if "--plateau-bs" not in args.extra_args and "--plateau-fracs" not in args.extra_args:
                cmd += ["--plateau-bs", f"0.0,{late_mean:.3f}"]
    extra_args = list(args.extra_args)
    if "--outdir" not in extra_args:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extra_args = ["--outdir", f"data/sim_sweep_{stamp}"] + extra_args
    cmd += extra_args
    if args.dry_run:
        print(" ".join(cmd))
        return
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
