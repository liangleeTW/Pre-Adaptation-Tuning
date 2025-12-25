"""Run sweep using group args derived from proprioceptive data."""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path


ARGS_PATH = Path("data/derived/sweep_args.txt")
RUN_SWEEP = Path("scripts/run_sweep.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sweep with derived group args.")
    parser.add_argument("--metric", choices=["pi", "logpi"], default="pi")
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
