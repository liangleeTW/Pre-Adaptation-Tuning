"""Summarize parameter recovery outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze recovery outputs.")
    parser.add_argument("--sweep-dir", type=str, default="data/sim_sweep")
    parser.add_argument("--infile", type=str, default="recovery.csv")
    parser.add_argument("--outfile", type=str, default="recovery_summary.csv")
    return parser.parse_args()


def summarize_param(df: pd.DataFrame, name: str, true_col: str) -> dict[str, float]:
    est = df[name].to_numpy(dtype=float)
    true = df[true_col].to_numpy(dtype=float)
    err = est - true
    return {
        f"{name}_bias": float(np.nanmean(err)),
        f"{name}_rmse": float(np.sqrt(np.nanmean(err**2))),
        f"{name}_corr": float(np.corrcoef(est, true)[0, 1]) if len(est) > 1 else float("nan"),
        f"{name}_sign_match": float(np.mean(np.sign(est) == np.sign(true))) if len(est) > 0 else float("nan"),
        f"{name}_n": float(len(est)),
    }


def main() -> None:
    args = parse_args()
    sweep_dir = Path(args.sweep_dir)
    in_path = sweep_dir / args.infile
    df = pd.read_csv(in_path)

    rows = []

    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model]
        row = {"model": model}
        if model == "M1":
            params = [c for c in sub.columns if c.startswith("beta_")]
            for name in params:
                row.update(summarize_param(sub, name, "true_beta"))
        elif model == "M2":
            params = [c for c in sub.columns if c.startswith("lam_")]
            for name in params:
                row.update(summarize_param(sub, name, "true_lam"))
        rows.append(row)

    summary = pd.DataFrame(rows)
    out_path = sweep_dir / args.outfile
    summary.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
