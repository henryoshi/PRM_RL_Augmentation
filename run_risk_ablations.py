"""
Run a predefined RiskAware ablation suite by calling benchmark.py.

This script does not modify any RL files. It only orchestrates benchmark runs
with different RiskAware parameter settings and writes one CSV per run.

Usage:
  python run_risk_ablations.py
  python run_risk_ablations.py --trials 10 --diff 3 --env office cityscape
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ABLATIONS = [
    {
        "name": "riskaware_default",
        "risk_beta": 0.5,
        "frontier_frac": 0.3,
        "frontier_sigma": 0.8,
        "repair_samples": 50,
    },
    {
        "name": "riskaware_no_risk_penalty",
        "risk_beta": 0.0,
        "frontier_frac": 0.3,
        "frontier_sigma": 0.8,
        "repair_samples": 50,
    },
    {
        "name": "riskaware_no_frontier",
        "risk_beta": 0.5,
        "frontier_frac": 0.0,
        "frontier_sigma": 0.8,
        "repair_samples": 50,
    },
    {
        "name": "riskaware_no_local_repair",
        "risk_beta": 0.5,
        "frontier_frac": 0.3,
        "frontier_sigma": 0.8,
        "repair_samples": 0,
    },
]


def build_cmd(args: argparse.Namespace, cfg: dict[str, float | int | str]) -> list[str]:
    csv_name = f"{cfg['name']}.csv"
    cmd = [
        sys.executable,
        "benchmark.py",
        "--env",
        *args.env,
        "--diff",
        *[str(d) for d in args.diff],
        "--planners",
        "Basic",
        "RiskAware",
        "--trials",
        str(args.trials),
        "--csv",
        str(Path(args.out_dir) / csv_name),
        "--risk-beta",
        str(cfg["risk_beta"]),
        "--frontier-frac",
        str(cfg["frontier_frac"]),
        "--frontier-sigma",
        str(cfg["frontier_sigma"]),
        "--repair-samples",
        str(cfg["repair_samples"]),
    ]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RiskAware ablation benchmarks")
    parser.add_argument("--env", nargs="+", default=["office", "cityscape"],
                        help="Environments to evaluate")
    parser.add_argument("--diff", nargs="+", type=int, default=[3],
                        choices=[0, 1, 2, 3],
                        help="Difficulty levels")
    parser.add_argument("--trials", type=int, default=10,
                        help="Trials per benchmark cell")
    parser.add_argument("--out-dir", default="ablations",
                        help="Directory for ablation CSV files")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for cfg in ABLATIONS:
        print(f"\n=== Running: {cfg['name']} ===")
        cmd = build_cmd(args, cfg)
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)

    print(f"\nDone. CSV files written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
