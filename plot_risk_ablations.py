"""
Generate figures for every CSV produced by run_risk_ablations.py.

Usage:
  python plot_risk_ablations.py
  python plot_risk_ablations.py --csv-dir ablations --out-dir ablation_figures
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot all RiskAware ablation CSV files")
    parser.add_argument("--csv-dir", default="ablations",
                        help="Directory containing ablation CSV files")
    parser.add_argument("--out-dir", default="ablation_figures",
                        help="Directory for generated figure folders")
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    out_dir = Path(args.out_dir)

    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {csv_dir}")
        return

    for csv_path in csv_files:
        target_dir = out_dir / csv_path.stem
        target_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "compare_prms.py",
            "--csv",
            str(csv_path),
            "--out",
            str(target_dir),
            "--no-show",
        ]

        print(f"\n=== Plotting {csv_path.name} ===")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)

    print(f"\nDone. Figures written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
