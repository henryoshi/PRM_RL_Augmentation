"""
PRM comparison visualiser.

Reads benchmark_results.csv (produced by benchmark.py) and generates
a set of publication-ready comparison figures. All trial logic lives in
benchmark.py — this script only post-processes the output CSV.

Usage:
    python compare_prms.py
    python compare_prms.py --csv my_results.csv --out figures/
"""

import argparse
import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps


# ── Data loading ─────────────────────────────────────────────────────

def load_csv(path):
    """Return list of row dicts; convert numeric fields to float."""
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for key in row:
                if key not in ("env", "difficulty", "planner"):
                    try:
                        row[key] = float(row[key])
                    except (ValueError, TypeError):
                        pass
            rows.append(row)
    return rows


def _scenarios(rows):
    """Sorted unique (env, difficulty) pairs."""
    return sorted({(r["env"], r["difficulty"]) for r in rows})


def _planners(rows):
    """Sorted unique planner names."""
    return sorted({r["planner"] for r in rows})


def _get(rows, env, diff, planner, metric):
    """Look up a single value; return 0.0 if not found."""
    for r in rows:
        if r["env"] == env and r["difficulty"] == diff \
                and r["planner"] == planner:
            v = r.get(metric, 0.0)
            return float(v) if v != "" else 0.0
    return 0.0


# ── Plot helpers ─────────────────────────────────────────────────────

def _bar_chart(rows, metric, err_metric=None, ylabel="", title="",
               pct=False, out=None, show=True):
    """
    Grouped bar chart: one group per scenario, one bar per planner.
    pct=True multiplies values by 100 (for success_rate display).
    """
    scenarios = _scenarios(rows)
    planners = _planners(rows)

    x = np.arange(len(scenarios))
    width = 0.8 / max(len(planners), 1)
    colors = colormaps["tab10"](np.linspace(0, 0.6, len(planners)))

    fig, ax = plt.subplots(figsize=(max(8, len(scenarios) * 1.4), 5))

    for i, planner in enumerate(planners):
        scale = 100.0 if pct else 1.0
        vals = [_get(rows, e, d, planner, metric) * scale
                for e, d in scenarios]
        errs = ([_get(rows, e, d, planner, err_metric) * scale
                 for e, d in scenarios]
                if err_metric else None)
        offset = (i - (len(planners) - 1) / 2) * width
        ax.bar(x + offset, vals, width * 0.9,
               label=planner, color=colors[i],
               yerr=errs, capsize=3, error_kw={"linewidth": 1})

    ax.set_xticks(x)
    ax.set_xticklabels([f"{e}\n{d}" for e, d in scenarios],
                       fontsize=8, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if out:
        os.makedirs(out, exist_ok=True)
        save_path = os.path.join(out, f"{metric}.png")
        plt.savefig(save_path, dpi=150)
        print(f"  Saved {save_path}")
    if show:
        plt.show()
    plt.close()


def _heatmap(rows, out=None, show=True):
    """Success-rate heatmap: rows = scenarios, cols = planners."""
    scenarios = _scenarios(rows)
    planners = _planners(rows)

    data = np.array([
        [_get(rows, e, d, p, "success_rate") * 100
         for p in planners]
        for e, d in scenarios
    ])

    fig, ax = plt.subplots(figsize=(max(4, len(planners) * 1.8),
                                    max(3, len(scenarios) * 0.9)))
    im = ax.imshow(data, vmin=0, vmax=100, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="Success rate (%)")

    ax.set_xticks(range(len(planners)))
    ax.set_xticklabels(planners, fontsize=9)
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels([f"{e} / {d}" for e, d in scenarios], fontsize=8)
    ax.set_title("Success Rate Heatmap (%)")

    for i in range(len(scenarios)):
        for j in range(len(planners)):
            val = data[i, j]
            txt_color = "black" if 25 < val < 80 else "white"
            ax.text(j, i, f"{val:.0f}%",
                    ha="center", va="center",
                    fontsize=8, color=txt_color)

    plt.tight_layout()

    if out:
        os.makedirs(out, exist_ok=True)
        save_path = os.path.join(out, "success_rate_heatmap.png")
        plt.savefig(save_path, dpi=150)
        print(f"  Saved {save_path}")
    if show:
        plt.show()
    plt.close()


# ── Entry point ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualise PRM benchmark results from CSV")
    parser.add_argument("--csv", default="benchmark_results.csv",
                        help="Path to benchmark CSV (default: benchmark_results.csv)")
    parser.add_argument("--out", default=None,
                        help="Directory to save figures (omit to show only)")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not open interactive plot windows")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: '{args.csv}' not found.")
        print("Run benchmark.py first:  python benchmark.py --planners Basic RiskAware")
        return

    rows = load_csv(args.csv)
    if not rows:
        print("No data found in CSV.")
        return

    planners = _planners(rows)
    scenarios = _scenarios(rows)
    print(f"Loaded {len(rows)} rows — "
          f"{len(planners)} planner(s): {', '.join(planners)}")
    print(f"Scenarios: {', '.join(f'{e}/{d}' for e, d in scenarios)}\n")

    charts = [
        # (metric_key,          err_key,               title,              y-label,            pct)
        ("success_rate",        None,                  "Success Rate",     "Rate (%)",          True),
        ("mission_time_mean",   "mission_time_std",    "Mission Time",     "Seconds",           False),
        ("min_clearance_mean",  "min_clearance_std",   "Min Clearance",    "Clearance (m)",     False),
        ("replans_mean",        "replans_std",         "Replan Count",     "Replans",           False),
        ("smoothness_mean",     "smoothness_std",      "Path Smoothness",  "Smoothness (rad)",  False),
    ]

    for metric, err, title, ylabel, pct in charts:
        _bar_chart(rows, metric, err_metric=err, ylabel=ylabel,
                   title=title, pct=pct, out=args.out,
                   show=(not args.no_show))

    _heatmap(rows, out=args.out, show=(not args.no_show))


if __name__ == "__main__":
    main()
