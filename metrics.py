"""
Metric computation from TrajectoryRecords.

Metrics per trial:
  • success       : bool
  • smoothness    : sum of angular deflections along executed path (lower = smoother)
  • min_clearance : minimum distance to any obstacle during execution
  • mission_time  : total wall-clock seconds (planning + execution)
  • replans       : number of replanning events

Aggregate over N trials:
  • success_rate  : fraction of successful trials
  • mean / std of smoothness, clearance, mission_time
"""

import numpy as np


def path_smoothness(positions, dim=2):
    """Sum of angular changes (radians) between consecutive segments."""
    if len(positions) < 3:
        return 0.0
    pts = [np.array(p[:dim]) for p in positions]
    total = 0.0
    for i in range(1, len(pts) - 1):
        v1 = pts[i] - pts[i - 1]
        v2 = pts[i + 1] - pts[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
        total += np.arccos(cos_a)
    return float(total)


def trial_metrics(record, dim=2):
    """Compute per-trial metrics from a TrajectoryRecord."""
    return {
        "success":       record.success,
        "smoothness":    path_smoothness(record.executed_positions, dim),
        "min_clearance": min(record.clearances) if record.clearances else 0.0,
        "mission_time":  record.total_time,
        "replans":       record.replans,
    }


def aggregate(trial_results):
    """Aggregate a list of per-trial metric dicts into summary stats."""
    n = len(trial_results)
    if n == 0:
        return {}

    successes = [t["success"] for t in trial_results]
    success_only = [t for t in trial_results if t["success"]]

    summary = {
        "n_trials":     n,
        "success_rate":  np.mean(successes),
    }

    for key in ("smoothness", "min_clearance", "mission_time", "replans"):
        vals = [t[key] for t in success_only] if success_only else [0]
        summary[f"{key}_mean"] = float(np.mean(vals))
        summary[f"{key}_std"]  = float(np.std(vals))

    return summary


def format_table(results_dict):
    """
    Pretty-print benchmark results.
    results_dict: {(env, difficulty, planner_name): summary_dict, ...}
    """
    header = (f"{'Env':<12} {'Diff':<14} {'Planner':<14} "
              f"{'Succ%':>6} {'Smooth':>8} {'MinClr':>8} "
              f"{'Time(s)':>8} {'Replans':>8}")
    sep = "─" * len(header)
    lines = [sep, header, sep]

    for (env, diff, name), s in sorted(results_dict.items()):
        lines.append(
            f"{env:<12} {diff:<14} {name:<14} "
            f"{s['success_rate']*100:>5.1f}% "
            f"{s['smoothness_mean']:>8.2f} "
            f"{s['min_clearance_mean']:>8.3f} "
            f"{s['mission_time_mean']:>8.2f} "
            f"{s['replans_mean']:>8.1f}"
        )
    lines.append(sep)
    return "\n".join(lines)


def save_csv(results_dict, path="benchmark_results.csv"):
    """Write benchmark results to CSV."""
    cols = ["env", "difficulty", "planner", "n_trials", "success_rate",
            "smoothness_mean", "smoothness_std",
            "min_clearance_mean", "min_clearance_std",
            "mission_time_mean", "mission_time_std",
            "replans_mean", "replans_std"]

    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for (env, diff, name), s in sorted(results_dict.items()):
            row = [env, diff, name] + [str(s.get(c, "")) for c in cols[3:]]
            f.write(",".join(row) + "\n")
    print(f"Results saved to {path}")
