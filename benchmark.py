"""
Benchmark runner — cold-start PRM comparison.
 
Each trial:
  1. Fresh environment + empty roadmap  (cold start)
  2. Build PRM from scratch             (timed)
  3. Find path                          (timed)
  4. Execute with noise + dynamic obs   (timed, metrics recorded)
 
Usage:
  python benchmark.py                          # all envs, all planners, diff=3
  python benchmark.py --env office --diff 2    # single scenario
  python benchmark.py --trials 10              # quick test
  python benchmark.py --planners Basic         # subset of planners
"""
 
import argparse
import time
from typing import Any
import numpy as np
import pybullet as p
 
from environments import ENV_BUILDERS, DIFFICULTY_LABELS
from prm_basic import BasicPRM
from prm_risk_aware import RiskAwarePRM
from prm_rl import PRMRL
from rl.local_policy import LocalPolicy, ReactivePolicy
from executor import PathExecutor
from metrics import trial_metrics, aggregate, format_table, save_csv
 
 
# ── Planner registry ────────────────────────────────────────────────
# Add your own planners here.  Each must subclass PRMBase (see prm_base.py)
# and implement construct() and find().
PLANNER_CLASSES = {
    "Basic":    BasicPRM,
    "RiskAware": RiskAwarePRM,
    "PRM-RL":   PRMRL,
}
 
 
def make_planner(name: str, env: dict[str, Any], policy_path=None):
    cls = PLANNER_CLASSES[name]
 
    if name == "PRM-RL":
        if policy_path:
            policy = LocalPolicy(policy_path)
        else:
            policy = ReactivePolicy()
        planner = cls(
            client_id=env["cid"],
            workspace_bounds=env["bounds"],
            robot_radius=env["robot_radius"],
            dim=env["dim"],
            policy=policy,
        )
    elif name == "RiskAware":
        planner = cls(
            client_id=env["cid"],
            workspace_bounds=env["bounds"],
            robot_radius=env["robot_radius"],
            dim=env["dim"],
            risk_beta=env.get("risk_beta", 0.5),
            frontier_frac=env.get("frontier_frac", 0.3),
            frontier_sigma=env.get("frontier_sigma", 0.8),
            repair_samples=env.get("repair_samples", 50),
        )
    else:
        planner = cls(
            client_id=env["cid"],
            workspace_bounds=env["bounds"],
            robot_radius=env["robot_radius"],
            dim=env["dim"],
        )
 
    all_ids = env["static_ids"] + env["dyn_manager"].body_ids
    planner.set_obstacles(all_ids)
    return planner
 
 
def run_trial(env_name, difficulty, planner_name, gui=False, policy_path=None,
              risk_beta=0.5, frontier_frac=0.3, frontier_sigma=0.8,
              repair_samples=50):
    """Run a single cold-start trial. Returns a per-trial metric dict."""
 
    # 1. Build fresh environment
    builder = ENV_BUILDERS[env_name]
    env: dict[str, Any] = builder(difficulty=difficulty, gui=gui)
    env["risk_beta"] = risk_beta
    env["frontier_frac"] = frontier_frac
    env["frontier_sigma"] = frontier_sigma
    env["repair_samples"] = repair_samples
 
    # 2. Create planner (cold start)
    planner = make_planner(planner_name, env, policy_path=policy_path)

    # Reset dynamic obstacles to t=0
    env["dyn_manager"].reset()

    t0 = time.perf_counter()

    # 3. Construct roadmap on the STATIC environment only.
    #    Dynamic obstacles are handled reactively during execution via replanning.
    #    This matches the cold-start narrative: build a static prior, adapt online.
    planner.set_obstacles(env["static_ids"])
    planner.construct(env["N"], env["dmax"], verbose=False)

    # 4. Find initial path (dynamic obstacles not yet in play)
    path, cost = planner.find(env["start"], env["goal"], env["dmax"])

    planning_dt = time.perf_counter() - t0

    # 5. Execute path, updates obstacle each step
    planner.set_obstacles(env["static_ids"] + env["dyn_manager"].body_ids)
    executor = PathExecutor(planner, env)
    record = executor.execute(path, cost)
    record.planning_time = planning_dt
    record.total_time = time.perf_counter() - t0
 
    # Tear down
    p.disconnect(env["cid"])
 
    return trial_metrics(record, dim=env["dim"])
 
 
def run_benchmark(envs, difficulties, planners, n_trials, gui=False,
                  policy_path=None, risk_beta=0.5, frontier_frac=0.3,
                  frontier_sigma=0.8, repair_samples=50):
    """Run the full benchmark matrix."""
    all_results = {}
 
    total = len(envs) * len(difficulties) * len(planners) * n_trials
    done = 0
 
    for env_name in envs:
        for diff in difficulties:
            diff_label = DIFFICULTY_LABELS[diff]
            for pname in planners:
                key = (env_name, diff_label, pname)
                trials = []
                for trial in range(n_trials):
                    done += 1
                    seed = trial * 1000 + diff * 100 + hash(env_name) % 100
                    np.random.seed(seed)
 
                    result = run_trial(env_name, diff, pname, gui=gui,
                                       policy_path=policy_path,
                                       risk_beta=risk_beta,
                                       frontier_frac=frontier_frac,
                                       frontier_sigma=frontier_sigma,
                                       repair_samples=repair_samples)
                    trials.append(result)
 
                    status = "OK" if result["success"] else "FAIL"
                    print(f"  [{done}/{total}] {env_name}/{diff_label}/"
                          f"{pname} trial {trial+1}: {status}  "
                          f"t={result['mission_time']:.2f}s  "
                          f"clr={result['min_clearance']:.3f}")
 
                all_results[key] = aggregate(trials)
 
    return all_results
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Cold-start PRM benchmark")
    parser.add_argument("--env", nargs="+",
                        choices=list(ENV_BUILDERS.keys()),
                        default=list(ENV_BUILDERS.keys()),
                        help="Environments to test")
    parser.add_argument("--diff", nargs="+", type=int,
                        choices=[0, 1, 2, 3], default=[0, 3],
                        help="Difficulty levels (0=static … 3=noise+dynamic)")
    parser.add_argument("--planners", nargs="+",
                        choices=list(PLANNER_CLASSES.keys()),
                        default=list(PLANNER_CLASSES.keys()),
                        help="Planner algorithms to benchmark")
    parser.add_argument("--trials", type=int, default=50,
                        help="Trials per (env, difficulty, planner)")
    parser.add_argument("--gui", action="store_true",
                        help="PyBullet GUI (useful for single-trial debug)")
    parser.add_argument("--csv", type=str, default="benchmark_results.csv",
                        help="Output CSV path")
    parser.add_argument("--policy", type=str, default=None,
                        help="Path to trained RL model .zip (for PRM-RL)")
    parser.add_argument("--risk-beta", type=float, default=0.5,
                        help="RiskAware: clearance penalty weight")
    parser.add_argument("--frontier-frac", type=float, default=0.3,
                        help="RiskAware: fraction of frontier-biased samples")
    parser.add_argument("--frontier-sigma", type=float, default=0.8,
                        help="RiskAware: Gaussian spread for frontier samples")
    parser.add_argument("--repair-samples", type=int, default=50,
                        help="RiskAware: local samples added during replan")
    args = parser.parse_args()

    if not (0.0 <= args.frontier_frac <= 1.0):
        parser.error("--frontier-frac must be in [0, 1]")
    if args.repair_samples < 0:
        parser.error("--repair-samples must be >= 0")
    if args.risk_beta < 0.0:
        parser.error("--risk-beta must be >= 0")
    if args.frontier_sigma < 0.0:
        parser.error("--frontier-sigma must be >= 0")
 
    print(f"Benchmark: {args.env} × diff {args.diff} × {args.planners}")
    print(f"Trials per cell: {args.trials}\n")
 
    results = run_benchmark(
        args.env, args.diff, args.planners, args.trials, gui=args.gui,
        policy_path=args.policy,
        risk_beta=args.risk_beta,
        frontier_frac=args.frontier_frac,
        frontier_sigma=args.frontier_sigma,
        repair_samples=args.repair_samples)
 
    save_csv(results, args.csv)
    print("\n" + format_table(results))
 
 
if __name__ == "__main__":
    main()
