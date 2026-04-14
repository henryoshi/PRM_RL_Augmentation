"""
Quick single-scenario runner for interactive testing and visualisation.

Usage:
  python run_single.py --env simple --diff 0 --planner Basic
  python run_single.py --env office --diff 2 --planner Basic --gui
"""

import argparse
import time
from typing import Any
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from environments import ENV_BUILDERS, DIFFICULTY_LABELS
from prm_basic import BasicPRM
from prm_risk_aware import RiskAwarePRM
from prm_rl import PRMRL
from rl.local_policy import LocalPolicy, ReactivePolicy
from executor import PathExecutor, OnlinePathExecutor
from metrics import trial_metrics


# ── Planner registry ────────────────────────────────────────────────
# Add your own planners here.  Each must subclass PRMBase (see prm_base.py)
# and implement construct() and find().
PLANNER_CLASSES = {
    "Basic":        BasicPRM,
    "RiskAware":    RiskAwarePRM,
    "PRM-RL":    PRMRL,
}


def _draw_rects(ax, rects, facecolor='#555555', edgecolor='#333333',
                label=None):
    """Draw obstacle rectangles on a matplotlib axes (top-down view).
    Each rect = (cx, cy, hx, hy, height)."""
    for i, (cx, cy, hx, hy, _h) in enumerate(rects):
        ax.add_patch(patches.Rectangle(
            (cx - hx, cy - hy), 2 * hx, 2 * hy,
            facecolor=facecolor, edgecolor=edgecolor, linewidth=0.6,
            label=label if i == 0 else None))


_DYN_TRAIL_COLORS = ['#e06000', '#9900cc', '#007777', '#cc6600', '#006699']


def _draw_dyn_trails(ax, dyn_positions_log, label_prefix="Dyn obs"):
    """
    Draw the recorded trajectory of each dynamic obstacle.

    dyn_positions_log : list of snapshots
        Each snapshot is a list of (x,y) tuples — one per obstacle per tick.
    """
    if not dyn_positions_log:
        return
    n_obs = len(dyn_positions_log[0])
    for oi in range(n_obs):
        xs = [snap[oi][0] for snap in dyn_positions_log if oi < len(snap)]
        ys = [snap[oi][1] for snap in dyn_positions_log if oi < len(snap)]
        color = _DYN_TRAIL_COLORS[oi % len(_DYN_TRAIL_COLORS)]
        ax.plot(xs, ys, color=color, lw=1.2, alpha=0.65, zorder=3,
                linestyle='--', label=f'{label_prefix} {oi+1}')
        # Mark start and end positions
        if xs:
            ax.plot(xs[0], ys[0], 'o', color=color, ms=5, zorder=4)
            ax.plot(xs[-1], ys[-1], 's', color=color, ms=5, zorder=4)


def plot_plan_2d(env, planner, path, title=""):
    """Top-down view of the roadmap and planned path, before execution."""
    fig, ax = plt.subplots(figsize=(8, 7))
    xmax = env["bounds"][0][1]
    ymax = env["bounds"][1][1]

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_aspect('equal')
    ax.set_facecolor('#f0f0f0')
    ax.set_title(title, fontsize=12)

    _draw_rects(ax, env.get("rects", []),
                facecolor='#555555', edgecolor='#333333', label='Static obstacle')
    _draw_rects(ax, env.get("dyn_rects", []),
                facecolor='#cc3333', edgecolor='#881111', label='Dyn obs (init)')

    for u, v in planner.G.edges:
        ax.plot([u[0], v[0]], [u[1], v[1]],
                color='steelblue', lw=0.25, alpha=0.4)
    xs = [n[0] for n in planner.G.nodes]
    ys = [n[1] for n in planner.G.nodes]
    ax.scatter(xs, ys, s=3, c='steelblue', zorder=3)

    if path:
        px = [wp[0] for wp in path]
        py = [wp[1] for wp in path]
        ax.plot(px, py, color='red', lw=2.2, zorder=5, label='Planned path')
        ax.scatter(px[1:-1], py[1:-1], s=18, c='red', zorder=6)

    s, g = env["start"], env["goal"]
    ax.plot(s[0], s[1], 'go', ms=11, zorder=7, label='Start')
    ax.plot(g[0], g[1], 'b^', ms=11, zorder=7, label='Goal')
    ax.legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig("result_plan.png", dpi=150)
    plt.show()


def plot_plan_3d(env, planner, path, title=""):
    """3D view of roadmap and planned path, before execution."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    buildings = env.get("buildings", [])
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    for (cx, cy, hx, hy, h) in buildings:
        x0, x1 = cx - hx, cx + hx
        y0, y1 = cy - hy, cy + hy
        verts = [
            [(x0,y0,0),(x1,y0,0),(x1,y1,0),(x0,y1,0)],
            [(x0,y0,h),(x1,y0,h),(x1,y1,h),(x0,y1,h)],
            [(x0,y0,0),(x1,y0,0),(x1,y0,h),(x0,y0,h)],
            [(x1,y0,0),(x1,y1,0),(x1,y1,h),(x1,y0,h)],
            [(x1,y1,0),(x0,y1,0),(x0,y1,h),(x1,y1,h)],
            [(x0,y1,0),(x0,y0,0),(x0,y0,h),(x0,y1,h)],
        ]
        ax.add_collection3d(Poly3DCollection(
            verts, alpha=0.3, facecolor='#7a8a9a', edgecolor='#4a5a6a',
            linewidth=0.3))

    edges = list(planner.G.edges)
    step = max(1, len(edges) // 2000)
    for u, v in edges[::step]:
        ax.plot([u[0], v[0]], [u[1], v[1]], [u[2], v[2]],
                color='steelblue', linewidth=0.2, alpha=0.25)

    if path:
        ax.plot([wp[0] for wp in path], [wp[1] for wp in path],
                [wp[2] for wp in path], color='red', lw=2.5, label='Planned path')

    s, g = env["start"], env["goal"]
    ax.scatter(*s, color='green', s=80, zorder=6, label='Start')
    ax.scatter(*g, color='blue',  s=80, zorder=6, label='Goal')
    ax.set_xlim(0, 40); ax.set_ylim(0, 40); ax.set_zlim(0, 25)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(title); ax.legend()
    plt.tight_layout()
    plt.savefig("result_plan_3d.png", dpi=150)
    plt.show()


def plot_2d(env, planner, record, title=""):
    """Top-down matplotlib visualisation for 2D scenarios."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    xmax = env["bounds"][0][1]
    ymax = env["bounds"][1][1]

    static_rects = env.get("rects", [])
    dyn_rects = env.get("dyn_rects", [])

    for ax_idx, ax in enumerate(axes):
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, ymax)
        ax.set_aspect('equal')
        ax.set_facecolor('#f0f0f0')

        # Draw static obstacles
        _draw_rects(ax, static_rects, facecolor='#555555',
                    edgecolor='#333333', label='Static obstacle')
        # Draw dynamic obstacle initial positions (snapshot from env)
        _draw_rects(ax, dyn_rects, facecolor='#cc3333',
                    edgecolor='#881111', label='Dyn obs (init)')

        if ax_idx == 0:
            ax.set_title("PRM Roadmap")
            for u, v in planner.G.edges:
                ax.plot([u[0], v[0]], [u[1], v[1]],
                        color='steelblue', lw=0.25, alpha=0.4)
            xs = [n[0] for n in planner.G.nodes]
            ys = [n[1] for n in planner.G.nodes]
            ax.scatter(xs, ys, s=3, c='steelblue', zorder=3)
        else:
            ax.set_title("Executed Trajectory + Dynamic Obstacle Trails")
            # Draw recorded dynamic obstacle trajectories only on trajectory panel
            _draw_dyn_trails(ax, record.dyn_obstacle_positions)

        if record.executed_positions:
            px = [pt[0] for pt in record.executed_positions]
            py = [pt[1] for pt in record.executed_positions]
            ax.plot(px, py, color='red', lw=1.8, zorder=4, label='Robot path')

        s, g = env["start"], env["goal"]
        ax.plot(s[0], s[1], 'go', ms=10, zorder=5, label='Start')
        ax.plot(g[0], g[1], 'b^', ms=10, zorder=5, label='Goal')
        ax.legend(loc='upper left', fontsize=7)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig("result_single.png", dpi=150)
    plt.show()


def plot_3d(env, planner, record, title=""):
    """3D matplotlib visualisation for cityscape."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    buildings = env.get("buildings", [])
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    for (cx, cy, hx, hy, h) in buildings:
        x0, x1 = cx - hx, cx + hx
        y0, y1 = cy - hy, cy + hy
        verts = [
            [(x0,y0,0),(x1,y0,0),(x1,y1,0),(x0,y1,0)],
            [(x0,y0,h),(x1,y0,h),(x1,y1,h),(x0,y1,h)],
            [(x0,y0,0),(x1,y0,0),(x1,y0,h),(x0,y0,h)],
            [(x1,y0,0),(x1,y1,0),(x1,y1,h),(x1,y0,h)],
            [(x1,y1,0),(x0,y1,0),(x0,y1,h),(x1,y1,h)],
            [(x0,y1,0),(x0,y0,0),(x0,y0,h),(x0,y1,h)],
        ]
        ax.add_collection3d(Poly3DCollection(
            verts, alpha=0.3, facecolor='#7a8a9a', edgecolor='#4a5a6a',
            linewidth=0.3))

    # PRM edges (subsample for clarity)
    edges = list(planner.G.edges)
    step = max(1, len(edges) // 2000)
    for u, v in edges[::step]:
        ax.plot([u[0], v[0]], [u[1], v[1]], [u[2], v[2]],
                color='steelblue', linewidth=0.2, alpha=0.25)

    if record.executed_positions:
        pts = record.executed_positions
        ax.plot([pt[0] for pt in pts], [pt[1] for pt in pts],
                [pt[2] for pt in pts], color='red', lw=2.5, label='Path')

    s, g = env["start"], env["goal"]
    ax.scatter(*s, color='green', s=80, zorder=6, label='Start')
    ax.scatter(*g, color='blue', s=80, zorder=6, label='Goal')
    ax.set_xlim(0, 40); ax.set_ylim(0, 40); ax.set_zlim(0, 25)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(title); ax.legend()
    plt.tight_layout()
    plt.savefig("result_single_3d.png", dpi=150)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="simple",
                        choices=list(ENV_BUILDERS.keys()))
    parser.add_argument("--diff", type=int, default=0, choices=[0,1,2,3])
    parser.add_argument("--planner", default="Basic",
                        choices=list(PLANNER_CLASSES.keys()))
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy", type=str, default=None,
                        help="Path to trained RL model .zip (for PRM-RL). "
                             "If omitted, uses ReactivePolicy fallback.")
    parser.add_argument("--risk-beta", type=float, default=0.5,
                        help="RiskAware: clearance penalty weight")
    parser.add_argument("--frontier-frac", type=float, default=0.3,
                        help="RiskAware: fraction of frontier-biased samples")
    parser.add_argument("--frontier-sigma", type=float, default=0.8,
                        help="RiskAware: Gaussian spread for frontier samples")
    parser.add_argument("--repair-samples", type=int, default=50,
                        help="RiskAware: local samples added during replan")
    parser.add_argument("--min-edge-len", type=float, default=0.0,
                        help="Skip edges shorter than this (m); prunes near-duplicate "
                             "connections when using higher N (e.g. robot_radius=0.2)")
    parser.add_argument("--mode", choices=["offline", "online"], default="offline",
                        help="offline (default): pre-build full roadmap then execute; "
                             "online: discover obstacles incrementally during execution "
                             "(only RiskAware planner supported)")
    parser.add_argument("--online-samples-per-tick", type=int, default=15,
                        help="Online mode: new roadmap nodes added near the robot "
                             "each execution tick (default: 15)")
    args = parser.parse_args()

    if not (0.0 <= args.frontier_frac <= 1.0):
        parser.error("--frontier-frac must be in [0, 1]")
    if args.repair_samples < 0:
        parser.error("--repair-samples must be >= 0")
    if args.risk_beta < 0.0:
        parser.error("--risk-beta must be >= 0")
    if args.frontier_sigma < 0.0:
        parser.error("--frontier-sigma must be >= 0")

    np.random.seed(args.seed)
    env: dict[str, Any] = ENV_BUILDERS[args.env](difficulty=args.diff,
                                                 gui=args.gui)
    if args.min_edge_len > 0.0:
        env["min_edge_len"] = args.min_edge_len  # CLI override wins over env default
    min_edge_len = env.get("min_edge_len", 0.0)
    max_neighbors = env.get("max_neighbors", None)

    cls = PLANNER_CLASSES[args.planner]

    # PRM-RL needs a policy object
    if args.planner == "PRM-RL":
        if args.policy:
            policy = LocalPolicy(args.policy)
        else:
            print("  (No --policy given, using ReactivePolicy fallback)")
            policy = ReactivePolicy()
        planner = cls(client_id=env["cid"], workspace_bounds=env["bounds"],
                      robot_radius=env["robot_radius"], dim=env["dim"],
                      policy=policy, max_neighbors=max_neighbors)
    elif args.planner == "RiskAware":
        planner = cls(client_id=env["cid"], workspace_bounds=env["bounds"],
                      robot_radius=env["robot_radius"], dim=env["dim"],
                      risk_beta=args.risk_beta,
                      frontier_frac=args.frontier_frac,
                      frontier_sigma=args.frontier_sigma,
                      repair_samples=args.repair_samples,
                      min_edge_len=min_edge_len,
                      max_neighbors=max_neighbors)
    else:
        planner = cls(client_id=env["cid"], workspace_bounds=env["bounds"],
                      robot_radius=env["robot_radius"], dim=env["dim"],
                      max_neighbors=max_neighbors)
    if args.mode == "online":
        if args.planner not in ("RiskAware",):
            parser.error("--mode online is only supported with --planner RiskAware")

        # Online mode: register the full world but reveal nothing yet.
        # The executor discovers obstacles incrementally as the robot moves.
        dyn_ids = env["dyn_manager"].body_ids
        planner.set_world_obstacles(env["static_ids"] + dyn_ids,
                                    dynamic_ids=dyn_ids)
        print(f"[{args.planner}] Online mode — "
              f"sense_radius={env.get('sense_radius', env['dmax']):.1f} m, "
              f"{args.online_samples_per_tick} samples/tick")
        print("Executing (roadmap grows during exploration)...")
        online_exec = OnlinePathExecutor(
            planner, env,
            online_samples_per_tick=args.online_samples_per_tick)
        record = online_exec.execute()
        record.total_time = record.execution_time
        path = None   # no pre-built path; use executed trajectory for plots

    else:
        # Offline mode (default): build full roadmap on static env, then execute
        planner.set_obstacles(env["static_ids"])

        print(f"[{args.planner}] Building PRM "
              f"(N={env['N']}, dmax={env['dmax']})...")
        planner.construct(env["N"], env["dmax"], verbose=True)
        print(f"  -> {planner.G.number_of_nodes()} nodes, "
              f"{planner.G.number_of_edges()} edges")

        print("Finding path...")
        path, cost = planner.find(env["start"], env["goal"], env["dmax"])
        if path:
            print(f"  -> length {cost:.2f}, {len(path)} waypoints")
        else:
            print("  -> no path found")

        # Show roadmap + planned path before execution begins
        plan_title = (f"{args.env} / {DIFFICULTY_LABELS[args.diff]} / {args.planner}"
                      f" — Roadmap & Planned Path")
        if env["dim"] == 2:
            plot_plan_2d(env, planner, path, plan_title)
        else:
            plot_plan_3d(env, planner, path, plan_title)

        # Restore full obstacle set before execution
        dyn_ids = env["dyn_manager"].body_ids
        planner.set_obstacles(env["static_ids"] + dyn_ids, dynamic_ids=dyn_ids)
        print("Executing...")
        executor = PathExecutor(planner, env)
        record = executor.execute(path, cost)
        record.total_time = record.execution_time

    m = trial_metrics(record, dim=env["dim"])
    print(f"\nResult: {'SUCCESS' if m['success'] else 'FAILURE'}")
    print(f"  Smoothness:    {m['smoothness']:.3f}")
    print(f"  Min clearance: {m['min_clearance']:.4f}")
    print(f"  Replans:       {m['replans']}")

    # Draw path + markers in PyBullet viewer
    if args.gui and record.executed_positions:
        pts = record.executed_positions
        for i in range(len(pts) - 1):
            a = list(pts[i]) if env["dim"] == 3 \
                else [pts[i][0], pts[i][1], 0.15]
            b = list(pts[i+1]) if env["dim"] == 3 \
                else [pts[i+1][0], pts[i+1][1], 0.15]
            p.addUserDebugLine(a, b, [1, 0, 0], lineWidth=3,
                               physicsClientId=env["cid"])
        for pos, clr in [(env["start"], [0, 1, 0]),
                         (env["goal"],  [0, 0, 1])]:
            pos3 = list(pos) if env["dim"] == 3 \
                else [pos[0], pos[1], 0.15]
            vs = p.createVisualShape(p.GEOM_SPHERE, radius=0.3,
                                     rgbaColor=clr + [0.8],
                                     physicsClientId=env["cid"])
            p.createMultiBody(baseMass=0, baseVisualShapeIndex=vs,
                              basePosition=pos3,
                              physicsClientId=env["cid"])

    title = (f"{args.env} / {DIFFICULTY_LABELS[args.diff]} / {args.planner}"
             f" — {'OK' if m['success'] else 'FAIL'}")

    if env["dim"] == 2:
        plot_2d(env, planner, record, title)
    else:
        plot_3d(env, planner, record, title)

    if args.gui:
        print("Press Ctrl-C to exit GUI.")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass

    p.disconnect(env["cid"])


if __name__ == "__main__":
    main()
