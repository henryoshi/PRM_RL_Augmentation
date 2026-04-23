"""
viz_local_repair.py — Local graph repair after a dynamic obstacle appears.

Two-panel figure:
  Left  – RiskAware roadmap with the initially planned path.
  Right – A new obstacle is placed mid-path; nodes inside the affected
          zone are removed (red ×), repair samples are added (green ★),
          and the repaired path is shown in orange.

Usage:
  python viz_local_repair.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pybullet as pb
import pybullet_data

from environments import build_simple
from prm_risk_aware import RiskAwarePRM

SEED = 7
N    = 300
np.random.seed(SEED)

# ── 1. Build environment and construct roadmap ────────────────────────
env = build_simple(difficulty=0)   # static only — we'll add the obstacle manually
rects  = env.get("rects", [])
BOUNDS = env["bounds"]

planner = RiskAwarePRM(
    env["cid"], env["bounds"], env["robot_radius"], dim=2,
    risk_beta=0.8, frontier_frac=0.3, frontier_sigma=0.8,
    repair_samples=20,
)
planner.set_obstacles(env["static_ids"])
print("Building roadmap...")
planner.construct(N, env["dmax"], verbose=False)
print(f"  {planner.G.number_of_nodes()} nodes, {planner.G.number_of_edges()} edges")

start, goal = env["start"], env["goal"]
path, cost = planner.find(start, goal, env["dmax"])
if path is None:
    raise RuntimeError("Initial path not found — try a different seed.")
print(f"  Path found: {len(path)} waypoints, cost={cost:.2f}")

# ── 2. Snapshot graph BEFORE repair ──────────────────────────────────
nodes_before = set(planner.G.nodes)
edges_before = {tuple(sorted(e)) for e in planner.G.edges}
path_before  = list(path)

# Store ALL edges for the left-panel drawing (captured now, before graph mutates)
all_edges_before = list(planner.G.edges)
all_nodes_before = list(planner.G.nodes)

# ── 3. Place a blocking obstacle on the path ──────────────────────────
# Pick a waypoint roughly 40% along the path as the obstacle centre.
block_idx = max(1, len(path) * 2 // 5)
obs_cx, obs_cy = path[block_idx][0], path[block_idx][1]
obs_hx, obs_hy = 1.0, 1.0   # big enough to invalidate nearby nodes

col = pb.createCollisionShape(
    pb.GEOM_BOX, halfExtents=[obs_hx, obs_hy, 0.5],
    physicsClientId=env["cid"])
vis = pb.createVisualShape(
    pb.GEOM_BOX, halfExtents=[obs_hx, obs_hy, 0.5],
    rgbaColor=[1, 0.2, 0.2, 0.9], physicsClientId=env["cid"])
obs_id = pb.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=col,
    baseVisualShapeIndex=vis,
    basePosition=[obs_cx, obs_cy, 0.5],
    physicsClientId=env["cid"])

# Update obstacle list so the planner sees the new body
planner.set_obstacles(env["static_ids"] + [obs_id])

# ── 4. Replan ─────────────────────────────────────────────────────────
affect_radius = env["dmax"] * 0.7
replan_dmax   = env["dmax"] * 1.5
print("Replanning...")
new_path, new_cost = planner.replan(
    start, goal, replan_dmax,
    dyn_positions=[[obs_cx, obs_cy]],
    affect_radius=affect_radius,
)
if new_path:
    print(f"  Repaired path: {len(new_path)} waypoints, cost={new_cost:.2f}")
else:
    print("  Replan failed — try a different seed or larger repair_samples.")

# ── 5. Diff graph state ───────────────────────────────────────────────
nodes_after  = set(planner.G.nodes)
edges_after  = {tuple(sorted(e)) for e in planner.G.edges}

removed_nodes = nodes_before - nodes_after
added_nodes   = nodes_after  - nodes_before
kept_nodes    = nodes_before & nodes_after
added_edges   = edges_after  - edges_before

print(f"  Removed nodes: {len(removed_nodes)}  |  "
      f"Added repair nodes: {len(added_nodes)}")

# ── 6. Draw ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

def draw_static_obstacles(ax):
    for cx, cy, hx, hy, _ in rects:
        ax.add_patch(patches.Rectangle(
            (cx - hx, cy - hy), 2 * hx, 2 * hy,
            facecolor='#555555', edgecolor='#333333', lw=0.8, zorder=2))


def setup_ax(ax):
    ax.set_xlim(*BOUNDS[0])
    ax.set_ylim(*BOUNDS[1])
    ax.set_aspect('equal')
    ax.set_facecolor('#f0f0f0')
    draw_static_obstacles(ax)


# ─── Left panel: initial roadmap ─────────────────────────────────────
ax = axes[0]
setup_ax(ax)
ax.set_title("Initial Roadmap + Planned Path", fontsize=12, pad=8)

for u, v in all_edges_before:
    ax.plot([u[0], v[0]], [u[1], v[1]],
            color='#4a90d9', lw=0.3, alpha=0.35, zorder=1)

xs_all = [n[0] for n in all_nodes_before]
ys_all = [n[1] for n in all_nodes_before]
ax.scatter(xs_all, ys_all, s=7, c='#4a90d9', zorder=3)

# Planned path
px = [wp[0] for wp in path_before]
py = [wp[1] for wp in path_before]
ax.plot(px, py, color='#c00000', lw=2.5, zorder=5, label='Planned path')

ax.plot(start[0], start[1], 'go', ms=12, zorder=6, label='Start')
ax.plot(goal[0],  goal[1],  'b^', ms=12, zorder=6, label='Goal')
ax.legend(fontsize=9, loc='upper left')


# ─── Right panel: after obstacle + repair ────────────────────────────
ax = axes[1]
setup_ax(ax)
ax.set_title("After Obstacle + Local Graph Repair", fontsize=12, pad=8)

# Surviving edges (light blue, thin)
for u, v in planner.G.edges:
    ax.plot([u[0], v[0]], [u[1], v[1]],
            color='#4a90d9', lw=0.3, alpha=0.30, zorder=1)

# New repair edges (green, slightly thicker)
for (u, v) in added_edges:
    ax.plot([u[0], v[0]], [u[1], v[1]],
            color='#22aa22', lw=1.0, alpha=0.75, zorder=2)

# Kept nodes (blue dots)
if kept_nodes:
    kx = [n[0] for n in kept_nodes]
    ky = [n[1] for n in kept_nodes]
    ax.scatter(kx, ky, s=6, c='#4a90d9', zorder=3)

# Removed nodes (red ×)
if removed_nodes:
    rx = [n[0] for n in removed_nodes]
    ry = [n[1] for n in removed_nodes]
    ax.scatter(rx, ry, s=90, c='#cc0000', marker='x', linewidths=1.8,
               zorder=6, label=f'Removed nodes ({len(removed_nodes)})')

# Added repair nodes (green ★)
if added_nodes:
    anx = [n[0] for n in added_nodes]
    any_ = [n[1] for n in added_nodes]
    ax.scatter(anx, any_, s=120, c='#00bb00', marker='*', zorder=7,
               label=f'Repair nodes ({len(added_nodes)})')

# New obstacle (red box)
ax.add_patch(patches.Rectangle(
    (obs_cx - obs_hx, obs_cy - obs_hy), 2 * obs_hx, 2 * obs_hy,
    facecolor='#dd2222', edgecolor='#991111', lw=1.5,
    alpha=0.85, zorder=4, label='Dynamic obstacle'))

# Original path (grayed out dashes)
ax.plot([wp[0] for wp in path_before],
        [wp[1] for wp in path_before],
        color='#aaaaaa', lw=1.5, linestyle='--', zorder=4,
        label='Original path (blocked)')

# Repaired path (orange)
if new_path:
    ax.plot([wp[0] for wp in new_path],
            [wp[1] for wp in new_path],
            color='#ff7700', lw=2.8, zorder=8, label='Repaired path')

ax.plot(start[0], start[1], 'go', ms=12, zorder=9)
ax.plot(goal[0],  goal[1],  'b^', ms=12, zorder=9)
ax.legend(fontsize=9, loc='upper left')

fig.suptitle(
    "Risk-Aware PRM — Local Graph Repair\n"
    "Nodes inside the obstacle zone are removed; targeted samples are added; A* re-queries",
    fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("viz_local_repair.png", dpi=150, bbox_inches='tight')
print("Saved: viz_local_repair.png")
plt.show()

pb.disconnect(env["cid"])
