"""
Simulation 1: Simple 2D Plane
A 12×12 m arena with a few walls and box obstacles.
The robot must navigate from bottom-left to top-right.
"""

import sys
import time
import numpy as np
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from prm_planner import PRMPlanner

# ── Environment parameters ──────────────────────────────────────────
ARENA = (12, 12)          # metres
START = (1.0, 1.0)
GOAL  = (11.0, 11.0)
N_SAMPLES = 300
DMAX = 3.0
ROBOT_RADIUS = 0.2
WALL_HEIGHT = 1.0
SEED = 42


def build_env(cid):
    """Create ground plane + obstacles. Returns (obstacle_ids, rect_patches)."""
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
    p.setGravity(0, 0, -9.81, physicsClientId=cid)
    p.loadURDF("plane.urdf", physicsClientId=cid)

    obstacles, rects = [], []

    def add_box(cx, cy, hx, hy, hz=WALL_HEIGHT):
        col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[hx, hy, hz], physicsClientId=cid)
        vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[hx, hy, hz],
            rgbaColor=[0.45, 0.45, 0.50, 1], physicsClientId=cid)
        body = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[cx, cy, hz], physicsClientId=cid)
        obstacles.append(body)
        # Store rectangle for matplotlib (x, y, width, height)
        rects.append((cx - hx, cy - hy, 2 * hx, 2 * hy))

    # --- obstacles ---
    # Horizontal wall across the middle, gap on the right
    add_box(3.5, 6.0, 3.5, 0.15)
    # Vertical wall on the right side, gap at the bottom
    add_box(8.0, 8.5, 0.15, 3.5)
    # Square pillar
    add_box(5.5, 3.0, 0.8, 0.8)
    # L-shaped barrier (two boxes)
    add_box(9.5, 3.0, 0.15, 2.0)
    add_box(9.5, 1.0, 1.5, 0.15)
    # Small block near goal
    add_box(10.0, 9.5, 0.6, 0.6)

    return obstacles, rects


def plot_result(rects, planner, path, title="Simple 2D PRM"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, ax in enumerate(axes):
        ax.set_xlim(0, ARENA[0])
        ax.set_ylim(0, ARENA[1])
        ax.set_aspect('equal')
        ax.set_facecolor('#f0f0f0')

        # Draw obstacles
        for (rx, ry, rw, rh) in rects:
            ax.add_patch(patches.Rectangle(
                (rx, ry), rw, rh, facecolor='#555555', edgecolor='black'))

        if ax_idx == 0:
            # Full PRM graph
            ax.set_title("PRM Roadmap")
            pos = {n: n for n in planner.G.nodes}
            for u, v in planner.G.edges:
                ax.plot([u[0], v[0]], [u[1], v[1]],
                        color='steelblue', linewidth=0.3, alpha=0.5)
            xs = [n[0] for n in planner.G.nodes]
            ys = [n[1] for n in planner.G.nodes]
            ax.scatter(xs, ys, s=4, c='steelblue', zorder=3)
        else:
            # Path only
            ax.set_title("Found Path")

        # Draw path on both panels
        if path:
            px = [pt[0] for pt in path]
            py = [pt[1] for pt in path]
            ax.plot(px, py, color='red', linewidth=2, zorder=4)

        ax.plot(*START, 'go', markersize=10, zorder=5, label='Start')
        ax.plot(*GOAL, 'b^', markersize=10, zorder=5, label='Goal')
        ax.legend(loc='upper left')

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig("result_simple.png", dpi=150)
    plt.show()


def main(use_gui=False):
    np.random.seed(SEED)
    mode = p.GUI if use_gui else p.DIRECT
    cid = p.connect(mode)

    obstacle_ids, rects = build_env(cid)

    planner = PRMPlanner(
        cid, workspace_bounds=[(0, ARENA[0]), (0, ARENA[1])],
        robot_radius=ROBOT_RADIUS, dim=2)
    planner.set_obstacles(obstacle_ids)

    print("Building PRM...")
    planner.ConstructPRM(N_SAMPLES, DMAX)
    print(f"PRM: {planner.G.number_of_nodes()} nodes, "
          f"{planner.G.number_of_edges()} edges")

    print("Searching for path...")
    path, dist = planner.find(START, GOAL, DMAX)
    print(f"Path found — length: {dist:.2f}, waypoints: {len(path)}")

    plot_result(rects, planner, path)

    if use_gui:
        # Draw path in PyBullet viewer
        for i in range(len(path) - 1):
            a = [path[i][0], path[i][1], 0.1]
            b = [path[i+1][0], path[i+1][1], 0.1]
            p.addUserDebugLine(a, b, [1, 0, 0], lineWidth=3,
                               physicsClientId=cid)
        print("Press Ctrl-C to exit GUI.")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass

    p.disconnect(cid)


if __name__ == "__main__":
    gui = "--gui" in sys.argv
    main(use_gui=gui)
