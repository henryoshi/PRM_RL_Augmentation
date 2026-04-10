"""
Simulation 2: Office Floor Plan (2D)
A 20×15 m office with rooms, corridors, and narrow doorways.
Tests PRM's ability to find paths through tight passages — a known
challenge for sampling-based planners.
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
ARENA = (20, 15)
START = (1.5, 1.5)
GOAL  = (18.5, 13.5)
N_SAMPLES = 600
DMAX = 3.0
ROBOT_RADIUS = 0.2
WALL_THICK = 0.12      # half-thickness of walls
WALL_HEIGHT = 1.2
DOOR_WIDTH = 0.9        # metres — tight squeeze for PRM
SEED = 42


def build_env(cid):
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
    p.setGravity(0, 0, -9.81, physicsClientId=cid)
    p.loadURDF("plane.urdf", physicsClientId=cid)

    obstacles, rects = [], []
    W = WALL_THICK
    H = WALL_HEIGHT

    def add_wall(cx, cy, hx, hy):
        col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[hx, hy, H], physicsClientId=cid)
        vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[hx, hy, H],
            rgbaColor=[0.70, 0.65, 0.55, 1], physicsClientId=cid)
        body = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[cx, cy, H], physicsClientId=cid)
        obstacles.append(body)
        rects.append((cx - hx, cy - hy, 2 * hx, 2 * hy))

    # ── Outer walls ─────────────────────────────────────────────────
    add_wall(10,  W,   10, W)       # bottom
    add_wall(10,  15 - W, 10, W)    # top
    add_wall(W,   7.5, W,  7.5)     # left
    add_wall(20 - W, 7.5, W, 7.5)   # right

    # ── Vertical partition at x=7 (door at y ∈ [6.5, 7.4]) ─────────
    door_lo, door_hi = 6.5, 6.5 + DOOR_WIDTH
    add_wall(7, (0 + door_lo) / 2,   W, door_lo / 2)
    add_wall(7, (door_hi + 15) / 2,  W, (15 - door_hi) / 2)

    # ── Vertical partition at x=14 (doors at y≈3.5 and y≈11.5) ─────
    d1_lo, d1_hi = 3.2, 3.2 + DOOR_WIDTH
    d2_lo, d2_hi = 11.0, 11.0 + DOOR_WIDTH
    add_wall(14, (0 + d1_lo) / 2,      W, d1_lo / 2)
    add_wall(14, (d1_hi + d2_lo) / 2,  W, (d2_lo - d1_hi) / 2)
    add_wall(14, (d2_hi + 15) / 2,     W, (15 - d2_hi) / 2)

    # ── Horizontal wall at y=10 from x=0→7 (door at x≈3.5) ────────
    hd_lo, hd_hi = 3.2, 3.2 + DOOR_WIDTH
    add_wall(hd_lo / 2,      10, hd_lo / 2,       W)
    add_wall((hd_hi + 7) / 2, 10, (7 - hd_hi) / 2, W)

    # ── Horizontal wall at y=4.5 from x=0→7 (door at x≈5) ─────────
    hd2_lo, hd2_hi = 4.8, 4.8 + DOOR_WIDTH
    add_wall(hd2_lo / 2,       4.5, hd2_lo / 2,       W)
    add_wall((hd2_hi + 7) / 2, 4.5, (7 - hd2_hi) / 2, W)

    # ── Horizontal wall at y=7.5 from x=14→20 (door at x≈17) ──────
    hd3_lo, hd3_hi = 16.8, 16.8 + DOOR_WIDTH
    add_wall((14 + hd3_lo) / 2, 7.5, (hd3_lo - 14) / 2, W)
    add_wall((hd3_hi + 20) / 2, 7.5, (20 - hd3_hi) / 2, W)

    # ── Furniture / cubicle blocks ──────────────────────────────────
    add_wall(2.5, 7.5, 0.6, 0.6)
    add_wall(10.5, 12.0, 1.0, 0.4)
    add_wall(10.5, 3.0, 0.5, 0.8)
    add_wall(17.0, 12.5, 0.7, 0.5)

    return obstacles, rects


def plot_result(rects, planner, path, title="Office Floor Plan PRM"):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax_idx, ax in enumerate(axes):
        ax.set_xlim(0, ARENA[0])
        ax.set_ylim(0, ARENA[1])
        ax.set_aspect('equal')
        ax.set_facecolor('#faf8f4')

        for (rx, ry, rw, rh) in rects:
            ax.add_patch(patches.Rectangle(
                (rx, ry), rw, rh,
                facecolor='#8B7D6B', edgecolor='#5C4033', linewidth=0.5))

        if ax_idx == 0:
            ax.set_title("PRM Roadmap")
            for u, v in planner.G.edges:
                ax.plot([u[0], v[0]], [u[1], v[1]],
                        color='cornflowerblue', linewidth=0.25, alpha=0.5)
            xs = [n[0] for n in planner.G.nodes]
            ys = [n[1] for n in planner.G.nodes]
            ax.scatter(xs, ys, s=3, c='cornflowerblue', zorder=3)
        else:
            ax.set_title("Found Path")

        if path:
            px = [pt[0] for pt in path]
            py = [pt[1] for pt in path]
            ax.plot(px, py, color='crimson', linewidth=2.2, zorder=4)

        ax.plot(*START, 'go', markersize=10, zorder=5, label='Start')
        ax.plot(*GOAL, 'b^', markersize=10, zorder=5, label='Goal')
        ax.legend(loc='upper left', fontsize=8)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig("result_office.png", dpi=150)
    plt.show()


def main(use_gui=False):
    np.random.seed(SEED)
    cid = p.connect(p.GUI if use_gui else p.DIRECT)

    obstacle_ids, rects = build_env(cid)

    planner = PRMPlanner(
        cid, workspace_bounds=[(0, ARENA[0]), (0, ARENA[1])],
        robot_radius=ROBOT_RADIUS, dim=2)
    planner.set_obstacles(obstacle_ids)

    print("Building PRM (office)...")
    planner.ConstructPRM(N_SAMPLES, DMAX)
    print(f"PRM: {planner.G.number_of_nodes()} nodes, "
          f"{planner.G.number_of_edges()} edges")

    print("Searching for path...")
    path, dist = planner.find(START, GOAL, DMAX)
    print(f"Path found — length: {dist:.2f}, waypoints: {len(path)}")

    plot_result(rects, planner, path)

    if use_gui:
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
    main(use_gui="--gui" in sys.argv)
