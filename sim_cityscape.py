"""
Simulation 3: Cityscape 3D — Drone Navigation
A 40×40×25 m urban environment with buildings of varying heights.
The drone PRM plans in full (x, y, z) space, requiring the planner
to route around and over structures.
"""

import sys
import time
import numpy as np
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from prm_planner import PRMPlanner

# ── Environment parameters ──────────────────────────────────────────
ARENA_XY = 40
ARENA_Z = 25
START = (2.0, 2.0, 5.0)
GOAL  = (38.0, 38.0, 15.0)
N_SAMPLES = 800
DMAX = 8.0
ROBOT_RADIUS = 0.4     # drone collision sphere
SEED = 42

# Building specs: (cx, cy, half_x, half_y, height)
BUILDINGS = [
    # Cluster 1 — near start
    (8,  6,  2.5, 2.5, 18),
    (8,  12, 1.5, 3.0, 10),
    (13, 8,  2.0, 2.0, 22),
    # Cluster 2 — centre
    (20, 18, 3.0, 3.0, 20),
    (20, 24, 2.0, 2.0, 12),
    (25, 20, 2.5, 4.0, 16),
    (18, 13, 1.5, 1.5, 8),
    # Cluster 3 — far side
    (32, 30, 3.0, 2.0, 24),
    (34, 36, 2.0, 2.0, 14),
    (28, 34, 2.5, 2.5, 10),
    # Scattered towers
    (15, 30, 1.5, 1.5, 20),
    (30, 10, 2.0, 3.0, 18),
    (5,  25, 2.0, 2.0, 15),
    (36, 20, 1.5, 1.5, 22),
    (22, 6,  2.0, 1.5, 11),
]

# Muted colour palette for buildings
BLDG_COLORS = [
    [0.55, 0.55, 0.60, 1],
    [0.50, 0.52, 0.58, 1],
    [0.60, 0.58, 0.55, 1],
    [0.48, 0.50, 0.55, 1],
    [0.58, 0.56, 0.52, 1],
]


def build_env(cid):
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
    p.setGravity(0, 0, -9.81, physicsClientId=cid)
    p.loadURDF("plane.urdf", physicsClientId=cid)

    obstacles = []
    building_data = []     # for matplotlib

    for i, (cx, cy, hx, hy, height) in enumerate(BUILDINGS):
        hz = height / 2.0
        color = BLDG_COLORS[i % len(BLDG_COLORS)]
        col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[hx, hy, hz], physicsClientId=cid)
        vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[hx, hy, hz],
            rgbaColor=color, physicsClientId=cid)
        body = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[cx, cy, hz], physicsClientId=cid)
        obstacles.append(body)
        building_data.append((cx, cy, hx, hy, height))

    return obstacles, building_data


def _box_faces(cx, cy, hx, hy, h):
    """Return 6 polygon faces of a box for Poly3DCollection."""
    x0, x1 = cx - hx, cx + hx
    y0, y1 = cy - hy, cy + hy
    verts = [
        [(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0)],  # bottom
        [(x0, y0, h), (x1, y0, h), (x1, y1, h), (x0, y1, h)],  # top
        [(x0, y0, 0), (x1, y0, 0), (x1, y0, h), (x0, y0, h)],
        [(x1, y0, 0), (x1, y1, 0), (x1, y1, h), (x1, y0, h)],
        [(x1, y1, 0), (x0, y1, 0), (x0, y1, h), (x1, y1, h)],
        [(x0, y1, 0), (x0, y0, 0), (x0, y0, h), (x0, y1, h)],
    ]
    return verts


def plot_result(building_data, planner, path,
                title="Cityscape 3D — Drone PRM"):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Draw buildings
    for (cx, cy, hx, hy, h) in building_data:
        faces = _box_faces(cx, cy, hx, hy, h)
        poly = Poly3DCollection(faces, alpha=0.35,
                                facecolor='#7a8a9a', edgecolor='#4a5a6a',
                                linewidth=0.4)
        ax.add_collection3d(poly)

    # Draw PRM edges (subset for clarity)
    edges = list(planner.G.edges)
    step = max(1, len(edges) // 2000)
    for u, v in edges[::step]:
        ax.plot([u[0], v[0]], [u[1], v[1]], [u[2], v[2]],
                color='steelblue', linewidth=0.2, alpha=0.3)

    # Draw path
    if path:
        px = [pt[0] for pt in path]
        py = [pt[1] for pt in path]
        pz = [pt[2] for pt in path]
        ax.plot(px, py, pz, color='red', linewidth=2.5, zorder=5,
                label='Path')

    ax.scatter(*START, color='green', s=80, zorder=6, label='Start')
    ax.scatter(*GOAL,  color='blue',  s=80, zorder=6, label='Goal')

    ax.set_xlim(0, ARENA_XY)
    ax.set_ylim(0, ARENA_XY)
    ax.set_zlim(0, ARENA_Z)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig("result_cityscape.png", dpi=150)
    plt.show()


def main(use_gui=False):
    np.random.seed(SEED)
    cid = p.connect(p.GUI if use_gui else p.DIRECT)

    if use_gui:
        p.resetDebugVisualizerCamera(
            cameraDistance=55, cameraYaw=45, cameraPitch=-35,
            cameraTargetPosition=[20, 20, 5], physicsClientId=cid)

    obstacle_ids, building_data = build_env(cid)

    planner = PRMPlanner(
        cid,
        workspace_bounds=[(0, ARENA_XY), (0, ARENA_XY), (1, ARENA_Z)],
        robot_radius=ROBOT_RADIUS,
        dim=3)
    planner.set_obstacles(obstacle_ids)

    print("Building 3D PRM (cityscape)...")
    planner.ConstructPRM(N_SAMPLES, DMAX)
    print(f"PRM: {planner.G.number_of_nodes()} nodes, "
          f"{planner.G.number_of_edges()} edges")

    print("Searching for path...")
    path, dist = planner.find(START, GOAL, DMAX)
    print(f"Path found — length: {dist:.2f}, waypoints: {len(path)}")

    plot_result(building_data, planner, path)

    if use_gui:
        # Draw path as debug lines in PyBullet viewer
        for i in range(len(path) - 1):
            p.addUserDebugLine(
                list(path[i]), list(path[i + 1]),
                [1, 0, 0], lineWidth=3, physicsClientId=cid)
        # Animate a sphere along the path
        drone_vis = p.createVisualShape(
            p.GEOM_SPHERE, radius=ROBOT_RADIUS * 2,
            rgbaColor=[0, 0.8, 0.2, 0.8], physicsClientId=cid)
        drone = p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=drone_vis,
            basePosition=list(START), physicsClientId=cid)

        print("Animating drone... (Ctrl-C to stop)")
        try:
            while True:
                for i in range(len(path) - 1):
                    a, b = np.array(path[i]), np.array(path[i + 1])
                    seg_len = np.linalg.norm(b - a)
                    n_frames = max(int(seg_len / 0.15), 2)
                    for f in range(n_frames):
                        t = f / n_frames
                        pos = a + t * (b - a)
                        p.resetBasePositionAndOrientation(
                            drone, pos.tolist(), [0, 0, 0, 1],
                            physicsClientId=cid)
                        time.sleep(0.02)
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass

    p.disconnect(cid)


if __name__ == "__main__":
    main(use_gui="--gui" in sys.argv)
