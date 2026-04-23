"""
Environment builder for all three scenarios.
Each function returns a consistent dict:
  {
    "cid":         PyBullet client id,
    "static_ids":  list of static obstacle body ids,
    "dyn_manager": DynamicObstacleManager (may be empty),
    "noise":       MotionNoise instance,
    "bounds":      workspace bounds [(lo,hi), ...],
    "start":       start config tuple,
    "goal":        goal config tuple,
    "dim":         2 or 3,
    "dmax":        connection radius,
    "N":           suggested sample count,
    "robot_radius": float,
  }

Difficulty levels:
  0 — static only
  1 — motion noise
  2 — dynamic obstacles
  3 — noise + dynamic obstacles
"""

import numpy as np
import pybullet as p
import pybullet_data
from dynamics import (DynamicObstacleManager, LinearObstacle, MotionNoise)

WALL_HEIGHT = 1.0

# ── Velocity helpers ─────────────────────────────────────────────────

def _rand_vel_2d(speed):
    angle = np.random.uniform(0, 2 * np.pi)
    return float(speed * np.cos(angle)), float(speed * np.sin(angle))

def _rand_vel_3d(speed):
    cos_theta = np.random.uniform(-1, 1)
    sin_theta = np.sqrt(max(0.0, 1 - cos_theta ** 2))
    phi = np.random.uniform(0, 2 * np.pi)
    return (float(speed * sin_theta * np.cos(phi)),
            float(speed * sin_theta * np.sin(phi)),
            float(speed * cos_theta))

# ── Helper ───────────────────────────────────────────────────────────

def _box(cid, cx, cy, hx, hy, hz=WALL_HEIGHT, color=None, cz=None):
    """Create a box obstacle. Returns (body_id, rect_tuple).
    rect_tuple = (cx, cy, hx, hy, height) for matplotlib plotting."""
    if color is None:
        color = [0.5, 0.5, 0.55, 1]
    if cz is None:
        cz = hz
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz],
                                 physicsClientId=cid)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, hz],
                              rgbaColor=color, physicsClientId=cid)
    body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis,
                             basePosition=[cx, cy, cz],
                             physicsClientId=cid)
    rect = (cx, cy, hx, hy, hz * 2)   # centre, half-extents, full height
    return body, rect


def _init_bullet(gui=False):
    cid = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                              physicsClientId=cid)
    p.setGravity(0, 0, -9.81, physicsClientId=cid)
    p.loadURDF("plane.urdf", physicsClientId=cid)
    return cid


# ═══════════════════════════════════════════════════════════════════════
#  1. SIMPLE 2D  (12 × 12 m)
# ═══════════════════════════════════════════════════════════════════════

def build_simple(difficulty=0, gui=False):
    cid = _init_bullet(gui)
    if gui:
        p.resetDebugVisualizerCamera(
            cameraDistance=16, cameraYaw=0, cameraPitch=-89,
            cameraTargetPosition=[6, 6, 0], physicsClientId=cid)
    static = []
    rects = []

    def add(cx, cy, hx, hy, **kw):
        body, rect = _box(cid, cx, cy, hx, hy, **kw)
        static.append(body)
        rects.append(rect)

    add(3.5, 6.0,  3.5, 0.15)
    add(8.0, 8.5,  0.15, 3.5)
    add(5.5, 3.0,  0.8,  0.8)
    add(9.5, 3.0,  0.15, 2.0)
    add(9.5, 1.0,  1.5,  0.15)
    add(10.0, 9.5, 0.6,  0.6)

    mgr = DynamicObstacleManager()
    dyn_rects = []
    noise_std = 0.0

    if difficulty >= 2:
        # 4 linear movers in key corridors, matching office-style dynamics.
        # Each is 0.3×0.3 m — small enough to not permanently block passages,
        # large enough to force replanning when they cross the robot's path.
        _dyn_starts = [
            (2.0,  3.0,  0.3, 0.3),
            (5.0,  8.0,  0.3, 0.3),
            (9.0,  8.0,  0.3, 0.3),
            (10.5, 4.0,  0.3, 0.3),
        ]
        _ws = [(0, 12), (0, 12)]
        for cx, cy, hx, hy in _dyn_starts:
            b, r = _box(cid, cx, cy, hx, hy, color=[0.9, 0.3, 0.1, 0.85])
            mgr.add(LinearObstacle(cid, b, velocity=_rand_vel_2d(0.4), bounds=_ws))
            dyn_rects.append(r)

    if difficulty in (1, 3):
        noise_std = 0.08

    return dict(
        cid=cid, static_ids=static, dyn_manager=mgr,
        noise=MotionNoise(noise_std, dim=2, bounds=[(0, 12), (0, 12)]),
        bounds=[(0, 12), (0, 12)], start=(1.0, 1.0), goal=(11.0, 11.0),
        dim=2, dmax=3.0, N=300, robot_radius=0.2,
        sense_radius=4.0,   # online mode: obstacles revealed within this radius
        rects=rects, dyn_rects=dyn_rects,
    )


# ═══════════════════════════════════════════════════════════════════════
#  2. OFFICE FLOOR PLAN  (20 × 15 m)
# ═══════════════════════════════════════════════════════════════════════

def build_office(difficulty=0, gui=False):
    cid = _init_bullet(gui)
    if gui:
        p.resetDebugVisualizerCamera(
            cameraDistance=22, cameraYaw=0, cameraPitch=-89,
            cameraTargetPosition=[10, 7.5, 0], physicsClientId=cid)
    static = []
    rects = []
    W = 0.12
    CLR = [0.70, 0.65, 0.55, 1]
    DOOR = 0.9

    def wall(cx, cy, hx, hy):
        b, r = _box(cid, cx, cy, hx, hy, color=CLR)
        static.append(b)
        rects.append(r)

    # Outer walls
    wall(10, W, 10, W)
    wall(10, 15 - W, 10, W)
    wall(W, 7.5, W, 7.5)
    wall(20 - W, 7.5, W, 7.5)

    # x = 7 partition (door y ∈ [6.5, 7.4])
    wall(7, 3.25, W, 3.25)
    wall(7, 11.2, W, 3.8)

    # x = 14 partition (doors at y≈3.5 and y≈11)
    wall(14, 1.6, W, 1.6)
    wall(14, 7.1, W, 3.0)
    wall(14, 13.45, W, 1.55)

    # y = 10 horizontal (door x≈3.5)
    wall(1.6, 10, 1.6, W)
    wall(5.65, 10, 1.35, W)

    # y = 4.5 horizontal (door x≈5)
    wall(2.4, 4.5, 2.4, W)
    wall(6.35, 4.5, 0.65, W)

    # y = 7.5 from x=14→20 (door x≈17)
    wall(15.4, 7.5, 1.4, W)
    wall(18.85, 7.5, 1.15, W)

    # Furniture
    wall(2.5, 7.5, 0.6, 0.6)
    wall(10.5, 12.0, 1.0, 0.4)
    wall(10.5, 3.0, 0.5, 0.8)
    wall(17.0, 12.5, 0.7, 0.5)

    mgr = DynamicObstacleManager()
    dyn_rects = []
    noise_std = 0.0

    if difficulty >= 2:
        # 5 random-walk obstacles spread across different rooms/corridors.
        # Each stays within its own room bounds so it wanders naturally
        # without permanently blocking any doorway.
        # Bounds are inset ~0.5 m from wall faces so momentum walkers
        # never reach a wall without the bounds bounce firing first.
        # No per-tick PyBullet collision check needed.
        _dyn_starts = [
            (3.0,  2.0,  0.35, 0.35),
            (3.0, 12.0,  0.35, 0.35),
            (10.0,  3.5, 0.35, 0.35),
            (10.5, 12.5, 0.35, 0.35),
            (17.0,  7.0, 0.35, 0.35),
        ]
        _ws = [(0, 20), (0, 15)]
        for cx, cy, hx, hy in _dyn_starts:
            b, r = _box(cid, cx, cy, hx, hy, color=[0.9, 0.3, 0.1, 0.85])
            mgr.add(LinearObstacle(cid, b, velocity=_rand_vel_2d(0.4), bounds=_ws))
            dyn_rects.append(r)

    if difficulty in (1, 3):
        noise_std = 0.06

    return dict(
        cid=cid, static_ids=static, dyn_manager=mgr,
        noise=MotionNoise(noise_std, dim=2, bounds=[(0, 20), (0, 15)]),
        bounds=[(0, 20), (0, 15)], start=(1.5, 1.5), goal=(18.5, 13.5),
        dim=2, dmax=3.5, N=1000, robot_radius=0.2, min_edge_len=0.5,
        frontier_frac=0.25,   # raised from 0.15: more doorway-biased samples
                              # needed to bridge narrow passages reliably
        max_neighbors=15,     # raised from 10: doorway-adjacent nodes need
                              # cross-wall neighbors in their k-NN candidate set
        sense_radius=5.0,     # online mode: obstacle reveal radius
        rects=rects, dyn_rects=dyn_rects,
    )


# ═══════════════════════════════════════════════════════════════════════
#  3. CITYSCAPE 3D  (40 × 40 × 25 m)
# ═══════════════════════════════════════════════════════════════════════

_BUILDINGS = [
    (8, 6, 2.5, 2.5, 18), (8, 12, 1.5, 3, 10), (13, 8, 2, 2, 22),
    (20, 18, 3, 3, 20), (20, 24, 2, 2, 12), (25, 20, 2.5, 4, 16),
    (18, 13, 1.5, 1.5, 8), (32, 30, 3, 2, 24), (34, 36, 2, 2, 14),
    (28, 34, 2.5, 2.5, 10), (15, 30, 1.5, 1.5, 20), (30, 10, 2, 3, 18),
    (5, 25, 2, 2, 15), (36, 20, 1.5, 1.5, 22), (22, 6, 2, 1.5, 11),
]

_BLDG_COLORS = [
    [0.55, 0.55, 0.60, 1], [0.50, 0.52, 0.58, 1],
    [0.60, 0.58, 0.55, 1], [0.48, 0.50, 0.55, 1],
]


def build_cityscape(difficulty=0, gui=False):
    cid = _init_bullet(gui)
    if gui:
        p.resetDebugVisualizerCamera(
            55, 45, -35, [20, 20, 5], physicsClientId=cid)
    static = []
    rects = []

    for i, (cx, cy, hx, hy, h) in enumerate(_BUILDINGS):
        clr = _BLDG_COLORS[i % len(_BLDG_COLORS)]
        b, r = _box(cid, cx, cy, hx, hy, hz=h / 2, color=clr, cz=h / 2)
        static.append(b)
        rects.append(r)

    mgr = DynamicObstacleManager()
    dyn_rects = []
    noise_std = 0.0

    _cs_bounds = [(0, 40), (0, 40), (1, 25)]
    if difficulty >= 2:
        d1, r1 = _box(cid, 15, 15, 0.6, 0.6, hz=0.6,
                       color=[1, 0.2, 0.2, 0.8], cz=12)
        mgr.add(LinearObstacle(cid, d1, velocity=_rand_vel_3d(2.5),
                               bounds=_cs_bounds, dim=3))
        d2, r2 = _box(cid, 25, 15, 0.5, 0.5, hz=0.5,
                       color=[1, 0.2, 0.2, 0.8], cz=8)
        mgr.add(LinearObstacle(cid, d2, velocity=_rand_vel_3d(2.5),
                               bounds=_cs_bounds, dim=3))
        d3, r3 = _box(cid, 20, 20, 0.5, 0.5, hz=0.5,
                       color=[1, 0.2, 0.2, 0.8], cz=15)
        mgr.add(LinearObstacle(cid, d3, velocity=_rand_vel_3d(2.5),
                               bounds=_cs_bounds, dim=3))
        dyn_rects.extend([r1, r2, r3])

    if difficulty in (1, 3):
        noise_std = 0.15

    return dict(
        cid=cid, static_ids=static, dyn_manager=mgr,
        noise=MotionNoise(noise_std, dim=3,
                          bounds=[(0, 40), (0, 40), (1, 25)]),
        bounds=[(0, 40), (0, 40), (1, 25)],
        start=(2.0, 2.0, 5.0), goal=(38.0, 38.0, 5.0),
        dim=3, dmax=6.0, N=1500, robot_radius=0.3, max_neighbors=12,
        sense_radius=14.0,
        max_nodes_multiplier=3,   # online node cap = N * this (3D needs more)
        rects=rects, dyn_rects=dyn_rects,
        buildings=_BUILDINGS,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Registry
# ═══════════════════════════════════════════════════════════════════════

ENV_BUILDERS = {
    "simple":    build_simple,
    "office":    build_office,
    "cityscape": build_cityscape,
}

DIFFICULTY_LABELS = {
    0: "static",
    1: "noise",
    2: "dynamic",
    3: "noise+dynamic",
}