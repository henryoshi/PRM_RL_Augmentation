"""
Base PRM infrastructure — collision checking, sampling, graph management.
All PRM variants inherit from this.
"""

import numpy as np
import networkx as nx
import pybullet as p


class PRMBase:
    name = "Base"

    def __init__(self, client_id, workspace_bounds, robot_radius=0.15,
                 dim=2, fixed_z=None):
        self.cid = client_id
        self.bounds = workspace_bounds
        self.radius = robot_radius
        self.dim = dim
        self.fixed_z = fixed_z if fixed_z is not None else robot_radius + 0.01
        self.G = nx.Graph()
        self.obstacle_ids = []

        self._col = p.createCollisionShape(
            p.GEOM_SPHERE, radius=robot_radius, physicsClientId=self.cid)
        self._probe = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=self._col,
            basePosition=[0, 0, -100], physicsClientId=self.cid)

    def set_obstacles(self, ids):
        self.obstacle_ids = list(ids)

    def reset_graph(self):
        self.G = nx.Graph()

    # ── Collision helpers ────────────────────────────────────────────
    def _to_3d(self, pt):
        if self.dim == 2:
            return [pt[0], pt[1], self.fixed_z]
        return list(pt[:3])

    def is_free(self, pt):
        pos = self._to_3d(pt)
        p.resetBasePositionAndOrientation(
            self._probe, pos, [0, 0, 0, 1], physicsClientId=self.cid)
        for oid in self.obstacle_ids:
            if p.getClosestPoints(
                    self._probe, oid, 0.0, physicsClientId=self.cid):
                return False
        return True

    def edge_free(self, v1, v2):
        a = np.asarray(v1, dtype=float)
        b = np.asarray(v2, dtype=float)
        length = np.linalg.norm(b - a)
        n = max(int(length / (self.radius * 0.5)), 2)
        for i in range(n + 1):
            if not self.is_free(tuple(a + (i / n) * (b - a))):
                return False
        return True

    def clearance(self, pt):
        """Min distance from robot centre at *pt* to any obstacle surface."""
        pos = self._to_3d(pt)
        p.resetBasePositionAndOrientation(
            self._probe, pos, [0, 0, 0, 1], physicsClientId=self.cid)
        min_d = float('inf')
        for oid in self.obstacle_ids:
            contacts = p.getClosestPoints(
                self._probe, oid, 100.0, physicsClientId=self.cid)
            for c in contacts:
                min_d = min(min_d, c[8] + self.radius)
        return min_d

    # ── Sampling ─────────────────────────────────────────────────────
    def random_sample(self):
        while True:
            pt = tuple(np.random.uniform(lo, hi)
                       for lo, hi in self.bounds[:self.dim])
            if self.is_free(pt):
                return pt

    def dist(self, a, b):
        return float(np.linalg.norm(np.array(a) - np.array(b)))

    # ── Interface (override in subclasses) ───────────────────────────
    def construct(self, N, dmax, **kw):
        raise NotImplementedError

    def find(self, start, goal, dmax, **kw):
        raise NotImplementedError

    def replan(self, current, goal, dmax, **kw):
        """Re-use existing roadmap to find a new path from *current*."""
        return self.find(current, goal, dmax, **kw)
