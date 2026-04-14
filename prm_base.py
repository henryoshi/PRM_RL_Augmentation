"""
Base PRM infrastructure — collision checking, sampling, graph management.
All PRM variants inherit from this.
"""

import numpy as np
import networkx as nx
import pybullet as p
from scipy.spatial import cKDTree


class PRMBase:
    name = "Base"

    def __init__(self, client_id, workspace_bounds, robot_radius=0.15,
                 dim=2, fixed_z=None, min_edge_len=0.0, max_neighbors=None):
        self.cid = client_id
        self.bounds = workspace_bounds
        self.radius = robot_radius
        self.dim = dim
        self.fixed_z = fixed_z if fixed_z is not None else robot_radius + 0.01
        self.min_edge_len = min_edge_len
        self.max_neighbors = max_neighbors
        self.G = nx.Graph()
        self.obstacle_ids = []

        self._col = p.createCollisionShape(
            p.GEOM_SPHERE, radius=robot_radius, physicsClientId=self.cid)
        self._probe = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=self._col,
            basePosition=[0, 0, -100], physicsClientId=self.cid)
        self._obs_aabbs = []   # populated by set_obstacles()

    def set_obstacles(self, ids, dynamic_ids=None):
        """Register obstacle body ids.

        Parameters
        ----------
        ids         : all obstacle ids (static + dynamic).
        dynamic_ids : subset that move (their AABBs are refreshed each query
                      rather than cached).  If None, all are treated as static.
        """
        self.obstacle_ids = list(ids)
        self._dynamic_ids = set(dynamic_ids) if dynamic_ids else set()
        # Precompute AABBs for static obstacles — avoids calling
        # getClosestPoints on every obstacle for every probe position.
        self._obs_aabbs = []
        for oid in self.obstacle_ids:
            if oid in self._dynamic_ids:
                self._obs_aabbs.append((oid, None, None))  # always include
                continue
            try:
                mn, mx = p.getAABB(oid, physicsClientId=self.cid)
                self._obs_aabbs.append((oid, np.array(mn), np.array(mx)))
            except Exception:
                self._obs_aabbs.append((oid, None, None))

    def _obstacles_near(self, pos_3d, radius):
        """Return obstacle ids whose AABB is within `radius` of pos_3d.

        Runs entirely in Python (no PyBullet calls) so it is cheap even for
        25 obstacles.  Callers then only invoke getClosestPoints on the small
        filtered set — typically 1-3 obstacles for a point in an open corridor.
        """
        if not self._obs_aabbs:
            return self.obstacle_ids
        px, py, pz = float(pos_3d[0]), float(pos_3d[1]), float(pos_3d[2])
        r2 = radius * radius
        result = []
        for oid, mn, mx in self._obs_aabbs:
            if mn is None:          # AABB unavailable — always include
                result.append(oid)
                continue
            # Closest point on the AABB to the query position
            cx = mn[0] if px < mn[0] else (mx[0] if px > mx[0] else px)
            cy = mn[1] if py < mn[1] else (mx[1] if py > mx[1] else py)
            cz = mn[2] if pz < mn[2] else (mx[2] if pz > mx[2] else pz)
            if (px-cx)**2 + (py-cy)**2 + (pz-cz)**2 <= r2:
                result.append(oid)
        return result

    def set_world_obstacles(self, ids, dynamic_ids=None):
        """Online mode: register all world obstacle IDs but reveal none yet.

        Call instead of set_obstacles() when using --mode online.
        Obstacles are gradually uncovered as the robot moves within
        sense_radius by calling _discover_obstacles() each tick.
        """
        self._dynamic_ids = set(dynamic_ids) if dynamic_ids else set()
        self._known_ids: set = set()
        # Cache AABBs for all world obstacles upfront.
        # Static obstacles → cached at this call; dynamic → None (refreshed live).
        self._world_aabbs: list = []
        for oid in ids:
            if oid in self._dynamic_ids:
                self._world_aabbs.append((oid, None, None))
                continue
            try:
                mn, mx = p.getAABB(oid, physicsClientId=self.cid)
                self._world_aabbs.append((oid, np.array(mn), np.array(mx)))
            except Exception:
                self._world_aabbs.append((oid, None, None))
        # Start with no known obstacles — is_free() sees nothing yet
        self.obstacle_ids = []
        self._obs_aabbs = []

    def _discover_obstacles(self, pos, sense_radius):
        """Reveal world obstacles newly within sense_radius of pos.

        Returns a list of newly discovered obstacle IDs (empty if none new).
        Updates obstacle_ids and _obs_aabbs immediately so that subsequent
        is_free() / edge_free() calls use the expanded known set.
        No-op if set_world_obstacles() was never called.
        """
        if not hasattr(self, '_world_aabbs'):
            return []
        pos_3d = self._to_3d(pos)
        px, py, pz = float(pos_3d[0]), float(pos_3d[1]), float(pos_3d[2])
        r2 = sense_radius * sense_radius
        newly_found = []
        for oid, mn, mx in self._world_aabbs:
            if oid in self._known_ids:
                continue
            if mn is None:
                # Dynamic obstacle — refresh AABB from live PyBullet state
                try:
                    mn2, mx2 = p.getAABB(oid, physicsClientId=self.cid)
                    mn_a, mx_a = np.array(mn2), np.array(mx2)
                except Exception:
                    newly_found.append(oid)
                    self._known_ids.add(oid)
                    continue
            else:
                mn_a, mx_a = mn, mx
            cx = mn_a[0] if px < mn_a[0] else (mx_a[0] if px > mx_a[0] else px)
            cy = mn_a[1] if py < mn_a[1] else (mx_a[1] if py > mx_a[1] else py)
            cz = mn_a[2] if pz < mn_a[2] else (mx_a[2] if pz > mx_a[2] else pz)
            if (px - cx) ** 2 + (py - cy) ** 2 + (pz - cz) ** 2 <= r2:
                newly_found.append(oid)
                self._known_ids.add(oid)
        if newly_found:
            self.obstacle_ids = list(self._known_ids)
            known = self._known_ids
            self._obs_aabbs = [
                (oid, mn, mx) for oid, mn, mx in self._world_aabbs
                if oid in known
            ]
        return newly_found

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
        # Only test obstacles whose AABB is within robot_radius of the probe
        # centre — cuts ~85% of getClosestPoints calls in open corridors.
        for oid in self._obstacles_near(pos, self.radius + 0.05):
            if p.getClosestPoints(
                    self._probe, oid, 0.0, physicsClientId=self.cid):
                return False
        return True

    def edge_free(self, v1, v2):
        a = np.asarray(v1, dtype=float)
        b = np.asarray(v2, dtype=float)
        length = np.linalg.norm(b - a)
        # Step = robot diameter (2×radius): safe since all obstacles are wider
        # than a single robot body, and we detect them before overlap.
        # Previously radius×0.5 — this is 2× faster with equivalent safety.
        n = max(int(length / (self.radius * 2.0)), 2)
        for i in range(n + 1):
            if not self.is_free(tuple(a + (i / n) * (b - a))):
                return False
        return True

    def _candidate_pairs(self, nodes, dmax, max_neighbors=None):
        """Return (i,j) index pairs within dmax via KD-tree.

        If max_neighbors is set, each node connects to at most that many
        nearest neighbours — caps edge count at N*max_neighbors/2 instead
        of O(N²) for dense roadmaps (e.g. office with N=1000, dmax=3.5).
        """
        if len(nodes) < 2:
            return []
        coords = np.array([n[:self.dim] for n in nodes], dtype=float)
        tree = cKDTree(coords)
        if max_neighbors is None:
            return list(tree.query_pairs(dmax))
        k = min(max_neighbors + 1, len(nodes))  # +1: query includes self
        dists, idxs = tree.query(coords, k=k)
        pairs = set()
        for i in range(len(nodes)):
            for j_pos in range(k):
                j = int(idxs[i, j_pos])
                d = float(dists[i, j_pos])
                if j != i and d <= dmax:
                    pairs.add((min(i, j), max(i, j)))
        return list(pairs)

    def clearance(self, pt):
        """Min distance from robot centre at *pt* to any obstacle surface.

        Capped at MAX_CLR (4 m): clearance beyond that is irrelevant to the
        risk formula (beta/clr → 0) and lets the AABB filter skip far walls.
        """
        _MAX_CLR = 4.0
        pos = self._to_3d(pt)
        p.resetBasePositionAndOrientation(
            self._probe, pos, [0, 0, 0, 1], physicsClientId=self.cid)
        min_d = _MAX_CLR
        for oid in self._obstacles_near(pos, _MAX_CLR):
            contacts = p.getClosestPoints(
                self._probe, oid, _MAX_CLR, physicsClientId=self.cid)
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
