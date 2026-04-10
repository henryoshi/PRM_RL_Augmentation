"""
PRM (Probabilistic Roadmap) planner using PyBullet for collision detection.
Adapted from occupancy-grid PRM to work with arbitrary PyBullet environments.
Supports both 2D (ground-plane navigation) and 3D (drone/aerial) planning.
"""

import numpy as np
import networkx as nx
import pybullet as p


class PRMPlanner:
    def __init__(self, client_id, workspace_bounds, robot_radius=0.15,
                 dim=2, fixed_z=None):
        """
        Parameters
        ----------
        client_id : int
            PyBullet physics client ID.
        workspace_bounds : list of (min, max)
            Sampling bounds per axis, e.g. [(0,10),(0,10)] for 2D.
        robot_radius : float
            Collision sphere radius representing the robot.
        dim : int
            2 for ground-plane planning, 3 for full 3D.
        fixed_z : float or None
            Height of the robot centre for 2D mode. Defaults to robot_radius.
        """
        self.cid = client_id
        self.bounds = workspace_bounds
        self.radius = robot_radius
        self.dim = dim
        self.fixed_z = fixed_z if fixed_z is not None else robot_radius + 0.01
        self.G = nx.Graph()
        self.obstacle_ids = []

        # Probe sphere used for point-collision queries
        self._col_shape = p.createCollisionShape(
            p.GEOM_SPHERE, radius=robot_radius, physicsClientId=self.cid)
        self._probe = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=self._col_shape,
            basePosition=[0, 0, -100],
            physicsClientId=self.cid)

    # ------------------------------------------------------------------
    # Environment interface
    # ------------------------------------------------------------------
    def set_obstacles(self, obstacle_ids):
        """Register PyBullet body IDs that act as obstacles."""
        self.obstacle_ids = list(obstacle_ids)

    # ------------------------------------------------------------------
    # Collision helpers
    # ------------------------------------------------------------------
    def _to_3d(self, point):
        if self.dim == 2:
            return [point[0], point[1], self.fixed_z]
        return [point[0], point[1], point[2]]

    def is_free(self, point):
        """Return True if placing the robot sphere at *point* causes no collision."""
        pos = self._to_3d(point)
        p.resetBasePositionAndOrientation(
            self._probe, pos, [0, 0, 0, 1], physicsClientId=self.cid)
        for obs_id in self.obstacle_ids:
            contacts = p.getClosestPoints(
                self._probe, obs_id, distance=0.0, physicsClientId=self.cid)
            if contacts:
                return False
        return True

    def edge_free(self, v1, v2):
        """Check collision along the straight-line segment v1 -> v2."""
        a, b = np.array(v1), np.array(v2)
        dist = np.linalg.norm(b - a)
        # Step size ≤ half the robot radius for thorough coverage
        n_steps = max(int(dist / (self.radius * 0.5)), 2)
        for i in range(n_steps + 1):
            t = i / n_steps
            pt = tuple(a + t * (b - a))
            if not self.is_free(pt):
                return False
        return True

    # ------------------------------------------------------------------
    # Core PRM methods (mirrors the original algorithm structure)
    # ------------------------------------------------------------------
    def _dist(self, v1, v2):
        return float(np.linalg.norm(np.array(v1) - np.array(v2)))

    def RandomSample(self):
        """Rejection-sample a collision-free configuration."""
        while True:
            pt = tuple(
                np.random.uniform(lo, hi)
                for lo, hi in self.bounds[:self.dim])
            if self.is_free(pt):
                return pt

    def AddVertex(self, vnew, dmax):
        """Add vnew to the graph, connecting to neighbours within dmax."""
        self.G.add_node(vnew)
        for v in list(self.G.nodes):
            if v != vnew and self._dist(v, vnew) < dmax:
                if self.edge_free(v, vnew):
                    self.G.add_edge(v, vnew, weight=self._dist(v, vnew))

    def ConstructPRM(self, N, dmax, verbose=True):
        """Build the roadmap with N random samples."""
        for k in range(N):
            vnew = self.RandomSample()
            self.AddVertex(vnew, dmax)
            if verbose and (k + 1) % 50 == 0:
                print(f"  PRM: {k+1}/{N} nodes, {self.G.number_of_edges()} edges")
        return self.G

    def find(self, start, goal, dmax, max_retries=10, samples_per_retry=100):
        """
        Connect start/goal into the roadmap and search for a path.
        Adds more samples on failure (same strategy as the original code).
        """
        self.AddVertex(start, dmax)
        self.AddVertex(goal, dmax)

        for attempt in range(max_retries):
            try:
                path = nx.astar_path(
                    self.G, start, goal,
                    heuristic=self._dist, weight='weight')
                dist = nx.astar_path_length(
                    self.G, start, goal,
                    heuristic=self._dist, weight='weight')
                return path, dist
            except nx.NetworkXNoPath:
                if verbose := True:
                    print(f"  No path (attempt {attempt+1}), "
                          f"adding {samples_per_retry} samples...")
                for _ in range(samples_per_retry):
                    self.AddVertex(self.RandomSample(), dmax)

        raise RuntimeError("Path not found after max retries.")
