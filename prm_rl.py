"""
PRM-RL planner — replaces the straight-line local planner with
a learned RL policy for edge validation.

From the proposal (Equation 3):
    P_success(qi, qj) = (1/K) * sum_{k=1}^{K} 1[agent reaches qj from qi]
    Edge added if P_success >= tau

Edge weight = average trajectory length of successful rollouts
(not Euclidean distance).

Inherits all collision checking, sampling, and graph infrastructure
from PRMBase.
"""

import numpy as np
import networkx as nx
import pybullet as p

from prm_base import PRMBase


class PRMRL(PRMBase):
    name = "PRM-RL"

    def __init__(self, client_id, workspace_bounds, robot_radius=0.15,
                 dim=2, fixed_z=None, policy=None, n_rollouts=10,
                 success_threshold=0.6, rollout_max_steps=100,
                 rollout_step_size=0.3, n_rays=16, ray_length=3.0,
                 goal_tol=None):
        """
        Parameters
        ----------
        policy : LocalPolicy or ReactivePolicy
            The trained RL local planner. Must have .predict(pos, goal, rays).
        n_rollouts : int
            K in Equation 3 — number of Monte Carlo rollouts per edge.
        success_threshold : float
            tau — minimum success rate to add an edge.
        rollout_max_steps : int
            Max steps per rollout before declaring failure.
        rollout_step_size : float
            How far the agent moves per step during rollouts.
        n_rays : int
            Number of lidar rays for observations.
        ray_length : float
            Max sensing range for lidar rays.
        goal_tol : float
            Distance threshold to count a rollout as "reached goal".
        """
        super().__init__(client_id, workspace_bounds, robot_radius,
                         dim, fixed_z)

        if policy is None:
            raise ValueError(
                "PRMRL requires a trained policy. Pass policy=LocalPolicy(...) "
                "or policy=ReactivePolicy() for testing."
            )
        self.policy = policy
        self.K = n_rollouts
        self.tau = success_threshold
        self.rollout_max_steps = rollout_max_steps
        self.rollout_step_size = rollout_step_size
        self.n_rays = n_rays
        self.ray_length = ray_length
        self.goal_tol = goal_tol or (robot_radius * 3)

        # Pre-compute ray angles (must match rl_env.py)
        self._ray_angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)

   

    def _cast_rays(self, pos):
        """Cast n_rays from pos, return normalised hit fractions."""
        dists = np.ones(self.n_rays, dtype=np.float64)
        z = self.fixed_z

        from_pos = [pos[0], pos[1], z]
        for i, angle in enumerate(self._ray_angles):
            dx = self.ray_length * np.cos(angle)
            dy = self.ray_length * np.sin(angle)
            to_pos = [pos[0] + dx, pos[1] + dy, z]

            result = p.rayTest(from_pos, to_pos, physicsClientId=self.cid)
            if result and result[0][0] != -1:
                dists[i] = result[0][2]

        return dists

    

    def _mc_validate_edge(self, v_start, v_goal):
        """
        Run K rollouts of the RL policy from v_start toward v_goal.

        Returns
        -------
        success_rate : float
            Fraction of rollouts that reached v_goal.
        avg_traj_length : float
            Average path length of successful rollouts.
            Returns inf if no rollout succeeded.
        """
        successes = 0
        traj_lengths = []

        start = np.array(v_start, dtype=np.float64)
        goal = np.array(v_goal, dtype=np.float64)

        for k in range(self.K):
            pos = start.copy()
            traj_len = 0.0
            reached = False

            for step in range(self.rollout_max_steps):
                # Check if we've reached the goal
                dist_to_goal = np.linalg.norm(pos - goal)
                if dist_to_goal < self.goal_tol:
                    reached = True
                    break

                # Get observation and action from policy
                rays = self._cast_rays(pos)
                action = self.policy.predict(pos, goal, rays)

                # Execute action
                action = np.clip(action, -1.0, 1.0)
                move = action * self.rollout_step_size
                new_pos = pos + move

                # Clamp to bounds
                for i in range(self.dim):
                    new_pos[i] = np.clip(
                        new_pos[i],
                        self.bounds[i][0],
                        self.bounds[i][1],
                    )

                # Collision check — if collision, rollout fails
                if not self.is_free(tuple(new_pos)):
                    break

                traj_len += np.linalg.norm(new_pos - pos)
                pos = new_pos

            if reached:
                successes += 1
                traj_lengths.append(traj_len)

        success_rate = successes / self.K
        avg_length = (
            float(np.mean(traj_lengths)) if traj_lengths else float("inf")
        )

        return success_rate, avg_length

    

    def construct(self, N, dmax, verbose=False, **kw):
        """
        Build roadmap with RL-validated edges.

        1. Sample N collision-free nodes (same as baseline).
        2. For each candidate edge (within dmax), run Monte Carlo
           rollouts instead of straight-line collision check.
        3. Add edge only if success_rate >= tau.
        4. Edge weight = average trajectory length (not Euclidean).
        """
        for k in range(N):
            vnew = self.random_sample()
            self.G.add_node(vnew)

            # Find neighbours within dmax
            neighbours = [
                v for v in self.G.nodes
                if v != vnew and self.dist(v, vnew) < dmax
            ]

            for v in neighbours:
                # Quick pre-filter: skip if straight line is definitely
                # blocked AND nodes are far apart (saves expensive rollouts)
                if not self.edge_free(v, vnew):
                    if self.dist(v, vnew) > dmax * 0.5:
                        continue

                # Monte Carlo validation (both directions)
                sr_fwd, len_fwd = self._mc_validate_edge(v, vnew)
                sr_bwd, len_bwd = self._mc_validate_edge(vnew, v)

                # Use minimum success rate (edge must work both ways)
                success_rate = min(sr_fwd, sr_bwd)

                if success_rate >= self.tau:
                    avg_len = (len_fwd + len_bwd) / 2
                    self.G.add_edge(v, vnew, weight=avg_len)

            if verbose and (k + 1) % 50 == 0:
                print(f"  [{self.name}] {k+1}/{N}  "
                      f"({self.G.number_of_edges()} edges)")

        return self.G

    

    def find(self, start, goal, dmax, max_retries=5, extra=50, **kw):
        """
        Connect start/goal into the graph and search.
        Uses standard collision checking for start/goal connections
        to keep query time reasonable.
        """
        for node in [start, goal]:
            if node not in self.G:
                self.G.add_node(node)
                for v in list(self.G.nodes):
                    if v != node and self.dist(v, node) < dmax:
                        if self.edge_free(v, node):
                            self.G.add_edge(
                                v, node, weight=self.dist(v, node)
                            )

        for attempt in range(max_retries):
            try:
                path = nx.astar_path(
                    self.G, start, goal,
                    heuristic=self.dist, weight="weight",
                )
                cost = nx.astar_path_length(
                    self.G, start, goal,
                    heuristic=self.dist, weight="weight",
                )
                return path, cost
            except nx.NetworkXNoPath:
                for _ in range(extra):
                    vnew = self.random_sample()
                    self.G.add_node(vnew)
                    for v in list(self.G.nodes):
                        if v != vnew and self.dist(v, vnew) < dmax:
                            if self.edge_free(v, vnew):
                                self.G.add_edge(
                                    v, vnew,
                                    weight=self.dist(v, vnew),
                                )

        return None, float("inf")

    def replan(self, current, goal, dmax, **kw):
        """Replan from current position, invalidating edges near
        dynamic obstacles first."""
        dyn_positions = kw.get("dyn_positions", [])
        affect_radius = kw.get("affect_radius", dmax * 0.6)

        if dyn_positions:
            self._invalidate_near(dyn_positions, affect_radius)

        return self.find(current, goal, dmax, **kw)

    def _invalidate_near(self, positions, radius):
        """Remove edges that pass near dynamic obstacle positions."""
        to_remove = []
        for u, v in self.G.edges:
            midpoint = tuple(
                (np.array(u) + np.array(v)) / 2
            )
            for obs_pos in positions:
                obs = np.array(obs_pos[:self.dim])
                if np.linalg.norm(np.array(midpoint) - obs) < radius:
                    to_remove.append((u, v))
                    break
        self.G.remove_edges_from(to_remove)

