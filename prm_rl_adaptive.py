"""
Adaptive PRM-RL — failure-guided sampling for PRM construction.

After an initial uniform sampling phase, the planner biases new
PRM nodes toward clusters of failure points.  This creates denser
roadmaps in regions where the RL policy struggles, giving A* more
route options around the hard spots.

Architecture:
  Phase 1 (70% of N): uniform sampling + MC validation (same as PRM-RL),
      collecting failure points along the way.
  Phase 2 (30% of N): failure-biased sampling — Gaussian samples
      centered on failure locations, then MC-validated as usual.
"""

import numpy as np
import networkx as nx
import pybullet as p

from prm_base import PRMBase


class AdaptivePRMRL(PRMBase):
    name = "AdaptiveRL"

    def __init__(self, client_id, workspace_bounds, robot_radius=0.15,
                 dim=2, fixed_z=None, policy=None, n_rollouts=10,
                 success_threshold=0.6, rollout_max_steps=100,
                 rollout_step_size=0.3, n_rays=16, ray_length=3.0,
                 goal_tol=None,
                 # Adaptive sampling params
                 failure_frac=0.3,
                 failure_sigma=0.8):
        """
        Parameters
        ----------
        policy : LocalPolicy or ReactivePolicy
        n_rollouts, success_threshold : MC edge validation params
        failure_frac : float
            Fraction of total samples reserved for failure-biased phase.
        failure_sigma : float
            Gaussian spread (metres) around failure points when sampling.
        """
        super().__init__(client_id, workspace_bounds, robot_radius,
                         dim, fixed_z)

        if policy is None:
            raise ValueError(
                "AdaptivePRMRL requires a policy. Pass policy=LocalPolicy(...) "
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
        self._ray_angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)

        # Adaptive sampling
        self.failure_frac = failure_frac
        self.failure_sigma = failure_sigma
        self.failure_points = []  # collected during MC validation



    def _cast_rays(self, pos):
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
        Records the position where each failed rollout ended.

        Returns
        -------
        success_rate : float
        avg_traj_length : float
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
                dist_to_goal = np.linalg.norm(pos - goal)
                if dist_to_goal < self.goal_tol:
                    reached = True
                    break

                rays = self._cast_rays(pos)
                action = self.policy.predict(pos, goal, rays)
                action = np.clip(action, -1.0, 1.0)
                new_pos = pos + action * self.rollout_step_size

                for i in range(self.dim):
                    new_pos[i] = np.clip(
                        new_pos[i],
                        self.bounds[i][0],
                        self.bounds[i][1],
                    )

                if not self.is_free(tuple(new_pos)):
                   
                    self.failure_points.append(tuple(pos))
                    break

                traj_len += np.linalg.norm(new_pos - pos)
                pos = new_pos

            if reached:
                successes += 1
                traj_lengths.append(traj_len)
            else:
                
                if not reached:
                    self.failure_points.append(tuple(pos))

        success_rate = successes / self.K
        avg_length = (
            float(np.mean(traj_lengths)) if traj_lengths else float("inf")
        )
        return success_rate, avg_length

    

    def _failure_biased_sample(self):
        """
        Sample a new node near a recorded failure point.
        Picks a random failure point and adds Gaussian noise.
        """
        if not self.failure_points:
            return self.random_sample()

        # Pick a random failure point
        idx = np.random.randint(len(self.failure_points))
        anchor = self.failure_points[idx]

        for _ in range(50):
            noise = np.random.normal(0, self.failure_sigma, size=self.dim)
            candidate = tuple(
                float(np.clip(anchor[i] + noise[i],
                              self.bounds[i][0], self.bounds[i][1]))
                for i in range(self.dim)
            )
            if self.is_free(candidate):
                return candidate

        return self.random_sample()

    

    def _add_vertex_mc(self, vnew, dmax):
        """Add vnew, validate edges with MC rollouts."""
        self.G.add_node(vnew)

        neighbours = [
            v for v in self.G.nodes
            if v != vnew and self.dist(v, vnew) < dmax
        ]

        for v in neighbours:
            if not self.edge_free(v, vnew):
                if self.dist(v, vnew) > dmax * 0.5:
                    continue

            sr_fwd, len_fwd = self._mc_validate_edge(v, vnew)
            sr_bwd, len_bwd = self._mc_validate_edge(vnew, v)
            success_rate = min(sr_fwd, sr_bwd)

            if success_rate >= self.tau:
                avg_len = (len_fwd + len_bwd) / 2
                self.G.add_edge(v, vnew, weight=avg_len)

    

    def construct(self, N, dmax, verbose=False, **kw):
        """
        Two-phase roadmap construction:

        Phase 1 (uniform): standard PRM-RL — uniform random samples
            with MC edge validation. Failure points are collected.

        Phase 2 (failure-biased): sample near failure locations to
            build denser roadmaps where the RL policy struggles.
        """
        n_uniform = int(N * (1.0 - self.failure_frac))
        n_failure = N - n_uniform

        # Phase 1: uniform sampling, collecting failures
        for k in range(n_uniform):
            vnew = self.random_sample()
            self._add_vertex_mc(vnew, dmax)

            if verbose and (k + 1) % 50 == 0:
                print(f"  [{self.name}] uniform {k+1}/{n_uniform}  "
                      f"({self.G.number_of_edges()} edges, "
                      f"{len(self.failure_points)} failures recorded)")

        if verbose:
            print(f"  [{self.name}] Phase 1 done: "
                  f"{len(self.failure_points)} failure points collected")
            print(f"  [{self.name}] Phase 2: {n_failure} failure-biased "
                  f"samples...")

        # Phase 2: failure-biased sampling
        for k in range(n_failure):
            vnew = self._failure_biased_sample()
            self._add_vertex_mc(vnew, dmax)

            if verbose and (k + 1) % 50 == 0:
                print(f"  [{self.name}] failure-biased {k+1}/{n_failure}  "
                      f"({self.G.number_of_edges()} edges)")

        if verbose:
            print(f"  [{self.name}] done — "
                  f"{self.G.number_of_nodes()} nodes, "
                  f"{self.G.number_of_edges()} edges")

        return self.G



    def find(self, start, goal, dmax, max_retries=5, extra=50, **kw):
        for node in [start, goal]:
            if node not in self.G:
                self.G.add_node(node)
                for v in list(self.G.nodes):
                    if v != node and self.dist(v, node) < dmax:
                        if self.edge_free(v, node):
                            self.G.add_edge(
                                v, node, weight=self.dist(v, node))

        for attempt in range(max_retries):
            try:
                path = nx.astar_path(
                    self.G, start, goal,
                    heuristic=self.dist, weight="weight")
                cost = nx.astar_path_length(
                    self.G, start, goal,
                    heuristic=self.dist, weight="weight")
                return path, cost
            except nx.NetworkXNoPath:
                # Use failure-biased samples for retries too
                for _ in range(extra):
                    if self.failure_points and np.random.random() < 0.5:
                        vnew = self._failure_biased_sample()
                    else:
                        vnew = self.random_sample()
                    self.G.add_node(vnew)
                    for v in list(self.G.nodes):
                        if v != vnew and self.dist(v, vnew) < dmax:
                            if self.edge_free(v, vnew):
                                self.G.add_edge(
                                    v, vnew, weight=self.dist(v, vnew))

        return None, float("inf")

    def replan(self, current, goal, dmax, **kw):
        dyn_positions = kw.get("dyn_positions", [])
        affect_radius = kw.get("affect_radius", dmax * 0.6)

        if dyn_positions:
            self._invalidate_near(dyn_positions, affect_radius)

        return self.find(current, goal, dmax, **kw)

    def _invalidate_near(self, positions, radius):
        to_remove = []
        for u, v in self.G.edges:
            midpoint = tuple((np.array(u) + np.array(v)) / 2)
            for obs_pos in positions:
                obs = np.array(obs_pos[:self.dim])
                if np.linalg.norm(np.array(midpoint) - obs) < radius:
                    to_remove.append((u, v))
                    break
        self.G.remove_edges_from(to_remove)