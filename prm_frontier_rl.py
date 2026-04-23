"""
Frontier-RL PRM — hybrid planner combining:

  1. RL-validated edges        (from PRM-RL)
  2. Risk-weighted edge costs  (from RiskAware PRM)
  3. Explicit collision-risk threshold (from Risk-Aware PRM paper)
  4. Incremental occupancy grid built from lidar during execution
  5. Frontier-biased sampling  — new PRM nodes placed at the boundary
     of explored/unexplored space, addressing the cold-start problem

Architecture:
  - OccupancyGrid: discretises the workspace; cells are UNKNOWN,
    FREE, or OCCUPIED.  Updated from lidar rays cast during
    construct() and replan().
  - Frontier detection: cells that are FREE and adjacent to UNKNOWN.
  - Sampling: 40 % uniform, 20 % frontier-biased, 20 % risk-biased,
    20 % failure-biased (near locations where RL rollouts failed —
    the planner learns its own weaknesses and densifies the roadmap
    where the policy struggles).
  - Edge admission: three-part pipeline:
      1. Feasibility: RL rollout success rate >= tau
      2. Risk: collision rate <= R_ad
      3. Preference: cost = avg_trajectory_length + beta / min_clearance
  - Edge weight: avg_trajectory_length + beta / min_clearance
    (combines PRM-RL trajectory cost with RiskAware penalty).
"""

import numpy as np
import networkx as nx
import pybullet as p

from prm_base import PRMBase


UNKNOWN  = 0
FREE     = 1
OCCUPIED = 2


class OccupancyGrid:
    """
    Simple 2D grid that tracks what the robot has observed.

    Parameters
    ----------
    bounds : list of (lo, hi)
        Workspace limits, e.g. [(0, 12), (0, 12)].
    resolution : float
        Cell size in metres.
    """

    def __init__(self, bounds, resolution=0.3):
        self.bounds = bounds
        self.res = resolution
        self.nx = int(np.ceil((bounds[0][1] - bounds[0][0]) / resolution))
        self.ny = int(np.ceil((bounds[1][1] - bounds[1][0]) / resolution))
        self.grid = np.full((self.nx, self.ny), UNKNOWN, dtype=np.int8)

    def _to_cell(self, x, y):
        ci = int((x - self.bounds[0][0]) / self.res)
        cj = int((y - self.bounds[1][0]) / self.res)
        ci = np.clip(ci, 0, self.nx - 1)
        cj = np.clip(cj, 0, self.ny - 1)
        return ci, cj

    def _to_world(self, ci, cj):
        x = self.bounds[0][0] + (ci + 0.5) * self.res
        y = self.bounds[1][0] + (cj + 0.5) * self.res
        return x, y

    def update_from_rays(self, robot_pos, ray_angles, ray_dists, ray_length):
        rx, ry = robot_pos[0], robot_pos[1]
        for angle, frac in zip(ray_angles, ray_dists):
            actual_dist = frac * ray_length
            n_steps = max(int(actual_dist / (self.res * 0.5)), 1)
            for s in range(n_steps + 1):
                d = (s / n_steps) * actual_dist
                px = rx + d * np.cos(angle)
                py = ry + d * np.sin(angle)
                ci, cj = self._to_cell(px, py)
                if s < n_steps:
                    if self.grid[ci, cj] == UNKNOWN:
                        self.grid[ci, cj] = FREE
                else:
                    if frac < 0.99:
                        self.grid[ci, cj] = OCCUPIED
                    else:
                        if self.grid[ci, cj] == UNKNOWN:
                            self.grid[ci, cj] = FREE

    def get_frontier_cells(self):
        frontiers = []
        for ci in range(1, self.nx - 1):
            for cj in range(1, self.ny - 1):
                if self.grid[ci, cj] != FREE:
                    continue
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if self.grid[ci + di, cj + dj] == UNKNOWN:
                        frontiers.append(self._to_world(ci, cj))
                        break
        return frontiers

    def explored_fraction(self):
        known = np.sum(self.grid != UNKNOWN)
        return known / (self.nx * self.ny)



class FrontierRLPRM(PRMBase):
    name = "FrontierRL"

    def __init__(self, client_id, workspace_bounds, robot_radius=0.15,
                 dim=2, fixed_z=None,
                 # RL params
                 policy=None, n_rollouts=5, success_threshold=0.5,
                 rollout_max_steps=100, rollout_step_size=0.3,
                 n_rays=16, ray_length=3.0, goal_tol=None,
                 # Risk-aware params
                 risk_beta=0.5,
                 risk_admissible=0.3,
                 # Sampling mix
                 uniform_frac=0.40,
                 frontier_frac=0.20,
                 risk_frac=0.20,
                 failure_frac=0.20,
                 frontier_sigma=0.5,
                 risk_sigma=0.8,
                 failure_sigma=0.8,
                 # Occupancy grid
                 grid_resolution=0.3,
                 # Repair
                 repair_samples=50):
        super().__init__(client_id, workspace_bounds, robot_radius,
                         dim, fixed_z)

        if policy is None:
            raise ValueError(
                "FrontierRLPRM requires a policy. Pass policy=LocalPolicy(...) "
                "or policy=ReactivePolicy() for testing."
            )

        # RL
        self.policy = policy
        self.K = n_rollouts
        self.tau = success_threshold
        self.rollout_max_steps = rollout_max_steps
        self.rollout_step_size = rollout_step_size
        self.n_rays = n_rays
        self.ray_length = ray_length
        self.goal_tol = goal_tol or (robot_radius * 3)
        self._ray_angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)

        # Risk
        self.risk_beta = risk_beta
        self.R_ad = risk_admissible

        # Sampling
        self.uniform_frac = uniform_frac
        self.frontier_frac = frontier_frac
        self.risk_frac = risk_frac
        self.failure_frac = failure_frac
        self.frontier_sigma = frontier_sigma
        self.risk_sigma = risk_sigma
        self.failure_sigma = failure_sigma

        # Failure tracking
        self.failure_points = []

        # Occupancy grid
        self.occ_grid = OccupancyGrid(workspace_bounds, grid_resolution)

        # Repair
        self.repair_samples = repair_samples



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

    def _sense_and_update(self, pos):
        rays = self._cast_rays(pos)
        self.occ_grid.update_from_rays(pos, self._ray_angles, rays,
                                        self.ray_length)
        return rays

    

    def _clearance_along_edge(self, u, v):
        a = np.asarray(u, dtype=float)
        b = np.asarray(v, dtype=float)
        length = np.linalg.norm(b - a)
        K = max(3, int(length / 0.5))
        min_clr = float('inf')
        for i in range(K + 1):
            pt = tuple(a + (i / K) * (b - a))
            min_clr = min(min_clr, self.clearance(pt))
        return min_clr

    

    def _mc_validate_edge(self, v_start, v_goal):
        """
        Run K rollouts. Returns (success_rate, collision_rate, avg_traj_length).

        success_rate  = # reached goal / K
        collision_rate = # ended in collision / K
        avg_traj_length = mean path length of successful rollouts
        """
        successes = 0
        collisions = 0
        traj_lengths = []
        start = np.array(v_start, dtype=np.float64)
        goal = np.array(v_goal, dtype=np.float64)

        for k in range(self.K):
            pos = start.copy()
            traj_len = 0.0
            reached = False
            collided = False

            for step in range(self.rollout_max_steps):
                if np.linalg.norm(pos - goal) < self.goal_tol:
                    reached = True
                    break

                rays = self._cast_rays(pos)
                action = self.policy.predict(pos, goal, rays)
                action = np.clip(action, -1.0, 1.0)
                new_pos = pos + action * self.rollout_step_size

                for i in range(self.dim):
                    new_pos[i] = np.clip(new_pos[i],
                                         self.bounds[i][0],
                                         self.bounds[i][1])

                if not self.is_free(tuple(new_pos)):
                    self.failure_points.append(tuple(pos))
                    collided = True
                    break

                traj_len += np.linalg.norm(new_pos - pos)
                pos = new_pos

            if reached:
                successes += 1
                traj_lengths.append(traj_len)
            else:
                self.failure_points.append(tuple(pos))

            if collided:
                collisions += 1

        success_rate = successes / self.K
        collision_rate = collisions / self.K
        avg_length = (float(np.mean(traj_lengths))
                      if traj_lengths else float("inf"))
        return success_rate, collision_rate, avg_length

    

    def _hybrid_edge_cost(self, u, v, traj_length):
        clr = max(self._clearance_along_edge(u, v), 1e-6)
        return traj_length + self.risk_beta / clr

    

    def _add_vertex_rl(self, v, dmax):
        """
        Three-part edge admission:
          1. Feasibility: success_rate >= tau
          2. Risk: collision_rate <= R_ad
          3. Preference: cost = traj_length + beta / clearance
        """
        self.G.add_node(v)

        neighbours = [
            u for u in self.G.nodes
            if u != v and self.dist(u, v) < dmax
        ]

        for u in neighbours:
            if not self.edge_free(u, v):
                if self.dist(u, v) > dmax * 0.5:
                    continue

            sr_fwd, cr_fwd, len_fwd = self._mc_validate_edge(u, v)
            sr_bwd, cr_bwd, len_bwd = self._mc_validate_edge(v, u)

            # 1. Feasibility: success rate must exceed threshold
            success_rate = min(sr_fwd, sr_bwd)
            # 2. Risk: collision rate must be below admissible bound
            collision_rate = max(cr_fwd, cr_bwd)

            if success_rate >= self.tau and collision_rate <= self.R_ad:
                # 3. Preference: hybrid cost
                avg_traj = (len_fwd + len_bwd) / 2
                cost = self._hybrid_edge_cost(u, v, avg_traj)
                self.G.add_edge(u, v, weight=cost)

    def _add_vertex_fast(self, v, dmax):
        self.G.add_node(v)
        for u in list(self.G.nodes):
            if u != v and self.dist(u, v) < dmax:
                if self.edge_free(u, v):
                    clr = max(self._clearance_along_edge(u, v), 1e-6)
                    cost = self.dist(u, v) + self.risk_beta / clr
                    self.G.add_edge(u, v, weight=cost)

    

    def _frontier_sample(self):
        frontiers = self.occ_grid.get_frontier_cells()
        if not frontiers:
            return self.random_sample()
        idx = np.random.randint(len(frontiers))
        anchor = frontiers[idx]
        for _ in range(50):
            noise = np.random.normal(0, self.frontier_sigma, size=self.dim)
            candidate = tuple(
                float(np.clip(anchor[i] + noise[i],
                              self.bounds[i][0], self.bounds[i][1]))
                for i in range(self.dim)
            )
            if self.is_free(candidate):
                return candidate
        return self.random_sample()

    def _risk_sample(self):
        nodes = list(self.G.nodes)
        if len(nodes) < 5:
            return self.random_sample()
        subset_size = min(len(nodes), 50)
        indices = np.random.choice(len(nodes), subset_size, replace=False)
        subset = [nodes[i] for i in indices]
        node_clr = {n: self.clearance(n) for n in subset}
        sorted_nodes = sorted(node_clr, key=lambda n: node_clr[n])
        tight_nodes = sorted_nodes[:max(1, len(sorted_nodes) // 5)]
        anchor = tight_nodes[np.random.randint(len(tight_nodes))]
        for _ in range(50):
            noise = np.random.normal(0, self.risk_sigma, size=self.dim)
            candidate = tuple(
                float(np.clip(anchor[i] + noise[i],
                              self.bounds[i][0], self.bounds[i][1]))
                for i in range(self.dim)
            )
            if self.is_free(candidate):
                return candidate
        return self.random_sample()

    def _failure_biased_sample(self):
        if not self.failure_points:
            return self.random_sample()
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

    

    def construct(self, N, dmax, verbose=False, **kw):
        n_uniform  = int(N * self.uniform_frac)
        n_frontier = int(N * self.frontier_frac)
        n_risk     = int(N * self.risk_frac)
        n_failure  = N - n_uniform - n_frontier - n_risk

        for k in range(n_uniform):
            vnew = self.random_sample()
            self._sense_and_update(vnew)
            self._add_vertex_rl(vnew, dmax)
            if verbose and (k + 1) % 50 == 0:
                explored = self.occ_grid.explored_fraction() * 100
                print(f"  [{self.name}] uniform {k+1}/{n_uniform}  "
                      f"({self.G.number_of_edges()} edges, "
                      f"{explored:.0f}% explored)")

        if verbose:
            frontiers = self.occ_grid.get_frontier_cells()
            print(f"  [{self.name}] frontier phase ({n_frontier} samples, "
                  f"{len(frontiers)} frontier cells)...")

        for k in range(n_frontier):
            vnew = self._frontier_sample()
            self._sense_and_update(vnew)
            self._add_vertex_rl(vnew, dmax)
            if verbose and (k + 1) % 50 == 0:
                explored = self.occ_grid.explored_fraction() * 100
                print(f"  [{self.name}] frontier {k+1}/{n_frontier}  "
                      f"({explored:.0f}% explored)")

        if verbose:
            print(f"  [{self.name}] risk phase ({n_risk} samples)...")

        for k in range(n_risk):
            vnew = self._risk_sample()
            self._sense_and_update(vnew)
            self._add_vertex_rl(vnew, dmax)

        if verbose:
            print(f"  [{self.name}] failure phase ({n_failure} samples, "
                  f"{len(self.failure_points)} failure points collected)...")

        for k in range(n_failure):
            vnew = self._failure_biased_sample()
            self._sense_and_update(vnew)
            self._add_vertex_rl(vnew, dmax)

        if verbose:
            explored = self.occ_grid.explored_fraction() * 100
            print(f"  [{self.name}] done — "
                  f"{self.G.number_of_nodes()} nodes, "
                  f"{self.G.number_of_edges()} edges, "
                  f"{explored:.0f}% explored")

        return self.G

    def find(self, start, goal, dmax, max_retries=5, extra=50, **kw):
        self._sense_and_update(start)
        self._sense_and_update(goal)
        for node in [start, goal]:
            if node not in self.G:
                self._add_vertex_fast(node, dmax)
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
                for _ in range(extra):
                    vnew = self._frontier_sample()
                    self._sense_and_update(vnew)
                    self._add_vertex_fast(vnew, dmax)
        return None, float("inf")

    def replan(self, current, goal, dmax, **kw):
        self._sense_and_update(current)
        dyn_positions = kw.get("dyn_positions", [])
        affect_radius = kw.get("affect_radius", dmax * 0.6)
        if dyn_positions:
            to_remove = []
            for u, v in self.G.edges:
                midpoint = tuple((np.array(u) + np.array(v)) / 2)
                for obs_pos in dyn_positions:
                    obs = np.array(obs_pos[:self.dim])
                    if np.linalg.norm(np.array(midpoint) - obs) < affect_radius:
                        to_remove.append((u, v))
                        break
            self.G.remove_edges_from(to_remove)
        added = 0
        attempts = 0
        while added < self.repair_samples and attempts < self.repair_samples * 5:
            attempts += 1
            roll = np.random.random()
            if roll < 0.33:
                candidate = self._frontier_sample()
            elif roll < 0.66 and self.failure_points:
                candidate = self._failure_biased_sample()
            else:
                noise = np.random.normal(0, dmax * 0.4, size=self.dim)
                candidate = tuple(
                    float(np.clip(current[i] + noise[i],
                                  self.bounds[i][0], self.bounds[i][1]))
                    for i in range(self.dim)
                )
                if not self.is_free(candidate):
                    continue
            self._sense_and_update(candidate)
            self._add_vertex_fast(candidate, dmax)
            added += 1
        return self.find(current, goal, dmax, **kw)