"""
Risk-Aware PRM — novel planner for cold-start navigation.

Three improvements over BasicPRM:

  1. Risk-weighted edges:   cost = dist(u,v) + beta / min_clearance(u→v)
     Paths near obstacles are penalised; A* prefers open corridors.

  2. Frontier-biased sampling:  (1-f)% uniform + f% Gaussian around the
     lowest-clearance nodes.  Adds density where the baseline fails.

  3. Local replan repair:  remove nodes near dynamic obstacles, add
     targeted samples, re-query A*.

All PyBullet interaction is delegated to PRMBase.
"""

import numpy as np
import networkx as nx
from scipy.spatial import KDTree as cKDTree
from prm_base import PRMBase


class RiskAwarePRM(PRMBase):
    name = "RiskAware"

    def __init__(self, client_id, workspace_bounds, robot_radius=0.15,
                 dim=2, fixed_z=None,
                 risk_beta=4.0,
                 frontier_frac=0.3,
                 frontier_sigma=0.8,
                 repair_samples=20,
                 min_edge_len=0.0,
                 max_neighbors=None,
                 min_clearance_threshold=None):
        super().__init__(client_id, workspace_bounds, robot_radius, dim,
                         fixed_z, min_edge_len=min_edge_len,
                         max_neighbors=max_neighbors)
        self.risk_beta = risk_beta
        self.frontier_frac = frontier_frac
        self.frontier_sigma = frontier_sigma
        self.repair_samples = repair_samples

        if min_clearance_threshold is None:
            self.min_clearance_threshold = robot_radius * 1.5
        else:
            self.min_clearance_threshold = min_clearance_threshold

        # Online-mode state (populated by online_step, None in offline mode)
        self._visited_positions = None
        self._visited_tree = None
        self._sense_radius = None

        # Sticky exploration target — prevents oscillation in online mode.
        # Once set, the robot follows this target until it arrives, the
        # target is pruned, or a full path to the goal is found.
        self._exploration_target = None

    # ── Risk-weighted edge helpers (M1) ──────────────────────────────

    def _clearance_along_edge(self, u, v):
        """Minimum clearance sampled along segment u→v."""
        a = np.asarray(u, dtype=float)
        b = np.asarray(v, dtype=float)
        length = np.linalg.norm(b - a)
        K = max(3, int(length / 1.0))
        min_clr = float('inf')
        for i in range(K + 1):
            pt = tuple(a + (i / K) * (b - a))
            min_clr = min(min_clr, self.clearance(pt))
        return min_clr

    def _risk_cost(self, u, v):
        """Public API: risk-weighted cost for edge u→v.

        cost = euclidean_dist + beta / min_clearance
        Used by verify_risk_aware.py tests.
        """
        d = self.dist(u, v)
        clr = max(self._clearance_along_edge(u, v), 1e-6)
        return d + self.risk_beta / (clr * clr)

    # Alias for external callers
    edge_cost = _risk_cost

    def _try_connect(self, u, v, d):
        """Connect u↔v with risk-weighted cost if the edge is free.

        Safety: if clearance < 2×radius, only allow short edges (< 4×radius)
        to prevent corner-cutting diagonals near walls.
        """
        if not self.edge_free(u, v):
            return
        clr = max(self._clearance_along_edge(u, v), 1e-6)
        if clr < self.radius * 2.0 and d > self.radius * 4.0:
            return
        weight = d + self.risk_beta / (clr * clr)
        self.G.add_edge(u, v, weight=weight)

    def _add_vertex(self, v, dmax):
        """Insert a node and connect to nearby neighbours."""
        self.G.add_node(v)
        nodes = [u for u in self.G.nodes if u != v]
        if not nodes:
            return
        if self.max_neighbors is not None:
            all_nodes = [v] + nodes
            for i, j in self._candidate_pairs(all_nodes, dmax, self.max_neighbors):
                if 0 in (i, j):
                    u = all_nodes[j if i == 0 else i]
                    d = self.dist(u, v)
                    if d >= self.min_edge_len:
                        self._try_connect(u, v, d)
        else:
            for u in nodes:
                d = self.dist(u, v)
                if self.min_edge_len <= d < dmax:
                    self._try_connect(u, v, d)

    # ── Frontier-biased sampling (M2) ────────────────────────────────

    def _collect_frontier_points(self, n_frontier):
        """Sample n_frontier free points biased toward tight regions."""
        nodes = list(self.G.nodes)
        if len(nodes) < 5:
            return [self.random_sample() for _ in range(n_frontier)]

        subset_size = min(len(nodes), 300) # was 100
        indices = np.random.choice(len(nodes), subset_size, replace=False)
        subset = [nodes[i] for i in indices]
        node_clr = {n: self.clearance(n) for n in subset}

        sorted_nodes = sorted(node_clr, key=lambda n: node_clr[n])
        tight_nodes = sorted_nodes[:max(1, len(sorted_nodes) // 5)]

        pts, attempts = [], 0
        while len(pts) < n_frontier and attempts < n_frontier * 10:
            attempts += 1
            anchor = tight_nodes[int(np.random.randint(len(tight_nodes)))]
            noise = np.random.normal(0, self.frontier_sigma, size=self.dim)
            candidate = tuple(
                float(np.clip(anchor[i] + noise[i],
                              self.bounds[i][0], self.bounds[i][1]))
                for i in range(self.dim)
            )
            if self.is_free(candidate):
                pts.append(candidate)
        return pts

    def random_sample(self):
        """Rejection sample with optional clearance gate."""
        while True:
            pt = tuple(float(np.random.uniform(lo, hi))
                       for lo, hi in self.bounds[:self.dim])
            if not self.is_free(pt):
                continue
            if (self.min_clearance_threshold > 0.0
                    and self.clearance(pt) < self.min_clearance_threshold):
                continue
            return pt

    # ── Graph node deduplication ────────────────────────────────────

    def _snap_to_graph(self, pos, dmax):
        """Return an existing graph node near pos, or add pos as a new node.

        Prevents node accumulation: when the robot's position shifts slightly
        each tick (due to step_size movement), this reuses the nearest existing
        node instead of creating a new one.  New nodes are only added when no
        existing node is within snap_radius.

        Returns the graph node to use as the A* source/target.
        """
        if pos in self.G:
            return pos
        snap_radius = dmax * 0.15   # same as the min_sep in sampling
        best_node = None
        best_dist = float('inf')
        for n in self.G.nodes:
            d = self.dist(n, pos)
            if d < best_dist:
                best_dist = d
                best_node = n
        if best_node is not None and best_dist < snap_radius:
            return best_node
        # No nearby node — add a new one
        self._add_vertex(pos, dmax)
        return pos

    def _record_visit(self, pos, sense_radius):
        """Record robot position for exploration scoring. Spatially downsampled."""
        pt = np.array(pos[:self.dim], dtype=float)
        if self._visited_positions is None:
            self._visited_positions = []
            self._visited_tree = None
            self._sense_radius = sense_radius
        if (len(self._visited_positions) == 0
                or np.linalg.norm(self._visited_positions[-1] - pt)
                > sense_radius * 0.4):
            self._visited_positions.append(pt)
            self._visited_tree = None  # invalidate cache

    def _get_visited_tree(self):
        """Lazily built KD-tree over visited positions."""
        if self._visited_tree is None and self._visited_positions:
            self._visited_tree = cKDTree(
                np.array(self._visited_positions, dtype=float))
        return self._visited_tree

    # ── PRMBase interface: construct / find / replan ──────────────────

    def construct(self, N, dmax, verbose=False, **kw):
        """Two-phase construction: uniform + frontier samples, KD-tree edges."""
        n_uniform = int(N * (1.0 - self.frontier_frac))
        n_frontier = N - n_uniform

        # Phase 1a: uniform samples
        uniform_samples = []
        while len(uniform_samples) < n_uniform:
            pt = self.random_sample()
            uniform_samples.append(pt)
            self.G.add_node(pt)
            if verbose and len(uniform_samples) % 100 == 0:
                print(f"  [{self.name}] uniform {len(uniform_samples)}/{n_uniform}")

        # Phase 1b: frontier-biased samples
        if verbose and n_frontier > 0:
            print(f"  [{self.name}] frontier phase ({n_frontier} samples)...")
        frontier_pts = self._collect_frontier_points(n_frontier)
        for pt in frontier_pts:
            self.G.add_node(pt)
        all_samples = uniform_samples + frontier_pts

        # Phase 2: KD-tree edge building
        if verbose:
            print(f"  [{self.name}] building edges ({len(all_samples)} nodes)...")
        for i, j in self._candidate_pairs(all_samples, dmax, self.max_neighbors):
            u, v = all_samples[i], all_samples[j]
            d = self.dist(u, v)
            if d >= self.min_edge_len:
                self._try_connect(u, v, d)

        if verbose:
            print(f"  [{self.name}] done — "
                  f"{self.G.number_of_nodes()} nodes, "
                  f"{self.G.number_of_edges()} edges")
        return self.G

    def find(self, start, goal, dmax, max_retries=5, extra=100, **kw):
        """A* on the risk-weighted graph."""
        start = self._snap_to_graph(start, dmax)

        # Online mode: don't add goal if it's in undiscovered space
        if goal not in self.G:
            if self._visited_positions is not None:
                goal_arr = np.array(goal[:self.dim], dtype=float)
                tree = self._get_visited_tree()
                if tree is not None:
                    d_nearest, _ = tree.query(goal_arr)
                    if d_nearest > self._sense_radius:
                        return None, float('inf')
            goal = self._snap_to_graph(goal, dmax)
        
        for _ in range(max_retries):
            try:
                path = nx.astar_path(self.G, start, goal,
                                     heuristic=self.dist, weight='weight')
                cost = nx.astar_path_length(self.G, start, goal,
                                            heuristic=self.dist, weight='weight')
                return path, cost
            except nx.NetworkXNoPath:
                for _ in range(extra):
                    self._add_vertex(self.random_sample(), dmax)
        return None, float('inf')

    def replan(self, current, goal, dmax,
               dyn_positions=None, affect_radius=None, **kw):
        """Local repair (M3): remove affected nodes, add targeted samples, re-query."""
        min_node_sep = dmax * 0.25

        # 1. Remove nodes inside the affected zone
        if dyn_positions and affect_radius is not None:
            nodes_to_remove = [
                n for n in list(self.G.nodes)
                if any(self.dist(n, tuple(dp[:self.dim])) < affect_radius
                       for dp in dyn_positions)
            ]
            self.G.remove_nodes_from(nodes_to_remove)

        # 2. Add repair samples (2/3 near obstacle, 1/3 near robot)
        if dyn_positions and affect_radius is not None:
            obs_anchors = [np.array(dp[:self.dim], dtype=float)
                           for dp in dyn_positions]
            inner_r = affect_radius * 1.2
            n_obs = (self.repair_samples * 2) // 3
            n_robot = self.repair_samples - n_obs
        else:
            obs_anchors, inner_r, n_obs = [], 0.0, 0
            n_robot = self.repair_samples

        robot_anchor = np.array(current[:self.dim], dtype=float)
        existing = list(self.G.nodes)

        def _ring_sample(anchor, r_inner, r_outer):
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(r_inner, r_outer)
            offset = np.zeros(self.dim)
            offset[0] = r * np.cos(angle)
            if self.dim >= 2:
                offset[1] = r * np.sin(angle)
            if self.dim == 3:
                offset[2] = np.random.normal(0, r_outer * 0.35)
            return tuple(
                float(np.clip(anchor[i] + offset[i],
                              self.bounds[i][0], self.bounds[i][1]))
                for i in range(self.dim))

        def _try_add(candidate):
            if not self.is_free(candidate):
                return False
            if any(self.dist(candidate, u) < min_node_sep for u in existing):
                return False
            self._add_vertex(candidate, dmax)
            existing.append(candidate)
            return True

        for budget, anchors, ir in [(n_obs, obs_anchors, inner_r),
                                     (n_robot, [robot_anchor], 0.0)]:
            if not anchors:
                continue
            added = attempts = 0
            outer = dmax if ir > 0 else dmax * 0.6
            while added < budget and attempts < budget * 10:
                attempts += 1
                anchor = anchors[int(np.random.randint(len(anchors)))]
                if _try_add(_ring_sample(anchor, ir, outer)):
                    added += 1

        # 3. Re-query
        return self.find(current, goal, dmax, **kw)

    # ── Online incremental exploration ───────────────────────────────

    def _pick_exploration_target(self, current_pos, goal, dmax):
        """Unified exploration target: score reachable nodes by goal proximity
        (primary) and novelty (secondary).

        The robot knows where the goal is, so exploration should be directed
        toward it. Novelty prevents the robot from stalling at dead ends —
        it breaks ties in favor of unexplored directions.

        Returns the best node, or None if no suitable target exists.
        """
        cur = tuple(current_pos[:self.dim])
        if self.G.number_of_nodes() < 3:
            return None

        tree = self._get_visited_tree()
        if tree is None:
            return None

        try:
            nearest = min(self.G.nodes, key=lambda n: self.dist(n, cur))
            reachable = list(nx.node_connected_component(self.G, nearest))
        except Exception:
            return None

        if len(reachable) < 2:
            return None

        sr = self._sense_radius
        goal_arr = np.array(goal[:self.dim], dtype=float)
        cur_arr = np.array(cur[:self.dim], dtype=float)
        max_goal_dist = max(np.linalg.norm(goal_arr - cur_arr), 1e-6)

        # Batch novelty computation
        node_coords = np.array([n[:self.dim] for n in reachable], dtype=float)
        visit_dists, _ = tree.query(node_coords, k=1)

        best_score, best_node = -1.0, None
        for idx, node in enumerate(reachable):
            if self.dist(node, cur) < dmax * 0.3:
                continue  # too close — won't make progress

            # Novelty: how far from any visited position (normalised by sense_radius)
            novelty = min(visit_dists[idx] / sr, 1.0)
            if novelty < 0.1:
                continue  # deep in explored territory

            # Goal proximity: closer to goal = higher score
            dist_to_goal = np.linalg.norm(goal_arr - node_coords[idx])
            goal_bonus = max(0.0, 1.0 - dist_to_goal / max_goal_dist)

            # Goal progress is primary; novelty breaks ties and prevents
            # stalling at dead-end walls facing the goal.
            score = 0.6 * goal_bonus + 0.4 * novelty
            if score > best_score:
                best_score, best_node = score, node

        return best_node

    def online_step(self, current_pos, goal, dmax, sense_radius,
                    n_samples=15, max_nodes=600, dyn_sense_radius=5.0):
        """One tick of online incremental exploration.

        0. Record visit.
        1. Discover obstacles within sense_radius.
        2. Prune invalidated nodes/edges.
        3. Grow roadmap near robot.
        4. Try A* to goal → return if found.
        5. Otherwise, pick a single exploration target and commit to it.
        """
        # 0. Record visit
        self._record_visit(current_pos, sense_radius)

        # 1. Discover obstacles
        newly_found = self._discover_obstacles(
            current_pos, sense_radius, dyn_sense_radius=dyn_sense_radius)

        # 2. Prune invalidated nodes and edges
        if newly_found:
            bad_nodes = [n for n in list(self.G.nodes) if not self.is_free(n)]
            self.G.remove_nodes_from(bad_nodes)
            bad_edges = [(u, v) for u, v in list(self.G.edges())
                         if not self.edge_free(u, v)]
            self.G.remove_edges_from(bad_edges)

            # Invalidate exploration target — world model changed, the
            # target might now be on the wrong side of a wall.
            if (self._exploration_target is not None
                    and self._exploration_target not in self.G):
                self._exploration_target = None

        # 3. Grow roadmap near robot — goal-directed sampling
        #
        # Three sampling strategies weighted by the situation:
        #   - Goal corridor (40%): points along the robot→goal line + noise.
        #     These build the roadmap toward the destination.
        #   - Goal-facing local (35%): disk samples biased toward the goal
        #     side of the robot (semicircle facing the goal). These give
        #     the robot local options for forward progress.
        #   - Exploration (25%): uniform disk around the robot. These fill
        #     gaps and enable backtracking when obstacles force detours.
        #
        # In 3D, z is biased toward the goal's altitude rather than spread
        # across the full workspace height. This concentrates nodes in the
        # altitude layer where they're needed for the final approach.
        anchor = np.array(current_pos[:self.dim], dtype=float)
        goal_arr = np.array(goal[:self.dim], dtype=float)
        dist_to_goal = np.linalg.norm(goal_arr - anchor)
        min_sep = dmax * 0.15
        existing = list(self.G.nodes)

        n_this_tick = n_samples if self.G.number_of_nodes() < max_nodes else 0

        sep_tree = (cKDTree(np.array([n[:self.dim] for n in existing],
                                      dtype=float))
                    if existing else None)

        # Goal direction unit vector (xy only for 3D, full for 2D)
        if dist_to_goal > 1e-6:
            goal_dir = (goal_arr - anchor) / dist_to_goal
        else:
            goal_dir = np.zeros(self.dim)

        added = attempts = 0
        while added < n_this_tick and attempts < n_this_tick * 8:
            attempts += 1
            roll = np.random.random()

            if dist_to_goal > 1e-6 and roll < 0.35:
                # ── Goal corridor: sample along robot→goal ray ──
                max_t = min(1.0, sense_radius / dist_to_goal)
                t = np.random.uniform(0.0, max_t)
                mid = anchor + t * (goal_arr - anchor)
                noise = np.random.randn(self.dim) * (dmax * 0.35)
                candidate = tuple(
                    float(np.clip(mid[i] + noise[i],
                                  self.bounds[i][0], self.bounds[i][1]))
                    for i in range(self.dim))

            elif dist_to_goal > 1e-6 and roll < 0.70:
                # ── Goal-facing local: semicircle toward goal ──
                # Sample a random direction, then flip it if it points
                # away from the goal. This creates a semicircular bias.
                angle = np.random.uniform(0, 2 * np.pi)
                r = np.random.uniform(0.0, dmax)
                offset = np.zeros(self.dim)
                offset[0] = r * np.cos(angle)
                offset[1] = r * np.sin(angle)
                # Dot product with goal direction — flip if negative
                dot = offset[0] * goal_dir[0] + offset[1] * goal_dir[1]
                if dot < 0:
                    offset[0] = -offset[0]
                    offset[1] = -offset[1]
                if self.dim >= 3:
                    # Bias z toward goal altitude with moderate spread
                    z_target = goal_arr[2]
                    offset[2] = np.random.normal(z_target - anchor[2],
                                                 dmax * 0.25)
                candidate = tuple(
                    float(np.clip(anchor[i] + offset[i],
                                  self.bounds[i][0], self.bounds[i][1]))
                    for i in range(self.dim))

            else:
                # ── Pure exploration: uniform disk around robot ──
                angle = np.random.uniform(0, 2 * np.pi)
                r = np.random.uniform(0.0, dmax)
                offset = np.zeros(self.dim)
                offset[0] = r * np.cos(angle)
                offset[1] = r * np.sin(angle)
                if self.dim >= 3:
                    offset[2] = np.random.normal(0, dmax * 0.35)
                candidate = tuple(
                    float(np.clip(anchor[i] + offset[i],
                                  self.bounds[i][0], self.bounds[i][1]))
                    for i in range(self.dim))

            if not self.is_free(candidate):
                continue
            if not self._los_free(current_pos, candidate):
                continue
            if (self.min_clearance_threshold > 0.0
                    and self.clearance(candidate) < self.min_clearance_threshold):
                continue
            if sep_tree is not None:
                d_nearest, _ = sep_tree.query(
                    np.array(candidate[:self.dim], dtype=float))
                if d_nearest < min_sep:
                    continue
            self._add_vertex(candidate, dmax)
            existing.append(candidate)
            added += 1

        # 4. Try A* to goal
        path, cost = self.find(current_pos, goal, dmax,
                               max_retries=1, extra=0)
        if path is not None:
            self._exploration_target = None  # goal is reachable, done exploring
            return path, cost

        # 5. Exploration fallback — single unified strategy with sticky commitment
        cur = self._snap_to_graph(tuple(current_pos[:self.dim]), dmax)

        # Check if we've arrived at (or lost) our committed target
        if self._exploration_target is not None:
            t = self._exploration_target
            if t not in self.G:
                self._exploration_target = None
            elif self.dist(cur, t) < dmax * 0.3:
                # Arrived — pick a new target next tick
                self._exploration_target = None
            else:
                # Still committed — route to target
                try:
                    path = nx.astar_path(self.G, cur, t,
                                         heuristic=self.dist, weight='weight')
                    cost = nx.astar_path_length(self.G, cur, t,
                                                heuristic=self.dist,
                                                weight='weight')
                    return path, cost
                except nx.NetworkXNoPath:
                    # Target became unreachable (e.g. edge pruned)
                    self._exploration_target = None

        # Pick a new exploration target
        target = self._pick_exploration_target(current_pos, goal, dmax)
        if target is not None:
            self._exploration_target = target
            try:
                path = nx.astar_path(self.G, cur, target,
                                     heuristic=self.dist, weight='weight')
                cost = nx.astar_path_length(self.G, cur, target,
                                            heuristic=self.dist, weight='weight')
                return path, cost
            except nx.NetworkXNoPath:
                self._exploration_target = None

        return None, float('inf')