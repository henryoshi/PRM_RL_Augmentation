"""
Risk-Aware PRM — novel planner for cold-start navigation.

Three improvements over BasicPRM (see revised_project_plan.txt):

  1. Risk-weighted edges:   cost = dist(u,v) + beta / min_clearance(u_v)
     Paths that run close to obstacles are penalised; A* naturally finds
     routes through the middle of open corridors.

  2. Frontier-biased sampling:  70 % uniform base + 30 % Gaussian
     anchored near the lowest-clearance nodes found.
     Adds roadmap density where the baseline tends to fail.

  3. Local replan repair:  rather than rebuilding the whole graph, remove
     only edges whose midpoint is within affect_radius of a dynamic obstacle,
     add a small number of targeted samples near the robot's current
     position, then re-query A*.
     NOTE: Potentially could look at local A* rather than NetworkXs built in
     solution to avoid entire re-comp.
All PyBullet interaction is delegated to PRMBase, no direct pybullet calls
here. This planner works with all three existing environments and all four
difficulty levels.
"""

import numpy as np
import networkx as nx
from prm_base import PRMBase


class RiskAwarePRM(PRMBase):
    name = "RiskAware"

    def __init__(self, client_id, workspace_bounds, robot_radius=0.15,
                 dim=2, fixed_z=None,
                 risk_beta=0.5,
                 frontier_frac=0.3,
                 frontier_sigma=0.8,
                 repair_samples=20,
                 min_edge_len=0.0,
                 max_neighbors=None):
        """
        Parameters (beyond PRMBase)
        ---------------------------
        risk_beta      : clearance penalty weight in edge cost.
        frontier_frac  : fraction of construct() samples that are
                         frontier-biased (rest are uniform).
        frontier_sigma : Gaussian spread (m) around anchor nodes when
                         generating frontier samples.
        repair_samples : samples added near the robot during local replan.
        min_edge_len   : edges shorter than this (m) are skipped; set to
                         ~robot_radius to prune redundant near-duplicate edges.
        max_neighbors  : if set, each node connects to at most this many
                         nearest neighbours during construction (caps edge
                         count for dense environments like office).
        """
        super().__init__(client_id, workspace_bounds, robot_radius, dim,
                         fixed_z, min_edge_len=min_edge_len,
                         max_neighbors=max_neighbors)
        self.risk_beta = risk_beta
        self.frontier_frac = frontier_frac
        self.frontier_sigma = frontier_sigma
        self.repair_samples = repair_samples

    # ── Internal helpers ─────────────────────────────────────────────

    def _clearance_along_edge(self, u, v):
        """Minimum clearance sampled at K points along segment u→v."""
        a = np.asarray(u, dtype=float)
        b = np.asarray(v, dtype=float)
        length = np.linalg.norm(b - a)
        K = max(3, int(length / 1.0))   # 1.0 m step — was 0.5 (2× faster)
        min_clr = float('inf')
        for i in range(K + 1):
            pt = tuple(a + (i / K) * (b - a))
            min_clr = min(min_clr, self.clearance(pt))
        return min_clr

    def _risk_cost(self, u, v):
        """Edge weight: Euclidean length + clearance penalty."""
        d = self.dist(u, v)
        clr = max(self._clearance_along_edge(u, v), 1e-6)
        return d + self.risk_beta / clr

    def _add_vertex(self, v, dmax):
        """Incremental insert — used only by find() and replan()."""
        self.G.add_node(v)
        for u in list(self.G.nodes):
            d = self.dist(u, v)
            if u != v and self.min_edge_len <= d < dmax:
                if self.edge_free(u, v):
                    self.G.add_edge(u, v, weight=self._risk_cost(u, v))

    def _try_connect(self, u, v, d):
        """Connect u↔v with risk-weighted cost. Override in subclasses."""
        if self.edge_free(u, v):
            self.G.add_edge(u, v, weight=self._risk_cost(u, v))

    def _collect_frontier_points(self, n_frontier):
        """
        Return n_frontier free sample POINTS biased toward tight regions.
        Does NOT modify the graph — pure sampling.
        """
        nodes = list(self.G.nodes)
        if len(nodes) < 5:
            pts = []
            while len(pts) < n_frontier:
                pts.append(self.random_sample())
            return pts

        subset_size = min(len(nodes), 100)
        indices = np.random.choice(len(nodes), subset_size, replace=False)
        subset = [nodes[i] for i in indices]
        node_clr = {n: self.clearance(n) for n in subset}

        sorted_nodes = sorted(node_clr, key=lambda n: node_clr[n])
        tight_nodes = sorted_nodes[:max(1, len(sorted_nodes) // 5)]

        pts, attempts = [], 0
        max_attempts = n_frontier * 10
        while len(pts) < n_frontier and attempts < max_attempts:
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

    def _frontier_samples(self, n_frontier, dmax):
        """Incremental frontier insert — used only during replan()."""
        for pt in self._collect_frontier_points(n_frontier):
            self._add_vertex(pt, dmax)

    # ── PRMBase interface ────────────────────────────────────────────

    def construct(self, N, dmax, verbose=False, **kw):
        """
        Two-phase roadmap construction with KD-tree edge building.

        Sampling  (70 % uniform + 30 % frontier-biased) is done first,
        then ALL edges are built in one batch via KD-tree query_pairs —
        replacing the O(N²) incremental loop with O(N log N + k).
        """
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

        # Phase 1b: frontier samples (needs uniform nodes already in graph
        #           to identify tight regions — but no edges yet)
        if verbose:
            print(f"  [{self.name}] frontier phase ({n_frontier} samples)...")
        frontier_pts = self._collect_frontier_points(n_frontier)
        for pt in frontier_pts:
            self.G.add_node(pt)
        all_samples = uniform_samples + frontier_pts

        # Phase 2: build all edges via KD-tree — single C-level pass
        if verbose:
            print(f"  [{self.name}] building edges "
                  f"({len(all_samples)} nodes)...")
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
        """A* on the risk-weighted graph, with up to max_retries expansions."""
        if start not in self.G:
            self._add_vertex(start, dmax)
        if goal not in self.G:
            self._add_vertex(goal, dmax)

        for _ in range(max_retries):
            try:
                path = nx.astar_path(self.G, start, goal,
                                     heuristic=self.dist, weight='weight')
                cost = nx.astar_path_length(self.G, start, goal,
                                            heuristic=self.dist,
                                            weight='weight')
                return path, cost
            except nx.NetworkXNoPath:
                for _ in range(extra):
                    self._add_vertex(self.random_sample(), dmax)

        return None, float('inf')

    def replan(self, current, goal, dmax,
               dyn_positions=None, affect_radius=None, **kw):
        """
        Local repair strategy — density-respecting:

          1. Remove nodes whose position falls inside the affected zone.
             Removing a node automatically removes all its incident edges, so
             no separate edge scan is needed.  This prevents ghost nodes from
             accumulating inside obstacles over successive replans.

          2. Add up to repair_samples new nodes near *current*, subject to a
             minimum separation guard: a candidate is skipped if any existing
             node is already within min_node_sep metres.  This keeps the graph
             uniformly sparse — repeated replans in the same spot add zero extra
             nodes once the local density is already sufficient.

          3. Re-query A* on the repaired graph.
        """
        min_node_sep = dmax * 0.25   # minimum distance between any two nodes

        if dyn_positions and affect_radius is not None:
            nodes_to_remove = [
                n for n in list(self.G.nodes)
                if any(
                    self.dist(n, tuple(dp[:self.dim])) < affect_radius
                    for dp in dyn_positions
                )
            ]
            # remove_nodes_from removes the node AND all its incident edges
            self.G.remove_nodes_from(nodes_to_remove)

        # Split repair budget between two concerns:
        #
        #   2/3 around the obstacle(s) — fix the topological gap by placing
        #       bypass candidates in a ring at inner_r..dmax around each
        #       obstacle.  Inner radius slightly outside affect_radius so
        #       nodes aren't placed right next to a still-moving obstacle.
        #
        #   1/3 near the robot — motion noise can push the robot into a
        #       sparsely sampled region where it can't reconnect to the graph
        #       at all.  A small local cloud ensures the robot always has
        #       nearby nodes to attach to regardless of drift.
        #
        if dyn_positions and affect_radius is not None:
            obs_anchors = [np.array(dp[:self.dim], dtype=float)
                           for dp in dyn_positions]
            inner_r = affect_radius * 1.2
            n_obs_samples   = (self.repair_samples * 2) // 3
            n_robot_samples = self.repair_samples - n_obs_samples
        else:
            obs_anchors     = []
            inner_r         = 0.0
            n_obs_samples   = 0
            n_robot_samples = self.repair_samples

        robot_anchor = np.array(current[:self.dim], dtype=float)

        def _ring_sample(anchor, inner_r, outer_r):
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(inner_r, outer_r)
            offset = np.zeros(self.dim)
            offset[0] = r * np.cos(angle)
            if self.dim >= 2:
                offset[1] = r * np.sin(angle)
            return tuple(
                float(np.clip(anchor[i] + offset[i],
                              self.bounds[i][0], self.bounds[i][1]))
                for i in range(self.dim)
            )

        existing_nodes = list(self.G.nodes)

        def _try_add(candidate):
            if not self.is_free(candidate):
                return False
            if any(self.dist(candidate, u) < min_node_sep
                   for u in existing_nodes):
                return False
            self._add_vertex(candidate, dmax)
            existing_nodes.append(candidate)
            return True

        # --- obstacle-vicinity samples ---
        added, attempts = 0, 0
        while added < n_obs_samples and attempts < n_obs_samples * 10:
            attempts += 1
            anchor = obs_anchors[int(np.random.randint(len(obs_anchors)))]
            if _try_add(_ring_sample(anchor, inner_r, dmax)):
                added += 1

        # --- robot-vicinity samples ---
        added, attempts = 0, 0
        while added < n_robot_samples and attempts < n_robot_samples * 10:
            attempts += 1
            if _try_add(_ring_sample(robot_anchor, 0.0, dmax * 0.6)):
                added += 1

        return self.find(current, goal, dmax, **kw)

    def online_step(self, current_pos, goal, dmax, sense_radius, n_samples=15):
        """
        One tick of online incremental exploration.

        Called each execution tick when running in --mode online instead of
        a pre-built roadmap.  Grows the roadmap one step at a time as the
        robot moves through the environment.

          1. Reveal obstacles newly within sense_radius (updates is_free).
          2. Prune roadmap nodes that are now in collision.
          3. Add up to n_samples free nodes near current_pos (density-guarded).
          4. Return (path, cost) via find() — (None, inf) if not connected yet.
        """
        # 1. Discover
        newly_found = self._discover_obstacles(current_pos, sense_radius)

        # 2. Prune nodes now in collision with the newly-revealed obstacles
        if newly_found:
            bad = [n for n in list(self.G.nodes) if not self.is_free(n)]
            self.G.remove_nodes_from(bad)

        # 3. Grow roadmap near robot (density-guarded: min_sep = dmax * 0.25)
        anchor = np.array(current_pos[:self.dim], dtype=float)
        min_sep = dmax * 0.25
        existing = list(self.G.nodes)
        added = attempts = 0
        while added < n_samples and attempts < n_samples * 8:
            attempts += 1
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0.0, dmax)
            offset = np.zeros(self.dim)
            offset[0] = r * np.cos(angle)
            if self.dim >= 2:
                offset[1] = r * np.sin(angle)
            candidate = tuple(
                float(np.clip(anchor[i] + offset[i],
                              self.bounds[i][0], self.bounds[i][1]))
                for i in range(self.dim)
            )
            if not self.is_free(candidate):
                continue
            if any(self.dist(candidate, u) < min_sep for u in existing):
                continue
            self._add_vertex(candidate, dmax)
            existing.append(candidate)
            added += 1

        # 4. Try to find path from current position to goal
        return self.find(current_pos, goal, dmax)
