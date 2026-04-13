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
                 repair_samples=50):
        """
        Parameters (beyond PRMBase)
        ---------------------------
        risk_beta      : clearance penalty weight in edge cost.
        frontier_frac  : fraction of construct() samples that are
                         frontier-biased (rest are uniform).
        frontier_sigma : Gaussian spread (m) around anchor nodes when
                         generating frontier samples.
        repair_samples : samples added near the robot during local replan.
        """
        super().__init__(client_id, workspace_bounds, robot_radius, dim,
                         fixed_z)
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
        K = max(3, int(length / 0.5))
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
        """Add v to the graph; connect to all neighbours within dmax."""
        self.G.add_node(v)
        for u in list(self.G.nodes):
            if u != v and self.dist(u, v) < dmax:
                if self.edge_free(u, v):
                    self.G.add_edge(u, v, weight=self._risk_cost(u, v))

    def _frontier_samples(self, n_frontier, dmax):
        """
        Add n_frontier samples biased toward tight regions.

        Strategy: compute clearance for up to 100 randomly chosen existing
        nodes; pick the bottom 20 % (lowest clearance = nearest to obstacles);
        draw Gaussian samples around those anchors.  Falls back to uniform
        sampling when the graph is too sparse to identify tight nodes.
        """
        nodes = list(self.G.nodes)
        if len(nodes) < 5:
            for _ in range(n_frontier):
                self._add_vertex(self.random_sample(), dmax)
            return

        # Compute clearance for a representative subset
        subset_size = min(len(nodes), 100)
        indices = np.random.choice(len(nodes), subset_size, replace=False)
        subset = [nodes[i] for i in indices]
        node_clr = {n: self.clearance(n) for n in subset}

        # Bottom 20 % by clearance are the anchor nodes
        sorted_nodes = sorted(node_clr, key=lambda n: node_clr[n])
        tight_nodes = sorted_nodes[:max(1, len(sorted_nodes) // 5)]

        added, attempts = 0, 0
        max_attempts = n_frontier * 10
        while added < n_frontier and attempts < max_attempts:
            attempts += 1
            anchor = tight_nodes[int(np.random.randint(len(tight_nodes)))]
            noise = np.random.normal(0, self.frontier_sigma, size=self.dim)
            candidate = tuple(
                float(np.clip(anchor[i] + noise[i],
                              self.bounds[i][0], self.bounds[i][1]))
                for i in range(self.dim)
            )
            if self.is_free(candidate):
                self._add_vertex(candidate, dmax)
                added += 1

    # ── PRMBase interface ────────────────────────────────────────────

    def construct(self, N, dmax, verbose=False, **kw):
        """
        Two-phase roadmap construction.

        Phase 1 (70 % of N): uniform random samples — same as BasicPRM.
        Phase 2 (30 % of N): frontier-biased samples near tight nodes.
        """
        n_uniform = int(N * (1.0 - self.frontier_frac))
        n_frontier = N - n_uniform

        for k in range(n_uniform):
            self._add_vertex(self.random_sample(), dmax)
            if verbose and (k + 1) % 100 == 0:
                print(f"  [{self.name}] uniform {k + 1}/{n_uniform}")

        if verbose:
            print(f"  [{self.name}] frontier phase ({n_frontier} samples)...")
        self._frontier_samples(n_frontier, dmax)

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
        Local repair strategy:
          1. Remove edges whose midpoint is within affect_radius of any
             dynamic obstacle (fast — no new collision queries).
          2. Add repair_samples new nodes concentrated near *current*.
          3. Re-query A* on the repaired graph.
        """
        if dyn_positions and affect_radius is not None:
            edges_to_remove = [
                (u, v)
                for u, v in list(self.G.edges)
                if any(
                    self.dist(
                        tuple((np.array(u) + np.array(v)) / 2),
                        tuple(dp[:self.dim])
                    ) < affect_radius
                    for dp in dyn_positions
                )
            ]
            self.G.remove_edges_from(edges_to_remove)

        added, attempts = 0, 0
        while added < self.repair_samples and attempts < self.repair_samples * 5:
            attempts += 1
            noise = np.random.normal(0, dmax * 0.4, size=self.dim)
            candidate = tuple(
                float(np.clip(current[i] + noise[i],
                              self.bounds[i][0], self.bounds[i][1]))
                for i in range(self.dim)
            )
            if self.is_free(candidate):
                self._add_vertex(candidate, dmax)
                added += 1

        return self.find(current, goal, dmax, **kw)
