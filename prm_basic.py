"""Standard PRM — baseline planner (mirrors the original algorithm)."""

import networkx as nx
from prm_base import PRMBase


class BasicPRM(PRMBase):
    name = "BasicPRM"

    def _add_vertex(self, v, dmax):
        self.G.add_node(v)
        for u in list(self.G.nodes):
            if u != v and self.dist(u, v) < dmax:
                if self.edge_free(u, v):
                    self.G.add_edge(u, v, weight=self.dist(u, v))

    def construct(self, N, dmax, verbose=False):
        for k in range(N):
            self._add_vertex(self.random_sample(), dmax)
            if verbose and (k + 1) % 100 == 0:
                print(f"  [{self.name}] {k+1}/{N}")
        return self.G

    def find(self, start, goal, dmax, max_retries=5, extra=100, **kw):
        if start not in self.G:
            self._add_vertex(start, dmax)
        if goal not in self.G:
            self._add_vertex(goal, dmax)

        for attempt in range(max_retries):
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
