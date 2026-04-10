"""
TEMPLATE: How to add your own PRM planner.

Copy this file, rename it (e.g. prm_myalgo.py), and fill in the three
methods below.  Then register it in run_single.py and benchmark.py:

    # --- run_single.py (and benchmark.py) ---
    from prm_myalgo import MyAlgoPRM          # 1. import

    PLANNER_CLASSES = {
        "Basic":  BasicPRM,
        "MyAlgo": MyAlgoPRM,                  # 2. register
    }

That's it.  Now you can run:

    python run_single.py --env office --diff 3 --planner MyAlgo --gui
    python benchmark.py  --planners Basic MyAlgo --trials 50

────────────────────────────────────────────────────────────────────────
What you inherit for free from PRMBase (see prm_base.py):
────────────────────────────────────────────────────────────────────────

  self.G              NetworkX graph — add/remove nodes and edges freely.
  self.bounds         Workspace limits, e.g. [(0,12),(0,12)] for 2D.
  self.dim            2 or 3.
  self.radius         Robot collision-sphere radius.
  self.obstacle_ids   PyBullet body IDs of all current obstacles.

  self.is_free(pt)        -> bool   Is this configuration collision-free?
  self.edge_free(v1, v2)  -> bool   Is the straight line v1→v2 clear?
  self.clearance(pt)      -> float  Min distance to nearest obstacle.
  self.random_sample()    -> tuple  Random collision-free configuration.
  self.dist(a, b)         -> float  Euclidean distance between configs.
  self.reset_graph()                Wipe self.G to start fresh.

These all use PyBullet under the hood — you never need to call PyBullet
directly in your planner unless you want to.

────────────────────────────────────────────────────────────────────────
What you must implement:
────────────────────────────────────────────────────────────────────────

  construct(N, dmax)           Build the roadmap (N samples, connect
                               neighbours within dmax).

  find(start, goal, dmax)      Query the roadmap for a path.
                               Return (path, cost) or (None, inf).

  replan(current, goal, dmax)  [optional override] Called by the executor
                               when dynamic obstacles block the current
                               path.  Default just calls find() again.
                               Override this if your algorithm can do
                               something smarter (local repair, etc).
"""

import networkx as nx
from prm_base import PRMBase


class MyAlgoPRM(PRMBase):
    name = "MyAlgo"       # shows up in benchmark tables and plot titles

    def construct(self, N, dmax, verbose=False, **kw):
        """
        Build the roadmap.

        Parameters
        ----------
        N    : int   — number of samples to draw.
        dmax : float — max connection radius.

        Use any strategy you want:
          - Uniform sampling    → self.random_sample()
          - Biased sampling     → write your own sampler
          - Halton / lattice    → generate deterministic sequences
          - Obstacle-aware      → use self.clearance(pt) to bias

        Connect nodes however you like — just add edges to self.G:
          self.G.add_node(v)
          self.G.add_edge(u, v, weight=self.dist(u, v))

        Always check self.edge_free(u, v) before adding an edge
        (unless you're doing lazy evaluation on purpose).
        """
        for k in range(N):
            vnew = self.random_sample()
            self.G.add_node(vnew)

            for v in list(self.G.nodes):
                if v != vnew and self.dist(v, vnew) < dmax:
                    if self.edge_free(v, vnew):
                        self.G.add_edge(v, vnew, weight=self.dist(v, vnew))

            if verbose and (k + 1) % 100 == 0:
                print(f"  [{self.name}] {k+1}/{N}")

        return self.G

    def find(self, start, goal, dmax, max_retries=5, extra=100, **kw):
        """
        Find a path from start to goal on the existing roadmap.

        1. Connect start and goal into the graph.
        2. Search (A*, Dijkstra, your own algorithm).
        3. Return (path_list, total_cost)  or  (None, float('inf')).

        The executor calls this once after construct(), and again via
        replan() whenever the path gets blocked by dynamic obstacles.
        """
        # Connect start/goal
        if start not in self.G:
            self.G.add_node(start)
            for v in list(self.G.nodes):
                if v != start and self.dist(v, start) < dmax:
                    if self.edge_free(v, start):
                        self.G.add_edge(v, start, weight=self.dist(v, start))

        if goal not in self.G:
            self.G.add_node(goal)
            for v in list(self.G.nodes):
                if v != goal and self.dist(v, goal) < dmax:
                    if self.edge_free(v, goal):
                        self.G.add_edge(v, goal, weight=self.dist(v, goal))

        # Search with retries
        for attempt in range(max_retries):
            try:
                path = nx.astar_path(self.G, start, goal,
                                     heuristic=self.dist, weight='weight')
                cost = nx.astar_path_length(self.G, start, goal,
                                            heuristic=self.dist, weight='weight')
                return path, cost
            except nx.NetworkXNoPath:
                # Add more samples and try again
                for _ in range(extra):
                    vnew = self.random_sample()
                    self.G.add_node(vnew)
                    for v in list(self.G.nodes):
                        if v != vnew and self.dist(v, vnew) < dmax:
                            if self.edge_free(v, vnew):
                                self.G.add_edge(v, vnew,
                                                weight=self.dist(v, vnew))

        return None, float('inf')

    def replan(self, current, goal, dmax, **kw):
        """
        [Optional] Called when dynamic obstacles block the active path.

        Default behaviour: just call find() from the new position.
        Override if your algorithm can do targeted repair, e.g.:
          - Remove edges near moving obstacles
          - Add samples only in the affected region
          - Re-weight edges based on obstacle proximity
        """
        return self.find(current, goal, dmax, **kw)
