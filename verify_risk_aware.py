"""
Phase 1 verification for RiskAwarePRM.

Tests all three algorithmic mechanisms described in the execution plan:
  M1. Edge cost increases near obstacles (risk_beta penalty).
  M2. Frontier-biased sampling concentrates nodes in tight regions.
  M3. Local replan removes edges near dynamic obstacles and re-queries A*.

Run with:
  python verify_risk_aware.py

Each test prints PASS or FAIL with a short diagnostic line.
"""

import numpy as np
import pybullet as p

from environments import build_simple
from prm_risk_aware import RiskAwarePRM

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def _make_planner(env, risk_beta=0.5, frontier_frac=0.3,
                  frontier_sigma=0.8, repair_samples=50):
    pl = RiskAwarePRM(
        client_id=env["cid"],
        workspace_bounds=env["bounds"],
        robot_radius=env["robot_radius"],
        dim=env["dim"],
        risk_beta=risk_beta,
        frontier_frac=frontier_frac,
        frontier_sigma=frontier_sigma,
        repair_samples=repair_samples,
    )
    all_ids = env["static_ids"] + env["dyn_manager"].body_ids
    pl.set_obstacles(all_ids)
    return pl


# ─────────────────────────────────────────────────────────────────────
# M1: Edge cost increases near obstacles
# ─────────────────────────────────────────────────────────────────────

def test_m1_edge_cost_near_obstacles():
    """
    A free-space edge far from obstacles must have lower _risk_cost than
    a same-length edge that passes close to an obstacle.

    Environment geometry (simple):
      - Horizontal wall at y=6.0, hx=3.5, hy=0.15 → occupies y=[5.85,6.15].
      - Box at (5.5, 3.0) with hx=hy=0.8.
      - Start is (1,1), open bottom-left area around (2,2) is far from all obs.

    Far edge  : (2.0,2.0)→(2.5,2.0)  — open area, clearance ~3+ m
    Near edge : (3.0,6.4)→(4.0,6.4)  — just above the horizontal wall top
                                         (wall top at y=6.15, robot r=0.2)
    Expected  : clr_near < clr_far  AND  cost_near > cost_far
    """
    env = build_simple(difficulty=0)
    pl = _make_planner(env, risk_beta=0.5)

    far_u  = (2.0, 2.0)
    far_v  = (2.5, 2.0)
    near_u = (3.0, 6.4)   # 0.25 m above wall top → clearance ~0.05 m
    near_v = (4.0, 6.4)

    # Verify test points are actually free (not inside obstacles)
    assert pl.is_free(far_u),  f"far_u {far_u} is not free — pick another point"
    assert pl.is_free(far_v),  f"far_v {far_v} is not free — pick another point"
    assert pl.is_free(near_u), f"near_u {near_u} is not free — pick another point"
    assert pl.is_free(near_v), f"near_v {near_v} is not free — pick another point"

    clr_far  = pl._clearance_along_edge(far_u, far_v)
    clr_near = pl._clearance_along_edge(near_u, near_v)
    cost_far  = pl._risk_cost(far_u, far_v)
    cost_near = pl._risk_cost(near_u, near_v)

    p.disconnect(env["cid"])

    ok = (clr_near < clr_far) and (cost_near > cost_far)
    tag = PASS if ok else FAIL
    print(f"[M1] {tag}  clr_far={clr_far:.3f}  clr_near={clr_near:.3f}  "
          f"cost_far={cost_far:.3f}  cost_near={cost_near:.3f}")
    return ok


# ─────────────────────────────────────────────────────────────────────
# M2: Frontier sampling concentrates nodes near tight regions
# ─────────────────────────────────────────────────────────────────────

def test_m2_frontier_bias():
    """
    Build two roadmaps of the same size:
      A) pure uniform (frontier_frac=0)
      B) 30 % frontier-biased (frontier_frac=0.3)
    The mean clearance of the bottom-10% nodes in B should be lower than
    in A, meaning B has MORE nodes in tight spots.
    Also the overall mean clearance of B should be lower (frontier nodes
    are anchored near low-clearance regions).
    """
    np.random.seed(0)
    N = 200

    env_a = build_simple(difficulty=0)
    pl_a = _make_planner(env_a, frontier_frac=0.0)
    pl_a.construct(N, env_a["dmax"], verbose=False)
    nodes_a = list(pl_a.G.nodes)
    clrs_a = [pl_a.clearance(n) for n in nodes_a]
    p.disconnect(env_a["cid"])

    env_b = build_simple(difficulty=0)
    pl_b = _make_planner(env_b, frontier_frac=0.3)
    pl_b.construct(N, env_b["dmax"], verbose=False)
    nodes_b = list(pl_b.G.nodes)
    clrs_b = [pl_b.clearance(n) for n in nodes_b]
    p.disconnect(env_b["cid"])

    pct = 0.10  # bottom 10 % = tightest nodes
    k_a = max(1, int(len(clrs_a) * pct))
    k_b = max(1, int(len(clrs_b) * pct))
    tight_a = float(np.mean(sorted(clrs_a)[:k_a]))
    tight_b = float(np.mean(sorted(clrs_b)[:k_b]))

    # Frontier-biased should have tighter bottom nodes (lower clearance)
    ok = tight_b <= tight_a
    tag = PASS if ok else FAIL
    print(f"[M2] {tag}  bottom-10% clearance: uniform={tight_a:.4f}  "
          f"frontier={tight_b:.4f}  "
          f"({'frontier tighter as expected' if ok else 'no bias detected'})")
    return ok


# ─────────────────────────────────────────────────────────────────────
# M3: Local replan removes edges near dynamic obstacles
# ─────────────────────────────────────────────────────────────────────

def test_m3_local_repair():
    """
    1. Build a small roadmap.
    2. Count edges with midpoints in a target region.
    3. Call replan() with a dynamic obstacle at that region's centre.
    4. Verify the edge count in that region decreased (edges were removed).
    5. Verify A* still returns a path (graph was repaired).
    """
    np.random.seed(42)
    env = build_simple(difficulty=0)
    pl = _make_planner(env, repair_samples=30)
    pl.construct(200, env["dmax"], verbose=False)

    # Place a fake dynamic obstacle in the middle of the map
    dyn_pos = (5.0, 5.0)
    affect_radius = 2.0

    def edges_near_dyn():
        count = 0
        for u, v in pl.G.edges:
            mid = tuple((np.array(u) + np.array(v)) / 2)
            if pl.dist(mid, dyn_pos) < affect_radius:
                count += 1
        return count

    edges_before = edges_near_dyn()
    edges_total_before = pl.G.number_of_edges()

    new_path, new_cost = pl.replan(
        env["start"], env["goal"], env["dmax"],
        dyn_positions=[dyn_pos],
        affect_radius=affect_radius,
    )

    edges_after  = edges_near_dyn()
    edges_total_after = pl.G.number_of_edges()

    p.disconnect(env["cid"])

    # Edges in the danger zone should drop significantly after removal.
    # Repair samples may re-add a small number of edges near the zone;
    # we accept up to 10 % of the original count remaining.
    removal_ok = edges_before > 0 and edges_after < max(1, edges_before * 0.10)
    repair_ok  = new_path is not None      # graph repaired, A* found path
    ok = removal_ok and repair_ok
    tag = PASS if ok else FAIL
    print(f"[M3] {tag}  edges near dyn: {edges_before}->{edges_after}  "
          f"total edges: {edges_total_before}->{edges_total_after}  "
          f"path={'found' if repair_ok else 'NOT FOUND'}")
    return ok


# ─────────────────────────────────────────────────────────────────────
# Bonus: End-to-end smoke check (simple/static, simple/dynamic)
# ─────────────────────────────────────────────────────────────────────

def test_e2e_simple_static():
    """RiskAware constructs and finds a path in simple/static."""
    np.random.seed(7)
    env = build_simple(difficulty=0)
    pl = _make_planner(env)
    pl.construct(env["N"], env["dmax"], verbose=False)
    path, cost = pl.find(env["start"], env["goal"], env["dmax"])
    p.disconnect(env["cid"])
    ok = path is not None and len(path) >= 2
    tag = PASS if ok else FAIL
    print(f"[E2E-static]  {tag}  path_len={len(path) if path else 0}  "
          f"cost={cost:.2f}")
    return ok


def test_e2e_simple_dynamic():
    """RiskAware constructs and finds a path in simple/noise+dynamic."""
    np.random.seed(7)
    env = build_simple(difficulty=3)
    pl = _make_planner(env)
    pl.construct(env["N"], env["dmax"], verbose=False)
    path, cost = pl.find(env["start"], env["goal"], env["dmax"])
    nodes = pl.G.number_of_nodes()
    edges = pl.G.number_of_edges()
    p.disconnect(env["cid"])
    ok = path is not None and len(path) >= 2
    tag = PASS if ok else FAIL
    print(f"[E2E-dynamic] {tag}  nodes={nodes}  edges={edges}  "
          f"path_len={len(path) if path else 0}  cost={cost:.2f}")
    return ok


# ─────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("RiskAwarePRM — Phase 1 Verification")
    print("=" * 60)

    results = {}
    results["M1 edge cost near obstacles"] = test_m1_edge_cost_near_obstacles()
    results["M2 frontier bias"]            = test_m2_frontier_bias()
    results["M3 local repair"]             = test_m3_local_repair()
    results["E2E simple/static"]           = test_e2e_simple_static()
    results["E2E simple/noise+dynamic"]    = test_e2e_simple_dynamic()

    print("=" * 60)
    passed = sum(results.values())
    total  = len(results)
    print(f"Result: {passed}/{total} passed")
    if passed == total:
        print("All mechanisms verified. Phase 1 exit criteria met.")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"Failed: {failed}")
    print("=" * 60)
