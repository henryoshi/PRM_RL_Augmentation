"""
Standalone visualization of frontier-biased sampling.

Produces a two-panel figure:
  Left  — uniform samples only (phase 1, 70%)
  Right — frontier-biased samples added (phase 2, 30%), coloured by
          the clearance of the seed node that attracted them.

No PyBullet required.  Obstacles are axis-aligned rectangles.
Run:
  python visualize_frontier_sampling.py
  python visualize_frontier_sampling.py --out frontier_sampling.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from scipy.spatial import cKDTree

# ── Environment definition (mirrors office layout, simplified) ────────
BOUNDS = [(0, 20), (0, 15)]

# (cx, cy, half-width, half-height)  — axis-aligned boxes
OBSTACLES = [
    # outer walls (thin)
    (10, 0.12, 10, 0.12),
    (10, 14.88, 10, 0.12),
    (0.12, 7.5, 0.12, 7.5),
    (19.88, 7.5, 0.12, 7.5),
    # interior partitions
    (7, 3.25, 0.12, 3.25),
    (7, 11.2, 0.12, 3.8),
    (14, 1.6, 0.12, 1.6),
    (14, 7.1, 0.12, 3.0),
    (14, 13.45, 0.12, 1.55),
    (1.6, 10, 1.6, 0.12),
    (5.65, 10, 1.35, 0.12),
    (2.4, 4.5, 2.4, 0.12),
    (6.35, 4.5, 0.65, 0.12),
    (15.4, 7.5, 1.4, 0.12),
    (18.85, 7.5, 1.15, 0.12),
    # furniture
    (2.5, 7.5, 0.6, 0.6),
    (10.5, 12.0, 1.0, 0.4),
    (10.5, 3.0, 0.5, 0.8),
    (17.0, 12.5, 0.7, 0.5),
]

ROBOT_RADIUS = 0.2
MIN_CLR      = ROBOT_RADIUS * 1.5   # hard gate threshold

# ── Geometry helpers ─────────────────────────────────────────────────

def _clearance_point(pt, obstacles, robot_radius):
    """Approximate clearance as min distance from pt to any obstacle surface."""
    px, py = pt
    min_d = 99.0
    for cx, cy, hw, hh in obstacles:
        # closest point on box to pt
        bx = np.clip(px, cx - hw, cx + hw)
        by = np.clip(py, cy - hh, cy + hh)
        d = np.hypot(px - bx, py - by) - robot_radius
        if d < min_d:
            min_d = d
    return max(min_d, 0.0)


def _is_free(pt, obstacles, robot_radius):
    return _clearance_point(pt, obstacles, robot_radius) > 0.0


def _random_free(n, rng, obstacles, robot_radius, bounds):
    pts = []
    while len(pts) < n:
        x = rng.uniform(bounds[0][0], bounds[0][1])
        y = rng.uniform(bounds[1][0], bounds[1][1])
        pt = (x, y)
        if _is_free(pt, obstacles, robot_radius):
            pts.append(pt)
    return np.array(pts)


# ── Sampling logic ───────────────────────────────────────────────────

def generate_samples(n_total=300, frontier_frac=0.30, frontier_sigma=0.8,
                     seed=42):
    rng = np.random.default_rng(seed)
    n_uniform  = int(n_total * (1 - frontier_frac))
    n_frontier = n_total - n_uniform

    # Phase 1 — uniform
    uniform_pts = _random_free(n_uniform, rng, OBSTACLES, ROBOT_RADIUS, BOUNDS)

    # Compute clearance of uniform samples → identify low-clearance seeds
    clr = np.array([_clearance_point(p, OBSTACLES, ROBOT_RADIUS)
                    for p in uniform_pts])

    # Bottom 20% by clearance → seed positions for phase 2
    threshold = np.percentile(clr, 20)
    seed_mask  = clr <= threshold
    seeds      = uniform_pts[seed_mask]

    # Phase 2 — Gaussian perturbations around seeds, hard-gated
    frontier_pts = []
    seed_labels  = []   # which seed attracted each frontier point
    attempts = 0
    while len(frontier_pts) < n_frontier and attempts < n_frontier * 30:
        attempts += 1
        idx = rng.integers(len(seeds))
        sx, sy = seeds[idx]
        x = rng.normal(sx, frontier_sigma)
        y = rng.normal(sy, frontier_sigma)
        pt = (
            np.clip(x, BOUNDS[0][0] + 0.05, BOUNDS[0][1] - 0.05),
            np.clip(y, BOUNDS[1][0] + 0.05, BOUNDS[1][1] - 0.05),
        )
        c = _clearance_point(pt, OBSTACLES, ROBOT_RADIUS)
        if c > MIN_CLR:          # hard gate
            frontier_pts.append(pt)
            seed_labels.append(idx)

    frontier_pts = np.array(frontier_pts) if frontier_pts else np.empty((0, 2))
    return uniform_pts, clr, seeds, seed_mask, frontier_pts


# ── Drawing helpers ──────────────────────────────────────────────────

def _draw_walls(ax, alpha=0.85):
    for cx, cy, hw, hh in OBSTACLES:
        rect = mpatches.Rectangle(
            (cx - hw, cy - hh), 2 * hw, 2 * hh,
            linewidth=0, facecolor="#7a7a82", alpha=alpha, zorder=2)
        ax.add_patch(rect)


def _axis_setup(ax, title):
    ax.set_xlim(BOUNDS[0])
    ax.set_ylim(BOUNDS[1])
    ax.set_aspect("equal")
    ax.set_facecolor("#f5f5f0")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#cccccc")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None,
                        help="Save figure to this path instead of showing")
    parser.add_argument("--n", type=int, default=300,
                        help="Total sample count (default: 300)")
    args = parser.parse_args()

    uniform_pts, clr, seeds, seed_mask, frontier_pts = generate_samples(
        n_total=args.n)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(wspace=0.18, left=0.03, right=0.97,
                        top=0.88, bottom=0.08)

    # ── Panel A: uniform phase ──────────────────────────────────────
    ax = axes[0]
    _axis_setup(ax, "Phase 1 — Uniform Samples (70%)")
    _draw_walls(ax)

    # Colour uniform nodes by clearance (low = warm, high = cool)
    sc = ax.scatter(uniform_pts[:, 0], uniform_pts[:, 1],
                    c=clr, cmap="RdYlGn_r", vmin=0, vmax=1.5,
                    s=18, zorder=3, linewidths=0, alpha=0.85)

    # Highlight low-clearance seeds
    ax.scatter(seeds[:, 0], seeds[:, 1],
               s=55, facecolors="none", edgecolors="#c0392b",
               linewidths=1.2, zorder=4)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Clearance (m)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.scatter(*[1.5], *[1.5], marker="*", s=180, color="#27ae60", zorder=5)
    ax.scatter(*[18.5], *[13.5], marker="*", s=180, color="#2980b9", zorder=5)

    # ── Panel B: frontier phase ─────────────────────────────────────
    ax = axes[1]
    _axis_setup(ax, "Phase 2 — Frontier-Biased Samples Added (30%)")
    _draw_walls(ax)

    # Uniform nodes (greyed out to let frontier nodes stand out)
    ax.scatter(uniform_pts[:, 0], uniform_pts[:, 1],
               c="#aaaaaa", s=12, zorder=3, linewidths=0, alpha=0.45)

    # Seed nodes
    ax.scatter(seeds[:, 0], seeds[:, 1],
               s=55, facecolors="none", edgecolors="#c0392b",
               linewidths=1.2, zorder=4)

    # Frontier nodes coloured by clearance
    if len(frontier_pts):
        f_clr = np.array([_clearance_point(p, OBSTACLES, ROBOT_RADIUS)
                          for p in frontier_pts])
        sc2 = ax.scatter(frontier_pts[:, 0], frontier_pts[:, 1],
                         c=f_clr, cmap="plasma_r", vmin=0, vmax=1.0,
                         s=28, zorder=5, linewidths=0, alpha=0.9)
        cbar2 = fig.colorbar(sc2, ax=ax, fraction=0.035, pad=0.04)
        cbar2.set_label("Clearance (m)", fontsize=8)
        cbar2.ax.tick_params(labelsize=7)

    ax.scatter(*[1.5], *[1.5], marker="*", s=180, color="#27ae60", zorder=6)
    ax.scatter(*[18.5], *[13.5], marker="*", s=180, color="#2980b9", zorder=6)

    fig.suptitle(
        "Frontier-Biased Sampling  -  Risk-Aware PRM",
        fontsize=13, fontweight="bold", y=0.97)

    if args.out:
        fig.savefig(args.out, dpi=180, bbox_inches="tight")
        print(f"Saved to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
