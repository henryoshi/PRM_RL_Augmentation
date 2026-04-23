"""
Path executor — walks the planned path step-by-step with:
  • motion noise applied at each step
  • dynamic obstacle updates every tick
  • collision detection along the way
  • replanning when the current path is blocked

Returns a TrajectoryRecord used by the metrics module.
"""

import time
import numpy as np
from dataclasses import dataclass, field


@dataclass
class TrajectoryRecord:
    success: bool = False
    executed_positions: list = field(default_factory=list)
    clearances: list = field(default_factory=list)
    replans: int = 0
    wait_ticks: int = 0             # ticks spent holding position for obstacles to clear
    planning_time: float = 0.0      # seconds spent in construct + find
    execution_time: float = 0.0     # seconds of simulated execution
    total_time: float = 0.0         # planning + execution wall-clock
    dyn_obstacle_positions: list = field(default_factory=list)
    # list of snapshots: each entry is a list of (x,y[,z]) — one per obstacle per tick


class PathExecutor:
    """
    Executes a PRM path in a dynamic, noisy PyBullet environment.

    Parameters
    ----------
    planner : PRMBase subclass instance (already constructed).
    env     : dict returned by an environment builder.
    step_size : float
        How far the robot moves per tick (metres).
    replan_horizon : int
        Check this many upcoming waypoints for validity.
    max_execution_steps : int
        Safety cap to avoid infinite loops.
    """

    def __init__(self, planner, env, step_size=0.3,
                 replan_horizon=3, max_execution_steps=3000,
                 max_wait_ticks=80, replan_dmax_factor=1.5):
        self.planner = planner
        self.env = env
        self.step_size = step_size
        self.replan_horizon = replan_horizon
        self.max_steps = max_execution_steps
        self.max_wait_ticks = max_wait_ticks
        self.replan_dmax_factor = replan_dmax_factor
        self.noise = env["noise"]
        self.dyn = env["dyn_manager"]
        self.dim = env["dim"]
        self.goal_tol = env["robot_radius"] * 3

    def _reached_goal(self, pos, goal):
        return np.linalg.norm(np.array(pos) - np.array(goal)) < self.goal_tol

    def _path_blocked_ahead(self, path, idx):
        """Check if the next few edges on the path are still collision-free."""
        end = min(idx + self.replan_horizon, len(path) - 1)
        for i in range(idx, end):
            if not self.planner.edge_free(path[i], path[i + 1]):
                return True
        return False

    def execute(self, path, cost):
        """
        Walk *path*, returning a TrajectoryRecord.
        If path is None (planning failed), returns a failed record.
        """
        rec = TrajectoryRecord()
        goal = self.env["goal"]
        dmax = self.env["dmax"]

        if path is None:
            return rec

        current = np.array(path[0], dtype=float)
        rec.executed_positions.append(tuple(current))
        wp_idx = 1                       # next waypoint index
        sim_t = 0.0

        for step in range(self.max_steps):
            # 1. Tick dynamic obstacles
            # dyn_manager.step() calls resetBasePositionAndOrientation for
            # each body — PyBullet automatically uses the updated positions
            # in subsequent getClosestPoints calls.  No need to rebuild the
            # obstacle/AABB cache every tick.
            dyn_positions = self.dyn.step(dt=0.05)
            if dyn_positions:
                rec.dyn_obstacle_positions.append(
                    [tuple(pos[:self.dim]) for pos in dyn_positions])

            # 2. Check if path ahead is blocked → replan
            if self._path_blocked_ahead(path, wp_idx - 1):
                rec.replans += 1
                cur_tuple = tuple(current)
                # Pass dynamic obstacle info to any planner whose replan()
                # accepts it (e.g. RiskAwarePRM local repair, future planners).
                # PRMBase.replan(**kw) absorbs unknown kwargs safely.#
                kwargs = {}
                # Change here to abstract for Aware PRM.
                if dyn_positions:
                    kwargs["dyn_positions"] = [
                        pos[:self.dim] for pos in dyn_positions]
                    kwargs["affect_radius"] = dmax * 0.6
                replan_dmax = dmax * self.replan_dmax_factor
                # max_retries=1, extra=0: replan on the existing roadmap only.
                # Allowing retries with extra nodes causes graph explosion —
                # up to 500 new nodes per replan × 80 wait ticks = 40k nodes,
                # turning each subsequent _add_vertex into thousands of PyBullet
                # calls (observed: >1000 s wall-clock for office/noise+dynamic).
                new_path, new_cost = self.planner.replan(
                    cur_tuple, goal, replan_dmax,
                    max_retries=1, extra=0, **kwargs)
                if new_path is None:
                    # Wait for dynamic obstacles to clear, retrying each tick
                    for _ in range(self.max_wait_ticks):
                        dyn_positions = self.dyn.step(dt=0.05)
                        sim_t += 0.05
                        rec.wait_ticks += 1
                        kw2 = {}
                        if dyn_positions:
                            kw2["dyn_positions"] = [
                                pos[:self.dim] for pos in dyn_positions]
                            kw2["affect_radius"] = replan_dmax * 0.6
                        new_path, _ = self.planner.replan(
                            cur_tuple, goal, replan_dmax,
                            max_retries=1, extra=0, **kw2)
                        if new_path is not None:
                            break
                if new_path is None:
                    break                    # timed out waiting → failure
                path = new_path
                wp_idx = 1

            # 3. Move toward next waypoint with noise
            if wp_idx >= len(path):
                break
            target = np.array(path[wp_idx], dtype=float)
            direction = target - current
            dist_to_wp = np.linalg.norm(direction)

            if dist_to_wp < self.step_size:
                commanded = tuple(target)
                wp_idx += 1
            else:
                commanded = tuple(current + self.step_size
                                  * direction / dist_to_wp)

            actual = self.noise.apply(commanded)
            current = np.array(actual, dtype=float)
            rec.executed_positions.append(tuple(current))

            # 4. Record clearance at actual position
            clr = self.planner.clearance(tuple(current))
            rec.clearances.append(clr)

            # 5. Collision? → failure
            if not self.planner.is_free(tuple(current)):
                break

            sim_t += 0.05

            # 6. Goal reached?
            if self._reached_goal(current, goal):
                rec.success = True
                break

        rec.execution_time = sim_t
        return rec


class OnlinePathExecutor:
    """
    Online incremental executor for RiskAwarePRM in --mode online.

    Requires planner.set_world_obstacles() (NOT set_obstacles()) and no
    upfront construct() call.  The roadmap starts empty and grows one tick
    at a time via planner.online_step().

    When no path to the goal exists yet, the planner routes the robot to
    exploration frontiers — reachable nodes that border unexplored space —
    so it discovers new corridors and eventually connects to the goal.
    No greedy goal-directed walk is used; the robot only moves along
    graph-planned paths or holds position while the roadmap grows.

    Parameters
    ----------
    planner              : RiskAwarePRM instance set up with set_world_obstacles().
    env                  : environment dict from a builder function.
    step_size            : metres per tick.
    max_execution_steps  : safety cap.
    sense_radius         : obstacle detection radius (m); defaults to env["sense_radius"].
    online_samples_per_tick : new nodes added near the robot each tick.
    """

    def __init__(self, planner, env, step_size=0.3,
                 max_execution_steps=3000, sense_radius=None,
                 online_samples_per_tick=15):
        self.planner = planner
        self.env = env
        self.step_size = step_size
        self.max_steps = max_execution_steps
        self.sense_radius = (sense_radius
                             if sense_radius is not None
                             else env.get("sense_radius", env["dmax"]))
        self.online_samples = online_samples_per_tick
        # Scale node cap with environment complexity: 2× the offline N gives the
        # online planner enough budget to cover the workspace while remaining
        # bounded.  simple→600, office→2000, cityscape→1600.
        self.max_nodes = env.get("N", 300) * env.get("max_nodes_multiplier", 2)
        self.noise = env["noise"]
        self.dyn = env["dyn_manager"]
        self.dim = env["dim"]
        self.goal_tol = env["robot_radius"] * 3

    def _reached_goal(self, pos, goal):
        return np.linalg.norm(np.array(pos) - np.array(goal)) < self.goal_tol

    def _path_ok(self, path, wp_idx):
        """Check if the next few edges on the current path are still free."""
        end = min(wp_idx + 3, len(path) - 1)
        for i in range(max(0, wp_idx - 1), end):
            if not self.planner.edge_free(path[i], path[i + 1]):
                return False
        return True

    def _is_goal_path(self, path, goal):
        """True if the path's final node is near the goal."""
        if not path:
            return False
        return np.linalg.norm(
            np.array(path[-1][:self.dim]) - np.array(goal[:self.dim])
        ) < self.goal_tol

    def execute(self):
        """Run online exploration from env start to goal; returns TrajectoryRecord."""
        rec = TrajectoryRecord()
        start = self.env["start"]
        goal  = self.env["goal"]
        dmax  = self.env["dmax"]

        current = np.array(start, dtype=float)
        rec.executed_positions.append(tuple(current))
        path   = None
        wp_idx = 1
        _step = 0

        for _step in range(self.max_steps):
            # 1. Tick dynamic obstacles
            dyn_positions = self.dyn.step(dt=0.05)
            if dyn_positions:
                rec.dyn_obstacle_positions.append(
                    [tuple(pos[:self.dim]) for pos in dyn_positions])

            # 2. Grow roadmap + attempt to plan
            new_path, _ = self.planner.online_step(
                tuple(current), goal, dmax,
                sense_radius=self.sense_radius,
                n_samples=self.online_samples,
                max_nodes=self.max_nodes)

            # 3. Decide whether to adopt the new path or keep following
            #    the current one.  Blindly adopting every tick causes
            #    oscillation: the fresh A* from a slightly shifted position
            #    may route backward through different edges.
            #
            #    Adopt the new path only when:
            #      (a) we have no current path (or it's exhausted)
            #      (b) the current path is blocked by a new obstacle
            #      (c) new path reaches the GOAL while old path doesn't
            #          (exploration → goal transition)
            need_new_path = False
            if path is None or wp_idx >= len(path):
                need_new_path = True                          # (a) no path
            elif not self._path_ok(path, wp_idx):
                need_new_path = True                          # (b) blocked
                rec.replans += 1
            elif (new_path is not None
                  and self._is_goal_path(new_path, goal)
                  and not self._is_goal_path(path, goal)):
                need_new_path = True                          # (c) goal found

            if need_new_path and new_path is not None:
                path   = new_path
                wp_idx = 1

            # 4. Move: follow path if available, else hold position
            if path is not None and wp_idx < len(path):
                target = np.array(path[wp_idx], dtype=float)
                direction = target - current
                dist_wp = np.linalg.norm(direction)
                at_wp = dist_wp < self.step_size
                proposed = (tuple(target) if at_wp
                            else tuple(current + self.step_size
                                       * direction / dist_wp))
                # For noisy environments, hold if the proposed step is
                # within 1-sigma of a wall.
                noise_std = getattr(self.noise, 'std', 0.0)
                robot_r   = getattr(self.planner, 'radius', 0.0)
                if (noise_std > 0.0
                        and self.planner.clearance(proposed) < robot_r + noise_std):
                    commanded = tuple(current)
                else:
                    commanded = proposed
                    if at_wp:
                        wp_idx += 1
            else:
                commanded = tuple(current)

            actual = self.noise.apply(commanded)
            current = np.array(actual, dtype=float)
            rec.executed_positions.append(tuple(current))

            # 5. Clearance + collision check
            clr = self.planner.clearance(tuple(current))
            rec.clearances.append(clr)
            if not self.planner.is_free(tuple(current)):
                break

            # 6. Goal reached?
            if self._reached_goal(current, goal):
                rec.success = True
                break

        rec.execution_time = float(_step) * 0.05
        return rec