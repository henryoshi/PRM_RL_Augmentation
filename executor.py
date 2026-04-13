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
    planning_time: float = 0.0      # seconds spent in construct + find
    execution_time: float = 0.0     # seconds of simulated execution
    total_time: float = 0.0         # planning + execution wall-clock


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
                 replan_horizon=3, max_execution_steps=3000):
        self.planner = planner
        self.env = env
        self.step_size = step_size
        self.replan_horizon = replan_horizon
        self.max_steps = max_execution_steps
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
            dyn_positions = self.dyn.step(dt=0.05)
            # Update planner's obstacle list (statics + dynamic bodies)
            all_ids = self.env["static_ids"] + self.dyn.body_ids
            self.planner.set_obstacles(all_ids)

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
                new_path, new_cost = self.planner.replan(
                    cur_tuple, goal, dmax, **kwargs)
                if new_path is None:
                    break                    # can't replan → failure
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
