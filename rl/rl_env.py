"""
Gym-style environment for training a local navigation policy.

Task: navigate from a start point to a goal point in a PyBullet workspace
      while avoiding obstacles.  This is the LOCAL planner — it handles
      short hops between PRM nodes, not full mission planning.

Observation (continuous, 2 + 2 + N_rays):
    [robot_x, robot_y,          # current position
     goal_dx, goal_dy,          # vector to goal (relative)
     ray_0_dist, ...,           # lidar-like obstacle sensing
     ray_N-1_dist]

Action (continuous, 2):
    [dx, dy]  — velocity command, clipped to max_step

Reward:
    +100  for reaching the goal
    -100  for collision
    -1    per step (encourages efficiency)
    + shaped bonus for getting closer to goal
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data


class LocalNavEnv(gym.Env):
    """
    A single short-range navigation episode between two points.

    Parameters
    ----------
    workspace_bounds : list of (lo, hi)
        e.g. [(0, 12), (0, 12)] for the simple environment.
    robot_radius : float
        Collision sphere radius.
    max_step : float
        Maximum distance the robot can move per action.
    n_rays : int
        Number of lidar-like distance sensors evenly spaced around robot.
    ray_length : float
        Maximum sensing range for each ray.
    max_episode_steps : int
        Episode terminates after this many steps (timeout).
    goal_tol : float
        Distance within which the goal is considered reached.
    obstacle_config : str
        Which environment to load: 'simple', 'office', or 'cityscape'.
    difficulty : int
        0-3 difficulty level (controls dynamic obstacles and noise).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        workspace_bounds=None,
        robot_radius=0.2,
        max_step=0.3,
        n_rays=16,
        ray_length=3.0,
        max_episode_steps=200,
        goal_tol=0.6,
        obstacle_config="simple",
        difficulty=0,
        gui=False,
    ):
        super().__init__()

        self.robot_radius = robot_radius
        self.max_step = max_step
        self.n_rays = n_rays
        self.ray_length = ray_length
        self.max_episode_steps = max_episode_steps
        self.goal_tol = goal_tol
        self.obstacle_config = obstacle_config
        self.difficulty = difficulty
        self.gui = gui

        # ── PyBullet setup ───────────────────────────────────────────
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self.cid
        )
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)
        p.loadURDF("plane.urdf", physicsClientId=self.cid)

        # Build the same obstacles as the PRM environments
        from environments import ENV_BUILDERS
        self._env_builder = ENV_BUILDERS[obstacle_config]

        # We'll rebuild obstacles each reset to keep things clean
        self.obstacle_ids = []
        self.bounds = workspace_bounds or [(0, 12), (0, 12)]
        self.dim = 2

        # Collision probe sphere (same as PRMBase)
        self._col = p.createCollisionShape(
            p.GEOM_SPHERE, radius=robot_radius, physicsClientId=self.cid
        )
        self._probe = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=self._col,
            basePosition=[0, 0, -100],
            physicsClientId=self.cid,
        )

        # ── Gym spaces ───────────────────────────────────────────────
        obs_dim = 2 + 2 + n_rays
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # action: [dx, dy] normalised to [-1, 1], scaled by max_step
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # ── Episode state ────────────────────────────────────────────
        self.robot_pos = None
        self.goal_pos = None
        self.step_count = 0
        self._prev_dist = None

        # Pre-compute ray angles
        self._ray_angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)

        # Will be populated on first reset
        self._env_data = None

    # ── Core Gym methods ─────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        # Rebuild environment obstacles from scratch
        self._rebuild_obstacles()

        # Sample random start and goal that are collision-free and
        # separated by a distance in the PRM edge-length regime
        dmax = self._env_data["dmax"]
        min_sep = dmax * 0.3
        max_sep = dmax * 1.0

        for _ in range(500):
            start = self._random_free_point()
            goal = self._random_free_point()
            sep = np.linalg.norm(np.array(start) - np.array(goal))
            if min_sep <= sep <= max_sep:
                break

        self.robot_pos = np.array(start, dtype=np.float64)
        self.goal_pos = np.array(goal, dtype=np.float64)
        self._prev_dist = np.linalg.norm(self.robot_pos - self.goal_pos)

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.step_count += 1

        # Scale action
        action = np.clip(action, -1.0, 1.0).astype(np.float64)
        move = action * self.max_step

        # Apply movement
        new_pos = self.robot_pos + move

        # Clamp to workspace bounds
        for i in range(self.dim):
            new_pos[i] = np.clip(
                new_pos[i], self.bounds[i][0], self.bounds[i][1]
            )

        self.robot_pos = new_pos

        # ── Check termination ────────────────────────────────────────
        dist_to_goal = np.linalg.norm(self.robot_pos - self.goal_pos)
        collision = not self._is_free(self.robot_pos)
        reached_goal = dist_to_goal < self.goal_tol
        timeout = self.step_count >= self.max_episode_steps

        # ── Reward ───────────────────────────────────────────────────
        reward = -1.0  # step penalty

        # Shaped: reward for getting closer
        dist_improvement = self._prev_dist - dist_to_goal
        reward += 10.0 * dist_improvement
        self._prev_dist = dist_to_goal

        if reached_goal:
            reward += 100.0
        if collision:
            reward -= 100.0

        terminated = reached_goal or collision
        truncated = timeout and not terminated

        obs = self._get_obs()
        info = {
            "reached_goal": reached_goal,
            "collision": collision,
            "dist_to_goal": dist_to_goal,
        }

        return obs, reward, terminated, truncated, info

    def close(self):
        if self.cid is not None:
            try:
                p.disconnect(self.cid)
            except Exception:
                pass
            self.cid = None

    # ── Observation ──────────────────────────────────────────────────

    def _get_obs(self):
        goal_delta = self.goal_pos - self.robot_pos
        ray_dists = self._cast_rays(self.robot_pos)
        obs = np.concatenate(
            [self.robot_pos, goal_delta, ray_dists]
        ).astype(np.float32)
        return obs

    def _cast_rays(self, pos):
        """Cast n_rays from pos, return normalised distances [0, 1]."""
        dists = np.ones(self.n_rays, dtype=np.float64)
        z = self.robot_radius + 0.01

        from_pos = [pos[0], pos[1], z]
        for i, angle in enumerate(self._ray_angles):
            dx = self.ray_length * np.cos(angle)
            dy = self.ray_length * np.sin(angle)
            to_pos = [pos[0] + dx, pos[1] + dy, z]

            result = p.rayTest(from_pos, to_pos, physicsClientId=self.cid)
            if result and result[0][0] != -1:
                dists[i] = result[0][2]

        return dists

    # ── Collision check ──────────────────────────────────────────────

    def _is_free(self, pos):
        pos3 = [pos[0], pos[1], self.robot_radius + 0.01]
        p.resetBasePositionAndOrientation(
            self._probe, pos3, [0, 0, 0, 1], physicsClientId=self.cid
        )
        for oid in self.obstacle_ids:
            if p.getClosestPoints(
                self._probe, oid, 0.0, physicsClientId=self.cid
            ):
                return False
        return True

    def _random_free_point(self):
        for _ in range(1000):
            pt = tuple(
                np.random.uniform(lo + self.robot_radius,
                                  hi - self.robot_radius)
                for lo, hi in self.bounds[:self.dim]
            )
            if self._is_free(np.array(pt)):
                return pt
        raise RuntimeError("Cannot find free point after 1000 attempts")

    # ── Environment obstacle building ────────────────────────────────

    def _rebuild_obstacles(self):
        """Recreate obstacles from the environment builder's rect data
        in our own PyBullet client."""

        # Remove old obstacles
        for oid in self.obstacle_ids:
            try:
                p.removeBody(oid, physicsClientId=self.cid)
            except Exception:
                pass
        self.obstacle_ids = []

        # Build a fresh env using the standard builder
        # (this creates its own pybullet client)
        env_data = self._env_builder(
            difficulty=self.difficulty, gui=False
        )
        self._env_data = env_data
        self.bounds = env_data["bounds"]

        # Disconnect the builder's client — we recreate obstacles in ours
        builder_cid = env_data["cid"]
        p.disconnect(builder_cid)

        # Recreate static obstacles from rect data
        rects = env_data.get("rects", [])
        for (cx, cy, hx, hy, height) in rects:
            hz = height / 2
            col = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[hx, hy, hz],
                physicsClientId=self.cid,
            )
            body = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                basePosition=[cx, cy, hz],
                physicsClientId=self.cid,
            )
            self.obstacle_ids.append(body)