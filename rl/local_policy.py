"""
Wrapper around the trained stable-baselines3 policy for use
in Monte Carlo rollouts during PRM-RL edge validation.

Also includes a ReactivePolicy fallback for testing the pipeline
without a trained model.
"""

import os
import numpy as np
from stable_baselines3 import PPO


class LocalPolicy:
    """
    Loads a trained PPO model and exposes a simple interface:

        action = policy.predict(robot_pos, goal_pos, ray_dists)

    Parameters
    ----------
    model_path : str
        Path to the .zip file saved by stable-baselines3.
    deterministic : bool
        If True, use the mean action (no sampling).  Recommended
        for Monte Carlo rollouts so variance comes from the
        environment (noise, dynamics) not the policy.
    """

    def __init__(self, model_path, deterministic=True):
        if not os.path.exists(model_path):
            if os.path.exists(model_path + ".zip"):
                model_path = model_path + ".zip"
            else:
                raise FileNotFoundError(
                    f"No trained model found at {model_path}. "
                    f"Run `python -m rl.train_local` first."
                )
        self.model = PPO.load(model_path)
        self.deterministic = deterministic

    def predict(self, robot_pos, goal_pos, ray_dists):
        """
        Get the action for the current state.

        Parameters
        ----------
        robot_pos : array-like, shape (2,)
        goal_pos  : array-like, shape (2,)
        ray_dists : array-like, shape (n_rays,)
            Normalised distances from lidar rays (0=touching, 1=max range).

        Returns
        -------
        action : np.ndarray, shape (2,)
            Velocity command (dx, dy), clipped to [-1, 1].
        """
        robot_pos = np.asarray(robot_pos, dtype=np.float32)
        goal_pos = np.asarray(goal_pos, dtype=np.float32)
        ray_dists = np.asarray(ray_dists, dtype=np.float32)

        goal_delta = goal_pos - robot_pos
        obs = np.concatenate([robot_pos, goal_delta, ray_dists])

        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return action


class ReactivePolicy:
    """
    A hand-coded fallback policy for testing the PRM-RL pipeline
    without needing a trained model.  Moves toward the goal and
    steers away from nearby obstacles detected by rays.

    Drop-in replacement for LocalPolicy — same predict() signature.
    """

    def __init__(self, obstacle_weight=0.8, goal_weight=1.0):
        self.obstacle_weight = obstacle_weight
        self.goal_weight = goal_weight

    def predict(self, robot_pos, goal_pos, ray_dists):
        robot_pos = np.asarray(robot_pos, dtype=np.float64)
        goal_pos = np.asarray(goal_pos, dtype=np.float64)
        ray_dists = np.asarray(ray_dists, dtype=np.float64)

        # Goal attraction
        delta = goal_pos - robot_pos
        dist = np.linalg.norm(delta)
        if dist > 1e-6:
            goal_dir = delta / dist
        else:
            goal_dir = np.zeros(2)

        # Obstacle repulsion from rays
        n_rays = len(ray_dists)
        angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
        repulsion = np.zeros(2)
        for i in range(n_rays):
            if ray_dists[i] < 0.6:  # close obstacle
                strength = (1.0 - ray_dists[i]) ** 2
                repulsion[0] -= strength * np.cos(angles[i])
                repulsion[1] -= strength * np.sin(angles[i])

        # Combine
        action = self.goal_weight * goal_dir + self.obstacle_weight * repulsion

        # Normalise to [-1, 1]
        max_val = np.max(np.abs(action)) if np.max(np.abs(action)) > 0 else 1
        action = np.clip(action / max_val, -1.0, 1.0)

        return action.astype(np.float32)