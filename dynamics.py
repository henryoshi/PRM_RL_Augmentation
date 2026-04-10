"""
Dynamic obstacle controllers and motion-noise model.
Each DynamicObstacle wraps a PyBullet body and updates its position
every simulation tick.
"""

import numpy as np
import pybullet as p


# ═══════════════════════════════════════════════════════════════════════
#  Dynamic obstacle types
# ═══════════════════════════════════════════════════════════════════════

class DynamicObstacle:
    """Base class — subclass and override `position_at(t)`."""

    def __init__(self, cid, body_id, dim=2):
        self.cid = cid
        self.body_id = body_id
        self.dim = dim
        self._base_pos, _ = p.getBasePositionAndOrientation(
            body_id, physicsClientId=cid)

    def position_at(self, t):
        """Return (x, y, z) position at time t."""
        return self._base_pos

    def update(self, t):
        pos = self.position_at(t)
        p.resetBasePositionAndOrientation(
            self.body_id, pos, [0, 0, 0, 1], physicsClientId=self.cid)
        return pos

    @property
    def pos_2d(self):
        pos, _ = p.getBasePositionAndOrientation(
            self.body_id, physicsClientId=self.cid)
        return pos[:2] if self.dim == 2 else pos


class PatrolObstacle(DynamicObstacle):
    """Moves linearly between two endpoints at constant speed."""

    def __init__(self, cid, body_id, start, end, speed=1.0, dim=2):
        super().__init__(cid, body_id, dim)
        self.start = np.array(start, dtype=float)
        self.end = np.array(end, dtype=float)
        self.speed = speed
        self._length = np.linalg.norm(self.end - self.start)
        self._period = (2 * self._length / speed) if speed > 0 else 1.0

    def position_at(self, t):
        phase = (t % self._period) / self._period   # 0 → 1 → 0
        frac = 1 - abs(2 * phase - 1)               # triangle wave
        xy = self.start + frac * (self.end - self.start)
        z = self._base_pos[2]
        if self.dim == 2:
            return [float(xy[0]), float(xy[1]), z]
        return [float(xy[0]), float(xy[1]), float(xy[2])]


class OscillateObstacle(DynamicObstacle):
    """Sinusoidal motion along one axis (good for doors / barriers)."""

    def __init__(self, cid, body_id, axis, amplitude, period, dim=2):
        super().__init__(cid, body_id, dim)
        self.axis = axis            # 0=x, 1=y, 2=z
        self.amplitude = amplitude
        self.period = period

    def position_at(self, t):
        pos = list(self._base_pos)
        pos[self.axis] += self.amplitude * np.sin(2 * np.pi * t / self.period)
        return pos


class RandomWalkObstacle(DynamicObstacle):
    """Bounded random walk — good for simulating pedestrians / drones."""

    def __init__(self, cid, body_id, bounds, step_size=0.15,
                 dim=2, seed=None):
        super().__init__(cid, body_id, dim)
        self._bounds = bounds            # [(lo, hi), ...]
        self._step = step_size
        self._rng = np.random.RandomState(seed)
        self._pos = np.array(self._base_pos[:3], dtype=float)

    def position_at(self, t):
        delta = self._rng.normal(0, self._step, size=3)
        if self.dim == 2:
            delta[2] = 0
        self._pos += delta
        for i in range(min(self.dim, len(self._bounds))):
            lo, hi = self._bounds[i]
            self._pos[i] = np.clip(self._pos[i], lo, hi)
        return self._pos.tolist()


# ═══════════════════════════════════════════════════════════════════════
#  Manager — ticks all dynamic obstacles at once
# ═══════════════════════════════════════════════════════════════════════

class DynamicObstacleManager:
    def __init__(self):
        self.obstacles: list[DynamicObstacle] = []
        self.t = 0.0

    def add(self, obs: DynamicObstacle):
        self.obstacles.append(obs)

    def step(self, dt=0.05):
        self.t += dt
        positions = []
        for obs in self.obstacles:
            pos = obs.update(self.t)
            positions.append(pos)
        return positions

    def current_positions(self):
        return [obs.pos_2d for obs in self.obstacles]

    @property
    def body_ids(self):
        return [obs.body_id for obs in self.obstacles]

    def reset(self):
        self.t = 0.0
        for obs in self.obstacles:
            obs.update(0.0)


# ═══════════════════════════════════════════════════════════════════════
#  Motion noise model
# ═══════════════════════════════════════════════════════════════════════

class MotionNoise:
    """Gaussian perturbation applied to commanded positions."""

    def __init__(self, std=0.0, dim=2, bounds=None):
        """
        Parameters
        ----------
        std : float
            Noise standard deviation (metres). 0 = deterministic.
        dim : int
            2 or 3.
        bounds : list of (lo, hi) or None
            Workspace bounds for clamping noisy positions.
        """
        self.std = std
        self.dim = dim
        self.bounds = bounds

    def apply(self, commanded_pos):
        if self.std <= 0:
            return commanded_pos
        noise = np.random.normal(0, self.std, size=self.dim)
        noisy = np.array(commanded_pos[:self.dim]) + noise
        if self.bounds:
            for i in range(self.dim):
                noisy[i] = np.clip(noisy[i], self.bounds[i][0],
                                   self.bounds[i][1])
        return tuple(noisy)
