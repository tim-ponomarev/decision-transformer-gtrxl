"""Simple GridWorld with hidden walls — a toy env demonstrating partial observability
and combinatorial state space, where vanilla value-based RL struggles."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ACTION_NOOP = 0
ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
NUM_ACTIONS = 5


@dataclass
class GridWorldConfig:
    size: int = 50
    max_steps: int = 200
    hidden_walls_prob: float = 0.05
    goal_reward: float = 1.0
    wall_penalty: float = -0.1
    step_penalty: float = -0.001


class GridWorld:
    """GridWorld with random walls hidden from the agent's observation.

    Observation: (agent_x, agent_y, goal_x, goal_y, step_count/max_steps)
                 — hidden walls are NOT in the observation, creating partial observability.
    """

    def __init__(self, config: GridWorldConfig | None = None, seed: int | None = None):
        self.config = config or GridWorldConfig()
        self.rng = np.random.default_rng(seed)
        self.state_dim = 5
        self.num_actions = NUM_ACTIONS
        self.reset()

    def reset(self) -> np.ndarray:
        size = self.config.size
        self.agent_pos = np.array([0, 0], dtype=np.int32)
        self.goal_pos = np.array([size - 1, size - 1], dtype=np.int32)
        self.walls = (self.rng.random((size, size)) < self.config.hidden_walls_prob).astype(bool)
        self.walls[tuple(self.agent_pos)] = False
        self.walls[tuple(self.goal_pos)] = False
        self.step_count = 0
        self.done = False
        return self._observe()

    def _observe(self) -> np.ndarray:
        return np.array([
            self.agent_pos[0] / self.config.size,
            self.agent_pos[1] / self.config.size,
            self.goal_pos[0] / self.config.size,
            self.goal_pos[1] / self.config.size,
            self.step_count / self.config.max_steps,
        ], dtype=np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        if self.done:
            raise RuntimeError("Episode done — call reset()")

        delta = {
            ACTION_NOOP: (0, 0),
            ACTION_UP: (-1, 0),
            ACTION_DOWN: (1, 0),
            ACTION_LEFT: (0, -1),
            ACTION_RIGHT: (0, 1),
        }[action]
        new_pos = self.agent_pos + np.array(delta, dtype=np.int32)

        size = self.config.size
        reward = self.config.step_penalty
        hit_wall = False

        if not (0 <= new_pos[0] < size and 0 <= new_pos[1] < size):
            hit_wall = True
        elif self.walls[tuple(new_pos)]:
            hit_wall = True
        else:
            self.agent_pos = new_pos

        if hit_wall:
            reward += self.config.wall_penalty

        self.step_count += 1
        reached_goal = bool(np.array_equal(self.agent_pos, self.goal_pos))
        if reached_goal:
            reward += self.config.goal_reward
            self.done = True
        elif self.step_count >= self.config.max_steps:
            self.done = True

        return self._observe(), reward, self.done, {"hit_wall": hit_wall, "reached_goal": reached_goal}
