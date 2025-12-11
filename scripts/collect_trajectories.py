"""Collect offline trajectories using a scripted (greedy towards goal) policy + noise.

The point is to have a diverse dataset of both good and mediocre episodes —
Decision Transformer learns to imitate based on return-to-go conditioning, so
variety helps more than optimality.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from any CWD by adding project src/ to path
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np
import torch
from tqdm import trange

from env import ACTION_DOWN, ACTION_LEFT, ACTION_NOOP, ACTION_RIGHT, ACTION_UP, GridWorld


def greedy_action(env: GridWorld) -> int:
    dx = env.goal_pos[0] - env.agent_pos[0]
    dy = env.goal_pos[1] - env.agent_pos[1]
    if abs(dx) > abs(dy):
        return ACTION_DOWN if dx > 0 else ACTION_UP
    if dy != 0:
        return ACTION_RIGHT if dy > 0 else ACTION_LEFT
    return ACTION_NOOP


def collect(num_episodes: int, noise: float, out_path: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    episodes = []
    for ep in trange(num_episodes, desc="Collecting"):
        env = GridWorld(seed=seed + ep)
        state = env.reset()
        states, actions, rewards = [state], [], []
        done = False
        while not done:
            if rng.random() < noise:
                action = int(rng.integers(0, env.num_actions))
            else:
                action = greedy_action(env)
            state, reward, done, _ = env.step(action)
            actions.append(action)
            rewards.append(reward)
            if not done:
                states.append(state)
            else:
                states.append(state)

        episodes.append({
            "states": np.array(states[:-1], dtype=np.float32),
            "actions": np.array(actions, dtype=np.int64),
            "rewards": np.array(rewards, dtype=np.float32),
        })

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"episodes": episodes}, out_path)
    total_steps = sum(len(e["actions"]) for e in episodes)
    mean_return = np.mean([e["rewards"].sum() for e in episodes])
    print(f"Saved {num_episodes} episodes ({total_steps} steps, mean return {mean_return:.3f}) to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--noise", type=float, default=0.3, help="Random action probability")
    parser.add_argument("--out", default="data/traj.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    collect(args.episodes, args.noise, args.out, args.seed)
