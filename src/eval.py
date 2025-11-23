"""Rollout a trained DT in the env and report win rate, reward distribution."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from env import GridWorld, GridWorldConfig
from model import DecisionTransformer


def rollout_one_episode(
    env: GridWorld,
    model: DecisionTransformer,
    target_return: float,
    device: torch.device,
    context_len: int = 30,
) -> tuple[float, bool, int]:
    state = env.reset()
    states = [state]
    actions = []
    returns = [target_return]
    total_reward = 0.0
    done = False
    t = 0

    while not done:
        # Build context window (last context_len steps)
        s_arr = np.array(states[-context_len:], dtype=np.float32)
        pad_needed = context_len - len(s_arr)
        if pad_needed > 0:
            s_arr = np.concatenate([np.zeros((pad_needed, env.state_dim), dtype=np.float32), s_arr])

        a_arr = np.zeros((context_len, env.num_actions), dtype=np.float32)
        for i, a in enumerate(actions[-context_len + 1:]):
            idx = i + max(0, context_len - len(actions) - 1)
            if idx < context_len:
                a_arr[idx, a] = 1.0

        r_arr = np.array(returns[-context_len:], dtype=np.float32)
        if len(r_arr) < context_len:
            r_arr = np.concatenate([np.zeros(context_len - len(r_arr), dtype=np.float32), r_arr])

        timesteps = np.arange(max(0, t - context_len + 1), t + 1)
        if len(timesteps) < context_len:
            timesteps = np.concatenate([np.zeros(context_len - len(timesteps), dtype=np.int64), timesteps])

        returns_t = torch.from_numpy(r_arr).unsqueeze(0).unsqueeze(-1).to(device)
        states_t = torch.from_numpy(s_arr).unsqueeze(0).to(device)
        actions_t = torch.from_numpy(a_arr).unsqueeze(0).to(device)
        timesteps_t = torch.from_numpy(timesteps.astype(np.int64)).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(returns_t, states_t, actions_t, timesteps_t)

        action_logits = out["action_logits"][0, -1]
        action = int(torch.argmax(action_logits).item())

        next_state, reward, done, info = env.step(action)
        total_reward += reward
        states.append(next_state)
        actions.append(action)
        returns.append(returns[-1] - reward)
        t += 1

    return total_reward, info.get("reached_goal", False), t


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/dt_gtrxl_wm.yaml")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--target-return", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model = DecisionTransformer(
        state_dim=cfg["state_dim"],
        action_dim=cfg["action_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        max_seq_len=cfg["max_seq_len"],
        max_timesteps=cfg.get("max_timesteps", 256),
        use_world_model=cfg["use_world_model"],
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    rewards = []
    wins = 0
    lengths = []

    for ep in range(args.episodes):
        env = GridWorld(seed=args.seed + ep)
        r, won, T = rollout_one_episode(env, model, args.target_return, device)
        rewards.append(r)
        lengths.append(T)
        if won:
            wins += 1

    print(f"\n=== Evaluation ({args.episodes} episodes) ===")
    print(f"  Win rate:        {wins / args.episodes:.2%}")
    print(f"  Mean reward:     {np.mean(rewards):.3f}")
    print(f"  Std reward:      {np.std(rewards):.3f}")
    print(f"  Mean length:     {np.mean(lengths):.1f}")


if __name__ == "__main__":
    main()
