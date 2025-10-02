"""Offline trajectory dataset for Decision Transformer.

Each sample is a contiguous window of `context_len` timesteps from a recorded
episode, with return-to-go computed on-the-fly so we can resample goals.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class OfflineTrajectoryDataset(Dataset):
    """Load `.pt` trajectories and sample context windows."""

    def __init__(self, path: str, context_len: int = 30):
        self.context_len = context_len
        # weights_only=False needed because trajectories are numpy arrays, not weights.
        # Safe here because we wrote the file ourselves in collect_trajectories.py.
        data = torch.load(path, map_location="cpu", weights_only=False)
        # Expected format: list of episodes, each with states, actions, rewards arrays
        self.episodes = data["episodes"]
        self.state_dim = self.episodes[0]["states"].shape[-1]
        self.action_dim = self._infer_action_dim(self.episodes[0])

        # Precompute valid (episode, start_index) pairs
        self.indices = []
        for ep_idx, ep in enumerate(self.episodes):
            T = len(ep["states"])
            if T < context_len + 1:
                continue
            for start in range(T - context_len):
                self.indices.append((ep_idx, start))

    @staticmethod
    def _infer_action_dim(ep: dict) -> int:
        actions = ep["actions"]
        if actions.ndim == 1:
            return int(actions.max()) + 1  # discrete
        return actions.shape[-1]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        ep_idx, start = self.indices[i]
        ep = self.episodes[ep_idx]
        end = start + self.context_len

        states = ep["states"][start:end]
        actions_raw = ep["actions"][start:end]
        rewards = ep["rewards"][start:end]
        next_states = ep["states"][start + 1:end + 1]

        # Return-to-go: sum of future rewards from step t to end of episode
        full_rewards = ep["rewards"]
        cum_future = np.cumsum(full_rewards[::-1])[::-1].copy()
        returns_to_go = cum_future[start:end]

        # Convert actions to one-hot if discrete
        if actions_raw.ndim == 1:
            n_actions = int(full_rewards.shape[0] and actions_raw.max() + 1)
            actions_oh = np.zeros((self.context_len, self.action_dim), dtype=np.float32)
            for t, a in enumerate(actions_raw):
                actions_oh[t, a] = 1.0
            actions_tensor = torch.from_numpy(actions_oh)
            target_actions = torch.from_numpy(actions_raw.astype(np.int64))
        else:
            actions_tensor = torch.from_numpy(actions_raw.astype(np.float32))
            target_actions = actions_tensor

        return {
            "returns": torch.from_numpy(returns_to_go.astype(np.float32)).unsqueeze(-1),
            "states": torch.from_numpy(states.astype(np.float32)),
            "actions": actions_tensor,
            "target_actions": target_actions,
            "next_states": torch.from_numpy(next_states.astype(np.float32)),
            "rewards": torch.from_numpy(rewards.astype(np.float32)),
            "timesteps": torch.arange(start, end, dtype=torch.long),
        }
