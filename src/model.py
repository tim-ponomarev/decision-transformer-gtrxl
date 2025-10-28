"""Decision Transformer with GTrXL backbone + optional World Model auxiliary head.

Input sequence: (R_1, s_1, a_1, R_2, s_2, a_2, ...) where R_t is return-to-go.
Predicts next action a_t given the history.

The World Model head is a side branch that predicts (next_state, reward) from
(state, action). It's trained jointly with the action prediction and acts as
a representation regularizer — encoder features must be useful for both
policy and dynamics prediction.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from gtrxl import GTrXLBlock, build_causal_mask


class DecisionTransformer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        max_seq_len: int = 600,  # 200 timesteps × 3 tokens per step (R, s, a)
        max_timesteps: int = 1024,  # max absolute timestep index (env episode length)
        dropout: float = 0.1,
        use_world_model: bool = True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.max_timesteps = max_timesteps
        self.use_world_model = use_world_model

        # Token embeddings for the 3 modalities
        self.return_embed = nn.Linear(1, hidden_dim)
        self.state_embed = nn.Linear(state_dim, hidden_dim)
        self.action_embed = nn.Linear(action_dim, hidden_dim)
        # Separate positional embedding sized to max episode length, not token sequence length
        self.timestep_embed = nn.Embedding(max_timesteps, hidden_dim)

        # Transformer backbone
        self.blocks = nn.ModuleList([
            GTrXLBlock(hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)

        # Action prediction head
        self.action_head = nn.Linear(hidden_dim, action_dim)

        # World model auxiliary heads (predict next_state and reward)
        if use_world_model:
            self.wm_next_state_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, state_dim),
            )
            self.wm_reward_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        returns: torch.Tensor,  # (B, T, 1)
        states: torch.Tensor,   # (B, T, state_dim)
        actions: torch.Tensor,  # (B, T, action_dim)
        timesteps: torch.Tensor,  # (B, T)
        memory: list[torch.Tensor] | None = None,
    ) -> dict:
        B, T, _ = states.shape

        # Embed each modality
        r_emb = self.return_embed(returns)
        s_emb = self.state_embed(states)
        a_emb = self.action_embed(actions)

        # Interleave as (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # Shape: (B, T, 3, D) -> (B, T*3, D)
        stacked = torch.stack([r_emb, s_emb, a_emb], dim=2).view(B, T * 3, self.hidden_dim)

        # Add timestep embeddings (same position for all 3 tokens of a timestep)
        t_emb = self.timestep_embed(timesteps)
        t_emb_expanded = t_emb.unsqueeze(2).expand(-1, -1, 3, -1).reshape(B, T * 3, self.hidden_dim)
        x = stacked + t_emb_expanded

        # Transformer blocks with causal mask
        seq_len = x.shape[1]
        mask = build_causal_mask(seq_len, mem_len=0, device=x.device)
        for block in self.blocks:
            x = block(x, memory=None, attn_mask=mask)
        x = self.final_norm(x)

        # Extract token positions: state tokens are at index 1, 4, 7, ... (every 3rd)
        state_token_idx = torch.arange(1, T * 3, 3, device=x.device)
        state_features = x[:, state_token_idx, :]  # (B, T, D)

        # Action prediction: from state token position, predict next action
        action_logits = self.action_head(state_features)

        out = {"action_logits": action_logits}

        # World Model: from action token positions, predict next state and reward
        if self.use_world_model:
            action_token_idx = torch.arange(2, T * 3, 3, device=x.device)
            action_features = x[:, action_token_idx, :]
            out["pred_next_state"] = self.wm_next_state_head(action_features)
            out["pred_reward"] = self.wm_reward_head(action_features).squeeze(-1)

        return out


def action_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy for discrete actions or MSE for continuous."""
    if targets.dtype == torch.long:
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    return F.mse_loss(logits, targets)


def world_model_loss(
    pred_next_state: torch.Tensor,
    true_next_state: torch.Tensor,
    pred_reward: torch.Tensor,
    true_reward: torch.Tensor,
    state_weight: float = 1.0,
    reward_weight: float = 0.1,
) -> torch.Tensor:
    state_loss = F.mse_loss(pred_next_state, true_next_state)
    reward_loss = F.mse_loss(pred_reward, true_reward)
    return state_weight * state_loss + reward_weight * reward_loss
