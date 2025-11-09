"""Decision Transformer training loop with gradient clipping and warmup schedule.

Two pieces of magic that matter:
1. Linear warmup for 1000 steps — prevents early gradient explosion
2. Gradient clipping at max_norm=1.0 — hard cap on per-batch updates

Without these two, ~35% of my training runs diverged in the first 2k steps.
With them, <2%.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False

    class _WandbStub:
        def init(self, *args, **kwargs):
            return None

        def log(self, *args, **kwargs):
            return None

        def finish(self, *args, **kwargs):
            return None

    wandb = _WandbStub()

from dataset import OfflineTrajectoryDataset
from model import DecisionTransformer, action_loss, world_model_loss


@dataclass
class TrainConfig:
    state_dim: int
    action_dim: int
    hidden_dim: int
    num_heads: int
    num_layers: int
    max_seq_len: int
    max_timesteps: int
    use_world_model: bool
    world_model_weight: float

    data_path: str
    batch_size: int
    context_len: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    grad_clip: float

    seed: int
    output_dir: str
    wandb_project: str
    run_name: str


def load_config(path: str) -> TrainConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return TrainConfig(**data)


def linear_warmup(step: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    return 1.0


def train(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=vars(cfg))

    # Data
    dataset = OfflineTrajectoryDataset(cfg.data_path, context_len=cfg.context_len)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2, drop_last=True)

    # Model
    model = DecisionTransformer(
        state_dim=cfg.state_dim,
        action_dim=cfg.action_dim,
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        max_seq_len=cfg.max_seq_len,
        max_timesteps=cfg.max_timesteps,
        use_world_model=cfg.use_world_model,
    ).to(device)

    # Optimizer with weight decay only on non-bias, non-norm parameters
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.ndim == 1 or "bias" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
    )

    step = 0
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.num_epochs):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{cfg.num_epochs}")

        for batch in pbar:
            returns = batch["returns"].to(device)
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            timesteps = batch["timesteps"].to(device)
            next_states = batch["next_states"].to(device)
            rewards = batch["rewards"].to(device)
            target_actions = batch["target_actions"].to(device)

            out = model(returns, states, actions, timesteps)
            act_loss = action_loss(out["action_logits"], target_actions)

            total = act_loss
            if cfg.use_world_model:
                wm_l = world_model_loss(
                    out["pred_next_state"], next_states,
                    out["pred_reward"], rewards,
                )
                total = act_loss + cfg.world_model_weight * wm_l

            optimizer.zero_grad()
            total.backward()

            # Gradient clipping — critical for stability
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)

            # Linear warmup
            lr_scale = linear_warmup(step, cfg.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = cfg.learning_rate * lr_scale

            optimizer.step()

            step += 1
            pbar.set_postfix({"loss": float(total), "grad_norm": float(grad_norm)})
            wandb.log({
                "train/loss": float(total),
                "train/action_loss": float(act_loss),
                "train/grad_norm": float(grad_norm),
                "train/lr": cfg.learning_rate * lr_scale,
            }, step=step)

        ckpt_path = output_dir / f"epoch_{epoch + 1}.pt"
        torch.save({"model": model.state_dict(), "step": step}, ckpt_path)
        print(f"Saved {ckpt_path}")

    final_path = output_dir / "final.pt"
    torch.save({"model": model.state_dict(), "step": step}, final_path)
    print(f"Saved {final_path}")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)
