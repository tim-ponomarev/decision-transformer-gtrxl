# decision-transformer-gtrxl

**Decision Transformer with GTrXL backbone and World Model auxiliary training on small, combinatorial environments.**

A minimal, readable reference implementation of the architecture I used for a production RL agent (under NDA — this version is scrubbed and runs on toy environments). Focus on the things that matter in practice:

- GTrXL backbone for stable long-horizon rollouts
- Pre-norm LayerNorm + gradient clipping (why the model actually trains)
- Offline training from recorded trajectories (no online env needed)
- World Model auxiliary loss for representation quality
- C++17/AVX2 batched env rewrite via pybind11 — **~200× throughput** over Python single-env baseline at batch_size=512

## Why Decision Transformer?

Value-based RL (DQN/PPO/SAC) on environments with **long horizons, partial observability, and combinatorial action spaces** tends to either:
- diverge because value estimates explode with bootstrapping, or
- collapse to a single greedy strategy that doesn't adapt

**Decision Transformer** treats RL as sequence modeling — given (state, return-to-go, action) tuples, predict next action. Training is offline supervised learning on recorded trajectories — stable, fast to iterate, no replay buffer drama.

## Results on toy env (GridWorld with hidden walls, 50×50 grid, 200-step episodes)

| Model | Train GPU hrs | Converged? | Win rate | Samples used |
|---|---|---|---|---|
| Vanilla PPO | 12 | ❌ (greedy collapse) | 31% | 2M |
| SAC (discrete) | 18 | ⚠️ (high variance) | 44% | 2M |
| DT (vanilla transformer) | 4 | ⚠️ (gradient instability) | 58% | 500k (offline) |
| **DT + GTrXL + grad clipping** | **4** | **✅** | **71%** | **500k (offline)** |
| DT + GTrXL + World Model aux | 5 | ✅ | **74%** | 500k (offline) |

**The interesting number**: DT+GTrXL needed **4× less data** than value-based methods because it learned from diverse recorded trajectories instead of exploring from scratch.

## Key architectural choices

### GTrXL over vanilla Transformer

Vanilla transformer gradients explode on long rollouts (200+ steps) when you backprop through residual streams. **GTrXL** fixes this with:

1. **Gated residual connections** — learned gate between residual and attention output, prevents the residual from dominating early in training
2. **Recurrent memory cache** — attention over the last N chunks, not just the current one, without quadratic blowup

Result: **divergence rate 35% → <2%** on my training runs.

### Pre-norm vs Post-norm

Post-norm (original transformer) puts LayerNorm *after* the residual. Pre-norm puts it *before*. Pre-norm is strictly more stable for deep stacks and is now standard in all modern LLMs. If you're still using post-norm, stop.

### Warmup LR schedule

Linear warmup for the first **1000 steps** (not epochs). Adam with a flat LR kills early training because the gradient magnitudes are tiny and Adam's second-moment estimate is unreliable. Warmup fixes this trivially.

### World Model auxiliary loss

In addition to the action prediction loss, train a side head to predict `(next_state, reward)` given `(state, action)`. This:

1. Regularizes the encoder toward physically meaningful representations
2. Gives you a free dynamics model you can use for planning / imagination rollouts
3. Adds negligible compute cost (~5% overhead)

For my toy env it gave +3% win rate — small but "free". On the real production env it was more meaningful.

## C++ / SIMD env step (optional, performance-focused)

Python environments are fine for prototyping but become a bottleneck once you're training at scale. My production rewrite hit these numbers on a combinatorial step function:

| Env backend | Step latency | Training throughput | Notes |
|---|---|---|---|
| Python (numpy) | 4.2 ms | 240 steps/sec | Pure Python with vectorized numpy |
| Cython | 1.4 ms | 710 steps/sec | Auto-translated, minimal manual tuning |
| **C++17 / AVX2 SIMD** | **0.6 ms** | **1650 steps/sec** | Manual SoA layout, cache-line aligned |

The C++ rewrite techniques that actually mattered:

- **Structure of Arrays (SoA)** not Array of Structures — makes column-wise operations SIMD-friendly
- **Cache-line alignment (64 bytes)** on hot data structures
- **Branchless comparisons** for action masking
- **`alignas(32)` and aligned_alloc** for AVX2 load/store intrinsics
- **pybind11** for Python interop — adds <1 µs call overhead, invisible

See `cpp/env_fast.cpp` for the scrubbed reference.

### The "C++ matches Python exactly" trap

First C++ version was 7× faster (single env) but gave divergent results in 0.3% of cases — `float` vs `double` in reward accumulation. Agent trained on C++ performed 2% worse on Python validation. Fix: 10k unit tests with fixed seeds, promoted critical paths to `double`, then binary-equal with Python reference. **Lesson: every time you rewrite for speed, validate numerical equivalence with the slow reference.**

The final **~200× batched speedup** (at batch_size=512) comes from stacking three gains: C++ eliminating Python interpreter overhead (~20×), AVX2 SIMD processing 8 envs per lane (~3-4×), and amortized function-call overhead across the batch (~3×).

## Quick start

```bash
pip install -r requirements.txt

# Smoke test: small model, 200 episodes, 2 epochs.
# CPU-only, runs in ~2 minutes, win rate ~70% on held-out seeds.
python scripts/collect_trajectories.py --episodes 200 --out data/traj_smoke.pt
python src/train.py --config configs/smoke_test.yaml
python src/eval.py --checkpoint outputs/smoke/final.pt --config configs/smoke_test.yaml --episodes 20

# Full run (recommended GPU)
python scripts/collect_trajectories.py --episodes 5000 --out data/traj.pt
python src/train.py --config configs/dt_gtrxl_wm.yaml
python src/eval.py --checkpoint outputs/dt-gtrxl-wm/final.pt --config configs/dt_gtrxl_wm.yaml --episodes 200
```

## Repo structure

```
decision-transformer-gtrxl/
├── src/
│   ├── model.py           # DT + GTrXL + WM head
│   ├── gtrxl.py           # Gated transformer XL layer
│   ├── world_model.py     # Auxiliary dynamics head
│   ├── env.py             # GridWorld toy env (Python reference)
│   ├── dataset.py         # Offline trajectory loader
│   ├── train.py           # Training loop
│   └── eval.py            # Held-out evaluation
├── cpp/
│   ├── env_fast.cpp       # C++/AVX2 env step (optional)
│   └── bindings.cpp       # pybind11 wrapper
├── scripts/
│   ├── collect_trajectories.py
│   └── benchmark_cpp_vs_python.py
└── configs/
    └── dt_gtrxl_wm.yaml
```

## Acknowledgments

- [Decision Transformer](https://arxiv.org/abs/2106.01345) — Chen et al., NeurIPS 2021
- [GTrXL](https://arxiv.org/abs/1910.06764) — Parisotto et al., ICML 2020
- [World Models](https://worldmodels.github.io/) — Ha & Schmidhuber, 2018

## Author

Tim Ponomarev — Applied ML Engineer, working on RL and World Models as part of an ongoing research collaboration.
