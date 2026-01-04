# cpp/ — fast batched env

C++17 + AVX2 SIMD reimplementation of the GridWorld env, vectorized over the **batch dimension** (one `__m256i` lane per env). Used optionally during training to bring the env step out of the Python critical path.

## Why this exists

The Python env (`src/env.py`) is fine for prototyping but becomes the bottleneck once batched rollout collection is in the loop. Profiling showed ~70% of training wall-time was spent in Python `step()` and numpy boxing on the smoke config. C++ rewrite drops it to <10%.

| Backend           | Step latency (batched, n=512) | Throughput        |
|-------------------|-------------------------------|-------------------|
| Python (numpy)    | 4.2 ms                        | 240 steps/sec     |
| Cython            | 1.4 ms                        | 710 steps/sec     |
| **C++17 / AVX2**  | **0.6 ms**                    | **1650 steps/sec**|

(Numbers from `scripts/benchmark_cpp_vs_python.py` on i9-9900K, AVX2 only.)

## Build

Requires CMake ≥ 3.16, a C++17 compiler with AVX2, and pybind11 (`pip install pybind11`).

```bash
cd cpp
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
cmake --install build
```

This produces `gridworld_fast.<so/pyd>` and copies it next to `src/`. Then from Python:

```python
import numpy as np
import gridworld_fast as ef

cfg = ef.GridWorldConfig()
cfg.size = 50

n   = 512
env = ef.BatchedGridWorld(num_envs=n, config=cfg, seed=42)

obs      = np.zeros((n, 5), dtype=np.float32)
rewards  = np.zeros(n,      dtype=np.float32)
dones    = np.zeros(n,      dtype=np.uint8)
hit_wall = np.zeros(n,      dtype=np.uint8)
actions  = np.zeros(n,      dtype=np.int32)

env.reset_all(obs)
for _ in range(200):
    actions[:] = np.random.randint(0, 5, size=n)
    env.step(actions, obs, rewards, dones, hit_wall)
    if dones.any():
        env.reset_masked(dones, obs)
```

## Numerical parity with Python reference

The first version of this rewrite was 7× faster but gave divergent results in 0.3% of cases — `float` vs `double` rounding in the reward calc. After promoting reward accumulation to `double` internally and rounding to `float` only on the output boundary, the C++ matches the Python reference bit-for-bit on 10k random seeds × 200-step rollouts. **Lesson: every speedup rewrite needs a parity test against the slow reference.** See `tests/test_parity.py` (not included in this scrubbed reference).

## Why not just use Cython?

Cython gets you to within 3× of hand-tuned C++ with way less effort (see table). For this project the extra 2× was worth it because the env was the actual bottleneck; on a project where it's a smaller fraction of training time, Cython would be the right choice.

## Files

- `env_fast.h`     — public interface, SoA columns, AlignedBuffer<T> wrapper
- `env_fast.cpp`   — AVX2 batched step, scalar wall lookups, branchless bounds
- `bindings.cpp`   — pybind11 wrapper (zero-copy numpy, no GIL trick yet)
- `CMakeLists.txt` — build with `/arch:AVX2` (MSVC) or `-mavx2 -mfma -O3` (GCC/Clang)
