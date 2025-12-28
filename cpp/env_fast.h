// env_fast.h — Batched GridWorld in C++17 with AVX2 SIMD over batch dimension.
//
// Design:
//   - Structure of Arrays (SoA) layout: each state field is its own contiguous,
//     32-byte-aligned array indexed by env_id. This makes column-wise operations
//     SIMD-friendly (one __m256i / __m256 covers 8 envs).
//   - Wall grids stored per-env as a flat bitmap (uint8_t) — random access is
//     unavoidable, but it's a single load per env per step.
//   - Branchless bounds and wall checks via bitmask blends.
//
// Numerical equivalence with Python reference is verified by `tests/test_parity.py`
// (10k random seeds, max abs diff < 1e-7 over 200-step rollouts).
#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <memory>

namespace gridworld_fast {

struct GridWorldConfig {
    int32_t size           = 50;
    int32_t max_steps      = 200;
    float   hidden_walls_p = 0.05f;
    float   goal_reward    = 1.0f;
    float   wall_penalty   = -0.1f;
    float   step_penalty   = -0.001f;
};

// Aligned-allocator wrapper for SoA columns. 32-byte alignment is required
// for AVX2 _mm256_load_si256 / _mm256_load_ps without faulting.
template <typename T>
class AlignedBuffer {
public:
    AlignedBuffer() = default;
    explicit AlignedBuffer(std::size_t n) { allocate(n); }
    ~AlignedBuffer() { release(); }

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    AlignedBuffer(AlignedBuffer&& o) noexcept : ptr_(o.ptr_), n_(o.n_) {
        o.ptr_ = nullptr; o.n_ = 0;
    }

    void allocate(std::size_t n);
    void release();

    T*       data()       noexcept { return ptr_; }
    const T* data() const noexcept { return ptr_; }
    std::size_t size() const noexcept { return n_; }

private:
    T*          ptr_ = nullptr;
    std::size_t n_   = 0;
};

// Batched GridWorld. All envs share the same config but have independent
// state and wall grids. Envs are stepped in lockstep — actions[i] applies
// to env i.
class BatchedGridWorld {
public:
    BatchedGridWorld(int32_t num_envs, GridWorldConfig cfg, uint64_t seed);

    // Reset every env. Fresh wall grid per env, agent at (0,0), goal at (size-1, size-1).
    // Writes flattened observations into `obs_out` of shape [num_envs, 5].
    void reset_all(float* obs_out);

    // Reset only envs whose `mask[i] != 0` (used after done).
    void reset_masked(const uint8_t* mask, float* obs_out);

    // Vectorized step. `actions` length = num_envs.
    // Outputs: obs_out[num_envs, 5], rewards[num_envs], dones[num_envs], hit_wall[num_envs].
    void step(const int32_t* actions,
              float*        obs_out,
              float*        rewards,
              uint8_t*      dones,
              uint8_t*      hit_wall);

    int32_t num_envs() const noexcept { return num_envs_; }
    int32_t size()     const noexcept { return cfg_.size; }

private:
    void observe_(int32_t i, float* dst) const;
    bool wall_at_(int32_t env, int32_t x, int32_t y) const;
    void seed_walls_(int32_t env);

    GridWorldConfig cfg_;
    int32_t         num_envs_;
    uint64_t        rng_state_;

    // SoA columns — all aligned to 32 bytes for AVX2 loads.
    AlignedBuffer<int32_t> agent_x_;
    AlignedBuffer<int32_t> agent_y_;
    AlignedBuffer<int32_t> goal_x_;
    AlignedBuffer<int32_t> goal_y_;
    AlignedBuffer<int32_t> step_count_;
    AlignedBuffer<uint8_t> done_;

    // Scratch for the SIMD prologue: tentative new positions before wall check.
    AlignedBuffer<int32_t> tent_x_;
    AlignedBuffer<int32_t> tent_y_;

    // Walls: one bitmap per env, flat row-major. Total size = num_envs * size * size.
    std::vector<uint8_t> walls_;
};

}  // namespace gridworld_fast
