// 85_online_softmax.cuh -- running max / sum + exp2 + rescale (FMHA softmax)
//
// ARCH: sm_90a
//
// FlashAttention online softmax core (per-row). Maintains running max/sum
// across K-sequence tiles. The corresponding "correction" warp uses the
// rescale factor returned per update to scale the running PV accumulator O.
//
// Two API styles are kept:
//   SoftmaxState         + softmax_state_init / _update / _finalize_inv
//                        Caller pre-computes row_max and row_sum_exp.
//   OnlineSoftmaxRow     + online_softmax_update_row
//                        Composite computes the per-tile max+exp2 itself.
//
// Plus warp-reduction helpers (warp_row_max / warp_row_sum / softmax_finalize).

#pragma once

// Source: knowledge/building_blocks/softmax_warp.md
// PTX:    9.7.18.8 (TMEM ld/st), 9.7.10.24 (cvt), 9.7.15.13 (redux.sync for max/sum)
//
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>
#include "../primitives/45_shfl_sync.cuh"

// =============================================================================
// Caller-supplied row stats (compact API)
// =============================================================================

struct SoftmaxState {
  float m;   // running max
  float l;   // running sum
};

// Initialize so the first max replaces m.
__device__ __forceinline__
void softmax_state_init(SoftmaxState& s) {
  s.m = -INFINITY;
  s.l = 0.f;
}

// Update with one K-tile's pre-computed stats for this row.
//   row_max     -- max(S) across the K-block for the current row.
//   row_sum_exp -- sum of exp2(S - row_max) across the K-block.
// Returns the correction factor that must be applied to the pre-existing
// PV accumulator (caller multiplies its accumulator by the returned value).
__device__ __forceinline__
float softmax_state_update(SoftmaxState& s,
                           float row_max, float row_sum_exp) {
  float new_m       = fmaxf(s.m, row_max);
  float correction  = exp2f(s.m - new_m);
  float row_renorm  = exp2f(row_max - new_m);
  s.l = s.l * correction + row_sum_exp * row_renorm;
  s.m = new_m;
  return correction;
}

// Finalize: return 1 / l for normalization of the output O.
__device__ __forceinline__
float softmax_state_finalize_inv(const SoftmaxState& s) {
  return 1.0f / s.l;
}

// =============================================================================
// Composite-driven row stats (API computes max + exp2 itself)
// =============================================================================

struct OnlineSoftmaxRow {
  float running_max;
  float running_sum;

  __device__ __forceinline__
  OnlineSoftmaxRow() : running_max(-INFINITY), running_sum(0.0f) {}
};

// Process one K-tile for a single row.
//   scale_log2  -- softmax temperature in log2 space (1/sqrt(d) * log2(e)).
//   tile_values -- in: S values for this K-tile; out: P = softmax values.
//   K_BLOCK     -- tile width.
// Returns the rescale factor for the correction warp to apply to the running
// O accumulator.
__device__ __forceinline__
float online_softmax_update_row(OnlineSoftmaxRow& state,
                                float* tile_values,
                                int K_BLOCK,
                                float scale_log2) {
  // 1. Compute tile max (scaled).
  float tile_max = -INFINITY;
  for (int k = 0; k < K_BLOCK; ++k) {
    float scaled = tile_values[k] * scale_log2;
    if (scaled > tile_max) tile_max = scaled;
  }
  // 2. Update running max.
  float old_max = state.running_max;
  float new_max = fmaxf(old_max, tile_max);
  // 3. Rescale factor for O accumulator.
  float rescale = exp2f(old_max - new_max);
  // 4. Rescale running sum.
  state.running_sum *= rescale;
  // 5. Exponentiate tile values; P = exp2(scale*S - new_max).
  for (int k = 0; k < K_BLOCK; ++k) {
    float p = exp2f(tile_values[k] * scale_log2 - new_max);
    tile_values[k] = p;
    state.running_sum += p;
  }
  state.running_max = new_max;
  return rescale;
}

// =============================================================================
// Warp-collective reductions and finalize
// =============================================================================

__device__ __forceinline__
float warp_row_max(float val) { return warp_reduce_max_f32(val); }

__device__ __forceinline__
float warp_row_sum(float val) { return warp_reduce_sum_f32(val); }

__device__ __forceinline__
float softmax_finalize(float p, float running_sum) {
  return p / running_sum;
}
