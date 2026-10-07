// 45_shfl_sync.cuh -- shfl.sync.{idx,up,down,bfly}.b32
//
// ARCH: sm_90a
//
// Warp shuffle primitives with explicit sync mask. Each lane donates one
// .b32 register; the destination lane is picked by the mode:
//   .idx   -- absolute lane index (broadcast / read from specific lane)
//   .up    -- source lane = self - delta (prefix-scan, clamped at 0)
//   .down  -- source lane = self + delta (suffix, clamped at 31)
//   .bfly  -- source lane = self XOR delta (butterfly reduction)

#pragma once

// Source: knowledge/instructions/warp/shfl_sync.md
// PTX:    9.7.10.6 (shfl.sync)
//
#include <cstdint>

__device__ __forceinline__
uint32_t shfl_sync_idx(uint32_t val, int src_lane,
                       uint32_t mask = 0xFFFFFFFFu) {
  uint32_t ret;
  asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1F, %3;\n"
               : "=r"(ret) : "r"(val), "r"(src_lane), "r"(mask));
  return ret;
}

__device__ __forceinline__
uint32_t shfl_sync_up(uint32_t val, int delta,
                      uint32_t mask = 0xFFFFFFFFu) {
  uint32_t ret;
  asm volatile("shfl.sync.up.b32 %0, %1, %2, 0, %3;\n"
               : "=r"(ret) : "r"(val), "r"(delta), "r"(mask));
  return ret;
}

__device__ __forceinline__
uint32_t shfl_sync_down(uint32_t val, int delta,
                        uint32_t mask = 0xFFFFFFFFu) {
  uint32_t ret;
  asm volatile("shfl.sync.down.b32 %0, %1, %2, 0x1F, %3;\n"
               : "=r"(ret) : "r"(val), "r"(delta), "r"(mask));
  return ret;
}

__device__ __forceinline__
uint32_t shfl_sync_bfly(uint32_t val, int lane_mask,
                        uint32_t mask = 0xFFFFFFFFu) {
  uint32_t ret;
  asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1F, %3;\n"
               : "=r"(ret) : "r"(val), "r"(lane_mask), "r"(mask));
  return ret;
}

// -- warp-wide reductions (built on butterfly shuffle) -----------------------

__device__ __forceinline__
float warp_reduce_sum_f32(float val) {
  uint32_t v = __float_as_uint(val);
  for (int delta = 16; delta >= 1; delta >>= 1) {
    uint32_t other = shfl_sync_bfly(v, delta);
    val += __uint_as_float(other);
    v = __float_as_uint(val);
  }
  return val;
}

__device__ __forceinline__
float warp_reduce_max_f32(float val) {
  uint32_t v = __float_as_uint(val);
  for (int delta = 16; delta >= 1; delta >>= 1) {
    uint32_t other = shfl_sync_bfly(v, delta);
    float other_f = __uint_as_float(other);
    val = fmaxf(val, other_f);
    v = __float_as_uint(val);
  }
  return val;
}
