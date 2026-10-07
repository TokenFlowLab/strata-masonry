// 22_tma_store.cuh -- cp.async.bulk.tensor.{2d,3d}.global.shared::cta.bulk_group
//
// ARCH: sm_90a
//
// One thread commits an SMEM tile to GMEM through the same tensormap
// descriptor used for TMA loads. Completion is tracked via the bulk
// async-group mechanism (cp.async.bulk.commit_group / wait_group, see #25).
// No 2SM variant exists for TMA stores.
//
// Caller must issue fence.proxy.async.shared::cta (#34) before this if the
// SMEM was written by generic-proxy instructions (stmatrix, st.shared).

#pragma once

// PTX:    9.7.10.28.5.3 (cp.async.bulk.tensor store)
//
#include <cuda.h>
#include <cstdint>

// Flat (non-tensor-map) FP32 bulk reduction. This is the exact primitive
// used by FA4's nondeterministic dQ drain: GMEM += contiguous SMEM. The
// operation participates in the cp.async bulk-group commit/wait protocol.
__device__ __forceinline__
void cpasync_reduce_bulk_add_f32(float* global_dst, uint32_t smem_src,
                                uint32_t bytes) {
  asm volatile(
      "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
      " [%0], [%1], %2;\n"
      :: "l"(global_dst), "r"(smem_src), "r"(bytes)
      : "memory");
}

// f16 variant (.noftz: denormals kept). The destination holds IEEE half.
__device__ __forceinline__
void cpasync_reduce_bulk_add_f16(uint16_t* global_dst, uint32_t smem_src,
                                uint32_t bytes) {
  asm volatile(
      "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.f16"
      " [%0], [%1], %2;\n"
      :: "l"(global_dst), "r"(smem_src), "r"(bytes)
      : "memory");
}

// Same reductions with an L2 cache-policy hint (createpolicy, primitive #68).
__device__ __forceinline__
void cpasync_reduce_bulk_add_f32_l2hint(float* global_dst, uint32_t smem_src,
                                       uint32_t bytes, uint64_t cache_policy) {
  asm volatile(
      "cp.reduce.async.bulk.global.shared::cta.bulk_group.L2::cache_hint.add.f32"
      " [%0], [%1], %2, %3;\n"
      :: "l"(global_dst), "r"(smem_src), "r"(bytes), "l"(cache_policy)
      : "memory");
}

__device__ __forceinline__
void cpasync_reduce_bulk_add_f16_l2hint(uint16_t* global_dst, uint32_t smem_src,
                                       uint32_t bytes, uint64_t cache_policy) {
  asm volatile(
      "cp.reduce.async.bulk.global.shared::cta.bulk_group.L2::cache_hint.add.noftz.f16"
      " [%0], [%1], %2, %3;\n"
      :: "l"(global_dst), "r"(smem_src), "r"(bytes), "l"(cache_policy)
      : "memory");
}

__device__ __forceinline__
void tma_store_2d(const void* tensormap_ptr, int coord_x, int coord_y,
                  uint32_t smem_src) {
  asm volatile(
    "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
    " [%0, {%1, %2}], [%3];\n"
    :: "l"(tensormap_ptr), "r"(coord_x), "r"(coord_y),
       "r"(smem_src)
    : "memory");
}

// 3D analog of tma_store_2d. Coords {c0, c1, c2} index the tensormap's dims
// fastest (c0) to slowest (c2); box extents come from the tensormap. Same fence
// requirement as the header note.
__device__ __forceinline__
void tma_store_3d(const void* tensormap_ptr, int c0, int c1, int c2,
                  uint32_t smem_src) {
  asm volatile(
    "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
    " [%0, {%1, %2, %3}], [%4];\n"
    :: "l"(tensormap_ptr), "r"(c0), "r"(c1), "r"(c2), "r"(smem_src)
    : "memory");
}

// 4D analog of tma_store_3d. Coords {c0..c3} fastest (c0) to slowest (c3); box
// extents come from the tensormap. Same fence requirement as the header note.
__device__ __forceinline__
void tma_store_4d(const void* tensormap_ptr, int c0, int c1, int c2, int c3,
                  uint32_t smem_src) {
  asm volatile(
    "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
    " [%0, {%1, %2, %3, %4}], [%5];\n"
    :: "l"(tensormap_ptr), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(smem_src)
    : "memory");
}

// Store with L2 cache hint.
__device__ __forceinline__
void tma_store_2d_l2hint(const void* tensormap_ptr, int coord_x, int coord_y,
                         uint32_t smem_src, uint64_t cache_policy) {
  asm volatile(
    "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group.L2::cache_hint"
    " [%0, {%1, %2}], [%3], %4;\n"
    :: "l"(tensormap_ptr), "r"(coord_x), "r"(coord_y),
       "r"(smem_src), "l"(cache_policy)
    : "memory");
}

// ---------------------------------------------------------------------------
// TMA reduction stores. PTX 9.7.10.28 `cp.reduce.async.bulk.tensor` family.
// Used by split-K accumulating epilogues: each tile-K partition issues a
// reduction store to the same global tile so the hardware adds/mins/maxes
// the partial result into the existing GMEM tile. Completion tracked via
// the same bulk async-group mechanism (#25).
// ---------------------------------------------------------------------------

// Reduction store: GMEM[tile] = GMEM[tile] + SMEM[smem_src]  (per-element).
__device__ __forceinline__
void tma_store_2d_add(const void* tensormap_ptr, int coord_x, int coord_y,
                      uint32_t smem_src) {
  asm volatile(
    "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group"
    " [%0, {%1, %2}], [%3];\n"
    :: "l"(tensormap_ptr), "r"(coord_x), "r"(coord_y),
       "r"(smem_src)
    : "memory");
}

// Reduction store: GMEM[tile] = min(GMEM[tile], SMEM[smem_src])  (per-element).
__device__ __forceinline__
void tma_store_2d_min(const void* tensormap_ptr, int coord_x, int coord_y,
                      uint32_t smem_src) {
  asm volatile(
    "cp.reduce.async.bulk.tensor.2d.global.shared::cta.min.tile.bulk_group"
    " [%0, {%1, %2}], [%3];\n"
    :: "l"(tensormap_ptr), "r"(coord_x), "r"(coord_y),
       "r"(smem_src)
    : "memory");
}

// Reduction store: GMEM[tile] = max(GMEM[tile], SMEM[smem_src])  (per-element).
__device__ __forceinline__
void tma_store_2d_max(const void* tensormap_ptr, int coord_x, int coord_y,
                      uint32_t smem_src) {
  asm volatile(
    "cp.reduce.async.bulk.tensor.2d.global.shared::cta.max.tile.bulk_group"
    " [%0, {%1, %2}], [%3];\n"
    :: "l"(tensormap_ptr), "r"(coord_x), "r"(coord_y),
       "r"(smem_src)
    : "memory");
}
