// 21_tma_load_prefetch.cuh -- cp.async.bulk.prefetch.tensor.2d.L2.global.tile
//
// ARCH: sm_90a
//
// Issues an L2 prefetch hint for a 2D tile described by a tensormap. No SMEM
// destination, no mbarrier -- purely advisory. One thread issues it. Typical
// use: before the first mainloop iteration, prefetch the next tile to hide
// DRAM latency.

#pragma once

// PTX:    9.7.10.28.5.5 (cp.async.bulk.prefetch.tensor)
//
#include <cuda.h>
#include <cstdint>

__device__ __forceinline__
void tma_prefetch_2d(const void* tensormap_ptr,
                     int coord_x, int coord_y) {
  asm volatile(
    "cp.async.bulk.prefetch.tensor.2d.L2.global.tile"
    " [%0, {%1, %2}];\n"
    :: "l"(tensormap_ptr), "r"(coord_x), "r"(coord_y)
    : "memory");
}

// Prefetch with L2 cache hint (eviction policy).
__device__ __forceinline__
void tma_prefetch_2d_l2hint(const void* tensormap_ptr,
                            int coord_x, int coord_y,
                            uint64_t cache_policy) {
  asm volatile(
    "cp.async.bulk.prefetch.tensor.2d.L2.global.tile.L2::cache_hint"
    " [%0, {%1, %2}], %3;\n"
    :: "l"(tensormap_ptr), "r"(coord_x), "r"(coord_y),
       "l"(cache_policy)
    : "memory");
}

// prefetch.tensormap [a]
//   Prefetch the 128 B tensormap itself. Typically issued for every descriptor at kernel
//   entry, unguarded, before any warp specialization.
// PTX: 9.7.10.29 (prefetch.tensormap)
__device__ __forceinline__
void prefetch_tensormap(const void* tensormap_ptr) {
  asm volatile("prefetch.tensormap [%0];\n" :: "l"(tensormap_ptr) : "memory");
}
