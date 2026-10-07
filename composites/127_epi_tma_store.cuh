// 127_epi_tma_store.cuh -- bar.sync + fence.proxy.async + TMA store + commit/wait
//
// ARCH: sm_90a
//
// Epilogue store sequence: sync the epilogue warps, cross-proxy fence,
// single-thread TMA store, close the bulk group, wait for GMEM landing.
//
// Three flavors:
//   epi_tma_store_2d<BAR_ID>          full sequence (threadIdx.x == 0 elects).
//   epi_tma_store_2d_elected          just fence + store, caller elects.
//   epi_tma_store_full<BAR_ID>        full sequence with explicit threads /
//                                     is_elected (caller-driven election).

#pragma once

// Source: knowledge/building_blocks/epi_warp.md
// PTX:    9.7.10.28.5.3 (cp.async.bulk.tensor store), 9.7.10.28.6.1 (commit_group), 9.7.10.28.6.2 (wait_group)
//
#include <cstdint>
#include "../primitives/22_tma_store.cuh"
#include "../primitives/25_tma_async_group.cuh"
#include "../primitives/34_fence_proxy_async.cuh"
#include "../primitives/37_bar_sync.cuh"

// Full sequence, threadIdx.x == 0 elects.
template <int BARRIER_ID>
__device__ __forceinline__
void epi_tma_store_2d(uint32_t thread_count_for_bar,
                      const void* tensormap, int coord_x, int coord_y,
                      uint32_t smem_src) {
  bar_sync<BARRIER_ID>(thread_count_for_bar);
  if (threadIdx.x == 0) {
    fence_proxy_async_shared_cta();
    tma_store_2d(tensormap, coord_x, coord_y, smem_src);
    cp_async_bulk_commit_group();
    cp_async_bulk_wait_group<0>();
  }
}

// Just fence + store -- caller is the elected thread.
__device__ __forceinline__
void epi_tma_store_2d_elected(const void* tensormap_ptr,
                              int coord_x, int coord_y,
                              uint32_t smem_src) {
  fence_proxy_async_shared_cta();
  tma_store_2d(tensormap_ptr, coord_x, coord_y, smem_src);
}

// Full sequence with explicit thread count + caller-driven election.
template <int BAR_ID = 1>
__device__ __forceinline__
void epi_tma_store_full(const void* tensormap_ptr,
                        int coord_x, int coord_y,
                        uint32_t smem_src,
                        uint32_t threads_in_group,
                        bool is_elected) {
  bar_sync<BAR_ID>(threads_in_group);
  if (is_elected) {
    fence_proxy_async_shared_cta();
    tma_store_2d(tensormap_ptr, coord_x, coord_y, smem_src);
    tma_store_commit_group();
    tma_store_wait_group<0>();
  }
}
