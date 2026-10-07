#pragma once
// 96_sched_warp_persistent.cuh -- persistent atomicAdd tile scheduler.
//
// ARCH: arch-agnostic (compiles on sm_90a / sm_100a / sm_103a)
//
// Persistent kernel block: each CTA stays resident, fetches tiles via an
// atomic counter, and continues until the counter exceeds total_tiles.
// Used by Hopper warp-specialized GEMMs (pre-CLC) and as a fallback on
// Blackwell. On Blackwell, prefer block #97 (CLC try_cancel).
//
// Block functions (per code/PLAN.md "Block function signature contract"):
//
//   __device__ __forceinline__ void
//   sched_warp_persistent_block(int total_tiles, int tiles_m, int tiles_n,
//                               uint32_t* counter, int* tile_log);
//
// Compact form: per-CTA loop, thread 0 atomic-adds the counter, rasterizes
// (col-major), encodes (blockIdx, m, n) into tile_log[tile_id]. Block is
// CTA-collective (uses __syncthreads + a local __shared__ scratch).
//
//   __device__ __forceinline__ void
//   sched_warp_persistent_all_block(uint32_t* counter,
//                                   int total_tiles, int tiles_m, int tiles_n,
//                                   int* out_tile_m, int* out_tile_n,
//                                   int* out_owner_cta, uint32_t* hit_grid,
//                                   uint32_t* claim_count);
//
// Comprehensive form: per-CTA loop on thread 0; rasterizes via swizzled<4>
// and records (m, n) + owner CTA + hit grid + claim count for each
// claimed tile. Caller is responsible for early-exit if non-thread-0 lanes
// shouldn't participate; this block self-gates on threadIdx.x == 0.
//
// Composes primitive 50 (atom_global_add_u32) + composite 82 (rasterize_*).
// Source: knowledge/building_blocks/sched_warp.md
// PTX:    9.7.15.5 (atom.global.add for persistent counter)

#include <cstdint>
#include "../primitives/50_atom_global.cuh"
#include "../composites/82_tile_rasterize.cuh"

__device__ __forceinline__
void sched_warp_persistent_block(int total_tiles, int tiles_m, int tiles_n,
                                 uint32_t* counter, int* tile_log) {
  __shared__ uint32_t my_tile;
  while (true) {
    if (threadIdx.x == 0) my_tile = atom_global_add_u32(counter, 1u);
    __syncthreads();
    uint32_t t = my_tile;
    if (t >= (uint32_t)total_tiles) break;
    int m, n;
    tile_rasterize_colmajor((int)t, tiles_m, tiles_n, m, n);
    if (threadIdx.x == 0) tile_log[t] = (blockIdx.x << 16) | (m << 8) | n;
    __syncthreads();
  }
}

__device__ __forceinline__
void sched_warp_persistent_all_block(
    uint32_t* tile_counter,
    int total_tiles, int tiles_m, int tiles_n,
    int* out_tile_m, int* out_tile_n, int* out_owner_cta,
    uint32_t* hit_grid, uint32_t* claim_count) {
  if (threadIdx.x != 0) return;
  int cta = blockIdx.x;
  uint32_t local_claims = 0;
  while (true) {
    uint32_t tid = atom_global_add_u32(tile_counter, 1u);
    if (tid >= (uint32_t)total_tiles) break;
    int tm, tn;
    tile_rasterize_swizzled<4>((int)tid, tiles_m, tiles_n, tm, tn);
    out_tile_m[tid]    = tm;
    out_tile_n[tid]    = tn;
    out_owner_cta[tid] = cta;
    atomicAdd(&hit_grid[tn * tiles_m + tm], 1u);
    local_claims++;
  }
  claim_count[cta] = local_claims;
}
