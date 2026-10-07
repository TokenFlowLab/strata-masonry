#pragma once
// 95_sched_warp_blockidx.cuh -- non-persistent blockIdx-based tile scheduler.
//
// ARCH: arch-agnostic (compiles on sm_90a / sm_100a / sm_103a)
//
// Each CTA owns one tile, identified by tile_id. The "warp" role is just a
// per-tile rasterization step: given tile_id + (tiles_m, tiles_n) and
// thread role, produce (m, n) coordinates and (optionally) record hit/log
// metadata. No scheduler state, no atomicAdd of a counter, no persistent
// kernel.
//
// Block function (per code/PLAN.md "Block function signature contract"):
//
//   __device__ __forceinline__ void
//   sched_warp_blockidx_block(int tile_id, int tiles_m, int tiles_n,
//                             int& m, int& n);
//
// Per-CTA, per-tile: compute (m, n) via column-major raster. Caller is
// expected to gate by `threadIdx.x == 0` if a single-thread write is
// desired.
//
// Auxiliary block function (caller-driven all-pattern recorder, used by
// the runtime correctness test):
//
//   __device__ __forceinline__ void
//   sched_warp_blockidx_all_patterns_block(
//       int tile_id, int tiles_m, int tiles_n,
//       int* out_col, int* out_row, int* out_sw, int* out_snake,
//       uint32_t* hit_col, uint32_t* hit_row,
//       uint32_t* hit_sw,  uint32_t* hit_snake);
//
// Source: knowledge/building_blocks/sched_warp.md
// PTX:    n/a (built-in blockIdx; no PTX issue)

#include <cstdint>
#include "../composites/82_tile_rasterize.cuh"

__device__ __forceinline__
void sched_warp_blockidx_block(int tile_id, int tiles_m, int tiles_n,
                               int& m, int& n) {
  tile_rasterize_colmajor(tile_id, tiles_m, tiles_n, m, n);
}

__device__ __forceinline__
void sched_warp_blockidx_all_patterns_block(
    int tile_id, int tiles_m, int tiles_n,
    int* out_col, int* out_row, int* out_sw, int* out_snake,
    uint32_t* hit_col, uint32_t* hit_row,
    uint32_t* hit_sw,  uint32_t* hit_snake) {
  int m, n;

  tile_rasterize_col_major(tile_id, tiles_m, tiles_n, m, n);
  out_col[tile_id * 2 + 0] = m; out_col[tile_id * 2 + 1] = n;
  atomicAdd(&hit_col[n * tiles_m + m], 1u);

  tile_rasterize_row_major(tile_id, tiles_m, tiles_n, m, n);
  out_row[tile_id * 2 + 0] = m; out_row[tile_id * 2 + 1] = n;
  atomicAdd(&hit_row[n * tiles_m + m], 1u);

  tile_rasterize_swizzled<4>(tile_id, tiles_m, tiles_n, m, n);
  out_sw[tile_id * 2 + 0] = m; out_sw[tile_id * 2 + 1] = n;
  atomicAdd(&hit_sw[n * tiles_m + m], 1u);

  tile_rasterize_snake(tile_id, tiles_m, tiles_n, m, n);
  out_snake[tile_id * 2 + 0] = m; out_snake[tile_id * 2 + 1] = n;
  atomicAdd(&hit_snake[n * tiles_m + m], 1u);
}
