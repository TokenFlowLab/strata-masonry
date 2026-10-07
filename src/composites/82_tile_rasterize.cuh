// 82_tile_rasterize.cuh -- linear tile_id -> 2D (tile_m, tile_n) coordinates
//
// ARCH: sm_90a
//
// Scope: this composite is for the LINEAR-INDEX scheduler family --
// blocks 95 (blockIdx) and 96 (persistent atomicAdd) -- where the
// scheduler picks a linear tile_id and then maps it to (m, n) via one
// of these helpers. Pick the mapping (col-major / row-major /
// swizzled / snake) to match the desired L2-reuse axis.
//
// **Not** for the CLC scheduler (block 97). CLC dispatches clusters in
// HW-determined order and returns the canceled cluster's (ctaid.x,
// ctaid.y) directly; there is no linear tile_id to remap. Any
// L2-locality optimization on the CLC path needs an axis-swap transform
// applied to the post-decoded (m_tile, n_tile) -- a different API
// (e.g. a `clc_rasterize` helper in composite 106 modeled on V56's
// `sched_swizzle_and_rasterize`).
//
// Column-major order (iterate M first) improves L2 locality for GEMM because
// A tile rows are reused across N-blocks within the same M-block.
//
// Two naming conventions for the basic raster:
//   tile_rasterize_colmajor   /  tile_rasterize_col_major
//   tile_rasterize_rowmajor   /  tile_rasterize_row_major
// Both pairs implement the same mapping (kept for caller-side stability).
//
// Swizzled rasters (CUTLASS "L2-aware raster" pattern, group of N
// consecutive N-tiles share the same M-block):
//   tile_rasterize_swizzled<RASTER_GROUP>          (no clamp)
//   tile_rasterize_swizzled_clamped<RASTER_GROUP>  (out_n clamped to
//                                                   tiles_n - 1)
//
// tile_rasterize_snake: boustrophedon (alternate M direction per N row).

#pragma once

// PTX:    n/a (tile-coord math; cite knowledge for col / row / swizzled raster)
//
#include <cstdint>

// =============================================================================
// Basic column-major / row-major (two name spellings)
// =============================================================================

__device__ __host__ __forceinline__
void tile_rasterize_colmajor(int tile_idx, int tiles_m, int tiles_n,
                             int& out_m, int& out_n) {
  out_n = tile_idx / tiles_m;
  out_m = tile_idx % tiles_m;
  (void)tiles_n;
}

__device__ __host__ __forceinline__
void tile_rasterize_col_major(int tile_id, int tiles_m, int tiles_n,
                              int& tile_m, int& tile_n) {
  tile_n = tile_id / tiles_m;
  tile_m = tile_id % tiles_m;
  (void)tiles_n;
}

__device__ __host__ __forceinline__
void tile_rasterize_rowmajor(int tile_idx, int tiles_m, int tiles_n,
                             int& out_m, int& out_n) {
  out_m = tile_idx / tiles_n;
  out_n = tile_idx % tiles_n;
  (void)tiles_m;
}

__device__ __host__ __forceinline__
void tile_rasterize_row_major(int tile_id, int tiles_m, int tiles_n,
                              int& tile_m, int& tile_n) {
  tile_m = tile_id / tiles_n;
  tile_n = tile_id % tiles_n;
  (void)tiles_m;
}

// =============================================================================
// Swizzled: group of RASTER_GROUP consecutive N-tiles share same M-block
// =============================================================================

// L2-aware raster (no clamp). Canonical version used by 95_sched_warp_blockidx.
template <int RASTER_GROUP>
__device__ __host__ __forceinline__
void tile_rasterize_swizzled(int tile_id, int tiles_m, int tiles_n,
                             int& tile_m, int& tile_n) {
  int tiles_per_group = tiles_m * RASTER_GROUP;
  int group_id        = tile_id / tiles_per_group;
  int tile_in_group   = tile_id % tiles_per_group;
  int col_in_group    = tile_in_group / tiles_m;
  tile_m = tile_in_group % tiles_m;
  tile_n = group_id * RASTER_GROUP + col_in_group;
  (void)tiles_n;
}

// Variant with out_n clamped to tiles_n - 1 (defensive when RASTER_GROUP
// doesn't divide tiles_n evenly).
template <int RASTER_GROUP>
__device__ __host__ __forceinline__
void tile_rasterize_swizzled_clamped(int tile_idx, int tiles_m, int tiles_n,
                                     int& out_m, int& out_n) {
  int block  = tile_idx / (tiles_m * RASTER_GROUP);
  int within = tile_idx % (tiles_m * RASTER_GROUP);
  out_m = within / RASTER_GROUP;
  out_n = block * RASTER_GROUP + (within % RASTER_GROUP);
  if (out_n >= tiles_n) out_n = tiles_n - 1;
}

// =============================================================================
// Snake (boustrophedon) raster
// =============================================================================

__device__ __host__ __forceinline__
void tile_rasterize_snake(int tile_id, int tiles_m, int tiles_n,
                          int& tile_m, int& tile_n) {
  tile_n = tile_id / tiles_m;
  int tile_m_raw = tile_id % tiles_m;
  // Reverse M on odd rows for better A-tile reuse.
  tile_m = (tile_n & 1) ? (tiles_m - 1 - tile_m_raw) : tile_m_raw;
  (void)tiles_n;
}
