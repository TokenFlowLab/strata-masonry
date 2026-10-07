// 83_grouped_gemm_tile_map.cuh -- tile_id -> (group_id, tile_m, tile_n)
//
// ARCH: sm_90a
//
// For MoE / grouped GEMM: a prefix-sum array of cumulative tile counts lets
// us binary-search which group a global tile_id belongs to. Two API styles:
//
//   - struct-returning form: assumes per-group tiles_m AND tiles_n (full
//     freedom for both dims to vary across experts).
//
//   - reference-out form: assumes per-group tiles_m only, shared tiles_n
//     across groups (common for "shared N,K, varying M" MoE).

#pragma once

// PTX:    n/a (prefix-sum search math; tensormap selection per group)
//
#include <cstdint>

// =============================================================================
// Struct-returning form (per-group tiles_m AND tiles_n)
// =============================================================================

struct GroupedTileCoord {
  int group;
  int tile_m;
  int tile_n;
};

// offsets: prefix-sum of tile counts, with offsets[0] = 0 and
// offsets[NUM_GROUPS] = total_tiles. tiles_m_per_g / tiles_n_per_g give the
// M x N layout of each group.
__device__ __forceinline__
GroupedTileCoord grouped_tile_map(int tile_idx,
                                  const int* offsets,         // [num_groups+1]
                                  const int* tiles_m_per_g,   // [num_groups]
                                  const int* tiles_n_per_g,   // [num_groups]
                                  int num_groups) {
  // Binary search.
  int lo = 0, hi = num_groups - 1;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (offsets[mid + 1] <= tile_idx) lo = mid + 1;
    else                              hi = mid;
  }
  int g = lo;
  int within = tile_idx - offsets[g];
  GroupedTileCoord c;
  c.group  = g;
  c.tile_m = within / tiles_n_per_g[g];
  c.tile_n = within % tiles_n_per_g[g];
  (void)tiles_m_per_g;
  return c;
}

// =============================================================================
// Reference-out form (per-group tiles_m, shared tiles_n)
// =============================================================================

// Binary search: which group does tile_id belong to?
// group_cumul_tiles[g] = total tile count for groups [0..g] (inclusive).
__device__ __forceinline__
int find_group_id(int tile_id, const int* group_cumul_tiles, int num_groups) {
  int lo = 0, hi = num_groups - 1;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (group_cumul_tiles[mid] <= tile_id) lo = mid + 1;
    else                                   hi = mid;
  }
  return lo;
}

// Full grouped GEMM tile map: tile_id -> (group_id, tile_m, tile_n).
// Each group has its own tiles_m (from group_tiles_m array); tiles_n is shared.
__device__ __forceinline__
void grouped_gemm_tile_map(int tile_id,
                           const int* group_cumul_tiles,
                           const int* group_tiles_m,
                           int tiles_n,
                           int num_groups,
                           int& group_id, int& tile_m, int& tile_n) {
  group_id = find_group_id(tile_id, group_cumul_tiles, num_groups);
  int group_start    = (group_id > 0) ? group_cumul_tiles[group_id - 1] : 0;
  int tile_in_group  = tile_id - group_start;
  int group_tm       = group_tiles_m[group_id];
  // Column-major within group.
  tile_n = tile_in_group / group_tm;
  tile_m = tile_in_group % group_tm;
  (void)tiles_n;
}

// Build a prefix-sum array of cumulative per-group tile counts.
__device__ __forceinline__
void compute_group_cumul_tiles(const int* group_tiles, int num_groups,
                               int* group_cumul_tiles_out) {
  int sum = 0;
  for (int g = 0; g < num_groups; ++g) {
    sum += group_tiles[g];
    group_cumul_tiles_out[g] = sum;
  }
}
