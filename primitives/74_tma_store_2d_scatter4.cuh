// 74_tma_store_2d_scatter4.cuh -- cp.async.bulk.tensor.2d.tile::scatter4
//
// ARCH: sm_100a
//
// Asynchronous 2D TMA store that scatters one SMEM tile to 4 NON-CONTIGUOUS
// rows of a GMEM tensor, using a HW-level scatter. Companion to #73
// tma_load_2d_gather4. Issued by ONE thread; completion tracked via the
// bulk async-group mechanism (cp.async.bulk.commit_group / wait_group, see
// #25).
//
// Caller must issue fence.proxy.async.shared::cta (#34) before this if the
// SMEM tile was written by generic-proxy instructions (stmatrix, st.shared).
//
// Coord layout (per PTX 9.x):
//   {col_idx, row0_idx, row1_idx, row2_idx, row3_idx}
// col_idx is the fast dim (inner / column) base; the 4 row indices are
// independent destination positions along the slow dim.

#pragma once

// PTX:    9.7.10.28.5.3 (cp.async.bulk.tensor store; tile::scatter4 modifier)
// CUTLASS: cute/arch/copy_sm100_tma.hpp:SM100_TMA_STORE_2D_SCATTER4
//
#include <cuda.h>
#include <cstdint>

// Basic scatter4 store (no L2 hint).
__device__ __forceinline__
void tma_store_2d_scatter4(const void* tensormap_ptr,
                           int coord_col,
                           int coord_row0, int coord_row1,
                           int coord_row2, int coord_row3,
                           uint32_t smem_src) {
  asm volatile(
    "cp.async.bulk.tensor.2d.global.shared::cta.tile::scatter4.bulk_group"
    " [%0, {%1, %2, %3, %4, %5}], [%6];\n"
    :: "l"(tensormap_ptr),
       "r"(coord_col),
       "r"(coord_row0), "r"(coord_row1),
       "r"(coord_row2), "r"(coord_row3),
       "r"(smem_src)
    : "memory");
}

// L2-cache-hint variant.
__device__ __forceinline__
void tma_store_2d_scatter4_l2hint(const void* tensormap_ptr,
                                  int coord_col,
                                  int coord_row0, int coord_row1,
                                  int coord_row2, int coord_row3,
                                  uint32_t smem_src,
                                  uint64_t cache_policy) {
  asm volatile(
    "cp.async.bulk.tensor.2d.global.shared::cta.tile::scatter4"
    ".bulk_group.L2::cache_hint"
    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;\n"
    :: "l"(tensormap_ptr),
       "r"(coord_col),
       "r"(coord_row0), "r"(coord_row1),
       "r"(coord_row2), "r"(coord_row3),
       "r"(smem_src),
       "l"(cache_policy)
    : "memory");
}
