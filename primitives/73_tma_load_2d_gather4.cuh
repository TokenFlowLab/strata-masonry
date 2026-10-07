// 73_tma_load_2d_gather4.cuh -- cp.async.bulk.tensor.2d.tile::gather4
//
// ARCH: sm_100a
//
// Asynchronous 2D TMA load that gathers 4 NON-CONTIGUOUS rows of a tensor
// into a single SMEM tile, using a HW-level gather. One TMA call fetches
// 4 rows whose indices are passed independently (1 col-index + 4 row-
// indices). Issued by ONE thread; completion is signaled via mbarrier
// tx-count (same protocol as #18 tma_load_2d).
//
// Before calling: the issuing thread must register the expected byte count
// via mbarrier.arrive.expect_tx (#31). expect_tx must cover all 4 rows.
//
// Coord layout (per PTX 9.x):
//   {col_idx, row0_idx, row1_idx, row2_idx, row3_idx}
// col_idx is the fast dim (inner / column) base; the 4 row indices are
// independent positions along the slow dim.
//
// Destination scope is .shared::cluster (matches #18 cluster form and
// CUTLASS SM100_TMA_LOAD_2D_GATHER4).

#pragma once

// PTX:    9.7.10.28.5.3 (cp.async.bulk.tensor; tile::gather4 modifier)
// CUTLASS: cute/arch/copy_sm100_tma.hpp:SM100_TMA_LOAD_2D_GATHER4
//
#include <cuda.h>
#include <cstdint>

// Basic gather4 load (no L2 hint).
__device__ __forceinline__
void tma_load_2d_gather4(uint32_t smem_dst, const void* tensormap_ptr,
                         uint32_t mbar_smem,
                         int coord_col,
                         int coord_row0, int coord_row1,
                         int coord_row2, int coord_row3) {
  asm volatile(
    "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
    ".mbarrier::complete_tx::bytes"
    " [%0], [%1, {%3, %4, %5, %6, %7}], [%2];\n"
    :: "r"(smem_dst), "l"(tensormap_ptr), "r"(mbar_smem),
       "r"(coord_col),
       "r"(coord_row0), "r"(coord_row1),
       "r"(coord_row2), "r"(coord_row3)
    : "memory");
}

// L2-cache-hint variant.
__device__ __forceinline__
void tma_load_2d_gather4_l2hint(uint32_t smem_dst, const void* tensormap_ptr,
                                uint32_t mbar_smem,
                                int coord_col,
                                int coord_row0, int coord_row1,
                                int coord_row2, int coord_row3,
                                uint64_t cache_policy) {
  asm volatile(
    "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
    ".mbarrier::complete_tx::bytes.L2::cache_hint"
    " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], %8;\n"
    :: "r"(smem_dst), "l"(tensormap_ptr), "r"(mbar_smem),
       "r"(coord_col),
       "r"(coord_row0), "r"(coord_row1),
       "r"(coord_row2), "r"(coord_row3),
       "l"(cache_policy)
    : "memory");
}
