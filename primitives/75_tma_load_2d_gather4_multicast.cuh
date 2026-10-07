// 75_tma_load_2d_gather4_multicast.cuh -- cp.async.bulk.tensor.2d
//   .tile::gather4.multicast::cluster
//
// ARCH: sm_100a
//
// Multicast variant of #73 tma_load_2d_gather4. One thread issues the
// gather4; the HW multicasts the fetched 4 rows to the destination CTAs
// indicated by `ctamask` and signals each destination CTA's mbarrier.
// Use when both peers of a `cta_group::2` cluster need the same 4 rows
// (e.g., MoE FC1 gather where both peers read the same A token rows).
//
// Coord layout (per PTX 9.x):
//   {col_idx, row0_idx, row1_idx, row2_idx, row3_idx}
//
// ctamask: 16-bit, bit i = destination CTA whose %cluster_ctarank is i.
// For cta_group::2 (cluster size = 2), the standard mask covering both
// peers is 0b11 = 0x3.

#pragma once

// PTX:    9.7.10.28.5.3 (cp.async.bulk.tensor; .tile::gather4 + .multicast::cluster)
// CUTLASS: cute/arch/copy_sm100_tma.hpp:SM100_TMA_LOAD_MULTICAST_2D_GATHER4
//
#include <cuda.h>
#include <cstdint>

// Multicast gather4 (no L2 hint).
__device__ __forceinline__
void tma_load_2d_gather4_multicast(uint32_t smem_dst,
                                   const void* tensormap_ptr,
                                   uint32_t mbar_smem,
                                   uint16_t ctamask,
                                   int coord_col,
                                   int coord_row0, int coord_row1,
                                   int coord_row2, int coord_row3) {
  asm volatile(
    "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
    ".mbarrier::complete_tx::bytes.multicast::cluster"
    " [%0], [%1, {%4, %5, %6, %7, %8}], [%2], %3;\n"
    :: "r"(smem_dst), "l"(tensormap_ptr), "r"(mbar_smem),
       "h"(ctamask),
       "r"(coord_col),
       "r"(coord_row0), "r"(coord_row1),
       "r"(coord_row2), "r"(coord_row3)
    : "memory");
}

// Multicast gather4 with L2 cache hint.
__device__ __forceinline__
void tma_load_2d_gather4_multicast_l2hint(uint32_t smem_dst,
                                          const void* tensormap_ptr,
                                          uint32_t mbar_smem,
                                          uint16_t ctamask,
                                          int coord_col,
                                          int coord_row0, int coord_row1,
                                          int coord_row2, int coord_row3,
                                          uint64_t cache_policy) {
  asm volatile(
    "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
    ".mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
    " [%0], [%1, {%4, %5, %6, %7, %8}], [%2], %3, %9;\n"
    :: "r"(smem_dst), "l"(tensormap_ptr), "r"(mbar_smem),
       "h"(ctamask),
       "r"(coord_col),
       "r"(coord_row0), "r"(coord_row1),
       "r"(coord_row2), "r"(coord_row3),
       "l"(cache_policy)
    : "memory");
}
