// 20_tma_load_multicast.cuh -- cp.async.bulk.tensor.2d ... .multicast::cluster
//
// ARCH: sm_90a
//
// Multicast TMA load: one GMEM read broadcasts to multiple CTAs in the
// cluster. ctamask is a 16-bit bitmask (bit i = destination CTA rank i).
// complete_tx arrives on each destination CTA's mbar at the same SMEM offset.
// Requires .shared::cluster destination (not .shared::cta).
//
// For 2SM + multicast, compose with #19 (add .cta_group::2 before multicast);
// kept as separate primitives for clarity.
//
// Two argument-ordering conventions are provided as overloads:
//   tma_load_2d_multicast(smem, tmap, mbar, x, y, mask)         smem-first
//   tma_load_2d_multicast(tmap, smem, mbar, x, y, mask)         tmap-first

#pragma once

// PTX:    9.7.10.28.5.3 (cp.async.bulk.tensor + .multicast::cluster)
//
#include <cuda.h>
#include <cstdint>

// -- smem-first convention ---------------------------------------------------

__device__ __forceinline__
void tma_load_2d_multicast(uint32_t smem_dst, const void* tensormap_ptr,
                           uint32_t mbar_smem,
                           int coord_x, int coord_y, uint16_t ctamask) {
  asm volatile(
    "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
    ".mbarrier::complete_tx::bytes.multicast::cluster"
    " [%0], [%1, {%3, %4}], [%2], %5;\n"
    :: "r"(smem_dst), "l"(tensormap_ptr),
       "r"(mbar_smem), "r"(coord_x), "r"(coord_y),
       "h"(ctamask)
    : "memory");
}

// -- tmap-first convention ---------------------------------------------------

__device__ __forceinline__
void tma_load_2d_multicast(void const* tensormap_ptr, uint32_t smem_dst,
                           uint32_t mbar_smem_addr,
                           int32_t coord_x, int32_t coord_y,
                           uint16_t cta_mask) {
  asm volatile(
    "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
    ".mbarrier::complete_tx::bytes.multicast::cluster"
    " [%0], [%1, {%3, %4}], [%2], %5;\n"
    :: "r"(smem_dst), "l"(tensormap_ptr),
       "r"(mbar_smem_addr),
       "r"(coord_x), "r"(coord_y),
       "h"(cta_mask)
    : "memory");
}

// Multicast variant with L2 cache hint.
__device__ __forceinline__
void tma_load_2d_multicast_l2hint(void const* tensormap_ptr,
                                  uint32_t smem_dst,
                                  uint32_t mbar_smem_addr,
                                  int32_t coord_x, int32_t coord_y,
                                  uint16_t cta_mask,
                                  uint64_t cache_policy) {
  asm volatile(
    "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
    ".mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
    " [%0], [%1, {%3, %4}], [%2], %5, %6;\n"
    :: "r"(smem_dst), "l"(tensormap_ptr),
       "r"(mbar_smem_addr),
       "r"(coord_x), "r"(coord_y),
       "h"(cta_mask),
       "l"(cache_policy)
    : "memory");
}

// -- 2SM + multicast composition (Blackwell only) ----------------------------
// `cta_group::2` (peer-bit on mbar address, see #19) coexists with
// `multicast::cluster` (16-bit ctamask). One GMEM read populates each
// peer-CTA-pair listed in ctamask at the same SMEM offset. Used by GB300
// 2SM grouped GEMM where the producer fans out to all consumer pairs.

#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)

__device__ __forceinline__
void tma_load_2d_2sm_multicast(void const* tensormap_ptr,
                               uint32_t smem_dst,
                               uint32_t mbar_smem_masked,
                               int32_t coord_x, int32_t coord_y,
                               uint16_t cta_mask) {
  asm volatile(
    "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile"
    ".mbarrier::complete_tx::bytes.multicast::cluster"
    " [%0], [%1, {%3, %4}], [%2], %5;\n"
    :: "r"(smem_dst), "l"(tensormap_ptr),
       "r"(mbar_smem_masked),
       "r"(coord_x), "r"(coord_y),
       "h"(cta_mask)
    : "memory");
}

// 2SM + multicast + L2 cache hint.
__device__ __forceinline__
void tma_load_2d_2sm_multicast_l2hint(void const* tensormap_ptr,
                                      uint32_t smem_dst,
                                      uint32_t mbar_smem_masked,
                                      int32_t coord_x, int32_t coord_y,
                                      uint16_t cta_mask,
                                      uint64_t cache_policy) {
  asm volatile(
    "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile"
    ".mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
    " [%0], [%1, {%3, %4}], [%2], %5, %6;\n"
    :: "r"(smem_dst), "l"(tensormap_ptr),
       "r"(mbar_smem_masked),
       "r"(coord_x), "r"(coord_y),
       "h"(cta_mask),
       "l"(cache_policy)
    : "memory");
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
