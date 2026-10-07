#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// 19_tma_load_2sm.cuh -- cp.async.bulk.tensor.{2d,3d}.cta_group::2 (Blackwell 2SM)
//
// ARCH: sm_100a
//
// 2SM TMA load: one GMEM read lands in the CTA-pair's SMEM (both peer CTAs).
// The complete_tx signal goes to the mbarrier at the address given; to route
// it to a specific peer CTA (e.g. always CTA 0 of the pair), callers apply
// the peer-bit mask 0xFEFFFFFF to the mbar address.
// Source: knowledge/instructions/tma/tma_load.md
// PTX:    9.7.10.28.5.3 (cp.async.bulk.tensor.cta_group::2)
//
#include <cstdint>

__device__ __forceinline__ void tma_load_2d_2sm(
    uint32_t smem_dst, const void* tensormap_ptr,
    uint32_t mbar_smem_masked, int coord_x, int coord_y) {
  asm volatile(
    "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global"
    ".mbarrier::complete_tx::bytes"
    " [%0], [%1, {%3, %4}], [%2];\n"
    :: "r"(smem_dst), "l"(tensormap_ptr),
       "r"(mbar_smem_masked), "r"(coord_x), "r"(coord_y)
    : "memory");
}

// 3D analog of tma_load_2d_2sm. Coords {c0,c1,c2} = tensormap dims fastest->
// slowest; route completion to a chosen peer by masking the mbar address
// (tma_peer_bit_mask), same as the 2D form.
__device__ __forceinline__ void tma_load_3d_2sm(
    uint32_t smem_dst, const void* tensormap_ptr,
    uint32_t mbar_smem_masked, int c0, int c1, int c2) {
  asm volatile(
    "cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.tile"
    ".mbarrier::complete_tx::bytes"
    " [%0], [%1, {%3, %4, %5}], [%2];\n"
    :: "r"(smem_dst), "l"(tensormap_ptr),
       "r"(mbar_smem_masked), "r"(c0), "r"(c1), "r"(c2)
    : "memory");
}

// 4D analog of tma_load_3d_2sm. Coords {c0..c3} = tensormap dims fastest->slowest.
__device__ __forceinline__ void tma_load_4d_2sm(
    uint32_t smem_dst, const void* tensormap_ptr,
    uint32_t mbar_smem_masked, int c0, int c1, int c2, int c3) {
  asm volatile(
    "cp.async.bulk.tensor.4d.cta_group::2.shared::cluster.global.tile"
    ".mbarrier::complete_tx::bytes"
    " [%0], [%1, {%3, %4, %5, %6}], [%2];\n"
    :: "r"(smem_dst), "l"(tensormap_ptr),
       "r"(mbar_smem_masked), "r"(c0), "r"(c1), "r"(c2), "r"(c3)
    : "memory");
}

// 5D analog of tma_load_4d_2sm. Coords {c0..c4} = tensormap dims fastest->slowest.
__device__ __forceinline__ void tma_load_5d_2sm(
    uint32_t smem_dst, const void* tensormap_ptr,
    uint32_t mbar_smem_masked, int c0, int c1, int c2, int c3, int c4) {
  asm volatile(
    "cp.async.bulk.tensor.5d.cta_group::2.shared::cluster.global.tile"
    ".mbarrier::complete_tx::bytes"
    " [%0], [%1, {%3, %4, %5, %6, %7}], [%2];\n"
    :: "r"(smem_dst), "l"(tensormap_ptr),
       "r"(mbar_smem_masked), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4)
    : "memory");
}

// 2SM TMA load with .L2::cache_hint. cache_policy is a 64-bit eviction
// policy descriptor (see CUDA driver `cuMemcpy3DAsync` cache hints / inline
// PTX `createpolicy`). PTX 9.7.10.27 .cta_group::2 + .L2::cache_hint form.
__device__ __forceinline__ void tma_load_2d_2sm_l2hint(
    uint32_t smem_dst, const void* tensormap_ptr,
    uint32_t mbar_smem_masked, int coord_x, int coord_y,
    uint64_t cache_policy) {
  asm volatile(
    "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global"
    ".mbarrier::complete_tx::bytes.L2::cache_hint"
    " [%0], [%1, {%3, %4}], [%2], %5;\n"
    :: "r"(smem_dst), "l"(tensormap_ptr),
       "r"(mbar_smem_masked), "r"(coord_x), "r"(coord_y),
       "l"(cache_policy)
    : "memory");
}

// Convenient constant for the peer-bit mask (see tma_load.md Gotchas).
__device__ __forceinline__ uint32_t tma_peer_bit_mask(uint32_t mbar_smem) {
  return mbar_smem & 0xFEFFFFFFu;
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
