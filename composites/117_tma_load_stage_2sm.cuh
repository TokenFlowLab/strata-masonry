#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// 117_tma_load_stage_2sm.cuh -- stage load using cta_group::2 TMA
//
// ARCH: sm_100a
//
// Same shape as #67 but issues the 2SM variant of TMA load. The mbar is
// steered to one CTA of the pair via the peer-bit mask (0xFEFFFFFF).
// PTX:    9.7.10.28.5.3 (cta_group::2), 9.7.15.16.14 (expect_tx + peer-bit mask)
//
#include <cstdint>
#include "../primitives/19_tma_load_2sm.cuh"
#include "../primitives/31_mbarrier_arrive_tx.cuh"

__device__ __forceinline__ void tma_load_stage_2sm_1tensor(
    uint32_t mbar_smem, uint32_t expected_bytes,
    uint32_t smem_dst, const void* tensormap, int x, int y) {
  mbarrier_arrive_expect_tx(mbar_smem, expected_bytes);
  uint32_t mbar_masked = tma_peer_bit_mask(mbar_smem);
  tma_load_2d_2sm(smem_dst, tensormap, mbar_masked, x, y);
}

__device__ __forceinline__ void tma_load_stage_2sm_2tensor(
    uint32_t mbar_smem, uint32_t expected_bytes_total,
    uint32_t smem_dst_a, const void* tm_a, int xa, int ya,
    uint32_t smem_dst_b, const void* tm_b, int xb, int yb) {
  mbarrier_arrive_expect_tx(mbar_smem, expected_bytes_total);
  uint32_t mbar_masked = tma_peer_bit_mask(mbar_smem);
  tma_load_2d_2sm(smem_dst_a, tm_a, mbar_masked, xa, ya);
  tma_load_2d_2sm(smem_dst_b, tm_b, mbar_masked, xb, yb);
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
