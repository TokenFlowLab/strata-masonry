#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// 0_tcgen05_alloc.cuh -- tcgen05.alloc.cta_group::{1,2}.sync.aligned.shared::cta.b32
//
// ARCH: sm_100a
//
// Warp-collective TMEM column allocator. Writes the TMEM base pointer
// (bits [31:16] = lane index = 0; bits [15:0] = column index = 0) to the
// caller-provided SMEM slot. For CTA_GROUP == 2, one warp from each peer
// CTA must issue the same call with the same n_cols; the first may block
// until the peer issues.
//
// Constraints (PTX 9.7.18.7):
//   n_cols in [32, 512] and a power of two
//   successive allocs within the same CTA must be non-increasing in n_cols
//
// Issuer: one warp per CTA (or one warp per peer-CTA pair for cta_group::2).
// Source: knowledge/instructions/tmem/tcgen05_tmem.md
// PTX:    9.7.18.7.1 (tcgen05.alloc)
//
#include <cstdint>

template <int CTA_GROUP>
__device__ __forceinline__ void tcgen05_alloc(uint32_t smem_dst_ptr,
                                              uint32_t n_cols) {
  static_assert(CTA_GROUP == 1 || CTA_GROUP == 2,
                "tcgen05_alloc: CTA_GROUP must be 1 or 2");
  if constexpr (CTA_GROUP == 1) {
    asm volatile(
      "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
      :: "r"(smem_dst_ptr), "r"(n_cols));
  } else {
    asm volatile(
      "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;\n"
      :: "r"(smem_dst_ptr), "r"(n_cols));
  }
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
