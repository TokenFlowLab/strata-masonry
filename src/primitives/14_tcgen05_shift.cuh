#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// 14_tcgen05_shift.cuh -- tcgen05.shift.cta_group::{1,2}.down [taddr]
//
// ARCH: sm_100a
//
// Async: shifts the 32-byte elements of the warp's TMEM matrix at taddr down
// one row, across all rows except the last. Moves data only; no scaling.
//
// Single-thread issued; cta_group::2 shifts this and the peer CTA's TMEM.
// taddr lane must be 32-aligned. Completion via tcgen05.commit + mbarrier.
// PTX:    9.7.18.9.3 (tcgen05.shift)
//
#include <cstdint>

template <int CTA_GROUP>
__device__ __forceinline__ void tcgen05_shift(uint32_t tmem_addr) {
  static_assert(CTA_GROUP == 1 || CTA_GROUP == 2,
                "tcgen05_shift: CTA_GROUP must be 1 or 2");
  if constexpr (CTA_GROUP == 1) {
    asm volatile("tcgen05.shift.cta_group::1.down [%0];\n"
                 :: "r"(tmem_addr));
  } else {
    asm volatile("tcgen05.shift.cta_group::2.down [%0];\n"
                 :: "r"(tmem_addr));
  }
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
