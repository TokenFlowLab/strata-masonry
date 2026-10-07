#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// 1_tcgen05_dealloc.cuh -- tcgen05.dealloc.cta_group::{1,2}.sync.aligned.b32
//
// ARCH: sm_100a
//
// Warp-collective TMEM column deallocator. Must be called before kernel
// exit for every prior tcgen05.alloc in the CTA. For CTA_GROUP == 2 both
// peer CTAs must synchronize TMEM accesses first (e.g. barrier.cluster)
// and neither may exit before the pair completes.
//
// Takes the base address previously returned by tcgen05_alloc and the
// same n_cols value that was allocated.
// Source: knowledge/instructions/tmem/tcgen05_tmem.md
// PTX:    9.7.18.7.1 (tcgen05.dealloc)
//
#include <cstdint>

template <int CTA_GROUP>
__device__ __forceinline__ void tcgen05_dealloc(uint32_t tmem_addr,
                                                uint32_t n_cols) {
  static_assert(CTA_GROUP == 1 || CTA_GROUP == 2,
                "tcgen05_dealloc: CTA_GROUP must be 1 or 2");
  if constexpr (CTA_GROUP == 1) {
    asm volatile(
      "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
      :: "r"(tmem_addr), "r"(n_cols));
  } else {
    asm volatile(
      "tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;\n"
      :: "r"(tmem_addr), "r"(n_cols));
  }
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
