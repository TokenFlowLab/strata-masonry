#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// 2_tcgen05_relinquish.cuh -- tcgen05.relinquish_alloc_permit.cta_group::{1,2}.sync.aligned
//
// ARCH: sm_100a
//
// Release the CTA's TMEM allocation permit so other CTAs / grids can
// begin rasterizing onto the same SM. Typically issued immediately after
// the last tcgen05.alloc, before the mainloop begins.
//
// No operands. After this call the CTA MUST NOT issue tcgen05.alloc again.
//
// PTX:    9.7.18.7.1 (tcgen05.alloc / dealloc / relinquish_alloc_permit)
//
template <int CTA_GROUP>
__device__ __forceinline__ void tcgen05_relinquish_alloc_permit() {
  static_assert(CTA_GROUP == 1 || CTA_GROUP == 2,
                "tcgen05_relinquish_alloc_permit: CTA_GROUP must be 1 or 2");
  if constexpr (CTA_GROUP == 1) {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::);
  } else {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;\n" ::);
  }
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
