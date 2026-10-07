// 29_mbarrier_init.cuh -- mbarrier.init / mbarrier.inval
//
// ARCH: sm_90a
//
// Initialize or invalidate an 8-byte, 8-byte-aligned mbarrier object in SMEM.
// Every mbarrier must be initialized before use and invalidated before the
// kernel exits. After init, follow with fence.mbarrier_init.release.cluster
// (#35) if the mbarrier must be visible to other CTAs in the cluster.

#pragma once

// PTX:    9.7.15.16.12 (mbarrier.init), 9.7.15.16.13 (mbarrier.inval)
//
#include <cstdint>

// Initialize mbarrier: set arrival count, reset phase to 0.
__device__ __forceinline__
void mbarrier_init(uint32_t mbar_smem, uint32_t arrive_count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
               :: "r"(mbar_smem), "r"(arrive_count) : "memory");
}

// Invalidate mbarrier; SMEM is reusable.
// Must be called before kernel exit for every initialized mbarrier.
__device__ __forceinline__
void mbarrier_inval(uint32_t mbar_smem) {
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n"
               :: "r"(mbar_smem) : "memory");
}
