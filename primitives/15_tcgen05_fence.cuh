#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// 15_tcgen05_fence.cuh -- tcgen05.fence::before_thread_sync / ::after_thread_sync
//
// ARCH: sm_100a
//
// Orders tcgen05 ops with respect to thread-sync boundaries. Place
// ::before_thread_sync before a bar.sync to make sure prior tcgen05 ops
// are visible across the warps that sync, and ::after_thread_sync after the
// bar.sync to make subsequent tcgen05 ops wait for the sync.
//
// These fences are single-thread (issuer-local) and do NOT require
// .cta_group -- they are independent of 1SM vs 2SM.
//
// Source: knowledge/instructions/tmem/tcgen05_tmem.md
// PTX:    9.7.18.11.1 (tcgen05.fence)
//
__device__ __forceinline__ void tcgen05_fence_before_thread_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");  // memory clobber: without it the compiler can detach/drop the fence from the surrounding stores (P-publish race, 2026-07-14)
}

__device__ __forceinline__ void tcgen05_fence_after_thread_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
