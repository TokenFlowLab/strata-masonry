#pragma once
#if defined(PL_AGENTIC_SM90A)
// 59_wgmma_fence_commit_wait.cuh -- wgmma.fence / .commit_group / .wait_group
//
// ARCH: sm_90a
//
// WGMMA lifecycle ops. wgmma.mma_async is asynchronous: results are
// not guaranteed visible until a wait_group of the appropriate
// committed group. This file provides:
//   wgmma_fence()       fence register/SMEM accesses BEFORE the first MMA
//   wgmma_commit_group()close current MMA batch into a tracked group
//   wgmma_wait_group<N>()wait until at most N committed groups remain
//
// Typical pipeline pattern:
//   fence -> mma_async (one or more) -> commit_group ;
//   ... (next iteration interleaves) ;
//   wait_group<1> ;  // backpressure: allow 1 in-flight group
//
// Constraints (PTX 9.7.17.7):
//   wgmma.fence must precede the first wgmma.mma_async in a group
//   .sync.aligned -- all 128 warpgroup threads must reach in lockstep
//
// Issuer: warp-group (128 threads).
// Source: knowledge/instructions/mma/wgmma.md
// PTX:    9.7.17.7 (wgmma.fence / .commit_group / .wait_group)
//
#include <cstdint>

// ---------------------------------------------------------------------------
// wgmma lifecycle instructions:
//   fence:        must precede wgmma.mma_async (orders regs + SMEM)
//   commit_group: closes current wgmma group
//   wait_group N: waits until at most N groups remain pending
// ---------------------------------------------------------------------------

// Fence before wgmma.mma_async (mandatory)
__device__ __forceinline__
void wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

// Close current WGMMA group
__device__ __forceinline__
void wgmma_commit_group() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

// Wait until at most N WGMMA groups remain pending
template <int N>
__device__ __forceinline__
void wgmma_wait_group() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}

#endif  // PL_AGENTIC_SM90A
