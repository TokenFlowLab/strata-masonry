// 34_fence_proxy_async.cuh -- fence.proxy.async.shared::cta
//
// ARCH: sm_90a
//
// Orders the generic proxy (st.shared, stmatrix) with respect to the async
// proxy (TMA, tcgen05.cp). Issue AFTER a generic-proxy SMEM write and BEFORE
// an async-proxy read of the same address -- or vice versa.
//
// Most common use: between stmatrix (epilogue SMEM write) and
// cp.async.bulk.tensor (TMA store). NOT required after TMA load +
// mbarrier.try_wait (implicit fence).
//
// Source: knowledge/instructions/fence/fence.md
// PTX:    9.7.15.4 (fence.proxy.async)
//

#pragma once

// Fence for shared::cta scope (most common: epilogue pattern).
__device__ __forceinline__
void fence_proxy_async_shared_cta() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

// Global-scope cross-proxy fence (entire GPU, no scope qualifier).
// Two names provided for the same instruction body.
__device__ __forceinline__
void fence_proxy_async() {
  asm volatile("fence.proxy.async;\n" ::: "memory");
}

// NOTE: this is the shared::cta-scoped fence (matches the name). The body was
// previously the UNSCOPED `fence.proxy.async;`, which ptxas lowers to a
// MEMBAR.ALL.CTA + MEMBAR.ALL.GPU pair (~263 membar stall samples/kernel vs
// FA4's 0). The generic->async ordering we need (st.shared sO before TMA store)
// only requires shared::cta scope, which lowers to a cheap FENCE.VIEW.ASYNC.S.
__device__ __forceinline__
void fence_proxy_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}
