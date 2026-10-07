// 114_fence_acq_rel.cuh -- fence.acq_rel.{cta,cluster,gpu}
//
// ARCH: sm_90a
//
// Acquire-release fence on the generic proxy: every load and store this thread
// issued before the fence completes before any it issues after it, at the given
// scope. SASS (sm_100a): MEMBAR.ALL.CTA; __threadfence_block() is the stronger
// membar.cta = MEMBAR.SC.CTA.
//
// Why a primitive: mbarrier.arrive is a release for prior STORES; a prior
// ld.shared whose value has not returned can still be reordered behind it. When
// the arrive frees a slot another warp will overwrite (the CLC response ring,
// composites/106), a fence between the load and the arrive completes the read.
// Composite 106 uses fence.proxy.async.shared::cta there (CUTLASS's choice): the
// slot's next writer is the async proxy, and that fence (MEMBAR.ALL.CTA +
// FENCE.VIEW.ASYNC.S) contains this one plus the generic -> async edge.
//
// Source: knowledge/instructions/fence/fence.md sec 4.4
// PTX:    9.7.15.4 (membar / fence)
//

#pragma once

__device__ __forceinline__
void fence_acq_rel_cta() {
  asm volatile("fence.acq_rel.cta;\n" ::: "memory");
}

__device__ __forceinline__
void fence_acq_rel_cluster() {
  asm volatile("fence.acq_rel.cluster;\n" ::: "memory");
}

__device__ __forceinline__
void fence_acq_rel_gpu() {
  asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
}
