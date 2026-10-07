// 35_fence_mbarrier_init.cuh -- fence.mbarrier_init.release.cluster
//
// ARCH: sm_90a
//
// Publishes a freshly-initialized mbarrier (mbarrier.init, #29) to peer CTAs
// in the cluster. Must be issued after mbarrier.init and before any peer CTA
// references the barrier (arrives from peer CTAs, multicast TMA completion).
// Only the .cluster scope is defined by the ISA; there is no .cta variant.
//
// Source: knowledge/instructions/fence/fence.md
// PTX:    9.7.15.4 (fence.mbarrier_init.release.cluster)
//

#pragma once

__device__ __forceinline__
void fence_mbarrier_init_release_cluster() {
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
