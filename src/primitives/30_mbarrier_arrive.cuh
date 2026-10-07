// 30_mbarrier_arrive.cuh -- mbarrier.arrive (plain, .release, .cluster)
//
// ARCH: sm_90a
//
// Decrement the mbarrier's arrival count by 1 (or by explicit count).
// The plain arrive uses .relaxed semantics; .release variants publish prior
// writes in this thread to threads that observe phase completion.
//
// All variants return the opaque 64-bit state token (callers may ignore it).

#pragma once

// PTX:    9.7.15.16.16 (mbarrier.arrive)
//
#include <cstdint>

// Plain arrive (count = 1). Returns state token.
__device__ __forceinline__
uint64_t mbarrier_arrive(uint32_t mbar_smem) {
  uint64_t state;
  asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];\n"
               : "=l"(state) : "r"(mbar_smem) : "memory");
  return state;
}

// Arrive with explicit count.
__device__ __forceinline__
uint64_t mbarrier_arrive_count(uint32_t mbar_smem, uint32_t count) {
  uint64_t state;
  asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1], %2;\n"
               : "=l"(state) : "r"(mbar_smem), "r"(count) : "memory");
  return state;
}

// .release arrive at CTA scope (count = 1). Two names provided.
__device__ __forceinline__
uint64_t mbarrier_arrive_release(uint32_t mbar_smem) {
  uint64_t state;
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 %0, [%1];\n"
               : "=l"(state) : "r"(mbar_smem) : "memory");
  return state;
}

__device__ __forceinline__
uint64_t mbarrier_arrive_release_cta(uint32_t mbar_smem) {
  uint64_t state;
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 %0, [%1];\n"
               : "=l"(state) : "r"(mbar_smem) : "memory");
  return state;
}

// .release arrive at cluster scope (for cluster-visible mbarriers).
__device__ __forceinline__
uint64_t mbarrier_arrive_release_cluster(uint32_t mbar_smem) {
  uint64_t state;
  asm volatile("mbarrier.arrive.release.cluster.shared::cta.b64 %0, [%1];\n"
               : "=l"(state) : "r"(mbar_smem) : "memory");
  return state;
}

// Arrive without capturing state (PTX `_` sink). Useful when state is unused.
__device__ __forceinline__
void mbarrier_arrive_nostate(uint32_t mbar_smem) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n"
               :: "r"(mbar_smem) : "memory");
}

// Cluster-shared mbarrier address forms. The `.shared::cluster` suffix
// declares the address operand as residing in the cluster's distributed
// shared memory window (i.e. usable from any CTA in the cluster, possibly
// after `mapa` or peer-bit-mask routing). Per PTX 9.7.15.16.16:
//   - `mbarrier.arrive` on a `.shared::cluster` mbarrier cannot return
//     state (sink `_` is mandatory).
//   - When `.sem` and `.scope` are both unspecified, defaults are
//     `.sem=.release, .scope=.cta` -- i.e. release-CTA-scope memory
//     ordering on a cluster-shared barrier.
//
// **Gotcha:** The explicit `.scope=.cluster` form (cluster-scope memory
// ordering) interacts with the tcgen05 alloc state machine on sm_100a:
// it triggers `phase_invalid_during_alloc` in persistent + warp-spec +
// cta_group::2 kernels.
// The default `.scope=.cta` form does NOT trigger the trap. For
// cross-CTA TMEM-free signaling in such kernels, use the default-scope
// form (`mbarrier_arrive_cluster_default`).

// Default-scope arrive on a cluster-shared mbarrier. PTX implicit
// .sem=.release, .scope=.cta. Address must point to .shared::cluster
// space (commonly produced by `mapa` or peer-bit-mask).
__device__ __forceinline__
void mbarrier_arrive_cluster_default(uint32_t cluster_smem_addr) {
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];\n"
               :: "r"(cluster_smem_addr) : "memory");
}

// Cluster-scope release arrive on a cluster-shared mbarrier. AVOID for
// persistent + warp-spec + cta_group::2 sm_100a kernels (triggers the
// alloc state-machine guardrail trap).
// Provided for completeness and for non-warp-spec cluster-coordination
// patterns where the cluster-wide memory ordering is required.
__device__ __forceinline__
void mbarrier_arrive_cluster_release_cluster_scope(uint32_t cluster_smem_addr) {
  asm volatile(
      "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];\n"
      :: "r"(cluster_smem_addr) : "memory");
}
