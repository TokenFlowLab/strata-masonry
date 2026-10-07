// 32_mbarrier_arrive_drop.cuh -- mbarrier.arrive_drop (permanent participant
//                                removal)
//
// ARCH: sm_90a
//
// Arrive on the mbarrier AND permanently reduce the arrival count for all
// subsequent phases. Used when a warp/CTA exits the pipeline early or
// conditionally withdraws.

#pragma once

// PTX:    9.7.15.16.17 (mbarrier.arrive_drop)
//
#include <cstdint>

// Arrive and drop (count = 1). Returns the opaque state token.
__device__ __forceinline__
uint64_t mbarrier_arrive_drop(uint32_t mbar_smem) {
  uint64_t state;
  asm volatile("mbarrier.arrive_drop.shared::cta.b64 %0, [%1];\n"
               : "=l"(state) : "r"(mbar_smem) : "memory");
  return state;
}

// Arrive and drop with explicit count. Returns the opaque state token.
__device__ __forceinline__
uint64_t mbarrier_arrive_drop_count(uint32_t mbar_smem, uint32_t count) {
  uint64_t state;
  asm volatile("mbarrier.arrive_drop.shared::cta.b64 %0, [%1], %2;\n"
               : "=l"(state) : "r"(mbar_smem), "r"(count) : "memory");
  return state;
}

// Arrive and drop without capturing state (PTX `_` sink).
__device__ __forceinline__
void mbarrier_arrive_drop_nostate(uint32_t mbar_smem) {
  asm volatile("mbarrier.arrive_drop.shared::cta.b64 _, [%0];\n"
               :: "r"(mbar_smem) : "memory");
}

// ---------------------------------------------------------------------------
// .release variants. When a CTA permanently exits a pipeline that was
// publishing writes to peers (e.g. TMEM scratch handed off to the
// epilogue, SMEM tile produced for downstream consumers), the drop must
// publish those writes. The `.release.shared::cta` form is for CTA-local
// pipelines; `.release.cluster.shared::cluster` for cluster-wide.
// ---------------------------------------------------------------------------

// Arrive and drop with .release.cta semantics. Publishes prior writes
// in this thread to threads in the same CTA that observe phase
// completion. Per ISA, .release with arrive_drop requires an explicit
// .cta or .cluster scope qualifier.
__device__ __forceinline__
void mbarrier_arrive_drop_release_shared_cta(uint32_t mbar_smem) {
  asm volatile("mbarrier.arrive_drop.release.cta.shared::cta.b64 _, [%0];\n"
               :: "r"(mbar_smem) : "memory");
}

// Arrive and drop with .release.cluster.shared::cluster semantics.
// Publishes prior writes to peer CTAs in the cluster.
__device__ __forceinline__
void mbarrier_arrive_drop_release_cluster_shared_cluster(uint32_t mbar_smem) {
  asm volatile("mbarrier.arrive_drop.release.cluster.shared::cluster.b64"
               " _, [%0];\n"
               :: "r"(mbar_smem) : "memory");
}
