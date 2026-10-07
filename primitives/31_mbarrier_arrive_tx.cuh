// 31_mbarrier_arrive_tx.cuh -- mbarrier.arrive.expect_tx + standalone
//                              expect_tx / complete_tx
//
// ARCH: sm_90a
//
// Combined arrive + expect_tx registers txBytes that TMA (or custom async
// producers) will credit via complete_tx. Pipelines typically call
// mbarrier_arrive_expect_tx once per TMA load before issuing the load.
//
//   mbarrier.arrive.expect_tx        arrive (count=1) AND register tx bytes.
//   mbarrier.expect_tx               register tx bytes WITHOUT consuming arrival.
//   mbarrier.complete_tx             explicitly credit completed bytes.

#pragma once

// Source: knowledge/instructions/barrier/mbarrier.md
// PTX:    9.7.15.16.14 (mbarrier.expect_tx)
//
#include <cstdint>

// Arrive + register expected bytes (most common: before TMA load).
__device__ __forceinline__
void mbarrier_arrive_expect_tx(uint32_t mbar_smem, uint32_t expected_bytes) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
               :: "r"(mbar_smem), "r"(expected_bytes) : "memory");
}

// Arrive + expect_tx, return the opaque 64-bit state token.
__device__ __forceinline__
uint64_t mbarrier_arrive_expect_tx_state(uint32_t mbar_smem,
                                         uint32_t expected_bytes) {
  uint64_t state;
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 %0, [%1], %2;\n"
               : "=l"(state) : "r"(mbar_smem), "r"(expected_bytes) : "memory");
  return state;
}

// Standalone expect_tx (no arrive). Register expected bytes only.
__device__ __forceinline__
void mbarrier_expect_tx(uint32_t mbar_smem, uint32_t expected_bytes) {
  asm volatile("mbarrier.expect_tx.shared::cta.b64 [%0], %1;\n"
               :: "r"(mbar_smem), "r"(expected_bytes) : "memory");
}

// Standalone complete_tx -- credit txBytes to the mbarrier's tx counter.
// Normally TMA hardware does this; this primitive lets custom async
// producers signal completion.
__device__ __forceinline__
void mbarrier_complete_tx(uint32_t mbar_smem, uint32_t bytes) {
  asm volatile("mbarrier.complete_tx.shared::cta.b64 [%0], %1;\n"
               :: "r"(mbar_smem), "r"(bytes) : "memory");
}

// Arrive + expect_tx with .release.cluster scope. Used by 2SM TMA load
// pipelines where the issue-side CTA arrives + registers expected bytes
// against an mbar that's visible to peer CTAs in the cluster (the
// .release publishes prior writes to peers; .cluster scope makes the
// arrival visible cluster-wide).
__device__ __forceinline__
void mbarrier_arrive_expect_tx_release_cluster(uint32_t mbar_smem,
                                               uint32_t expected_bytes) {
  asm volatile("mbarrier.arrive.expect_tx.release.cluster.shared::cta.b64"
               " _, [%0], %1;\n"
               :: "r"(mbar_smem), "r"(expected_bytes) : "memory");
}

// Arrive + expect_tx on a cluster-shared mbar. The mbar address is in
// .shared::cluster -- i.e., remote, typically produced by mapa.shared::
// cluster from a local SMEM address. Used by cluster-coordinated
// producers (e.g. CLC sched warp's expect_tx setup on each peer's
// clc_full mbar) where lanes 0..K-1 each remap to peer 0..K-1 and
// arrive on the remote mbar.
//
// Sem / scope: defaults (.release / .cta per PTX 9.7.15.16.16). Per-thread
// instruction; not .sync.aligned.
__device__ __forceinline__
void mbarrier_arrive_expect_tx_cluster(uint32_t cluster_smem_addr,
                                       uint32_t expected_bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%0], %1;\n"
               :: "r"(cluster_smem_addr), "r"(expected_bytes) : "memory");
}

// mbarrier.complete_tx.shared::cluster.relaxed.cluster.b64 [mbar], bytes
//   Credit `bytes` to a PEER CTA's transaction barrier without storing data. Used to top up a
//   barrier armed for the MAXIMUM cluster size when the actual cluster is smaller.
//   Distinct from mbarrier_complete_tx above (.shared::cta scope) -- not substitutable.
__device__ __forceinline__
void mbarrier_complete_tx_cluster_relaxed(uint32_t remote_mbar, uint32_t bytes) {
  asm volatile("mbarrier.complete_tx.shared::cluster.relaxed.cluster.b64 [%0], %1;\n"
               :: "r"(remote_mbar), "r"(bytes) : "memory");
}
