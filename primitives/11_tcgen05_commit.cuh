#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// 11_tcgen05_commit.cuh -- tcgen05.commit.cta_group::{1,2}.mbarrier::arrive::one
//
// ARCH: sm_100a
//
// One-thread commit. Arrives on an mbarrier when every prior tcgen05.mma /
// tcgen05.cp / tcgen05.shift issued (in program order from this thread) has
// retired. Used to signal MMA completion to the epilogue warp.
//
// Variants:
//   plain              -- arrive on the local SMEM mbarrier.
//   multicast::cluster -- arrive on every peer CTA's mbarrier via a cluster
//                         bit mask (one bit per peer CTA, lowest bit = self).
//                         The mbarrier must be in .shared::cluster.
// Source: knowledge/instructions/mma/tcgen05_mma.md
// PTX:    9.7.18.12.1 (tcgen05.commit)
//
#include <cstdint>

template <int CTA_GROUP>
__device__ __forceinline__ void tcgen05_commit(uint32_t mbar_smem_addr) {
  static_assert(CTA_GROUP == 1 || CTA_GROUP == 2,
                "tcgen05_commit: CTA_GROUP must be 1 or 2");
  if constexpr (CTA_GROUP == 1) {
    asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];\n"
      :: "r"(mbar_smem_addr));
  } else {
    asm volatile(
      "tcgen05.commit.cta_group::2.mbarrier::arrive::one.b64 [%0];\n"
      :: "r"(mbar_smem_addr));
  }
}

// Multicast variant: arrives on the mbarrier in every CTA whose bit is set
// in ctamask (bit i = peer CTA i in the cluster). Mbarrier must reside in
// .shared::cluster (use a cluster-translated address).
template <int CTA_GROUP>
__device__ __forceinline__ void tcgen05_commit_multicast(
    uint32_t mbar_smem_addr, uint16_t ctamask) {
  static_assert(CTA_GROUP == 1 || CTA_GROUP == 2,
                "tcgen05_commit_multicast: CTA_GROUP must be 1 or 2");
  if constexpr (CTA_GROUP == 1) {
    asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster."
      "multicast::cluster.b64 [%0], %1;\n"
      :: "r"(mbar_smem_addr), "h"(ctamask));
  } else {
    asm volatile(
      "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster."
      "multicast::cluster.b64 [%0], %1;\n"
      :: "r"(mbar_smem_addr), "h"(ctamask));
  }
}

// Lead-thread-predicated cta_group::1 commit: only lane `lead`!=0 arrives.
__device__ __forceinline__ void tcgen05_commit1_lead(uint32_t lead, uint32_t mbar_smem_addr) {
  asm volatile(
    "{\n\t"
    ".reg .pred q;\n\t"
    "setp.ne.b32 q, %0, 0;\n\t"
    "@q tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%1];\n\t"
    "}\n"
    :: "r"(lead), "r"(mbar_smem_addr));
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
