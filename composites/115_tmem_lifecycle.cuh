#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// 115_tmem_lifecycle.cuh -- alloc -> relinquish -> (use) -> barrier.cluster -> dealloc
//
// ARCH: sm_100a
//
// Full TMEM lifecycle in one call: allocate a column range, relinquish the
// alloc permit (so other CTAs can start), run `body`, cluster-sync the pair
// (for cta_group::2), dealloc.
// Source: knowledge/building_blocks/pipeline.md
// PTX:    9.7.18.7.1 (alloc / dealloc), 9.7.15.3 (barrier.cluster)
//
#include <cstdint>
#include "../primitives/0_tcgen05_alloc.cuh"
#include "../primitives/1_tcgen05_dealloc.cuh"
#include "../primitives/2_tcgen05_relinquish.cuh"
#include "../primitives/38_barrier_cluster.cuh"

template <int CTA_GROUP, typename Body>
__device__ __forceinline__ void tmem_lifecycle(
    uint32_t smem_slot, uint32_t n_cols, Body&& body) {
  // Warp 0 (lanes 0..31) allocates; broadcasts base via shared slot.
  if (threadIdx.x < 32) {
    tcgen05_alloc<CTA_GROUP>(smem_slot, n_cols);
    tcgen05_relinquish_alloc_permit<CTA_GROUP>();
  }
  __syncthreads();

  uint32_t tmem_base = *reinterpret_cast<uint32_t*>(
      (char*)__cvta_shared_to_generic(smem_slot));
  body(tmem_base);

  // For 2SM, coordinate exits.
  if constexpr (CTA_GROUP == 2) {
    barrier_cluster_arrive();
    barrier_cluster_wait();
  }
  __syncthreads();

  if (threadIdx.x < 32) {
    tcgen05_dealloc<CTA_GROUP>(tmem_base, n_cols);
  }
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
