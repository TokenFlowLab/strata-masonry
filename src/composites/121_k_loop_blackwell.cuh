#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// 121_k_loop_blackwell.cuh -- wait(full) + MMA x K_BLOCKS + commit.multicast
//
// ARCH: sm_100a
//
// A single tile's K-loop on Blackwell: consume K_BLOCKS worth of TMA-loaded
// stages via tcgen05.mma (caller supplies desc builders), then commit the
// accumulator via tcgen05.commit.multicast.
// PTX:    9.7.18.10.10.1 (tcgen05.mma), 9.7.18.6.2 (pipelined pairs)
//
#include <cstdint>
#include "../primitives/3_tcgen05_mma_f16.cuh"
#include "../primitives/11_tcgen05_commit.cuh"
#include "../primitives/33_mbarrier_try_wait.cuh"
#include "./118_mbarrier_phase_tracking.cuh"

// Call inside the MMA warp. `issue_mma` is a functor (int k_block) called
// after each wait; it builds descs and invokes tcgen05_mma_* itself.
template <int NUM_STAGES, int CTA_GROUP, typename IssueMma>
__device__ __forceinline__ void k_loop_blackwell(
    uint64_t* full_bars, uint64_t* empty_bars,
    uint32_t acc_mbar, uint16_t ctamask,
    int k_blocks,
    MbarrierPhaseTracker<NUM_STAGES>& phase,
    IssueMma&& issue_mma) {
  for (int k = 0; k < k_blocks; ++k) {
    int stage = phase.stage();
    uint32_t fmb = smem_ptr_u32(&full_bars[stage]);
    uint32_t emb = smem_ptr_u32(&empty_bars[stage]);
    mbarrier_wait_parity(fmb, phase.current_phase());

    issue_mma(k);  // caller issues tcgen05.mma with stage's descriptors

    (void)emb;  // the consumer MMA warp does NOT arrive on empty; the
                // dedicated scheduler warp does after commit lands.
    phase.advance();
  }
  // Commit the full K tile via multicast (peer CTA also receives signal).
  tcgen05_commit_multicast<CTA_GROUP>(acc_mbar, ctamask);
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
