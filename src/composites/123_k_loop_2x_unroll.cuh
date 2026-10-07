#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// 123_k_loop_2x_unroll.cuh -- 2x unrolled Blackwell K-loop (two issues per wait)
//
// ARCH: sm_100a
//
// Some MMA patterns can issue two tcgen05.mma against the same stage's SMEM
// before the empty mbarrier needs to fire. This composite unrolls the inner
// wait/issue pair by two to amortize the mbarrier spin overhead.
// PTX:    9.7.18.10.10.1 (tcgen05.mma), 9.7.18.6.2 (pipelined pairs)
//
#include <cstdint>
#include "../primitives/11_tcgen05_commit.cuh"
#include "../primitives/33_mbarrier_try_wait.cuh"
#include "./118_mbarrier_phase_tracking.cuh"

template <int NUM_STAGES, int CTA_GROUP, typename IssueMma>
__device__ __forceinline__ void k_loop_blackwell_2x(
    uint64_t* full_bars, uint64_t* empty_bars,
    uint32_t acc_mbar, uint16_t ctamask,
    int k_blocks,
    MbarrierPhaseTracker<NUM_STAGES>& phase,
    IssueMma&& issue_mma) {
  (void)empty_bars;
  int k = 0;
  for (; k + 1 < k_blocks; k += 2) {
    int stage = phase.stage();
    mbarrier_wait_parity(smem_ptr_u32(&full_bars[stage]),
                                   phase.current_phase());
    issue_mma(k);
    issue_mma(k + 1);
    phase.advance();
  }
  for (; k < k_blocks; ++k) {
    int stage = phase.stage();
    mbarrier_wait_parity(smem_ptr_u32(&full_bars[stage]),
                                   phase.current_phase());
    issue_mma(k);
    phase.advance();
  }
  tcgen05_commit_multicast<CTA_GROUP>(acc_mbar, ctamask);
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
