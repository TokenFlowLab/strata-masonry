#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// 103_clc_throttle_blackwell.cuh -- Blackwell CLC throttle pipeline.
//
// ARCH: sm_100a
//
// 2-stage throttle barrier pair that bounds the load-warp drift relative
// to the scheduler/MMA across persistent-loop tile boundaries. Without
// this, the load warp can race ahead by an unbounded number of tiles
// (limited only by SMEM stage count); the alloc state machine's phase
// invariant requires a tighter bound.
//
// Pattern:
//   Init: throttle_full[N] arrive_count=1, throttle_empty[N] arrive_count=1.
//         Pre-arrive throttle_empty[N] times (all "free" at start).
//   Load warp (per tile):
//     wait throttle_empty[s]; arrive throttle_full[s]; advance s.
//   Sched warp (per tile):
//     wait throttle_full[s]; arrive throttle_empty[s]; advance s.
//   Both bars cycle in lockstep with N stages of headroom.
//
// PTX:    9.7.15.16 (mbarrier).

#include <cstdint>
#include "../primitives/29_mbarrier_init.cuh"
#include "../primitives/30_mbarrier_arrive.cuh"
#include "../primitives/33_mbarrier_try_wait.cuh"

template <int N_STAGES>
struct ClcThrottleBars {
  uint64_t* throttle_full;   // [N_STAGES]
  uint64_t* throttle_empty;  // [N_STAGES]
};

template <int N_STAGES>
struct ClcThrottleState {
  int s;                              // current stage index
  int phase[N_STAGES];                // per-stage parity tracker
};

template <int N_STAGES>
__device__ __forceinline__
ClcThrottleState<N_STAGES> clc_throttle_state_init() {
  ClcThrottleState<N_STAGES> st;
  st.s = 0;
  #pragma unroll
  for (int i = 0; i < N_STAGES; ++i) st.phase[i] = 0;
  return st;
}

// One-time init from a single thread. Pre-arrives `pre_arrive` times on
// EACH throttle_empty bar (typically pre_arrive = 1, so all stages are
// "free" at start; load warp's first wait succeeds without waiting for
// sched).
template <int N_STAGES>
__device__ __forceinline__
void clc_throttle_init(ClcThrottleBars<N_STAGES> bars, int pre_arrive = 1) {
  for (int i = 0; i < N_STAGES; ++i) {
    mbarrier_init(static_cast<uint32_t>(__cvta_generic_to_shared(&bars.throttle_full[i])),  /*arrive_count=*/1);
    mbarrier_init(static_cast<uint32_t>(__cvta_generic_to_shared(&bars.throttle_empty[i])), /*arrive_count=*/1);
    for (int p = 0; p < pre_arrive; ++p) {
      mbarrier_arrive(static_cast<uint32_t>(__cvta_generic_to_shared(&bars.throttle_empty[i])));
    }
  }
}

// === Load-warp side (producer of throttle_full, consumer of throttle_empty) ===

// Wait for the throttle_empty[s] slot to be free (sched has consumed
// the prior tile's "request").
template <int N_STAGES>
__device__ __forceinline__
void clc_throttle_load_acquire(ClcThrottleBars<N_STAGES> bars,
                               const ClcThrottleState<N_STAGES>& st) {
  uint32_t bar_addr = static_cast<uint32_t>(
      __cvta_generic_to_shared(&bars.throttle_empty[st.s]));
  mbarrier_wait_parity(bar_addr, st.phase[st.s]);
}

// Signal throttle_full[s]: this tile is in flight, sched can pick up.
template <int N_STAGES>
__device__ __forceinline__
void clc_throttle_load_commit(ClcThrottleBars<N_STAGES> bars,
                              const ClcThrottleState<N_STAGES>& st) {
  uint32_t bar_addr = static_cast<uint32_t>(
      __cvta_generic_to_shared(&bars.throttle_full[st.s]));
  mbarrier_arrive(bar_addr);
}

// === Sched-warp side (consumer of throttle_full, producer of throttle_empty) ===

template <int N_STAGES>
__device__ __forceinline__
void clc_throttle_sched_wait(ClcThrottleBars<N_STAGES> bars,
                             const ClcThrottleState<N_STAGES>& st) {
  uint32_t bar_addr = static_cast<uint32_t>(
      __cvta_generic_to_shared(&bars.throttle_full[st.s]));
  mbarrier_wait_parity(bar_addr, st.phase[st.s]);
}

template <int N_STAGES>
__device__ __forceinline__
void clc_throttle_sched_release(ClcThrottleBars<N_STAGES> bars,
                                const ClcThrottleState<N_STAGES>& st) {
  uint32_t bar_addr = static_cast<uint32_t>(
      __cvta_generic_to_shared(&bars.throttle_empty[st.s]));
  mbarrier_arrive(bar_addr);
}

// === Shared advance ===

template <int N_STAGES>
__device__ __forceinline__
void clc_throttle_state_advance(ClcThrottleState<N_STAGES>& st) {
  st.phase[st.s] ^= 1;
  st.s = (st.s + 1) % N_STAGES;
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
