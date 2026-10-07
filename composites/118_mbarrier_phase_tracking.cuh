// 118_mbarrier_phase_tracking.cuh -- phase parity flip + stage wraparound
//
// ARCH: sm_90a
//
// Maintains expected parity for try_wait.parity in multi-stage pipelines.
//
//   idx2stage / idx2phase  stateless (slot, phase) from an explicit
//                                  ring-iteration index `idx`; order-independent.
//
// and two stateful tracker styles:
//
//   MbarrierPhaseTracker<N>  per-stage independent parity (phase[i]).
//                            Each stage flips its own parity on completion.
//                            Useful when stages can complete out of order.
//
//   PhaseTracker<N>          single global phase that flips when the stage
//   EmptyPhaseTracker<N>     index wraps past N-1. The Empty form starts
//                            at parity 1 (producer waiting on "consumed").

#pragma once

// Source: knowledge/building_blocks/pipeline.md
// PTX:    9.7.15.16.19 (mbarrier.try_wait.parity)
//
#include <cstdint>

// =============================================================================
// Per-stage independent phase tracker
// =============================================================================

template <int NUM_STAGES>
struct MbarrierPhaseTracker {
  // phase[s] = current expected parity for stage s. Flips each cycle.
  uint32_t phase[NUM_STAGES];
  int idx;  // current stage index

  __device__ __forceinline__
  void init() {
    for (int i = 0; i < NUM_STAGES; ++i) phase[i] = 0;
    idx = 0;
  }

  // Parity to pass to try_wait.parity for the current stage.
  __device__ __forceinline__
  uint32_t current_phase() const { return phase[idx]; }

  // Call after a successful wait: flip parity for the consumed stage and
  // advance the stage index with wraparound.
  __device__ __forceinline__
  void advance() {
    phase[idx] ^= 1u;
    idx = (idx + 1) % NUM_STAGES;
  }

  __device__ __forceinline__
  int stage() const { return idx; }
};

// =============================================================================
// Stateless (slot, phase) from an explicit ring-iteration index
//
// `idx` is the running ring-iteration index 0,1,2,.. -- one per pipeline stage /
// ring slot consumed (unbounded, unlike the wrapped stage index in the trackers).
// Map it to its (slot, phase) for a depth-num_stages ring: slot is idx % N; the
// phase bit flips each time a slot is reused, i.e. every N steps, so (idx / N) & 1.
//
// Order-independent: because they take the idx directly, they stay correct even
// if iterations are visited out of order or skipped. Prefer these when the access
// pattern is irregular, or when the idx is already in hand (e.g. idx = 2*kblock
// for an interleaved K/V ring). The stateful PhaseTracker below is the
// lockstep-only alternative.
// =============================================================================

__device__ __forceinline__
int idx2stage(int idx, int num_stages) {
  return idx % num_stages;
}

__device__ __forceinline__
uint32_t idx2phase(int idx, int num_stages) {
  return (idx / num_stages) & 1;
}

// =============================================================================
// Global-phase tracker (parity flips on wrap)
//
// Stateful (slot, phase) for a depth-NUM_STAGES ring: advance() walks slot
// 0,1,..,N-1,0,.. and flips phase on each wrap. After k advances it equals the
// stateless pair above: get_stage()==idx2stage(k,N), get_phase()==idx2phase(k,N).
//
// CORRECT ONLY IF iterations are consumed in strict order, exactly one advance()
// each. It tracks a *count*, not the idx, so any skipped or out-of-order iteration
// desyncs it. When the consumption order is fixed (e.g. an MMA warp draining a K/V
// ring K0,V0,K1,V1,..) this is the simplest option. When it is not, use the
// stateless idx2stage/idx2phase above.
// =============================================================================

template <int NUM_STAGES>
struct PhaseTracker {
  int stage;
  uint32_t phase;

  __device__ __forceinline__
  PhaseTracker() : stage(0), phase(0) {}

  __device__ __forceinline__
  void advance() {
    stage++;
    if (stage == NUM_STAGES) {
      stage = 0;
      phase ^= 1;
    }
  }

  __device__ __forceinline__
  int get_stage() const { return stage; }

  __device__ __forceinline__
  uint32_t get_phase() const { return phase; }
};

// Inverted phase for empty_mbar (producer waits on "consumed").
template <int NUM_STAGES>
struct EmptyPhaseTracker {
  int stage;
  uint32_t phase;

  __device__ __forceinline__
  EmptyPhaseTracker() : stage(0), phase(1) {}

  __device__ __forceinline__
  void advance() {
    stage++;
    if (stage == NUM_STAGES) {
      stage = 0;
      phase ^= 1;
    }
  }

  __device__ __forceinline__
  int get_stage() const { return stage; }

  __device__ __forceinline__
  uint32_t get_phase() const { return phase; }
};

// In-place advance for the (int stage, uint32_t phase) pair used inline by
// kernels and warp-role blocks (alternative to the stateful PhaseTracker
// struct when the caller already owns the two variables).
//
// Equivalent to `stage = (stage + 1) % STAGES; if (stage == 0) phase ^= 1`
// but uses explicit "compare with STAGES" so the SASS is identical regardless
// of whether STAGES is a power of 2 (no Barrett reduction for non-pow2 N).
// Matches the form used in the sibling PhaseTracker<N>::advance() struct.
template <int STAGES>
__device__ __forceinline__
void advance_stage_phase(int& stage, uint32_t& phase) {
  ++stage;
  if (stage == STAGES) {
    stage = 0;
    phase ^= 1u;
  }
}
