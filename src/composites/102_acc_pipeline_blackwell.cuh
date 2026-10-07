#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// 102_acc_pipeline_blackwell.cuh -- Blackwell accumulator pipeline.
//
// ARCH: sm_100a
//
// 2-stage TMEM-accumulator pipeline that escapes the tcgen05 alloc
// state machine guardrail trap when composed with persistent multi-tile
// kernels. Each stage owns one half of a 2x N_TILE_CLUSTER TMEM
// allocation; the producer (MMA warp) and consumer (epilogue warps)
// hand off via per-stage acc_full/acc_empty mbar pairs.
//
// Per-stage parity tracking is encoded in PipelineState{index, phase}.
// Producer skips its first 2 acquire waits (pipeline warm-up; bars
// are NOT pre-arrived). Consumer release uses the peer-bit-mask
// arrive pattern: every consumer thread arrives on peer 0's
// acc_empty[stage] bar (`Sm100MmaPeerBitMask = 0xFEFFFFFF`),
// independent of which peer the issuing thread runs on. This
// produces 2 * num_consumer_threads_per_cta total arrives on peer
// 0's bar, with arrive_count set accordingly.
//
// PTX:    9.7.15.16 (mbarrier), 9.7.18.12.1 (tcgen05.commit), 9.7.18.5
//         (Issue Granularity)

#include <cstdint>
#include "../primitives/11_tcgen05_commit.cuh"
#include "../primitives/29_mbarrier_init.cuh"
#include "../primitives/30_mbarrier_arrive.cuh"
#include "../primitives/33_mbarrier_try_wait.cuh"

// Mask matching cute::Sm100MmaPeerBitMask. AND a local SMEM addr with
// this to remap to peer 0's SMEM (clears the cluster peer bit). Used
// for the consumer_release arrive so all consumer threads (across both
// peers) arrive on a single bar in peer 0's SMEM.
static constexpr uint32_t SM100_ACC_PIPE_PEER_MASK = 0xFEFFFFFF;

// SMEM bar layout: 2 stages of (full, empty).
struct AccPipelineBars {
  uint64_t* acc_full;   // [2]
  uint64_t* acc_empty;  // [2]
};

// PipelineState-style cursor. Producer holds one; consumer holds another;
// each advances independently. `count` is the visit count; the current
// stage and that stage's expected phase are derived from it:
//   index = count & 1
//   stage_phase = (count >> 1) & 1   (consumer wait = previous bar phase)
struct AccPipelineState {
  int count;   // visit count (also encodes stage and phase)
};

__device__ __forceinline__
AccPipelineState acc_pipeline_state_init() {
  return AccPipelineState{0};
}

__device__ __forceinline__
int acc_pipeline_state_index(const AccPipelineState& s) {
  return s.count & 1;
}

__device__ __forceinline__
int acc_pipeline_state_phase(const AccPipelineState& s) {
  return (s.count >> 1) & 1;
}

__device__ __forceinline__
void acc_pipeline_state_advance(AccPipelineState& s) {
  s.count++;
}

// One-time init of the four bars. Call from a single thread; pair with
// __syncthreads() and fence_mbarrier_init.release.cluster on the caller's
// side. `consumer_arv_count` = 2 * num_consumer_threads_per_cta (e.g.
// 256 for 4 epi warps x 2 CTAs of 32 threads).
__device__ __forceinline__
void acc_pipeline_init(AccPipelineBars bars, uint32_t consumer_arv_count) {
  for (int s = 0; s < 2; ++s) {
    mbarrier_init(static_cast<uint32_t>(__cvta_generic_to_shared(&bars.acc_full[s])),  /*arrive_count=*/1);
    mbarrier_init(static_cast<uint32_t>(__cvta_generic_to_shared(&bars.acc_empty[s])), consumer_arv_count);
  }
}

// === PRODUCER (MMA warp) ===

// Acquire the next acc stage. Skips the wait for the first 2 visits
// (warm-up phase, bars NOT pre-arrived). Issued by ONE thread of the
// MMA warp.
__device__ __forceinline__
void acc_pipeline_producer_acquire(AccPipelineBars bars,
                                   const AccPipelineState& s) {
  if (s.count < 2) return;
  // Visit n on this stage (n = count >> 1) waits for phase (n - 1) & 1.
  // For count in {2, 3, 4, 5}: stage_visit = {1, 1, 2, 2}; wait_phase
  // = {0, 0, 1, 1}.
  int wait_phase = ((s.count >> 1) - 1) & 1;
  uint32_t bar_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(
          &bars.acc_empty[acc_pipeline_state_index(s)]));
  mbarrier_wait_parity(bar_addr, wait_phase);
}

// Commit the current acc stage. Issued by ONE thread of the MMA warp,
// AFTER the K-loop's tcgen05.mma instructions for this tile have been
// issued. Performs `tcgen05.commit.cta_group::2.multicast::cluster`
// which signals acc_full[index] in BOTH peers' SMEM.
template <int CTA_GROUP>
__device__ __forceinline__
void acc_pipeline_producer_commit(AccPipelineBars bars,
                                  const AccPipelineState& s,
                                  uint16_t ctamask) {
  uint32_t bar_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(
          &bars.acc_full[acc_pipeline_state_index(s)]));
  tcgen05_commit_multicast<CTA_GROUP>(bar_addr, ctamask);
}

// === CONSUMER (epilogue warps) ===

// Wait for the next acc stage to be ready. Caller's phase tracker.
__device__ __forceinline__
void acc_pipeline_consumer_wait(AccPipelineBars bars,
                                const AccPipelineState& s) {
  uint32_t bar_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(
          &bars.acc_full[acc_pipeline_state_index(s)]));
  mbarrier_wait_parity(bar_addr, acc_pipeline_state_phase(s));
}

// Release the current acc stage. Each consumer thread issues. Routes
// the arrive to PEER 0's acc_empty[index] via Sm100MmaPeerBitMask
// (clears bit 24 of the SMEM address). Both peers' threads thus arrive
// on the same bar; `arrive_count = 2 * num_consumer_threads_per_cta`
// must be set so the bar triggers when all have arrived.
__device__ __forceinline__
void acc_pipeline_consumer_release(AccPipelineBars bars,
                                   const AccPipelineState& s) {
  uint32_t local_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(
          &bars.acc_empty[acc_pipeline_state_index(s)]));
  uint32_t peer0_addr = local_addr & SM100_ACC_PIPE_PEER_MASK;
  mbarrier_arrive_cluster_release_cluster_scope(peer0_addr);
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
