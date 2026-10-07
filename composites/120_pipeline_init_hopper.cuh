#pragma once
#if defined(PL_AGENTIC_SM90A)
// 120_pipeline_init_hopper.cuh -- Hopper pipeline barrier initialization
//
// ARCH: sm_90a
//
// Initializes the SMEM mbarrier array for a Hopper warp-specialized
// pipeline. Hopper pipelines need only full_mbar (load -> MMA) and
// empty_mbar (MMA -> load); no CLC/throttle/acc-double-buffer barriers
// (those are Blackwell-only). After init, fences ensure barriers are
// visible to all warps before they enter their roles.
//
// Constraints:
//   one thread (typically tid==0) issues mbarrier.init for each barrier
//   followed by fence.mbarrier_init.release.cta and __syncthreads()
//   barrier_count must match the number of consumers per producer
//
// Issuer: one thread per CTA (broadcast to all via syncthreads).
// Source: knowledge/building_blocks/pipeline.md
// PTX:    9.7.15.16.12 (mbarrier.init), 9.7.15.4 (fence.mbarrier_init)
//
#include <cstdint>
#include "29_mbarrier_init.cuh"
#include "35_fence_mbarrier_init.cuh"
#include "37_bar_sync.cuh"

// ---------------------------------------------------------------------------
// Initialize mbarrier array for Hopper warp-specialized pipeline.
// Simpler than Blackwell: only full/empty barriers (no CLC, no throttle,
// no double-buffered accumulator in TMEM).
//
// Arrival counts:
//   full_mbar: 1 (single TMA load arrives via complete_tx)
//   empty_mbar: consumer count (typically 1-2 warp groups = 128-256 threads)
// ---------------------------------------------------------------------------

// Initialize a single-stage full/empty barrier pair
__device__ __forceinline__
void pipeline_init_hopper_1stage(
    uint32_t full_mbar, uint32_t empty_mbar,
    uint32_t empty_arrival_count) {
    mbarrier_init(full_mbar, 1);
    mbarrier_init(empty_mbar, empty_arrival_count);
}

// Initialize multi-stage pipeline barriers
// full_mbar_base: SMEM address of first full mbar (array of NUM_STAGES)
// empty_mbar_base: SMEM address of first empty mbar (array of NUM_STAGES)
template <int NUM_STAGES>
__device__ __forceinline__
void pipeline_init_hopper(
    uint32_t full_mbar_base,
    uint32_t empty_mbar_base,
    uint32_t empty_arrival_count) {
    #pragma unroll
    for (int s = 0; s < NUM_STAGES; s++) {
        mbarrier_init(full_mbar_base + s * 8, 1);
        mbarrier_init(empty_mbar_base + s * 8, empty_arrival_count);
    }
}

// Full init sequence: init + fence + CTA sync
template <int NUM_STAGES>
__device__ __forceinline__
void pipeline_init_hopper_full(
    uint32_t full_mbar_base,
    uint32_t empty_mbar_base,
    uint32_t empty_arrival_count,
    uint32_t total_threads) {
    // Only elected thread initializes
    if (threadIdx.x == 0) {
        pipeline_init_hopper<NUM_STAGES>(full_mbar_base, empty_mbar_base,
                                           empty_arrival_count);
    }
    // Sync all threads in CTA before pipeline starts
    bar_sync<0>(total_threads);
}

#endif  // PL_AGENTIC_SM90A
