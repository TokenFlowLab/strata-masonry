#pragma once
#if defined(PL_AGENTIC_SM90A)
// 122_k_loop_hopper.cuh -- Hopper one-K-tile inner loop body
//
// ARCH: sm_90a
//
// One iteration of the K-dimension reduction loop for Hopper WGMMA:
//   1. mbarrier.try_wait.parity on full_mbar[stage]
//   2. wgmma.fence
//   3. wgmma.mma_async (one or more sub-K blocks)
//   4. wgmma.commit_group
//   5. wgmma.wait_group<N> (pipelined backpressure)
//   6. mbarrier.arrive on empty_mbar[stage] (signal load warp)
//
// Composes #33 (mbarrier_try_wait), #30 (mbarrier_arrive), #53-#59
// (wgmma family), and #43 (smem_desc_hopper).
//
// Constraints:
//   wait_group<1> allows 1 group in-flight (the previous iter's MMA);
//     adjust to wait_group<0> for stricter ordering at small cost
//   empty_mbar arrive must happen AFTER the wait_group covering the
//     stage's SMEM reads, else load may overwrite mid-MMA
//
// Issuer: warp-group (128 threads) -- the MMA consumer.
// Source: knowledge/building_blocks/mma_warp.md
// PTX:    9.7.17.5 (wgmma.mma_async), 9.7.17.7 (fence/commit/wait)
//
#include <cstdint>
#include "33_mbarrier_try_wait.cuh"
#include "30_mbarrier_arrive.cuh"
#include "53_wgmma_f16_ss.cuh"
#include "59_wgmma_fence_commit_wait.cuh"
#include "43_smem_desc_hopper.cuh"

// ---------------------------------------------------------------------------
// Hopper K-loop for one output tile.
// Consumes one mainloop pipeline stage: wait for full_mbar, issue WGMMA
// for all K sub-tiles within the SMEM stage, signal empty_mbar when done.
//
// NUM_K_BLOCKS: number of K=16 MMA atoms per SMEM stage
// ---------------------------------------------------------------------------

// Single-stage K-loop iteration for FP16 SS-form WGMMA m64n64k16
// Caller builds a_desc, b_desc from SMEM base + K offset
template <int NUM_K_BLOCKS>
__device__ __forceinline__
void k_loop_hopper_f16_n64_stage(
    float (&d)[32],
    uint32_t full_mbar, uint32_t empty_mbar,
    uint32_t phase,
    uint32_t smem_a_base, uint32_t smem_b_base,
    uint32_t stride_a_bytes, uint32_t stride_b_bytes,
    bool scale_d = true) {
    // Wait for stage to be filled
    mbarrier_wait_parity(full_mbar, phase);
    // Fence before MMA (orders regs + SMEM)
    wgmma_fence();
    // K-block loop: iterate over K sub-tiles within this stage
    #pragma unroll
    for (int kb = 0; kb < NUM_K_BLOCKS; kb++) {
        uint64_t a_desc = build_smem_desc_hopper_b128(
            smem_a_base + kb * 16 * 2 /* FP16 */, stride_a_bytes);
        uint64_t b_desc = build_smem_desc_hopper_b128(
            smem_b_base + kb * 16 * 2, stride_b_bytes);
        // First MMA: accumulate from caller; subsequent: always accumulate
        bool sd = (kb == 0) ? scale_d : true;
        wgmma_f16_ss_m64n64k16(d, a_desc, b_desc, sd);
    }
    // Commit the group
    wgmma_commit_group();
    // Wait for all pending WGMMA to complete
    wgmma_wait_group<0>();
    // Signal stage is consumed (empty)
    mbarrier_arrive_nostate(empty_mbar);
}

// Simpler variant without fence-commit-wait (caller manages group)
template <int NUM_K_BLOCKS>
__device__ __forceinline__
void k_loop_hopper_f16_n64_body(
    float (&d)[32],
    uint32_t smem_a_base, uint32_t smem_b_base,
    uint32_t stride_a_bytes, uint32_t stride_b_bytes,
    bool scale_d = true) {
    #pragma unroll
    for (int kb = 0; kb < NUM_K_BLOCKS; kb++) {
        uint64_t a_desc = build_smem_desc_hopper_b128(
            smem_a_base + kb * 16 * 2, stride_a_bytes);
        uint64_t b_desc = build_smem_desc_hopper_b128(
            smem_b_base + kb * 16 * 2, stride_b_bytes);
        bool sd = (kb == 0) ? scale_d : true;
        wgmma_f16_ss_m64n64k16(d, a_desc, b_desc, sd);
    }
}

#endif  // PL_AGENTIC_SM90A
