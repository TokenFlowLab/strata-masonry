#pragma once
#if defined(PL_AGENTIC_SM90A)
// 92_mma_warp_hopper.cuh -- Hopper WGMMA consumer warp role
//
// ARCH: sm_90a
//
// Hopper warp-group MMA-consumer role. Iterates a K-loop, waiting on
// per-stage full mbarriers, issuing wgmma m64n64k16 atoms (SS variant)
// against caller-supplied SMEM A/B tiles, then signalling consumed via
// the empty mbarrier. All 128 threads of the warpgroup participate.
//
// Composes #29-#33 (mbarrier), #43 (smem_desc_hopper), #53 (wgmma_f16_ss),
// #59 (wgmma_fence_commit_wait), #69 (mbarrier_phase_tracking).
//
// Block function (per code/PLAN.md "Block function signature contract"):
//
//   template <int NUM_STAGES, int TILE_A_BYTES, int TILE_B_BYTES>
//   __device__ __forceinline__ void
//   mma_warp_hopper_block(char* smem_a, char* smem_b,
//                         uint64_t* full_mb, uint64_t* empty_mb,
//                         int num_k_tiles, float (&d)[32], int tid);
//
// - smem_a / smem_b: caller-allocated multi-stage SMEM regions
//   (NUM_STAGES * TILE_X_BYTES each).
// - full_mb / empty_mb: NUM_STAGES-deep mbarrier arrays (caller-init'd).
// - num_k_tiles: K-loop trip count.
// - d: caller-allocated 32-reg f32 accumulator (per-thread).
// - tid: thread index within the warpgroup (0..127).
//
// Caller-collective on the 128-thread warpgroup (warp 0..3).
//
// Source: knowledge/building_blocks/mma_warp.md
// PTX:    9.7.17.5 (wgmma.mma_async), 9.7.17.7 (fence/commit/wait)

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../primitives/29_mbarrier_init.cuh"
#include "../primitives/30_mbarrier_arrive.cuh"
#include "../primitives/33_mbarrier_try_wait.cuh"
#include "../primitives/43_smem_desc_hopper.cuh"
#include "../primitives/53_wgmma_f16_ss.cuh"
#include "../primitives/59_wgmma_fence_commit_wait.cuh"
#include "../composites/118_mbarrier_phase_tracking.cuh"

// One full K-tile = one WGMMA m64n64k16 atom. SS variant: A,B both in SMEM.
// Flat row-major layout (swizzle=0): LBO=row-stride bytes=32, SBO=8-row=256.
__device__ __forceinline__
void mma_warp_hopper_stage(float (&d)[32], uint32_t sa, uint32_t sb,
                            bool scale_d) {
  uint64_t ad = build_smem_desc_hopper(sa, 32, 256, 0);
  uint64_t bd = build_smem_desc_hopper(sb, 32, 256, 0);
  wgmma_f16_ss_m64n64k16(d, ad, bd, scale_d);
}

template <int NUM_STAGES, int TILE_A_BYTES, int TILE_B_BYTES>
__device__ __forceinline__
void mma_warp_hopper_block(char* smem_a, char* smem_b,
                           uint64_t* full_mb, uint64_t* empty_mb,
                           int num_k_tiles, float (&d)[32], int tid) {
  uint32_t fb = static_cast<uint32_t>(__cvta_generic_to_shared(full_mb));
  uint32_t eb = static_cast<uint32_t>(__cvta_generic_to_shared(empty_mb));
  PhaseTracker<NUM_STAGES> tr;
  for (int k = 0; k < num_k_tiles; ++k) {
    int s = tr.get_stage();
    mbarrier_wait_parity(fb + s * 8, tr.get_phase());
    wgmma_fence();
    uint32_t sa = static_cast<uint32_t>(
        __cvta_generic_to_shared(smem_a + s * TILE_A_BYTES));
    uint32_t sb = static_cast<uint32_t>(
        __cvta_generic_to_shared(smem_b + s * TILE_B_BYTES));
    mma_warp_hopper_stage(d, sa, sb, (k > 0));
    wgmma_commit_group();
    wgmma_wait_group<0>();
    if (tid == 0) mbarrier_arrive_nostate(eb + s * 8);
    tr.advance();
    __syncthreads();
    if (tid == 0) mbarrier_arrive_nostate(fb + s * 8);
    __syncthreads();
  }
}

#endif  // PL_AGENTIC_SM90A
