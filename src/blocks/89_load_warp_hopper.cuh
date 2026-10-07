#pragma once
#if defined(PL_AGENTIC_SM90A)
// 89_load_warp_hopper.cuh -- Hopper TMA producer warp role (1SM)
//
// ARCH: sm_90a
//
// Hopper warp-specialized load-warp role. Issues NUM_STAGES-deep
// TMA-pipelined loads from a 2D tensor map, signalling completion via
// a per-stage full mbarrier and waiting on a per-stage empty mbarrier
// for back-pressure from the consumer.
//
// Composes #18 (tma_load), #29-#33 (mbarrier), #44 (elect_sync),
// #116 (tma_load_stage), #118 (phase tracking).
//
// Block functions:
//
//   template <int NUM_STAGES, int TILE_BYTES, int TILE_K>
//   __device__ __forceinline__ void
//   load_warp_hopper_block(const CUtensorMap& tma_a,
//                          char* smem_a,
//                          uint64_t* full_mb, uint64_t* empty_mb,
//                          int num_k_tiles, int lane);
//
// - tma_a: caller-built 2D tensor map descriptor.
// - smem_a: caller-allocated SMEM region of NUM_STAGES * TILE_BYTES bytes.
// - full_mb / empty_mb: caller-allocated SMEM mbarrier arrays of length
//   NUM_STAGES (caller initialises them).
// - num_k_tiles: number of K-tiles to load.
// - lane: thread's lane within the load warp (0..31). Caller gates by
//   warp; only lane 0 issues TMA + expect_tx.
//
//   template <int NUM_STAGES, int TILE_M, int TILE_K, int TILE_BYTES>
//   __device__ __forceinline__ void
//   load_warp_hopper_consumer_block(half* __restrict__ out,
//                                   char* smem_a,
//                                   uint64_t* full_mb, uint64_t* empty_mb,
//                                   int num_k_tiles, int lane);
//
// Caller-gated warp-collective consumer body. Waits each stage's full
// mbarrier, copies SMEM -> GMEM (lane-stride 32), arrives empty.
//
// PTX:    9.7.10.28.5.3 (TMA), 9.7.15.16.14 (expect_tx)

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../primitives/23_tma_tensormap.cuh"
#include "../primitives/29_mbarrier_init.cuh"
#include "../primitives/30_mbarrier_arrive.cuh"
#include "../primitives/33_mbarrier_try_wait.cuh"
#include "../primitives/44_elect_sync.cuh"
#include "../composites/116_tma_load_stage.cuh"
#include "../composites/118_mbarrier_phase_tracking.cuh"

template <int NUM_STAGES, int TILE_BYTES, int TILE_K>
__device__ __forceinline__
void load_warp_hopper_block(const CUtensorMap& tma_a,
                            char* smem_a,
                            uint64_t* full_mb, uint64_t* empty_mb,
                            int num_k_tiles, int lane) {
  if (lane != 0) return;
  uint32_t fb = static_cast<uint32_t>(__cvta_generic_to_shared(full_mb));
  uint32_t eb = static_cast<uint32_t>(__cvta_generic_to_shared(empty_mb));
  EmptyPhaseTracker<NUM_STAGES> tr;
  for (int k = 0; k < num_k_tiles; ++k) {
    int s = tr.get_stage();
    mbarrier_wait_parity(eb + s * 8, tr.get_phase());
    uint32_t smem_dst = static_cast<uint32_t>(
        __cvta_generic_to_shared(smem_a + s * TILE_BYTES));
    tma_load_stage_2d(&tma_a, smem_dst, fb + s * 8,
                      (uint32_t)TILE_BYTES,
                      /*coord_x=*/k * TILE_K,
                      /*coord_y=*/0);
    tr.advance();
  }
}

template <int NUM_STAGES, int TILE_M, int TILE_K, int TILE_BYTES>
__device__ __forceinline__
void load_warp_hopper_consumer_block(half* __restrict__ out,
                                     char* smem_a,
                                     uint64_t* full_mb, uint64_t* empty_mb,
                                     int num_k_tiles, int lane) {
  uint32_t fb = static_cast<uint32_t>(__cvta_generic_to_shared(full_mb));
  uint32_t eb = static_cast<uint32_t>(__cvta_generic_to_shared(empty_mb));
  PhaseTracker<NUM_STAGES> tr;
  constexpr int TILE_ELEMS = TILE_M * TILE_K;
  for (int k = 0; k < num_k_tiles; ++k) {
    int s = tr.get_stage();
    mbarrier_wait_parity(fb + s * 8, tr.get_phase());
    half* smem_ptr = reinterpret_cast<half*>(smem_a + s * TILE_BYTES);
    half* dst      = out + k * TILE_ELEMS;
    for (int i = lane; i < TILE_ELEMS; i += 32)
      dst[i] = smem_ptr[i];
    __syncwarp();
    if (lane == 0) mbarrier_arrive_nostate(eb + s * 8);
    tr.advance();
  }
}

#endif  // PL_AGENTIC_SM90A
