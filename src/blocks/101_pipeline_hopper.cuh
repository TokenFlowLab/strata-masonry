#pragma once
#if defined(PL_AGENTIC_SM90A)
// 101_pipeline_hopper.cuh -- full Hopper warp-specialized GEMM pipeline
//
// ARCH: sm_90a
//
// Crown-jewel block: 3-warpgroup Hopper pipeline (WG0=load, WG1=MMA,
// WG2=idle). The block wires load (#89), MMA (#92), and the standard
// mbarrier protocol, dumping the raw WG1 accumulator to
// out_accum[t_wg*32 + slot] for host verification.
//
// Composes #23 (tma_tensormap), #29-#33 (mbarrier), #37 (bar_sync),
// #44 (elect_sync), #43 (smem_desc_hopper), #53/#59 (wgmma family),
// #116 (tma_load_stage), #118 (phase tracking), #122 (k_loop_hopper),
// #84 (warp_dispatch).
//
// Block function:
//
//   template <int NUM_STAGES, int TILE_M, int TILE_N, int TILE_K,
//             int TILE_A_BYTES, int TILE_B_BYTES>
//   __device__ __forceinline__ void
//   pipeline_hopper_block(const CUtensorMap& tma_a, const CUtensorMap& tma_b,
//                         int num_k_tiles, int tile_m_idx, int tile_n_idx,
//                         char* smem_a, char* smem_b,
//                         uint64_t* full_mb, uint64_t* empty_mb,
//                         float* out_accum);
//
// Self-gated: the block uses hopper_apply_regbudget_3wg + role dispatch
// internally and routes load/MMA/idle warps appropriately. The caller
// launches with blockDim.x == 3 * THREADS_PER_WG (= 384).
//
// PTX:    9.7.17 (wgmma family), 9.7.15.16 (mbarrier), 9.7.10.28.5.3 (TMA)

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../primitives/23_tma_tensormap.cuh"
#include "../primitives/29_mbarrier_init.cuh"
#include "../primitives/30_mbarrier_arrive.cuh"
#include "../primitives/33_mbarrier_try_wait.cuh"
#include "../primitives/37_bar_sync.cuh"
#include "../primitives/44_elect_sync.cuh"
#include "../primitives/43_smem_desc_hopper.cuh"
#include "../primitives/53_wgmma_f16_ss.cuh"
#include "../primitives/59_wgmma_fence_commit_wait.cuh"
#include "../composites/116_tma_load_stage.cuh"
#include "../composites/118_mbarrier_phase_tracking.cuh"
#include "../composites/120_pipeline_init_hopper.cuh"
#include "../composites/122_k_loop_hopper.cuh"
#include "../composites/126_epi_subtile_hopper.cuh"
#include "../composites/127_epi_tma_store.cuh"
#include "../composites/84_warp_dispatch.cuh"

template <int NUM_STAGES, int TILE_M, int TILE_N, int TILE_K,
          int TILE_A_BYTES, int TILE_B_BYTES>
__device__ __forceinline__
void pipeline_hopper_block(const CUtensorMap& tma_a,
                           const CUtensorMap& tma_b,
                           int num_k_tiles,
                           int tile_m_idx, int tile_n_idx,
                           char* smem_a, char* smem_b,
                           uint64_t* full_mb, uint64_t* empty_mb,
                           float* __restrict__ out_accum) {
  constexpr int K_BLOCKS = TILE_K / 16;

  hopper_apply_regbudget_3wg();
  WarpRole role = hopper_role_dispatch_3wg();

  uint32_t fb = static_cast<uint32_t>(__cvta_generic_to_shared(full_mb));
  uint32_t eb = static_cast<uint32_t>(__cvta_generic_to_shared(empty_mb));

  if (threadIdx.x == 0) {
    #pragma unroll
    for (int s = 0; s < NUM_STAGES; ++s) {
      mbarrier_init(fb + s * 8, 1);
      mbarrier_init(eb + s * 8, THREADS_PER_WG);
    }
  }
  bar_sync<0>(blockDim.x);

  if (role == WarpRole::Load) {
    int warp_in_wg = get_warp_id() % WARPS_PER_WG;
    if (warp_in_wg == 0 && elect_one_sync()) {
      EmptyPhaseTracker<NUM_STAGES> tr;
      for (int k = 0; k < num_k_tiles; ++k) {
        int s = tr.get_stage();
        mbarrier_wait_parity(eb + s * 8, tr.get_phase());
        uint32_t sA = static_cast<uint32_t>(
            __cvta_generic_to_shared(smem_a + s * TILE_A_BYTES));
        uint32_t sB = static_cast<uint32_t>(
            __cvta_generic_to_shared(smem_b + s * TILE_B_BYTES));
        tma_load_stage_ab(&tma_a, &tma_b, sA, sB,
                          fb + s * 8, TILE_A_BYTES, TILE_B_BYTES,
                          /*A*/ k * TILE_K, tile_m_idx * TILE_M,
                          /*B*/ tile_n_idx * TILE_N, k * TILE_K);
        tr.advance();
      }
    }
  } else if (role == WarpRole::Mma && get_warpgroup_id() == 1) {
    float d[32] = {};
    int t_wg = threadIdx.x - THREADS_PER_WG;
    PhaseTracker<NUM_STAGES> tr;
    for (int k = 0; k < num_k_tiles; ++k) {
      int s = tr.get_stage();
      uint32_t sA = static_cast<uint32_t>(
          __cvta_generic_to_shared(smem_a + s * TILE_A_BYTES));
      uint32_t sB = static_cast<uint32_t>(
          __cvta_generic_to_shared(smem_b + s * TILE_B_BYTES));
      k_loop_hopper_f16_n64_stage<K_BLOCKS>(
          d, fb + s * 8, eb + s * 8, tr.get_phase(),
          sA, sB, TILE_K * 2, TILE_K * 2, k > 0);
      tr.advance();
    }
    #pragma unroll
    for (int slot = 0; slot < 32; ++slot)
      out_accum[t_wg * 32 + slot] = d[slot];
  }
}

#endif  // PL_AGENTIC_SM90A
