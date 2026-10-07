// 100_pipeline_blackwell.cuh -- canonical Blackwell pipeline scaffold.
//
// ARCH: sm_100a / sm_103a
//
// Faithful implementation of `knowledge/building_blocks/pipeline.md` --
// the FULL Blackwell warp-specialized pipeline mbarrier suite, with
// all five warp roles wired correctly, but with NO real bodies:
//   - MMA warp: per-stage full/empty handshake; NO tcgen05.mma.
//   - LOAD warp: throttle + per-stage empty/full handshake; NO TMA loads.
//   - SCHED warp: throttle + CLC try_cancel scheduler.
//   - IDLE warp: CLC empty handshake participant only.
//   - EPI warps (4): acc_full / acc_empty handshake; NO TMEM ld, NO
//     stmatrix, NO TMA stores.
//
// What this scaffold proves: every mbarrier in pipeline.md sec 2 can be
// init'd, cycled by its real producer/consumer warp roles, and torn
// down across `num_tiles` iterations + tcgen05 alloc/dealloc + cluster
// barriers without hanging. Equivalent to the "skeleton" K0 used in
// early bring-up before TMA / MMA / EPI bodies were added.
//
// USAGE: copy-as-template scaffold, NOT a #include target. New kernels
// start by copy-pasting this body, then replace the role stubs:
//   MMA arrive-only -> tcgen05.mma + final commit on acc_full
//   LOAD arrive-only -> TMA load + arrive_expect_tx
//   EPI arrive-only -> TMEM ld + cvt + stmatrix + TMA store
//
// SMEM barrier layout (per pipeline.md sec 2):
//   full[NUM_STAGES]            mainloop: load -> MMA
//   empty[NUM_STAGES]           mainloop: MMA -> load
//   acc_full[2]                 accumulator: MMA -> epi
//   acc_empty[2]                accumulator: epi -> MMA
//   clc_full[2]                 CLC: HW -> sched
//   clc_empty[2]                CLC: sched + consumers -> HW
//   throttle_full[2]            CLC throttle: load -> sched
//   throttle_empty[2]           CLC throttle: sched -> load
//   clc_response[8]             2 stages x 4 uint32 (16-byte CLC result)
//
// Source: knowledge/building_blocks/pipeline.md
// PTX:    9.7.15.16 (mbarrier.{init,arrive,arrive_tx,try_wait.parity}),
//         9.7.15.18 (clusterlaunchcontrol.try_cancel),
//         9.7.18    (tcgen05.{alloc,relinquish_alloc_permit,dealloc,commit}),
//         9.7.15.3  (barrier.cluster.{arrive,wait}),
//         9.7.15.4  (fence.mbarrier_init.release.cluster),
//         9.7.15.14 (griddepcontrol).
//
#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)

#include <cstdint>
#include <cuda_runtime.h>

#include "../primitives/0_tcgen05_alloc.cuh"
#include "../primitives/1_tcgen05_dealloc.cuh"
#include "../primitives/2_tcgen05_relinquish.cuh"
#include "../primitives/11_tcgen05_commit.cuh"
#include "../primitives/25_tma_async_group.cuh"
#include "../primitives/29_mbarrier_init.cuh"
#include "../primitives/30_mbarrier_arrive.cuh"
#include "../primitives/31_mbarrier_arrive_tx.cuh"
#include "../primitives/33_mbarrier_try_wait.cuh"
#include "../primitives/35_fence_mbarrier_init.cuh"
#include "../primitives/38_barrier_cluster.cuh"
#include "../primitives/44_elect_sync.cuh"
#include "../primitives/61_clc_try_cancel.cuh"
#include "../primitives/62_clc_query_cancel.cuh"
#include "../primitives/69_griddepcontrol.cuh"
#include "../primitives/70_smem_ptr.cuh"

#include "../composites/118_mbarrier_phase_tracking.cuh"
#include "../composites/119_pipeline_init_blackwell.cuh"
#include "../composites/106_clc_fetch_next_tile.cuh"

// ============================================================================
// Kernel: pipeline_blackwell_skeleton
//
// Template:
//   NUM_STAGES -- mainloop ring depth (typical 4-6).
// Args:
//   num_tiles  -- number of CLC-served tiles to cycle (per persistent
//                 cluster). Skeleton uses CLC try_cancel like K0; the
//                 host is expected to launch (gridDim.x = 2 * num_tiles)
//                 so all tiles get cancelled before exit.
//
// Cluster: __cluster_dims__(2, 1, 1)         -- 2-CTA cluster.
// Block:   256 threads = 8 warps per CTA.
// Warp roles (per pipeline.md sec 4.5 + warp-role pages):
//   warp 0     : MMA driver
//   warp 1     : SCHED (CLC + throttle producer)
//   warp 2     : LOAD (TMA stand-in + throttle consumer)
//   warp 3     : IDLE (CLC consumer participant only)
//   warps 4-7  : EPI (drain TMEM "accumulator" + ack acc_empty)
// ============================================================================

template <int NUM_STAGES>
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(256, 1)
pipeline_blackwell_skeleton_kernel() {
  constexpr int CLC_STAGES      = 2;
  constexpr int ACC_STAGES      = 2;
  constexpr int THROTTLE_STAGES = 2;

  // SMEM layout: barriers at offset 0, 16-byte CLC response slots after.
  extern __shared__ __align__(16) uint8_t pipe_skel_smem[];
  uint64_t* mbar           = reinterpret_cast<uint64_t*>(pipe_skel_smem);
  uint64_t* full           = mbar + 0;                                       // [NUM_STAGES]
  uint64_t* empty          = mbar + NUM_STAGES;                              // [NUM_STAGES]
  uint64_t* acc_full       = mbar + 2 * NUM_STAGES + 0;                      // [2]
  uint64_t* acc_empty      = mbar + 2 * NUM_STAGES + ACC_STAGES;             // [2]
  uint64_t* clc_full       = mbar + 2 * NUM_STAGES + 2 * ACC_STAGES;         // [2]
  uint64_t* clc_empty      = mbar + 2 * NUM_STAGES + 2 * ACC_STAGES + CLC_STAGES; // [2]
  uint64_t* throttle_full  = mbar + 2 * NUM_STAGES + 2 * ACC_STAGES + 2 * CLC_STAGES; // [2]
  uint64_t* throttle_empty = mbar + 2 * NUM_STAGES + 2 * ACC_STAGES + 2 * CLC_STAGES + THROTTLE_STAGES;
  uint32_t* clc_response   = reinterpret_cast<uint32_t*>(
                                 throttle_empty + THROTTLE_STAGES);          // [CLC_STAGES * 4]
  __shared__ uint32_t tmem_slot;

  const int peer = blockIdx.x & 1;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

  // --- MBAR INIT (per pipeline.md sec 4.1) -----------------------------
  BlackwellPipelineBars bars{
      full, empty,
      acc_full, acc_empty,
      clc_full, clc_empty,
      throttle_full, throttle_empty
  };
  if (threadIdx.x == 0) {
    for (int i = 0; i < CLC_STAGES * 4; ++i) clc_response[i] = 0;
  }
  pipeline_init_blackwell<NUM_STAGES, /*CTA_GROUP=*/2>(bars);

  // ====================================================================
  // Warp role dispatch (per pipeline.md sec 4.5)
  // ====================================================================
  if (warp == 0) {
    // TMEM alloc owned by MMA warp; signal EPI via named barrier 6.
    tcgen05_alloc<2>(smem_ptr_u32(&tmem_slot), /*nCols=*/512);
    asm volatile("bar.arrive 6, 160;\n" ::: "memory");
    uint32_t tmem_base = tmem_slot; (void)tmem_base;

    PhaseTracker<NUM_STAGES> full_ph;
    int      acc_prod_stage = 0;
    uint32_t acc_prod_phase = 0;
    int      clc_cons_stage = 0;
    uint32_t clc_cons_phase = 0;

    while (true) {
      mbarrier_wait_parity(smem_ptr_u32(&acc_empty[acc_prod_stage]),
                           acc_prod_phase ^ 1u);

      #pragma unroll 1
      for (int k = 0; k < 1; ++k) {
        const int s = full_ph.get_stage();
        mbarrier_wait_parity(smem_ptr_u32(&full[s]),
                             full_ph.get_phase());
        if (elect_one_sync()) {
          tcgen05_commit_multicast<2>(smem_ptr_u32(&empty[s]),
                                      /*ctamask=*/0x3);
        }
        full_ph.advance();
      }
      if (elect_one_sync()) {
        tcgen05_commit_multicast<2>(smem_ptr_u32(&acc_full[acc_prod_stage]),
                                    /*ctamask=*/0x3);
      }
      advance_stage_phase<ACC_STAGES>(acc_prod_stage, acc_prod_phase);

      ClcTileInfo next = clc_fetch_next_tile<1, 2, ClcRasterOrder::AlongN>(
          clc_full, clc_empty, clc_response,
          clc_cons_stage, clc_cons_phase, /*do_release=*/elect_one_sync());
      clc_fetch_next_tile_advance(clc_cons_stage, clc_cons_phase);
      if (!next.valid) break;
    }
  } else if (warp == 1) {
    griddepcontrol_wait();

    int      thr_cons_stage = 0;
    uint32_t thr_cons_phase = 0;
    int      clc_prod_stage = 0;
    uint32_t clc_prod_phase = 1;
    int      clc_cons_stage = 0;
    uint32_t clc_cons_phase = 0;

    while (true) {
      if (peer == 0) {
        mbarrier_wait_parity(smem_ptr_u32(&throttle_full[thr_cons_stage]),
                             thr_cons_phase);
        mbarrier_arrive_nostate(smem_ptr_u32(&throttle_empty[thr_cons_stage]));
        advance_stage_phase<THROTTLE_STAGES>(thr_cons_stage, thr_cons_phase);

        if (lane == 0) {
          mbarrier_wait_parity(smem_ptr_u32(&clc_empty[clc_prod_stage]),
                               clc_prod_phase);
        }
        __syncwarp();
        const uint32_t full_addr = smem_ptr_u32(&clc_full[clc_prod_stage]);
        clc_arrive_expect_tx_cluster(full_addr, /*tx_bytes=*/16);
        if (lane == 0) {
          const uint32_t resp_addr = smem_ptr_u32(&clc_response[clc_prod_stage * 4]);
          clc_try_cancel_multicast_all(resp_addr, full_addr);
        }
        advance_stage_phase<CLC_STAGES>(clc_prod_stage, clc_prod_phase);
      }

      ClcTileInfo next = clc_fetch_next_tile<1, 2, ClcRasterOrder::AlongN>(
          clc_full, clc_empty, clc_response,
          clc_cons_stage, clc_cons_phase, /*do_release=*/elect_one_sync());
      clc_fetch_next_tile_advance(clc_cons_stage, clc_cons_phase);
      if (!next.valid) break;
    }
    // Drain in-flight clc_empty arrives so the next launch's tcgen05.alloc
    // doesn't trip the alloc-state-machine guardrail
    // (kernels/gemm/sm100a/tcgen05_local_memory_trap.md).
    if (peer == 0) {
      for (int s = 0; s < CLC_STAGES; ++s) {
        if (lane == 0) {
          mbarrier_wait_parity(smem_ptr_u32(&clc_empty[clc_prod_stage]),
                               clc_prod_phase);
        }
        __syncwarp();
        advance_stage_phase<CLC_STAGES>(clc_prod_stage, clc_prod_phase);
      }
    }
  } else if (warp == 2) {
    EmptyPhaseTracker<NUM_STAGES> empty_ph;
    int      thr_prod_stage = 0;
    uint32_t thr_prod_phase = 1;
    int      clc_cons_stage = 0;
    uint32_t clc_cons_phase = 0;

    while (true) {
      if (peer == 0) {
        mbarrier_wait_parity(smem_ptr_u32(&throttle_empty[thr_prod_stage]),
                             thr_prod_phase);
        mbarrier_arrive_nostate(smem_ptr_u32(&throttle_full[thr_prod_stage]));
        advance_stage_phase<THROTTLE_STAGES>(thr_prod_stage, thr_prod_phase);
      }

      #pragma unroll 1
      for (int k = 0; k < 1; ++k) {
        const int s = empty_ph.get_stage();
        mbarrier_wait_parity(smem_ptr_u32(&empty[s]),
                             empty_ph.get_phase());
        // Real K0's TMA auto-arrives full[s] with expect_tx; here we
        // substitute a manual arrive, lane-gated to keep the count at 1
        // (arrive_count=1; an unguarded warp arrive would land 32x and
        // flip the bar's phase 32 times).
        if (elect_one_sync()) {
          mbarrier_arrive_nostate(smem_ptr_u32(&full[s]));
        }
        empty_ph.advance();
      }

      ClcTileInfo next = clc_fetch_next_tile<1, 2, ClcRasterOrder::AlongN>(
          clc_full, clc_empty, clc_response,
          clc_cons_stage, clc_cons_phase, /*do_release=*/elect_one_sync());
      clc_fetch_next_tile_advance(clc_cons_stage, clc_cons_phase);
      if (!next.valid) break;
    }
  } else if (warp == 3) {
    int      clc_cons_stage = 0;
    uint32_t clc_cons_phase = 0;
    while (true) {
      ClcTileInfo next = clc_fetch_next_tile<1, 2, ClcRasterOrder::AlongN>(
          clc_full, clc_empty, clc_response,
          clc_cons_stage, clc_cons_phase, /*do_release=*/elect_one_sync());
      clc_fetch_next_tile_advance(clc_cons_stage, clc_cons_phase);
      if (!next.valid) break;
    }
  } else {
    // Wait for MMA warp's tcgen05.alloc to publish tmem_base.
    asm volatile("bar.sync 6, 160;\n" ::: "memory");
    uint32_t tmem_base = tmem_slot; (void)tmem_base;

    int      acc_cons_stage = 0;
    uint32_t acc_cons_phase = 0;
    int      clc_cons_stage = 0;
    uint32_t clc_cons_phase = 0;
    while (true) {
      mbarrier_wait_parity(smem_ptr_u32(&acc_full[acc_cons_stage]),
                           acc_cons_phase);
      mbarrier_arrive_nostate(smem_ptr_u32(&acc_empty[acc_cons_stage]));
      advance_stage_phase<ACC_STAGES>(acc_cons_stage, acc_cons_phase);

      ClcTileInfo next = clc_fetch_next_tile<1, 2, ClcRasterOrder::AlongN>(
          clc_full, clc_empty, clc_response,
          clc_cons_stage, clc_cons_phase, /*do_release=*/(warp == 4 && lane == 0));
      clc_fetch_next_tile_advance(clc_cons_stage, clc_cons_phase);
      if (!next.valid) break;
    }
  }

  // --- Teardown (cluster-barrier variant) ------------------------------
  // Per-warp tail drains live in load_warp.md sec 6, mma_warp.md sec 6,
  // sched_warp.md sec 6. The cluster_arrive/wait below is the larger-
  // blast-radius alternative (see mma_warp.md sec 6.2 "Variant: cluster
  // barrier"). Drains in-flight TMA stores first so a straggler CTA's
  // pending stores don't race the cluster barrier.
  if (warp == 4 && lane == 0) {
    cp_async_bulk_wait_group<0>();
  }
  __syncthreads();
  barrier_cluster_arrive();
  barrier_cluster_wait();

  if (warp == 0) {
    tcgen05_relinquish_alloc_permit<2>();
    tcgen05_dealloc<2>(tmem_slot, /*nCols=*/512);
  }
  if (lane == 0) griddepcontrol_launch_dependents();
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
