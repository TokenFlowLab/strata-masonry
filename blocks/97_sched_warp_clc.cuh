// 97_sched_warp_clc.cuh -- Blackwell CLC (clusterlaunchcontrol) hardware
// scheduler building block. __cluster_dims__(1, 1, 1).
//
// ARCH: sm_100a
//
// The "sched warp" role using the Blackwell CLC primitive:
//   - mbarrier_init for the CLC try_cancel completion mbarrier
//   - clc_fetch_next_tile() (#81 composite): try_cancel + wait + query +
//     return ClcTile {canceled, ctaid_x, ctaid_y, ctaid_z}
//   - mbarrier.inval at end
//
// Parametric form: `sched_warp_clc(out_canceled)`. The kernel writes the
// returned `canceled` flag to `*out_canceled` so a caller can confirm CLC
// was reached without faulting.
//
// Issuer: 1 CTA, 32 threads. clc_fetch_next_tile is collective on warp 0;
// the rest of the CTA spins on the same mbarrier through the helper.
//
// PTX:    9.7.15.18 (clusterlaunchcontrol.try_cancel + .query_cancel)

#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)

#include <cstdint>
#include "../primitives/29_mbarrier_init.cuh"
#include "../primitives/30_mbarrier_arrive.cuh"
#include "../primitives/33_mbarrier_try_wait.cuh"
#include "../primitives/44_elect_sync.cuh"
#include "../primitives/69_griddepcontrol.cuh"
#include "../primitives/70_smem_ptr.cuh"
#include "../primitives/_warp_prof_noop.cuh"
#include "../composites/81_clc_scheduler_loop.cuh"
#include "../composites/106_clc_fetch_next_tile.cuh"

/* ============================================================================
 * sched_warp_clc_block(wpc, slot, mbar, out_canceled)
 *
 * Header-only __device__ form of the legacy CLC try_cancel + wait +
 * query_cancel smoke envelope. Caller owns SMEM (`slot[4]` for the
 * 16-byte response and `mbar` for the completion barrier).
 *
 * Caller responsibilities:
 *   - launch with 1x1x1 cluster, 32 threads/CTA.
 *   - allocate `__shared__ uint32_t slot[4]` and `__shared__ uint64_t mbar`,
 *     initialize slot to zeros and call mbarrier_init(mbar, 1).
 *   - issue __syncthreads BEFORE this block.
 *   - mbarrier.inval `mbar` AFTER this block returns.
 * ============================================================================ */
__device__ __forceinline__
void sched_warp_clc_block(WpCtx& wpc,uint32_t* slot,
                          uint64_t* mbar,
                          uint32_t* out_canceled) {
  ClcTile t = clc_fetch_next_tile(smem_ptr_u32(slot),
                                  smem_ptr_u32(mbar), 0);
  if (threadIdx.x == 0 && out_canceled != nullptr)
    *out_canceled = t.canceled;
}

/* ============================================================================
 * sched_warp_clc_blackwell_ntiles_2sm_bf16<>(wpc, ...)
 *
 * Production __device__ body for the SCHED warp role of a 2SM Blackwell
 * BF16 GEMM kernel with independent per-warp persistent loops.
 * Owns the CLC dispatch producer pipeline:
 *   - Wait clc_empty[prod_stage] (phase-1 init idiom: first 2 iters
 *     pass through without pre-arrives).
 *   - arrive_expect_tx clc_full[prod_stage] (lanes 0/1 -> CTA0/CTA1
 *     mbarrier.arrive.expect_tx via mapa::cluster).
 *   - try_cancel.async.multicast::cluster::all (one elected lane).
 *   - Advance prod cursor; consume one fetch via clc_fetch_next_tile;
 *     break if !valid.
 *
 * Naming: <role>_<arch>_<#tiles>_<cluster>_<dtype>.
 *   role    = sched_warp_clc      (CLC try_cancel-driven scheduler)
 *   arch    = blackwell           (sm_100a / sm_103a)
 *   #tiles  = ntiles              (self-driven do-while; persistent)
 *   cluster = 2sm                 (cta_group::2)
 *   dtype   = bf16                (kernel-level dtype; sched warp itself
 *                                  is dtype-agnostic; suffix kept for
 *                                  consistency with sibling blocks)
 *
 * Threading model:
 *   Caller invokes from ALL 32 lanes of the sched warp in BOTH peer CTAs.
 *   Body internally:
 *     - Leader CTA only does the producer side (wait clc_empty + arrive_
 *       expect_tx + try_cancel). Follower CTA's sched warp falls
 *       through.
 *     - All 32 lanes (both CTAs) call clc_fetch_next_tile each iter --
 *       contributes 64 arrives on clc_empty per iter (32 lanes x 2
 *       CTAs).
 *
 * Args:
 *   clc_full_bar, clc_empty_bar
 *     2-stage CLC pipeline mbars in cluster shared SMEM.
 *
 *   clc_response
 *     [2 * 4 u32]: 16 bytes per stage for the try_cancel response.
 *
 *   cluster_rank, lane
 *     `cluster_rank == 0` is the leader CTA; only leader's sched warp
 *     issues CLC. `lane` is intra-warp lane index.
 *
 * PTX:    9.7.15.18 (clusterlaunchcontrol.try_cancel.async.multicast::
 *                    cluster::all),
 *         9.7.15.16.16 (mbarrier.arrive.expect_tx + scope/sem defaults),
 *         9.7.15.16.19 (mbarrier.try_wait.parity).
 * ============================================================================ */
template <bool USE_GRIDDEP_WAIT = false,
          int CLUSTER_SHAPE_M = 1, int CLUSTER_SHAPE_N = 2,
          ClcRasterOrder ORDER = ClcRasterOrder::AlongN,
          int CTA_GROUP = 2, bool SUSPEND = false, int CLC_STAGES = 2>
__device__ inline
void sched_warp_clc_blackwell_ntiles_2sm_bf16(WpCtx& wpc,
    uint64_t* clc_full_bar, uint64_t* clc_empty_bar,
    uint32_t* clc_response,
    uint64_t* throttle_full, uint64_t* throttle_empty,
    int cluster_rank, int lane) {
  // Wait for the prior kernel in this stream to drain + commit. Pairs
  // with the host-side cudaLaunchAttributeProgrammaticStreamSerialization
  // attribute (set on each launch) so the dependent grid's setup
  // (mbar init, tcgen05.alloc, throttle handshakes) overlaps with the
  // prerequisite's tail (final TMA stores, dealloc). Without the host
  // attribute the wait is a no-op. Issue on lane 0 of the sched warp
  // -- griddepcontrol is per-thread blocking, so any single thread is
  // enough; the rest of the warp converges below at __syncwarp() on
  // the first iter.
  if constexpr (USE_GRIDDEP_WAIT) {
    if (lane == 0) griddepcontrol_wait();
  }

  int      clc_prod_stage = 0;
  uint32_t clc_prod_phase = 1;
  int      clc_cons_stage = 0;
  uint32_t clc_cons_phase = 0;
  int      thr_cons_stage = 0;
  uint32_t thr_cons_phase = 0;

  while (true) {
    if (cluster_rank == 0) {
      // Throttle handshake: wait throttle_full, arrive throttle_empty.
      // ALL 32 lanes participate so the warp converges before the CLC
      // issue.
      wp_begin(wpc, WP_SCHED_WAIT_THROTTLE);
      if constexpr (SUSPEND) mbarrier_wait_parity_suspend(smem_ptr_u32(&throttle_full[thr_cons_stage]), thr_cons_phase);
      else                   mbarrier_wait_parity(smem_ptr_u32(&throttle_full[thr_cons_stage]), thr_cons_phase);
      wp_end(wpc, WP_SCHED_WAIT_THROTTLE);
      mbarrier_arrive_nostate(smem_ptr_u32(&throttle_empty[thr_cons_stage]));
      advance_stage_phase<2>(thr_cons_stage, thr_cons_phase);

      wp_begin(wpc, WP_SCHED_WAIT_CLC);
      if (lane == 0) {
        const uint32_t empty_addr = smem_ptr_u32(
            &clc_empty_bar[clc_prod_stage]);
        if constexpr (SUSPEND) mbarrier_wait_parity_suspend(empty_addr, clc_prod_phase);
        else                   mbarrier_wait_parity(empty_addr, clc_prod_phase);
      }
      __syncwarp();
      wp_end(wpc, WP_SCHED_WAIT_CLC);
      wp_begin(wpc, WP_SCHED_ISSUE);
      const uint32_t full_addr = smem_ptr_u32(
          &clc_full_bar[clc_prod_stage]);
      if constexpr (CTA_GROUP == 1) {
        clc_arrive_expect_tx_cta(full_addr, /*tx_bytes=*/16);
      } else {
        clc_arrive_expect_tx_cluster(full_addr, /*tx_bytes=*/16);
      }
      if (lane == 0) {
        const uint32_t resp_addr = smem_ptr_u32(
            &clc_response[clc_prod_stage * 4]);
        if constexpr (CTA_GROUP == 1) {
          // 1SM (no cluster): non-multicast variant.
          clc_try_cancel_async(resp_addr, full_addr);
        } else {
          clc_try_cancel_multicast_all(resp_addr, full_addr);
        }
      }
      advance_stage_phase<CLC_STAGES>(clc_prod_stage, clc_prod_phase);
      wp_end(wpc, WP_SCHED_ISSUE);
    }

    wp_begin(wpc, WP_CLC_FETCH);
    ClcTileInfo next = clc_fetch_next_tile<
        CLUSTER_SHAPE_M, CLUSTER_SHAPE_N, ORDER, CTA_GROUP, SUSPEND>(
        clc_full_bar, clc_empty_bar, clc_response,
        clc_cons_stage, clc_cons_phase, /*do_release=*/elect_one_sync());
    wp_end(wpc, WP_CLC_FETCH);
    clc_fetch_next_tile_advance<CLC_STAGES>(clc_cons_stage, clc_cons_phase);
    if (!next.valid) break;
  }

  // SCHED tail drain: wait clc_empty for CLC_STAGES (=2) more rounds.
  // Drains any in-flight consumer arrives that may not have landed yet
  // when the main loop broke, ensuring CLC HW state is clean at kernel
  // exit. Without this, the next launch's tcgen05.alloc may trip the
  // alloc-state-machine guardrail observed in multi-launch sanitizer
  // tests.
  if (cluster_rank == 0) {
    for (int s = 0; s < CLC_STAGES; ++s) {
      wp_begin(wpc, WP_SCHED_WAIT_CLC);  // tail drain: clc_empty
      if (lane == 0) {
        const uint32_t empty_addr = smem_ptr_u32(
            &clc_empty_bar[clc_prod_stage]);
        if constexpr (SUSPEND) mbarrier_wait_parity_suspend(empty_addr, clc_prod_phase);
        else                   mbarrier_wait_parity(empty_addr, clc_prod_phase);
      }
      __syncwarp();
      wp_end(wpc, WP_SCHED_WAIT_CLC);
      advance_stage_phase<CLC_STAGES>(clc_prod_stage, clc_prod_phase);
    }
  }
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
