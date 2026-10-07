#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// 107_idle_warp_blackwell.cuh -- Blackwell idle warp building block.
//
// ARCH: sm_100a
//
// The "idle warp" role for independent per-warp persistent loops in
// 2SM Blackwell GEMM kernels: a warp that has no per-tile work
// of its own (no load, mma, epi, sched), but MUST participate in the
// cluster-wide CLC handshake (`fetch_next_tile`) so that
// `clc_empty[stage]` mbarrier arrive_count = #threads in cluster that
// participate is satisfied. Without this warp, an 8-warp design that
// only assigns roles to warps {0, 1, 2, 4, 5, 6, 7} would leave warp 3
// silent at every tile boundary, under-counting the clc_empty arrives.
//
// The idle warp's purpose is purely architectural -- contribute its 32
// lanes' arrives to the cluster-wide handshake on each tile boundary.
// `setmaxnreg_dec` is the only other lifecycle event (drops register
// budget so heavier MMA/epi consumers can `setmaxnreg_inc` from the
// CTA pool).

// Source: knowledge/building_blocks/idle_warp.md (primary).
//         knowledge/building_blocks/sched_warp.md (dispatch peer).
//         knowledge/instructions/tmem/tcgen05_tmem.md sec 8.3.
// PTX:    9.7.15.16.16 (mbarrier.arrive on cluster-shared mbar),
//         9.7.21.5     (setmaxnreg.dec.sync.aligned).
//
#include <cstdint>
#include "../primitives/44_elect_sync.cuh"
#include "../primitives/_warp_prof_noop.cuh"
#include "../primitives/46_setmaxnreg.cuh"
#include "../composites/106_clc_fetch_next_tile.cuh"

/* ============================================================================
 * idle_warp_blackwell_ntiles_2sm_bf16<int IDLE_REG_BUDGET = 24>(wpc, ...)
 *
 * Production __device__ body for the IDLE warp role. Calls
 * clc_fetch_next_tile() in a do-while loop; breaks on
 * `!next.valid`. Each iteration contributes 32 arrives on
 * clc_empty[stage] (lanes 0..31 of this warp).
 *
 * Naming: <role>_<arch>_<#tiles>_<cluster>_<dtype>.
 *   role    = idle_warp           (CLC handshake participant; no per-tile work)
 *   arch    = blackwell           (sm_100a / sm_103a)
 *   #tiles  = ntiles              (self-driven do-while; persistent)
 *   cluster = 2sm                 (cta_group::2)
 *   dtype   = bf16                (kernel-level dtype; idle warp is
 *                                  dtype-agnostic; suffix kept for
 *                                  consistency with sibling blocks)
 *
 * Threading model:
 *   Caller invokes from ALL 32 lanes of the idle warp in BOTH peer CTAs.
 *   The body has no internal lane gating -- every lane calls
 *   clc_fetch_next_tile each iter (so every lane arrives on
 *   clc_empty once per tile).
 *
 * Args (none beyond the CLC dispatch state):
 *   clc_full_bar, clc_empty_bar, clc_response
 *     [V41_CLC_STAGES=2] cluster-shared CLC mbars + 16-byte response
 *     slots. Same SMEM layout shared with sched_warp + load_warp_ntiles
 *     + mma_warp_ntiles + epi_warp_ntiles. Caller-owned init.
 *
 * Template parameters:
 *   IDLE_REG_BUDGET (default 24)
 *     Register budget the idle warp keeps after setmaxnreg.dec. Must
 *     be in [24, 256] and a multiple of 8 (PTX 9.7.21.5). Default 24
 *     is the minimum -- idle warp has no register pressure beyond
 *     bookkeeping for the CLC consumer cursor.
 *
 * Preconditions:
 *   - clc_full_bar / clc_empty_bar mbarrier_init done with the kernel-
 *     wide arrive_count for clc_empty (e.g., 512 = 8 warps * 32 lanes
 *     * 2 CTAs if all warps fetch).
 *   - Caller invokes from all 32 lanes of THIS warp.
 * ============================================================================ */
template <int IDLE_REG_BUDGET = 24,
          int CLUSTER_SHAPE_M = 1, int CLUSTER_SHAPE_N = 2,
          ClcRasterOrder ORDER = ClcRasterOrder::AlongN,
          int CTA_GROUP = 2, int CLC_STAGES = 2>
__device__ inline
void idle_warp_blackwell_ntiles_2sm_bf16(WpCtx& wpc,
    uint64_t* clc_full_bar, uint64_t* clc_empty_bar,
    uint32_t* clc_response) {
  setmaxnreg_dec<IDLE_REG_BUDGET>();

  int      clc_cons_stage = 0;
  uint32_t clc_cons_phase = 0;

  while (true) {
    wp_begin(wpc, WP_CLC_FETCH);
    ClcTileInfo next = clc_fetch_next_tile<
        CLUSTER_SHAPE_M, CLUSTER_SHAPE_N, ORDER, CTA_GROUP>(
        clc_full_bar, clc_empty_bar, clc_response,
        clc_cons_stage, clc_cons_phase, /*do_release=*/elect_one_sync());
    wp_end(wpc, WP_CLC_FETCH);
    // CLC ring depth must match the sched producer (default 2 keeps GEMM callers byte-identical;
    // FMHA persistent kernels use CLC_STAGES=4).
    clc_fetch_next_tile_advance<CLC_STAGES>(clc_cons_stage, clc_cons_phase);
    if (!next.valid) break;
  }
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
