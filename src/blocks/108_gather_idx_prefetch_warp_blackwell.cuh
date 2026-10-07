#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// 108_gather_idx_prefetch_warp_blackwell.cuh -- Blackwell gather-index
// prefetch warp building block.
//
// ARCH: sm_100a
//
// Role for the cp.async gather-fused MoE FC1: a dedicated warp that
// runs the persistent CLC tile loop and, for each tile, stages that tile's
// M_TILE_PER_CTA gather row-indices (perm[]) into a small SMEM ring so the
// cp.async load warps never stall on perm[] GMEM latency. Mirrors QuACK's
// `a_prefetch` warp (quack/gemm_sm100.py:
// a_prefetch_warp_id / make_a_prefetch_pipeline).
//
// It also INHERITS the idle warp's CLC arrive-count role: it drives
// clc_fetch_next_tile every tile, contributing its 32 lanes' arrives to
// clc_empty[stage] so the cluster-wide handshake stays balanced when this
// warp replaces the former idle warp (see 107_idle_warp_blackwell.cuh).
//
// perm[] is K-INVARIANT (the 128 row indices for a tile do not depend on
// the K-block), so the indices are loaded ONCE per tile here and reused by
// the load warps across all K_BLOCKS -- removing redundant per-K-block
// perm reads.
//
// Index pipeline (this warp = PRODUCER, load warps = CONSUMER):
//   idx_smem  : int[NUM_IDX_STAGES * M_TILE_PER_CTA] ring of row indices.
//   idx_full  : mbar[NUM_IDX_STAGES] -- producer arrives when a slot is
//               filled; load warps wait on it before reading idx_smem.
//   idx_empty : mbar[NUM_IDX_STAGES] -- load warps arrive when done with a
//               slot; producer waits on it before refilling.
// Producer follows the established producer-side phase convention in this
// codebase (cf. the throttle handshake in 88_load_warp_blackwell.cuh):
// start stage=0, phase=1; wait idx_empty[stage], fill, arrive idx_full[stage].
//
// PTX:    9.7.15.16.16 (mbarrier.arrive), 9.7.21.5 (setmaxnreg.dec).
//
#include <cstdint>
#include <cuda_bf16.h>
#include "../primitives/44_elect_sync.cuh"
#include "../primitives/46_setmaxnreg.cuh"
#include "../primitives/30_mbarrier_arrive.cuh"
#include "../primitives/33_mbarrier_try_wait.cuh"
#include "../primitives/70_smem_ptr.cuh"
#include "../primitives/27_cp_async_cg.cuh"   // prefetch_global_l2
#include "../composites/118_mbarrier_phase_tracking.cuh"
#include "../composites/82_tile_rasterize.cuh"
#include "../composites/106_clc_fetch_next_tile.cuh"
#include "../primitives/_warp_prof_noop.cuh"

/* ============================================================================
 * gather_idx_prefetch_warp_blackwell_1sm<...>(wpc, ...)
 *
 * Production __device__ body for the gather-index prefetch warp (1SM /
 * cta_group::1). Persistent CLC loop; per tile it stages the tile's
 * perm[] row-indices into the idx_smem ring and hands the slot to the load
 * warps via idx_full/idx_empty.
 *
 * Threading model:
 *   Invoked from ALL 32 lanes of the prefetch warp. The 32 lanes
 *   cooperatively copy the M_TILE_PER_CTA indices (lane i, i+32, ...);
 *   one elected lane arrives idx_full and drives the CLC release.
 *
 * Template parameters:
 *   NUM_IDX_STAGES    depth of the index ring (>=2 to let this warp run a
 *                     tile ahead of the load warps).
 *   M_TILE_PER_CTA    rows per CTA tile (== indices per tile).
 *   M_TILE_CLUSTER    1SM: == M_TILE_PER_CTA (no peer split).
 *   PREFETCH_REG_BUDGET  setmaxnreg.dec budget (default 32; light warp).
 *   CLUSTER_SHAPE_M/N, ORDER, CTA_GROUP  CLC dispatch config (1SM ->
 *                     CTA_GROUP=1, CLUSTER_SHAPE_*=1).
 *
 * Args:
 *   idx_smem        ring of NUM_IDX_STAGES * M_TILE_PER_CTA ints.
 *   idx_full/idx_empty  index-pipeline mbars [NUM_IDX_STAGES].
 *   clc_full_bar/clc_empty_bar/clc_response  shared CLC dispatch state.
 *   perm            [M_padded] gather indices (packed row -> source row,
 *                   -1 for pad). Read-only.
 *   m_tile_remap    optional persistent-tile remap (nullptr = identity).
 *
 * Preconditions:
 *   - idx_full initialized arrive_count=1 (one producer lane arrives).
 *   - idx_empty initialized so all NUM_IDX_STAGES slots start AVAILABLE
 *     for the producer's first waits (consumer pre-arrive or init phase),
 *     arrive_count matching the load-side releaser (one elected lane).
 *   - clc_empty arrive_count includes this warp's 32 lanes (it replaces
 *     the idle warp 1:1 in the CLC handshake).
 * ============================================================================ */
template <int NUM_IDX_STAGES, int M_TILE_PER_CTA, int M_TILE_CLUSTER,
          int PREFETCH_REG_BUDGET = 32,
          int CLUSTER_SHAPE_M = 1, int CLUSTER_SHAPE_N = 1,
          ClcRasterOrder ORDER = ClcRasterOrder::AlongN,
          int CTA_GROUP = 1>
__device__ inline
void gather_idx_prefetch_warp_blackwell_1sm(WpCtx& wpc,
    int* __restrict__ idx_smem,
    uint64_t* idx_full, uint64_t* idx_empty,
    uint64_t* clc_full_bar, uint64_t* clc_empty_bar, uint32_t* clc_response,
    const int* __restrict__ perm,
    const int* __restrict__ m_tile_remap = nullptr) {
  setmaxnreg_dec<PREFETCH_REG_BUDGET>();
  static_assert(M_TILE_CLUSTER == M_TILE_PER_CTA,
                "1SM: cluster tile == per-CTA tile (no peer split)");
  static_assert(NUM_IDX_STAGES >= 2,
                "index ring needs >= 2 stages to run ahead of the load warps");

  const int lane = threadIdx.x & 31;

  // CLC consumer cursor (same as idle/load warps).
  int      clc_cons_stage = 0;
  uint32_t clc_cons_phase = 0;
  // Index-pipeline producer cursor (producer-side convention: phase starts 1).
  int      idx_stage = 0;
  uint32_t idx_phase = 1;

  auto remap = [&](int p1) -> int {
    return m_tile_remap ? m_tile_remap[p1] : p1;
  };

  int m_tile, n_tile;
  if constexpr (ORDER == ClcRasterOrder::AlongN) {
    m_tile = remap((int)blockIdx.y);
    n_tile = (int)blockIdx.x;
  } else {
    m_tile = remap((int)blockIdx.x);
    n_tile = (int)blockIdx.y;
  }
  (void)n_tile;  // index prefetch depends only on m_tile

  while (true) {
    const int packed_row_base = m_tile * M_TILE_CLUSTER;

    // Wait for this ring slot to be free, then stage the tile's indices.
    mbarrier_wait_parity(smem_ptr_u32(&idx_empty[idx_stage]), idx_phase);
    int* dst = idx_smem + idx_stage * M_TILE_PER_CTA;
    #pragma unroll
    for (int i = lane; i < M_TILE_PER_CTA; i += 32) {
      dst[i] = perm[packed_row_base + i];
    }
    __syncwarp();
    if (elect_one_sync()) {
      mbarrier_arrive_nostate(smem_ptr_u32(&idx_full[idx_stage]));
    }
    advance_stage_phase<NUM_IDX_STAGES>(idx_stage, idx_phase);

    // Advance CLC (also contributes this warp's arrives, the idle-warp role).
    ClcTileInfo next = clc_fetch_next_tile<
        CLUSTER_SHAPE_M, CLUSTER_SHAPE_N, ORDER, CTA_GROUP>(
        clc_full_bar, clc_empty_bar, clc_response,
        clc_cons_stage, clc_cons_phase, /*do_release=*/elect_one_sync());
    clc_fetch_next_tile_advance(clc_cons_stage, clc_cons_phase);
    if (!next.valid) break;
    m_tile = remap(next.m_tile);
    n_tile = next.n_tile;
  }
}

/* ============================================================================
 * gather_idx_prefetch_warp_blackwell_2sm<...>(wpc, ...)
 *
 * 2SM (cta_group::2) variant: each cluster CTA stages ITS OWN M_TILE_PER_CTA
 * index slice -- packed rows [m_tile*M_TILE_CLUSTER + peer*M_TILE_PER_CTA,
 * +M_TILE_PER_CTA) -- matching the per-CTA cp.async gather in
 * load_warp_blackwell_ntiles_2sm_bf16_cpasync. CLC is CTA_GROUP=2.
 * ============================================================================ */
// PREFETCH_A_L2=true: after staging a tile's gather indices, this warp also
// issues prefetch.global.L2 over each source row's K-extent, warming L2 ~2
// tiles ahead of the cp.async load warp (the MMA otherwise stalls on
// cp.async completion behind a DRAM-miss tail; prefetching the gather rows
// converts that tail to L2 hits). Needs A_base + K + the row-stride (= K).
template <int NUM_IDX_STAGES, int M_TILE_PER_CTA, int M_TILE_CLUSTER,
          int PREFETCH_REG_BUDGET = 32,
          int CLUSTER_SHAPE_M = 1, int CLUSTER_SHAPE_N = 2,
          ClcRasterOrder ORDER = ClcRasterOrder::AlongN,
          int CTA_GROUP = 2, bool PREFETCH_A_L2 = false>
__device__ inline
void gather_idx_prefetch_warp_blackwell_2sm(WpCtx& wpc,
    int* __restrict__ idx_smem,
    uint64_t* idx_full, uint64_t* idx_empty,
    uint64_t* clc_full_bar, uint64_t* clc_empty_bar, uint32_t* clc_response,
    const int* __restrict__ perm, int peer,
    const int* __restrict__ m_tile_remap = nullptr,
    const __nv_bfloat16* __restrict__ A_base = nullptr, int K = 0) {
  setmaxnreg_dec<PREFETCH_REG_BUDGET>();
  static_assert(NUM_IDX_STAGES >= 2, "index ring needs >= 2 stages");
  const int lane = threadIdx.x & 31;
  // bf16 per 128B L2 line = 64; one prefetch per line covers the row's K span.
  constexpr int ELEMS_PER_LINE = 64;
  int      clc_cons_stage = 0; uint32_t clc_cons_phase = 0;
  int      idx_stage = 0; uint32_t idx_phase = 1;
  auto remap = [&](int p1) -> int { return m_tile_remap ? m_tile_remap[p1] : p1; };

  int m_tile;
  if constexpr (ORDER == ClcRasterOrder::AlongN) m_tile = remap((int)blockIdx.y);
  else                                           m_tile = remap((int)blockIdx.x >> 1);

  while (true) {
    const int packed_row_base = m_tile * M_TILE_CLUSTER + peer * M_TILE_PER_CTA;
    // Stage this tile's gather row-indices (perm[]) into the SMEM ring slot.
    wp_begin(wpc, WP_PREFETCH_IDX);
    mbarrier_wait_parity(smem_ptr_u32(&idx_empty[idx_stage]), idx_phase);
    int* dst = idx_smem + idx_stage * M_TILE_PER_CTA;
    #pragma unroll
    for (int i = lane; i < M_TILE_PER_CTA; i += 32) dst[i] = perm[packed_row_base + i];
    __syncwarp();
    if (elect_one_sync()) mbarrier_arrive_nostate(smem_ptr_u32(&idx_full[idx_stage]));
    advance_stage_phase<NUM_IDX_STAGES>(idx_stage, idx_phase);
    wp_end(wpc, WP_PREFETCH_IDX);

    // Warm L2 for this tile's gather rows (issued AFTER handing off the index
    // slot, so the load warps are never gated on it).
    if constexpr (PREFETCH_A_L2) {
      wp_begin(wpc, WP_PREFETCH_L2);
      const int lines_per_row = K / ELEMS_PER_LINE;
      #pragma unroll 1
      for (int i = lane; i < M_TILE_PER_CTA; i += 32) {
        const int sr = perm[packed_row_base + i];
        if (sr >= 0) {
          const __nv_bfloat16* rowp = A_base + (size_t)sr * K;
          for (int c = 0; c < lines_per_row; ++c)
            prefetch_global_l2(rowp + c * ELEMS_PER_LINE);
        }
      }
      wp_end(wpc, WP_PREFETCH_L2);
    }

    wp_begin(wpc, WP_CLC_FETCH);
    ClcTileInfo next = clc_fetch_next_tile<
        CLUSTER_SHAPE_M, CLUSTER_SHAPE_N, ORDER, CTA_GROUP>(
        clc_full_bar, clc_empty_bar, clc_response,
        clc_cons_stage, clc_cons_phase, /*do_release=*/elect_one_sync());
    wp_end(wpc, WP_CLC_FETCH);
    clc_fetch_next_tile_advance(clc_cons_stage, clc_cons_phase);
    if (!next.valid) break;
    m_tile = remap(next.m_tile);
  }
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
