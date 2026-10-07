#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// 90_mma_warp_blackwell.cuh -- Blackwell tcgen05.mma consumer warp building
// block.
//
// ARCH: sm_100a
//
// The "MMA warp" role for a Blackwell pipeline:
//   - tcgen05.alloc<1>     allocate 128 TMEM columns (warp-collective)
//   - tcgen05.relinquish_alloc_permit<1>  let other CTAs grab TMEM
//   - (issue tcgen05.mma)  caller-side MMAs land here in real kernels;
//                          this skeleton is the alloc/commit/wait/dealloc
//                          envelope around them
//   - tcgen05.commit<1>    produce a tracked completion on the mbarrier
//   - mbarrier.try_wait.parity   wait for the commit to land
//   - tcgen05.dealloc<1>   free TMEM
//
// Parametric form: `mma_warp_blackwell(ok, out_base)`. On success the kernel
// writes 1 to `*ok` (after the commit-wait completes) and the TMEM base
// returned by tcgen05.alloc to `*out_base` (so a caller can sanity-check
// that alloc succeeded).
//
// Issuer: 1 CTA, 128 threads. tcgen05.alloc / .relinquish issued by the
// first warp; commit + dealloc by lane 0 of warp 0.

// PTX:    9.7.18.10.10.1 (tcgen05.mma), 9.7.18.12.1 (commit), 9.7.15.3 (barrier.cluster)
//
#include <cstdint>
#include "../primitives/0_tcgen05_alloc.cuh"
#include "../primitives/1_tcgen05_dealloc.cuh"
#include "../primitives/2_tcgen05_relinquish.cuh"
#include "../primitives/3_tcgen05_mma_f16.cuh"
#include "../primitives/8_tcgen05_mma_idesc.cuh"
#include "../primitives/15_tcgen05_fence.cuh"
#include "../primitives/11_tcgen05_commit.cuh"
#include "../primitives/30_mbarrier_arrive.cuh"
#include "../primitives/33_mbarrier_try_wait.cuh"
#include "../primitives/44_elect_sync.cuh"
#include "../primitives/70_smem_ptr.cuh"
#include "../primitives/_warp_prof_noop.cuh"
#include "../composites/118_mbarrier_phase_tracking.cuh"
#include "../composites/104_acc_pipeline_2bank_blackwell.cuh"
#include "../composites/106_clc_fetch_next_tile.cuh"

/* ============================================================================
 * mma_warp_blackwell_block(wpc, slot, mbar, ok, out_base)
 *
 * Header-only __device__ form of the legacy alloc / commit / wait /
 * dealloc smoke envelope. Caller owns the SMEM `slot` (TMEM alloc
 * destination, 16-byte aligned) + `mbar` (commit / wait barrier).
 *
 * Caller responsibilities:
 *   - launch with 128 threads/CTA, single CTA (no cluster).
 *   - allocate `__shared__ uint32_t slot` and `__shared__ uint64_t mbar`,
 *     initialize slot to 0 and call mbarrier_init(mbar, 1) before this
 *     block.
 *   - issue __syncthreads BEFORE this block.
 *
 * The block runs a complete tcgen05 envelope:
 *   tcgen05_alloc<1>(slot, 128)
 *   tcgen05_relinquish_alloc_permit<1>()
 *   tcgen05_commit<1>(mbar)
 *   mbarrier_wait_parity(mbar, 0)
 *   tcgen05_dealloc<1>(tbase, 128)
 *
 * Writes the alloc base to `*out_base` (if non-null) and 1 to `*ok`
 * (if non-null) after the commit/wait chain completes.
 * ============================================================================ */
__device__ __forceinline__
void mma_warp_blackwell_block(WpCtx& wpc,uint32_t* slot,
                              uint64_t* mbar,
                              int* ok,
                              uint32_t* out_base) {
  if (threadIdx.x < 32) {
    tcgen05_alloc<1>(smem_ptr_u32(slot), 128);
    tcgen05_relinquish_alloc_permit<1>();
  }
  __syncthreads();
  uint32_t tbase = *slot;
  if (threadIdx.x == 0) {
    // (real consumer issues tcgen05.mma here -- placeholder for the
    // commit-and-wait test envelope)
    tcgen05_commit<1>(smem_ptr_u32(mbar));
    if (out_base != nullptr) *out_base = tbase;
  }
  mbarrier_wait_parity(smem_ptr_u32(mbar), 0);
  __syncthreads();
  if (threadIdx.x == 0) {
    if (ok != nullptr) *ok = 1;
    tcgen05_dealloc<1>(tbase, 128);
  }
}

/* ============================================================================
 * mma_warp_blackwell_1tile_2sm_bf16<NUM_STAGES, M_TILE_CLUSTER,
 *                                   N_TILE_CLUSTER, K_TILE>(wpc, ...)
 *
 * Production __device__ body for 2SM Blackwell BF16 GEMM kernels.
 * Issues the K-loop tcgen05.mma + commit per stage for ONE tile, plus
 * the composite-104 acc pipeline producer handshake (acquire on
 * acc_empty[bank] before, commit on acc_full[bank] after).
 *
 * Naming: <role>_<arch>_<#tiles>_<cluster>_<dtype>.
 *   role    = mma_warp           (tcgen05.mma producer; consumer of full_bar
 *                                  + producer of empty_bar via commit
 *                                  + producer of acc_full via composite 104)
 *   arch    = blackwell          (sm_100a / sm_103a)
 *   #tiles  = 1tile              (one tile's K-loop; caller drives outer)
 *   cluster = 2sm                (cta_group::2; cluster_dims(2,1,1))
 *   dtype   = bf16               (BF16 operands -> FP32 accumulator)
 *
 * Invocation: call with the WHOLE MMA warp of the leader CTA (all 32
 * lanes), gated to peer 0 only -- e.g.:
 *
 *     if (warp == 0 && peer == 0)
 *         mma_warp_blackwell_1tile_2sm_bf16<...>(wpc, ...);
 *
 *   The body needs all 32 lanes: the per-K-block mbarrier waits are
 *   warp-wide, and the single-thread tcgen05.mma.cta_group::2 + commit
 *   (PTX 9.7.18.5: one thread per cluster pair) are picked INSIDE via
 *   elect_one_sync(). Do NOT gate the call to lane 0.
 *
 * Args:
 *   desc_a, desc_b
 *     [NUM_STAGES]-length arrays of pre-built SMEM matrix descriptors
 *     (caller computes via primitive 42 build_smem_desc_blackwell).
 *     Descriptors don't change across tiles -- compute once at kernel
 *     entry and reuse.
 *
 *   full_bar, empty_bar
 *     [NUM_STAGES] mainloop pipeline mbars. Body waits full_bar[s]
 *     (consumer of load warp's TMA-delivered complete_tx + arrive),
 *     then issues tcgen05.commit.cta_group::2.multicast::cluster on
 *     empty_bar[s] (producer; load warp's consumer side waits this).
 *
 *   acc_bars, prod_state
 *     Composite 104 acc-pipeline (2-bank TMEM acc with peer-bit-mask
 *     CTA-scope arrive on acc_empty). Body issues:
 *       - acc_pipeline_2bank_producer_acquire(acc_bars, prod_state)
 *         BEFORE the K-loop (waits acc_empty[bank]).
 *       - acc_pipeline_2bank_producer_commit_cluster<2>(acc_bars, prod_state,
 *         ctamask=0x3) AFTER the K-loop (multicast complete_tx to
 *         acc_full[bank] in both peers' SMEM).
 *     prod_state.count advances once per call; persists across tiles
 *     in caller's scope.
 *
 *   tmem_base
 *     TMEM allocation base from tcgen05.alloc.cta_group::2 (nCols=512).
 *     Body computes `tmem_c = tmem_base + acc_stage * 256` for the
 *     2-bank acc pipeline (bank 0: TMEM cols 0..255; bank 1: 256..511).
 *
 *   K, full_ph
 *     Problem K dim (K_BLOCKS = K / K_TILE) and caller-owned
 *     PhaseTracker<NUM_STAGES> for full_bar phase (default-constructs
 *     to {stage=0, parity=0}). Body advances full_ph once per K-block.
 *     Persists across tiles in caller's scope.
 *
 * Notes:  descriptor-stride trick (stage0 + STAGE_DELTA avoids LDL/STL
 *         near tcgen05.alloc); TMEM bank stride = N_TILE_CLUSTER, <= 256.
 * PTX:    9.7.18.10.10.1 (tcgen05.mma.cta_group::2.kind::f16),
 *         9.7.18.12.1   (tcgen05.commit.cta_group::2.multicast::cluster),
 *         9.7.15.16.19  (mbarrier.try_wait.parity).
 * ============================================================================ */
// PEER1SYNC=true adds the cross-CTA peer1 A-ready wait (gather kernels, where
// A is gathered per-CTA so peer 0's MMA must also wait peer 1's A via the
// relay on peer1_done_bar). PEER1SYNC=false (default) is the non-gather path
// (A arrives 2SM-multicast on full_bar, no cross-CTA handshake). The
// peer1_done_* args are only read when PEER1SYNC=true.
template <int NUM_STAGES, int M_TILE_CLUSTER, int N_TILE_CLUSTER, int K_TILE,
          uint64_t A_STAGE_DELTA, uint64_t B_STAGE_DELTA, bool PEER1SYNC = false>
__device__ inline
void mma_warp_blackwell_1tile_2sm_bf16(WpCtx& wpc,
    uint64_t desc_a_stage0, uint64_t desc_b_stage0,
    uint64_t* full_bar, uint64_t* empty_bar,
    AccPipeline2BankBars acc_bars,
    AccPipeline2BankState& prod_state,
    uint32_t tmem_base, int K,
    PhaseTracker<NUM_STAGES>& full_ph,
    uint64_t* peer1_done_bar = nullptr,                   // PEER1SYNC only
    PhaseTracker<NUM_STAGES>* peer1_done_ph = nullptr) {  // PEER1SYNC only
  static_assert(N_TILE_CLUSTER <= 256, "N_TILE_CLUSTER <= 256: TMEM has 512 cols, 2-bank pipeline -> 256 cols per bank");
  constexpr int K_ATOMS_PER_TILE = K_TILE / 16;
  const int K_BLOCKS = K / K_TILE;
  const uint16_t ctamask = 0x3;
  const uint32_t idesc = make_idesc_bf16_f32(
      M_TILE_CLUSTER, N_TILE_CLUSTER, /*ta=*/false, /*tb=*/false);

  wp_begin(wpc, WP_MMA_WAIT_ACC);
  acc_pipeline_2bank_producer_acquire(acc_bars, prod_state);
  wp_end(wpc, WP_MMA_WAIT_ACC);
  const int acc_stage = acc_pipeline_2bank_state_index(prod_state);
  const uint32_t tmem_c = tmem_base + (uint32_t)(acc_stage * N_TILE_CLUSTER);

  for (int k = 0; k < K_BLOCKS; ++k) {
    const int s = full_ph.get_stage();
    // Wait for peer 0's own A + cta_group::2 B contributions.
    wp_begin(wpc, WP_MMA_WAIT_FULL);
    mbarrier_wait_parity(smem_ptr_u32(&full_bar[s]),
                         full_ph.get_phase());
    if constexpr (PEER1SYNC) {
      // Cross-CTA: peer 1's A gather landed; its MMA warp relayed
      // mbarrier.arrive.shared::cluster to peer 0's peer1_done_bar[s].
      mbarrier_wait_parity(smem_ptr_u32(&peer1_done_bar[s]),
                           peer1_done_ph->get_phase());
    }
    wp_end(wpc, WP_MMA_WAIT_FULL);
    // Compute per-stage descriptors on the fly (no local-memory array
    // -> no LDL/STL traffic that can stall the MMA pipeline and trip
    // the tcgen05.alloc HW guardrail).
    const uint64_t da_s = desc_a_stage0 + s * A_STAGE_DELTA;
    const uint64_t db_s = desc_b_stage0 + s * B_STAGE_DELTA;
    wp_begin(wpc, WP_MMA_ISSUE);
    if (elect_one_sync()) {
      #pragma unroll
      for (int ki = 0; ki < K_ATOMS_PER_TILE; ++ki) {
        const bool enable_d = (k != 0) || (ki != 0);
        tcgen05_mma_f16_ss<2>(tmem_c,
                              da_s + 2 * ki,
                              db_s + 2 * ki,
                              idesc, enable_d);
      }
      tcgen05_commit_multicast<2>(
          smem_ptr_u32(&empty_bar[s]), ctamask);
    }
    wp_end(wpc, WP_MMA_ISSUE);
    full_ph.advance();
    if constexpr (PEER1SYNC) peer1_done_ph->advance();
  }
  if (elect_one_sync()) {
    acc_pipeline_2bank_producer_commit_cluster<2>(acc_bars, prod_state, ctamask);
  }
}

/* ============================================================================
 * mma_warp_blackwell_1tile_1sm_bf16<NUM_STAGES, M_TILE_CLUSTER,
 *                                   N_TILE_CLUSTER, K_TILE,
 *                                   A_STAGE_DELTA, B_STAGE_DELTA>(wpc, ...)
 *
 * 1SM (cta_group::1) counterpart of `_1tile_2sm_bf16`. Used by grouped
 * GEMM phase-2 tail kernel where each tile is processed by a single CTA.
 *
 * Differences from _1tile_2sm_:
 *   - tcgen05.mma.cta_group::1 (not ::2; no multicast).
 *   - tcgen05.commit<1> on empty_bar (not commit_multicast<2>).
 *   - tcgen05_fence_before_thread_sync() before the acc_full commit so
 *     this warp's MMAs retire before the cross-warp handoff to EPI.
 *   - acc_full commit via direct tcgen05_commit<1> on acc_full[bank]
 *     (composite 104's producer_commit is multicast-only).
 *
 * Caller responsibility: produce_state_advance() after this call (state
 * is observable by the caller).
 *
 * PTX:    9.7.18.10.10.1 (tcgen05.mma.cta_group::1.kind::f16),
 *         9.7.18.12.1   (tcgen05.commit.cta_group::1).
 * ============================================================================ */
template <int NUM_STAGES, int M_TILE_CLUSTER, int N_TILE_CLUSTER, int K_TILE,
          uint64_t A_STAGE_DELTA, uint64_t B_STAGE_DELTA>
__device__ inline
void mma_warp_blackwell_1tile_1sm_bf16(WpCtx& wpc,
    uint64_t desc_a_stage0, uint64_t desc_b_stage0,
    uint64_t* full_bar, uint64_t* empty_bar,
    AccPipeline2BankBars acc_bars,
    AccPipeline2BankState& prod_state,
    uint32_t tmem_base, int K,
    PhaseTracker<NUM_STAGES>& full_ph) {
  static_assert(N_TILE_CLUSTER <= 256, "N_TILE_CLUSTER <= 256: TMEM has 512 cols, 2-bank pipeline -> 256 cols per bank");
  constexpr int K_ATOMS_PER_TILE = K_TILE / 16;
  const int K_BLOCKS = K / K_TILE;
  const uint32_t idesc = make_idesc_bf16_f32(
      M_TILE_CLUSTER, N_TILE_CLUSTER, /*ta=*/false, /*tb=*/false);

  acc_pipeline_2bank_producer_acquire(acc_bars, prod_state);
  const int bank = acc_pipeline_2bank_state_index(prod_state);
  const uint32_t tmem_c = tmem_base + (uint32_t)(bank * N_TILE_CLUSTER);

  for (int k = 0; k < K_BLOCKS; ++k) {
    const int s = full_ph.get_stage();
    mbarrier_wait_parity(smem_ptr_u32(&full_bar[s]),
                         full_ph.get_phase());
    const uint64_t da_s = desc_a_stage0 + s * A_STAGE_DELTA;
    const uint64_t db_s = desc_b_stage0 + s * B_STAGE_DELTA;
    if (elect_one_sync()) {
      #pragma unroll
      for (int ki = 0; ki < K_ATOMS_PER_TILE; ++ki) {
        const bool enable_d = (k != 0) || (ki != 0);
        tcgen05_mma_f16_ss<1>(tmem_c,
                              da_s + 2 * ki,
                              db_s + 2 * ki,
                              idesc, enable_d);
      }
      tcgen05_commit<1>(smem_ptr_u32(&empty_bar[s]));
    }
    full_ph.advance();
  }
  if (elect_one_sync()) {
    acc_pipeline_2bank_producer_commit_cta<1>(acc_bars, prod_state);
  }
}

/* ============================================================================
 * mma_warp_blackwell_ntiles_2sm_bf16<...>(wpc, ...)
 *
 * Self-driving do-while wrapper around the _1tile_ body for per-warp
 * persistent loops. Each iteration:
 *   1. fetch_next_tile (cluster-wide CLC handshake; ALL lanes).
 *   2. If leader CTA's lane 0: call _1tile_ body (K-loop + acc commit).
 *   3. Advance acc-pipeline state (all threads).
 *   4. Break if !next.valid.
 *
 * Threading model:
 *   Caller invokes from ALL 32 lanes of warp 0 in BOTH peer CTAs. Body
 *   internally:
 *     - All lanes (both CTAs): fetch_next_tile + acc cursor advance.
 *     - Lane 0 of leader CTA (peer 0): _1tile_ body call.
 *     - Other lanes / peers: no per-tile work; participate only in CLC
 *       handshake.
 * ============================================================================ */
// Both A and B are K-major (TMA loads BT in N x K row-major into SMEM,
// so K is the inner/contiguous dim in each B SMEM tile, same as A).
// Per-K-atom SMEM-descriptor delta is +2 (16 K-elements * 2 bytes = 32
// bytes; SMEM-desc start_address is in 16-byte units in low 14 bits).
// Tail at function exit (after the persistent loop):
//   1. Leader-CTA acc_empty drain (2 stages) -- proves both CTAs' EPI
//      done with TMEM.
//   2. tmem_dealloc_bar handshake -- symmetric arrive-on-peer +
//      wait-on-local; leader's arrive carries the "TMEM done" proof
//      to the follower.
//   3. tcgen05.relinquish_alloc_permit + tcgen05.dealloc<2>.
template <int NUM_STAGES, int M_TILE_CLUSTER, int N_TILE_CLUSTER, int K_TILE,
          uint64_t A_STAGE_DELTA, uint64_t B_STAGE_DELTA,
          int TMEM_NCOLS = 2 * N_TILE_CLUSTER,
          int CLUSTER_SHAPE_M = 1, int CLUSTER_SHAPE_N = 2,
          ClcRasterOrder ORDER = ClcRasterOrder::AlongN>
__device__ inline
void mma_warp_blackwell_ntiles_2sm_bf16(WpCtx& wpc,
    uint64_t desc_a_stage0, uint64_t desc_b_stage0,
    uint64_t* full_bar, uint64_t* empty_bar,
    AccPipeline2BankBars acc_bars,
    uint64_t* clc_full_bar, uint64_t* clc_empty_bar, uint32_t* clc_response,
    uint64_t* tmem_dealloc_bar,
    uint32_t tmem_base, int K, int peer, int lane,
    uint32_t* tmem_slot = nullptr) {  // if set, this block owns the TMEM alloc
  static_assert(N_TILE_CLUSTER <= 256, "N_TILE_CLUSTER <= 256: TMEM has 512 cols, 2-bank pipeline -> 256 cols per bank");
  AccPipeline2BankState prod_state = acc_pipeline_2bank_state_init();
  PhaseTracker<NUM_STAGES> full_ph;

  int      clc_cons_stage = 0;
  uint32_t clc_cons_phase = 0;

  (void)lane;
  // MMA warp owns the TMEM alloc (cta_group::2) when tmem_slot is given, and
  // signals EPI via named bar 6 that tmem_base is published. Symmetric with the
  // tail dealloc below. Legacy callers do the alloc and pass tmem_slot=nullptr.
  if (tmem_slot != nullptr) {
    wp_begin(wpc, WP_MMA_TMEM_2CTA_ALLOC);
    tcgen05_alloc<2>(smem_ptr_u32(tmem_slot), /*nCols=*/TMEM_NCOLS);
    asm volatile("bar.arrive 6, 160;\n" ::: "memory");
    tmem_base = *tmem_slot;
    wp_end(wpc, WP_MMA_TMEM_2CTA_ALLOC);
  }

  while (true) {
    if (peer == 0) {
      mma_warp_blackwell_1tile_2sm_bf16<
          NUM_STAGES, M_TILE_CLUSTER, N_TILE_CLUSTER, K_TILE,
          A_STAGE_DELTA, B_STAGE_DELTA>(wpc, desc_a_stage0, desc_b_stage0, full_bar, empty_bar,
          acc_bars, prod_state, tmem_base, K, full_ph);
    }
    acc_pipeline_2bank_state_advance(prod_state);
    wp_begin(wpc, WP_CLC_FETCH);
    ClcTileInfo next = clc_fetch_next_tile<
        CLUSTER_SHAPE_M, CLUSTER_SHAPE_N, ORDER>(
        clc_full_bar, clc_empty_bar, clc_response,
        clc_cons_stage, clc_cons_phase, /*do_release=*/elect_one_sync());
    wp_end(wpc, WP_CLC_FETCH);
    clc_fetch_next_tile_advance(clc_cons_stage, clc_cons_phase);
    if (!next.valid) break;
  }

  // --- Tail teardown ---------------------------------------------------
  // 1. Leader CTA: drain both acc_empty stages so EPI is proven done
  //    with TMEM on both CTAs. prod_state was advanced inside every
  //    iter, so it points at the next-to-acquire stage.
  wp_begin(wpc, WP_MMA_WAIT_ACC);  // tail: drain both acc banks
  if (peer == 0) {
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
      acc_pipeline_2bank_producer_acquire(acc_bars, prod_state);
      acc_pipeline_2bank_state_advance(prod_state);
    }
  }
  wp_end(wpc, WP_MMA_WAIT_ACC);

  // 2. Symmetric cross-CTA handshake on tmem_dealloc_bar. Leader's
  //    arrive happens after its drain, carrying the "TMEM done on both
  //    CTAs" guarantee to the follower. Peer-bit XOR (0x01000000) maps
  //    a local SMEM address to the peer CTA's view.
  constexpr uint32_t SM100_PEER_BIT = 0x01000000u;
  const uint32_t bar_local = smem_ptr_u32(tmem_dealloc_bar);
  const uint32_t bar_peer  = bar_local ^ SM100_PEER_BIT;
  mbarrier_arrive_cluster_default(bar_peer);
  wp_begin(wpc, WP_MMA_TMEM_2CTA_FREE);  // tail: cross-CTA TMEM-done handshake
  mbarrier_wait_parity(bar_local, /*phase=*/0);
  wp_end(wpc, WP_MMA_TMEM_2CTA_FREE);

  // 3. Both CTAs collectively dealloc.
  tcgen05_relinquish_alloc_permit<2>();
  tcgen05_dealloc<2>(tmem_base, /*nCols=*/TMEM_NCOLS);
}

/* ============================================================================
 * mma_warp_blackwell_ntiles_2sm_bf16_peer1relay<...>(wpc, ...)
 *
 * SonicMoE-style 2-CTA sync for the cp.async gather path. Pair with the load
 * warp `load_warp_blackwell_ntiles_2sm_bf16_cpasync`, whose peer-1 load
 * is fully fire-and-forget (no deferred wait, no peer1_done arrive).
 *
 *   peer 0 (leader)     : runs the matmul (_1tile_2sm_bf16<...,PEER1SYNC=true>), which waits
 *                         BOTH its own full_bar[s] (own A + 2SM B) AND
 *                         peer1_done_bar[s] (peer 1's A, relayed below).
 *   peer 1 (non-leader) : its MMA warp does NO matmul -- it just relays. Per
 *                         k-block it waits its OWN full_bar[s] (peer 1's cp.async
 *                         gather landed) then cross-CTA arrives peer 0's
 *                         peer1_done_bar[s].
 *
 * Moving the wait off the load warp lets the gather run NUM_STAGES deep (the
 * cp.async latency is hidden by the pipeline instead of stalling the load).
 * ============================================================================ */
template <int NUM_STAGES, int M_TILE_CLUSTER, int N_TILE_CLUSTER, int K_TILE,
          uint64_t A_STAGE_DELTA, uint64_t B_STAGE_DELTA,
          int TMEM_NCOLS = 2 * N_TILE_CLUSTER,
          int CLUSTER_SHAPE_M = 1, int CLUSTER_SHAPE_N = 2,
          ClcRasterOrder ORDER = ClcRasterOrder::AlongN>
__device__ inline
void mma_warp_blackwell_ntiles_2sm_bf16_peer1relay(WpCtx& wpc,
    uint64_t desc_a_stage0, uint64_t desc_b_stage0,
    uint64_t* full_bar, uint64_t* empty_bar,
    uint64_t* peer1_done_bar,
    AccPipeline2BankBars acc_bars,
    uint64_t* clc_full_bar, uint64_t* clc_empty_bar, uint32_t* clc_response,
    uint64_t* tmem_dealloc_bar,
    uint32_t tmem_base, int K, int peer, int lane) {
  static_assert(N_TILE_CLUSTER <= 256, "N_TILE_CLUSTER <= 256");
  AccPipeline2BankState prod_state = acc_pipeline_2bank_state_init();
  PhaseTracker<NUM_STAGES> full_ph;          // peer 0: own full
  PhaseTracker<NUM_STAGES> peer1_done_ph;    // peer 0: peer1_done
  PhaseTracker<NUM_STAGES> relay_full_ph;    // peer 1: own full (relay consumes)
  const int K_BLOCKS = K / K_TILE;
  constexpr uint32_t SM100_PEER_BIT = 0x01000000u;

  int      clc_cons_stage = 0;
  uint32_t clc_cons_phase = 0;
  (void)lane;

  while (true) {
    if (peer == 0) {
      mma_warp_blackwell_1tile_2sm_bf16<
          NUM_STAGES, M_TILE_CLUSTER, N_TILE_CLUSTER, K_TILE,
          A_STAGE_DELTA, B_STAGE_DELTA, /*PEER1SYNC=*/true>(wpc, desc_a_stage0, desc_b_stage0, full_bar, empty_bar,
          acc_bars, prod_state, tmem_base, K, full_ph,
          peer1_done_bar, &peer1_done_ph);
    } else {
      // peer 1 relay: wait own full[s] (A landed) -> arrive peer 0's peer1_done[s].
      wp_begin(wpc, WP_MMA_WAIT_FULL);
      for (int k = 0; k < K_BLOCKS; ++k) {
        const int s = relay_full_ph.get_stage();
        mbarrier_wait_parity(smem_ptr_u32(&full_bar[s]), relay_full_ph.get_phase());
        if (elect_one_sync()) {
          const uint32_t pd_peer0 =
              smem_ptr_u32(&peer1_done_bar[s]) ^ SM100_PEER_BIT;
          mbarrier_arrive_cluster_default(pd_peer0);
        }
        relay_full_ph.advance();
      }
      wp_end(wpc, WP_MMA_WAIT_FULL);
    }
    acc_pipeline_2bank_state_advance(prod_state);
    wp_begin(wpc, WP_CLC_FETCH);
    ClcTileInfo next = clc_fetch_next_tile<
        CLUSTER_SHAPE_M, CLUSTER_SHAPE_N, ORDER>(
        clc_full_bar, clc_empty_bar, clc_response,
        clc_cons_stage, clc_cons_phase, /*do_release=*/elect_one_sync());
    wp_end(wpc, WP_CLC_FETCH);
    clc_fetch_next_tile_advance(clc_cons_stage, clc_cons_phase);
    if (!next.valid) break;
  }

  // --- Tail teardown (identical to _ntiles_2sm_bf16_gather) ----------------
  wp_begin(wpc, WP_MMA_WAIT_ACC);  // tail: drain both acc banks
  if (peer == 0) {
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
      acc_pipeline_2bank_producer_acquire(acc_bars, prod_state);
      acc_pipeline_2bank_state_advance(prod_state);
    }
  }
  wp_end(wpc, WP_MMA_WAIT_ACC);
  wp_begin(wpc, WP_MMA_TMEM_2CTA_FREE);  // tail: cross-CTA TMEM-done handshake + dealloc
  const uint32_t bar_local = smem_ptr_u32(tmem_dealloc_bar);
  const uint32_t bar_peer  = bar_local ^ SM100_PEER_BIT;
  mbarrier_arrive_cluster_default(bar_peer);
  mbarrier_wait_parity(bar_local, /*phase=*/0);
  tcgen05_relinquish_alloc_permit<2>();
  tcgen05_dealloc<2>(tmem_base, /*nCols=*/TMEM_NCOLS);
  wp_end(wpc, WP_MMA_TMEM_2CTA_FREE);
}

/* ============================================================================
 * mma_warp_blackwell_ntiles_1sm_bf16<...>(wpc, ...)
 *
 * 1SM (cta_group::1, no cluster) counterpart of `_ntiles_2sm_bf16`. Same
 * CLC-driven persistent loop, no peer / no cluster handshake. Per-iter
 * K-loop delegates to `_1tile_1sm_bf16`.
 *
 * Tail: drain acc_empty + tcgen05.relinquish<1> + tcgen05.dealloc<1>.
 * ============================================================================ */
template <int NUM_STAGES, int M_TILE_CLUSTER, int N_TILE_CLUSTER, int K_TILE,
          uint64_t A_STAGE_DELTA, uint64_t B_STAGE_DELTA,
          int TMEM_NCOLS = 2 * N_TILE_CLUSTER,
          int CLUSTER_SHAPE_M = 1, int CLUSTER_SHAPE_N = 1,
          ClcRasterOrder ORDER = ClcRasterOrder::AlongN>
__device__ inline
void mma_warp_blackwell_ntiles_1sm_bf16(WpCtx& wpc,
    uint64_t desc_a_stage0, uint64_t desc_b_stage0,
    uint64_t* full_bar, uint64_t* empty_bar,
    AccPipeline2BankBars acc_bars,
    uint64_t* clc_full_bar, uint64_t* clc_empty_bar, uint32_t* clc_response,
    uint32_t tmem_base, int K, int lane) {
  static_assert(N_TILE_CLUSTER <= 256, "N_TILE_CLUSTER <= 256: TMEM has 512 cols, 2-bank pipeline -> 256 cols per bank");
  AccPipeline2BankState prod_state = acc_pipeline_2bank_state_init();
  PhaseTracker<NUM_STAGES> full_ph;

  int      clc_cons_stage = 0;
  uint32_t clc_cons_phase = 0;

  (void)lane;

  while (true) {
    mma_warp_blackwell_1tile_1sm_bf16<
        NUM_STAGES, M_TILE_CLUSTER, N_TILE_CLUSTER, K_TILE,
        A_STAGE_DELTA, B_STAGE_DELTA>(wpc, desc_a_stage0, desc_b_stage0, full_bar, empty_bar,
        acc_bars, prod_state, tmem_base, K, full_ph);
    acc_pipeline_2bank_state_advance(prod_state);
    ClcTileInfo next = clc_fetch_next_tile<
        CLUSTER_SHAPE_M, CLUSTER_SHAPE_N, ORDER, /*CTA_GROUP=*/1>(
        clc_full_bar, clc_empty_bar, clc_response,
        clc_cons_stage, clc_cons_phase, /*do_release=*/elect_one_sync());
    clc_fetch_next_tile_advance(clc_cons_stage, clc_cons_phase);
    if (!next.valid) break;
  }

  // Tail: drain both acc_empty stages so EPI is proven done with TMEM.
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    acc_pipeline_2bank_producer_acquire(acc_bars, prod_state);
    acc_pipeline_2bank_state_advance(prod_state);
  }

  tcgen05_relinquish_alloc_permit<1>();
  tcgen05_dealloc<1>(tmem_base, /*nCols=*/TMEM_NCOLS);
}

/* ============================================================================
 * mma_warp_blackwell_ntiles_2sm_bf16_natoms<N_ATOMS_PER_TILE>(...)
 *
 * Variant of mma_warp_blackwell_ntiles_2sm_bf16 that issues N_ATOMS_PER_TILE
 * atoms per K-tile in the N direction. Enables NTC > 256 (per-CTA N > 128)
 * by stitching multiple Layout A atoms (each M=256 N=128 cta_group::2).
 *
 * Per atom:
 *   tmem_c += n_atom * 128 cols (each atom covers 128 N-cols in TMEM).
 *   db += n_atom * (128 N-rows * K_TILE bytes / 16-byte units) = n_atom * 8*K_TILE.
 *
 * Backward compat: existing callers passing N_ATOMS_PER_TILE=1 get the
 * same behavior as the non-_natoms variant.
 * ============================================================================ */
template <int N_ATOMS_PER_TILE,
          int NUM_STAGES, int M_TILE_CLUSTER, int N_TILE_CLUSTER, int K_TILE,
          uint64_t A_STAGE_DELTA, uint64_t B_STAGE_DELTA,
          int TMEM_NCOLS = 2 * N_TILE_CLUSTER,
          int CLUSTER_SHAPE_M = 1, int CLUSTER_SHAPE_N = 2,
          ClcRasterOrder ORDER = ClcRasterOrder::AlongN>
__device__ inline
void mma_warp_blackwell_ntiles_2sm_bf16_natoms(
    uint64_t desc_a_stage0, uint64_t desc_b_stage0,
    uint64_t* full_bar, uint64_t* empty_bar,
    AccPipeline2BankBars acc_bars,
    uint64_t* clc_full_bar, uint64_t* clc_empty_bar, uint32_t* clc_response,
    uint64_t* tmem_dealloc_bar,
    uint32_t tmem_base, int K, int peer, int lane) {
  static_assert(N_TILE_CLUSTER <= 256, "N_TILE_CLUSTER <= 256: TMEM has 512 cols, 2-bank pipeline -> 256 cols per bank");
  AccPipeline2BankState prod_state = acc_pipeline_2bank_state_init();
  PhaseTracker<NUM_STAGES> full_ph;

  int      clc_cons_stage = 0;
  uint32_t clc_cons_phase = 0;

  constexpr int K_ATOMS_PER_TILE = K_TILE / 16;
  // Per-N-atom: each Layout A atom is M_TILE_CLUSTER x 128 (atom-N=128).
  // N_TILE_CLUSTER must == N_ATOMS_PER_TILE * 128 * 1 (peers don't split N
  // when N_ATOMS_PER_TILE > 1).
  // B SMEM is N-major: row stride = K_TILE * 2 bytes. Step per N-atom =
  // 128 rows * K_TILE * 2 bytes = 256 * K_TILE bytes = 16 * K_TILE 16-byte units.
  constexpr uint64_t B_N_ATOM_DELTA = (uint64_t)16 * (uint64_t)K_TILE;
  // TMEM lane offset per N-atom: 128 (since each atom covers 128 N-cols).
  constexpr uint32_t TMEM_N_ATOM_OFFSET = 128;
  const int K_BLOCKS = K / K_TILE;
  const uint16_t ctamask = 0x3;
  // The idesc encodes atom shape M=M_TILE_CLUSTER N=128. ONE atom at a
  // time covers atom-N=128 even though cluster covers N_TILE_CLUSTER.
  const uint32_t idesc = make_idesc_bf16_f32(
      M_TILE_CLUSTER, /*atom-N=*/128, /*ta=*/false, /*tb=*/false);
  (void)lane;

  while (true) {
    ClcTileInfo next = clc_fetch_next_tile<
        CLUSTER_SHAPE_M, CLUSTER_SHAPE_N, ORDER>(
        clc_full_bar, clc_empty_bar, clc_response,
        clc_cons_stage, clc_cons_phase, /*do_release=*/elect_one_sync());
    clc_fetch_next_tile_advance(clc_cons_stage, clc_cons_phase);

    if (peer == 0) {
      acc_pipeline_2bank_producer_acquire(acc_bars, prod_state);
      const int acc_stage = acc_pipeline_2bank_state_index(prod_state);
      const uint32_t tmem_c_base = tmem_base + (uint32_t)(acc_stage * N_TILE_CLUSTER);

      for (int k = 0; k < K_BLOCKS; ++k) {
        const int s = full_ph.get_stage();
        mbarrier_wait_parity(smem_ptr_u32(&full_bar[s]),
                             full_ph.get_phase());
        const uint64_t da_s = desc_a_stage0 + s * A_STAGE_DELTA;
        const uint64_t db_s = desc_b_stage0 + s * B_STAGE_DELTA;
        if (elect_one_sync()) {
          #pragma unroll
          for (int na = 0; na < N_ATOMS_PER_TILE; ++na) {
            const uint32_t tmem_c_na = tmem_c_base + na * TMEM_N_ATOM_OFFSET;
            const uint64_t db_na = db_s + na * B_N_ATOM_DELTA;
            #pragma unroll
            for (int ki = 0; ki < K_ATOMS_PER_TILE; ++ki) {
              const bool enable_d = (k != 0) || (ki != 0);
              tcgen05_mma_f16_ss<2>(tmem_c_na,
                                    da_s + 2 * ki,
                                    db_na + 2 * ki,
                                    idesc, enable_d);
            }
          }
          tcgen05_commit_multicast<2>(
              smem_ptr_u32(&empty_bar[s]), ctamask);
        }
        full_ph.advance();
      }
      if (elect_one_sync()) {
        acc_pipeline_2bank_producer_commit_cluster<2>(acc_bars, prod_state, ctamask);
      }
    }
    acc_pipeline_2bank_state_advance(prod_state);
    if (!next.valid) break;
  }

  // Tail teardown (matches main variant).
  if (peer == 0) {
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
      acc_pipeline_2bank_producer_acquire(acc_bars, prod_state);
      acc_pipeline_2bank_state_advance(prod_state);
    }
  }

  constexpr uint32_t SM100_PEER_BIT_NA = 0x01000000u;
  const uint32_t bar_local_na = smem_ptr_u32(tmem_dealloc_bar);
  const uint32_t bar_peer_na  = bar_local_na ^ SM100_PEER_BIT_NA;
  mbarrier_arrive_cluster_default(bar_peer_na);
  mbarrier_wait_parity(bar_local_na, /*phase=*/0);

  tcgen05_relinquish_alloc_permit<2>();
  tcgen05_dealloc<2>(tmem_base, /*nCols=*/TMEM_NCOLS);
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
