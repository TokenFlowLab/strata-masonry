#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// 88_load_warp_blackwell.cuh -- Blackwell 2SM TMA producer warp building
// block. __cluster_dims__(2, 1, 1), __launch_bounds__(128, 1).
//
// ARCH: sm_100a
//
// The "load warp" role for a 2SM Blackwell pipeline:
//   - setmaxnreg.dec<40>      free registers for the heavier consumer warps
//   - tma_prefetch_2d          warm L2 with the first stages
//   - per stage: mbarrier_arrive_expect_tx on full[s], cp.async.bulk.tensor
//     .2d.cta_group::2 (#19 tma_load_2d_2sm) into smemA[s], advance phase
//   - on the last tile: writes the final SMEM tile to `out` so a caller
//     can verify the load (caller passes a 16x16-float-shaped buffer).
//
// Parametric form: `load_warp_blackwell(tma, out, k_tiles)`. The kernel is
// callable from any test/launcher; the test in tests/88_*.cu just provides
// the tensormap and the verification buffer.
//
// Issuer: cluster of 2 CTAs, each 128 threads. Only thread 0 of CTA 0 issues
// the TMA load + arrive.expect_tx; the other CTA's TMA-side participation is
// implicit via the cta_group::2 modifier.

// PTX:    9.7.10.28.5.3 (TMA cta_group::2), 9.7.15.16.14 (expect_tx)
//
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "../primitives/18_tma_load.cuh"
#include "../primitives/19_tma_load_2sm.cuh"
#include "../primitives/21_tma_load_prefetch.cuh"
#include "../primitives/73_tma_load_2d_gather4.cuh"
#include "../primitives/68_l2cache_policy.cuh"
#include "../primitives/29_mbarrier_init.cuh"
#include "../primitives/30_mbarrier_arrive.cuh"
#include "../primitives/31_mbarrier_arrive_tx.cuh"
#include "../primitives/33_mbarrier_try_wait.cuh"
#include "../primitives/44_elect_sync.cuh"
#include "../primitives/46_setmaxnreg.cuh"
#include "../primitives/70_smem_ptr.cuh"
#include "../primitives/_warp_prof_noop.cuh"
#include "../primitives/27_cp_async_cg.cuh"
#include "../primitives/28_cp_async_commit_wait.cuh"
#include "../primitives/41_smem_swizzle.cuh"
#include "../primitives/67_mapa.cuh"
#include "../composites/118_mbarrier_phase_tracking.cuh"
#include "../composites/83_grouped_gemm_tile_map.cuh"
#include "../composites/106_clc_fetch_next_tile.cuh"
#include "../composites/109_fastdivmod.cuh"
#include "../composites/110_fmha_workitem_decode.cuh"

/* ============================================================================
 * load_warp_blackwell_block<NUM_STAGES, TILE_FLOATS>(wpc, ...)
 *
 * Header-only __device__ form of the legacy single-tile A-only smoke
 * body. Caller owns SMEM + mbar arrays + init + fence + sync; this block
 * just runs the producer-consumer per-stage TMA load loop and (on the
 * last tile) writes the stage's SMEM contents through `out_last_tile`.
 *
 * Caller responsibilities:
 *   - launch with cluster_dims(2, 1, 1), 128 threads/CTA.
 *   - allocate `smem_a[NUM_STAGES * TILE_FLOATS]`, `full_bar[NUM_STAGES]`,
 *     `empty_bar[NUM_STAGES]` in SMEM (16/128-byte aligned per type).
 *   - mbarrier_init each full / empty slot with arrive_count=1.
 *   - issue `fence.mbarrier_init.release.cluster` + `__syncthreads`.
 *   - mbarrier.inval each slot AFTER this block returns.
 *
 * Args:
 *   tma_A           CUDA tensor map for the A operand (FP32 in the test
 *                   harness; the block is dtype-agnostic per the
 *                   TILE_FLOATS template -- TILE_FLOATS counts elements,
 *                   not bytes).
 *   smem_a          NUM_STAGES * TILE_FLOATS contiguous floats in this
 *                   CTA's shared memory.
 *   full_bar        NUM_STAGES uint64_t mbarriers (producer signals the
 *                   stage is loaded). Caller-init'd, arrive_count=1.
 *   empty_bar       NUM_STAGES uint64_t mbarriers (reserved for the
 *                   symmetric consumer; not consumed in this body --
 *                   present so the contract matches the production
 *                   load_warp signature).
 *   k_tiles         number of K tiles to load.
 *   out_last_tile   testability hook: on the final iter, the leader
 *                   CTA's first TILE_FLOATS lanes write SMEM stage s
 *                   through this pointer. Pass nullptr in production.
 * ============================================================================ */
template <int NUM_STAGES, int TILE_FLOATS>
__device__ __forceinline__
void load_warp_blackwell_block(WpCtx& wpc,const CUtensorMap& tma_A,
                               float* smem_a,
                               uint64_t* full_bar,
                               uint64_t* empty_bar,
                               int k_tiles,
                               float* out_last_tile) {
  (void)empty_bar;

  setmaxnreg_dec<40>();

  // Prefetch first NUM_STAGES tiles into L2.
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    #pragma unroll
    for (int s = 0; s < NUM_STAGES; ++s)
      tma_prefetch_2d(&tma_A, 0, s * 16);
  }

  MbarrierPhaseTracker<NUM_STAGES> ph; ph.init();
  for (int k = 0; k < k_tiles; ++k) {
    int s = ph.stage();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar[s]),
                                TILE_FLOATS * sizeof(float));
      uint32_t mbar = tma_peer_bit_mask(smem_ptr_u32(&full_bar[s]));
      tma_load_2d_2sm(smem_ptr_u32(&smem_a[s * TILE_FLOATS]),
                      &tma_A, mbar, 0, k * 16);
    }
    if (blockIdx.x == 0)
      mbarrier_wait_parity(smem_ptr_u32(&full_bar[s]),
                           ph.current_phase());
    if (out_last_tile != nullptr && blockIdx.x == 0 && k == k_tiles - 1
        && threadIdx.x < TILE_FLOATS) {
      out_last_tile[threadIdx.x] = smem_a[s * TILE_FLOATS + threadIdx.x];
    }
    ph.advance();
  }
}

/* ============================================================================
 * load_warp_blackwell_1tile_2sm_bf16<NUM_STAGES, M_TILE_PER_CTA,
 *                                   N_TILE_PER_CTA, K_TILE>(wpc, ...)
 *
 * Production __device__ body for 2SM Blackwell BF16 GEMM kernels.
 * Naming: <role>_<arch>_<#tiles>_<cluster>_<dtype>.
 *   role    = load_warp                 (TMA producer)
 *   arch    = blackwell                 (sm_100a; sm_103a compatible)
 *   #tiles  = 1tile                     (this function loads ONE tile;
 *                                        caller drives the outer loop --
 *                                        works for both shared-outer-
 *                                        loop kernels (kernel-level
 *                                        `while`) AND per-warp
 *                                        self-driven kernels (each
 *                                        warp body contains its own
 *                                        do-while). Sibling `_ntiles_*`
 *                                        has the do-while INSIDE.)
 *   cluster = 2sm                       (cta_group::2; cluster_dims(2,1,1))
 *   dtype   = bf16                      (sizeof(__nv_bfloat16) per element)
 * Future siblings in this file follow the same axes:
 *   load_warp_blackwell_ntiles_2sm_bf16 -- self-driven do-while wrapper.
 *   load_warp_blackwell_1tile_1sm_bf16  -- cta_group::1 variant.
 *   load_warp_blackwell_1tile_2sm_fp8   -- fp8 dtype variant.
 * One tile's worth of TMA loads of A and B operands (BF16 elements)
 * into NUM_STAGES SMEM stages. Composes primitives + composite 118
 * already included by this file.
 *
 * Generalizations vs the existing __global__ test in this same file:
 *   - parametric NUM_STAGES, M_TILE_PER_CTA, N_TILE_PER_CTA, K_TILE
 *     (template) -- replaces hardcoded 2 stages / 16x16.
 *   - both A and B operands -- the test only exercises A.
 *   - caller-owned SMEM, mbars, and EmptyPhaseTracker (so phase
 *     persists across persistent-loop tiles).
 *
 * Dtype contract (encoded in the `_bf16` suffix): SMEM stages are
 * sized as M_TILE_PER_CTA * K_TILE * sizeof(__nv_bfloat16) bytes per
 * A stage and N_TILE_PER_CTA * K_TILE * sizeof(__nv_bfloat16) per B
 * stage. FP16 happens to match (sizeof(half) == 2) but callers
 * should rename / specialize before relying on it. FP8 / FP4 require
 * a separate `_fp8` / `_fp4` body (different byte widths).
 *
 * Roles on bars (canonical 2SM load-warp pattern):
 *   - PRODUCER on full_bar[s]: arrives expect_tx + tma_load_2d_2sm
 *     (whose .cta_group::2 multicast complete_tx delivers bytes).
 *   - CONSUMER on empty_bar[s]: waits parity for MMA to release.
 *     Phase-1 init idiom (EmptyPhaseTracker starts at parity 1)
 *     lets the first NUM_STAGES iters pass through without
 *     pre-arrives.
 *
 * Caller responsibilities (NOT in body):
 *   - mbarrier_init + fence.mbarrier_init.release.cluster (kernel
 *     entry).
 *   - mbarrier_inval (kernel exit, if needed).
 *   - setmaxnreg_dec (load warp's reg-budget; once-per-warp, not per
 *     tile -- planned for load_warp_persistent_blackwell_bf16 wrapper).
 *   - tma_prefetch_2d (L2 warm-up policy; once at start, not per
 *     tile).
 *   - persistent loop / fetch_next_tile / tail drain (kernel- or
 *     persistent-wrapper-level).
 *
 * Preconditions (load-bearing assumptions; violating these is UB):
 *   - empty_ph initial parity == 1. EmptyPhaseTracker default-
 *     constructs to {stage=0, parity=1}, so the canonical
 *     `EmptyPhaseTracker<NUM_STAGES> empty_ph;` declaration
 *     satisfies this. The body's first NUM_STAGES iters rely on
 *     parity=1 mismatching the freshly-init'd empty_bar's incomplete
 *     phase 0 (try_wait.parity returns true -> no-op pass-through).
 *     If the caller initializes empty_ph with parity=0 instead, the
 *     first wait will block forever (or, if pre-arrives populate the
 *     bars, advance the wrong stage). The phase-1 contract MUST be
 *     preserved when the tracker is shared across persistent-loop
 *     tiles.
 *   - full_bar / empty_bar mbarrier_init done with arrive_count=1
 *     each, before kernel-entry cluster sync.
 *   - smem_a / smem_b base pointers 128-byte aligned.
 *
 * Args:
 *   tma_a, tma_b
 *     CUDA tensor map handles for the A and B GMEM operands. The
 *     body issues `tma_load_2d_2sm.cta_group::2` against these. The
 *     tensormaps must be set up by the host (per the kernel's
 *     tile-shape contract); this body does not modify them.
 *
 *   smem_a, smem_b
 *     Byte pointers to the base of each operand's NUM_STAGES-deep
 *     contiguous buffer in this CTA's shared memory. Each stage
 *     holds one per-CTA tile of operand data:
 *       - smem_a: NUM_STAGES * (M_TILE_PER_CTA * K_TILE *
 *                 sizeof(__nv_bfloat16)) bytes total. Stage s
 *                 starts at byte offset
 *                 s * M_TILE_PER_CTA * K_TILE * sizeof(__nv_bfloat16).
 *       - smem_b: NUM_STAGES * (N_TILE_PER_CTA * K_TILE *
 *                 sizeof(__nv_bfloat16)) bytes total. Same per-stage
 *                 indexing.
 *     Caller owns the layout. Buffers must be 128-byte aligned for
 *     B128 swizzle compatibility with the consumer's tcgen05.mma
 *     SMEM descriptors (PTX 9.7.18.4.1). The body issues TMA
 *     writes to these regions; a downstream MMA warp reads them via
 *     SMEM descriptors that must point to the same base addresses.
 *
 *   full_bar, empty_bar
 *     Arrays of NUM_STAGES uint64_t mbarriers each, in this CTA's
 *     shared memory. Indexed by pipeline stage s in [0, NUM_STAGES).
 *     Caller is responsible for mbarrier_init + the
 *     `fence.mbarrier_init.release.cluster` at kernel entry.
 *
 *     full_bar[s] semantics: "stage s SMEM has been TMA-loaded with
 *     valid A+B operand data; ready for MMA to consume."
 *       - PRODUCER (this body): leader CTA's lane 0 issues
 *         `mbarrier.arrive.expect_tx` for
 *         `2 * (A_TILE_BYTES + B_TILE_BYTES)`; the cta_group::2
 *         multicast TMA load's `complete_tx::bytes` delivers the
 *         byte arrivals to full_bar[s] on BOTH peers' SMEM via the
 *         peer-bit-mask routing (tma_peer_bit_mask). Configured
 *         with arrive_count=1 at init.
 *       - CONSUMER (downstream MMA warp, NOT this body): waits
 *         try_wait.parity on full_bar[s]; once cleared, issues
 *         tcgen05.mma against smem_a[s] / smem_b[s].
 *
 *     empty_bar[s] semantics: "stage s SMEM has been consumed by
 *     MMA; safe for the load warp to refill."
 *       - PRODUCER (downstream MMA warp, NOT this body): MMA's
 *         tcgen05.commit.multicast::cluster signals empty_bar[s] in
 *         both peers' SMEM after the K-block atoms targeting that
 *         stage have been issued. Configured with arrive_count=1 at
 *         init.
 *       - CONSUMER (this body): lane 0 waits try_wait.parity on
 *         empty_bar[s] before refilling it. Phase-1 init idiom
 *         (EmptyPhaseTracker starts at parity 1) lets the first
 *         NUM_STAGES iters pass through without pre-arrives -- the
 *         bar's fresh phase 0 doesn't match the expected parity 1,
 *         so try_wait.parity returns true and the call is a no-op.
 *         Subsequent iters wait for real arrives from MMA.
 *
 *   K, m_offset, n_offset_b
 *     Problem K dimension and per-CTA tile coordinates in GMEM.
 *     m_offset is the row offset in A (leader CTA + peer offset
 *     already folded in); n_offset_b is the row offset in B^T (TMA
 *     descriptor B is stored transposed, indexed by n on the
 *     leading axis).
 *
 *   peer
 *     Cluster rank for this CTA: 0 (leader) or 1 (follower) under
 *     the cta_group::2 cluster_dims(2,1,1) contract. Leader
 *     arrives expect_tx on full_bar; both peers issue
 *     tma_load_2d_2sm (cta_group::2 multicasts the load).
 *
 *   empty_ph
 *     Caller-owned EmptyPhaseTracker reference. Caller declares it
 *     once at warp entry (e.g., `EmptyPhaseTracker<NUM_STAGES>
 *     empty_ph;` -- default-constructs with stage=0, parity=1) and
 *     passes by reference each tile so phase persists across the
 *     persistent loop. Body advances the tracker once per K-block.
 *
 * Single-thread invocation (load warp's elected thread):
 *   The body issues per-stage `mbarrier_arrive_expect_tx` and
 *   `tma_load_2d_2sm`; both must come from EXACTLY ONE thread per
 *   warp. All lanes of the selected load warp must enter this function;
 *   it elects the issuing lane internally and converges the warp before
 *   returning:
 *
 *     if (warp == load_warp)
 *         load_warp_blackwell_1tile_2sm_bf16<...>(wpc, ...);
 *
 * PTX:    9.7.10.28.5.3 (TMA cta_group::2),
 *         9.7.15.16.14 (mbarrier.arrive.expect_tx),
 *         9.7.15.16.19 (mbarrier.try_wait.parity).
 * ============================================================================ */
template <int NUM_STAGES, int M_TILE_PER_CTA, int N_TILE_PER_CTA, int K_TILE,
          bool USE_L2_HINT = false,
          int L2CACHE_POLICY_A = 0, int L2CACHE_POLICY_B = 0>
__device__ inline
void load_warp_blackwell_1tile_2sm_bf16(WpCtx& wpc,
    const CUtensorMap* tma_a, const CUtensorMap* tma_b,
    uint8_t* smem_a, uint8_t* smem_b,
    uint64_t* full_bar, uint64_t* empty_bar,
    int K, int m_offset, int n_offset_b, int peer,
    EmptyPhaseTracker<NUM_STAGES>& empty_ph) {
  constexpr int kElemBytes = static_cast<int>(sizeof(__nv_bfloat16));
  constexpr int kStageABytes = M_TILE_PER_CTA * K_TILE * kElemBytes;
  constexpr int kStageBBytes = N_TILE_PER_CTA * K_TILE * kElemBytes;
  const uint32_t expect_tx = 2 * (kStageABytes + kStageBBytes);
  const int K_BLOCKS = K / K_TILE;
  const uint64_t cache_policy_a = make_l2cache_policy<L2CACHE_POLICY_A>();
  const uint64_t cache_policy_b = make_l2cache_policy<L2CACHE_POLICY_B>();

  for (int k = 0; k < K_BLOCKS; ++k) {
    const int      s          = empty_ph.get_stage();
    const uint32_t full       = smem_ptr_u32(&full_bar[s]);
    const uint32_t full_route = tma_peer_bit_mask(full);
    const int      k_off      = k * K_TILE;

    wp_begin(wpc, WP_LOAD_WAIT);
    mbarrier_wait_parity(smem_ptr_u32(&empty_bar[s]),
                         empty_ph.get_phase());
    wp_end(wpc, WP_LOAD_WAIT);
    wp_begin(wpc, WP_LOAD_ISSUE);
    if (elect_one_sync()) {
      if (peer == 0) mbarrier_arrive_expect_tx(full, expect_tx);
      if constexpr (USE_L2_HINT) {
        tma_load_2d_2sm_l2hint(smem_ptr_u32(smem_a + s * kStageABytes),
                               tma_a, full_route, k_off, m_offset, cache_policy_a);
        tma_load_2d_2sm_l2hint(smem_ptr_u32(smem_b + s * kStageBBytes),
                               tma_b, full_route, k_off, n_offset_b, cache_policy_b);
      } else {
        tma_load_2d_2sm(smem_ptr_u32(smem_a + s * kStageABytes),
                        tma_a, full_route, k_off, m_offset);
        tma_load_2d_2sm(smem_ptr_u32(smem_b + s * kStageBBytes),
                        tma_b, full_route, k_off, n_offset_b);
      }
    }
    wp_end(wpc, WP_LOAD_ISSUE);
    empty_ph.advance();
  }
  __syncwarp();  // converge after the elect-issued TMA loads
}

/* ============================================================================
 * load_warp_blackwell_1tile_1sm_bf16<NUM_STAGES, M_TILE_PER_CTA,
 *                                    N_TILE_PER_CTA, K_TILE,
 *                                    USE_L2_HINT, L2CACHE_POLICY_A,
 *                                    L2CACHE_POLICY_B>(wpc, ...)
 *
 * 1SM (cta_group::1, no cluster) counterpart of the _1tile_2sm_ form.
 * Used by phase-2 of grouped GEMM where each tile is processed by a
 * single CTA.
 *
 * Differences from _1tile_2sm_:
 *   - TMA via `tma_load_2d_cta{,_l2hint}` (.shared::cta scope; single CTA,
 *     no cluster).
 *   - expect_tx = 1 x (A + B) bytes (single CTA, no peer multicast).
 *   - No `peer` / `tma_peer_bit_mask` -- the full_bar address is local.
 *   - Caller must `elect_one_sync()` before calling (single-issuer).
 *
 * L2 cache policy:
 *   USE_L2_HINT = true selects the .L2::cache_hint TMA form, with per-
 *   operand policies from primitive 68's menu (0=evict_last_full,
 *   1=evict_normal_full, ...). When false, no hint is applied.
 * ============================================================================ */
template <int NUM_STAGES, int M_TILE_PER_CTA, int N_TILE_PER_CTA, int K_TILE,
          bool USE_L2_HINT = false,
          int L2CACHE_POLICY_A = 0, int L2CACHE_POLICY_B = 0>
__device__ inline
void load_warp_blackwell_1tile_1sm_bf16(WpCtx& wpc,
    const CUtensorMap* tma_a, const CUtensorMap* tma_b,
    uint8_t* smem_a, uint8_t* smem_b,
    uint64_t* full_bar, uint64_t* empty_bar,
    int K, int m_offset, int n_offset_b,
    EmptyPhaseTracker<NUM_STAGES>& empty_ph) {
  constexpr int kElemBytes = static_cast<int>(sizeof(__nv_bfloat16));
  constexpr int kStageABytes = M_TILE_PER_CTA * K_TILE * kElemBytes;
  constexpr int kStageBBytes = N_TILE_PER_CTA * K_TILE * kElemBytes;
  const uint32_t expect_tx = kStageABytes + kStageBBytes;
  const int K_BLOCKS = K / K_TILE;
  const uint64_t cache_policy_a = make_l2cache_policy<L2CACHE_POLICY_A>();
  const uint64_t cache_policy_b = make_l2cache_policy<L2CACHE_POLICY_B>();

  for (int k = 0; k < K_BLOCKS; ++k) {
    const int      s    = empty_ph.get_stage();
    const uint32_t full = smem_ptr_u32(&full_bar[s]);
    const int      k_off = k * K_TILE;

    mbarrier_wait_parity(smem_ptr_u32(&empty_bar[s]),
                         empty_ph.get_phase());
    if (elect_one_sync()) {
      mbarrier_arrive_expect_tx(full, expect_tx);
      if constexpr (USE_L2_HINT) {
        tma_load_2d_cta_l2hint(tma_a, smem_ptr_u32(smem_a + s * kStageABytes),
                               full, k_off, m_offset, cache_policy_a);
        tma_load_2d_cta_l2hint(tma_b, smem_ptr_u32(smem_b + s * kStageBBytes),
                               full, k_off, n_offset_b, cache_policy_b);
      } else {
        tma_load_2d_cta(tma_a, smem_ptr_u32(smem_a + s * kStageABytes),
                        full, k_off, m_offset);
        tma_load_2d_cta(tma_b, smem_ptr_u32(smem_b + s * kStageBBytes),
                        full, k_off, n_offset_b);
      }
    }
    empty_ph.advance();
  }
  __syncwarp();  // converge after the elect-issued TMA loads
}

/* ============================================================================
 * load_warp_blackwell_ntiles_2sm_bf16<...>(wpc, ...)
 *
 * Self-driving do-while wrapper around the _1tile_ body for per-warp
 * persistent loops. Each iteration:
 *   1. Compute m_offset / n_offset_b for the CURRENT tile coords.
 *   2. (lane 0 only) call _1tile_ body (K-loop TMA loads).
 *   3. fetch_next_tile (cluster-wide CLC handshake; ALL lanes).
 *   4. Update m_tile, n_tile from next; break if !next.valid.
 *   5. (lane 0 only) tail drain on empty_bar (NUM_STAGES iters).
 *
 * Threading model:
 *   Caller invokes from ALL 32 lanes of the load warp (e.g., warp 2) in
 *   BOTH peer CTAs (= 64 threads/cluster contributing to clc_empty).
 *   - All lanes (both CTAs): setmaxnreg, fetch_next_tile, m/n update.
 *   - Lane 0 only: _1tile_ body call + empty_ph state + tail drain.
 *
 * Template parameters:
 *   NUM_STAGES, M_TILE_PER_CTA, N_TILE_PER_CTA, K_TILE
 *     Forwarded to _1tile_ body.
 *   M_TILE_CLUSTER, N_TILE_CLUSTER
 *     Cluster-level tile shape (per-CTA = cluster / 2 under cta_group::2).
 *   LOAD_REG_BUDGET (default 40)
 *     Register budget after setmaxnreg.dec. [24, 256], multiple of 8.
 *     Default 40 -- load warp is light (TMA ops + mbar bookkeeping;
 *     no MMA / cvt / stmatrix pressure).
 * ============================================================================ */
// L2CACHE_POLICY_A / L2CACHE_POLICY_B (compile-time, only used when
// USE_L2_HINT=true): pick the L2 cache hint per operand from the standard
// menu:
//   0 = evict_last (default; bias working set to stay in L2)
//   1 = evict_normal
//   2 = fractional evict_last/unchanged @ 0.5
//   3 = fractional evict_last/unchanged @ 0.25
//   4 = evict_unchanged (pure stream)
// Dense GEMM callers leave both at 0. Grouped GEMM typically picks
// A=evict_unchanged (pure stream) and B=evict_last (cross-tile expert
// weight reuse).
//
// GROUPED_GEMM (compile-time): when true, the loop decodes expert_id per
// tile via composite 83's find_group_id and shifts n_offset_b by
// `expert_id * N` so B's flat (E*N, K) layout reads the right expert's
// weights. `m_tile_remap` (runtime, nullptr = identity): see block 93's
// `m_tile_remap` doc for the idx-space details. N / E / expert_cumul_smem
// are only read when GROUPED_GEMM=true; pass defaults for dense GEMM use.
template <int NUM_STAGES, int M_TILE_PER_CTA, int N_TILE_PER_CTA, int K_TILE,
          int M_TILE_CLUSTER, int N_TILE_CLUSTER,
          int LOAD_REG_BUDGET = 40,
          bool USE_L2_HINT = false,
          int CLUSTER_SHAPE_M = 1, int CLUSTER_SHAPE_N = 2,
          ClcRasterOrder ORDER = ClcRasterOrder::AlongN,
          bool GROUPED_GEMM = false,
          int L2CACHE_POLICY_A = 0, int L2CACHE_POLICY_B = 0>
__device__ inline
void load_warp_blackwell_ntiles_2sm_bf16(WpCtx& wpc,
    const CUtensorMap* tma_a, const CUtensorMap* tma_b,
    uint8_t* smem_a, uint8_t* smem_b,
    uint64_t* full_bar, uint64_t* empty_bar,
    uint64_t* clc_full_bar, uint64_t* clc_empty_bar, uint32_t* clc_response,
    uint64_t* throttle_full, uint64_t* throttle_empty,
    int K, int peer,
    int N = 0, int E = 0,
    const int* __restrict__ expert_cumul_smem = nullptr,
    const int* __restrict__ m_tile_remap = nullptr) {
  setmaxnreg_dec<LOAD_REG_BUDGET>();

  const int lane = threadIdx.x & 31;
  EmptyPhaseTracker<NUM_STAGES> empty_ph;
  int      clc_cons_stage = 0;
  uint32_t clc_cons_phase = 0;
  int      thr_prod_stage = 0;
  uint32_t thr_prod_phase = 1;

  auto remap = [&](int p1) -> int {
    return m_tile_remap ? m_tile_remap[p1] : p1;
  };

  int m_tile, n_tile;
  if constexpr (ORDER == ClcRasterOrder::AlongN) {
    m_tile = remap((int)blockIdx.y);
    n_tile = (int)(blockIdx.x >> 1);
  } else {  // AlongM: cluster.x -> M
    m_tile = remap((int)(blockIdx.x >> 1));
    n_tile = (int)blockIdx.y;
  }

  const int K_BLOCKS = K / K_TILE;
  (void)lane;
  (void)K_BLOCKS;

  while (true) {
    const int m_offset = m_tile * M_TILE_CLUSTER + peer * M_TILE_PER_CTA;
    int n_offset_b     = n_tile * N_TILE_CLUSTER + peer * N_TILE_PER_CTA;
    if constexpr (GROUPED_GEMM) {
      const int expert_id = find_group_id(m_tile, expert_cumul_smem, E);
      n_offset_b += expert_id * N;
    }

    // Throttle handshake (producer side): wait throttle_empty, arrive
    // throttle_full. ALL 32 lanes participate so the warp converges
    // before the K-loop.
    if (peer == 0) {
      wp_begin(wpc, WP_LOAD_WAIT_THROTTLE);  // scheduler pacing (not backpressure)
      mbarrier_wait_parity(smem_ptr_u32(&throttle_empty[thr_prod_stage]),
                           thr_prod_phase);
      wp_end(wpc, WP_LOAD_WAIT_THROTTLE);
      mbarrier_arrive_nostate(smem_ptr_u32(&throttle_full[thr_prod_stage]));
      advance_stage_phase<2>(thr_prod_stage, thr_prod_phase);
    }

    // K-loop: ALL 32 lanes enter helper. Helper waits empty_bar and
    // advances empty_ph on all lanes (warp-uniform); elect_one_sync()
    // inside the helper gates arrive_expect_tx + TMA loads.
    load_warp_blackwell_1tile_2sm_bf16<
        NUM_STAGES, M_TILE_PER_CTA, N_TILE_PER_CTA, K_TILE,
        USE_L2_HINT, L2CACHE_POLICY_A, L2CACHE_POLICY_B>(wpc, tma_a, tma_b, smem_a, smem_b, full_bar, empty_bar,
        K, m_offset, n_offset_b, peer, empty_ph);

    wp_begin(wpc, WP_CLC_FETCH);
    ClcTileInfo next = clc_fetch_next_tile<
        CLUSTER_SHAPE_M, CLUSTER_SHAPE_N, ORDER>(
        clc_full_bar, clc_empty_bar, clc_response,
        clc_cons_stage, clc_cons_phase, /*do_release=*/elect_one_sync());
    wp_end(wpc, WP_CLC_FETCH);
    clc_fetch_next_tile_advance(clc_cons_stage, clc_cons_phase);
    if (!next.valid) break;
    m_tile = remap(next.m_tile);
    n_tile = next.n_tile;
  }

  // Tail drain: ALL 32 lanes wait remaining empty_bar slots so the
  // pipeline drains cleanly before the kernel exits.
  wp_begin(wpc, WP_LOAD_WAIT);  // tail drain: wait remaining stages
  #pragma unroll
  for (int i = 0; i < NUM_STAGES; ++i) {
    const int s = empty_ph.get_stage();
    mbarrier_wait_parity(smem_ptr_u32(&empty_bar[s]),
                                  empty_ph.get_phase());
    empty_ph.advance();
  }
  wp_end(wpc, WP_LOAD_WAIT);
}

/* ============================================================================
 * load_warp_blackwell_ntiles_2sm_bf16_swiglu_m1<...>(wpc, ...)
 *
 * M1-layout grouped + SwiGLU load warp. Caller supplies B in the
 * framework-native concat-halves layout:
 *   B shape  = [E, 2 * I_moe, K] row-major (N = 2 * I_moe)
 *   B[e, 0..I_moe-1, :]      = gate weights for expert e
 *   B[e, I_moe..2*I_moe-1, :] = up   weights for expert e
 *
 * No host pre-pack is needed -- this matches Megatron / vLLM / SGLang
 * fused MoE conventions verbatim.
 *
 * Per N-tile, the two cta_group::2 peers fetch DIFFERENT N-sections of
 * the same B (mapped to TMEM cols [0, 128) = up, [128, 256) = gate per
 * peer's MMA share, matching the layout the M2 EPI consumes):
 *   peer 0 -> up   rows: n_offset_b = expert_id * N + I_moe + n_tile * (NTC/2)
 *   peer 1 -> gate rows: n_offset_b = expert_id * N + 0     + n_tile * (NTC/2)
 *
 * Everything else (TMA descriptor count per peer = 1, mbarrier
 * transaction count, full / empty handshake, throttle pipeline,
 * CLC fetch) is identical to `_ntiles_2sm_bf16`. The new function
 * exists only to swap the per-peer offset arithmetic.
 *
 * Pairs with `epi_warp_blackwell_ntiles_2sm_bf16_swiglu_chunked` --
 * the TMEM accumulator layout is identical ([up | gate] split at
 * N_TILE_CLUSTER/2), so the EPI consumer is the same.
 * ============================================================================ */
template <int NUM_STAGES, int M_TILE_PER_CTA, int N_TILE_PER_CTA, int K_TILE,
          int M_TILE_CLUSTER, int N_TILE_CLUSTER,
          int LOAD_REG_BUDGET = 40,
          bool USE_L2_HINT = false,
          int CLUSTER_SHAPE_M = 1, int CLUSTER_SHAPE_N = 2,
          ClcRasterOrder ORDER = ClcRasterOrder::AlongN,
          int L2CACHE_POLICY_A = 0, int L2CACHE_POLICY_B = 0>
__device__ inline
void load_warp_blackwell_ntiles_2sm_bf16_swiglu_m1(WpCtx& wpc,
    const CUtensorMap* tma_a, const CUtensorMap* tma_b,
    uint8_t* smem_a, uint8_t* smem_b,
    uint64_t* full_bar, uint64_t* empty_bar,
    uint64_t* clc_full_bar, uint64_t* clc_empty_bar, uint32_t* clc_response,
    uint64_t* throttle_full, uint64_t* throttle_empty,
    int K, int peer,
    int N, int E,
    const int* __restrict__ expert_cumul_smem,
    const int* __restrict__ m_tile_remap = nullptr) {
  setmaxnreg_dec<LOAD_REG_BUDGET>();
  static_assert(N_TILE_PER_CTA == N_TILE_CLUSTER / 2,
                "M1 SwiGLU load assumes peer-split N at N_TILE_CLUSTER/2");
  // I_moe = post-SwiGLU N = N / 2. N must be even for the half split.

  const int lane = threadIdx.x & 31;
  EmptyPhaseTracker<NUM_STAGES> empty_ph;
  int      clc_cons_stage = 0;
  uint32_t clc_cons_phase = 0;
  int      thr_prod_stage = 0;
  uint32_t thr_prod_phase = 1;

  auto remap = [&](int p1) -> int {
    return m_tile_remap ? m_tile_remap[p1] : p1;
  };

  int m_tile, n_tile;
  if constexpr (ORDER == ClcRasterOrder::AlongN) {
    m_tile = remap((int)blockIdx.y);
    n_tile = (int)(blockIdx.x >> 1);
  } else {  // AlongM: cluster.x -> M
    m_tile = remap((int)(blockIdx.x >> 1));
    n_tile = (int)blockIdx.y;
  }

  const int K_BLOCKS = K / K_TILE;
  const int I_moe    = N / 2;
  (void)lane;
  (void)K_BLOCKS;

  while (true) {
    const int expert_id  = find_group_id(m_tile, expert_cumul_smem, E);
    const int m_offset   = m_tile * M_TILE_CLUSTER + peer * M_TILE_PER_CTA;
    // Peer 0 -> up section (rows [I_moe, 2*I_moe)); peer 1 -> gate
    // section (rows [0, I_moe)). Matches the M2 EPI's TMEM-col convention
    // (cols 0..NTC/2 = up, NTC/2..NTC = gate).
    const int section    = (peer == 0) ? I_moe : 0;
    const int n_offset_b = expert_id * N + section + n_tile * N_TILE_PER_CTA;

    if (peer == 0) {
      mbarrier_wait_parity(smem_ptr_u32(&throttle_empty[thr_prod_stage]),
                           thr_prod_phase);
      mbarrier_arrive_nostate(smem_ptr_u32(&throttle_full[thr_prod_stage]));
      advance_stage_phase<2>(thr_prod_stage, thr_prod_phase);
    }

    load_warp_blackwell_1tile_2sm_bf16<
        NUM_STAGES, M_TILE_PER_CTA, N_TILE_PER_CTA, K_TILE,
        USE_L2_HINT, L2CACHE_POLICY_A, L2CACHE_POLICY_B>(wpc, tma_a, tma_b, smem_a, smem_b, full_bar, empty_bar,
        K, m_offset, n_offset_b, peer, empty_ph);

    ClcTileInfo next = clc_fetch_next_tile<
        CLUSTER_SHAPE_M, CLUSTER_SHAPE_N, ORDER>(
        clc_full_bar, clc_empty_bar, clc_response,
        clc_cons_stage, clc_cons_phase, /*do_release=*/elect_one_sync());
    clc_fetch_next_tile_advance(clc_cons_stage, clc_cons_phase);
    if (!next.valid) break;
    m_tile = remap(next.m_tile);
    n_tile = next.n_tile;
  }

  // Tail drain.
  #pragma unroll
  for (int i = 0; i < NUM_STAGES; ++i) {
    const int s = empty_ph.get_stage();
    mbarrier_wait_parity(smem_ptr_u32(&empty_bar[s]),
                                  empty_ph.get_phase());
    empty_ph.advance();
  }
}

/* ============================================================================
 * load_warp_blackwell_ntiles_2sm_bf16_gather4<...>(wpc, ...)
 *
 * Gather-fused 2SM load warp for MoE FC1. Fuses the per-token gather
 * (packed -> unpermuted token index lookup) into the load path so the
 * activation matrix A no longer needs to be materialized in
 * per-expert packed order in GMEM.
 *
 *   A : (M_in, K) row-major, UNPERMUTED activations. Single 2D
 *       tensormap with box = (1, K_TILE), SWIZZLE_128B (matches the
 *       MMA's A-side SMEM layout; verified by tests/76). Each gather4
 *       call fetches 4 independent 1-row strips. Pad slots in `perm[]`
 *       use -1 and the descriptor's OOB-fill (zero) handles them.
 *   B : same chunked M2 layout as block 93's SwiGLU EPI
 *       (cols 0..N/2-1 = up, N/2..N-1 = gate, pre-shuffled per N-tile),
 *       SWIZZLE_128B.
 *
 * Per K-block per peer the load warp issues:
 *   - M_TILE_PER_CTA / 4 gather4 calls for A (e.g. 128/4 = 32).
 *     Each call signals THIS peer's LOCAL full mbar (gather4 has no
 *     .cta_group::2 form, so cluster-scope mbar routing via the
 *     peer-bit mask does NOT cross-signal; see tests/78).
 *   - 1 standard 2SM `tma_load_2d_2sm` for B. This DOES use
 *     .cta_group::2 and routes via the peer-bit mask to peer 0's mbar.
 *
 * Per-peer expect_tx:
 *   peer 0: kStageABytes + 2 * kStageBBytes
 *           (own A gather + both peers' cta_group::2 B contributions)
 *   peer 1: kStageABytes
 *           (own A gather only; no B routes to peer 1's mbar)
 *
 * CAVEAT: peer 0's MMA (cta_group::2) reads A from BOTH peers' SMEMs.
 * Peer 0's mbar flips when peer 0's own A and both peers' B have
 * arrived -- it has NO visibility into peer 1's A-gather completion.
 * Without a cross-peer sync this races and the kernel hangs. Closing
 * this requires either (a) a cluster_barrier between gather and MMA,
 * (b) a dedicated cross-peer "both A done" mbar with peer 1
 * arriving on it after waiting for its own local mbar, or (c) a
 * non-TMA gather (e.g. cp.async cluster-cooperative).
 *
 * `perm[packed_idx]` = source token row in A (or -1 for padding).
 * `packed_idx` runs from 0 to total_padded_tokens - 1. Per tile,
 * the peer reads packed indices `[m_tile * MTC + peer * MTC_PER_CTA,
 * + MTC_PER_CTA)`.
 *
 * PTX:    9.7.10.28.5.3 (TMA tile::gather4; no cta_group::2 form),
 *         9.7.15.16.14 (mbarrier.arrive.expect_tx),
 *         9.7.15.16.19 (mbarrier.try_wait.parity).
 * ============================================================================ */
template <int NUM_STAGES, int M_TILE_PER_CTA, int N_TILE_PER_CTA, int K_TILE,
          int M_TILE_CLUSTER, int N_TILE_CLUSTER,
          int LOAD_REG_BUDGET = 40,
          bool USE_L2_HINT = false,
          int CLUSTER_SHAPE_M = 1, int CLUSTER_SHAPE_N = 2,
          ClcRasterOrder ORDER = ClcRasterOrder::AlongN,
          int L2CACHE_POLICY_A = 0, int L2CACHE_POLICY_B = 0>
__device__ inline
void load_warp_blackwell_ntiles_2sm_bf16_gather4(WpCtx& wpc,
    const CUtensorMap* tma_a_gather,  // A: box=(1, K_TILE), SWIZZLE_128B
    const CUtensorMap* tma_b,         // B: standard chunked, SWIZZLE_128B
    uint8_t* smem_a, uint8_t* smem_b,
    uint64_t* full_bar, uint64_t* empty_bar,
    uint64_t* peer1_done_bar,         // NUM_STAGES; arrive_count=1
    uint64_t* clc_full_bar, uint64_t* clc_empty_bar, uint32_t* clc_response,
    uint64_t* throttle_full, uint64_t* throttle_empty,
    int K, int peer,
    int N, int E,
    const int* __restrict__ perm,                    // packed -> token idx
    const int* __restrict__ expert_cumul_smem,
    const int* __restrict__ m_tile_remap = nullptr) {
  setmaxnreg_dec<LOAD_REG_BUDGET>();
  static_assert(M_TILE_PER_CTA % 4 == 0,
                "gather4 fetches 4 rows per call; M_TILE_PER_CTA must be /4");

  constexpr int N_GATHER_PER_KBLOCK = M_TILE_PER_CTA / 4;
  constexpr int kElemBytes   = static_cast<int>(sizeof(__nv_bfloat16));
  constexpr int kStageABytes = M_TILE_PER_CTA * K_TILE * kElemBytes;
  constexpr int kStageBBytes = N_TILE_PER_CTA * K_TILE * kElemBytes;
  // gather4 (no .cta_group::2 form in PTX 9.x) only signals its LOCAL
  // CTA's mbar -- tests/78 confirmed cluster-scope mbar routing via
  // peer-bit toggle does NOT cross-signal for gather4. So per-peer
  // accounting:
  //   peer 0 mbar: peer 0's A gather (local)  + peer 0's B 2SM cta_group::2
  //                                            + peer 1's B 2SM cta_group::2
  //              = kStageABytes + 2*kStageBBytes.
  //   peer 1 mbar: peer 1's A gather (local).
  //              = kStageABytes.
  // Only peer 0's mbar is consumed by MMA (MMA runs on peer 0 only).
  // Peer 1's mbar is consumed by peer 1's MMA warp (the relay): it waits
  // peer 1's full_bar then arrives peer 0's peer1_done_bar once peer 1's
  // gather has landed -- see mma_warp_blackwell_ntiles_2sm_bf16_peer1relay.
  const uint32_t expect_tx_peer0 =
      (uint32_t)(kStageABytes + 2 * kStageBBytes);
  const uint32_t expect_tx_peer1 = (uint32_t)kStageABytes;
  const uint32_t expect_tx_local =
      (peer == 0) ? expect_tx_peer0 : expect_tx_peer1;
  const int K_BLOCKS = K / K_TILE;
  const uint64_t cache_policy_a = make_l2cache_policy<L2CACHE_POLICY_A>();
  const uint64_t cache_policy_b = make_l2cache_policy<L2CACHE_POLICY_B>();
  (void)cache_policy_a;
  (void)cache_policy_b;

  const int lane = threadIdx.x & 31;
  EmptyPhaseTracker<NUM_STAGES> empty_ph;
  int      clc_cons_stage = 0;
  uint32_t clc_cons_phase = 0;
  int      thr_prod_stage = 0;
  uint32_t thr_prod_phase = 1;

  auto remap = [&](int p1) -> int {
    return m_tile_remap ? m_tile_remap[p1] : p1;
  };

  int m_tile, n_tile;
  if constexpr (ORDER == ClcRasterOrder::AlongN) {
    m_tile = remap((int)blockIdx.y);
    n_tile = (int)(blockIdx.x >> 1);
  } else {
    m_tile = remap((int)(blockIdx.x >> 1));
    n_tile = (int)blockIdx.y;
  }
  (void)lane;

  while (true) {
    const int expert_id       = find_group_id(m_tile, expert_cumul_smem, E);
    // Packed-token base index for THIS peer's MTC_PER_CTA rows.
    const int packed_row_base = m_tile * M_TILE_CLUSTER
                              + peer * M_TILE_PER_CTA;
    // B uses the chunked M2 layout: cluster N tile = N_TILE_CLUSTER,
    // per-peer half = N_TILE_PER_CTA = N_TILE_CLUSTER / 2.
    const int n_offset_b      = expert_id * N
                              + n_tile * N_TILE_CLUSTER
                              + peer * N_TILE_PER_CTA;

    if (peer == 0) {
      mbarrier_wait_parity(smem_ptr_u32(&throttle_empty[thr_prod_stage]),
                           thr_prod_phase);
      mbarrier_arrive_nostate(smem_ptr_u32(&throttle_full[thr_prod_stage]));
      advance_stage_phase<2>(thr_prod_stage, thr_prod_phase);
    }

    // K-block loop: gather4 x N_GATHER_PER_KBLOCK for A + 2SM TMA for B.
    for (int k = 0; k < K_BLOCKS; ++k) {
      const int      s          = empty_ph.get_stage();
      const uint32_t full       = smem_ptr_u32(&full_bar[s]);
      const uint32_t full_route = tma_peer_bit_mask(full);
      const int      k_off      = k * K_TILE;
      const uint32_t smem_a_s   = smem_ptr_u32(smem_a + s * kStageABytes);
      const uint32_t smem_b_s   = smem_ptr_u32(smem_b + s * kStageBBytes);

      mbarrier_wait_parity(smem_ptr_u32(&empty_bar[s]),
                           empty_ph.get_phase());
      if (elect_one_sync()) {
        // Each peer arms its OWN local mbar (gather4 mbar must be
        // local per tests/78; see expect_tx_local computation above).
        mbarrier_arrive_expect_tx(full, expect_tx_local);

        // A: 32 gather4 calls signaling THIS peer's local mbar.
        // Pad slots (perm[i] == -1) use the descriptor's OOB-fill = 0.
        #pragma unroll
        for (int g = 0; g < N_GATHER_PER_KBLOCK; ++g) {
          const int r0 = perm[packed_row_base + g * 4 + 0];
          const int r1 = perm[packed_row_base + g * 4 + 1];
          const int r2 = perm[packed_row_base + g * 4 + 2];
          const int r3 = perm[packed_row_base + g * 4 + 3];
          const uint32_t dst = smem_a_s
                             + (uint32_t)(g * 4 * K_TILE * kElemBytes);
          if constexpr (USE_L2_HINT) {
            tma_load_2d_gather4_l2hint(dst, tma_a_gather, full,
                                       k_off, r0, r1, r2, r3,
                                       cache_policy_a);
          } else {
            tma_load_2d_gather4(dst, tma_a_gather, full,
                                k_off, r0, r1, r2, r3);
          }
        }

        // B: standard 2SM multicast TMA (cta_group::2). Signals peer 0's
        // full_bar via the peer-bit-masked address.
        if constexpr (USE_L2_HINT) {
          tma_load_2d_2sm_l2hint(smem_b_s, tma_b, full_route,
                                 k_off, n_offset_b, cache_policy_b);
        } else {
          tma_load_2d_2sm(smem_b_s, tma_b, full_route,
                          k_off, n_offset_b);
        }
      }

      // Fire-and-forget: each peer just armed its OWN full_bar. The
      // peer1->peer0 A-ready relay is on peer1's MMA warp (..._gather_async),
      // not here (gather4 can't cluster-route its mbar -- tests/78,81).
      empty_ph.advance();
    }
    __syncwarp();

    const bool epi_release = elect_one_sync();
    ClcTileInfo next = clc_fetch_next_tile<
        CLUSTER_SHAPE_M, CLUSTER_SHAPE_N, ORDER>(
        clc_full_bar, clc_empty_bar, clc_response,
        clc_cons_stage, clc_cons_phase, /*do_release=*/epi_release);
    clc_fetch_next_tile_advance(clc_cons_stage, clc_cons_phase);
    if (!next.valid) break;
    m_tile = remap(next.m_tile);
    n_tile = next.n_tile;
  }

  // Tail drain.
  #pragma unroll
  for (int i = 0; i < NUM_STAGES; ++i) {
    const int s = empty_ph.get_stage();
    mbarrier_wait_parity(smem_ptr_u32(&empty_bar[s]),
                         empty_ph.get_phase());
    empty_ph.advance();
  }
}

/* ============================================================================
 * load_warp_blackwell_ntiles_2sm_bf16_gather4_parallel<...>(wpc, ...)
 *
 * Lane-parallel sibling of `_ntiles_2sm_bf16_gather4`. Same 2SM peer
 * accounting, peer-bit B multicast, and fire-and-forget peer1-relay
 * handshake (peer 1's MMA warp relays A-ready to peer 0). The only change
 * is the A gather4 issue, ported from `_1sm_bf16_gather4_parallel`:
 *   (i)  hoist this lane's 4 perm row indices ONCE per tile (K-invariant),
 *        held in registers instead of re-read every K-block; and
 *   (ii) lane g issues gather4 #g (rows [g*4, g*4+4)) so all
 *        N_GATHER_PER_KBLOCK (= M_TILE_PER_CTA/4) issue across the warp in
 *        parallel rather than serially from one elected lane.
 * mbar accounting is unchanged: one elected lane arrives expect_tx_local
 * (ordered before any issue by __syncwarp); each gather4 + the B 2SM load
 * accumulate tx-bytes into the same per-peer full_bar.
 *
 * Parallel issue + 2SM peer1-relay. Requires N_GATHER_PER_KBLOCK <= 32.
 * ============================================================================ */
template <int NUM_STAGES, int M_TILE_PER_CTA, int N_TILE_PER_CTA, int K_TILE,
          int M_TILE_CLUSTER, int N_TILE_CLUSTER,
          int LOAD_REG_BUDGET = 40,
          bool USE_L2_HINT = false,
          int CLUSTER_SHAPE_M = 1, int CLUSTER_SHAPE_N = 2,
          ClcRasterOrder ORDER = ClcRasterOrder::AlongN,
          int L2CACHE_POLICY_A = 0, int L2CACHE_POLICY_B = 0>
__device__ inline
void load_warp_blackwell_ntiles_2sm_bf16_gather4_parallel(WpCtx& wpc,
    const CUtensorMap* tma_a_gather,  // A: box=(1, K_TILE), SWIZZLE_128B
    const CUtensorMap* tma_b,         // B: standard chunked, SWIZZLE_128B
    uint8_t* smem_a, uint8_t* smem_b,
    uint64_t* full_bar, uint64_t* empty_bar,
    uint64_t* peer1_done_bar,         // NUM_STAGES; arrive_count=1
    uint64_t* clc_full_bar, uint64_t* clc_empty_bar, uint32_t* clc_response,
    uint64_t* throttle_full, uint64_t* throttle_empty,
    int K, int peer,
    int N, int E,
    const int* __restrict__ perm,
    const int* __restrict__ expert_cumul_smem,
    const int* __restrict__ m_tile_remap = nullptr) {
  setmaxnreg_dec<LOAD_REG_BUDGET>();
  static_assert(M_TILE_PER_CTA % 4 == 0,
                "gather4 fetches 4 rows per call; M_TILE_PER_CTA must be /4");
  constexpr int N_GATHER_PER_KBLOCK = M_TILE_PER_CTA / 4;
  static_assert(N_GATHER_PER_KBLOCK <= 32,
                "4A maps one gather group per lane; need N_GATHER <= warp size");

  constexpr int kElemBytes   = static_cast<int>(sizeof(__nv_bfloat16));
  constexpr int kStageABytes = M_TILE_PER_CTA * K_TILE * kElemBytes;
  constexpr int kStageBBytes = N_TILE_PER_CTA * K_TILE * kElemBytes;
  // Per-peer mbar accounting (gather4 cannot cta_group::2-route its mbar --
  // tests/78): peer 0 = own A gather + BOTH B halves (multicast routes to
  // peer 0); peer 1 = own A gather only.
  const uint32_t expect_tx_peer0 =
      (uint32_t)(kStageABytes + 2 * kStageBBytes);
  const uint32_t expect_tx_peer1 = (uint32_t)kStageABytes;
  const uint32_t expect_tx_local =
      (peer == 0) ? expect_tx_peer0 : expect_tx_peer1;
  const int K_BLOCKS = K / K_TILE;
  const uint64_t cache_policy_a = make_l2cache_policy<L2CACHE_POLICY_A>();
  const uint64_t cache_policy_b = make_l2cache_policy<L2CACHE_POLICY_B>();
  (void)cache_policy_a;
  (void)cache_policy_b;

  const int lane   = threadIdx.x & 31;
  const int g      = lane;                          // gather group for this lane
  const bool active = (g < N_GATHER_PER_KBLOCK);    // lanes >= N_GATHER idle on A
  EmptyPhaseTracker<NUM_STAGES> empty_ph;
  int      clc_cons_stage = 0;
  uint32_t clc_cons_phase = 0;
  int      thr_prod_stage = 0;
  uint32_t thr_prod_phase = 1;

  auto remap = [&](int p1) -> int {
    return m_tile_remap ? m_tile_remap[p1] : p1;
  };

  int m_tile, n_tile;
  if constexpr (ORDER == ClcRasterOrder::AlongN) {
    m_tile = remap((int)blockIdx.y);
    n_tile = (int)(blockIdx.x >> 1);
  } else {
    m_tile = remap((int)(blockIdx.x >> 1));
    n_tile = (int)blockIdx.y;
  }

  while (true) {
    const int expert_id       = find_group_id(m_tile, expert_cumul_smem, E);
    const int packed_row_base = m_tile * M_TILE_CLUSTER
                              + peer * M_TILE_PER_CTA;
    const int n_offset_b      = expert_id * N
                              + n_tile * N_TILE_CLUSTER
                              + peer * N_TILE_PER_CTA;

    // (i) Index prefetch: this lane's 4 row indices ONCE per tile.
    int rr0 = -1, rr1 = -1, rr2 = -1, rr3 = -1;
    if (active) {
      rr0 = perm[packed_row_base + g * 4 + 0];
      rr1 = perm[packed_row_base + g * 4 + 1];
      rr2 = perm[packed_row_base + g * 4 + 2];
      rr3 = perm[packed_row_base + g * 4 + 3];
    }

    if (peer == 0) {
      mbarrier_wait_parity(smem_ptr_u32(&throttle_empty[thr_prod_stage]),
                           thr_prod_phase);
      mbarrier_arrive_nostate(smem_ptr_u32(&throttle_full[thr_prod_stage]));
      advance_stage_phase<2>(thr_prod_stage, thr_prod_phase);
    }

    for (int k = 0; k < K_BLOCKS; ++k) {
      const int      s          = empty_ph.get_stage();
      const uint32_t full       = smem_ptr_u32(&full_bar[s]);
      const uint32_t full_route = tma_peer_bit_mask(full);
      const int      k_off      = k * K_TILE;
      const uint32_t smem_a_s   = smem_ptr_u32(smem_a + s * kStageABytes);
      const uint32_t smem_b_s   = smem_ptr_u32(smem_b + s * kStageBBytes);

      mbarrier_wait_parity(smem_ptr_u32(&empty_bar[s]),
                           empty_ph.get_phase());
      // (ii) one lane arms expect_tx, ordered before any issue.
      if (elect_one_sync()) {
        mbarrier_arrive_expect_tx(full, expect_tx_local);
      }
      __syncwarp();
      // lane-parallel gather4: lane g issues its own 4-row strip to the
      // LOCAL full_bar (gather4 mbar must be local per tests/78).
      if (active) {
        const uint32_t dst = smem_a_s
                           + (uint32_t)(g * 4 * K_TILE * kElemBytes);
        if constexpr (USE_L2_HINT) {
          tma_load_2d_gather4_l2hint(dst, tma_a_gather, full,
                                     k_off, rr0, rr1, rr2, rr3,
                                     cache_policy_a);
        } else {
          tma_load_2d_gather4(dst, tma_a_gather, full,
                              k_off, rr0, rr1, rr2, rr3);
        }
      }
      // B: 2SM multicast (cta_group::2), signals peer 0's full_bar via the
      // peer-bit-masked route. Both peers issue.
      if (elect_one_sync()) {
        if constexpr (USE_L2_HINT) {
          tma_load_2d_2sm_l2hint(smem_b_s, tma_b, full_route,
                                 k_off, n_offset_b, cache_policy_b);
        } else {
          tma_load_2d_2sm(smem_b_s, tma_b, full_route,
                          k_off, n_offset_b);
        }
      }
      __syncwarp();
      // Fire-and-forget (peer1->peer0 relay is on peer1's MMA warp).
      empty_ph.advance();
    }
    __syncwarp();

    const bool epi_release = elect_one_sync();
    ClcTileInfo next = clc_fetch_next_tile<
        CLUSTER_SHAPE_M, CLUSTER_SHAPE_N, ORDER>(
        clc_full_bar, clc_empty_bar, clc_response,
        clc_cons_stage, clc_cons_phase, /*do_release=*/epi_release);
    clc_fetch_next_tile_advance(clc_cons_stage, clc_cons_phase);
    if (!next.valid) break;
    m_tile = remap(next.m_tile);
    n_tile = next.n_tile;
  }

  // Tail drain.
  #pragma unroll
  for (int i = 0; i < NUM_STAGES; ++i) {
    const int s = empty_ph.get_stage();
    mbarrier_wait_parity(smem_ptr_u32(&empty_bar[s]),
                         empty_ph.get_phase());
    empty_ph.advance();
  }
}

/* ============================================================================
 * load_warp_blackwell_ntiles_1sm_bf16<...>(wpc, ...)
 *
 * 1SM (cta_group::1) counterpart of `_ntiles_2sm_bf16`. Same CLC-driven
 * persistent loop, just no peer / no cta_group::2 multicast / no
 * peer-bit mask. Per-iteration K-loop delegates to `_1tile_1sm_bf16`.
 *
 * Generic over GROUPED_GEMM via the same `expert_cumul_smem` /
 * `m_tile_remap` plumbing as the 2sm version. Phase-2 of grouped GEMM
 * (m_tile_remap pre-computed by host) and dense 1SM GEMM both fit.
 *
 * For 1SM, M_TILE_CLUSTER == M_TILE_PER_CTA and N_TILE_CLUSTER ==
 * N_TILE_PER_CTA (no peer split). They remain separate template params
 * for API symmetry with the 2sm version.
 * ============================================================================ */
template <int NUM_STAGES, int M_TILE_PER_CTA, int N_TILE_PER_CTA, int K_TILE,
          int M_TILE_CLUSTER, int N_TILE_CLUSTER,
          int LOAD_REG_BUDGET = 40,
          bool USE_L2_HINT = false,
          int CLUSTER_SHAPE_M = 1, int CLUSTER_SHAPE_N = 1,
          ClcRasterOrder ORDER = ClcRasterOrder::AlongN,
          bool GROUPED_GEMM = false,
          int L2CACHE_POLICY_A = 0, int L2CACHE_POLICY_B = 0>
__device__ inline
void load_warp_blackwell_ntiles_1sm_bf16(WpCtx& wpc,
    const CUtensorMap* tma_a, const CUtensorMap* tma_b,
    uint8_t* smem_a, uint8_t* smem_b,
    uint64_t* full_bar, uint64_t* empty_bar,
    uint64_t* clc_full_bar, uint64_t* clc_empty_bar, uint32_t* clc_response,
    uint64_t* throttle_full, uint64_t* throttle_empty,
    int K,
    int N = 0, int E = 0,
    const int* __restrict__ expert_cumul_smem = nullptr,
    const int* __restrict__ m_tile_remap = nullptr) {
  setmaxnreg_dec<LOAD_REG_BUDGET>();

  EmptyPhaseTracker<NUM_STAGES> empty_ph;
  int      clc_cons_stage = 0;
  uint32_t clc_cons_phase = 0;
  int      thr_prod_stage = 0;
  uint32_t thr_prod_phase = 1;

  auto remap = [&](int p1) -> int {
    return m_tile_remap ? m_tile_remap[p1] : p1;
  };

  int m_tile, n_tile;
  if constexpr (ORDER == ClcRasterOrder::AlongN) {
    m_tile = remap((int)blockIdx.y);
    n_tile = (int)blockIdx.x;
  } else {  // AlongM: blockIdx.x -> M (no /2 peer split for 1SM)
    m_tile = remap((int)blockIdx.x);
    n_tile = (int)blockIdx.y;
  }

  while (true) {
    const int m_offset = m_tile * M_TILE_CLUSTER;
    int n_offset_b     = n_tile * N_TILE_CLUSTER;
    if constexpr (GROUPED_GEMM) {
      const int expert_id = find_group_id(m_tile, expert_cumul_smem, E);
      n_offset_b += expert_id * N;
    }

    // Throttle handshake (producer side): ALL 32 lanes wait
    // throttle_empty, arrive throttle_full. Drives SCHED's throttle
    // consumer side. No peer gate for 1SM (single CTA).
    mbarrier_wait_parity(smem_ptr_u32(&throttle_empty[thr_prod_stage]),
                         thr_prod_phase);
    mbarrier_arrive_nostate(smem_ptr_u32(&throttle_full[thr_prod_stage]));
    advance_stage_phase<2>(thr_prod_stage, thr_prod_phase);

    // K-loop: ALL 32 lanes enter helper. Helper waits empty_bar and
    // advances empty_ph on all lanes (warp-uniform); elect_one_sync()
    // inside the helper gates arrive_expect_tx + TMA loads.
    load_warp_blackwell_1tile_1sm_bf16<
        NUM_STAGES, M_TILE_PER_CTA, N_TILE_PER_CTA, K_TILE,
        USE_L2_HINT, L2CACHE_POLICY_A, L2CACHE_POLICY_B>(wpc, tma_a, tma_b, smem_a, smem_b, full_bar, empty_bar,
        K, m_offset, n_offset_b, empty_ph);

    ClcTileInfo next = clc_fetch_next_tile<
        CLUSTER_SHAPE_M, CLUSTER_SHAPE_N, ORDER>(
        clc_full_bar, clc_empty_bar, clc_response,
        clc_cons_stage, clc_cons_phase, /*do_release=*/elect_one_sync());
    clc_fetch_next_tile_advance(clc_cons_stage, clc_cons_phase);
    if (!next.valid) break;
    m_tile = remap(next.m_tile);
    n_tile = next.n_tile;
  }

  // Tail drain so the pipeline drains cleanly before kernel exit.
  #pragma unroll
  for (int i = 0; i < NUM_STAGES; ++i) {
    const int s = empty_ph.get_stage();
    mbarrier_wait_parity(smem_ptr_u32(&empty_bar[s]),
                         empty_ph.get_phase());
    empty_ph.advance();
  }
}

/* ============================================================================
 * load_warp_blackwell_ntiles_1sm_bf16_gather4<...>(wpc, ...)
 *
 * 1SM (cta_group::1, no cluster) gather-fused load warp for MoE FC1.
 * Same gather4-driven A path as `_ntiles_2sm_bf16_gather`, but:
 *   - No peer / no peer1_done_bar / no cross-CTA handshake (1SM only
 *     issues into its own CTA's mbar, which is what gather4 supports
 *     natively -- the whole reason 2SM gather needed extra sync).
 *   - expect_tx = kStageABytes + kStageBBytes (own A gather + own B).
 *   - B uses single-CTA TMA (primitive 18 `tma_load_2d`), not 2SM
 *     multicast.
 *
 * A descriptor: box = (1, K_TILE), SWIZZLE_128B. M_TILE_PER_CTA / 4
 * gather4 calls per K-block (each fetches 4 rows).
 *
 * perm[packed_idx] = source row in A (-1 for pad slots, handled by
 * the descriptor's OOB-fill).
 *
 * PTX:    9.7.10.28.5.3 (TMA tile::gather4).
 * ============================================================================ */
template <int NUM_STAGES, int M_TILE_PER_CTA, int N_TILE_PER_CTA, int K_TILE,
          int M_TILE_CLUSTER, int N_TILE_CLUSTER,
          int LOAD_REG_BUDGET = 40,
          bool USE_L2_HINT = false,
          int CLUSTER_SHAPE_M = 1, int CLUSTER_SHAPE_N = 1,
          ClcRasterOrder ORDER = ClcRasterOrder::AlongN,
          int L2CACHE_POLICY_A = 0, int L2CACHE_POLICY_B = 0>
__device__ inline
void load_warp_blackwell_ntiles_1sm_bf16_gather4(WpCtx& wpc,
    const CUtensorMap* tma_a_gather,  // A: box=(1, K_TILE), SWIZZLE_128B
    const CUtensorMap* tma_b,         // B: standard, SWIZZLE_128B
    uint8_t* smem_a, uint8_t* smem_b,
    uint64_t* full_bar, uint64_t* empty_bar,
    uint64_t* clc_full_bar, uint64_t* clc_empty_bar, uint32_t* clc_response,
    uint64_t* throttle_full, uint64_t* throttle_empty,
    int K,
    int N, int E,
    const int* __restrict__ perm,
    const int* __restrict__ expert_cumul_smem,
    const int* __restrict__ m_tile_remap = nullptr) {
  setmaxnreg_dec<LOAD_REG_BUDGET>();
  static_assert(M_TILE_PER_CTA % 4 == 0,
                "gather4 fetches 4 rows per call; M_TILE_PER_CTA must be /4");
  static_assert(M_TILE_CLUSTER == M_TILE_PER_CTA && N_TILE_CLUSTER == N_TILE_PER_CTA,
                "1SM: cluster tile == per-CTA tile (no peer split)");

  constexpr int N_GATHER_PER_KBLOCK = M_TILE_PER_CTA / 4;
  constexpr int kElemBytes   = static_cast<int>(sizeof(__nv_bfloat16));
  constexpr int kStageABytes = M_TILE_PER_CTA * K_TILE * kElemBytes;
  constexpr int kStageBBytes = N_TILE_PER_CTA * K_TILE * kElemBytes;
  // 1SM: gather4 + B both signal LOCAL full_bar. No peer / no cross-CTA.
  const uint32_t expect_tx_local =
      (uint32_t)(kStageABytes + kStageBBytes);
  const int K_BLOCKS = K / K_TILE;
  const uint64_t cache_policy_a = make_l2cache_policy<L2CACHE_POLICY_A>();
  const uint64_t cache_policy_b = make_l2cache_policy<L2CACHE_POLICY_B>();
  (void)cache_policy_a;
  (void)cache_policy_b;

  EmptyPhaseTracker<NUM_STAGES> empty_ph;
  int      clc_cons_stage = 0;
  uint32_t clc_cons_phase = 0;
  int      thr_prod_stage = 0;
  uint32_t thr_prod_phase = 1;

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

  while (true) {
    const int expert_id       = find_group_id(m_tile, expert_cumul_smem, E);
    // Packed-token base index for THIS CTA's M_TILE_PER_CTA rows.
    const int packed_row_base = m_tile * M_TILE_CLUSTER;
    const int n_offset_b      = expert_id * N + n_tile * N_TILE_CLUSTER;

    mbarrier_wait_parity(smem_ptr_u32(&throttle_empty[thr_prod_stage]),
                         thr_prod_phase);
    mbarrier_arrive_nostate(smem_ptr_u32(&throttle_full[thr_prod_stage]));
    advance_stage_phase<2>(thr_prod_stage, thr_prod_phase);

    for (int k = 0; k < K_BLOCKS; ++k) {
      const int      s        = empty_ph.get_stage();
      const uint32_t full     = smem_ptr_u32(&full_bar[s]);
      const int      k_off    = k * K_TILE;
      const uint32_t smem_a_s = smem_ptr_u32(smem_a + s * kStageABytes);
      const uint32_t smem_b_s = smem_ptr_u32(smem_b + s * kStageBBytes);

      mbarrier_wait_parity(smem_ptr_u32(&empty_bar[s]),
                           empty_ph.get_phase());
      if (elect_one_sync()) {
        mbarrier_arrive_expect_tx(full, expect_tx_local);

        #pragma unroll
        for (int g = 0; g < N_GATHER_PER_KBLOCK; ++g) {
          const int r0 = perm[packed_row_base + g * 4 + 0];
          const int r1 = perm[packed_row_base + g * 4 + 1];
          const int r2 = perm[packed_row_base + g * 4 + 2];
          const int r3 = perm[packed_row_base + g * 4 + 3];
          const uint32_t dst = smem_a_s
                             + (uint32_t)(g * 4 * K_TILE * kElemBytes);
          if constexpr (USE_L2_HINT) {
            tma_load_2d_gather4_l2hint(dst, tma_a_gather, full,
                                       k_off, r0, r1, r2, r3,
                                       cache_policy_a);
          } else {
            tma_load_2d_gather4(dst, tma_a_gather, full,
                                k_off, r0, r1, r2, r3);
          }
        }

        if constexpr (USE_L2_HINT) {
          tma_load_2d_cta_l2hint(tma_b, smem_b_s, full,
                                 k_off, n_offset_b, cache_policy_b);
        } else {
          tma_load_2d_cta(tma_b, smem_b_s, full, k_off, n_offset_b);
        }
      }
      empty_ph.advance();
    }
    __syncwarp();

    const bool epi_release = elect_one_sync();
    ClcTileInfo next = clc_fetch_next_tile<
        CLUSTER_SHAPE_M, CLUSTER_SHAPE_N, ORDER, /*CTA_GROUP=*/1>(
        clc_full_bar, clc_empty_bar, clc_response,
        clc_cons_stage, clc_cons_phase, /*do_release=*/epi_release);
    clc_fetch_next_tile_advance(clc_cons_stage, clc_cons_phase);
    if (!next.valid) break;
    m_tile = remap(next.m_tile);
    n_tile = next.n_tile;
  }

  #pragma unroll
  for (int i = 0; i < NUM_STAGES; ++i) {
    const int s = empty_ph.get_stage();
    mbarrier_wait_parity(smem_ptr_u32(&empty_bar[s]),
                         empty_ph.get_phase());
    empty_ph.advance();
  }
}

/* ============================================================================
 * load_warp_blackwell_ntiles_1sm_bf16_gather4_parallel<...>(wpc, ...)
 *
 * Same contract as load_warp_blackwell_ntiles_1sm_bf16_gather4, but fixes its
 * single-thread serial gather4 bottleneck:
 *
 *   serial: one elected lane issues all N_GATHER_PER_KBLOCK (=32 for
 *        M_TILE=128) gather4 calls per K-block, re-reading perm[] each
 *        K-block -> 1024 serial issues + 4096 redundant perm reads per CTA.
 *
 *   here: (i) hoist perm[] indices out of the K loop -- each lane reads its
 *            own 4 row indices ONCE per tile (perm[] is K-invariant), held
 *            in registers; and
 *        (ii) lane-parallel issue -- lane g issues gather4 #g (rows
 *            [g*4, g*4+4)) so all 32 gather4 issue across the warp in
 *            parallel instead of serially from lane 0.
 *
 * mbar accounting unchanged: one elected lane arrives expect_tx(A+B) before
 * any issue (ordered by __syncwarp); all gather4 + B signal the same
 * full_bar[s], tx-bytes accumulate. Requires N_GATHER_PER_KBLOCK <= 32 so
 * each lane owns at most one gather group.
 * ============================================================================ */
template <int NUM_STAGES, int M_TILE_PER_CTA, int N_TILE_PER_CTA, int K_TILE,
          int M_TILE_CLUSTER, int N_TILE_CLUSTER,
          int LOAD_REG_BUDGET = 48,
          bool USE_L2_HINT = false,
          int CLUSTER_SHAPE_M = 1, int CLUSTER_SHAPE_N = 1,
          ClcRasterOrder ORDER = ClcRasterOrder::AlongN,
          int L2CACHE_POLICY_A = 0, int L2CACHE_POLICY_B = 0>
__device__ inline
void load_warp_blackwell_ntiles_1sm_bf16_gather4_parallel(WpCtx& wpc,
    const CUtensorMap* tma_a_gather,  // A: box=(1, K_TILE), SWIZZLE_128B
    const CUtensorMap* tma_b,         // B: standard, SWIZZLE_128B
    uint8_t* smem_a, uint8_t* smem_b,
    uint64_t* full_bar, uint64_t* empty_bar,
    uint64_t* clc_full_bar, uint64_t* clc_empty_bar, uint32_t* clc_response,
    uint64_t* throttle_full, uint64_t* throttle_empty,
    int K,
    int N, int E,
    const int* __restrict__ perm,
    const int* __restrict__ expert_cumul_smem,
    const int* __restrict__ m_tile_remap = nullptr) {
  setmaxnreg_dec<LOAD_REG_BUDGET>();
  static_assert(M_TILE_PER_CTA % 4 == 0,
                "gather4 fetches 4 rows per call; M_TILE_PER_CTA must be /4");
  static_assert(M_TILE_CLUSTER == M_TILE_PER_CTA && N_TILE_CLUSTER == N_TILE_PER_CTA,
                "1SM: cluster tile == per-CTA tile (no peer split)");
  constexpr int N_GATHER_PER_KBLOCK = M_TILE_PER_CTA / 4;
  static_assert(N_GATHER_PER_KBLOCK <= 32,
                "4A maps one gather group per lane; need N_GATHER <= warp size");

  constexpr int kElemBytes   = static_cast<int>(sizeof(__nv_bfloat16));
  constexpr int kStageABytes = M_TILE_PER_CTA * K_TILE * kElemBytes;
  constexpr int kStageBBytes = N_TILE_PER_CTA * K_TILE * kElemBytes;
  const uint32_t expect_tx_local =
      (uint32_t)(kStageABytes + kStageBBytes);
  const int K_BLOCKS = K / K_TILE;
  const uint64_t cache_policy_a = make_l2cache_policy<L2CACHE_POLICY_A>();
  const uint64_t cache_policy_b = make_l2cache_policy<L2CACHE_POLICY_B>();
  (void)cache_policy_a;
  (void)cache_policy_b;

  const int lane = threadIdx.x & 31;
  const int g    = lane;                          // gather group for this lane
  const bool active = (g < N_GATHER_PER_KBLOCK);  // lanes >= N_GATHER idle on A

  EmptyPhaseTracker<NUM_STAGES> empty_ph;
  int      clc_cons_stage = 0;
  uint32_t clc_cons_phase = 0;
  int      thr_prod_stage = 0;
  uint32_t thr_prod_phase = 1;

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

  while (true) {
    const int expert_id       = find_group_id(m_tile, expert_cumul_smem, E);
    const int packed_row_base = m_tile * M_TILE_CLUSTER;
    const int n_offset_b      = expert_id * N + n_tile * N_TILE_CLUSTER;

    // (i) Index prefetch: hoist this lane's 4 row indices ONCE per tile
    // (perm[] is K-invariant). Pad slots stay -1 (descriptor OOB-fill).
    int rr0 = -1, rr1 = -1, rr2 = -1, rr3 = -1;
    if (active) {
      rr0 = perm[packed_row_base + g * 4 + 0];
      rr1 = perm[packed_row_base + g * 4 + 1];
      rr2 = perm[packed_row_base + g * 4 + 2];
      rr3 = perm[packed_row_base + g * 4 + 3];
    }

    mbarrier_wait_parity(smem_ptr_u32(&throttle_empty[thr_prod_stage]),
                         thr_prod_phase);
    mbarrier_arrive_nostate(smem_ptr_u32(&throttle_full[thr_prod_stage]));
    advance_stage_phase<2>(thr_prod_stage, thr_prod_phase);

    for (int k = 0; k < K_BLOCKS; ++k) {
      const int      s        = empty_ph.get_stage();
      const uint32_t full     = smem_ptr_u32(&full_bar[s]);
      const int      k_off    = k * K_TILE;
      const uint32_t smem_a_s = smem_ptr_u32(smem_a + s * kStageABytes);
      const uint32_t smem_b_s = smem_ptr_u32(smem_b + s * kStageBBytes);

      mbarrier_wait_parity(smem_ptr_u32(&empty_bar[s]),
                           empty_ph.get_phase());
      // (ii) one lane sets expect_tx, ordered before any issue.
      if (elect_one_sync()) {
        mbarrier_arrive_expect_tx(full, expect_tx_local);
      }
      __syncwarp();
      // lane-parallel gather4: lane g issues its own 4-row strip.
      if (active) {
        const uint32_t dst = smem_a_s
                           + (uint32_t)(g * 4 * K_TILE * kElemBytes);
        if constexpr (USE_L2_HINT) {
          tma_load_2d_gather4_l2hint(dst, tma_a_gather, full,
                                     k_off, rr0, rr1, rr2, rr3,
                                     cache_policy_a);
        } else {
          tma_load_2d_gather4(dst, tma_a_gather, full,
                              k_off, rr0, rr1, rr2, rr3);
        }
      }
      if (elect_one_sync()) {
        if constexpr (USE_L2_HINT) {
          tma_load_2d_cta_l2hint(tma_b, smem_b_s, full,
                                 k_off, n_offset_b, cache_policy_b);
        } else {
          tma_load_2d_cta(tma_b, smem_b_s, full, k_off, n_offset_b);
        }
      }
      __syncwarp();
      empty_ph.advance();
    }
    __syncwarp();

    const bool epi_release = elect_one_sync();
    ClcTileInfo next = clc_fetch_next_tile<
        CLUSTER_SHAPE_M, CLUSTER_SHAPE_N, ORDER, /*CTA_GROUP=*/1>(
        clc_full_bar, clc_empty_bar, clc_response,
        clc_cons_stage, clc_cons_phase, /*do_release=*/epi_release);
    clc_fetch_next_tile_advance(clc_cons_stage, clc_cons_phase);
    if (!next.valid) break;
    m_tile = remap(next.m_tile);
    n_tile = next.n_tile;
  }

  #pragma unroll
  for (int i = 0; i < NUM_STAGES; ++i) {
    const int s = empty_ph.get_stage();
    mbarrier_wait_parity(smem_ptr_u32(&empty_bar[s]),
                         empty_ph.get_phase());
    empty_ph.advance();
  }
}

/* ============================================================================
 * load_warp_blackwell_ntiles_1sm_bf16_cpasync<...>(wpc, ...)
 *
 * cp.async gather-fused 1SM load warps for MoE FC1. Replaces the TMA
 * gather4 A-load with a TILED cp.async gather: NUM_LOAD_WARPS
 * warps (default 4 = 128 threads) cooperatively copy the A tile from the
 * UNPERMUTED activation, each row's GMEM address indirected through the
 * gather index. B stays on TMA. This attacks the DMA-bound gather (it is
 * throughput-bound, not issue-bound) by moving the bytes with
 * the full LSU of 128 threads instead of the TMA HW-gather. Mirrors QuACK's
 * use_tma_gather=False path (gemm_sm100.py: load_AB_gather_A /
 * copy_utils.py: gather_m_get_copy_fn).
 *
 * Thread map (128 load threads, K-major A tile of M_TILE x K_TILE bf16):
 *   THREADS_PER_ROW = (K_TILE*2)/16          (8 for K_TILE=64; 1 row=128B=8x16B)
 *   ROWS_PER_PASS   = 128 / THREADS_PER_ROW  (16)
 *   PASSES          = M_TILE / ROWS_PER_PASS (8)
 *   thread t: chunk = t % THREADS_PER_ROW (k = chunk*8), row_in_pass = t / TPR.
 *   The THREADS_PER_ROW threads of a row load adjacent 16B chunks ->
 *   each row is one coalesced 128B transaction.
 *
 * Index pipeline (CONSUMER; prefetch warp = producer, block 108):
 *   waits idx_full[idx_stage], caches each thread's PASSES src-rows from
 *   idx_smem_ring (perm[] is K-invariant -> read once per tile, reused over
 *   all K_BLOCKS), then releases idx_empty[idx_stage].
 *
 * full_bar[s] MIXED completion (keeps the MMA consumer unchanged):
 *   - B: one elected thread `mbarrier_arrive_expect_tx(full, B_bytes)` + TMA.
 *   - A: each of the 128 load threads `cp_async_mbarrier_arrive_noinc(full)`
 *        after its cp.async -> 128 arrives gated on cp.async completion
 *        (NOINC: pre-counted in full_bar's init arrive_count of 129).
 *   => full_bar init arrive_count MUST be 128 + 1 = 129; expected tx = B_bytes.
 *
 * A SMEM swizzle: cp.async writes addresses WE compute, so they must match
 * the MMA A descriptor's SWIZZLE_128B (smem_swizzle_b128 over byte offset
 * row*A_ROW_BYTES + chunk*16). A_ROW_BYTES = K_TILE*2 = one 128B atom.
 *
 * Named barrier LOAD_BAR_ID (default 9) syncs the NUM_LOAD_THREADS load
 * threads (index-read ordering + expect_tx-before-issue). Must not collide
 * with the kernel's other bar ids (MMA/epi use bar 6).
 *
 * Caller passes the RAW A base pointer (mA, M_unpermuted x K row-major),
 * NOT a TMA descriptor, for the A side.
 *
 * PTX: 9.7.10.28.3.1 (cp.async.cg), 9.7.10.28.3.x (cp.async.mbarrier.arrive).
 * ============================================================================ */
template <int NUM_STAGES, int M_TILE_PER_CTA, int N_TILE_PER_CTA, int K_TILE,
          int NUM_IDX_STAGES,
          int NUM_LOAD_WARPS = 4, int LOAD_WARP_ID0 = 2,
          int LOAD_BAR_ID = 9,
          int LOAD_REG_BUDGET = 48,
          bool USE_L2_HINT = false,
          ClcRasterOrder ORDER = ClcRasterOrder::AlongN,
          int L2CACHE_POLICY_A = 0, int L2CACHE_POLICY_B = 0>
__device__ inline
void load_warp_blackwell_ntiles_1sm_bf16_cpasync(WpCtx& wpc,
    const __nv_bfloat16* __restrict__ A_base,  // raw A (M_unpermuted, K) row-major
    const CUtensorMap* tma_b,                   // B: standard TMA, SWIZZLE_128B
    uint8_t* smem_a, uint8_t* smem_b,
    uint64_t* full_bar, uint64_t* empty_bar,
    const int* __restrict__ idx_smem_ring,      // [NUM_IDX_STAGES * M_TILE_PER_CTA]
    uint64_t* idx_full, uint64_t* idx_empty,
    uint64_t* clc_full_bar, uint64_t* clc_empty_bar, uint32_t* clc_response,
    uint64_t* throttle_full, uint64_t* throttle_empty,
    int K, int N, int E,
    const int* __restrict__ expert_cumul_smem,
    const int* __restrict__ m_tile_remap = nullptr) {
  setmaxnreg_dec<LOAD_REG_BUDGET>();

  constexpr int NUM_LOAD_THREADS = NUM_LOAD_WARPS * 32;
  constexpr int kElemBytes    = static_cast<int>(sizeof(__nv_bfloat16));
  constexpr int ELEMS_PER_CHUNK = 16 / kElemBytes;          // 8 bf16 per 16B
  constexpr int THREADS_PER_ROW = (K_TILE * kElemBytes) / 16;
  constexpr int ROWS_PER_PASS   = NUM_LOAD_THREADS / THREADS_PER_ROW;
  constexpr int PASSES          = M_TILE_PER_CTA / ROWS_PER_PASS;
  constexpr int A_ROW_BYTES     = K_TILE * kElemBytes;       // 128 for K_TILE=64
  constexpr int kStageABytes    = M_TILE_PER_CTA * K_TILE * kElemBytes;
  constexpr int kStageBBytes    = N_TILE_PER_CTA * K_TILE * kElemBytes;
  static_assert((K_TILE * kElemBytes) % 16 == 0, "row must be a multiple of 16B");
  static_assert(NUM_LOAD_THREADS % THREADS_PER_ROW == 0, "thread map: rows");
  static_assert(M_TILE_PER_CTA % ROWS_PER_PASS == 0, "thread map: passes");

  const int tid         = (int)threadIdx.x - LOAD_WARP_ID0 * 32;  // 0..127
  const int chunk       = tid % THREADS_PER_ROW;                  // 16B col -> k=chunk*8
  const int row_in_pass = tid / THREADS_PER_ROW;                  // 0..ROWS_PER_PASS-1

  const int K_BLOCKS = K / K_TILE;
  const uint64_t cache_policy_a = make_l2cache_policy<L2CACHE_POLICY_A>();
  const uint64_t cache_policy_b = make_l2cache_policy<L2CACHE_POLICY_B>();
  (void)cache_policy_a; (void)cache_policy_b;

  EmptyPhaseTracker<NUM_STAGES> empty_ph;
  int      clc_cons_stage = 0; uint32_t clc_cons_phase = 0;
  int      thr_prod_stage = 0; uint32_t thr_prod_phase = 1;
  int      idx_cons_stage = 0; uint32_t idx_cons_phase = 0;

  auto remap = [&](int p1) -> int {
    return m_tile_remap ? m_tile_remap[p1] : p1;
  };

  int m_tile, n_tile;
  if constexpr (ORDER == ClcRasterOrder::AlongN) {
    m_tile = remap((int)blockIdx.y); n_tile = (int)blockIdx.x;
  } else {
    m_tile = remap((int)blockIdx.x); n_tile = (int)blockIdx.y;
  }

  while (true) {
    const int expert_id  = find_group_id(m_tile, expert_cumul_smem, E);
    const int n_offset_b = expert_id * N + n_tile * N_TILE_PER_CTA;

    // ---- consume index slot: cache this thread's PASSES gather rows ----
    mbarrier_wait_parity(smem_ptr_u32(&idx_full[idx_cons_stage]), idx_cons_phase);
    const int* idx_tile = idx_smem_ring + idx_cons_stage * M_TILE_PER_CTA;
    int src_row[PASSES];
    #pragma unroll
    for (int p = 0; p < PASSES; ++p) {
      src_row[p] = idx_tile[p * ROWS_PER_PASS + row_in_pass];
    }
    asm volatile("bar.sync %0, %1;\n" :: "n"(LOAD_BAR_ID), "n"(NUM_LOAD_THREADS) : "memory");
    if (tid == 0) mbarrier_arrive_nostate(smem_ptr_u32(&idx_empty[idx_cons_stage]));
    advance_stage_phase<NUM_IDX_STAGES>(idx_cons_stage, idx_cons_phase);

    // ---- throttle (32 arrives, matches the single-warp sched config) ----
    if (tid < 32) {
      mbarrier_wait_parity(smem_ptr_u32(&throttle_empty[thr_prod_stage]),
                           thr_prod_phase);
      mbarrier_arrive_nostate(smem_ptr_u32(&throttle_full[thr_prod_stage]));
      advance_stage_phase<2>(thr_prod_stage, thr_prod_phase);
    }

    for (int k = 0; k < K_BLOCKS; ++k) {
      const int      s        = empty_ph.get_stage();
      const uint32_t full     = smem_ptr_u32(&full_bar[s]);
      const int      k_off    = k * K_TILE;
      const uint32_t smem_a_s = smem_ptr_u32(smem_a + s * kStageABytes);
      const uint32_t smem_b_s = smem_ptr_u32(smem_b + s * kStageBBytes);

      mbarrier_wait_parity(smem_ptr_u32(&empty_bar[s]), empty_ph.get_phase());

      // B: one elected thread sets expect_tx (B bytes only) then issues TMA.
      if (tid == 0) mbarrier_arrive_expect_tx(full, (uint32_t)kStageBBytes);
      asm volatile("bar.sync %0, %1;\n" :: "n"(LOAD_BAR_ID), "n"(NUM_LOAD_THREADS) : "memory");
      if (tid == 0) {
        if constexpr (USE_L2_HINT)
          tma_load_2d_cta_l2hint(tma_b, smem_b_s, full, k_off, n_offset_b, cache_policy_b);
        else
          tma_load_2d_cta(tma_b, smem_b_s, full, k_off, n_offset_b);
      }

      // A: tiled cp.async gather, swizzled SMEM dst to match the MMA desc.
      #pragma unroll
      for (int p = 0; p < PASSES; ++p) {
        const int row = p * ROWS_PER_PASS + row_in_pass;
        const int sr  = src_row[p];
        const uint32_t dst = smem_a_s
            + smem_swizzle_b128((uint32_t)(row * A_ROW_BYTES + chunk * 16));
        if (sr >= 0) {
          const __nv_bfloat16* gsrc =
              A_base + (size_t)sr * K + k_off + chunk * ELEMS_PER_CHUNK;
          if constexpr (USE_L2_HINT) cp_async_cg_16_l2hint(dst, gsrc, cache_policy_a);
          else                       cp_async_cg_16(dst, gsrc);
        } else {
          cp_async_cg_16_zfill(dst, A_base, /*zero_fill=*/true);  // pad row
        }
      }
      // A completion -> full_bar: cp.async.mbarrier.arrive.noinc defers an
      // arrive until this thread's cp.async land (NOINC: full_bar arrive_count
      // is pre-counted 129 = 1 expect_tx + 128 cp.async arrives). Async, so
      // the load warps keep pipelining instead of draining each k-block.
      cp_async_mbarrier_arrive_noinc(full);
      empty_ph.advance();
    }

    // CLC: all load lanes participate (arrive count), one releases.
    ClcTileInfo next = clc_fetch_next_tile<
        /*CSM=*/1, /*CSN=*/1, ORDER, /*CTA_GROUP=*/1>(
        clc_full_bar, clc_empty_bar, clc_response,
        clc_cons_stage, clc_cons_phase, /*do_release=*/(tid == 0));
    clc_fetch_next_tile_advance(clc_cons_stage, clc_cons_phase);
    if (!next.valid) break;
    m_tile = remap(next.m_tile);
    n_tile = next.n_tile;
  }

  #pragma unroll
  for (int i = 0; i < NUM_STAGES; ++i) {
    const int s = empty_ph.get_stage();
    mbarrier_wait_parity(smem_ptr_u32(&empty_bar[s]), empty_ph.get_phase());
    empty_ph.advance();
  }
}

/* ============================================================================
 * load_warp_blackwell_ntiles_2sm_bf16_cpasync<...>(wpc, ...)
 *
 * 2SM (cta_group::2) cp.async gather-fused load warps for MoE FC1. Each of
 * the 2 cluster CTAs gathers ITS OWN M_TILE_PER_CTA rows of A via cp.async
 * (the 2SM TMA splits the MTC-row A tile per-peer, so per-CTA-local gather
 * is correct); B is loaded by the 2SM multicast TMA. The MMA is cta_group::2.
 *
 * A-completion: cp.async.mbarrier.arrive is
 * shared::cta, so CTA1 cannot directly signal the LEADER's full_bar. The load
 * warp does NOT solve this -- it stays fully fire-and-forget: each peer arms
 * only its OWN local full_bar (128 cp.async.mbarrier.arrive.noinc + peer 1's 1
 * plain -> arrive_count = 129; peer 0 also carries the B expect_tx =
 * 2*B_bytes). The peer1->peer0 A-ready relay is done by peer 1's (otherwise
 * idle) MMA warp -- it waits its own full_bar[s] then
 * mbarrier.arrive.shared::cluster on peer 0's peer1_done_bar[s]. See
 * mma_warp_blackwell_ntiles_2sm_bf16_peer1relay. Keeping the wait off the
 * load warp is what makes the gather fully async.
 *
 * PTX: 9.7.10.28.5.3 (cta_group::2 TMA), 9.7.15.16.16 (mbarrier.arrive
 *      .shared::cluster), 9.7.10.28.3.* (cp.async).
 * ============================================================================ */
template <int NUM_STAGES, int M_TILE_PER_CTA, int N_TILE_PER_CTA, int K_TILE,
          int M_TILE_CLUSTER, int N_TILE_CLUSTER, int NUM_IDX_STAGES,
          int NUM_LOAD_WARPS = 4, int LOAD_WARP_ID0 = 2, int LOAD_BAR_ID = 9,
          int LOAD_REG_BUDGET = 48, bool USE_L2_HINT = false,
          int CLUSTER_SHAPE_M = 1, int CLUSTER_SHAPE_N = 2,
          ClcRasterOrder ORDER = ClcRasterOrder::AlongN,
          int L2CACHE_POLICY_A = 0, int L2CACHE_POLICY_B = 0>
__device__ inline
void load_warp_blackwell_ntiles_2sm_bf16_cpasync(WpCtx& wpc,
    const __nv_bfloat16* __restrict__ A_base,   // raw A (M_unpermuted, K)
    const CUtensorMap* tma_b,                    // B: cta_group::2 TMA
    uint8_t* smem_a, uint8_t* smem_b,
    uint64_t* full_bar, uint64_t* empty_bar,
    uint64_t* peer1_done_bar,                     // NUM_STAGES; arrive_count=1
    const int* __restrict__ idx_smem_ring,       // [NUM_IDX_STAGES * M_TILE_PER_CTA]
    uint64_t* idx_full, uint64_t* idx_empty,
    uint64_t* clc_full_bar, uint64_t* clc_empty_bar, uint32_t* clc_response,
    uint64_t* throttle_full, uint64_t* throttle_empty,
    int K, int N, int E, int peer,
    const int* __restrict__ expert_cumul_smem,
    const int* __restrict__ m_tile_remap = nullptr) {
  setmaxnreg_dec<LOAD_REG_BUDGET>();
  constexpr int NUM_LOAD_THREADS = NUM_LOAD_WARPS * 32;
  constexpr int kElemBytes    = static_cast<int>(sizeof(__nv_bfloat16));
  constexpr int ELEMS_PER_CHUNK = 16 / kElemBytes;
  constexpr int THREADS_PER_ROW = (K_TILE * kElemBytes) / 16;
  constexpr int ROWS_PER_PASS   = NUM_LOAD_THREADS / THREADS_PER_ROW;
  constexpr int PASSES          = M_TILE_PER_CTA / ROWS_PER_PASS;
  constexpr int A_ROW_BYTES     = K_TILE * kElemBytes;
  constexpr int kStageABytes    = M_TILE_PER_CTA * K_TILE * kElemBytes;
  constexpr int kStageBBytes    = N_TILE_PER_CTA * K_TILE * kElemBytes;
  // B only: 2*B_bytes (both peers via the cta_group::2 TMA).
  const uint32_t expect_tx_b = (uint32_t)(2 * kStageBBytes);
  static_assert(M_TILE_PER_CTA % ROWS_PER_PASS == 0, "thread map");

  const int tid         = (int)threadIdx.x - LOAD_WARP_ID0 * 32;
  const int chunk       = tid % THREADS_PER_ROW;
  const int row_in_pass = tid / THREADS_PER_ROW;
  const int K_BLOCKS    = K / K_TILE;
  const uint64_t cache_policy_a = make_l2cache_policy<L2CACHE_POLICY_A>();
  const uint64_t cache_policy_b = make_l2cache_policy<L2CACHE_POLICY_B>();
  (void)cache_policy_a; (void)cache_policy_b;

  EmptyPhaseTracker<NUM_STAGES> empty_ph;
  int      clc_cons_stage = 0; uint32_t clc_cons_phase = 0;
  int      thr_prod_stage = 0; uint32_t thr_prod_phase = 1;
  int      idx_cons_stage = 0; uint32_t idx_cons_phase = 0;

  auto remap = [&](int p1) -> int { return m_tile_remap ? m_tile_remap[p1] : p1; };
  int m_tile, n_tile;
  if constexpr (ORDER == ClcRasterOrder::AlongN) {
    m_tile = remap((int)blockIdx.y); n_tile = (int)blockIdx.x >> 1;  // /2: peers split N
  } else {
    m_tile = remap((int)blockIdx.x >> 1); n_tile = (int)blockIdx.y;
  }

  while (true) {
    const int expert_id  = find_group_id(m_tile, expert_cumul_smem, E);
    // 2SM: each peer loads its own contiguous N-half of the cluster B tile.
    const int n_offset_b = expert_id * N + n_tile * N_TILE_CLUSTER
                         + peer * N_TILE_PER_CTA;

    // Consume this CTA's index slice (prefetch staged peer's M_TILE_PER_CTA rows).
    wp_begin(wpc, WP_LOAD_WAIT);  // wait for prefetch warp to stage this tile's idx ring slot
    mbarrier_wait_parity(smem_ptr_u32(&idx_full[idx_cons_stage]), idx_cons_phase);
    wp_end(wpc, WP_LOAD_WAIT);
    const int* idx_tile = idx_smem_ring + idx_cons_stage * M_TILE_PER_CTA;
    int src_row[PASSES];
    #pragma unroll
    for (int p = 0; p < PASSES; ++p)
      src_row[p] = idx_tile[p * ROWS_PER_PASS + row_in_pass];
    asm volatile("bar.sync %0, %1;\n" :: "n"(LOAD_BAR_ID), "n"(NUM_LOAD_THREADS) : "memory");
    if (tid == 0) mbarrier_arrive_nostate(smem_ptr_u32(&idx_empty[idx_cons_stage]));
    advance_stage_phase<NUM_IDX_STAGES>(idx_cons_stage, idx_cons_phase);

    // Throttle handshake: ONLY peer 0 (its sched warp throttles on
    // cluster_rank==0). 32 arrives = one load warp's worth (warp LOAD_WARP_ID0).
    if (peer == 0 && tid < 32) {
      wp_begin(wpc, WP_LOAD_WAIT_THROTTLE);  // scheduler pacing (peer 0 warp 2 only)
      mbarrier_wait_parity(smem_ptr_u32(&throttle_empty[thr_prod_stage]), thr_prod_phase);
      mbarrier_arrive_nostate(smem_ptr_u32(&throttle_full[thr_prod_stage]));
      advance_stage_phase<2>(thr_prod_stage, thr_prod_phase);
      wp_end(wpc, WP_LOAD_WAIT_THROTTLE);
    }

    for (int k = 0; k < K_BLOCKS; ++k) {
      const int      s        = empty_ph.get_stage();
      const uint32_t full     = smem_ptr_u32(&full_bar[s]);
      const int      k_off    = k * K_TILE;
      const uint32_t smem_a_s = smem_ptr_u32(smem_a + s * kStageABytes);
      const uint32_t smem_b_s = smem_ptr_u32(smem_b + s * kStageBBytes);

      wp_begin(wpc, WP_LOAD_WAIT);  // backpressure: wait for a free SMEM stage (empty_bar)
      mbarrier_wait_parity(smem_ptr_u32(&empty_bar[s]), empty_ph.get_phase());
      wp_end(wpc, WP_LOAD_WAIT);

      // B: peer 0 sets expect_tx (B only); elected thread issues the 2SM TMA.
      wp_begin(wpc, WP_LOAD_ISSUE);
      if (tid == 0 && peer == 0) mbarrier_arrive_expect_tx(full, expect_tx_b);
      asm volatile("bar.sync %0, %1;\n" :: "n"(LOAD_BAR_ID), "n"(NUM_LOAD_THREADS) : "memory");
      if (tid == 0) {
        const uint32_t full_route = tma_peer_bit_mask(full);
        if constexpr (USE_L2_HINT)
          tma_load_2d_2sm_l2hint(smem_b_s, tma_b, full_route, k_off, n_offset_b, cache_policy_b);
        else
          tma_load_2d_2sm(smem_b_s, tma_b, full_route, k_off, n_offset_b);
      }
      wp_end(wpc, WP_LOAD_ISSUE);

      // A: per-CTA cp.async gather (swizzled dst), then drain + leader arrive.
      wp_begin(wpc, WP_LOAD_GATHER);
      #pragma unroll
      for (int p = 0; p < PASSES; ++p) {
        const int row = p * ROWS_PER_PASS + row_in_pass;
        const int sr  = src_row[p];
        const uint32_t dst = smem_a_s
            + smem_swizzle_b128((uint32_t)(row * A_ROW_BYTES + chunk * 16));
        if (sr >= 0) {
          const __nv_bfloat16* gsrc = A_base + (size_t)sr * K + k_off + chunk * ELEMS_PER_CHUNK;
          if constexpr (USE_L2_HINT) cp_async_cg_16_l2hint(dst, gsrc, cache_policy_a);
          else                       cp_async_cg_16(dst, gsrc);
        } else {
          cp_async_cg_16_zfill(dst, A_base, /*zero_fill=*/true);
        }
      }
      // A-ready: each peer arms its OWN full_bar, fire-and-forget. count=129:
      // 128 cp.async.noinc + (peer0: B expect_tx | peer1: 1 plain). The
      // peer1->peer0 relay is on peer1's MMA warp (..._gather_async), not here.
      // Keep BRANCHED on peer -- faster than unconditional (codegen).
      if (peer == 0) {
        cp_async_mbarrier_arrive_noinc(full);   // 128 .noinc; B expect_tx is the 129th
      } else {
        cp_async_mbarrier_arrive_noinc(full);
        // peer1 has no B expect_tx, so the 128 .noinc leave full at 128/129;
        // tid0's 1 plain arrive supplies the missing 129th (else it stalls).
        if (tid == 0) mbarrier_arrive_nostate(full);
      }
      wp_end(wpc, WP_LOAD_GATHER);
      empty_ph.advance();
    }

    wp_begin(wpc, WP_CLC_FETCH);
    ClcTileInfo next = clc_fetch_next_tile<
        CLUSTER_SHAPE_M, CLUSTER_SHAPE_N, ORDER, /*CTA_GROUP=*/2>(
        clc_full_bar, clc_empty_bar, clc_response,
        clc_cons_stage, clc_cons_phase, /*do_release=*/(tid == 0));
    wp_end(wpc, WP_CLC_FETCH);
    clc_fetch_next_tile_advance(clc_cons_stage, clc_cons_phase);
    if (!next.valid) break;
    m_tile = remap(next.m_tile);
    n_tile = next.n_tile;
  }

  wp_begin(wpc, WP_LOAD_WAIT);  // tail: drain remaining empty_bar slots
  #pragma unroll
  for (int i = 0; i < NUM_STAGES; ++i) {
    const int s = empty_ph.get_stage();
    mbarrier_wait_parity(smem_ptr_u32(&empty_bar[s]), empty_ph.get_phase());
    empty_ph.advance();
  }
  wp_end(wpc, WP_LOAD_WAIT);
}

/* ============================================================================
 * load_warp_blackwell_ntiles_1sm_tail_bf16<...>(wpc, ...)
 *
 * 1SM CLC-driven load warp for grouped-GEMM phase-2 tail. Same persistent
 * loop as `_ntiles_1sm_bf16`, but decodes per-tile (m_offset, n_offset_b)
 * from `tail_m_offset[m_tile]` + `tail_expert_id[m_tile] * N` lookup tables
 * (host-built; one entry per phase-2 cluster).
 *
 * Throttle handshake driven by ALL 32 lanes (matches sched_warp_clc's
 * 32-arrive throttle_full consumer side).
 * ============================================================================ */
template <int NUM_STAGES, int M_TILE_PER_CTA, int N_TILE_PER_CTA, int K_TILE,
          int LOAD_REG_BUDGET = 40,
          bool USE_L2_HINT = false,
          int CLUSTER_SHAPE_M = 1, int CLUSTER_SHAPE_N = 1,
          ClcRasterOrder ORDER = ClcRasterOrder::AlongN,
          int L2CACHE_POLICY_A = 0, int L2CACHE_POLICY_B = 0>
__device__ inline
void load_warp_blackwell_ntiles_1sm_tail_bf16(WpCtx& wpc,
    const CUtensorMap* tma_a, const CUtensorMap* tma_b,
    uint8_t* smem_a, uint8_t* smem_b,
    uint64_t* full_bar, uint64_t* empty_bar,
    uint64_t* clc_full_bar, uint64_t* clc_empty_bar, uint32_t* clc_response,
    uint64_t* throttle_full, uint64_t* throttle_empty,
    int N, int K,
    const int* __restrict__ tail_m_offset,
    const int* __restrict__ tail_expert_id) {
  setmaxnreg_dec<LOAD_REG_BUDGET>();

  EmptyPhaseTracker<NUM_STAGES> empty_ph;
  int      clc_cons_stage = 0;
  uint32_t clc_cons_phase = 0;
  int      thr_prod_stage = 0;
  uint32_t thr_prod_phase = 1;

  int m_tile, n_tile;
  if constexpr (ORDER == ClcRasterOrder::AlongN) {
    m_tile = (int)blockIdx.y;
    n_tile = (int)blockIdx.x;
  } else {
    m_tile = (int)blockIdx.x;
    n_tile = (int)blockIdx.y;
  }

  while (true) {
    const int m_offset   = tail_m_offset[m_tile];
    const int expert_id  = tail_expert_id[m_tile];
    const int n_offset_b = expert_id * N + n_tile * N_TILE_PER_CTA;

    // Throttle handshake (producer side): ALL 32 lanes wait
    // throttle_empty, arrive throttle_full.
    mbarrier_wait_parity(smem_ptr_u32(&throttle_empty[thr_prod_stage]),
                         thr_prod_phase);
    mbarrier_arrive_nostate(smem_ptr_u32(&throttle_full[thr_prod_stage]));
    advance_stage_phase<2>(thr_prod_stage, thr_prod_phase);

    load_warp_blackwell_1tile_1sm_bf16<
        NUM_STAGES, M_TILE_PER_CTA, N_TILE_PER_CTA, K_TILE,
        USE_L2_HINT, L2CACHE_POLICY_A, L2CACHE_POLICY_B>(wpc, tma_a, tma_b, smem_a, smem_b, full_bar, empty_bar,
        K, m_offset, n_offset_b, empty_ph);

    ClcTileInfo next = clc_fetch_next_tile<
        CLUSTER_SHAPE_M, CLUSTER_SHAPE_N, ORDER, /*CTA_GROUP=*/1>(
        clc_full_bar, clc_empty_bar, clc_response,
        clc_cons_stage, clc_cons_phase, /*do_release=*/elect_one_sync());
    clc_fetch_next_tile_advance(clc_cons_stage, clc_cons_phase);
    if (!next.valid) break;
    m_tile = next.m_tile;
    n_tile = next.n_tile;
  }

  // Tail drain.
  #pragma unroll
  for (int i = 0; i < NUM_STAGES; ++i) {
    const int s = empty_ph.get_stage();
    mbarrier_wait_parity(smem_ptr_u32(&empty_bar[s]),
                         empty_ph.get_phase());
    empty_ph.advance();
  }
}

/* ============================================================================
 * load_warp_blackwell_1tile_1sm_bf16_fmha<>(wpc, ...)
 *
 * One FMHA work-item's loads (decoded coords in): K and V time-share one
 * NUM_KV_STAGES ring (two stages consumed per K-block: K then V), issued in
 * DESCENDING token order; Q[m] on per-M-tile single-slot pipelines
 * (full_bar_q/empty_bar_q), loaded once at k==0, interleaved between K and V;
 * 4D per-sample Q box (rows past seqlen_q zero-fill, not the next sample).
 * Load order: K[K_TILES-1], Q0, Q1, V[K_TILES-1], K[K_TILES-2], V[K_TILES-2],
 * ..., K[0], V[0].
 *
 * ALL 32 lanes enter (elect_one_sync inside gates arrive_expect_tx + TMA).
 * Trackers are caller-persistent (by reference) so phase carries across the
 * persistent loop.
 * ============================================================================ */
template <int NUM_KV_STAGES, int M_TILES_PER_CTA, int M_TILE, int K_TILE, int HEAD_DIM>
__device__ inline
void load_warp_blackwell_1tile_1sm_bf16_fmha(WpCtx& wpc,
    const CUtensorMap* tmap_q, const CUtensorMap* tmap_k, const CUtensorMap* tmap_v_t,
    uint8_t* sQ0, uint8_t* sKV,
    uint64_t* full_bar, uint64_t* empty_bar,
    uint64_t* full_bar_q, uint64_t* empty_bar_q,
    int sample, int h_kv, int q_tile_base, int K_TILES,
    int seqlen_kv, int q_tile_per_mtile, int gqa_group_size,
    EmptyPhaseTracker<NUM_KV_STAGES>& kv_empty_ph, EmptyPhaseTracker<1>& q_empty_ph,
    int v_sample_stride = 0) {
  // B128 swizzle atom = 128 bytes = 64 bf16: all SMEM tiles are laid out in
  // 64-wide sub-tiles along the contiguous dim.
  constexpr int SUB_COLS_BF16   = 64;
  constexpr int SUB_COLS_BYTES  = SUB_COLS_BF16 * (int)sizeof(__nv_bfloat16);
  constexpr int Q_SUBTILES      = HEAD_DIM / SUB_COLS_BF16;
  constexpr int K_SUBTILES      = HEAD_DIM / SUB_COLS_BF16;
  constexpr int V_SUBTILES      = K_TILE / SUB_COLS_BF16;
  constexpr int Q_SUB_COLS_BYTES = M_TILE * SUB_COLS_BYTES;
  constexpr int V_SUB_COLS_BYTES = HEAD_DIM * SUB_COLS_BYTES;
  constexpr int Q_TILE_BYTES    = Q_SUBTILES * Q_SUB_COLS_BYTES;
  constexpr int K_TILE_BYTES    = K_SUBTILES * (K_TILE * SUB_COLS_BYTES);
  constexpr int V_TILE_BYTES    = V_SUBTILES * V_SUB_COLS_BYTES;
  const int k_start = sample * seqlen_kv;
  // V_T may pad each sample to meet the TMA box's 16-byte GMEM alignment.
  const int v_start = sample * (v_sample_stride > 0 ? v_sample_stride : seqlen_kv);

  for (int k = 0; k < K_TILES; ++k) {
    const int k_offset = (K_TILES - 1 - k) * K_TILE;
    int kv_stage = kv_empty_ph.get_stage();

    wp_begin(wpc, WP_LOAD_WAIT);
    mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar[kv_stage]), kv_empty_ph.get_phase());
    kv_empty_ph.advance();
    wp_end(wpc, WP_LOAD_WAIT);

    wp_begin(wpc, WP_LOAD_ISSUE_K);
    if (elect_one_sync()) {
      mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar[kv_stage]), K_TILE_BYTES);
      // ONE 3D copy folds both head-dim swizzle atoms (vs 2 x 2D). coords
      // {atom-col 0, token, atom h_kv*K_SUBTILES}; box [SUB_COLS_BF16, K_TILE, K_SUBTILES].
      tma_load_3d(smem_ptr_u32(sKV + kv_stage * K_TILE_BYTES), tmap_k, smem_ptr_u32(&full_bar[kv_stage]),
                  0, k_start + k_offset, h_kv * K_SUBTILES);
    }
    wp_end(wpc, WP_LOAD_ISSUE_K);

    if (k == 0 && elect_one_sync()) {
      // 4D Q box (token-in-sample dim): rows past seqlen_q zero-fill, not the next sample.
      #pragma unroll
      for (int m = 0; m < M_TILES_PER_CTA; ++m) {
        wp_begin(wpc, WP_LOAD_WAIT);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_q[m]), q_empty_ph.get_phase());
        wp_end(wpc, WP_LOAD_WAIT);

        wp_begin(wpc, WP_LOAD_ISSUE_Q);
        mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar_q[m]), Q_TILE_BYTES);
        const int tok0 = q_tile_base + m * q_tile_per_mtile;
        // box [hd/2 x gqa_group_size x q_tile_per_mtile x 1], qh-inner packed rows.
        #pragma unroll
        for (int s = 0; s < Q_SUBTILES; ++s) {
          tma_load_4d(smem_ptr_u32(sQ0 + m * Q_TILE_BYTES + s * Q_SUB_COLS_BYTES), tmap_q,
                      smem_ptr_u32(&full_bar_q[m]),
                      s * SUB_COLS_BF16, h_kv * gqa_group_size, tok0, sample);
        }
        wp_end(wpc, WP_LOAD_ISSUE_Q);
      }
      q_empty_ph.advance();
    }
    kv_stage = kv_empty_ph.get_stage();

    wp_begin(wpc, WP_LOAD_WAIT);
    mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar[kv_stage]), kv_empty_ph.get_phase());
    kv_empty_ph.advance();
    wp_end(wpc, WP_LOAD_WAIT);

    wp_begin(wpc, WP_LOAD_ISSUE_V);
    if (elect_one_sync()) {
      mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar[kv_stage]), V_TILE_BYTES);
      #pragma unroll
      for (int s = 0; s < V_SUBTILES; ++s) {
        tma_load_2d(smem_ptr_u32(sKV + kv_stage * K_TILE_BYTES + s * V_SUB_COLS_BYTES), tmap_v_t,
                    smem_ptr_u32(&full_bar[kv_stage]),
                    v_start + k_offset + s * SUB_COLS_BF16, h_kv * HEAD_DIM);
      }
    }
    wp_end(wpc, WP_LOAD_ISSUE_V);
  }
  __syncwarp();  // converge after the elect-issued TMA loads
}

// Forward decl: the 2SM 1tile body is defined below, but the unified ntiles
// wrapper (CLUSTER_N == 2 branch) instantiates it above the definition.
template <int NUM_KV_STAGES, int M_TILES_PER_CTA, int M_TILE, int K_TILE, int HEAD_DIM>
__device__ inline
void load_warp_blackwell_1tile_2sm_bf16_fmha(WpCtx& wpc,
    const CUtensorMap* tmap_q, const CUtensorMap* tmap_k, const CUtensorMap* tmap_v_t,
    uint8_t* const* sQ, uint8_t* sKV,
    uint64_t* full_bar, uint64_t* empty_bar, uint64_t* full_bar_q, uint64_t* empty_bar_q,
    int peer, int sample, int h_kv, int q_tb_eff, int K_TILES,
    int k_start, int q_tile_per_mtile, int gqa_group_size,
    EmptyPhaseTracker<NUM_KV_STAGES>& kv_empty_ph, EmptyPhaseTracker<1>& q_empty_ph);

/* ============================================================================
 * load_warp_blackwell_ntiles_1sm2sm_bf16_fmha<>(wpc, ...)
 *
 * Production __device__ body for the LOAD warp of the Blackwell BF16 FMHA
 * context kernels (uniform seqlen, 1SM + 2SM): persistent loop of
 * decode_workitem (#110) + throttle producer, driven by CLC (#106) or grid-stride.
 *
 * CLUSTER_N selects the per-work-item 1tile body:
 *   CLUSTER_N == 1 -> load_warp_blackwell_1tile_1sm_bf16_fmha (wpc, full Q/K/V tiles).
 *   CLUSTER_N == 2 -> load_warp_blackwell_1tile_2sm_bf16_fmha (wpc, cta_group::2:
 *     Q M-split, K/V N-split half-boxes, peer bit mask, peer-0 arrive_expect_tx).
 * Only the wrapper is shared; the two 1tile bodies stay separate because their
 * inner-loop TMA instructions differ (2sm variants + half-box coords).
 * ============================================================================ */
template <int NUM_KV_STAGES, int M_TILES_PER_CTA, int M_TILE, int K_TILE, int HEAD_DIM,
          bool USE_CLC, int CLC_STAGES, bool Q_RASTER, bool IS_CAUSAL, bool LPT,
          int LOAD_REG_BUDGET = 48, bool TAIL_DRAIN = true, int CLUSTER_N = 1>
__device__ inline
void load_warp_blackwell_ntiles_1sm2sm_bf16_fmha(WpCtx& wpc,
    const CUtensorMap* tmap_q, const CUtensorMap* tmap_k, const CUtensorMap* tmap_v_t,
    uint8_t* sQ0, uint8_t* sKV,
    uint64_t* full_bar, uint64_t* empty_bar,
    uint64_t* full_bar_q, uint64_t* empty_bar_q,
    uint64_t* clc_full, uint64_t* clc_empty, uint32_t* clc_response,
    uint64_t* throttle_full, uint64_t* throttle_empty,
    int seqlen_kv, int num_q_heads, int num_kv_heads, int packed_idx_per_seq, int num_samples,
    unsigned long long magic0, unsigned long long magic1, unsigned long long magic2,
    int seqlen_q = -1, int v_sample_stride = 0) {
  // seqlen_q: 2SM padding-peer Q-validity clamp; <0 -> = seqlen_kv (self-attn).
  // v_sample_stride: optional 1SM V_T storage stride; zero keeps the legacy packed layout.
  setmaxnreg_dec<LOAD_REG_BUDGET>();
  const int peer = (int)(blockIdx.x % CLUSTER_N);
  if (seqlen_q < 0) seqlen_q = seqlen_kv;

  const int gqa_group_size    = num_q_heads / num_kv_heads;
  const int q_tile_per_mtile  = M_TILE / gqa_group_size;
  const int q_tile_per_cta    = M_TILES_PER_CTA * q_tile_per_mtile;
  const int total_tiles       = num_samples * packed_idx_per_seq * num_kv_heads;

  EmptyPhaseTracker<NUM_KV_STAGES> kv_empty_ph;
  EmptyPhaseTracker<1> q_empty_ph;
  [[maybe_unused]] int clc_stage = 0;
  [[maybe_unused]] uint32_t clc_phase = 0;
  [[maybe_unused]] int thr_prod_stage = 0;
  [[maybe_unused]] uint32_t thr_prod_phase = 1;
  int tile_id = (int)(blockIdx.x / CLUSTER_N);
  while (true) {
    int sample, h_kv, q_tile_base, K_TILES;
    decode_workitem<K_TILE, Q_RASTER, IS_CAUSAL, LPT, CLUSTER_N>(tile_id, seqlen_kv, num_kv_heads,
        packed_idx_per_seq, q_tile_per_cta, magic0, magic1, magic2,
        sample, h_kv, q_tile_base, K_TILES, peer);

    // Throttle producer paces the sched warp; only the leader CTA runs it. peer is
    // always 0 for CLUSTER_N == 1, so `peer == 0` covers both the 1SM and 2SM paths.
    if constexpr (USE_CLC) {
      if (peer == 0) {
        wp_begin(wpc, WP_LOAD_WAIT_THROTTLE);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&throttle_empty[thr_prod_stage]), thr_prod_phase);
        wp_end(wpc, WP_LOAD_WAIT_THROTTLE);
        mbarrier_arrive_nostate(smem_ptr_u32(&throttle_full[thr_prod_stage]));
        advance_stage_phase<2>(thr_prod_stage, thr_prod_phase);
      }
    }

    if constexpr (CLUSTER_N == 2) {
      // 2SM: the unpaired padding peer must NOT bail (the leader's cta_group::2 MMA +
      // cross-CTA barriers would deadlock); it loads a clamped in-bounds PADDING Q and
      // runs in lockstep. sQ0/sQ1 are contiguous, so rebuild the per-m-tile array here.
      constexpr int Q_TILE_BYTES = (HEAD_DIM / 64) * (M_TILE * 64 * (int)sizeof(__nv_bfloat16));
      const int q_tb_eff = (q_tile_base < seqlen_q) ? q_tile_base : 0;   // Q validity (cross: seqlen_q)
      const int k_start  = sample * seqlen_kv;                              // K packing stride (seqlen_kv)
      uint8_t* sQ[M_TILES_PER_CTA];
      #pragma unroll
      for (int m = 0; m < M_TILES_PER_CTA; ++m) sQ[m] = sQ0 + m * Q_TILE_BYTES;
      load_warp_blackwell_1tile_2sm_bf16_fmha<
          NUM_KV_STAGES, M_TILES_PER_CTA, M_TILE, K_TILE, HEAD_DIM>(wpc, tmap_q, tmap_k, tmap_v_t, sQ, sKV,
          full_bar, empty_bar, full_bar_q, empty_bar_q,
          peer, sample, h_kv, q_tb_eff, K_TILES, k_start, q_tile_per_mtile, gqa_group_size,
          kv_empty_ph, q_empty_ph);
    } else {
      load_warp_blackwell_1tile_1sm_bf16_fmha<
          NUM_KV_STAGES, M_TILES_PER_CTA, M_TILE, K_TILE, HEAD_DIM>(wpc, tmap_q, tmap_k, tmap_v_t, sQ0, sKV,
          full_bar, empty_bar, full_bar_q, empty_bar_q,
          sample, h_kv, q_tile_base, K_TILES,
          seqlen_kv, q_tile_per_mtile, gqa_group_size,
          kv_empty_ph, q_empty_ph, v_sample_stride);
    }

    if constexpr (USE_CLC) {
      ClcTileInfo next = clc_fetch_next_tile<1, CLUSTER_N, ClcRasterOrder::AlongN, /*CTA_GROUP=*/CLUSTER_N>(
          clc_full, clc_empty, clc_response, clc_stage, clc_phase, /*do_release=*/elect_one_sync());
      clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
      if (!next.valid) break;
      tile_id = next.n_tile;
    } else {
      tile_id += (int)(gridDim.x / CLUSTER_N);
      if (tile_id >= total_tiles) break;
    }
  }
  // Tail drain: MMA's last empty_bar / empty_bar_q arrives (phase-init borrowed
  // NUM_KV_STAGES + M_TILES_PER_CTA up front). empty_bar is all-lanes; empty_bar_q
  // is elect-lane only, matching the main loop (only that lane advances q_empty_ph).
  if constexpr (TAIL_DRAIN) {
    wp_begin(wpc, WP_LOAD_WAIT);
    #pragma unroll
    for (int s = 0; s < NUM_KV_STAGES; ++s) {
      const int kv_stage = kv_empty_ph.get_stage();
      mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar[kv_stage]), kv_empty_ph.get_phase());
      kv_empty_ph.advance();
    }
    if (elect_one_sync()) {
      #pragma unroll
      for (int m = 0; m < M_TILES_PER_CTA; ++m) {
        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_q[m]), q_empty_ph.get_phase());
      }
    }
    wp_end(wpc, WP_LOAD_WAIT);
  }
}

/* ============================================================================
 * load_warp_blackwell_1tile_2sm_bf16_fmha<>(wpc, ...)
 *
 * One FMHA work-item's TMA loads for the 2SM (cta_group::2) cluster:
 *   - Q  M-split : each peer loads its own 128-row Q tile (tma_load_4d_2sm);
 *                  peer 0 sizes arrive_expect_tx for BOTH peers' Q (2*Q_TILE_BYTES).
 *   - K  N-split : each peer loads a K_TILE/2 half-box along kv-token (tma_load_3d_2sm).
 *   - V  N-split : each peer loads a HEAD_DIM/2 half along head-dim (tma_load_2d_2sm).
 * Only peer 0 issues arrive_expect_tx; every TMA routes via tma_peer_bit_mask so
 * cta_group::2 delivers complete_tx to peer 0's barrier. Trackers by reference.
 * ============================================================================ */
template <int NUM_KV_STAGES, int M_TILES_PER_CTA, int M_TILE, int K_TILE, int HEAD_DIM>
__device__ inline
void load_warp_blackwell_1tile_2sm_bf16_fmha(WpCtx& wpc,
    const CUtensorMap* tmap_q, const CUtensorMap* tmap_k, const CUtensorMap* tmap_v_t,
    uint8_t* const* sQ, uint8_t* sKV,
    uint64_t* full_bar, uint64_t* empty_bar, uint64_t* full_bar_q, uint64_t* empty_bar_q,
    int peer, int sample, int h_kv, int q_tb_eff, int K_TILES,
    int k_start, int q_tile_per_mtile, int gqa_group_size,
    EmptyPhaseTracker<NUM_KV_STAGES>& kv_empty_ph, EmptyPhaseTracker<1>& q_empty_ph) {
  constexpr int SUB_COLS_BF16    = 64;
  constexpr int SUB_COLS_BYTES   = SUB_COLS_BF16 * (int)sizeof(__nv_bfloat16);
  constexpr int Q_SUBTILES       = HEAD_DIM / SUB_COLS_BF16;
  constexpr int K_SUBTILES       = HEAD_DIM / SUB_COLS_BF16;
  constexpr int V_SUBTILES       = K_TILE / SUB_COLS_BF16;
  constexpr int Q_SUB_COLS_BYTES = M_TILE * SUB_COLS_BYTES;
  constexpr int Q_TILE_BYTES     = Q_SUBTILES * Q_SUB_COLS_BYTES;
  constexpr int K_TILE_BYTES     = K_SUBTILES * (K_TILE * SUB_COLS_BYTES);
  constexpr int V_TILE_BYTES     = V_SUBTILES * Q_SUB_COLS_BYTES;

  for (int k = 0; k < K_TILES; ++k) {
    const int k_offset = (K_TILES - 1 - k) * K_TILE;
    int kv_stage = kv_empty_ph.get_stage();

    wp_begin(wpc, WP_LOAD_WAIT);
    mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar[kv_stage]), kv_empty_ph.get_phase());
    kv_empty_ph.advance();
    wp_end(wpc, WP_LOAD_WAIT);

    wp_begin(wpc, WP_LOAD_ISSUE_K);
    if (elect_one_sync()) {
      if (peer == 0) mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar[kv_stage]), K_TILE_BYTES);
      const uint32_t kbar = tma_peer_bit_mask(smem_ptr_u32(&full_bar[kv_stage]));
      tma_load_3d_2sm(smem_ptr_u32(sKV + kv_stage * K_TILE_BYTES), tmap_k, kbar,
                      0, k_start + k_offset + peer * (K_TILE / 2), h_kv * K_SUBTILES);
    }
    wp_end(wpc, WP_LOAD_ISSUE_K);

    if (k == 0 && elect_one_sync()) {
      #pragma unroll
      for (int m = 0; m < M_TILES_PER_CTA; ++m) {
        wp_begin(wpc, WP_LOAD_WAIT);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_q[m]), q_empty_ph.get_phase());
        wp_end(wpc, WP_LOAD_WAIT);

        wp_begin(wpc, WP_LOAD_ISSUE_Q);
        if (peer == 0) mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar_q[m]), 2 * Q_TILE_BYTES);
        const uint32_t qbar = tma_peer_bit_mask(smem_ptr_u32(&full_bar_q[m]));
        const int tok0 = q_tb_eff + m * q_tile_per_mtile;
        #pragma unroll
        for (int s = 0; s < Q_SUBTILES; ++s) {
          tma_load_4d_2sm(smem_ptr_u32(sQ[m] + s * Q_SUB_COLS_BYTES), tmap_q, qbar,
                          s * SUB_COLS_BF16, h_kv * gqa_group_size, tok0, sample);
        }
        wp_end(wpc, WP_LOAD_ISSUE_Q);
      }
      q_empty_ph.advance();
    }
    kv_stage = kv_empty_ph.get_stage();

    wp_begin(wpc, WP_LOAD_WAIT);
    mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar[kv_stage]), kv_empty_ph.get_phase());
    kv_empty_ph.advance();
    wp_end(wpc, WP_LOAD_WAIT);

    wp_begin(wpc, WP_LOAD_ISSUE_V);
    if (elect_one_sync()) {
      if (peer == 0) mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar[kv_stage]), V_TILE_BYTES);
      const uint32_t vbar = tma_peer_bit_mask(smem_ptr_u32(&full_bar[kv_stage]));
      #pragma unroll
      for (int s = 0; s < V_SUBTILES; ++s) {
        tma_load_2d_2sm(smem_ptr_u32(sKV + kv_stage * K_TILE_BYTES + s * (Q_SUB_COLS_BYTES / 2)), tmap_v_t, vbar,
                        k_start + k_offset + s * SUB_COLS_BF16, h_kv * HEAD_DIM + peer * (HEAD_DIM / 2));
      }
    }
    wp_end(wpc, WP_LOAD_ISSUE_V);
  }
  __syncwarp();  // converge after the elect-issued TMA loads
}

/* ============================================================================
 * load_warp_blackwell_ntiles_1sm_varlen_bf16_fmha<>(wpc, ...)
 *
 * LOAD warp for the varlen 1SM FMHA context kernel (fmha_context_bf16_varlen.cu).
 * Separate from the uniform load block because varlen packs samples densely: Q is a
 * 3D FLAT-GLOBAL map (tma_load_3d, tok0 = cu_seqlens_q[sample] + q_tile_base + ...),
 * not the uniform 4D per-sample box, and the K/V start comes from a k_base[] prefix
 * array. Short-sample `if (q_tile_base < seqlen_q)` guard; warp-0-owned TMEM so no tail
 * drain. Single self-contained ntiles body (not shared, so no 1tile/ntiles split).
 * ============================================================================ */
template <int NUM_KV_STAGES, int M_TILES_PER_CTA, int M_TILE, int K_TILE, int HEAD_DIM,
          bool USE_CLC, int CLC_STAGES, bool Q_RASTER, bool IS_CAUSAL, bool LPT,
          int LOAD_REG_BUDGET = 56>
__device__ inline
void load_warp_blackwell_ntiles_1sm_varlen_bf16_fmha(WpCtx& wpc,
    const CUtensorMap* tmap_q, const CUtensorMap* tmap_k, const CUtensorMap* tmap_v_t,
    uint8_t* sQ0, uint8_t* sKV,
    uint64_t* full_bar, uint64_t* empty_bar, uint64_t* full_bar_q, uint64_t* empty_bar_q,
    uint64_t* clc_full, uint64_t* clc_empty, uint32_t* clc_response,
    uint64_t* throttle_full, uint64_t* throttle_empty,
    int num_q_heads, int num_kv_heads, int packed_mtiles_per_seq, int num_samples,
    const int* cu_seqlens_q, const int* k_base, const int* seqlens_kv) {
  setmaxnreg_dec<LOAD_REG_BUDGET>();
  constexpr int SUB_COLS_BF16    = 64;
  constexpr int SUB_COLS_BYTES   = SUB_COLS_BF16 * (int)sizeof(__nv_bfloat16);
  constexpr int Q_SUBTILES       = HEAD_DIM / SUB_COLS_BF16;
  constexpr int K_SUBTILES       = HEAD_DIM / SUB_COLS_BF16;
  constexpr int V_SUBTILES       = K_TILE / SUB_COLS_BF16;
  constexpr int Q_SUB_COLS_BYTES = M_TILE * SUB_COLS_BYTES;
  constexpr int Q_TILE_BYTES     = Q_SUBTILES * Q_SUB_COLS_BYTES;
  constexpr int K_TILE_BYTES     = K_SUBTILES * (K_TILE * SUB_COLS_BYTES);
  constexpr int V_TILE_BYTES     = V_SUBTILES * Q_SUB_COLS_BYTES;

  const int gqa_group_size   = num_q_heads / num_kv_heads;
  const int q_tile_per_mtile = M_TILE / gqa_group_size;
  const int q_tile_per_cta   = M_TILES_PER_CTA * q_tile_per_mtile;
  const int packed_mtiles_per_sample = packed_mtiles_per_seq * num_kv_heads;
  const int total_tiles = num_samples * packed_mtiles_per_sample;

  EmptyPhaseTracker<NUM_KV_STAGES> kv_empty_ph;
  EmptyPhaseTracker<1> q_empty_ph;
  [[maybe_unused]] int clc_stage = 0;
  [[maybe_unused]] uint32_t clc_phase = 0;
  [[maybe_unused]] int thr_prod_stage = 0;
  [[maybe_unused]] uint32_t thr_prod_phase = 1;
  int tile_id = (int)blockIdx.x;
  while (true) {
    int sample, h_kv, q_tile_base, seqlen_q, K_TILES;
    decode_workitem_varlen<K_TILE, Q_RASTER, IS_CAUSAL, LPT>(tile_id, num_kv_heads,
        packed_mtiles_per_seq, packed_mtiles_per_sample, q_tile_per_cta,
        cu_seqlens_q, seqlens_kv, sample, h_kv, q_tile_base, seqlen_q, K_TILES);
    const int q_start = cu_seqlens_q[sample];

    // Throttle producer paces the sched warp (block 97 consumer): one round per work-item.
    if constexpr (USE_CLC) {
      wp_begin(wpc, WP_LOAD_WAIT_THROTTLE);
      mbarrier_wait_parity_suspend(smem_ptr_u32(&throttle_empty[thr_prod_stage]), thr_prod_phase);
      wp_end(wpc, WP_LOAD_WAIT_THROTTLE);
      mbarrier_arrive_nostate(smem_ptr_u32(&throttle_full[thr_prod_stage]));
      advance_stage_phase<2>(thr_prod_stage, thr_prod_phase);
    }

    if (q_tile_base < seqlen_q) {
      const int k_start = k_base[sample];
      // Load order: K[K_TILES-1], Q0, Q1, V[K_TILES-1], K[K_TILES-2], V[..], ..., K[0], V[0].
      for (int k = 0; k < K_TILES; ++k) {
        const int k_offset = (K_TILES - 1 - k) * K_TILE;
        int kv_stage = kv_empty_ph.get_stage();
        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar[kv_stage]), kv_empty_ph.get_phase());
        kv_empty_ph.advance();
        if (elect_one_sync()) {
          mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar[kv_stage]), K_TILE_BYTES);
          tma_load_3d(smem_ptr_u32(sKV + kv_stage * K_TILE_BYTES), tmap_k, smem_ptr_u32(&full_bar[kv_stage]),
                      0, k_start + k_offset, h_kv * K_SUBTILES);
        }
        if (k == 0 && elect_one_sync()) {
          #pragma unroll
          for (int m = 0; m < M_TILES_PER_CTA; ++m) {
            mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_q[m]), q_empty_ph.get_phase());
            mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar_q[m]), Q_TILE_BYTES);
            const int tok0 = q_start + q_tile_base + m * q_tile_per_mtile;
            #pragma unroll
            for (int s = 0; s < Q_SUBTILES; ++s) {
              tma_load_3d(smem_ptr_u32(sQ0 + m * Q_TILE_BYTES + s * Q_SUB_COLS_BYTES), tmap_q,
                          smem_ptr_u32(&full_bar_q[m]), s * SUB_COLS_BF16, h_kv * gqa_group_size, tok0);
            }
          }
          q_empty_ph.advance();
        }
        kv_stage = kv_empty_ph.get_stage();
        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar[kv_stage]), kv_empty_ph.get_phase());
        kv_empty_ph.advance();
        if (elect_one_sync()) {
          mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar[kv_stage]), V_TILE_BYTES);
          #pragma unroll
          for (int s = 0; s < V_SUBTILES; ++s) {
            tma_load_2d(smem_ptr_u32(sKV + kv_stage * K_TILE_BYTES + s * Q_SUB_COLS_BYTES), tmap_v_t,
                        smem_ptr_u32(&full_bar[kv_stage]), k_start + k_offset + s * SUB_COLS_BF16, h_kv * HEAD_DIM);
          }
        }
      }
    }
    if constexpr (USE_CLC) {
      ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, /*CTA_GROUP=*/1>(
          clc_full, clc_empty, clc_response, clc_stage, clc_phase, /*do_release=*/elect_one_sync());
      clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
      if (!next.valid) break;
      tile_id = next.n_tile;
    } else {
      tile_id += (int)gridDim.x;
      if (tile_id >= total_tiles) break;
    }
  }
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
