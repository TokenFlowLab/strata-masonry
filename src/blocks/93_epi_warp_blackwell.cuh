#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// 93_epi_warp_blackwell.cuh -- Blackwell epilogue building block.
// TMEM accumulator -> registers -> cvt(F32 -> FP16) -> stmatrix -> SMEM.
//
// ARCH: sm_100a
//
// The "epilogue warp" role: drains a TMEM-resident FP32 accumulator tile
// through the cvt + stmatrix path into a 64-entry SMEM staging buffer,
// laid out for a downstream TMA store. Wraps the alloc / wait / dealloc
// envelope and uses #125 epi_subtile_blackwell_fp16 for the cvt + stmatrix
// kernel-side body.
//
// Parametric form: `epi_warp_blackwell(out)`. The kernel writes the SMEM
// staging buffer's 64 FP16 words into `out` (uint16_t[64]) so a caller
// can verify the cvt + stmatrix path produced non-zero data.
//
// Issuer: 1 CTA, 32 threads (one warp; epi_subtile_blackwell_fp16 is a
// warp-collective stmatrix helper).

// PTX:    9.7.18.8 (tcgen05.ld), 9.7.10.24 (cvt), 9.7.16.5.16 (stmatrix), 9.7.10.28.5.3 (TMA store)
//
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "../primitives/0_tcgen05_alloc.cuh"
#include "../primitives/1_tcgen05_dealloc.cuh"
#include "../primitives/2_tcgen05_relinquish.cuh"
#include "../primitives/9_tcgen05_ld.cuh"
#include "../primitives/10_tcgen05_st.cuh"
#include "../primitives/12_tcgen05_wait.cuh"
#include "../primitives/22_tma_store.cuh"
#include "../primitives/25_tma_async_group.cuh"
#include "../primitives/34_fence_proxy_async.cuh"
#include "../primitives/63_cvt_f32_to_f16_bf16.cuh"
#include "../composites/125_epi_subtile_blackwell.cuh"
#include "../primitives/70_smem_ptr.cuh"
#include "../primitives/_warp_prof_noop.cuh"
#include "../composites/82_tile_rasterize.cuh"
#include "../composites/130_swiglu_act.cuh"
#include "../composites/104_acc_pipeline_2bank_blackwell.cuh"
#include "../primitives/30_mbarrier_arrive.cuh"
#include "../primitives/33_mbarrier_try_wait.cuh"
#include "../primitives/44_elect_sync.cuh"
#include "../primitives/46_setmaxnreg.cuh"
#include "../composites/118_mbarrier_phase_tracking.cuh"
#include "../composites/106_clc_fetch_next_tile.cuh"
#include "../composites/110_fmha_workitem_decode.cuh"
#include "../composites/106_clc_fetch_next_tile.cuh"

/* ============================================================================
 * epi_warp_blackwell_block(wpc, slot, smem_stage, out)
 *
 * Header-only __device__ form of the legacy alloc / tcgen05.st seed /
 * cvt+stmatrix / dealloc smoke envelope. Caller owns SMEM (`slot` and
 * `smem_stage[64]`).
 *
 * Caller responsibilities:
 *   - launch with 32 threads/CTA, single CTA.
 *   - allocate `__shared__ uint32_t slot` (16-byte aligned) and
 *     `__shared__ uint16_t smem_stage[64]` (1024-byte aligned).
 *   - initialize slot to 0.
 *   - issue __syncthreads BEFORE this block.
 *
 * The block runs a single-warp epilogue smoke:
 *   tcgen05_alloc<1>(slot, 32) + relinquish
 *   tcgen05_st_32x32b_x8(tbase, ...) + tcgen05_wait_st
 *   epi_subtile_blackwell_fp16(tbase, smem_stage)
 *   drain smem_stage -> out (per-lane copy)
 *   tcgen05_dealloc<1>(tbase, 32)
 * ============================================================================ */
__device__ __forceinline__
void epi_warp_blackwell_block(WpCtx& wpc,uint32_t* slot,
                              uint16_t* smem_stage,
                              uint16_t* out) {
  if (threadIdx.x < 32) {
    tcgen05_alloc<1>(smem_ptr_u32(slot), 32);
    tcgen05_relinquish_alloc_permit<1>();
  }
  __syncthreads();
  uint32_t tbase = *slot;

  // Seed TMEM with a known FP32 pattern (one register vector per lane).
  if (threadIdx.x < 32) {
    uint32_t in[8];
    float* f = reinterpret_cast<float*>(in);
    #pragma unroll
    for (int i = 0; i < 8; ++i) f[i] = (float)(threadIdx.x + i + 1);
    tcgen05_st_32x32b_x8(tbase, in);
    tcgen05_wait_st();
    epi_subtile_blackwell_fp16(tbase, smem_ptr_u32(smem_stage));
  }
  __syncthreads();

  // Drain the SMEM staging buffer to the caller's output.
  for (int i = threadIdx.x; i < 64; i += blockDim.x) {
    if (out != nullptr) out[i] = smem_stage[i];
  }
  __syncthreads();
  if (threadIdx.x == 0) tcgen05_dealloc<1>(tbase, 32);
}

/* ============================================================================
 * epi_warp_blackwell_1tile_1sm2sm_bf16<M_TILE_PER_CTA, N_TILE_CLUSTER,
 *                                   EPI_SUB_COLS, EPI_NUM_BUFS>(wpc, ...)
 *
 * Production __device__ body for 2SM Blackwell BF16 GEMM kernels.
 * Drains ONE tile of TMEM accumulator (FP32) -> reg -> cvt(F32->BF16x2)
 * -> SMEM staged sub-tile -> TMA store to GMEM. Composes composite 104
 * for the acc-pipeline consumer handshake (wait acc_full[bank] before;
 * arrive acc_empty[bank] from leader epi thread per CTA after).
 *
 * Naming: <role>_<arch>_<#tiles>_<cluster>_<dtype>.
 *   role    = epi_warp           (TMEM consumer, GMEM producer via TMA)
 *   arch    = blackwell
 *   #tiles  = 1tile               (one tile's epilogue; caller drives outer)
 *   cluster = 2sm                 (cta_group::2; cluster_dims(2,1,1))
 *   dtype   = bf16                (FP32 acc -> BF16 GMEM output)
 *
 * Threading model:
 *   Caller invokes from ALL epi-warp threads in BOTH peer CTAs (i.e.,
 *   warps 4-7, 32 lanes each, 2 CTAs = 256 threads/cluster). The body
 *   internally gates:
 *     - Per-thread: tcgen05.ld + cvt + STS to row [epi_warp*32 + lane].
 *     - Warp-collective bar.sync 1, 128 between sub-tiles.
 *     - Warp 4 lane 0 of each peer: TMA store + cp.async.bulk.commit_group
 *       and the composite-104 consumer_release on acc_empty[bank].
 *
 * Args:
 *   tma_d
 *     CUDA tensormap for the GMEM output operand D. Body issues
 *     `cp.async.bulk.tensor.2d.global.shared::cta` (TMA store) per
 *     sub-tile from warp 4 lane 0.
 *
 *   smem_d
 *     Byte pointer to the dual-buffer epi SMEM region (size =
 *     EPI_NUM_BUFS * M_TILE_PER_CTA * EPI_SUB_COLS * sizeof(bf16)).
 *     Buffers ping-pong via `buf = sub & (EPI_NUM_BUFS - 1)`.
 *
 *   acc_bars, cons_state
 *     Composite 104 acc-pipeline. Body issues:
 *       - acc_pipeline_2bank_consumer_wait(acc_bars, cons_state) BEFORE
 *         the sub-tile loop (waits acc_full[bank]).
 *       - acc_pipeline_2bank_consumer_release(acc_bars, cons_state) from
 *         warp 4 lane 0 only AFTER the sub-tile loop drains. Routes
 *         the arrive to CTA 0's acc_empty[bank] via peer-bit-mask
 *         (count=2 -> one arrive from each CTA's epi leader).
 *       - acc_pipeline_2bank_state_advance(cons_state) at end (all
 *         threads).
 *
 *   tmem_base
 *     TMEM allocation base from tcgen05.alloc.cta_group::2. Body
 *     computes per-thread `my_tmem = tmem_base + (epi_warp*32) << 16
 *     + acc_stage * 256 + sub * EPI_SUB_COLS` (row-shift in upper 16
 *     bits, col offset in lower 16 bits per the TMEM addressing
 *     convention).
 *
 *   peer, warp, lane
 *     Cluster rank, intra-CTA warp index [0,7], intra-warp lane [0,31].
 *     Body uses (warp - 4) for the row index, lane for the column.
 *
 *   m_offset, n_offset_d
 *     GMEM output coords for the current tile.
 *
 * Notes:  TMEM bank stride = N_TILE_CLUSTER, <= 256.
 * PTX:    9.7.18.8.x   (tcgen05.ld + tcgen05.wait::ld),
 *         9.7.10.24     (cvt.rn.bf16x2.f32),
 *         9.7.10.28.5.3 (cp.async.bulk.tensor.2d -- TMA store),
 *         9.7.15.16.16 (mbarrier.arrive on acc_empty via composite 104).
 * ============================================================================ */
// DRAIN_PER_TILE selects how SMEM-buf reuse is fenced across tile
// boundaries. On entry to the next tile there are EPI_NUM_BUFS TMA
// stores still in flight from this tile's last EPI_NUM_BUFS subs:
//
//   true (default): drain those in-flight stores at every tile boundary
//     via wait_group<0> at end of _1tile_. In-sub-loop wait_group is
//     guarded (`sub >= EPI_NUM_BUFS`), so it fires only on within-tile
//     reuse. Cost = serializing tail TMA stores against next tile's
//     sub=0 tcgen05.ld + cvt+STS.
//
//   false: skip the per-tile drain and let those EPI_NUM_BUFS stores
//     stay in flight into the next tile. In-sub-loop wait_group runs
//     every iter (no guard); the per-thread commit count carries across
//     tiles, so sub=0/1 of non-first tiles correctly throttle SMEM-buf
//     reuse. Final drain amortizes once at end of _ntiles_. Wins when
//     the tail TMA stores' GMEM-commit latency is longer than next
//     tile's TMEM->regs + cvt+STS path -- the overlap hides it.
//
// Default `true`: 2x the wait_group calls (EPI_SUB_COUNT=4 / EPI_NUM_BUFS=2) cost more than the
// overlap saves on typical K; `false` wins when the per-tile drain dominates (small K, fused epilogues).
template <int M_TILE_PER_CTA, int N_TILE_CLUSTER, int EPI_SUB_COLS, int EPI_NUM_BUFS,
          bool DRAIN_PER_TILE = true>
__device__ inline
void epi_warp_blackwell_1tile_1sm2sm_bf16(WpCtx& wpc,
    const CUtensorMap* tma_d,
    uint8_t* smem_d,
    AccPipeline2BankBars acc_bars,
    AccPipeline2BankState& cons_state,
    uint32_t tmem_base,
    int peer, int warp, int lane,
    int m_offset, int n_offset_d) {
  static_assert(N_TILE_CLUSTER <= 256, "N_TILE_CLUSTER <= 256: TMEM has 512 cols, 2-bank pipeline -> 256 cols per bank");
  constexpr int EPI_SUB_COUNT = N_TILE_CLUSTER / EPI_SUB_COLS;
  constexpr int EPI_BUF_BYTES = M_TILE_PER_CTA * EPI_SUB_COLS *
                                 static_cast<int>(sizeof(__nv_bfloat16));
  (void)peer;

  const int epi_warp = (warp >= 4) ? (warp - 4) : 0;
  const uint32_t my_tmem_row_offset = ((uint32_t)(epi_warp * 32) << 16);
  const int row = epi_warp * 32 + lane;

  __nv_bfloat16* d_smem_buf[EPI_NUM_BUFS];
  #pragma unroll
  for (int b = 0; b < EPI_NUM_BUFS; ++b) {
    d_smem_buf[b] = reinterpret_cast<__nv_bfloat16*>(smem_d + b * EPI_BUF_BYTES);
  }

  wp_begin(wpc, WP_EPI_WAIT_ACC);
  acc_pipeline_2bank_consumer_wait(acc_bars, cons_state);
  wp_end(wpc, WP_EPI_WAIT_ACC);
  const int acc_stage_cons = acc_pipeline_2bank_state_index(cons_state);
  // TMEM bank stride in cols = N_TILE_CLUSTER (one full N-tile per bank).
  // Phase-1 2SM: N_TILE_CLUSTER=256. Phase-2 1SM: N_TILE_CLUSTER=128.
  const uint32_t my_tmem = tmem_base + my_tmem_row_offset
                           + (uint32_t)(acc_stage_cons * N_TILE_CLUSTER);

  #pragma unroll
  for (int sub = 0; sub < EPI_SUB_COUNT; ++sub) {
    const int buf  = sub & (EPI_NUM_BUFS - 1);
    const int col0 = sub * EPI_SUB_COLS;
    uint32_t regs[EPI_SUB_COLS];
    const uint32_t taddr = my_tmem + (uint32_t)col0;
    // tcgen05.ld.32x32b.xN: N fp32 per lane = N cols per row, sized to
    // match EPI_SUB_COLS. PTX 9.7.18.8.3 / primitive 9.
    static_assert(EPI_SUB_COLS == 1   || EPI_SUB_COLS == 2   ||
                  EPI_SUB_COLS == 4   || EPI_SUB_COLS == 8   ||
                  EPI_SUB_COLS == 16  || EPI_SUB_COLS == 32  ||
                  EPI_SUB_COLS == 64  || EPI_SUB_COLS == 128,
                  "EPI_SUB_COLS must be 1/2/4/8/16/32/64/128");
    wp_begin(wpc, WP_EPI_TMEM_LD);
    if      constexpr (EPI_SUB_COLS == 1)   tcgen05_ld_32x32b_x1  (taddr, regs);
    else if constexpr (EPI_SUB_COLS == 2)   tcgen05_ld_32x32b_x2  (taddr, regs);
    else if constexpr (EPI_SUB_COLS == 4)   tcgen05_ld_32x32b_x4  (taddr, regs);
    else if constexpr (EPI_SUB_COLS == 8)   tcgen05_ld_32x32b_x8  (taddr, regs);
    else if constexpr (EPI_SUB_COLS == 16)  tcgen05_ld_32x32b_x16 (taddr, regs);
    else if constexpr (EPI_SUB_COLS == 32)  tcgen05_ld_32x32b_x32 (taddr, regs);
    else if constexpr (EPI_SUB_COLS == 64)  tcgen05_ld_32x32b_x64 (taddr, regs);
    else if constexpr (EPI_SUB_COLS == 128) tcgen05_ld_32x32b_x128(taddr, regs);
    tcgen05_wait_ld();
    wp_end(wpc, WP_EPI_TMEM_LD);

    // Early acc_empty release: TMEM[acc_stage] is fully read after the
    // LAST sub's tcgen05.wait::ld. The cvt+STS+TMA chain that follows
    // reads regs / SMEM, never TMEM, so the producer (MMA) can refill
    // TMEM in parallel. arrive_count=256 is consumed across all 4 epi
    // warps x 32 lanes x 2 CTAs (one arrive per thread).
    if (sub == EPI_SUB_COUNT - 1) {
      acc_pipeline_2bank_consumer_release(acc_bars, cons_state);
      acc_pipeline_2bank_state_advance(cons_state);
    }

    // tma_store_2d and cp_async_bulk_commit_group are per-thread, and
    // only warp 4 lane 0 issues them -- so it's the sole holder of TMA
    // commits. cp_async_bulk_wait_group is also per-thread; non-issuer
    // threads have 0 commits and their wait_group passes instantly.
    // bar.sync 1, 128 makes all 128 EPI threads wait for the issuer's
    // drain so the next sub's STS does not race the in-flight TMA's
    // SMEM read. Gating wait_group on the issuer is a micro-opt; the
    // ungated form
    //   cp_async_bulk_wait_group<EPI_NUM_BUFS - 1>();
    //   asm volatile("bar.sync 1, 128;\n" ::: "memory");
    // is also correct.
    wp_begin(wpc, WP_EPI_WAIT_STORE);
    if constexpr (DRAIN_PER_TILE) {
      if (sub >= EPI_NUM_BUFS) {
        if (warp == 4 && lane == 0) {
          cp_async_bulk_wait_group<EPI_NUM_BUFS - 1>();
        }
        asm volatile("bar.sync 1, 128;\n" ::: "memory");
      }
    } else {
      if (warp == 4 && lane == 0) {
        cp_async_bulk_wait_group<EPI_NUM_BUFS - 1>();
      }
      asm volatile("bar.sync 1, 128;\n" ::: "memory");
    }
    wp_end(wpc, WP_EPI_WAIT_STORE);

    wp_begin(wpc, WP_EPI_STORE);
    uint32_t* d_row_u32 = reinterpret_cast<uint32_t*>(
        d_smem_buf[buf] + row * EPI_SUB_COLS);
    #pragma unroll
    for (int j = 0; j < EPI_SUB_COLS; j += 2) {
      const float a = __int_as_float(regs[j]);
      const float b = __int_as_float(regs[j + 1]);
      d_row_u32[j >> 1] = cvt_pack_f32_to_bf16x2(a, b);
    }
    asm volatile("bar.sync 1, 128;\n" ::: "memory");

    if (warp == 4 && lane == 0) {
      fence_proxy_async_shared_cta();
      const uint32_t d_smem_addr = smem_ptr_u32(d_smem_buf[buf]);
      tma_store_2d(tma_d, /*x=*/n_offset_d + col0, /*y=*/m_offset, d_smem_addr);
      cp_async_bulk_commit_group();
    }
    wp_end(wpc, WP_EPI_STORE);
    // No post-TMA-issue bar.sync: next sub's STS races safely; same-buf
    // reuse is guarded by the wait_group+bar.sync above; end-of-tile
    // drain catches warp 4 lane 0.
    //asm volatile("bar.sync 1, 128;\n" ::: "memory");
  }
  wp_begin(wpc, WP_EPI_WAIT_STORE);
  if constexpr (DRAIN_PER_TILE) {
    if (warp == 4 && lane == 0) {
      cp_async_bulk_wait_group<0>();
    }
    asm volatile("bar.sync 1, 128;\n" ::: "memory");
  }
  wp_end(wpc, WP_EPI_WAIT_STORE);
}

/* ============================================================================
 * epi_warp_blackwell_ntiles_2sm_bf16<...>(wpc, ...)
 *
 * Self-driving do-while wrapper around the _1tile_ body for per-warp
 * persistent loops. Each iteration:
 *   1. Compute m_offset / n_offset_d for the CURRENT tile coords.
 *   2. Call _1tile_ body (acc consumer wait + sub-tile loop + TMA store
 *      + acc consumer release).
 *   3. fetch_next_tile (cluster-wide CLC handshake).
 *   4. Update m_tile, n_tile from next; break if !next.valid.
 *
 * Caller invokes from ALL 32 lanes of warps 4-7 in BOTH peer CTAs
 * (= 256 threads/cluster). Each thread participates in fetch_next_tile.
 *
 * `m_tile_remap` (optional, default nullptr -> identity): grouped-GEMM
 * 2-phase mode shrinks the CLC grid to phase-1-only m-clusters (skipping
 * each expert's partial tail, handled by a separate 1SM phase-2 kernel),
 * while A/D keep the full padded layout. The two idx spaces diverge once
 * any expert has a tail. Remap maps the CLC's phase-1 idx back to the
 * full-layout idx so `m_offset = remap(idx) * MTC + peer * MTC/2` lands
 * on the right rows of D. Caller passes a device pointer to an int[
 * m_clusters_p1] array (host-built); nullptr for single-phase / non-
 * grouped use.
 *
 * Example, E=2, MTC=256, m_e={1330, 832}:
 *   Full layout: e0 -> idx 0..5 (5 full + 1 tail), e1 -> idx 6..9 (3+1).
 *   Phase-1:     e0 -> p1 0..4,                    e1 -> p1 5..7.
 *   m_tile_remap[0..7] = {0,1,2,3,4, 6,7,8}    (skip idx 5: e0's tail).
 * ============================================================================ */
// Raster-templated overload: caller picks (CLUSTER_SHAPE_M, CLUSTER_SHAPE_N,
// ORDER) for the CLC decode. Defaults match the un-templated form
// (1, 2, AlongN) so existing call sites are unchanged.
// DRAIN_PER_TILE: see _1tile_ doc comment. Default `true` is faster on
// typical shapes; `false` overlaps next tile's sub=0 with prior tile's tail
// TMA stores (useful when the per-tile drain dominates).
template <int M_TILE_PER_CTA, int N_TILE_PER_CTA, int K_TILE,
          int M_TILE_CLUSTER, int N_TILE_CLUSTER,
          int EPI_SUB_COLS, int EPI_NUM_BUFS,
          int CLUSTER_SHAPE_M = 1, int CLUSTER_SHAPE_N = 2,
          ClcRasterOrder ORDER = ClcRasterOrder::AlongN,
          bool DRAIN_PER_TILE = true>
__device__ inline
void epi_warp_blackwell_ntiles_2sm_bf16(WpCtx& wpc,
    const CUtensorMap* tma_d,
    uint8_t* smem_d,
    AccPipeline2BankBars acc_bars,
    uint64_t* clc_full_bar, uint64_t* clc_empty_bar, uint32_t* clc_response,
    uint32_t tmem_base,
    int peer, int warp, int lane,
    const int* __restrict__ m_tile_remap = nullptr,
    uint32_t* tmem_slot = nullptr) {  // if set, this block waits for + reads tmem_base
  static_assert(N_TILE_CLUSTER <= 256, "N_TILE_CLUSTER <= 256: TMEM has 512 cols, 2-bank pipeline -> 256 cols per bank");
  AccPipeline2BankState cons_state = acc_pipeline_2bank_state_init();

  // EPI waits for MMA's tcgen05.alloc to publish tmem_base (named bar 6) when
  // tmem_slot is given. Symmetric with mma owning the alloc. Legacy callers do
  // the bar.sync themselves and pass tmem_base + tmem_slot=nullptr.
  if (tmem_slot != nullptr) {
    wp_begin(wpc, WP_EPI_WAIT_TMEM);
    asm volatile("bar.sync 6, 160;\n" ::: "memory");
    tmem_base = *tmem_slot;
    wp_end(wpc, WP_EPI_WAIT_TMEM);
  }

  auto remap = [&](int p1) -> int {
    return (m_tile_remap != nullptr) ? m_tile_remap[p1] : p1;
  };

  int      clc_cons_stage = 0;
  uint32_t clc_cons_phase = 0;
  int m_tile, n_tile;
  if constexpr (ORDER == ClcRasterOrder::AlongN) {
    m_tile = remap((int)blockIdx.y);
    n_tile = (int)(blockIdx.x >> 1);
  } else {  // AlongM: cluster.x -> M
    m_tile = remap((int)(blockIdx.x >> 1));
    n_tile = (int)blockIdx.y;
  }

  while (true) {
    const int m_offset   = m_tile * M_TILE_CLUSTER + peer * M_TILE_PER_CTA;
    const int n_offset_d = n_tile * N_TILE_CLUSTER;

    epi_warp_blackwell_1tile_1sm2sm_bf16<
        M_TILE_PER_CTA, N_TILE_CLUSTER, EPI_SUB_COLS, EPI_NUM_BUFS,
        DRAIN_PER_TILE>(wpc, tma_d, smem_d, acc_bars, cons_state, tmem_base,
        peer, warp, lane, m_offset, n_offset_d);

    // Only ONE thread of the 4 epi warps arrives clc_empty (warp 4
    // lane 0); other epi threads pass do_release=false. The init
    // arrive_count for clc_empty is set accordingly at the kernel level.
    const bool epi_release = (warp == 4 && lane == 0);
    wp_begin(wpc, WP_CLC_FETCH);
    ClcTileInfo next = clc_fetch_next_tile<
        CLUSTER_SHAPE_M, CLUSTER_SHAPE_N, ORDER>(
        clc_full_bar, clc_empty_bar, clc_response,
        clc_cons_stage, clc_cons_phase, /*do_release=*/epi_release);
    wp_end(wpc, WP_CLC_FETCH);
    clc_fetch_next_tile_advance(clc_cons_stage, clc_cons_phase);
    if (!next.valid) break;
    m_tile = remap(next.m_tile);
    n_tile = next.n_tile;
  }
  if constexpr (!DRAIN_PER_TILE) {
    // Final drain: per-tile drain was skipped, so flush the last
    // tile's TMA stores once at the end of the persistent loop.
    wp_begin(wpc, WP_EPI_WAIT_STORE);
    if (warp == 4 && lane == 0) {
      cp_async_bulk_wait_group<0>();
    }
    asm volatile("bar.sync 1, 128;\n" ::: "memory");
    wp_end(wpc, WP_EPI_WAIT_STORE);
  }
}

/* ============================================================================
 * epi_warp_blackwell_1tile_1sm2sm_bf16_swiglu<M_TILE_PER_CTA, N_TILE_CLUSTER,
 *                                             EPI_SUB_COLS, EPI_NUM_BUFS>(wpc, ...)
 *
 * SwiGLU-fused 2SM Blackwell BF16 epilogue. Differs from
 * `_1tile_1sm2sm_bf16` only in the per-row pack inner loop and the
 * SMEM / TMA-store col stride.
 *
 * B is laid out as a column-by-column interleave
 * with even col = up, odd col = gate. The MMA accumulator therefore
 * arrives in TMEM with the same parity: TMEM col 2n = up_n,
 * 2n+1 = gate_n. Per (m, n_out) output position, fuse:
 *   D[m, n_out] = bf16( silu(gate_n) * up_n )    where silu(z) = z * sigmoid(z)
 *
 * Each iteration of the per-row pack loop consumes 4 fp32 TMEM cols
 * (up0, gate0, up1, gate1 -> 2 SwiGLU outputs -> one bf16x2). Requires
 * EPI_SUB_COLS divisible by 4.
 *
 * SMEM EPI buf width is EPI_SUB_COLS / 2 bf16 per row (vs EPI_SUB_COLS
 * in the non-SwiGLU body). TMA store coord uses post-SwiGLU N:
 *   x = n_offset_d_out + sub * (EPI_SUB_COLS / 2)
 *
 * Caller is responsible for:
 *   - Allocating smem_d with size EPI_NUM_BUFS * M_TILE_PER_CTA *
 *     (EPI_SUB_COLS / 2) * sizeof(bf16).
 *   - Building D's TMA tensormap with width = post-SwiGLU N (= half
 *     of pre-SwiGLU N).
 *
 * PTX:    9.7.18.8.x   (tcgen05.ld + tcgen05.wait::ld),
 *         9.7.10.24     (cvt.rn.bf16x2.f32),
 *         9.7.10.28.5.3 (cp.async.bulk.tensor.2d -- TMA store).
 * ============================================================================ */
template <int M_TILE_PER_CTA, int N_TILE_CLUSTER, int EPI_SUB_COLS, int EPI_NUM_BUFS,
          bool DRAIN_PER_TILE = true>
__device__ inline
void epi_warp_blackwell_1tile_1sm2sm_bf16_swiglu(WpCtx& wpc,
    const CUtensorMap* tma_d,
    uint8_t* smem_d,
    AccPipeline2BankBars acc_bars,
    AccPipeline2BankState& cons_state,
    uint32_t tmem_base,
    int peer, int warp, int lane,
    int m_offset, int n_offset_d_out) {
  static_assert(N_TILE_CLUSTER <= 256, "N_TILE_CLUSTER <= 256: TMEM has 512 cols, 2-bank pipeline -> 256 cols per bank");
  static_assert(EPI_SUB_COLS % 4 == 0, "SwiGLU EPI requires EPI_SUB_COLS divisible by 4 (each iter consumes 4 input cols)");
  constexpr int EPI_SUB_COUNT     = N_TILE_CLUSTER / EPI_SUB_COLS;
  constexpr int EPI_SUB_COLS_OUT  = EPI_SUB_COLS / 2;
  constexpr int EPI_BUF_BYTES_OUT = M_TILE_PER_CTA * EPI_SUB_COLS_OUT *
                                     static_cast<int>(sizeof(__nv_bfloat16));
  (void)peer;

  const int epi_warp = (warp >= 4) ? (warp - 4) : 0;
  const uint32_t my_tmem_row_offset = ((uint32_t)(epi_warp * 32) << 16);
  const int row = epi_warp * 32 + lane;

  __nv_bfloat16* d_smem_buf[EPI_NUM_BUFS];
  #pragma unroll
  for (int b = 0; b < EPI_NUM_BUFS; ++b) {
    d_smem_buf[b] = reinterpret_cast<__nv_bfloat16*>(smem_d + b * EPI_BUF_BYTES_OUT);
  }

  wp_begin(wpc, WP_EPI_WAIT_ACC);
  acc_pipeline_2bank_consumer_wait(acc_bars, cons_state);
  wp_end(wpc, WP_EPI_WAIT_ACC);
  const int acc_stage_cons = acc_pipeline_2bank_state_index(cons_state);
  const uint32_t my_tmem = tmem_base + my_tmem_row_offset
                           + (uint32_t)(acc_stage_cons * N_TILE_CLUSTER);

  #pragma unroll
  for (int sub = 0; sub < EPI_SUB_COUNT; ++sub) {
    const int buf      = sub & (EPI_NUM_BUFS - 1);
    const int col0     = sub * EPI_SUB_COLS;
    const int col0_out = sub * EPI_SUB_COLS_OUT;
    uint32_t regs[EPI_SUB_COLS];
    const uint32_t taddr = my_tmem + (uint32_t)col0;
    static_assert(EPI_SUB_COLS == 4   || EPI_SUB_COLS == 8   ||
                  EPI_SUB_COLS == 16  || EPI_SUB_COLS == 32  ||
                  EPI_SUB_COLS == 64  || EPI_SUB_COLS == 128,
                  "EPI_SUB_COLS must be 4/8/16/32/64/128 for SwiGLU EPI");
    wp_begin(wpc, WP_EPI_TMEM_LD);
    if      constexpr (EPI_SUB_COLS == 4)   tcgen05_ld_32x32b_x4  (taddr, regs);
    else if constexpr (EPI_SUB_COLS == 8)   tcgen05_ld_32x32b_x8  (taddr, regs);
    else if constexpr (EPI_SUB_COLS == 16)  tcgen05_ld_32x32b_x16 (taddr, regs);
    else if constexpr (EPI_SUB_COLS == 32)  tcgen05_ld_32x32b_x32 (taddr, regs);
    else if constexpr (EPI_SUB_COLS == 64)  tcgen05_ld_32x32b_x64 (taddr, regs);
    else if constexpr (EPI_SUB_COLS == 128) tcgen05_ld_32x32b_x128(taddr, regs);
    tcgen05_wait_ld();
    wp_end(wpc, WP_EPI_TMEM_LD);

    // Early acc_empty release on last sub (TMEM is fully read; the
    // SwiGLU compute and STS only touch regs/SMEM).
    if (sub == EPI_SUB_COUNT - 1) {
      acc_pipeline_2bank_consumer_release(acc_bars, cons_state);
      acc_pipeline_2bank_state_advance(cons_state);
    }

    // See `_1tile_1sm2sm_bf16` for the rationale: bar.sync 1, 128 forces
    // all 128 EPI threads to wait for warp 4 lane 0's wait_group so the
    // next sub's STS does not race with the in-flight TMA's SMEM read.
    wp_begin(wpc, WP_EPI_WAIT_STORE);
    if constexpr (DRAIN_PER_TILE) {
      if (sub >= EPI_NUM_BUFS) {
        if (warp == 4 && lane == 0) {
          cp_async_bulk_wait_group<EPI_NUM_BUFS - 1>();
        }
        asm volatile("bar.sync 1, 128;\n" ::: "memory");
      }
    } else {
      if (warp == 4 && lane == 0) {
        cp_async_bulk_wait_group<EPI_NUM_BUFS - 1>();
      }
      asm volatile("bar.sync 1, 128;\n" ::: "memory");
    }
    wp_end(wpc, WP_EPI_WAIT_STORE);

    // SwiGLU fusion: even col = up, odd col = gate. Each iter consumes
    // 4 input fp32 (up0, gate0, up1, gate1) and produces 1 bf16x2
    // (silu(gate0)*up0, silu(gate1)*up1). The SMEM row stride is
    // EPI_SUB_COLS_OUT = EPI_SUB_COLS/2 bf16 cols.
    wp_begin(wpc, WP_EPI_SWIGLU);
    uint32_t* d_row_u32 = reinterpret_cast<uint32_t*>(
        d_smem_buf[buf] + row * EPI_SUB_COLS_OUT);
    #pragma unroll
    for (int j = 0; j < EPI_SUB_COLS; j += 4) {
      const float up0   = __int_as_float(regs[j]);
      const float gate0 = __int_as_float(regs[j + 1]);
      const float up1   = __int_as_float(regs[j + 2]);
      const float gate1 = __int_as_float(regs[j + 3]);
      // SwiGLU = silu(gate) * up. Tanh-based silu (composite 130)
      // avoids the __expf overflow-to-Inf intermediate that bites
      // on large accumulator magnitudes.
      const float out0 = swiglu_act_f32(gate0, up0);
      const float out1 = swiglu_act_f32(gate1, up1);
      d_row_u32[j >> 2] = cvt_pack_f32_to_bf16x2(out0, out1);
    }
    wp_end(wpc, WP_EPI_SWIGLU);
    wp_begin(wpc, WP_EPI_STORE);
    asm volatile("bar.sync 1, 128;\n" ::: "memory");
    if (warp == 4 && lane == 0) {
      fence_proxy_async_shared_cta();
      const uint32_t d_smem_addr = smem_ptr_u32(d_smem_buf[buf]);
      tma_store_2d(tma_d, /*x=*/n_offset_d_out + col0_out, /*y=*/m_offset, d_smem_addr);
      cp_async_bulk_commit_group();
    }
    wp_end(wpc, WP_EPI_STORE);
    // No post-TMA-issue bar.sync: next sub's STS races safely; same-buf
    // reuse is guarded by the wait_group+bar.sync above; end-of-tile
    // drain catches warp 4 lane 0.
    //asm volatile("bar.sync 1, 128;\n" ::: "memory");
  }
  wp_begin(wpc, WP_EPI_WAIT_STORE);
  if constexpr (DRAIN_PER_TILE) {
    if (warp == 4 && lane == 0) {
      cp_async_bulk_wait_group<0>();
    }
    asm volatile("bar.sync 1, 128;\n" ::: "memory");
  }
  wp_end(wpc, WP_EPI_WAIT_STORE);
}

/* ============================================================================
 * epi_warp_blackwell_ntiles_2sm_bf16_swiglu<...>(wpc, ...)
 *
 * Self-driving wrapper around `_1tile_1sm2sm_bf16_swiglu`. Mirrors
 * `_ntiles_2sm_bf16` but computes the per-tile output N-offset in
 * post-SwiGLU coordinates (n_offset_d_out = n_tile * N_TILE_CLUSTER / 2).
 * The CLC raster still walks the pre-SwiGLU N (N_TILE_CLUSTER cols per
 * tile), so grid math is unchanged from the non-fused kernel.
 * ============================================================================ */
template <int M_TILE_PER_CTA, int N_TILE_PER_CTA, int K_TILE,
          int M_TILE_CLUSTER, int N_TILE_CLUSTER,
          int EPI_SUB_COLS, int EPI_NUM_BUFS,
          int CLUSTER_SHAPE_M = 1, int CLUSTER_SHAPE_N = 2,
          ClcRasterOrder ORDER = ClcRasterOrder::AlongN,
          bool DRAIN_PER_TILE = true>
__device__ inline
void epi_warp_blackwell_ntiles_2sm_bf16_swiglu(WpCtx& wpc,
    const CUtensorMap* tma_d,
    uint8_t* smem_d,
    AccPipeline2BankBars acc_bars,
    uint64_t* clc_full_bar, uint64_t* clc_empty_bar, uint32_t* clc_response,
    uint32_t tmem_base,
    int peer, int warp, int lane,
    const int* __restrict__ m_tile_remap = nullptr,
    uint32_t* tmem_slot = nullptr) {  // if set, this block waits for + reads tmem_base
  static_assert(N_TILE_CLUSTER <= 256, "N_TILE_CLUSTER <= 256: TMEM has 512 cols, 2-bank pipeline -> 256 cols per bank");
  static_assert(N_TILE_CLUSTER % 2 == 0, "SwiGLU EPI requires N_TILE_CLUSTER even");
  AccPipeline2BankState cons_state = acc_pipeline_2bank_state_init();

  // EPI waits for MMA's tcgen05.alloc to publish tmem_base (named bar 6) when
  // tmem_slot is given -- symmetric with the base epi block.
  if (tmem_slot != nullptr) {
    wp_begin(wpc, WP_EPI_WAIT_TMEM);
    asm volatile("bar.sync 6, 160;\n" ::: "memory");
    tmem_base = *tmem_slot;
    wp_end(wpc, WP_EPI_WAIT_TMEM);
  }

  auto remap = [&](int p1) -> int {
    return (m_tile_remap != nullptr) ? m_tile_remap[p1] : p1;
  };

  int      clc_cons_stage = 0;
  uint32_t clc_cons_phase = 0;
  int m_tile, n_tile;
  if constexpr (ORDER == ClcRasterOrder::AlongN) {
    m_tile = remap((int)blockIdx.y);
    n_tile = (int)(blockIdx.x >> 1);
  } else {  // AlongM: cluster.x -> M
    m_tile = remap((int)(blockIdx.x >> 1));
    n_tile = (int)blockIdx.y;
  }

  while (true) {
    const int m_offset       = m_tile * M_TILE_CLUSTER + peer * M_TILE_PER_CTA;
    const int n_offset_d_out = n_tile * (N_TILE_CLUSTER / 2);

    epi_warp_blackwell_1tile_1sm2sm_bf16_swiglu<
        M_TILE_PER_CTA, N_TILE_CLUSTER, EPI_SUB_COLS, EPI_NUM_BUFS,
        DRAIN_PER_TILE>(wpc, tma_d, smem_d, acc_bars, cons_state, tmem_base,
        peer, warp, lane, m_offset, n_offset_d_out);

    const bool epi_release = (warp == 4 && lane == 0);
    wp_begin(wpc, WP_CLC_FETCH);
    ClcTileInfo next = clc_fetch_next_tile<
        CLUSTER_SHAPE_M, CLUSTER_SHAPE_N, ORDER>(
        clc_full_bar, clc_empty_bar, clc_response,
        clc_cons_stage, clc_cons_phase, /*do_release=*/epi_release);
    wp_end(wpc, WP_CLC_FETCH);
    clc_fetch_next_tile_advance(clc_cons_stage, clc_cons_phase);
    if (!next.valid) break;
    m_tile = remap(next.m_tile);
    n_tile = next.n_tile;
  }
  if constexpr (!DRAIN_PER_TILE) {
    wp_begin(wpc, WP_EPI_WAIT_STORE);
    if (warp == 4 && lane == 0) {
      cp_async_bulk_wait_group<0>();
    }
    asm volatile("bar.sync 1, 128;\n" ::: "memory");
    wp_end(wpc, WP_EPI_WAIT_STORE);
  }
}

/* ============================================================================
 * epi_warp_blackwell_1tile_1sm2sm_bf16_swiglu_chunked<M_TILE_PER_CTA, N_TILE_CLUSTER,
 *                                                EPI_SUB_COLS, EPI_NUM_BUFS>(wpc, ...)
 *
 * SwiGLU-fused 2SM BF16 epilogue, chunked variant (128-col chunk
 * pre-pack; "M2").
 *
 * B layout ("M2"): within each 256-wide
 * N-tile, cols 0..127 are up and cols 128..255 are gate. The MMA
 * accumulator therefore arrives in TMEM with the same split: TMEM cols
 * [0, N_TILE_CLUSTER/2) = up, [N_TILE_CLUSTER/2, N_TILE_CLUSTER) = gate.
 * Per (m, n_out) output position, fuse:
 *   D[m, n_out] = bf16( silu(gate[n_out]) * up[n_out] )
 *
 * EPI loop iterates EPI_SUB_COUNT_OUT = (N_TILE_CLUSTER / 2) / EPI_SUB_COLS
 * sub-tiles. Each sub issues two tcgen05.ld reads: one at TMEM offset
 * `col0` (up slice of width EPI_SUB_COLS) and one at `col0 +
 * N_TILE_CLUSTER/2` (gate slice). The fused output is `EPI_SUB_COLS` bf16
 * cols per sub (same width as the input slices, NOT halved like the
 * interleaved sibling).
 *
 * SMEM EPI buf width = `EPI_SUB_COLS` bf16 per row (matches the non-fused
 * kernel). TMA store box = `EPI_SUB_COLS` cols at post-SwiGLU N offsets:
 *   x = n_offset_d_out + sub * EPI_SUB_COLS
 *
 * Caller is responsible for:
 *   - Allocating smem_d with size EPI_NUM_BUFS * M_TILE_PER_CTA *
 *     EPI_SUB_COLS * sizeof(bf16) (= same as non-fused kernel).
 *   - Building D's TMA tensormap with width = post-SwiGLU N.
 *
 * Trade-off vs interleaved (`_swiglu`): same MMA / load / SMEM A+B, but
 * two TMEM reads per output sub (vs one). Interleaved reads
 * EPI_SUB_COLS cols and produces EPI_SUB_COLS/2 outputs (half-cols
 * stored); chunked reads 2*EPI_SUB_COLS cols and produces EPI_SUB_COLS
 * outputs (same-cols stored). Net TMEM read volume is identical;
 * chunked wins on TMA store burst width, interleaved wins on TMEM read
 * instruction count.
 * ============================================================================ */
template <int M_TILE_PER_CTA, int N_TILE_CLUSTER, int EPI_SUB_COLS, int EPI_NUM_BUFS,
          bool DRAIN_PER_TILE = true>
__device__ inline
void epi_warp_blackwell_1tile_1sm2sm_bf16_swiglu_chunked(WpCtx& wpc,
    const CUtensorMap* tma_d,
    uint8_t* smem_d,
    AccPipeline2BankBars acc_bars,
    AccPipeline2BankState& cons_state,
    uint32_t tmem_base,
    int peer, int warp, int lane,
    int m_offset, int n_offset_d_out) {
  static_assert(N_TILE_CLUSTER <= 256, "N_TILE_CLUSTER <= 256: TMEM has 512 cols, 2-bank pipeline -> 256 cols per bank");
  static_assert(N_TILE_CLUSTER % 2 == 0, "chunked SwiGLU EPI requires N_TILE_CLUSTER even (up half / gate half split)");
  static_assert((N_TILE_CLUSTER / 2) % EPI_SUB_COLS == 0,
                "chunked SwiGLU: (N_TILE_CLUSTER/2) must be a multiple of EPI_SUB_COLS");
  static_assert(EPI_SUB_COLS % 2 == 0, "EPI_SUB_COLS must be even (cvt_pack_f32_to_bf16x2 consumes pairs)");
  constexpr int GATE_BASE         = N_TILE_CLUSTER / 2;
  constexpr int EPI_SUB_COUNT_OUT = (N_TILE_CLUSTER / 2) / EPI_SUB_COLS;
  constexpr int EPI_BUF_BYTES_OUT = M_TILE_PER_CTA * EPI_SUB_COLS *
                                     static_cast<int>(sizeof(__nv_bfloat16));
  (void)peer;

  const int epi_warp = (warp >= 4) ? (warp - 4) : 0;
  const uint32_t my_tmem_row_offset = ((uint32_t)(epi_warp * 32) << 16);
  const int row = epi_warp * 32 + lane;

  __nv_bfloat16* d_smem_buf[EPI_NUM_BUFS];
  #pragma unroll
  for (int b = 0; b < EPI_NUM_BUFS; ++b) {
    d_smem_buf[b] = reinterpret_cast<__nv_bfloat16*>(smem_d + b * EPI_BUF_BYTES_OUT);
  }

  acc_pipeline_2bank_consumer_wait(acc_bars, cons_state);
  const int acc_stage_cons = acc_pipeline_2bank_state_index(cons_state);
  const uint32_t my_tmem = tmem_base + my_tmem_row_offset
                           + (uint32_t)(acc_stage_cons * N_TILE_CLUSTER);

  #pragma unroll
  for (int sub = 0; sub < EPI_SUB_COUNT_OUT; ++sub) {
    const int buf  = sub & (EPI_NUM_BUFS - 1);
    const int col0 = sub * EPI_SUB_COLS;
    uint32_t regs_up  [EPI_SUB_COLS];
    uint32_t regs_gate[EPI_SUB_COLS];
    const uint32_t taddr_up   = my_tmem + (uint32_t)col0;
    const uint32_t taddr_gate = my_tmem + (uint32_t)(GATE_BASE + col0);
    static_assert(EPI_SUB_COLS == 2   || EPI_SUB_COLS == 4   ||
                  EPI_SUB_COLS == 8   || EPI_SUB_COLS == 16  ||
                  EPI_SUB_COLS == 32  || EPI_SUB_COLS == 64  ||
                  EPI_SUB_COLS == 128,
                  "EPI_SUB_COLS must be 2/4/8/16/32/64/128 for chunked SwiGLU EPI");
    if      constexpr (EPI_SUB_COLS == 2)   { tcgen05_ld_32x32b_x2  (taddr_up, regs_up);
                                              tcgen05_ld_32x32b_x2  (taddr_gate, regs_gate); }
    else if constexpr (EPI_SUB_COLS == 4)   { tcgen05_ld_32x32b_x4  (taddr_up, regs_up);
                                              tcgen05_ld_32x32b_x4  (taddr_gate, regs_gate); }
    else if constexpr (EPI_SUB_COLS == 8)   { tcgen05_ld_32x32b_x8  (taddr_up, regs_up);
                                              tcgen05_ld_32x32b_x8  (taddr_gate, regs_gate); }
    else if constexpr (EPI_SUB_COLS == 16)  { tcgen05_ld_32x32b_x16 (taddr_up, regs_up);
                                              tcgen05_ld_32x32b_x16 (taddr_gate, regs_gate); }
    else if constexpr (EPI_SUB_COLS == 32)  { tcgen05_ld_32x32b_x32 (taddr_up, regs_up);
                                              tcgen05_ld_32x32b_x32 (taddr_gate, regs_gate); }
    else if constexpr (EPI_SUB_COLS == 64)  { tcgen05_ld_32x32b_x64 (taddr_up, regs_up);
                                              tcgen05_ld_32x32b_x64 (taddr_gate, regs_gate); }
    else if constexpr (EPI_SUB_COLS == 128) { tcgen05_ld_32x32b_x128(taddr_up, regs_up);
                                              tcgen05_ld_32x32b_x128(taddr_gate, regs_gate); }
    tcgen05_wait_ld();

    // Early acc_empty release on last sub (TMEM fully read by then).
    if (sub == EPI_SUB_COUNT_OUT - 1) {
      acc_pipeline_2bank_consumer_release(acc_bars, cons_state);
      acc_pipeline_2bank_state_advance(cons_state);
    }

    // See `_1tile_1sm2sm_bf16` for the rationale: bar.sync 1, 128 forces
    // all 128 EPI threads to wait for warp 4 lane 0's wait_group so the
    // next sub's STS does not race with the in-flight TMA's SMEM read.
    if constexpr (DRAIN_PER_TILE) {
      if (sub >= EPI_NUM_BUFS) {
        if (warp == 4 && lane == 0) {
          cp_async_bulk_wait_group<EPI_NUM_BUFS - 1>();
        }
        asm volatile("bar.sync 1, 128;\n" ::: "memory");
      }
    } else {
      if (warp == 4 && lane == 0) {
        cp_async_bulk_wait_group<EPI_NUM_BUFS - 1>();
      }
      asm volatile("bar.sync 1, 128;\n" ::: "memory");
    }

    // Fuse silu(gate) * up; pack each consecutive pair into one bf16x2
    // and write SMEM. Output stride = EPI_SUB_COLS bf16 cols per row.
    uint32_t* d_row_u32 = reinterpret_cast<uint32_t*>(
        d_smem_buf[buf] + row * EPI_SUB_COLS);
    #pragma unroll
    for (int j = 0; j < EPI_SUB_COLS; j += 2) {
      const float u0 = __int_as_float(regs_up  [j]);
      const float g0 = __int_as_float(regs_gate[j]);
      const float u1 = __int_as_float(regs_up  [j + 1]);
      const float g1 = __int_as_float(regs_gate[j + 1]);
      // Tanh-based SwiGLU via composite 130; same math as
      // silu(gate)*up but avoids the __expf overflow-to-Inf path
      // that triggers a slow special-value handler in the SM.
      d_row_u32[j >> 1] = cvt_pack_f32_to_bf16x2(
          swiglu_act_f32(g0, u0), swiglu_act_f32(g1, u1));
    }
    asm volatile("bar.sync 1, 128;\n" ::: "memory");
    if (warp == 4 && lane == 0) {
      fence_proxy_async_shared_cta();
      const uint32_t d_smem_addr = smem_ptr_u32(d_smem_buf[buf]);
      tma_store_2d(tma_d, /*x=*/n_offset_d_out + col0, /*y=*/m_offset, d_smem_addr);
      cp_async_bulk_commit_group();
    }
    // No post-TMA-issue bar.sync: next sub's STS races safely; same-buf
    // reuse is guarded by the wait_group+bar.sync above; end-of-tile
    // drain catches warp 4 lane 0.
    //asm volatile("bar.sync 1, 128;\n" ::: "memory");
  }
  if constexpr (DRAIN_PER_TILE) {
    if (warp == 4 && lane == 0) {
      cp_async_bulk_wait_group<0>();
    }
    asm volatile("bar.sync 1, 128;\n" ::: "memory");
  }
}

/* ============================================================================
 * epi_warp_blackwell_ntiles_2sm_bf16_swiglu_chunked<...>(wpc, ...)
 *
 * Self-driving wrapper around `_1tile_1sm2sm_bf16_swiglu_chunked`.
 * Mirrors `_ntiles_2sm_bf16_swiglu` (the interleaved sibling);
 * identical CLC raster, identical post-SwiGLU n-offset arithmetic.
 * Only the per-tile body differs.
 * ============================================================================ */
template <int M_TILE_PER_CTA, int N_TILE_PER_CTA, int K_TILE,
          int M_TILE_CLUSTER, int N_TILE_CLUSTER,
          int EPI_SUB_COLS, int EPI_NUM_BUFS,
          int CLUSTER_SHAPE_M = 1, int CLUSTER_SHAPE_N = 2,
          ClcRasterOrder ORDER = ClcRasterOrder::AlongN,
          bool DRAIN_PER_TILE = true>
__device__ inline
void epi_warp_blackwell_ntiles_2sm_bf16_swiglu_chunked(WpCtx& wpc,
    const CUtensorMap* tma_d,
    uint8_t* smem_d,
    AccPipeline2BankBars acc_bars,
    uint64_t* clc_full_bar, uint64_t* clc_empty_bar, uint32_t* clc_response,
    uint32_t tmem_base,
    int peer, int warp, int lane,
    const int* __restrict__ m_tile_remap = nullptr) {
  static_assert(N_TILE_CLUSTER <= 256, "N_TILE_CLUSTER <= 256: TMEM has 512 cols, 2-bank pipeline -> 256 cols per bank");
  static_assert(N_TILE_CLUSTER % 2 == 0, "chunked SwiGLU EPI requires N_TILE_CLUSTER even");
  AccPipeline2BankState cons_state = acc_pipeline_2bank_state_init();

  auto remap = [&](int p1) -> int {
    return (m_tile_remap != nullptr) ? m_tile_remap[p1] : p1;
  };

  int      clc_cons_stage = 0;
  uint32_t clc_cons_phase = 0;
  int m_tile, n_tile;
  if constexpr (ORDER == ClcRasterOrder::AlongN) {
    m_tile = remap((int)blockIdx.y);
    n_tile = (int)(blockIdx.x >> 1);
  } else {  // AlongM: cluster.x -> M
    m_tile = remap((int)(blockIdx.x >> 1));
    n_tile = (int)blockIdx.y;
  }

  while (true) {
    const int m_offset       = m_tile * M_TILE_CLUSTER + peer * M_TILE_PER_CTA;
    const int n_offset_d_out = n_tile * (N_TILE_CLUSTER / 2);

    epi_warp_blackwell_1tile_1sm2sm_bf16_swiglu_chunked<
        M_TILE_PER_CTA, N_TILE_CLUSTER, EPI_SUB_COLS, EPI_NUM_BUFS,
        DRAIN_PER_TILE>(wpc, tma_d, smem_d, acc_bars, cons_state, tmem_base,
        peer, warp, lane, m_offset, n_offset_d_out);

    const bool epi_release = (warp == 4 && lane == 0);
    ClcTileInfo next = clc_fetch_next_tile<
        CLUSTER_SHAPE_M, CLUSTER_SHAPE_N, ORDER>(
        clc_full_bar, clc_empty_bar, clc_response,
        clc_cons_stage, clc_cons_phase, /*do_release=*/epi_release);
    clc_fetch_next_tile_advance(clc_cons_stage, clc_cons_phase);
    if (!next.valid) break;
    m_tile = remap(next.m_tile);
    n_tile = next.n_tile;
  }
  if constexpr (!DRAIN_PER_TILE) {
    if (warp == 4 && lane == 0) {
      cp_async_bulk_wait_group<0>();
    }
    asm volatile("bar.sync 1, 128;\n" ::: "memory");
  }
}

/* ============================================================================
 * epi_warp_blackwell_ntiles_1sm_bf16<...>(wpc, ...)
 *
 * 1SM (cta_group::1, no cluster) counterpart of `_ntiles_2sm_bf16`. No
 * peer / no /2 in blockIdx; m_offset/n_offset computed directly from
 * the CLC-returned tile coords. Per-tile body delegates to the existing
 * `_1tile_1sm2sm_bf16` helper with peer=0.
 * ============================================================================ */
template <int M_TILE_PER_CTA, int N_TILE_PER_CTA,
          int EPI_SUB_COLS, int EPI_NUM_BUFS,
          int CLUSTER_SHAPE_M = 1, int CLUSTER_SHAPE_N = 1,
          ClcRasterOrder ORDER = ClcRasterOrder::AlongN,
          bool DRAIN_PER_TILE = true>
__device__ inline
void epi_warp_blackwell_ntiles_1sm_bf16(WpCtx& wpc,
    const CUtensorMap* tma_d,
    uint8_t* smem_d,
    AccPipeline2BankBars acc_bars,
    uint64_t* clc_full_bar, uint64_t* clc_empty_bar, uint32_t* clc_response,
    uint32_t tmem_base,
    int warp, int lane) {
  static_assert(N_TILE_PER_CTA <= 256, "N_TILE_PER_CTA <= 256: TMEM has 512 cols, 2-bank pipeline -> 256 cols per bank");
  AccPipeline2BankState cons_state = acc_pipeline_2bank_state_init();

  int      clc_cons_stage = 0;
  uint32_t clc_cons_phase = 0;
  int m_tile, n_tile;
  if constexpr (ORDER == ClcRasterOrder::AlongN) {
    m_tile = (int)blockIdx.y;
    n_tile = (int)blockIdx.x;
  } else {  // AlongM
    m_tile = (int)blockIdx.x;
    n_tile = (int)blockIdx.y;
  }

  while (true) {
    const int m_offset = m_tile * M_TILE_PER_CTA;
    const int n_offset = n_tile * N_TILE_PER_CTA;

    epi_warp_blackwell_1tile_1sm2sm_bf16<
        M_TILE_PER_CTA, N_TILE_PER_CTA, EPI_SUB_COLS, EPI_NUM_BUFS,
        DRAIN_PER_TILE>(wpc, tma_d, smem_d, acc_bars, cons_state, tmem_base,
        /*peer=*/0, warp, lane, m_offset, n_offset);

    const bool epi_release = (warp == 4 && lane == 0);
    ClcTileInfo next = clc_fetch_next_tile<
        CLUSTER_SHAPE_M, CLUSTER_SHAPE_N, ORDER, /*CTA_GROUP=*/1>(
        clc_full_bar, clc_empty_bar, clc_response,
        clc_cons_stage, clc_cons_phase, /*do_release=*/epi_release);
    clc_fetch_next_tile_advance(clc_cons_stage, clc_cons_phase);
    if (!next.valid) break;
    m_tile = next.m_tile;
    n_tile = next.n_tile;
  }
  if constexpr (!DRAIN_PER_TILE) {
    if (warp == 4 && lane == 0) {
      cp_async_bulk_wait_group<0>();
    }
    asm volatile("bar.sync 1, 128;\n" ::: "memory");
  }
}

/* ============================================================================
 * epi_warp_blackwell_1tile_1sm2sm_bf16_swiglu_interleaved<...>(wpc, ...)
 *
 * SwiGLU-fused 2SM/1SM BF16 epilogue, M3 1-col interleaved variant.
 *
 * B layout: per N-tile, columns alternate up/gate. Pack convention is
 * up = even col, gate = odd col. The MMA accumulator therefore arrives
 * in TMEM with the same interleave: TMEM col 2n = up[n],
 * col 2n+1 = gate[n] for n in [0, N_TILE_CLUSTER / 2).
 *
 * Per output col n_out in [0, EPI_SUB_COLS), fuse:
 *   D[m, sub * EPI_SUB_COLS + n_out] =
 *     bf16( silu(gate[n_out]) * up[n_out] )
 *
 * Each sub issues ONE tcgen05.ld reading 2 * EPI_SUB_COLS adjacent cols
 * (up/gate pairs) -- vs M2's TWO reads at strided cols. This is the
 * M3-vs-M2 advantage: the wider single read can saturate `tcgen05.ld`'s
 * 32x32b vector lanes more cleanly.
 *
 * Caller responsibilities (identical to M2 chunked):
 *   - smem_d size = EPI_NUM_BUFS * M_TILE_PER_CTA * EPI_SUB_COLS *
 *     sizeof(bf16).
 *   - D's TMA tensormap width = post-SwiGLU N (= pre-activation N / 2).
 *
 * Constraints:
 *   - 2 * EPI_SUB_COLS must be a valid tcgen05.ld 32x32b x-suffix:
 *     2 * EPI_SUB_COLS in {2,4,8,16,32,64,128} => EPI_SUB_COLS in
 *     {1,2,4,8,16,32,64}. EPI_SUB_COLS must also be even (cvt pair pack).
 *   - (N_TILE_CLUSTER / 2) % EPI_SUB_COLS == 0 (sub count is integer).
 * ============================================================================ */
template <int M_TILE_PER_CTA, int N_TILE_CLUSTER, int EPI_SUB_COLS, int EPI_NUM_BUFS,
          bool DRAIN_PER_TILE = true>
__device__ inline
void epi_warp_blackwell_1tile_1sm2sm_bf16_swiglu_interleaved(WpCtx& wpc,
    const CUtensorMap* tma_d,
    uint8_t* smem_d,
    AccPipeline2BankBars acc_bars,
    AccPipeline2BankState& cons_state,
    uint32_t tmem_base,
    int peer, int warp, int lane,
    int m_offset, int n_offset_d_out) {
  static_assert(N_TILE_CLUSTER <= 256, "N_TILE_CLUSTER <= 256");
  static_assert(N_TILE_CLUSTER % 2 == 0,
                "interleaved SwiGLU EPI requires N_TILE_CLUSTER even");
  static_assert((N_TILE_CLUSTER / 2) % EPI_SUB_COLS == 0,
                "interleaved SwiGLU: (N_TILE_CLUSTER/2) must be a multiple of EPI_SUB_COLS");
  static_assert(EPI_SUB_COLS % 2 == 0,
                "EPI_SUB_COLS must be even (cvt_pack_f32_to_bf16x2 consumes pairs)");
  static_assert(EPI_SUB_COLS == 2  || EPI_SUB_COLS == 4  ||
                EPI_SUB_COLS == 8  || EPI_SUB_COLS == 16 ||
                EPI_SUB_COLS == 32 || EPI_SUB_COLS == 64,
                "EPI_SUB_COLS must be 2/4/8/16/32/64 (2*ESC <= 128 = max tcgen05.ld x)");
  constexpr int EPI_SUB_COUNT_OUT = (N_TILE_CLUSTER / 2) / EPI_SUB_COLS;
  constexpr int EPI_BUF_BYTES_OUT = M_TILE_PER_CTA * EPI_SUB_COLS *
                                     static_cast<int>(sizeof(__nv_bfloat16));
  constexpr int LD_WIDTH = 2 * EPI_SUB_COLS;  // cols read per tcgen05.ld
  (void)peer;

  const int epi_warp = (warp >= 4) ? (warp - 4) : 0;
  const uint32_t my_tmem_row_offset = ((uint32_t)(epi_warp * 32) << 16);
  const int row = epi_warp * 32 + lane;

  __nv_bfloat16* d_smem_buf[EPI_NUM_BUFS];
  #pragma unroll
  for (int b = 0; b < EPI_NUM_BUFS; ++b) {
    d_smem_buf[b] = reinterpret_cast<__nv_bfloat16*>(smem_d + b * EPI_BUF_BYTES_OUT);
  }

  wp_begin(wpc, WP_EPI_WAIT_ACC);
  acc_pipeline_2bank_consumer_wait(acc_bars, cons_state);
  wp_end(wpc, WP_EPI_WAIT_ACC);
  const int acc_stage_cons = acc_pipeline_2bank_state_index(cons_state);
  const uint32_t my_tmem = tmem_base + my_tmem_row_offset
                           + (uint32_t)(acc_stage_cons * N_TILE_CLUSTER);

  #pragma unroll
  for (int sub = 0; sub < EPI_SUB_COUNT_OUT; ++sub) {
    const int buf  = sub & (EPI_NUM_BUFS - 1);
    // TMEM input col0 = sub * 2 * EPI_SUB_COLS (pairs are 2 cols wide).
    const int tmem_col0 = sub * LD_WIDTH;
    uint32_t regs[LD_WIDTH];  // interleaved: regs[2i]=up_i, regs[2i+1]=gate_i
    const uint32_t taddr = my_tmem + (uint32_t)tmem_col0;
    wp_begin(wpc, WP_EPI_TMEM_LD);
    if      constexpr (LD_WIDTH == 4)   tcgen05_ld_32x32b_x4  (taddr, regs);
    else if constexpr (LD_WIDTH == 8)   tcgen05_ld_32x32b_x8  (taddr, regs);
    else if constexpr (LD_WIDTH == 16)  tcgen05_ld_32x32b_x16 (taddr, regs);
    else if constexpr (LD_WIDTH == 32)  tcgen05_ld_32x32b_x32 (taddr, regs);
    else if constexpr (LD_WIDTH == 64)  tcgen05_ld_32x32b_x64 (taddr, regs);
    else if constexpr (LD_WIDTH == 128) tcgen05_ld_32x32b_x128(taddr, regs);
    tcgen05_wait_ld();
    wp_end(wpc, WP_EPI_TMEM_LD);

    if (sub == EPI_SUB_COUNT_OUT - 1) {
      acc_pipeline_2bank_consumer_release(acc_bars, cons_state);
      acc_pipeline_2bank_state_advance(cons_state);
    }

    wp_begin(wpc, WP_EPI_WAIT_STORE);
    if constexpr (DRAIN_PER_TILE) {
      if (sub >= EPI_NUM_BUFS) {
        if (warp == 4 && lane == 0) {
          cp_async_bulk_wait_group<EPI_NUM_BUFS - 1>();
        }
        asm volatile("bar.sync 1, 128;\n" ::: "memory");
      }
    } else {
      if (warp == 4 && lane == 0) {
        cp_async_bulk_wait_group<EPI_NUM_BUFS - 1>();
      }
      asm volatile("bar.sync 1, 128;\n" ::: "memory");
    }
    wp_end(wpc, WP_EPI_WAIT_STORE);

    // Pack: for each output pair (j, j+1), fuse silu(gate)*up. Input
    // regs are interleaved (up=even, gate=odd), so the j-th output uses
    // regs[2j] (up) and regs[2j+1] (gate).
    wp_begin(wpc, WP_EPI_SWIGLU);
    uint32_t* d_row_u32 = reinterpret_cast<uint32_t*>(
        d_smem_buf[buf] + row * EPI_SUB_COLS);
    #pragma unroll
    for (int j = 0; j < EPI_SUB_COLS; j += 2) {
      const float u0 = __int_as_float(regs[2 * j + 0]);
      const float g0 = __int_as_float(regs[2 * j + 1]);
      const float u1 = __int_as_float(regs[2 * j + 2]);
      const float g1 = __int_as_float(regs[2 * j + 3]);
      d_row_u32[j >> 1] = cvt_pack_f32_to_bf16x2(
          swiglu_act_f32(g0, u0), swiglu_act_f32(g1, u1));
    }
    wp_end(wpc, WP_EPI_SWIGLU);
    wp_begin(wpc, WP_EPI_STORE);
    asm volatile("bar.sync 1, 128;\n" ::: "memory");
    if (warp == 4 && lane == 0) {
      fence_proxy_async_shared_cta();
      const uint32_t d_smem_addr = smem_ptr_u32(d_smem_buf[buf]);
      tma_store_2d(tma_d,
                   /*x=*/n_offset_d_out + sub * EPI_SUB_COLS,
                   /*y=*/m_offset, d_smem_addr);
      cp_async_bulk_commit_group();
    }
    wp_end(wpc, WP_EPI_STORE);
  }
  wp_begin(wpc, WP_EPI_WAIT_STORE);
  if constexpr (DRAIN_PER_TILE) {
    if (warp == 4 && lane == 0) {
      cp_async_bulk_wait_group<0>();
    }
    asm volatile("bar.sync 1, 128;\n" ::: "memory");
  }
  wp_end(wpc, WP_EPI_WAIT_STORE);
}

/* ============================================================================
 * epi_warp_blackwell_ntiles_1sm_bf16_swiglu_interleaved<...>(wpc, ...)
 *
 * 1SM CLC-driven wrapper for the M3 (1-col interleaved) SwiGLU EPI.
 * Mirrors `_ntiles_1sm_bf16_swiglu_chunked` -- same CLC handshake,
 * same persistent loop. Output stride is N_TILE_PER_CTA / 2 (post-
 * SwiGLU width per tile).
 * ============================================================================ */
template <int M_TILE_PER_CTA, int N_TILE_PER_CTA,
          int EPI_SUB_COLS, int EPI_NUM_BUFS,
          int CLUSTER_SHAPE_M = 1, int CLUSTER_SHAPE_N = 1,
          ClcRasterOrder ORDER = ClcRasterOrder::AlongN,
          bool DRAIN_PER_TILE = true>
__device__ inline
void epi_warp_blackwell_ntiles_1sm_bf16_swiglu_interleaved(WpCtx& wpc,
    const CUtensorMap* tma_d,
    uint8_t* smem_d,
    AccPipeline2BankBars acc_bars,
    uint64_t* clc_full_bar, uint64_t* clc_empty_bar, uint32_t* clc_response,
    uint32_t tmem_base,
    int warp, int lane) {
  static_assert(N_TILE_PER_CTA <= 256, "N_TILE_PER_CTA <= 256");
  static_assert(N_TILE_PER_CTA % 2 == 0,
                "interleaved SwiGLU EPI requires N_TILE_PER_CTA even");
  AccPipeline2BankState cons_state = acc_pipeline_2bank_state_init();

  int      clc_cons_stage = 0;
  uint32_t clc_cons_phase = 0;
  int m_tile, n_tile;
  if constexpr (ORDER == ClcRasterOrder::AlongN) {
    m_tile = (int)blockIdx.y;
    n_tile = (int)blockIdx.x;
  } else {
    m_tile = (int)blockIdx.x;
    n_tile = (int)blockIdx.y;
  }

  while (true) {
    const int m_offset       = m_tile * M_TILE_PER_CTA;
    const int n_offset_d_out = n_tile * (N_TILE_PER_CTA / 2);

    epi_warp_blackwell_1tile_1sm2sm_bf16_swiglu_interleaved<
        M_TILE_PER_CTA, N_TILE_PER_CTA, EPI_SUB_COLS, EPI_NUM_BUFS,
        DRAIN_PER_TILE>(wpc, tma_d, smem_d, acc_bars, cons_state, tmem_base,
        /*peer=*/0, warp, lane, m_offset, n_offset_d_out);

    const bool epi_release = (warp == 4 && lane == 0);
    ClcTileInfo next = clc_fetch_next_tile<
        CLUSTER_SHAPE_M, CLUSTER_SHAPE_N, ORDER, /*CTA_GROUP=*/1>(
        clc_full_bar, clc_empty_bar, clc_response,
        clc_cons_stage, clc_cons_phase, /*do_release=*/epi_release);
    clc_fetch_next_tile_advance(clc_cons_stage, clc_cons_phase);
    if (!next.valid) break;
    m_tile = next.m_tile;
    n_tile = next.n_tile;
  }
  if constexpr (!DRAIN_PER_TILE) {
    if (warp == 4 && lane == 0) {
      cp_async_bulk_wait_group<0>();
    }
    asm volatile("bar.sync 1, 128;\n" ::: "memory");
  }
}

/* ============================================================================
 * epi_warp_blackwell_ntiles_2sm_bf16_swiglu_interleaved<...>(wpc, ...)
 *
 * 2SM CLC-driven wrapper for the M3 (1-col interleaved) SwiGLU EPI.
 * Mirrors `_ntiles_2sm_bf16_swiglu_chunked`; identical CLC raster and
 * per-CTA n-offset arithmetic. Only the per-tile body differs --
 * delegates to `_1tile_1sm2sm_bf16_swiglu_interleaved` (the M3 helper).
 * Output stride is N_TILE_CLUSTER / 2 (post-SwiGLU width).
 * ============================================================================ */
template <int M_TILE_PER_CTA, int N_TILE_PER_CTA, int K_TILE,
          int M_TILE_CLUSTER, int N_TILE_CLUSTER,
          int EPI_SUB_COLS, int EPI_NUM_BUFS,
          int CLUSTER_SHAPE_M = 1, int CLUSTER_SHAPE_N = 2,
          ClcRasterOrder ORDER = ClcRasterOrder::AlongN,
          bool DRAIN_PER_TILE = true>
__device__ inline
void epi_warp_blackwell_ntiles_2sm_bf16_swiglu_interleaved(WpCtx& wpc,
    const CUtensorMap* tma_d,
    uint8_t* smem_d,
    AccPipeline2BankBars acc_bars,
    uint64_t* clc_full_bar, uint64_t* clc_empty_bar, uint32_t* clc_response,
    uint32_t tmem_base,
    int peer, int warp, int lane,
    const int* __restrict__ m_tile_remap = nullptr) {
  static_assert(N_TILE_CLUSTER <= 256, "N_TILE_CLUSTER <= 256");
  static_assert(N_TILE_CLUSTER % 2 == 0,
                "interleaved SwiGLU EPI requires N_TILE_CLUSTER even");
  AccPipeline2BankState cons_state = acc_pipeline_2bank_state_init();

  auto remap = [&](int p1) -> int {
    return (m_tile_remap != nullptr) ? m_tile_remap[p1] : p1;
  };

  int      clc_cons_stage = 0;
  uint32_t clc_cons_phase = 0;
  int m_tile, n_tile;
  if constexpr (ORDER == ClcRasterOrder::AlongN) {
    m_tile = remap((int)blockIdx.y);
    n_tile = (int)(blockIdx.x >> 1);
  } else {
    m_tile = remap((int)(blockIdx.x >> 1));
    n_tile = (int)blockIdx.y;
  }

  while (true) {
    const int m_offset       = m_tile * M_TILE_CLUSTER + peer * M_TILE_PER_CTA;
    const int n_offset_d_out = n_tile * (N_TILE_CLUSTER / 2);

    epi_warp_blackwell_1tile_1sm2sm_bf16_swiglu_interleaved<
        M_TILE_PER_CTA, N_TILE_CLUSTER, EPI_SUB_COLS, EPI_NUM_BUFS,
        DRAIN_PER_TILE>(wpc, tma_d, smem_d, acc_bars, cons_state, tmem_base,
        peer, warp, lane, m_offset, n_offset_d_out);

    const bool epi_release = (warp == 4 && lane == 0);
    wp_begin(wpc, WP_CLC_FETCH);
    ClcTileInfo next = clc_fetch_next_tile<
        CLUSTER_SHAPE_M, CLUSTER_SHAPE_N, ORDER>(
        clc_full_bar, clc_empty_bar, clc_response,
        clc_cons_stage, clc_cons_phase, /*do_release=*/epi_release);
    wp_end(wpc, WP_CLC_FETCH);
    clc_fetch_next_tile_advance(clc_cons_stage, clc_cons_phase);
    if (!next.valid) break;
    m_tile = remap(next.m_tile);
    n_tile = next.n_tile;
  }
  if constexpr (!DRAIN_PER_TILE) {
    wp_begin(wpc, WP_EPI_WAIT_STORE);
    if (warp == 4 && lane == 0) {
      cp_async_bulk_wait_group<0>();
    }
    asm volatile("bar.sync 1, 128;\n" ::: "memory");
    wp_end(wpc, WP_EPI_WAIT_STORE);
  }
}

/* ============================================================================
 * epi_warp_blackwell_ntiles_1sm_bf16_swiglu_chunked<...>(wpc, ...)
 *
 * 1SM (cta_group::1, no cluster) counterpart of
 * `_ntiles_2sm_bf16_swiglu_chunked`. Same chunked B layout (up in cols
 * [0, N/2), gate in [N/2, N)) and same per-tile body via
 * `_1tile_1sm2sm_bf16_swiglu_chunked` with peer=0. Output D width is
 * N_TILE_PER_CTA / 2 (post-SwiGLU).
 * ============================================================================ */
template <int M_TILE_PER_CTA, int N_TILE_PER_CTA,
          int EPI_SUB_COLS, int EPI_NUM_BUFS,
          int CLUSTER_SHAPE_M = 1, int CLUSTER_SHAPE_N = 1,
          ClcRasterOrder ORDER = ClcRasterOrder::AlongN,
          bool DRAIN_PER_TILE = true>
__device__ inline
void epi_warp_blackwell_ntiles_1sm_bf16_swiglu_chunked(WpCtx& wpc,
    const CUtensorMap* tma_d,
    uint8_t* smem_d,
    AccPipeline2BankBars acc_bars,
    uint64_t* clc_full_bar, uint64_t* clc_empty_bar, uint32_t* clc_response,
    uint32_t tmem_base,
    int warp, int lane) {
  static_assert(N_TILE_PER_CTA <= 256, "N_TILE_PER_CTA <= 256");
  static_assert(N_TILE_PER_CTA % 2 == 0,
                "chunked SwiGLU EPI requires N_TILE_PER_CTA even");
  AccPipeline2BankState cons_state = acc_pipeline_2bank_state_init();

  int      clc_cons_stage = 0;
  uint32_t clc_cons_phase = 0;
  int m_tile, n_tile;
  if constexpr (ORDER == ClcRasterOrder::AlongN) {
    m_tile = (int)blockIdx.y;
    n_tile = (int)blockIdx.x;
  } else {
    m_tile = (int)blockIdx.x;
    n_tile = (int)blockIdx.y;
  }

  while (true) {
    const int m_offset       = m_tile * M_TILE_PER_CTA;
    const int n_offset_d_out = n_tile * (N_TILE_PER_CTA / 2);

    epi_warp_blackwell_1tile_1sm2sm_bf16_swiglu_chunked<
        M_TILE_PER_CTA, N_TILE_PER_CTA, EPI_SUB_COLS, EPI_NUM_BUFS,
        DRAIN_PER_TILE>(wpc, tma_d, smem_d, acc_bars, cons_state, tmem_base,
        /*peer=*/0, warp, lane, m_offset, n_offset_d_out);

    const bool epi_release = (warp == 4 && lane == 0);
    ClcTileInfo next = clc_fetch_next_tile<
        CLUSTER_SHAPE_M, CLUSTER_SHAPE_N, ORDER, /*CTA_GROUP=*/1>(
        clc_full_bar, clc_empty_bar, clc_response,
        clc_cons_stage, clc_cons_phase, /*do_release=*/epi_release);
    clc_fetch_next_tile_advance(clc_cons_stage, clc_cons_phase);
    if (!next.valid) break;
    m_tile = next.m_tile;
    n_tile = next.n_tile;
  }
  if constexpr (!DRAIN_PER_TILE) {
    if (warp == 4 && lane == 0) {
      cp_async_bulk_wait_group<0>();
    }
    asm volatile("bar.sync 1, 128;\n" ::: "memory");
  }
}

/* ============================================================================
 * epi_warp_blackwell_ntiles_1sm_tail_bf16<...>(wpc, ...)
 *
 * 1SM CLC-driven EPI warp for grouped-GEMM phase-2 tail. Same persistent
 * loop as `_ntiles_2sm_bf16`, but:
 *   - 1SM (no peer; m_offset uses tail_m_offset[m_tile] lookup directly).
 *   - n_offset_d = n_tile * N_TILE_PER_CTA (no peer split).
 *   - Cluster shape defaults to 1x1.
 *
 * Caller invokes from ALL 32 lanes of warps 4-7 (128 threads).
 * ============================================================================ */
template <int M_TILE_PER_CTA, int N_TILE_PER_CTA,
          int EPI_SUB_COLS, int EPI_NUM_BUFS,
          int CLUSTER_SHAPE_M = 1, int CLUSTER_SHAPE_N = 1,
          ClcRasterOrder ORDER = ClcRasterOrder::AlongN,
          bool DRAIN_PER_TILE = true>
__device__ inline
void epi_warp_blackwell_ntiles_1sm_tail_bf16(WpCtx& wpc,
    const CUtensorMap* tma_d,
    uint8_t* smem_d,
    AccPipeline2BankBars acc_bars,
    uint64_t* clc_full_bar, uint64_t* clc_empty_bar, uint32_t* clc_response,
    uint32_t tmem_base,
    int warp, int lane,
    const int* __restrict__ tail_m_offset) {
  static_assert(N_TILE_PER_CTA <= 256, "N_TILE_PER_CTA <= 256: TMEM has 512 cols, 2-bank pipeline -> 256 cols per bank");
  AccPipeline2BankState cons_state = acc_pipeline_2bank_state_init();

  int      clc_cons_stage = 0;
  uint32_t clc_cons_phase = 0;
  int m_tile, n_tile;
  if constexpr (ORDER == ClcRasterOrder::AlongN) {
    m_tile = (int)blockIdx.y;
    n_tile = (int)blockIdx.x;
  } else {
    m_tile = (int)blockIdx.x;
    n_tile = (int)blockIdx.y;
  }

  while (true) {
    const int m_offset = tail_m_offset[m_tile];
    const int n_offset = n_tile * N_TILE_PER_CTA;

    epi_warp_blackwell_1tile_1sm2sm_bf16<
        M_TILE_PER_CTA, N_TILE_PER_CTA, EPI_SUB_COLS, EPI_NUM_BUFS,
        DRAIN_PER_TILE>(wpc, tma_d, smem_d, acc_bars, cons_state, tmem_base,
        /*peer=*/0, warp, lane, m_offset, n_offset);

    const bool epi_release = (warp == 4 && lane == 0);
    ClcTileInfo next = clc_fetch_next_tile<
        CLUSTER_SHAPE_M, CLUSTER_SHAPE_N, ORDER, /*CTA_GROUP=*/1>(
        clc_full_bar, clc_empty_bar, clc_response,
        clc_cons_stage, clc_cons_phase, /*do_release=*/epi_release);
    clc_fetch_next_tile_advance(clc_cons_stage, clc_cons_phase);
    if (!next.valid) break;
    m_tile = next.m_tile;
    n_tile = next.n_tile;
  }
  if constexpr (!DRAIN_PER_TILE) {
    if (warp == 4 && lane == 0) {
      cp_async_bulk_wait_group<0>();
    }
    asm volatile("bar.sync 1, 128;\n" ::: "memory");
  }
}

/* ============================================================================
 * epi_store_warp_blackwell_1tile_1sm2sm_bf16_fmha<>(wpc, ...)
 *
 * One FMHA work-item's O store. STORE-ONLY: the TMEM -> *=1/l -> bf16 pack
 * lives in the correction warps; this warp only TMA-stores the corr-packed
 * sO staging buffers. Per M-tile: wait full_bar_o_epi[m] (corr signals pack
 * done) -> Q_SUBTILES x tma_store_4d (per-sample 4D map: rows past seqlen_q are
 * clamped, not written into the next sample) -> commit group. Then drain each
 * commit group and release sO[m] to corr as ITS store completes (empty_bar_o_epi[m]).
 * ============================================================================ */
template <int M_TILES_PER_CTA, int M_TILE, int HEAD_DIM>
__device__ inline
void epi_store_warp_blackwell_1tile_1sm2sm_bf16_fmha(WpCtx& wpc,
    const CUtensorMap* tmap_o, __nv_bfloat16* const* sO_bufs,
    uint64_t* full_bar_o_epi, uint64_t* empty_bar_o_epi,
    int sample, int h_kv, int q_tile_base, int q_tile_per_mtile, int gqa_group_size,
    PhaseTracker<1>& full_o_ph, bool do_store = true) {
  static_assert(M_TILES_PER_CTA == 2, "split-drain below assumes 2 M-tiles");
  constexpr int SUB_COLS_BF16    = 64;   // B128 swizzle atom = 128 B = 64 bf16
  constexpr int SUB_COLS_BYTES   = SUB_COLS_BF16 * (int)sizeof(__nv_bfloat16);
  constexpr int Q_SUBTILES       = HEAD_DIM / SUB_COLS_BF16;
  constexpr int Q_SUB_COLS_BYTES = M_TILE * SUB_COLS_BYTES;

  #pragma unroll
  for (int m = 0; m < M_TILES_PER_CTA; ++m) {
    wp_begin(wpc, WP_EPI_WAIT_TMEM);
    mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_o_epi[m]), full_o_ph.get_phase());
    wp_end(wpc, WP_EPI_WAIT_TMEM);

    wp_begin(wpc, WP_EPI_STORE);
    if (do_store && elect_one_sync()) {
      const int tok0 = q_tile_base + m * q_tile_per_mtile;
      #pragma unroll
      for (int s = 0; s < Q_SUBTILES; ++s) {
        tma_store_4d(tmap_o, s * SUB_COLS_BF16, h_kv * gqa_group_size, tok0, sample,
                     smem_ptr_u32(reinterpret_cast<const uint8_t*>(sO_bufs[m]) + s * Q_SUB_COLS_BYTES));
      }
      cp_async_bulk_commit_group();
    }
    wp_end(wpc, WP_EPI_STORE);
  }

  wp_begin(wpc, WP_EPI_WAIT_STORE);
  // Drain per-m commit groups; release each sO slot to corr as ITS store completes
  // (without this, corr's next pack races the in-flight store at tiny causal K-loops).
  if (elect_one_sync()) {
    cp_async_bulk_wait_group_read<1>();
    mbarrier_arrive(smem_ptr_u32(&empty_bar_o_epi[0]));
    cp_async_bulk_wait_group_read<0>();
    mbarrier_arrive(smem_ptr_u32(&empty_bar_o_epi[1]));
  }
  wp_end(wpc, WP_EPI_WAIT_STORE);

  full_o_ph.advance();
}

/* ============================================================================
 * epi_store_warp_blackwell_ntiles_1sm_bf16_fmha<>(...)
 *
 * Production __device__ body for the EPI (store) warp of the 1SM Blackwell BF16
 * FMHA context kernel: prime empty_bar_o_epi once, then a persistent loop of
 * decode_workitem (#110) + epi_store_warp_blackwell_1tile_1sm2sm_bf16_fmha, driven
 * by CLC (#106) or grid-stride.
 * ============================================================================ */
template <int M_TILES_PER_CTA, int M_TILE, int HEAD_DIM, int K_TILE,
          bool USE_CLC, int CLC_STAGES, bool Q_RASTER, bool IS_CAUSAL, bool LPT,
          int EPI_REG_BUDGET = 48, int CLUSTER_N = 1>
__device__ inline
void epi_store_warp_blackwell_ntiles_1sm2sm_bf16_fmha(WpCtx& wpc,
    const CUtensorMap* tmap_o, __nv_bfloat16* const* sO_bufs,
    uint64_t* full_bar_o_epi, uint64_t* empty_bar_o_epi,
    uint64_t* clc_full, uint64_t* clc_empty, uint32_t* clc_response,
    int seqlen_q, int num_q_heads, int num_kv_heads, int packed_idx_per_seq, int num_samples,
    unsigned long long magic0, unsigned long long magic1, unsigned long long magic2) {
  setmaxnreg_dec<EPI_REG_BUDGET>();
  // CLUSTER_N=1 (1SM): peer=0, tile_id=blockIdx.x. CLUSTER_N=2 (2SM): peer=blockIdx.x&1,
  // tile_id=cluster_id; the ntiles pipeline is otherwise identical.
  const int peer = (int)(blockIdx.x % CLUSTER_N);

  const int gqa_group_size    = num_q_heads / num_kv_heads;
  const int q_tile_per_mtile  = M_TILE / gqa_group_size;
  const int q_tile_per_cta    = M_TILES_PER_CTA * q_tile_per_mtile;
  const int total_tiles       = num_samples * packed_idx_per_seq * num_kv_heads;  // packed-M tiles (1SM) / clusters (2SM)

  PhaseTracker<1> full_o_ph;
  // Prime empty_bar_o_epi once: corr's first sO pack must not block (no prior store in flight).
  if (elect_one_sync()) {
    #pragma unroll
    for (int m = 0; m < M_TILES_PER_CTA; ++m)
      mbarrier_arrive(smem_ptr_u32(&empty_bar_o_epi[m]));
  }
  [[maybe_unused]] int clc_stage = 0;
  [[maybe_unused]] uint32_t clc_phase = 0;
  int tile_id = (int)(blockIdx.x / CLUSTER_N);
  while (true) {
    int sample, h_kv, q_tile_base, K_TILES;
    decode_workitem<K_TILE, Q_RASTER, IS_CAUSAL, LPT, CLUSTER_N>(tile_id, seqlen_q, num_kv_heads,
        packed_idx_per_seq, q_tile_per_cta, magic0, magic1, magic2,
        sample, h_kv, q_tile_base, K_TILES, peer);

    // 2SM: the unpaired padding peer (q_tile_base past seqlen_q) waits in lockstep but does NOT store.
    const bool do_store = (CLUSTER_N == 1) || (q_tile_base < seqlen_q);
    epi_store_warp_blackwell_1tile_1sm2sm_bf16_fmha<M_TILES_PER_CTA, M_TILE, HEAD_DIM>(wpc, tmap_o, sO_bufs, full_bar_o_epi, empty_bar_o_epi,
        sample, h_kv, q_tile_base, q_tile_per_mtile, gqa_group_size, full_o_ph, do_store);

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
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
