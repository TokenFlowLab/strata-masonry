#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// 99_correction_warp.cuh -- FMHA accumulator-correction warp building block.
//
// ARCH: sm_100a
//
// The "correction warp" role for FMHA: rescale the running PV accumulator
// (resident in TMEM) by the factor returned by the softmax warp's online-
// softmax update (#85, #98). The factor is exp2(old_max - new_max) when
// the running row max grew; multiplying the prior accumulator by that
// factor before adding the next P*V correctly stitches the per-tile
// softmax into the row-global softmax.
//
// Source: knowledge/building_blocks/correction_warp.md
// PTX:    9.7.18.8 (TMEM ld/st)
//
// Block function (per code/PLAN.md "Block function signature contract"):
//
//   __device__ __forceinline__ void
//   correction_warp_block(uint32_t tmem_addr, float factor);
//
// - tmem_addr: 32-bit TMEM address (lane << 16 | col) of a 32-row x 8-col
//   FP32 accumulator slice owned by the caller.
// - factor: scalar rescale multiplier (typically exp2(old_max - new_max)).
//
// Post-condition: every cell of the 32x8 slice multiplied by `factor`
// in-place. Pre-condition: caller has issued tcgen05.wait::st (or
// equivalent) so prior writes to the slice are observable.
//
// Caller-gated. Caller must invoke from all 32 lanes of the warp that
// owns the slice (tcgen05.{ld,st}.sync.aligned are warp-collective).
// The block does not allocate / deallocate TMEM and does not own any
// SMEM region; the calling kernel does.

#include <cstdint>
#include "../composites/86_acc_correction.cuh"

__device__ __forceinline__
void correction_warp_block(uint32_t tmem_addr, float factor) {
  acc_correction_fp32_x8(tmem_addr, factor);
}

// ============================================================================
// Production FMHA correction warp (kernels/fmha/sm100a/fmha_context_bf16_uniform.cu).
// Owns the online-softmax O rescale + final epilogue for a 32-row band:
//   per K block, read alpha (softmax) and rescale the running O accumulator in
//   TMEM (LDTM -> FMUL2 -> STTM, x16 chunks); at the end, O *= 1/l -> bf16 ->
//   swizzled sO staging buffer -> signal the epi (store) warp (full_bar_o_epi).
// ============================================================================
#include <cuda_bf16.h>
#include "../primitives/9_tcgen05_ld.cuh"
#include "../primitives/10_tcgen05_st.cuh"
#include "../primitives/12_tcgen05_wait.cuh"
#include "../primitives/15_tcgen05_fence.cuh"
#include "../primitives/34_fence_proxy_async.cuh"
#include "../primitives/63_cvt_f32_to_f16_bf16.cuh"
#include "../primitives/76_packed_f32x2.cuh"
#include "../primitives/78_rcp_approx.cuh"
#include "../primitives/30_mbarrier_arrive.cuh"
#include "../primitives/37_bar_sync.cuh"
#include "../primitives/67_mapa.cuh"
#include "../primitives/33_mbarrier_try_wait.cuh"
#include "../primitives/44_elect_sync.cuh"
#include "../primitives/46_setmaxnreg.cuh"
#include "../primitives/70_smem_ptr.cuh"
#include "../primitives/_warp_prof_noop.cuh"
#include "../composites/118_mbarrier_phase_tracking.cuh"
#include "../composites/106_clc_fetch_next_tile.cuh"
#include "../composites/110_fmha_workitem_decode.cuh"
#include "../composites/112_fmha_softmax_utils.cuh"

// One FMHA work-item's correction for this warp's 32-row band: block-0 alpha
// consume, per-block O rescale (K_TILES-1 blocks), then the O*=1/l epilogue pack
// to sO. Trackers persist across work-items (by reference).
template <int M_TILE, int M_TILES_PER_CTA, int HEAD_DIM, int K_TILE,
          int SUB_COLS_BF16, bool FULL_NAMED_BAR, bool SOFTMAX_THROTTLE, bool USE_2CTA = false>
__device__ inline
void correction_warp_blackwell_1tile_1sm2sm_bf16_fmha(WpCtx& wpc,
    uint32_t tmem_base, int corr_warp_id, int lane, int K_TILES,
    float* alpha_and_l_smem, __nv_bfloat16* const* sO_bufs,
    uint64_t* full_bar_alpha, uint64_t* full_bar_l, uint64_t* full_bar_o_acc,
    uint64_t* full_bar_o_epi, uint64_t* empty_bar_spo, uint64_t* empty_bar_alpha_and_l,
    uint64_t* empty_bar_o_epi,
    PhaseTracker<1>& alpha_ph, PhaseTracker<1>& o_acc_ph, PhaseTracker<1>& o_epi_empty_ph) {
  // block 0: no rescale (no prior O); consume alpha + release the scale slot.
  #pragma unroll
  for (int i = 0; i < M_TILES_PER_CTA; ++i) {
    wp_begin(wpc, WP_CORR_WAIT);
    if constexpr (FULL_NAMED_BAR) full_bar_wait(i, corr_warp_id);
    else mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_alpha[i]), alpha_ph.get_phase());
    mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
    wp_end(wpc, WP_CORR_WAIT);
  }
  if constexpr (!FULL_NAMED_BAR) alpha_ph.advance();

  for (int k = 1; k < K_TILES; ++k) {
    #pragma unroll
    for (int i = 0; i < M_TILES_PER_CTA; ++i) {
      wp_begin(wpc, WP_CORR_WAIT);
      if constexpr (FULL_NAMED_BAR) full_bar_wait(i, corr_warp_id);
      else mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_alpha[i]), alpha_ph.get_phase());
      wp_end(wpc, WP_CORR_WAIT);

      wp_begin(wpc, WP_CORR_READ_ALPHA);
      float alpha = alpha_and_l_smem[i * M_TILE + corr_warp_id * 32 + lane];
      if constexpr (!SOFTMAX_THROTTLE) mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
      wp_end(wpc, WP_CORR_READ_ALPHA);

      wp_begin(wpc, WP_CORR_O_SCALE);
      bool skip = __all_sync(0xffffffffu, alpha == 1.0f);
      if (!skip) {
        // O(g-1) is done: BMM1(g) trails BMM2(g-1) in the in-order tcgen05 pipe.
        const uint32_t o_tmem_addr = tmem_base + (uint32_t)(2 * K_TILE + i * HEAD_DIM) + ((uint32_t)(corr_warp_id * 32) << 16);
        // x16 chunks (16 regs live): a 64-reg chunk overflows the 80-reg corr budget (spills).
        const float2 alpha2 = f32x2_splat(alpha);
        // per-chunk LDTM -> FMUL2 -> STTM, no per-chunk waits; one trailing wait::st drains.
        #pragma unroll
        for (int c0 = 0; c0 < HEAD_DIM; c0 += 16) {
          uint32_t o_regs[16];
          tcgen05_ld_32x32b_x16(o_tmem_addr + (uint32_t)c0, o_regs);
          float2* o2 = reinterpret_cast<float2*>(o_regs);
          #pragma unroll
          for (int e = 0; e < 8; ++e) o2[e] = fmul2(o2[e], alpha2);
          tcgen05_st_32x32b_x16(o_tmem_addr + (uint32_t)c0, o_regs);
        }
        tcgen05_wait_st();
        // publish rescaled O for BMM2
        tcgen05_fence_before_thread_sync();
      }
      if constexpr (SOFTMAX_THROTTLE) mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
      if constexpr (USE_2CTA) mbarrier_arrive_cluster_default(mapa_shared_cluster_u32(smem_ptr_u32(&empty_bar_spo[i]), 0));
      else mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[i]));
      wp_end(wpc, WP_CORR_O_SCALE);
    }
    if constexpr (!FULL_NAMED_BAR) alpha_ph.advance();
  }

  // epilogue: O *= 1/l -> bf16 -> sO[i] -> signal W_EPI (full_bar_o_epi).
  #pragma unroll
  for (int i = 0; i < M_TILES_PER_CTA; ++i) {
    wp_begin(wpc, WP_CORR_WAIT);
    mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_o_acc[i]), o_acc_ph.get_phase());
    if constexpr (FULL_NAMED_BAR) full_bar_wait(i, corr_warp_id);
    else mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_l[i]), o_acc_ph.get_phase());
    wp_end(wpc, WP_CORR_WAIT);

    wp_begin(wpc, WP_CORR_EPI);
    const int corr_tid = corr_warp_id * 32 + lane;
    float l = alpha_and_l_smem[i * M_TILE + corr_tid];
    if constexpr (!SOFTMAX_THROTTLE) mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
    float inv_l = (l > 0.f) ? rcp_approx_ftz_f32(l) : 0.f;
    const float2 inv_l2 = f32x2_splat(inv_l);
    const uint32_t o_tmem_addr = tmem_base + (uint32_t)(2 * K_TILE + i * HEAD_DIM) + ((uint32_t)(corr_warp_id * 32) << 16);
    #pragma unroll
    for (int c0 = 0; c0 < HEAD_DIM; c0 += 16) {
      uint32_t o_regs[16];
      tcgen05_ld_32x32b_x16(o_tmem_addr + (uint32_t)c0, o_regs);
      if (c0 == 0) mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_o_epi[i]), o_epi_empty_ph.get_phase());
      float2* o2 = reinterpret_cast<float2*>(o_regs);
      const int s = c0 / SUB_COLS_BF16;
      const int v_base = (c0 % SUB_COLS_BF16) / 8;
      __nv_bfloat16* so_sub = sO_bufs[i] + s * (M_TILE * SUB_COLS_BF16);
      #pragma unroll
      for (int vv = 0; vv < 2; ++vv) {
        const int v = v_base + vv;
        const float2 r0 = fmul2(o2[vv * 4 + 0], inv_l2);
        const float2 r1 = fmul2(o2[vv * 4 + 1], inv_l2);
        const float2 r2 = fmul2(o2[vv * 4 + 2], inv_l2);
        const float2 r3 = fmul2(o2[vv * 4 + 3], inv_l2);
        uint4 packed;
        packed.x = cvt_f32x2_to_bf16x2(r0.x, r0.y);
        packed.y = cvt_f32x2_to_bf16x2(r1.x, r1.y);
        packed.z = cvt_f32x2_to_bf16x2(r2.x, r2.y);
        packed.w = cvt_f32x2_to_bf16x2(r3.x, r3.y);
        *reinterpret_cast<uint4*>(&so_sub[corr_tid * SUB_COLS_BF16 + (v ^ (corr_tid & 7)) * 8]) = packed;
      }
    }
    // O ld done -> MMA may reuse the O slot
    tcgen05_fence_before_thread_sync();

    if constexpr (SOFTMAX_THROTTLE) mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
    if constexpr (USE_2CTA) mbarrier_arrive_cluster_default(mapa_shared_cluster_u32(smem_ptr_u32(&empty_bar_spo[i]), 0));
    else mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[i]));

    // order this thread's st.shared writes (generic proxy) before TMA store (async proxy)
    fence_proxy_async_shared();

    mbarrier_arrive(smem_ptr_u32(&full_bar_o_epi[i]));
    wp_end(wpc, WP_CORR_EPI);
  }
  o_acc_ph.advance();
  o_epi_empty_ph.advance();
}

// Production __device__ body for the correction warp: prime the return barriers
// once, then a persistent loop of decode_workitem (#110, K_TILES only) +
// correction_warp_blackwell_1tile_1sm2sm_bf16_fmha, driven by CLC (#106) or grid-stride.
// CLUSTER_N == 2 selects the 2SM (cta_group::2) path: decode pairs adjacent packed-M
// tiles across the 2 peers, the 1tile body routes cross-CTA (USE_2CTA), and the
// empty_bar_spo prime is routed to peer 0 (the dual-producer barrier's sole consumer).
template <int M_TILE, int M_TILES_PER_CTA, int HEAD_DIM, int K_TILE,
          int SUB_COLS_BF16, bool FULL_NAMED_BAR, bool SOFTMAX_THROTTLE,
          bool USE_CLC, int CLC_STAGES, bool Q_RASTER, bool IS_CAUSAL, bool LPT,
          int CORR_REG_BUDGET = 80, bool TAIL_DRAIN = true, int CLUSTER_N = 1>
__device__ inline
void correction_warp_blackwell_ntiles_1sm2sm_bf16_fmha(WpCtx& wpc,
    uint32_t* tmem_slot, int corr_warp_id, int lane,
    float* alpha_and_l_smem, __nv_bfloat16* const* sO_bufs,
    uint64_t* full_bar_alpha, uint64_t* full_bar_l, uint64_t* full_bar_o_acc,
    uint64_t* full_bar_o_epi, uint64_t* empty_bar_spo, uint64_t* empty_bar_alpha_and_l,
    uint64_t* empty_bar_o_epi,
    uint64_t* clc_full, uint64_t* clc_empty, uint32_t* clc_response,
    int seqlen_kv, int num_q_heads, int num_kv_heads, int packed_idx_per_seq, int num_samples,
    unsigned long long magic0, unsigned long long magic1, unsigned long long magic2) {
  setmaxnreg_dec<CORR_REG_BUDGET>();
  const int peer = (int)(blockIdx.x % CLUSTER_N);

  const int q_tile_per_cta = M_TILES_PER_CTA * (M_TILE / (num_q_heads / num_kv_heads));
  const int total_tiles    = num_samples * packed_idx_per_seq * num_kv_heads;

  bar_sync<9>(416);
  const uint32_t tmem_base = *tmem_slot;

  [[maybe_unused]] PhaseTracker<1> alpha_ph;
  PhaseTracker<1> o_acc_ph;
  PhaseTracker<1> o_epi_empty_ph;
  [[maybe_unused]] int clc_stage = 0;
  [[maybe_unused]] uint32_t clc_phase = 0;

  // Prime the return barriers once (first BMM2 / first softmax stat write).
  #pragma unroll
  for (int i = 0; i < M_TILES_PER_CTA; ++i) {
    if constexpr (CLUSTER_N == 2)
      mbarrier_arrive_cluster_default(mapa_shared_cluster_u32(smem_ptr_u32(&empty_bar_spo[i]), 0));
    else
      mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[i]));
    mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
  }

  int tile_id = (int)(blockIdx.x / CLUSTER_N);
  while (true) {
    int sample, h_kv, q_tile_base, K_TILES;
    decode_workitem<K_TILE, Q_RASTER, IS_CAUSAL, LPT, CLUSTER_N>(tile_id, seqlen_kv, num_kv_heads,
        packed_idx_per_seq, q_tile_per_cta, magic0, magic1, magic2,
        sample, h_kv, q_tile_base, K_TILES, peer);

    correction_warp_blackwell_1tile_1sm2sm_bf16_fmha<
        M_TILE, M_TILES_PER_CTA, HEAD_DIM, K_TILE, SUB_COLS_BF16, FULL_NAMED_BAR, SOFTMAX_THROTTLE,
        /*USE_2CTA=*/(CLUSTER_N == 2)>(wpc, tmem_base, corr_warp_id, lane, K_TILES, alpha_and_l_smem, sO_bufs,
        full_bar_alpha, full_bar_l, full_bar_o_acc, full_bar_o_epi,
        empty_bar_spo, empty_bar_alpha_and_l, empty_bar_o_epi,
        alpha_ph, o_acc_ph, o_epi_empty_ph);

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
  bar_sync<9>(416);

  if constexpr (TAIL_DRAIN) {
    // Tail drain: corr->epi (empty_bar_o_epi).
    wp_begin(wpc, WP_CORR_WAIT);
    #pragma unroll
    for (int i = 0; i < M_TILES_PER_CTA; ++i) {
      mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_o_epi[i]),
                           o_epi_empty_ph.get_phase());
    }
    wp_end(wpc, WP_CORR_WAIT);
  }
}

// varlen O epilogue PASS 2: re-tile the swizzled sO -> gmem O_out, 16 threads/row (8 bf16 each),
// per-token `tok < seqlen_q` predicate (ragged tail). `tid` in [0,128): the 128-thread corr store
// passes corr_tid; the 32-thread epi store loops g=0..3 with tid = g*32 + lane.
template <bool MHA, int M_TILE, int HEAD_DIM>
__device__ __forceinline__ void varlen_o_store_pass2(
    int tid, const __nv_bfloat16* sO, __nv_bfloat16* O_out,
    int q_start, int q_tile_base, int m_tile, int h_kv,
    int num_q_heads, int gqa_group_size, int q_tile_per_mtile, int seqlen_q) {
  const int qhi = tid >> 4;
  const int c   = tid & 15;
  const int tok0 = q_tile_base + m_tile * q_tile_per_mtile;
  const long gstride = (long)num_q_heads * HEAD_DIM;
  const uint4* srow = reinterpret_cast<const uint4*>(&sO[qhi * HEAD_DIM + (c ^ qhi) * 8]);
  if constexpr (MHA) {
    long ob = (long)(q_start + tok0 + qhi) * gstride + (long)h_kv * HEAD_DIM + (long)(c * 8);
    #pragma unroll
    for (int iter = 0; iter < M_TILE / 8; ++iter) {
      if (tok0 + qhi + 8 * iter < seqlen_q)
        *reinterpret_cast<uint4*>(&O_out[ob]) = *srow;
      ob   += 8 * gstride;    // token += 8
      srow += HEAD_DIM;
    }
  } else {
    const int q_head = h_kv * gqa_group_size + qhi;
    long ob = (long)(q_start + tok0) * gstride + (long)q_head * HEAD_DIM + (long)(c * 8);
    #pragma unroll
    for (int iter = 0; iter < M_TILE / 8; ++iter) {
      if (tok0 + iter < seqlen_q)
        *reinterpret_cast<uint4*>(&O_out[ob]) = *srow;
      ob   += gstride;        // token += 1
      srow += HEAD_DIM;
    }
  }
}

/* ============================================================================
 * correction_warp_blackwell_ntiles_1sm_varlen_bf16_fmha<>(...)
 *
 * Correction warp for the varlen 1SM FMHA context kernel (fmha_context_bf16_varlen.cu).
 * Kept SEPARATE from the uniform corr block: the epilogue is a PREDICATED STG re-tile
 * store done ON the corr warps (no sO_bufs, no full_bar_o_epi, no TMA epi warp) --
 *   PASS 1: TMEM O *= 1/l -> bf16 -> contiguous swizzled sO (chunk ^ (row&7));
 *   PASS 2: sO -> gmem O_out, 16 threads/row, per-token `tok < seqlen_q` predicate,
 *           MHA vs GQA addressing. Two bar_sync<10>(128) fence the 4 corr warps' sO.
 * The per-K O-rescale skeleton (alpha consume, __all_sync skip, LDTM->FMUL2->STTM) and
 * the O*=1/l math are the same as uniform; only the store path diverges. Warp-0-owned
 * TMEM (tmem_base passed in). Short-sample `if (q_tile_base < seqlen_q)` guard.
 * ============================================================================ */
template <int M_TILE, int M_TILES_PER_CTA, int HEAD_DIM, int K_TILE, bool FULL_NAMED_BAR,
          bool USE_CLC, int CLC_STAGES, bool Q_RASTER, bool IS_CAUSAL, bool LPT, bool MHA,
          int CORR_REG_BUDGET = 72>
__device__ inline
void correction_warp_blackwell_ntiles_1sm_varlen_bf16_fmha(
    uint32_t* tmem_slot, int corr_warp_id, int lane,
    float* alpha_and_l_smem, __nv_bfloat16* sO, __nv_bfloat16* O_out,
    uint64_t* full_bar_alpha, uint64_t* full_bar_l, uint64_t* full_bar_o_acc,
    uint64_t* empty_bar_spo, uint64_t* empty_bar_alpha_and_l,
    uint64_t* clc_full, uint64_t* clc_empty, uint32_t* clc_response,
    int num_q_heads, int num_kv_heads, int packed_mtiles_per_seq, int num_samples,
    const int* cu_seqlens_q, const int* seqlens_kv) {
  setmaxnreg_dec<CORR_REG_BUDGET>();
  bar_sync<9>(416);   // wait the MMA warp's tcgen05.alloc<1> publish (named bar 9)
  const uint32_t tmem_base = *tmem_slot;
  constexpr int S_COLS = K_TILE;
  constexpr int O_COLS = HEAD_DIM;

  const int gqa_group_size   = num_q_heads / num_kv_heads;
  const int q_tile_per_mtile = M_TILE / gqa_group_size;
  const int q_tile_per_cta   = M_TILES_PER_CTA * q_tile_per_mtile;
  const int packed_mtiles_per_sample = packed_mtiles_per_seq * num_kv_heads;
  const int total_tiles = num_samples * packed_mtiles_per_sample;

  [[maybe_unused]] PhaseTracker<1> alpha_ph;
  PhaseTracker<1> o_acc_ph;
  // Prime the return barriers once (first BMM2 / first softmax stat write).
  #pragma unroll
  for (int i = 0; i < M_TILES_PER_CTA; ++i) {
    mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[i]));
    mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
  }

  [[maybe_unused]] int clc_stage = 0;
  [[maybe_unused]] uint32_t clc_phase = 0;
  int tile_id = (int)blockIdx.x;
  while (true) {
    int sample, h_kv, q_tile_base, seqlen_q, K_TILES;
    decode_workitem_varlen<K_TILE, Q_RASTER, IS_CAUSAL, LPT>(tile_id, num_kv_heads,
        packed_mtiles_per_seq, packed_mtiles_per_sample, q_tile_per_cta,
        cu_seqlens_q, seqlens_kv, sample, h_kv, q_tile_base, seqlen_q, K_TILES);
    const int q_start = cu_seqlens_q[sample];
    if (q_tile_base < seqlen_q) {
    // block 0: no rescale (no prior O); consume alpha + release the scale slot.
    #pragma unroll
    for (int i = 0; i < M_TILES_PER_CTA; ++i) {
      if constexpr (FULL_NAMED_BAR) full_bar_wait(i, corr_warp_id);
      else mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_alpha[i]), alpha_ph.get_phase());
      mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
    }
    if constexpr (!FULL_NAMED_BAR) alpha_ph.advance();

    for (int k = 1; k < K_TILES; ++k) {
      #pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i) {
        if constexpr (FULL_NAMED_BAR) full_bar_wait(i, corr_warp_id);
        else mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_alpha[i]), alpha_ph.get_phase());

        float alpha = alpha_and_l_smem[i * M_TILE + corr_warp_id * 32 + lane];
        mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));

        bool skip = __all_sync(0xffffffffu, alpha == 1.0f);
        if (!skip) {
          // O(g-1) is done: BMM1(g) trails BMM2(g-1) in the in-order tcgen05 pipe.
          tcgen05_fence_after_thread_sync();
          const uint32_t o_tmem_addr = tmem_base + (uint32_t)(2 * S_COLS + i * O_COLS) + ((uint32_t)(corr_warp_id * 32) << 16);
          // SERIAL ld -> wait_ld -> mul -> st -> wait_st: inline-asm tcgen05.ld/st are opaque to ptxas -- no overlap.
          const float2 alpha2 = f32x2_splat(alpha);
          #pragma unroll
          for (int c0 = 0; c0 < HEAD_DIM; c0 += 64) {
            uint32_t o_regs[64];
            tcgen05_ld_32x32b_x64(o_tmem_addr + (uint32_t)c0, *reinterpret_cast<uint32_t(*)[64]>(o_regs));
            tcgen05_wait_ld();
            float2* o2 = reinterpret_cast<float2*>(o_regs);
            #pragma unroll
            for (int e = 0; e < 32; ++e) o2[e] = fmul2(o2[e], alpha2);
            tcgen05_st_32x32b_x64(o_tmem_addr + (uint32_t)c0, *reinterpret_cast<uint32_t(*)[64]>(o_regs));
            tcgen05_wait_st();
          }
          // publish rescaled O for BMM2
          tcgen05_fence_before_thread_sync();
        }
        mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[i]));
      }
      if constexpr (!FULL_NAMED_BAR) alpha_ph.advance();
    }

    // epilogue: O *= 1/l, then store to gmem.
    #pragma unroll
    for (int i = 0; i < M_TILES_PER_CTA; ++i) {
      mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_o_acc[i]), o_acc_ph.get_phase());
      if constexpr (FULL_NAMED_BAR) full_bar_wait(i, corr_warp_id);
      else mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_l[i]), o_acc_ph.get_phase());

      const int corr_tid = corr_warp_id * 32 + lane;
      float l = alpha_and_l_smem[i * M_TILE + corr_tid];
      mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
      float inv_l = (l > 0.f) ? rcp_approx_ftz_f32(l) : 0.f;
      // PASS 1: read this thread's O row from TMEM, O *= 1/l, FP32 -> bf16 -> sO[row].
      tcgen05_fence_after_thread_sync();           // order MMA's final BMM2 O before our ld
      const uint32_t o_tmem_addr = tmem_base + (uint32_t)(2 * S_COLS + i * O_COLS) + ((uint32_t)(corr_warp_id * 32) << 16);
      // x16 chunks (16 regs live) ease spill; sO chunk c swizzled to c ^ (row&7) (same map in PASS 2).
      const float2 inv_l2 = f32x2_splat(inv_l);
      #pragma unroll
      for (int c0 = 0; c0 < HEAD_DIM; c0 += 16) {
        uint32_t o_regs[16];
        tcgen05_ld_32x32b_x16(o_tmem_addr + (uint32_t)c0, *reinterpret_cast<uint32_t(*)[16]>(o_regs));
        tcgen05_wait_ld();
        float2* o2 = reinterpret_cast<float2*>(o_regs);
        #pragma unroll
        for (int v = 0; v < 2; ++v) {
          const int chunk = c0 / 8 + v;
          const float2 s0 = fmul2(o2[v * 4 + 0], inv_l2);
          const float2 s1 = fmul2(o2[v * 4 + 1], inv_l2);
          const float2 s2 = fmul2(o2[v * 4 + 2], inv_l2);
          const float2 s3 = fmul2(o2[v * 4 + 3], inv_l2);
          uint4 packed;
          packed.x = cvt_f32x2_to_bf16x2(s0.x, s0.y);
          packed.y = cvt_f32x2_to_bf16x2(s1.x, s1.y);
          packed.z = cvt_f32x2_to_bf16x2(s2.x, s2.y);
          packed.w = cvt_f32x2_to_bf16x2(s3.x, s3.y);
          *reinterpret_cast<uint4*>(&sO[corr_tid * HEAD_DIM + (chunk ^ (corr_tid & 7)) * 8]) = packed;
        }
      }
      // O ld done -> MMA may reuse the O slot
      tcgen05_fence_before_thread_sync();

      mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[i]));
      bar_sync<10>(128);   // pass1 done: all 4 correction warps' sO rows visible
      varlen_o_store_pass2<MHA, M_TILE, HEAD_DIM>(corr_tid, sO, O_out,
          q_start, q_tile_base, i, h_kv, num_q_heads, gqa_group_size, q_tile_per_mtile, seqlen_q);
      bar_sync<10>(128);   // pass2 done: sO free for the next M-tile
    }
    o_acc_ph.advance();
    }  // guarded body
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
  bar_sync<9>(416);   // pair the MMA warp's dealloc handshake
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
