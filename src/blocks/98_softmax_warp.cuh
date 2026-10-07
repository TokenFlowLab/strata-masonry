#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// 98_softmax_warp.cuh -- FMHA online softmax warp building block.
//
// ARCH: sm_100a
//
// The "softmax warp" role for FlashAttention-style FMHA. Walks a row of
// score values 32 lanes at a time, maintaining the running max + sum-of-
// exp2 state across K-tiles using #85 SoftmaxState (composite). Per tile:
//   - load 32 scores per lane (or -INFINITY for tail lanes)
//   - redux_sync_max_f32 for tile max
//   - lane-local exp2(x - tile_max) + butterfly reduction for tile sum
//   - softmax_state_update merges (tile_max, tile_sum) into the running
//     state, returning the rescale factor that the correction warp would
//     apply to the running PV accumulator
//
// PTX:    9.7.18.8 (TMEM ld/st), 9.7.10.24 (cvt), 9.7.15.13 (redux.sync)
//
// Block function:
//
//   __device__ __forceinline__ void
//   softmax_warp_block(const float* scores, int K, SoftmaxState& state);
//
// - scores: GMEM pointer to a contiguous row of K scores.
// - K: number of scores in the row.
// - state: caller-allocated, caller-initialized SoftmaxState; updated
//   in-place across all K-tiles. Caller reads `state.m` / `state.l`
//   after the block returns.
//
// Caller-gated. Warp-collective: invoke from all 32 lanes of the
// softmax warp.

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include "../primitives/49_redux_sync_f32.cuh"
#include "../composites/85_online_softmax.cuh"

__device__ __forceinline__
void softmax_warp_block(const float* __restrict__ scores, int K,
                        SoftmaxState& state) {
  for (int k = 0; k < K; k += 32) {
    float x = (k + threadIdx.x < K) ? scores[k + threadIdx.x] : -INFINITY;
    float m = redux_sync_max_f32(x);
    float se = exp2f(x - m);
    for (int off = 16; off > 0; off >>= 1)
      se += __shfl_xor_sync(0xFFFFFFFFu, se, off);
    softmax_state_update(state, m, se);
  }
}

// ============================================================================
// Production FMHA softmax warp (kernels/fmha/sm100a/fmha_context_bf16_varlen.cu).
// One thread owns a whole K_TILE-key S row (lane-local rmax + rowsum, no cross-
// lane reduction). Reads S from TMEM, masks the ragged/causal last tile, keeps a
// running (m, l) online-softmax state, publishes alpha per K block via
// alpha_and_l_smem + full_bar_alpha, then fuses ffma2(scale)+exp2+rowsum+bf16
// pack and tcgen05.st's P back into the S TMEM slot (split-P: two chunks).
// ============================================================================
#include "../primitives/9_tcgen05_ld.cuh"
#include "../primitives/10_tcgen05_st.cuh"
#include "../primitives/12_tcgen05_wait.cuh"
#include "../primitives/15_tcgen05_fence.cuh"
#include "../primitives/30_mbarrier_arrive.cuh"
#include "../primitives/67_mapa.cuh"
#include "../primitives/33_mbarrier_try_wait.cuh"
#include "../primitives/44_elect_sync.cuh"
#include "../primitives/46_setmaxnreg.cuh"
#include "../primitives/63_cvt_f32_to_f16_bf16.cuh"
#include "../primitives/76_packed_f32x2.cuh"
#include "../primitives/77_ex2_approx.cuh"
#include "../primitives/70_smem_ptr.cuh"
#include "../primitives/_warp_prof_noop.cuh"
#include "../composites/118_mbarrier_phase_tracking.cuh"
#include "../composites/106_clc_fetch_next_tile.cuh"
#include "../composites/110_fmha_workitem_decode.cuh"
#include "../composites/112_fmha_softmax_utils.cuh"

// One FMHA work-item's softmax for this warp's 32-row band: wait the scale slot,
// run the online softmax over K_TILES KV blocks (descending), then publish the
// final row-sum (full_bar_l). Trackers persist across work-items (by reference).
template <int K_TILE, int M_TILE, int S_LD_COLS, bool FULL_NAMED_BAR, bool SPLIT_P,
          bool EX2_EMU, bool IS_CAUSAL, bool USE_2CTA = false>
__device__ inline
void softmax_warp_blackwell_1tile_1sm2sm_bf16_fmha(WpCtx& wpc,
    uint32_t s_tmem_addr, int m_tile, int warp_in_group, int row_in_m_tile, int lane,
    float scale_log2, int gqa_group_size, int q_tile_per_mtile, int q_tile_base,
    int seqlen_kv, int K_TILES, float* alpha_and_l_smem,
    uint64_t* full_bar_spo, uint64_t* full_bar_alpha, uint64_t* full_bar_l,
    uint64_t* empty_bar_spo, uint64_t* empty_bar_alpha_and_l, uint64_t* full_bar_p_last,
    PhaseTracker<1>& spo_ph, PhaseTracker<1>& scale_empty_ph) {
  constexpr int SPLIT_P_COL   = (K_TILE / 4 * 3) / 2;
  constexpr int EX2_FRG_PAIRS = 16;          // 32 elts / fragment = 16 pairs
  constexpr int EX2_FRG_CNT   = K_TILE / 32;
  constexpr int EX2_FREQ      = 16;          // FA4 ex2_emu_freq
  constexpr int EX2_RES       = 4;           // FA4 ex2_emu_res

  // causal: this row's query-token position (pack-GQA qh-inner); unused when IS_CAUSAL=false.
  const int q_pos = q_tile_base + m_tile * q_tile_per_mtile + row_in_m_tile / gqa_group_size;
  float m_run = -INFINITY, l_run = 0.f;
  wp_begin(wpc, WP_SM_WAIT_SCALE);
  mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_alpha_and_l[m_tile]), scale_empty_ph.get_phase());
  scale_empty_ph.advance();
  wp_end(wpc, WP_SM_WAIT_SCALE);
  for (int k = 0; k < K_TILES; ++k) {
    const int k_offset = (K_TILES - 1 - k) * K_TILE;
    wp_begin(wpc, WP_SM_WAIT_S);
    mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_spo[m_tile]), spo_ph.get_phase());
    wp_end(wpc, WP_SM_WAIT_S);

    wp_begin(wpc, WP_SM_SOFTMAX);
    uint32_t s_regs[K_TILE];
    #pragma unroll
    for (int c0 = 0; c0 < K_TILE; c0 += S_LD_COLS) {
      const uint32_t taddr = s_tmem_addr + (uint32_t)c0;
      if      constexpr (S_LD_COLS == 32)  tcgen05_ld_32x32b_x32 (taddr, *reinterpret_cast<uint32_t(*)[32]>(&s_regs[c0]));
      else if constexpr (S_LD_COLS == 64)  tcgen05_ld_32x32b_x64 (taddr, *reinterpret_cast<uint32_t(*)[64]>(&s_regs[c0]));
      else if constexpr (S_LD_COLS == 128) tcgen05_ld_32x32b_x128(taddr, *reinterpret_cast<uint32_t(*)[128]>(&s_regs[c0]));
    }
    // no wait_ld: rmax/exp2 below scoreboard-wait the S LDTM (RAW).

    float* scores = reinterpret_cast<float*>(s_regs);
    float2* scores2 = reinterpret_cast<float2*>(s_regs);
    // order this S read ahead of the later tcgen05.st that overwrites the slot with P
    tcgen05_fence_before_thread_sync();

    // A work-item can span multiple causal diagonal tiles (e.g. 256 Q tokens in MHA).
    // Mask until the entire K tile precedes this M-tile's first query token.
    const int first_q = q_tile_base + m_tile * q_tile_per_mtile;
    if (k == 0 || (IS_CAUSAL && k_offset + K_TILE > first_q))
      mask_s_row_r2p<IS_CAUSAL, K_TILE>(scores, k_offset, q_pos, seqlen_kv);

    // rmax via 4 independent FMNMX3 accumulators (4-way ILP). K_TILE % 8 == 0.
    float rmax0 = -INFINITY, rmax1 = -INFINITY, rmax2 = -INFINITY, rmax3 = -INFINITY;
    #pragma unroll
    for (int j = 0; j < K_TILE; j += 8) {
      rmax0 = fmaxf(fmaxf(rmax0, scores[j + 0]), scores[j + 1]);
      rmax1 = fmaxf(fmaxf(rmax1, scores[j + 2]), scores[j + 3]);
      rmax2 = fmaxf(fmaxf(rmax2, scores[j + 4]), scores[j + 5]);
      rmax3 = fmaxf(fmaxf(rmax3, scores[j + 6]), scores[j + 7]);
    }
    float rmax = fmaxf(fmaxf(rmax0, rmax1), fmaxf(rmax2, rmax3));
    float new_m = fmaxf(m_run, rmax);
    // Descending traversal can start with fully masked rows. Keep m_run=-inf, but
    // use zero in the exponent so P=0 rather than exp2(-inf - -inf)=NaN.
    const float row_max_safe = new_m == -INFINITY ? 0.0f : new_m;
    float alpha = 0.0f;
    if (k != 0) alpha = ex2_approx_f32((m_run - row_max_safe) * scale_log2);
    alpha_and_l_smem[m_tile * M_TILE + row_in_m_tile] = alpha;
    if constexpr (FULL_NAMED_BAR) full_bar_arrive(m_tile, warp_in_group);
    else mbarrier_arrive(smem_ptr_u32(&full_bar_alpha[m_tile]));

    // fused ffma2(scale) + exp2 + row-sum + bf16 pack: 128 keys -> 64 u32 P cols.
    const float2 scale2 = f32x2_splat(scale_log2);
    const float2 neg_m_scaled2 = f32x2_splat(-row_max_safe * scale_log2);
    float2 lt2 = make_float2(0.f, 0.f);
    uint32_t p_regs[K_TILE / 2];
    #pragma unroll
    for (int c = 0; c < K_TILE / 2; ++c) {
      const float2 a2 = ffma2(scores2[c], scale2, neg_m_scaled2);
      float2 e2;
      if constexpr (EX2_EMU) {
        const int jj = c / EX2_FRG_PAIRS;
        const int kk = 2 * (c % EX2_FRG_PAIRS);
        const bool use_hw = (kk % EX2_FREQ < EX2_FREQ - EX2_RES) || (jj >= EX2_FRG_CNT - 1);
        e2 = use_hw ? make_float2(ex2_approx_f32(a2.x), ex2_approx_f32(a2.y))
                    : ex2_emu_f32x2(a2.x, a2.y);
      } else {
        e2 = make_float2(ex2_approx_f32(a2.x), ex2_approx_f32(a2.y));
      }
      lt2 = fadd2(lt2, e2);
      p_regs[c] = cvt_f32x2_to_bf16x2(e2.x, e2.y);
    }
    uint32_t p_tmem_addr = s_tmem_addr;
    wp_end(wpc, WP_SM_SOFTMAX);

    wp_begin(wpc, WP_SM_STORE_P);
    if constexpr (SPLIT_P) {
      tcgen05_st_32x32b_x32(p_tmem_addr, *reinterpret_cast<uint32_t(*)[32]>(&p_regs[0]));
      tcgen05_st_32x32b_x16(p_tmem_addr + 32, *reinterpret_cast<uint32_t(*)[16]>(&p_regs[32]));
      tcgen05_wait_st();   // RACE FIX f22fb2e: fence orders but does NOT complete the async STTM
      tcgen05_fence_before_thread_sync();
      if constexpr (USE_2CTA) mbarrier_arrive_cluster_default(mapa_shared_cluster_u32(smem_ptr_u32(&empty_bar_spo[m_tile]), 0));
      else mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[m_tile]));
      tcgen05_st_32x32b_x16(p_tmem_addr + SPLIT_P_COL, *reinterpret_cast<uint32_t(*)[16]>(&p_regs[SPLIT_P_COL]));
      tcgen05_wait_st();   // RACE FIX f22fb2e
      tcgen05_fence_before_thread_sync();
      if constexpr (USE_2CTA) mbarrier_arrive_cluster_default(mapa_shared_cluster_u32(smem_ptr_u32(&full_bar_p_last[m_tile]), 0));
      else mbarrier_arrive(smem_ptr_u32(&full_bar_p_last[m_tile]));
    } else {
      tcgen05_st_32x32b_x32(p_tmem_addr, *reinterpret_cast<uint32_t(*)[32]>(&p_regs[0]));
      tcgen05_st_32x32b_x32(p_tmem_addr + 32, *reinterpret_cast<uint32_t(*)[32]>(&p_regs[32]));
      tcgen05_wait_st();   // RACE FIX f22fb2e
      tcgen05_fence_before_thread_sync();
      if constexpr (USE_2CTA) mbarrier_arrive_cluster_default(mapa_shared_cluster_u32(smem_ptr_u32(&empty_bar_spo[m_tile]), 0));
      else mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[m_tile]));
    }
    wp_end(wpc, WP_SM_STORE_P);

    spo_ph.advance();
    float lt = lt2.x + lt2.y;
    l_run = alpha * l_run + lt; m_run = new_m;
    wp_begin(wpc, WP_SM_WAIT_SCALE);
    mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_alpha_and_l[m_tile]), scale_empty_ph.get_phase());
    scale_empty_ph.advance();
    wp_end(wpc, WP_SM_WAIT_SCALE);
  }

  wp_begin(wpc, WP_SM_READ_L);
  alpha_and_l_smem[m_tile * M_TILE + row_in_m_tile] = l_run;
  if constexpr (FULL_NAMED_BAR) full_bar_arrive(m_tile, warp_in_group);
  else mbarrier_arrive(smem_ptr_u32(&full_bar_l[m_tile]));
  wp_end(wpc, WP_SM_READ_L);
}

// Production __device__ body for the softmax warp: derive this warp's 32-row band,
// then a persistent loop of decode_workitem (#110) +
// softmax_warp_blackwell_1tile_1sm2sm_bf16_fmha, driven by CLC (#106) or grid-stride.
// VARLEN=true selects the varlen decode (prefix-sum cu_seqlens_q/seqlens_kv, per-sample seqlen_q,
// ragged K_TILES) + the short-sample `q_tile_base < seqlen_q` guard, and masks with seqlens_kv[sample]
// instead of the scalar `seqlen`. 1SM only (CLUSTER_N=1, TAIL_DRAIN=false). The 1tile body + loop
// skeleton are shared with uniform/2sm.
template <int K_TILE, int M_TILE, int M_TILES_PER_CTA, int S_LD_COLS, bool FULL_NAMED_BAR, bool SPLIT_P,
          bool EX2_EMU, bool USE_CLC, int CLC_STAGES, bool Q_RASTER, bool IS_CAUSAL, bool LPT,
          int SM_REG_BUDGET = 192, bool TAIL_DRAIN = true, int CLUSTER_N = 1, bool VARLEN = false>
__device__ inline
void softmax_warp_blackwell_ntiles_1sm2sm_bf16_fmha(WpCtx& wpc,
    uint32_t* tmem_slot, int warp_id, int lane, float scale_log2, float* alpha_and_l_smem,
    uint64_t* full_bar_spo, uint64_t* full_bar_alpha, uint64_t* full_bar_l,
    uint64_t* empty_bar_spo, uint64_t* empty_bar_alpha_and_l, uint64_t* full_bar_p_last,
    uint64_t* clc_full, uint64_t* clc_empty, uint32_t* clc_response,
    int seqlen_kv, int num_q_heads, int num_kv_heads, int packed_idx_per_seq, int num_samples,
    unsigned long long magic0, unsigned long long magic1, unsigned long long magic2,
    const int* cu_seqlens_q = nullptr, const int* seqlens_kv = nullptr) {
  setmaxnreg_inc<SM_REG_BUDGET>();
  const int peer = (int)(blockIdx.x % CLUSTER_N);

  const int gqa_group_size    = num_q_heads / num_kv_heads;
  const int q_tile_per_mtile  = M_TILE / gqa_group_size;
  const int q_tile_per_cta    = M_TILES_PER_CTA * q_tile_per_mtile;
  const int total_tiles       = num_samples * packed_idx_per_seq * num_kv_heads;

  bar_sync<9>(416);
  const uint32_t tmem_base = *tmem_slot;
  constexpr int S_COLS = K_TILE;
  constexpr int WARPS_PER_MTILE = M_TILE / 32;
  const int m_tile = warp_id / WARPS_PER_MTILE;
  const int warp_in_group = warp_id % WARPS_PER_MTILE;
  const int row_in_m_tile = warp_in_group * 32 + lane;
  const uint32_t s_tmem_addr = tmem_base + (uint32_t)(m_tile * S_COLS) + ((uint32_t)(warp_in_group * 32) << 16);
  PhaseTracker<1> spo_ph;
  PhaseTracker<1> scale_empty_ph;
  [[maybe_unused]] int clc_stage = 0;
  [[maybe_unused]] uint32_t clc_phase = 0;

  int tile_id = (int)(blockIdx.x / CLUSTER_N);
  while (true) {
    int sample, h_kv, q_tile_base, K_TILES, seqlen_q = 0, seqlen_kv_wi = seqlen_kv;
    if constexpr (VARLEN) {
      decode_workitem_varlen<K_TILE, Q_RASTER, IS_CAUSAL, LPT>(tile_id, num_kv_heads,
          packed_idx_per_seq, packed_idx_per_seq * num_kv_heads, q_tile_per_cta,
          cu_seqlens_q, seqlens_kv, sample, h_kv, q_tile_base, seqlen_q, K_TILES);
      seqlen_kv_wi = seqlens_kv[sample];
    } else {
      decode_workitem<K_TILE, Q_RASTER, IS_CAUSAL, LPT, CLUSTER_N>(tile_id, seqlen_kv, num_kv_heads,
          packed_idx_per_seq, q_tile_per_cta, magic0, magic1, magic2,
          sample, h_kv, q_tile_base, K_TILES, peer);
    }

    if (!VARLEN || q_tile_base < seqlen_q)
    softmax_warp_blackwell_1tile_1sm2sm_bf16_fmha<
        K_TILE, M_TILE, S_LD_COLS, FULL_NAMED_BAR, SPLIT_P, EX2_EMU, IS_CAUSAL, /*USE_2CTA=*/(CLUSTER_N == 2)>(wpc, s_tmem_addr, m_tile, warp_in_group, row_in_m_tile, lane, scale_log2, gqa_group_size,
        q_tile_per_mtile, q_tile_base, seqlen_kv_wi, K_TILES, alpha_and_l_smem,
        full_bar_spo, full_bar_alpha, full_bar_l, empty_bar_spo, empty_bar_alpha_and_l, full_bar_p_last,
        spo_ph, scale_empty_ph);

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
    // Closing prime: empty_bar_spo +128 (mirror of corr's start prime) so MMA can drain it.
    // After bar_sync<9>: MMA's real epilogue waits are done, so only its tail drain consumes it.
    // 2SM: route to peer 0 (the dual-producer barrier's sole consumer is the leader MMA warp).
    if constexpr (CLUSTER_N == 2)
      mbarrier_arrive_cluster_default(mapa_shared_cluster_u32(smem_ptr_u32(&empty_bar_spo[m_tile]), 0));
    else
      mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[m_tile]));

    // Tail drain: softmax->corr (empty_bar_alpha_and_l).
    wp_begin(wpc, WP_SM_WAIT_SCALE);
    mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_alpha_and_l[m_tile]),
                         scale_empty_ph.get_phase());
    wp_end(wpc, WP_SM_WAIT_SCALE);
  }
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
