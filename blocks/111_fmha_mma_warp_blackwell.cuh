#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// 111_fmha_mma_warp_blackwell.cuh -- Blackwell tcgen05.mma driver warp for the
// 1SM BF16 FMHA context kernel (kernels/fmha/sm100a/fmha_context_bf16_uniform.cu).
//
// ARCH: sm_100a
//
// Unlike the dense GEMM mma warp (90_mma_warp_blackwell.cuh, one SS K-loop into
// one accumulator), the FMHA mma warp runs the TWO chained GEMMs of attention as
// a BMM1-ahead software pipeline over M_TILES_PER_CTA q-tiles:
//   BMM1 (Q@K -> S, SS form, accumulator S[i] reset per KV block) and
//   BMM2 (P@V -> O, TS form with a TMEM P operand, O[i] accumulated across blocks),
// interleaved so BMM1 of KV-block k+1 issues right after BMM2 of block k. The
// shared K/V ring alternately delivers K (BMM1) and V (BMM2) tiles; softmax's P
// arrives in TMEM (empty_bar_spo / full_bar_p_last, split-P). TMEM layout:
// S[i] at i*S_COLS, O[i] at 2*S_COLS + i*O_COLS (512 cols for 2 M-tiles at 128/128).
//
// PTX: 9.7.18.4 (tcgen05.mma.ss / .ts), 9.7.18.6 (tcgen05.commit),
//      9.7.15.16.19 (mbarrier.try_wait.parity)

#include <cstdint>
#include <cuda_bf16.h>
#include "../primitives/0_tcgen05_alloc.cuh"
#include "../primitives/1_tcgen05_dealloc.cuh"
#include "../primitives/2_tcgen05_relinquish.cuh"
#include "../primitives/3_tcgen05_mma_f16.cuh"
#include "../primitives/8_tcgen05_mma_idesc.cuh"
#include "../primitives/11_tcgen05_commit.cuh"
#include "../primitives/37_bar_sync.cuh"
#include "../primitives/38_barrier_cluster.cuh"
#include "../primitives/33_mbarrier_try_wait.cuh"
#include "../primitives/42_smem_desc_blackwell.cuh"
#include "../primitives/44_elect_sync.cuh"
#include "../primitives/46_setmaxnreg.cuh"
#include "../primitives/70_smem_ptr.cuh"
#include "../primitives/_warp_prof_noop.cuh"
#include "../composites/118_mbarrier_phase_tracking.cuh"
#include "../composites/106_clc_fetch_next_tile.cuh"
#include "../composites/110_fmha_workitem_decode.cuh"

/* ============================================================================
 * fmha_mma_warp_blackwell_1tile_1sm_bf16<>(wpc, ...)
 *
 * One FMHA work-item's MMA pipeline (K_TILES KV blocks): prologue BMM1 of block
 * 0 for each M-tile, steady-state BMM2(k)+BMM1-ahead(k+1) per M-tile, epilogue
 * BMM2 of the last block -> final O. Only the leader lane issues tcgen05.mma /
 * commit (all 32 lanes drive the mbarrier waits). Phase trackers persist across
 * work-items (passed by reference).
 * ============================================================================ */
template <int NUM_KV_STAGES, int M_TILES_PER_CTA, int M_TILE, int K_TILE, int HEAD_DIM, bool SPLIT_P>
__device__ inline
void fmha_mma_warp_blackwell_1tile_1sm_bf16(WpCtx& wpc,
    uint32_t tmem_base, bool lead,
    uint64_t desc_q_stage0, uint64_t desc_kv_stage0, uint32_t idesc_qk, uint32_t idesc_pv,
    int K_TILES,
    uint64_t* full_bar, uint64_t* empty_bar, uint64_t* full_bar_q, uint64_t* empty_bar_q,
    uint64_t* full_bar_spo, uint64_t* empty_bar_spo, uint64_t* full_bar_p_last,
    uint64_t* full_bar_o_acc,
    PhaseTracker<NUM_KV_STAGES>& kv_ph, PhaseTracker<1>& q_ph, PhaseTracker<1>& spo_ph) {
  // B128 swizzle atom = 128 B = 64 bf16; all SMEM tiles laid out in 64-wide sub-tiles.
  constexpr int SUB_COLS_BF16     = 64;
  constexpr int SUB_COLS_BYTES    = SUB_COLS_BF16 * (int)sizeof(__nv_bfloat16);
  constexpr int Q_SUBTILES        = HEAD_DIM / SUB_COLS_BF16;
  constexpr int K_SUBTILES        = HEAD_DIM / SUB_COLS_BF16;
  constexpr int V_SUBTILES        = K_TILE / SUB_COLS_BF16;
  constexpr int K_ATOMS_PER_TILE  = SUB_COLS_BF16 / 16;
  constexpr int Q_SUB_COLS_BYTES  = M_TILE * SUB_COLS_BYTES;
  constexpr int K_SUB_COLS_BYTES  = K_TILE * SUB_COLS_BYTES;
  // >> 4: the SMEM descriptor address field is in 16-byte units (addr >> 4).
  constexpr uint64_t KV_STAGE_DELTA = (uint64_t)(K_SUBTILES * K_SUB_COLS_BYTES) >> 4;  // KV ring stage
  constexpr uint64_t Q_MTILE_DELTA  = (uint64_t)(Q_SUBTILES * Q_SUB_COLS_BYTES) >> 4;  // Q per M-tile
  constexpr uint64_t SUB_DELTA      = (uint64_t)Q_SUB_COLS_BYTES >> 4;                 // within-tile swizzle atom
  constexpr int S_COLS       = K_TILE;
  constexpr int O_COLS       = HEAD_DIM;
  constexpr int SPLIT_P_ATOM = (K_TILE / 4 * 3) / 16;

  // prologue: BMM1 of K-block 0 for every M-tile.
  int kv_stage = kv_ph.get_stage();
  wp_begin(wpc, WP_MMA_WAIT_FULL_K);
  mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar[kv_stage]), kv_ph.get_phase());
  kv_ph.advance();
  wp_end(wpc, WP_MMA_WAIT_FULL_K);

  #pragma unroll
  for (int i = 0; i < M_TILES_PER_CTA; ++i) {
    wp_begin(wpc, WP_MMA_WAIT_FULL_Q);
    mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_q[i]), q_ph.get_phase());
    wp_end(wpc, WP_MMA_WAIT_FULL_Q);

    wp_begin(wpc, WP_MMA_ISSUE);
    if (lead) {
      const uint32_t s_tmem_addr = tmem_base + (uint32_t)(i * S_COLS);
      const uint64_t da_base = desc_q_stage0 + (uint64_t)i * Q_MTILE_DELTA;
      const uint64_t db_base = desc_kv_stage0 + (uint64_t)kv_stage * KV_STAGE_DELTA;

      #pragma unroll
      for (int s = 0; s < Q_SUBTILES; ++s) {
        const uint64_t da = da_base + (uint64_t)s * SUB_DELTA;
        const uint64_t db = db_base + (uint64_t)s * SUB_DELTA;
        #pragma unroll
        for (int ki = 0; ki < K_ATOMS_PER_TILE; ++ki) {
          const bool enable_d = (s != 0) || (ki != 0);
          tcgen05_mma_f16_ss<1>(s_tmem_addr, da + 2 * ki, db + 2 * ki, idesc_qk, enable_d);
        }
      }
    }
    wp_end(wpc, WP_MMA_ISSUE);

    wp_begin(wpc, WP_MMA_COMMIT);
    if (lead) tcgen05_commit<1>(smem_ptr_u32(&full_bar_spo[i]));
    wp_end(wpc, WP_MMA_COMMIT);
  }

  wp_begin(wpc, WP_MMA_COMMIT);
  if (lead) tcgen05_commit<1>(smem_ptr_u32(&empty_bar[kv_stage]));
  wp_end(wpc, WP_MMA_COMMIT);

  // main loop: BMM2(tile) then BMM1(next tile)
  for (int k_tile_id = 0; k_tile_id + 1 < K_TILES; ++k_tile_id) {
    // ring: kv_stage = V(current) for BMM2, kv_stage_next = K(next) for BMM1-ahead.
    const int kv_stage = kv_ph.get_stage();

    wp_begin(wpc, WP_MMA_WAIT_FULL_V);
    mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar[kv_stage]), kv_ph.get_phase());
    kv_ph.advance();
    wp_end(wpc, WP_MMA_WAIT_FULL_V);

    int kv_stage_next = 0;
    #pragma unroll
    for (int i = 0; i < M_TILES_PER_CTA; ++i) {
      wp_begin(wpc, WP_MMA_WAIT_P);
      mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_spo[i]), spo_ph.get_phase());
      wp_end(wpc, WP_MMA_WAIT_P);

      wp_begin(wpc, WP_MMA_ISSUE);
      const uint32_t s_tmem_addr = tmem_base + (uint32_t)(i * S_COLS);
      const uint32_t o_tmem_addr = tmem_base + (uint32_t)(2 * S_COLS + i * O_COLS);
      const uint64_t dbV = desc_kv_stage0 + (uint64_t)kv_stage * KV_STAGE_DELTA;
      #pragma unroll
      for (int s = 0; s < V_SUBTILES; ++s) {
        const uint64_t db_s = dbV + (uint64_t)s * SUB_DELTA;
        #pragma unroll
        for (int ki = 0; ki < K_ATOMS_PER_TILE; ++ki) {
          const int a = s * K_ATOMS_PER_TILE + ki;
          if constexpr (SPLIT_P) if (a == SPLIT_P_ATOM) {
            wp_end(wpc, WP_MMA_ISSUE);
            wp_begin(wpc, WP_MMA_WAIT_P);
            mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_p_last[i]), spo_ph.get_phase());
            wp_end(wpc, WP_MMA_WAIT_P);
            wp_begin(wpc, WP_MMA_ISSUE);
          }
          const bool accumulate = (k_tile_id != 0) || (a != 0);
          if (lead) {
            tcgen05_mma_f16_ts_1sm(o_tmem_addr, s_tmem_addr + (uint32_t)(a * 8), db_s + 2 * ki, idesc_pv, accumulate, 0, 0, 0, 0);
          }
        }
      }
      wp_end(wpc, WP_MMA_ISSUE);

      // K(next) is shared by both M-tiles: only i==0 waits + advances the ring.
      if (i == 0) {
        kv_stage_next = kv_ph.get_stage();
        wp_begin(wpc, WP_MMA_WAIT_FULL_K);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar[kv_stage_next]), kv_ph.get_phase());
        wp_end(wpc, WP_MMA_WAIT_FULL_K);
        kv_ph.advance();
      }

      if (lead && i == M_TILES_PER_CTA - 1) {
        wp_begin(wpc, WP_MMA_COMMIT);
        tcgen05_commit<1>(smem_ptr_u32(&empty_bar[kv_stage]));
        wp_end(wpc, WP_MMA_COMMIT);
      }

      // BMM1(next): Q@K -> S
      wp_begin(wpc, WP_MMA_ISSUE);
      if (lead) {
        const uint64_t da_base = desc_q_stage0 + (uint64_t)i * Q_MTILE_DELTA;
        const uint64_t db_base = desc_kv_stage0 + (uint64_t)kv_stage_next * KV_STAGE_DELTA;
        #pragma unroll
        for (int s = 0; s < Q_SUBTILES; ++s) {
          const uint64_t da = da_base + (uint64_t)s * SUB_DELTA;
          const uint64_t db = db_base + (uint64_t)s * SUB_DELTA;
          #pragma unroll
          for (int ki = 0; ki < K_ATOMS_PER_TILE; ++ki) {
            const bool enable_d = (s != 0) || (ki != 0);
            tcgen05_mma_f16_ss<1>(s_tmem_addr, da + 2 * ki, db + 2 * ki, idesc_qk, enable_d);
          }
        }
      }
      wp_end(wpc, WP_MMA_ISSUE);

      wp_begin(wpc, WP_MMA_COMMIT);
      if (lead) tcgen05_commit<1>(smem_ptr_u32(&full_bar_spo[i]));
      wp_end(wpc, WP_MMA_COMMIT);
    }

    wp_begin(wpc, WP_MMA_COMMIT);
    if (lead) tcgen05_commit<1>(smem_ptr_u32(&empty_bar[kv_stage_next]));
    wp_end(wpc, WP_MMA_COMMIT);

    spo_ph.advance();
  }

  wp_begin(wpc, WP_MMA_COMMIT);
  if (lead) {
    #pragma unroll
    for (int i = 0; i < M_TILES_PER_CTA; ++i) {
      tcgen05_commit<1>(smem_ptr_u32(&empty_bar_q[i]));
    }
  }
  wp_end(wpc, WP_MMA_COMMIT);

  // epilogue: BMM2 of the last K-block -> final O
  kv_stage = kv_ph.get_stage();

  wp_begin(wpc, WP_MMA_WAIT_FULL_V);
  mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar[kv_stage]), kv_ph.get_phase());
  kv_ph.advance();
  wp_end(wpc, WP_MMA_WAIT_FULL_V);

  #pragma unroll
  for (int i = 0; i < M_TILES_PER_CTA; ++i) {
    wp_begin(wpc, WP_MMA_WAIT_P);
    mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_spo[i]), spo_ph.get_phase());
    wp_end(wpc, WP_MMA_WAIT_P);

    wp_begin(wpc, WP_MMA_ISSUE);
    const uint32_t s_tmem_addr = tmem_base + (uint32_t)(i * S_COLS);
    const uint32_t o_tmem_addr = tmem_base + (uint32_t)(2 * S_COLS + i * O_COLS);
    const uint64_t dbV = desc_kv_stage0 + (uint64_t)kv_stage * KV_STAGE_DELTA;
    #pragma unroll
    for (int s = 0; s < V_SUBTILES; ++s) {
      const uint64_t db_s = dbV + (uint64_t)s * SUB_DELTA;
      #pragma unroll
      for (int ki = 0; ki < K_ATOMS_PER_TILE; ++ki) {
        const int a = s * K_ATOMS_PER_TILE + ki;   // flat atom (split-P + P addr)
        if constexpr (SPLIT_P) if (a == SPLIT_P_ATOM) {
          wp_end(wpc, WP_MMA_ISSUE);
          wp_begin(wpc, WP_MMA_WAIT_P);
          mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_p_last[i]), spo_ph.get_phase());
          wp_end(wpc, WP_MMA_WAIT_P);
          wp_begin(wpc, WP_MMA_ISSUE);
        }
        const bool accumulate = (K_TILES != 1) || (a != 0);
        if (lead) {
          tcgen05_mma_f16_ts_1sm(o_tmem_addr, s_tmem_addr + (uint32_t)(a * 8), db_s + 2 * ki, idesc_pv, accumulate, 0, 0, 0, 0);
        }
      }
    }
    wp_end(wpc, WP_MMA_ISSUE);

    wp_begin(wpc, WP_MMA_COMMIT);
    if (lead) tcgen05_commit<1>(smem_ptr_u32(&full_bar_o_acc[i]));
    wp_end(wpc, WP_MMA_COMMIT);
  }

  wp_begin(wpc, WP_MMA_COMMIT);
  if (lead) tcgen05_commit<1>(smem_ptr_u32(&empty_bar[kv_stage]));
  wp_end(wpc, WP_MMA_COMMIT);

  spo_ph.advance();
  q_ph.advance();
}

/* ============================================================================
 * fmha_mma_warp_blackwell_ntiles_1sm_bf16<>(wpc, ...)
 *
 * Production __device__ body for the MMA warp. Owns the TMEM lifecycle: allocs
 * TMEM (writes tmem_slot), publishes it to the softmax + correction warps via a
 * bar.sync scoped to the TMEM_ALLOC_THREADS TMEM-user warps, then a persistent
 * loop of decode_workitem (#110) + fmha_mma_warp_blackwell_1tile_1sm_bf16; a
 * closing bar.sync waits those warps out before the tail tcgen05.dealloc.
 * ============================================================================ */
// VARLEN=true selects the varlen decode (prefix-sum cu_seqlens_q/seqlens_kv, ragged K_TILES) +
// the short-sample `q_tile_base < seqlen_q` guard; TAIL_DRAIN=false (varlen has no closing-prime
// empty_bar_spo drain). The alloc/publish/teardown + 1tile body are shared with uniform.
template <int NUM_KV_STAGES, int M_TILES_PER_CTA, int M_TILE, int K_TILE, int HEAD_DIM,
          bool SPLIT_P, bool USE_CLC, int CLC_STAGES, bool Q_RASTER, bool IS_CAUSAL, bool LPT,
          int TMEM_TOTAL_COLS, int MMA_REG_BUDGET = 48, bool TAIL_DRAIN = true, bool VARLEN = false>
__device__ inline
void fmha_mma_warp_blackwell_ntiles_1sm_bf16(WpCtx& wpc,
    uint32_t* tmem_slot, const uint8_t* sQ0, const uint8_t* sKV,
    uint64_t* full_bar, uint64_t* empty_bar, uint64_t* full_bar_q, uint64_t* empty_bar_q,
    uint64_t* full_bar_spo, uint64_t* empty_bar_spo, uint64_t* full_bar_p_last,
    uint64_t* full_bar_o_acc,
    uint64_t* clc_full, uint64_t* clc_empty, uint32_t* clc_response,
    int seqlen_kv, int num_q_heads, int num_kv_heads, int packed_mtiles_per_seq, int num_samples,
    unsigned long long magic0, unsigned long long magic1, unsigned long long magic2,
    const int* cu_seqlens_q = nullptr, const int* seqlens_kv = nullptr) {
  setmaxnreg_dec<MMA_REG_BUDGET>();

  const int q_tile_per_cta = M_TILES_PER_CTA * (M_TILE / (num_q_heads / num_kv_heads));
  const int total_packed_mtiles = num_samples * packed_mtiles_per_seq * num_kv_heads;

  const bool lead = elect_one_sync();
  // This warp owns the TMEM alloc when tmem_slot is given (nullptr -> the caller
  // allocated and pre-populated *tmem_slot itself, GEMM 90-style). bar.arrive
  // (not sync) so this warp starts the pipeline without waiting the consumers.
  uint32_t tmem_base = 0;
  if (tmem_slot != nullptr) {
    wp_begin(wpc, WP_MMA_TMEM_2CTA_ALLOC);
    tcgen05_alloc<1>(smem_ptr_u32(tmem_slot), TMEM_TOTAL_COLS);
    bar_arrive<9>(416);   // softmax (w0-7) + corr (w8-11) + mma (w12): 13 warps
    tmem_base = *tmem_slot;
    wp_end(wpc, WP_MMA_TMEM_2CTA_ALLOC);
  }

  constexpr uint32_t DESC_SBO = 1024, DESC_LBO = 16;
  const uint32_t idesc_qk = make_idesc_bf16_f32(M_TILE, K_TILE, false, false);
  const uint32_t idesc_pv = make_idesc_bf16_f32(M_TILE, HEAD_DIM, false, false);
  const uint64_t desc_q_stage0  = build_smem_desc_blackwell(smem_ptr_u32(sQ0), DESC_SBO, DESC_LBO, SmemSwizzleBlackwell::B128);
  const uint64_t desc_kv_stage0 = build_smem_desc_blackwell(smem_ptr_u32(sKV), DESC_SBO, DESC_LBO, SmemSwizzleBlackwell::B128);

  PhaseTracker<NUM_KV_STAGES> kv_ph;
  PhaseTracker<1> q_ph;
  PhaseTracker<1> spo_ph;
  [[maybe_unused]] int clc_stage = 0;
  [[maybe_unused]] uint32_t clc_phase = 0;
  int tile_id = (int)blockIdx.x;
  while (true) {
    int sample, h_kv, q_tile_base, K_TILES, seqlen_q = 0;
    if constexpr (VARLEN) {
      decode_workitem_varlen<K_TILE, Q_RASTER, IS_CAUSAL, LPT>(tile_id, num_kv_heads,
          packed_mtiles_per_seq, packed_mtiles_per_seq * num_kv_heads, q_tile_per_cta,
          cu_seqlens_q, seqlens_kv, sample, h_kv, q_tile_base, seqlen_q, K_TILES);
    } else {
      decode_workitem<K_TILE, Q_RASTER, IS_CAUSAL, LPT>(tile_id, seqlen_kv, num_kv_heads,
          packed_mtiles_per_seq, q_tile_per_cta, magic0, magic1, magic2,
          sample, h_kv, q_tile_base, K_TILES);
    }

    if (!VARLEN || q_tile_base < seqlen_q)
    fmha_mma_warp_blackwell_1tile_1sm_bf16<NUM_KV_STAGES, M_TILES_PER_CTA, M_TILE, K_TILE, HEAD_DIM, SPLIT_P>(wpc, tmem_base, lead, desc_q_stage0, desc_kv_stage0, idesc_qk, idesc_pv, K_TILES,
        full_bar, empty_bar, full_bar_q, empty_bar_q,
        full_bar_spo, empty_bar_spo, full_bar_p_last, full_bar_o_acc,
        kv_ph, q_ph, spo_ph);

    if constexpr (USE_CLC) {
      ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, /*CTA_GROUP=*/1>(
          clc_full, clc_empty, clc_response, clc_stage, clc_phase, /*do_release=*/elect_one_sync());
      clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
      if (!next.valid) break;
      tile_id = next.n_tile;
    } else {
      tile_id += gridDim.x;
      if (tile_id >= total_packed_mtiles) break;
    }
  }

  wp_begin(wpc, WP_MMA_TMEM_2CTA_FREE);
  bar_sync<9>(416);
  if (tmem_slot != nullptr) {
    tcgen05_relinquish_alloc_permit<1>();
    tcgen05_dealloc<1>(tmem_base, TMEM_TOTAL_COLS);
  }
  wp_end(wpc, WP_MMA_TMEM_2CTA_FREE);

  if constexpr (TAIL_DRAIN) {
    // Tail drain: empty_bar_spo (corr dangling + softmax closing prime -> full 256).
    wp_begin(wpc, WP_MMA_WAIT_P);
    #pragma unroll
    for (int i = 0; i < M_TILES_PER_CTA; ++i) {
      mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_spo[i]), spo_ph.get_phase());
    }
    wp_end(wpc, WP_MMA_WAIT_P);
  }
}

/* ============================================================================
 * fmha_mma_warp_blackwell_1tile_2sm_bf16<>(wpc, ...)
 *
 * One FMHA work-item's cta_group::2 MMA pipeline (LEADER CTA / peer 0 only):
 * BMM1-ahead over M_TILES_PER_CTA q-tiles at joint M = 2*M_TILE (256 rows).
 * BMM1 = Q@K (SS, mma<2>), BMM2 = P@V (TS with a TMEM P operand, ts_2sm). Every
 * commit is tcgen05_commit_multicast<2>(bar, 0x3) so both CTAs' barrier copies
 * see it. K/V descriptors step by the half-box KV subtile (8KB); Q by the full
 * 16KB. idesc M = 256 is built by the caller. Trackers persist by reference.
 * ============================================================================ */
template <int NUM_KV_STAGES, int M_TILES_PER_CTA, int M_TILE, int K_TILE, int HEAD_DIM, bool SPLIT_P>
__device__ inline
void fmha_mma_warp_blackwell_1tile_2sm_bf16(WpCtx& wpc,
    uint32_t tmem_base, bool lead,
    uint64_t desc_q0, uint64_t desc_kv0, uint32_t idesc_qk, uint32_t idesc_pv,
    int K_TILES,
    uint64_t* full_bar, uint64_t* empty_bar, uint64_t* full_bar_q, uint64_t* empty_bar_q,
    uint64_t* full_bar_spo, uint64_t* empty_bar_spo, uint64_t* full_bar_p_last,
    uint64_t* full_bar_o_acc,
    PhaseTracker<NUM_KV_STAGES>& kv_ph, PhaseTracker<1>& q_ph, PhaseTracker<1>& spo_ph) {
  constexpr int SUB_COLS_BF16     = 64;
  constexpr int SUB_COLS_BYTES    = SUB_COLS_BF16 * (int)sizeof(__nv_bfloat16);
  constexpr int Q_SUBTILES        = HEAD_DIM / SUB_COLS_BF16;
  constexpr int V_SUBTILES        = K_TILE / SUB_COLS_BF16;
  constexpr int K_ATOMS_PER_TILE  = SUB_COLS_BF16 / 16;
  constexpr int Q_SUB_COLS_BYTES  = M_TILE * SUB_COLS_BYTES;
  constexpr int K_TILE_BYTES      = (HEAD_DIM / SUB_COLS_BF16) * (K_TILE * SUB_COLS_BYTES);
  constexpr int S_COLS            = K_TILE;
  constexpr int O_COLS            = HEAD_DIM;
  constexpr int SPLIT_P_ATOM      = (K_TILE / 4 * 3) / 16;
  constexpr uint64_t KV_DESC_DELTA      = (uint64_t)K_TILE_BYTES >> 4;            // ring-slot stride (32KB, half-used)
  constexpr uint64_t SUB_DESC_DELTA_Q   = (uint64_t)Q_SUB_COLS_BYTES >> 4;        // Q head-dim subtile (16KB)
  constexpr uint64_t SUB_DESC_DELTA_KV  = (uint64_t)(Q_SUB_COLS_BYTES / 2) >> 4;  // K/V head-dim subtile (8KB, half-box)
  constexpr uint64_t Q_MTILE_DESC_DELTA = (uint64_t)(Q_SUBTILES * Q_SUB_COLS_BYTES) >> 4;

  // prologue: BMM1 of K-block 0 for every M-tile.
  int kv_stage = kv_ph.get_stage();
  wp_begin(wpc, WP_MMA_WAIT_FULL_K);
  mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar[kv_stage]), kv_ph.get_phase());
  kv_ph.advance();
  wp_end(wpc, WP_MMA_WAIT_FULL_K);

  #pragma unroll
  for (int i = 0; i < M_TILES_PER_CTA; ++i) {
    wp_begin(wpc, WP_MMA_WAIT_FULL_Q);
    mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_q[i]), q_ph.get_phase());
    wp_end(wpc, WP_MMA_WAIT_FULL_Q);

    wp_begin(wpc, WP_MMA_ISSUE);
    if (lead) {
      const uint32_t s_tmem_addr = tmem_base + (uint32_t)(i * S_COLS);
      const uint64_t da_base = desc_q0 + (uint64_t)i * Q_MTILE_DESC_DELTA;
      const uint64_t db_base = desc_kv0 + (uint64_t)kv_stage * KV_DESC_DELTA;
      #pragma unroll
      for (int s = 0; s < Q_SUBTILES; ++s) {
        const uint64_t da = da_base + (uint64_t)s * SUB_DESC_DELTA_Q;
        const uint64_t db = db_base + (uint64_t)s * SUB_DESC_DELTA_KV;
        #pragma unroll
        for (int ki = 0; ki < K_ATOMS_PER_TILE; ++ki) {
          const bool enable_d = (s != 0) || (ki != 0);
          tcgen05_mma_f16_ss<2>(s_tmem_addr, da + 2 * ki, db + 2 * ki, idesc_qk, enable_d);
        }
      }
    }
    wp_end(wpc, WP_MMA_ISSUE);

    wp_begin(wpc, WP_MMA_COMMIT);
    if (lead) tcgen05_commit_multicast<2>(smem_ptr_u32(&full_bar_spo[i]), 0x3);
    wp_end(wpc, WP_MMA_COMMIT);
  }

  wp_begin(wpc, WP_MMA_COMMIT);
  if (lead) tcgen05_commit_multicast<2>(smem_ptr_u32(&empty_bar[kv_stage]), 0x3);
  wp_end(wpc, WP_MMA_COMMIT);

  // main loop: BMM2(tile) then BMM1(next tile)
  for (int k_tile_id = 0; k_tile_id + 1 < K_TILES; ++k_tile_id) {
    const int kv_stage = kv_ph.get_stage();
    wp_begin(wpc, WP_MMA_WAIT_FULL_V);
    mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar[kv_stage]), kv_ph.get_phase());
    kv_ph.advance();
    wp_end(wpc, WP_MMA_WAIT_FULL_V);

    int kv_stage_next = 0;
    #pragma unroll
    for (int i = 0; i < M_TILES_PER_CTA; ++i) {
      wp_begin(wpc, WP_MMA_WAIT_P);
      mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_spo[i]), spo_ph.get_phase());
      wp_end(wpc, WP_MMA_WAIT_P);

      wp_begin(wpc, WP_MMA_ISSUE);
      const uint32_t s_tmem_addr = tmem_base + (uint32_t)(i * S_COLS);
      const uint32_t o_tmem_addr = tmem_base + (uint32_t)(2 * S_COLS + i * O_COLS);
      const uint64_t dbV = desc_kv0 + (uint64_t)kv_stage * KV_DESC_DELTA;
      #pragma unroll
      for (int s = 0; s < V_SUBTILES; ++s) {
        const uint64_t db_s = dbV + (uint64_t)s * SUB_DESC_DELTA_KV;
        #pragma unroll
        for (int ki = 0; ki < K_ATOMS_PER_TILE; ++ki) {
          const int a = s * K_ATOMS_PER_TILE + ki;
          if constexpr (SPLIT_P) if (a == SPLIT_P_ATOM) {
            wp_end(wpc, WP_MMA_ISSUE);
            wp_begin(wpc, WP_MMA_WAIT_P);
            mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_p_last[i]), spo_ph.get_phase());
            wp_end(wpc, WP_MMA_WAIT_P);
            wp_begin(wpc, WP_MMA_ISSUE);
          }
          const bool accumulate = (k_tile_id != 0) || (a != 0);
          if (lead) {
            tcgen05_mma_f16_ts_2sm(o_tmem_addr, s_tmem_addr + (uint32_t)(a * 8), db_s + 2 * ki, idesc_pv, accumulate, 0, 0, 0, 0, 0, 0, 0, 0);
          }
        }
      }
      wp_end(wpc, WP_MMA_ISSUE);

      if (i == 0) {
        kv_stage_next = kv_ph.get_stage();
        wp_begin(wpc, WP_MMA_WAIT_FULL_K);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar[kv_stage_next]), kv_ph.get_phase());
        wp_end(wpc, WP_MMA_WAIT_FULL_K);
        kv_ph.advance();
      }

      if (lead && i == M_TILES_PER_CTA - 1) {
        wp_begin(wpc, WP_MMA_COMMIT);
        tcgen05_commit_multicast<2>(smem_ptr_u32(&empty_bar[kv_stage]), 0x3);
        wp_end(wpc, WP_MMA_COMMIT);
      }

      // BMM1(next): Q@K -> S
      wp_begin(wpc, WP_MMA_ISSUE);
      if (lead) {
        const uint64_t da_base = desc_q0 + (uint64_t)i * Q_MTILE_DESC_DELTA;
        const uint64_t db_base = desc_kv0 + (uint64_t)kv_stage_next * KV_DESC_DELTA;
        #pragma unroll
        for (int s = 0; s < Q_SUBTILES; ++s) {
          const uint64_t da = da_base + (uint64_t)s * SUB_DESC_DELTA_Q;
          const uint64_t db = db_base + (uint64_t)s * SUB_DESC_DELTA_KV;
          #pragma unroll
          for (int ki = 0; ki < K_ATOMS_PER_TILE; ++ki) {
            const bool enable_d = (s != 0) || (ki != 0);
            tcgen05_mma_f16_ss<2>(s_tmem_addr, da + 2 * ki, db + 2 * ki, idesc_qk, enable_d);
          }
        }
      }
      wp_end(wpc, WP_MMA_ISSUE);

      wp_begin(wpc, WP_MMA_COMMIT);
      if (lead) tcgen05_commit_multicast<2>(smem_ptr_u32(&full_bar_spo[i]), 0x3);
      wp_end(wpc, WP_MMA_COMMIT);
    }

    wp_begin(wpc, WP_MMA_COMMIT);
    if (lead) tcgen05_commit_multicast<2>(smem_ptr_u32(&empty_bar[kv_stage_next]), 0x3);
    wp_end(wpc, WP_MMA_COMMIT);

    spo_ph.advance();
  }

  wp_begin(wpc, WP_MMA_COMMIT);
  if (lead) {
    #pragma unroll
    for (int i = 0; i < M_TILES_PER_CTA; ++i) {
      tcgen05_commit_multicast<2>(smem_ptr_u32(&empty_bar_q[i]), 0x3);
    }
  }
  wp_end(wpc, WP_MMA_COMMIT);

  // epilogue: BMM2 of the last K-block -> final O
  kv_stage = kv_ph.get_stage();
  wp_begin(wpc, WP_MMA_WAIT_FULL_V);
  mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar[kv_stage]), kv_ph.get_phase());
  kv_ph.advance();
  wp_end(wpc, WP_MMA_WAIT_FULL_V);

  #pragma unroll
  for (int i = 0; i < M_TILES_PER_CTA; ++i) {
    wp_begin(wpc, WP_MMA_WAIT_P);
    mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_spo[i]), spo_ph.get_phase());
    wp_end(wpc, WP_MMA_WAIT_P);

    wp_begin(wpc, WP_MMA_ISSUE);
    const uint32_t s_tmem_addr = tmem_base + (uint32_t)(i * S_COLS);
    const uint32_t o_tmem_addr = tmem_base + (uint32_t)(2 * S_COLS + i * O_COLS);
    const uint64_t dbV = desc_kv0 + (uint64_t)kv_stage * KV_DESC_DELTA;
    #pragma unroll
    for (int s = 0; s < V_SUBTILES; ++s) {
      const uint64_t db_s = dbV + (uint64_t)s * SUB_DESC_DELTA_KV;
      #pragma unroll
      for (int ki = 0; ki < K_ATOMS_PER_TILE; ++ki) {
        const int a = s * K_ATOMS_PER_TILE + ki;
        if constexpr (SPLIT_P) if (a == SPLIT_P_ATOM) {
          wp_end(wpc, WP_MMA_ISSUE);
          wp_begin(wpc, WP_MMA_WAIT_P);
          mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_p_last[i]), spo_ph.get_phase());
          wp_end(wpc, WP_MMA_WAIT_P);
          wp_begin(wpc, WP_MMA_ISSUE);
        }
        const bool accumulate = (K_TILES != 1) || (a != 0);
        if (lead) {
          tcgen05_mma_f16_ts_2sm(o_tmem_addr, s_tmem_addr + (uint32_t)(a * 8), db_s + 2 * ki, idesc_pv, accumulate, 0, 0, 0, 0, 0, 0, 0, 0);
        }
      }
    }
    wp_end(wpc, WP_MMA_ISSUE);

    wp_begin(wpc, WP_MMA_COMMIT);
    if (lead) tcgen05_commit_multicast<2>(smem_ptr_u32(&full_bar_o_acc[i]), 0x3);
    wp_end(wpc, WP_MMA_COMMIT);
  }

  wp_begin(wpc, WP_MMA_COMMIT);
  if (lead) tcgen05_commit_multicast<2>(smem_ptr_u32(&empty_bar[kv_stage]), 0x3);
  wp_end(wpc, WP_MMA_COMMIT);

  spo_ph.advance();
  q_ph.advance();
}

/* ============================================================================
 * fmha_mma_warp_blackwell_ntiles_2sm_bf16<>(wpc, ...)
 *
 * Production __device__ body for the MMA warp of the 2SM (cta_group::2) FMHA
 * context kernel (fmha_context_bf16_uniform_2sm.cu). Kept SEPARATE from the 1SM
 * ntiles wrapper because the 2SM TMEM lifecycle has no 1SM analog: BOTH CTAs'
 * MMA warps run a cluster-collective tcgen05.alloc<2>; only the leader CTA
 * (peer 0) issues the cta_group::2 MMAs while peer 1 runs an idle CLC consumer
 * for count-balance; teardown pairs a symmetric cross-CTA handshake (peer-bit
 * XOR on tmem_dealloc_bar) before tcgen05.dealloc<2>. Persistent loop of
 * decode_workitem (#110) + fmha_mma_warp_blackwell_1tile_2sm_bf16.
 * ============================================================================ */
template <int NUM_KV_STAGES, int M_TILES_PER_CTA, int M_TILE, int K_TILE, int HEAD_DIM,
          bool SPLIT_P, bool USE_CLC, int CLC_STAGES, bool Q_RASTER, bool IS_CAUSAL,
          int TMEM_TOTAL_COLS, int MMA_REG_BUDGET = 48, bool TAIL_DRAIN = true>
__device__ inline
void fmha_mma_warp_blackwell_ntiles_2sm_bf16(WpCtx& wpc,
    uint32_t* tmem_slot, uint64_t* tmem_dealloc_bar,
    const uint8_t* sQ0, const uint8_t* sKV,
    uint64_t* full_bar, uint64_t* empty_bar, uint64_t* full_bar_q, uint64_t* empty_bar_q,
    uint64_t* full_bar_spo, uint64_t* empty_bar_spo, uint64_t* full_bar_p_last,
    uint64_t* full_bar_o_acc,
    uint64_t* clc_full, uint64_t* clc_empty, uint32_t* clc_response,
    int seqlen_kv, int num_q_heads, int num_kv_heads, int packed_idx_per_seq, int num_samples,
    unsigned long long magic0, unsigned long long magic1, unsigned long long magic2) {
  setmaxnreg_dec<MMA_REG_BUDGET>();
  constexpr int M_TILE_CLUSTER = 2 * M_TILE;   // joint cta_group::2 MMA M-dim (256)
  const int peer           = (int)(blockIdx.x & 1);
  const int q_tile_per_cta = M_TILES_PER_CTA * (M_TILE / (num_q_heads / num_kv_heads));
  const int num_clusters   = (int)(gridDim.x >> 1);
  const int total_clusters = num_samples * packed_idx_per_seq * num_kv_heads;

  // Both CTAs' MMA warps own the cta_group::2 TMEM alloc (cluster-collective) and
  // publish tmem_base to this CTA's softmax + correction warps via named barrier 9.
  wp_begin(wpc, WP_MMA_TMEM_2CTA_ALLOC);
  tcgen05_alloc<2>(smem_ptr_u32(tmem_slot), TMEM_TOTAL_COLS);
  bar_arrive<9>(416);
  const uint32_t tmem_base = *tmem_slot;
  wp_end(wpc, WP_MMA_TMEM_2CTA_ALLOC);
  PhaseTracker<1> spo_ph;   // shared by the peer-0 MMA loop and the tail empty_bar_spo drain

  // Only the LEADER CTA (peer 0) issues the cta_group::2 MMAs + commits.
  if (peer == 0) {
    const bool lead = elect_one_sync();
    constexpr uint32_t DESC_SBO = 1024, DESC_LBO = 16;
    // 2SM: idesc M = M_TILE_CLUSTER (256); N stays 128 -- each CTA supplies its half of B.
    const uint32_t idesc_qk = make_idesc_bf16_f32(M_TILE_CLUSTER, K_TILE, false, false);
    const uint32_t idesc_pv = make_idesc_bf16_f32(M_TILE_CLUSTER, HEAD_DIM, false, false);
    const uint64_t desc_q0  = build_smem_desc_blackwell(smem_ptr_u32(sQ0), DESC_SBO, DESC_LBO, SmemSwizzleBlackwell::B128);
    const uint64_t desc_kv0 = build_smem_desc_blackwell(smem_ptr_u32(sKV), DESC_SBO, DESC_LBO, SmemSwizzleBlackwell::B128);

    PhaseTracker<NUM_KV_STAGES> kv_ph;
    PhaseTracker<1> q_ph;
    int cluster_tile_id = (int)(blockIdx.x >> 1);
    [[maybe_unused]] int clc_stage = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;
    while (true) {
      int sample, h_kv, q_tile_base, K_TILES;
      decode_workitem<K_TILE, Q_RASTER, IS_CAUSAL, /*LPT=*/false, /*CLUSTER_N=*/2>(
          cluster_tile_id, seqlen_kv, num_kv_heads, packed_idx_per_seq, q_tile_per_cta,
          magic0, magic1, magic2, sample, h_kv, q_tile_base, K_TILES, /*peer=*/0);
      (void)sample; (void)h_kv;
      (void)q_tile_base;   // peer 0 (leader) is always paired (even tile in range) -- no bail.

      fmha_mma_warp_blackwell_1tile_2sm_bf16<NUM_KV_STAGES, M_TILES_PER_CTA, M_TILE, K_TILE, HEAD_DIM, SPLIT_P>(wpc, tmem_base, lead, desc_q0, desc_kv0, idesc_qk, idesc_pv, K_TILES,
          full_bar, empty_bar, full_bar_q, empty_bar_q,
          full_bar_spo, empty_bar_spo, full_bar_p_last, full_bar_o_acc,
          kv_ph, q_ph, spo_ph);

      if constexpr (USE_CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 2, ClcRasterOrder::AlongN, /*CTA_GROUP=*/2>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, /*do_release=*/elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        if (!next.valid) break;
        cluster_tile_id = next.n_tile;
      } else {
        cluster_tile_id += num_clusters;
        if (cluster_tile_id >= total_clusters) break;
      }
    }
  }
  // 2SM count-balance: peer 1 issues no MMAs but under CLC MUST still run as an idle CLC
  // consumer (clc_empty expects 32 arrives per work-item; 31 -> SILENT HANG).
  else if constexpr (USE_CLC) {   // peer == 1
    int clc_stage = 0; uint32_t clc_phase = 0;
    while (true) {
      ClcTileInfo next = clc_fetch_next_tile<1, 2, ClcRasterOrder::AlongN, /*CTA_GROUP=*/2>(
          clc_full, clc_empty, clc_response, clc_stage, clc_phase, /*do_release=*/elect_one_sync());
      clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
      if (!next.valid) break;
    }
  }

  // 2SM TMEM teardown: bar 9 proves this CTA's softmax + corr are done with TMEM; then a
  // symmetric cross-CTA handshake (peer-bit XOR) proves BOTH CTAs are done before dealloc<2>.
  wp_begin(wpc, WP_MMA_TMEM_2CTA_FREE);
  bar_sync<9>(416);
  {
    const uint32_t bar_local = smem_ptr_u32(tmem_dealloc_bar);
    mbarrier_arrive_cluster_default(bar_local ^ 0x01000000u);
    mbarrier_wait_parity_suspend(bar_local, /*phase=*/0);
  }
  tcgen05_relinquish_alloc_permit<2>();
  tcgen05_dealloc<2>(tmem_base, TMEM_TOTAL_COLS);
  wp_end(wpc, WP_MMA_TMEM_2CTA_FREE);

  if constexpr (TAIL_DRAIN) {
    // Tail drain: empty_bar_spo (peer 0 only). Its 512 flip = both CTAs' corr dangling epilogue
    // (256) + both CTAs' softmax closing primes (256), all routed to peer 0 via mapa.
    if (peer == 0) {
      wp_begin(wpc, WP_MMA_WAIT_P);
      #pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i) {
        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_spo[i]), spo_ph.get_phase());
      }
      wp_end(wpc, WP_MMA_WAIT_P);
    }
  }
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
