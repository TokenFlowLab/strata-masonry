// fmha_context_bf16_uniform_2sm.cu -- K2 FMHA context BF16, sm_100a.
//
// 2-CTA (cta_group::2) sibling of fmha_context_bf16_uniform.cu: same feature set
// (GQA/MHA via MHA; full/causal via IS_CAUSAL; USE_CLC / Q_RASTER scheduler), but a 2SM cluster
// pairs two CTAs on one (sample, kv_head) with a joint 256-row cta_group::2 MMA.
//
// ASSUMES all seqlens EQUAL (no varlen) -> uniform K_TILES (varlen lives in
// fmha_context_bf16_varlen.cu). IS_CAUSAL=true adds the triangular mask + a K-loop cap;
// the cap comes from the higher (odd / peer1) tile so both peers keep the same K_TILES (lockstep
// for the joint MMA), and each CTA masks its own rows with its own q_pos.
//
// GEN-shape variant. Warp-specialized 16-warp body + barrier contract are the SAME as
// fmha_context_bf16_gqa_nonpersistent.cu (see its header for Terminology / Layout / flow / barriers).
//   - PERSISTENT: CLC / cluster-stride over work tiles (phase trackers persist).
//   - TMA-store epilogue: valid only because full/equal-seqlen tiles are non-ragged (varlen kernels
//     use a predicated STG re-tile instead).
//
// Barrier contract (additions to gqa_nonpersistent's -- unique to the TMA-sO epilogue):
//   - empty_bar_o_epi[m] (count 1): epi -> corr, "sO[m]'s TMA store drained, slot reusable" --
//     see fmha_context_bf16_uniform.cu's header. Both peers run it (the padding peer's waits
//     are no-ops).
//
// Budgets:
//   - Registers: per-warp budgets sum to exactly the SM file (65536 = 128*512):
//       softmax inc<176> (x8) + correction dec<88> (x4) + four single warps dec<72> (x4)
//       = 32 * (8*176 + 4*88 + 4*72) = 32 * 2048 = 65536.
//     This ladder differs from the BLOCK BASE fmha_context_bf16_uniform_2sm.cu, which really does
//     run 192/80/48 (it passes SM_REG_BUDGET=192 / CORR_REG_BUDGET=80 / *_REG_BUDGET=48 to the
//     blocks/*). Both ladders total 65536; this fork just shifts registers from the softmax warps
//     to correction and the single warps. This header claimed the base's 192/80/48 until 2026-07-25
//     -- if you are comparing the two kernels, read the setmaxnreg calls, not the prose.
//     WARP_PROF: the softmax warp sits at its cap, so wp_begin/wp_end there can fault as an illegal
//     instruction; raise the budget before profiling. (The old "single ~R26 / corr ~R77 / softmax
//     ~R186" usage figures were measured under the 192/80/48 ladder and no longer apply -- ~R186
//     does not even fit a 176 budget.)
//   - TMEM: each M-tile needs S + O = K_TILE + HEAD_DIM cols of the 512, so
//     M_TILES_PER_CTA <= 512 / (K_TILE + HEAD_DIM) = 2 for 128/128 (adjacent q-tiles of the
//     SAME sequence).
//
// Work-item math (query-row tiling, per (sample, kv_head)):
//   Three granularities, each 2x coarser than the last:
//     raw M_TILE (128 rows)  --/ M_TILES_PER_CTA (2)-->  packed_mtile (1 CTA's work)
//     packed_mtile           --/ 2 CTAs per cluster --> packed_mpair  (1 cluster's work)
//   So a packed_mtile = 2 M_TILEs, and a packed_mpair = 2 packed_mtiles = 4 M_TILEs.
//   q_tile_per_cta       = M_TILES_PER_CTA * q_tile_per_mtile   (tokens one CTA covers)
//   packed_mtiles_per_seq = ceil(seqlen / q_tile_per_cta)       (single-CTA tiles per seq)
//   packed_mpairs_per_seq = ceil(packed_mtiles_per_seq / 2)     (cluster work-items per seq)
//   A cluster processes one packed_mpair: peer 0 = even packed_mtile, peer 1 = odd (clamped if
//   the count is odd). The scheduler enumerates cluster_workitem_id over
//   num_samples * packed_mpairs_per_seq * num_kv_heads; decode_workitem() splits it back into
//   (sample, kv_head, packed_mpairs_index) and q_tile_base = (2*index + peer) * q_tile_per_cta.
//
// EX2_EMU hybrid exp2 (FA4 apply_exp2_convert):
//   - Per 32-elt fragment, a fraction of pairs use the f32x2 ALU emulation (ex2_emu_f32x2)
//     instead of MUFU.EX2 to relieve the EX2 pipe.
//   - Gate: HW unless (k%EX2_FREQ >= EX2_FREQ-EX2_RES) AND (fragment < last), where for
//     softmax pair c: fragment j = c/EX2_FRG_PAIRS, in-fragment elt k = 2*(c%EX2_FRG_PAIRS).

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <type_traits>
#include <cstring>
#include <cmath>
#include <vector>
#include "../../../tests/test_utils.cuh"
#include "../../../primitives/0_tcgen05_alloc.cuh"
#include "../../../primitives/1_tcgen05_dealloc.cuh"
#include "../../../primitives/2_tcgen05_relinquish.cuh"
#include "../../../primitives/3_tcgen05_mma_f16.cuh"
#include "../../../primitives/8_tcgen05_mma_idesc.cuh"
#include "../../../primitives/9_tcgen05_ld.cuh"
#include "../../../primitives/10_tcgen05_st.cuh"
#include "../../../primitives/11_tcgen05_commit.cuh"
#include "../../../primitives/12_tcgen05_wait.cuh"
#include "../../../primitives/15_tcgen05_fence.cuh"
#include "../../../primitives/18_tma_load.cuh"
#include "../../../primitives/19_tma_load_2sm.cuh"
#include "../../../primitives/22_tma_store.cuh"
#include "../../../primitives/23_tma_tensormap.cuh"
#include "../../../primitives/25_tma_async_group.cuh"
#include "../../../primitives/29_mbarrier_init.cuh"
#include "../../../primitives/30_mbarrier_arrive.cuh"
#include "../../../primitives/31_mbarrier_arrive_tx.cuh"
#include "../../../primitives/33_mbarrier_try_wait.cuh"
#include "../../../primitives/34_fence_proxy_async.cuh"
#include "../../../primitives/35_fence_mbarrier_init.cuh"
#include "../../../primitives/42_smem_desc_blackwell.cuh"
#include "../../../primitives/44_elect_sync.cuh"
#include "../../../primitives/37_bar_sync.cuh"
#include "../../../primitives/38_barrier_cluster.cuh"
#include "../../../primitives/67_mapa.cuh"
#include "../../../primitives/63_cvt_f32_to_f16_bf16.cuh"
#include "../../../composites/118_mbarrier_phase_tracking.cuh"
#include "../../../composites/106_clc_fetch_next_tile.cuh"
#include "../../../primitives/46_setmaxnreg.cuh"
#include "../../../primitives/76_packed_f32x2.cuh"
#include "../../../primitives/77_ex2_approx.cuh"
#include "../../../primitives/78_rcp_approx.cuh"
#include "../../../primitives/_warp_prof_noop.cuh"
#include "fmha_utils.cuh"


constexpr int M_TILE = 128;
constexpr int M_TILE_CLUSTER = 2 * M_TILE; // 256
constexpr int M_TILES_PER_CTA = 2;
constexpr int K_TILE = 128;
constexpr int HEAD_DIM = 128;
// B128 swizzle atom = 128 bytes = 64 bf16: all SMEM tiles are laid out in
// 64-wide sub-tiles along the contiguous dim.
constexpr int SUB_COLS_BF16 = 64;
constexpr int SUB_COLS_BYTES = SUB_COLS_BF16 * (int)sizeof(__nv_bfloat16);   // 128 B (one swizzle atom)
constexpr int Q_SUBTILES = HEAD_DIM / SUB_COLS_BF16;  // 2
constexpr int K_SUBTILES = HEAD_DIM / SUB_COLS_BF16;  // 2 (K tile is K_TILE tokens x head_dim)
constexpr int V_SUBTILES = K_TILE / SUB_COLS_BF16;    // 2 (V_T tile is head_dim x K_TILE tokens)
constexpr int P_SUBTILES = K_TILE / SUB_COLS_BF16;    // 2
constexpr int Q_SUB_COLS_BYTES = M_TILE * SUB_COLS_BYTES;       // 16 KB
constexpr int K_SUB_COLS_BYTES = K_TILE * SUB_COLS_BYTES;       // 16 KB
constexpr int Q_TILE_BYTES = Q_SUBTILES * Q_SUB_COLS_BYTES;     // 32 KB
constexpr int K_TILE_BYTES = K_SUBTILES * K_SUB_COLS_BYTES;     // 32 KB
constexpr int KV_SLOT_BYTES = K_TILE_BYTES / 2;                 // 16 KB (half-box per CTA)
constexpr int V_TILE_BYTES = V_SUBTILES * Q_SUB_COLS_BYTES;     // 32 KB
constexpr int P_TILE_BYTES = P_SUBTILES * Q_SUB_COLS_BYTES;     // 32 KB
constexpr int K_ATOMS_PER_TILE = SUB_COLS_BF16 / 16;    // 4
constexpr int SPLIT_P_N    = K_TILE / 4 * 3;            // 96
constexpr int SPLIT_P_ATOM = SPLIT_P_N / 16;            // 6 (BMM2 atom at the split)
constexpr int SPLIT_P_COL  = SPLIT_P_N / 2;     // 48 (u32 P cols written before the empty_bar_spo signal)
constexpr int EX2_FRG_PAIRS = 16;          // 32 elts / fragment = 16 pairs
constexpr int EX2_FRG_CNT   = K_TILE / 32; // = 4 for K_TILE=128
// EX2_FREQ=10 uses more FFMA-emu than 1CTA's 16. Measured +1.2-1.8% (8~=10 plateau, 12~=16 lower):
// the 2SM softmax is MUFU/SFU-bound, so shifting exp2 off MUFU onto FMA wins.
constexpr int EX2_FREQ      = 10;          // FA4-2CTA ex2_emu_freq
constexpr int EX2_RES       = 4;           // FA4 ex2_emu_res (default)
constexpr int EX2_START_FRG = 1;           // FA4-2CTA ex2_emu_start_frg (fragment 0 pure-HW)
constexpr int NUM_KV_STAGES = 6;
constexpr int S_COLS = K_TILE;
constexpr int O_COLS = HEAD_DIM;
constexpr int TMEM_TOTAL = 512;
constexpr int W_CORR0 = 8, W_MMA = 12, W_EPI = 13, W_LOAD = 14, W_SCHED = 15;
constexpr int N_WARPS = 16;
constexpr int CLC_STAGES = 4;

extern __shared__ __align__(1024) uint8_t fmha_smem[];

template <bool Q_RASTER, bool IS_CAUSAL>
__device__ __forceinline__ void decode_workitem(
    int cluster_workitem_id, int peer, int seqlen, int num_kv_heads, 
    int packed_mpairs_per_seq, int packed_mpairs_per_sample, int q_tile_per_cta, 
    unsigned long long magic0, unsigned long long magic1, unsigned long long magic2,
    int& sample, int& h_kv, int& q_tile_base, int& k_tiles) {
  sample = (int)fdiv((unsigned)cluster_workitem_id, magic0);
  const int rr = cluster_workitem_id - sample * packed_mpairs_per_sample;
  int packed_mpairs_index;
  if constexpr (Q_RASTER) {
    h_kv                = (int)fdiv((unsigned)rr, magic1);
    packed_mpairs_index = rr - h_kv * packed_mpairs_per_seq;
  } else {
    packed_mpairs_index = (int)fdiv((unsigned)rr, magic2);
    h_kv                = rr - packed_mpairs_index * num_kv_heads;
  }
  q_tile_base = (2 * packed_mpairs_index + peer) * q_tile_per_cta;
  k_tiles = (seqlen + K_TILE - 1) / K_TILE;
  if constexpr (IS_CAUSAL) {
    const int cluster_max_q_base = (2 * packed_mpairs_index + 1) * q_tile_per_cta;
    const int causal_k_tiles_cap = (cluster_max_q_base + q_tile_per_cta - 1) / K_TILE + 1;
    if (causal_k_tiles_cap < k_tiles) k_tiles = causal_k_tiles_cap;
  }
}

// 64-bit SMEM descriptor; k-loop walks add to the LOW word only (hi/swizzle word is
// unchanged -- all slot/subtile/atom deltas stay < 2^14 desc units). Union keeps it one object.
union SmemDescPair { uint64_t u64; uint2 w; };

// Compile-time kernel config (template args, set in run()'s `constexpr` block):
//   S_LD_COLS        : cols per softmax tcgen05.ld of the S row (32/64 compile; 128 aborts ptxas).
//   FULL_NAMED_BAR   : softmax->corr "scale ready": true = HW named barrier (per-band), false =
//                      mbarrier (full_bar_alpha/full_bar_l). Both use alpha_and_l_smem.
//   EX2_EMU          : route a fraction of softmax exp2 through FFMA f32x2 emulation (vs MUFU.EX2).
//   SPLIT_P          : softmax publishes P in two chunks (96+32 keys); BMM2 starts on the first,
//                      full_bar_p_last gates the tail atoms.
//   SOFTMAX_THROTTLE : FA4 pacing -- corr defers releasing the alpha/l slot until after it consumes,
//                      holding softmax ~1 stage behind correction.
//   USE_CLC          : true = cluster CLC work-stealing sched (leader w15 -> CLC_STAGES tile ring;
//                      both CTAs' 16 warps release clc_empty, arrive=32); false = static (w15 idle).
//   Q_RASTER         : true = M-pair-innermost decode (same (sample, kv_head) packed-M pairs are
//                      consecutive -> hot L2 on K/V across stolen tiles); false = kv-head-innermost.
//   MHA              : true = HQ==HK, so gqa_group_size folds to 1 (q_tile_per_mtile = 128);
//                      false = GQA (runtime HQ/HK ratio).
//   IS_CAUSAL        : true = triangular causal mask + K-loop cap (uniform seqlen; still non-ragged).
template <int S_LD_COLS = 32, bool FULL_NAMED_BAR = false, bool EX2_EMU = false, bool SPLIT_P = true,
          bool SOFTMAX_THROTTLE = false, bool USE_CLC = true, bool Q_RASTER = true, bool MHA = false,
          bool IS_CAUSAL = false, int RESCALE_THRESHOLD = 8>
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(N_WARPS * 32, 1)
fmha_context_bf16_gen_kernel(const __grid_constant__ CUtensorMap tmap_q,
    const __grid_constant__ CUtensorMap tmap_k, const __grid_constant__ CUtensorMap tmap_v_t,
    const __grid_constant__ CUtensorMap tmap_o, int seqlen,
    int num_q_heads, int num_kv_heads, float scale_log2,
    int packed_mtiles_per_seq, int num_samples,
    unsigned long long magic0, unsigned long long magic1, unsigned long long magic2) {
  const int gqa_group_size = MHA ? 1 : (num_q_heads / num_kv_heads);
  const int q_tile_per_mtile = M_TILE / gqa_group_size;        // GQA(8): 16 / MHA: 128
  const int q_tile_per_cta   = M_TILES_PER_CTA * q_tile_per_mtile;     // GQA(8): 32 / MHA: 256

  // 2SM: a cluster's 2 CTAs share one (sample, kv_head): peer 0 owns even packed-M tiles, peer 1
  // odd; the joint MMA pairs peer0.mtile_i with peer1.mtile_i (256-row M); K/V is N-split per CTA.
  const int peer           = blockIdx.x & 1;
  const int cluster_id     = blockIdx.x >> 1;
  const int num_clusters   = gridDim.x >> 1;
  const int packed_mpairs_per_seq       = (packed_mtiles_per_seq + 1) >> 1;
  const int packed_mpairs_per_sample    = packed_mpairs_per_seq * num_kv_heads;
  const int total_workitems = num_samples * packed_mpairs_per_sample;
  (void)num_clusters; (void)total_workitems; (void)magic2; (void)packed_mpairs_per_seq;

  uint8_t* sQ0 = fmha_smem;
  uint8_t* sQ1 = sQ0 + Q_TILE_BYTES;
  uint8_t* sQ[2] = { sQ0, sQ1 };
  uint8_t* sKV = sQ1 + Q_TILE_BYTES;
  __nv_bfloat16* sO0 = reinterpret_cast<__nv_bfloat16*>(sKV + NUM_KV_STAGES * KV_SLOT_BYTES);
  __nv_bfloat16* sO1 = sO0 + M_TILE * HEAD_DIM;
  __nv_bfloat16* sO_bufs[2] = { sO0, sO1 };
  uint64_t* full_bar = reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(sO1) + M_TILE * HEAD_DIM * sizeof(__nv_bfloat16));
  uint64_t* empty_bar= full_bar + NUM_KV_STAGES;
  uint64_t* full_bar_q  = empty_bar + NUM_KV_STAGES;
  uint64_t* empty_bar_q   = full_bar_q + 2;
  uint64_t* full_bar_spo  = empty_bar_q + 2;
  uint64_t* empty_bar_spo = full_bar_spo + 2;
  uint64_t* full_bar_o_acc   = empty_bar_spo + 2;
  uint64_t* full_bar_alpha = full_bar_o_acc + 2;
  uint64_t* full_bar_l   = full_bar_alpha + 2;
  uint64_t* full_bar_p_last    = full_bar_l + 2;
  uint64_t* empty_bar_alpha_and_l = full_bar_p_last + 2;
  uint64_t* full_bar_o_epi  = empty_bar_alpha_and_l + 2;
  uint64_t* empty_bar_o_epi = full_bar_o_epi + 2;
  uint64_t* clc_full  = empty_bar_o_epi + 2;
  uint64_t* clc_empty = clc_full + CLC_STAGES;
  uint32_t* clc_response = reinterpret_cast<uint32_t*>(
      (reinterpret_cast<uintptr_t>(clc_empty + CLC_STAGES) + 15u) & ~uintptr_t(15u));
  uint32_t* tmem_slot = clc_response + CLC_STAGES * 4;
  float* alpha_and_l_smem = reinterpret_cast<float*>(tmem_slot + 2);
  // NOTE: 1CTA's wake-granule isolation of empty_bar_alpha_and_l does NOT port here -- measured it
  // INVERTS on 2SM (r1 +10 but r2 -25.5, r3 -27.5), losing the throughput rows 2SM exists for. Left out.

  // Keep the array. 1CTA computes sKV + kv_stage*BYTES on the fly (no LDL, +11 r2/+20 r4: it is
  // scoreboard-bound); 2SM has wait slack so that form is off the critical path (measured r2 -3/r4 -8).
  uint8_t* smem_kv[NUM_KV_STAGES];
  for (int s = 0; s < NUM_KV_STAGES; ++s) smem_kv[s] = sKV + s * KV_SLOT_BYTES;


  const int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;
  WpCtx wpc = wp_ctx_init();

  if (warp_id == 0) {
    tcgen05_alloc<2>(smem_ptr_u32(tmem_slot), TMEM_TOTAL);
    tcgen05_relinquish_alloc_permit<2>();
  }

  if (tid < NUM_KV_STAGES) {
    mbarrier_init(smem_ptr_u32(&full_bar[tid]), 1);
    mbarrier_init(smem_ptr_u32(&empty_bar[tid]), 1);
  } else if (tid >= 32 && tid < 34) {
    const int i = tid - 32;
    mbarrier_init(smem_ptr_u32(&full_bar_q[i]), 1);
    mbarrier_init(smem_ptr_u32(&empty_bar_q[i]), 1);
    mbarrier_init(smem_ptr_u32(&full_bar_l[i]), 128);
    mbarrier_init(smem_ptr_u32(&full_bar_spo[i]), 1);
    mbarrier_init(smem_ptr_u32(&empty_bar_spo[i]), 512);         // 2SM: BOTH CTAs' 4 softmax + 4 corr warps
  } else if (tid >= 64 && tid < 66) {
    const int i = tid - 64;
    mbarrier_init(smem_ptr_u32(&full_bar_o_acc[i]), 1);
    mbarrier_init(smem_ptr_u32(&full_bar_alpha[i]), 128);
    mbarrier_init(smem_ptr_u32(&empty_bar_alpha_and_l[i]), 128);
    mbarrier_init(smem_ptr_u32(&full_bar_p_last[i]), 256);           // 2SM: BOTH CTAs' 4 softmax warps
    mbarrier_init(smem_ptr_u32(&full_bar_o_epi[i]), 128);
    mbarrier_init(smem_ptr_u32(&empty_bar_o_epi[i]), 1);
  }
  if (tid == 96) {
    if constexpr (USE_CLC) {
      #pragma unroll
      for (int s = 0; s < CLC_STAGES; ++s) {
        mbarrier_init(smem_ptr_u32(&clc_full[s]), 1);
        mbarrier_init(smem_ptr_u32(&clc_empty[s]), N_WARPS * 2);
      }
      #pragma unroll
      for (int i = 0; i < CLC_STAGES * 4; ++i)
        clc_response[i] = 0;
    }
  }
  fence_mbarrier_init_release_cluster();
  __syncthreads();
  const uint32_t tmem_base = *tmem_slot;
  // 2SM: both CTAs must finish alloc + mbarrier init before any cross-CTA MMA/mbarrier use.
  barrier_cluster_arrive_relaxed_aligned();   // control-only join: no release fence
  barrier_cluster_wait_aligned();             // (FA4 has no MEMBAR.ALL pair here)

  if (warp_id == W_LOAD) {
    setmaxnreg_dec<72>();

    EmptyPhaseTracker<NUM_KV_STAGES> kv_empty_ph;
    EmptyPhaseTracker<1> q_empty_ph;
    int cluster_workitem_id = cluster_id;
    [[maybe_unused]] int clc_stage = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;

    while (true) {
      int sample, h_kv, q_tile_base, K_TILES;
      decode_workitem<Q_RASTER, IS_CAUSAL>(cluster_workitem_id, peer, seqlen, num_kv_heads, packed_mpairs_per_seq,
          packed_mpairs_per_sample, q_tile_per_cta, magic0, magic1, magic2,
          sample, h_kv, q_tile_base, K_TILES);
      // clamp padding peer's OOB tile to a dummy in-bounds Q (can't bail).
      const int q_tile_base_safe = (q_tile_base < seqlen) ? q_tile_base : 0;
      // padded K/V token base (driver lays samples at seqlen_pad stride; K_TILE multiple)
      const int k_start = sample * (((seqlen + 127) >> 7) << 7);

      for (int k = 0; k < K_TILES; ++k) {
        const int k_offset = (K_TILES - 1 - k) * K_TILE;
        int kv_stage = kv_empty_ph.get_stage();

        wp_begin(wpc, WP_LOAD_WAIT);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar[kv_stage]), kv_empty_ph.get_phase());
        kv_empty_ph.advance();
        wp_end(wpc, WP_LOAD_WAIT);

        wp_begin(wpc, WP_LOAD_ISSUE_K);
        const uint32_t kbar = tma_peer_bit_mask(smem_ptr_u32(&full_bar[kv_stage]));
        const uint32_t kdst = smem_ptr_u32(smem_kv[kv_stage]);
        const int      k_token     = k_start + k_offset + peer * (K_TILE / 2);
        const int      k_head_atom = h_kv * K_SUBTILES;
        if (elect_one_sync()) {
          if (peer == 0) mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar[kv_stage]), K_TILE_BYTES);
          tma_load_3d_2sm(kdst, &tmap_k, kbar, 0, k_token, k_head_atom);
        }
        wp_end(wpc, WP_LOAD_ISSUE_K);

        if (k == 0) {
          #pragma unroll
          for (int m = 0; m < M_TILES_PER_CTA; ++m) {
            wp_begin(wpc, WP_LOAD_WAIT);
            mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_q[m]), q_empty_ph.get_phase());
            wp_end(wpc, WP_LOAD_WAIT);
            wp_begin(wpc, WP_LOAD_ISSUE_Q);
            const uint32_t qbar = tma_peer_bit_mask(smem_ptr_u32(&full_bar_q[m]));
            const int q_token = q_tile_base_safe + m * q_tile_per_mtile;
            const int q_head  = h_kv * gqa_group_size;
            if (elect_one_sync()) {
              if (peer == 0) mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar_q[m]), 2 * Q_TILE_BYTES);
              tma_load_5d_2sm(smem_ptr_u32(sQ[m]), &tmap_q, qbar, 0, q_head, q_token, sample, 0);
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
        const uint32_t vbar = tma_peer_bit_mask(smem_ptr_u32(&full_bar[kv_stage]));
        const uint32_t vdst = smem_ptr_u32(smem_kv[kv_stage]);
        const int      v_head_row  = h_kv * HEAD_DIM + peer * (HEAD_DIM / 2);
        const int      v_token_atom = (k_start + k_offset) / SUB_COLS_BF16;
        if (elect_one_sync()) {
          if (peer == 0) mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar[kv_stage]), V_TILE_BYTES);
          tma_load_3d_2sm(vdst, &tmap_v_t, vbar, 0, v_head_row, v_token_atom);
        }
        wp_end(wpc, WP_LOAD_ISSUE_V);
      }

      if constexpr (USE_CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 2, ClcRasterOrder::AlongN, /*CTA_GROUP=*/2, /*SUSPEND=*/true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, /*do_release=*/elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        if (!next.valid) break;
        cluster_workitem_id = next.n_tile;
      } else {
        cluster_workitem_id += num_clusters;
        if (cluster_workitem_id >= total_workitems) break;
      }
    }
  }
  else if (warp_id == W_MMA) {
    setmaxnreg_dec<72>();

    const bool lead = elect_one_sync();
    constexpr uint32_t DESC_SBO = 1024, DESC_LBO = 16;
    // TMEM: S[i] at i*128, O[i] at 256+i*128 (512 cols).
    // 2SM idesc: M = M_TILE_CLUSTER (256), N = 128. A M-split (each CTA = its 128-row half),
    // B N-split (each = its 64-wide half); MMA reads both peers.
    const uint32_t idesc_qk = make_idesc_bf16_f32(M_TILE_CLUSTER, K_TILE, false, false);
    const uint32_t idesc_pv = make_idesc_bf16_f32(M_TILE_CLUSTER, HEAD_DIM,  false, false);
    const uint64_t desc_q0  = build_smem_desc_blackwell(smem_ptr_u32(sQ0), DESC_SBO, DESC_LBO, SmemSwizzleBlackwell::B128);
    const uint64_t desc_kv0 = build_smem_desc_blackwell(smem_ptr_u32(sKV), DESC_SBO, DESC_LBO, SmemSwizzleBlackwell::B128);
    // >> 4: the SMEM descriptor address field is in 16-byte units (addr >> 4).
    constexpr uint64_t KV_DESC_DELTA      = KV_SLOT_BYTES >> 4;
    constexpr uint64_t SUB_DESC_DELTA_Q   = Q_SUB_COLS_BYTES >> 4;
    constexpr uint64_t SUB_DESC_DELTA_KV  = (Q_SUB_COLS_BYTES / 2) >> 4;
    constexpr uint64_t Q_MTILE_DESC_DELTA = Q_TILE_BYTES >> 4;

    PhaseTracker<NUM_KV_STAGES> kv_ph;
    PhaseTracker<1> q_ph;
    PhaseTracker<1> spo_ph;
    int cluster_workitem_id = cluster_id;
    [[maybe_unused]] int clc_stage = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;
    while (true) {
      int sample, h_kv, q_tile_base, K_TILES;
      decode_workitem<Q_RASTER, IS_CAUSAL>(cluster_workitem_id, peer, seqlen, num_kv_heads, packed_mpairs_per_seq,
          packed_mpairs_per_sample, q_tile_per_cta, magic0, magic1, magic2,
          sample, h_kv, q_tile_base, K_TILES);
      (void)sample; (void)h_kv;
      // peer 0's even tile is always in range, so the leader never bails.
      (void)q_tile_base;

      if (peer == 0) {
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
            SmemDescPair da, db;
            da.u64 = desc_q0;  da.w.x += (uint32_t)(i * Q_MTILE_DESC_DELTA);
            db.u64 = desc_kv0; db.w.x += (uint32_t)(kv_stage * (int)KV_DESC_DELTA);

            #pragma unroll
            for (int s = 0; s < Q_SUBTILES; ++s) {
              #pragma unroll
              for (int ki = 0; ki < K_ATOMS_PER_TILE; ++ki) {
                const bool enable_d = (s != 0) || (ki != 0);
                tcgen05_mma_f16_ss<2>(s_tmem_addr, da.u64, db.u64, idesc_qk, enable_d);
                da.w.x += 2; db.w.x += 2;
              }
              da.w.x += (uint32_t)(SUB_DESC_DELTA_Q  - 2 * K_ATOMS_PER_TILE);
              db.w.x += (uint32_t)(SUB_DESC_DELTA_KV - 2 * K_ATOMS_PER_TILE);
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
            SmemDescPair dbv;
            dbv.u64 = desc_kv0;
            dbv.w.x += (uint32_t)(kv_stage * (int)KV_DESC_DELTA);

            #pragma unroll
            for (int s = 0; s < V_SUBTILES; ++s) {
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
                  tcgen05_mma_f16_ts_2sm(o_tmem_addr, s_tmem_addr + (uint32_t)(a * 8), dbv.u64, idesc_pv, accumulate, 0, 0, 0, 0, 0, 0, 0, 0);
                }
                dbv.w.x += 2;
              }
              dbv.w.x += (uint32_t)(SUB_DESC_DELTA_KV - 2 * K_ATOMS_PER_TILE);
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
              tcgen05_commit_multicast<2>(smem_ptr_u32(&empty_bar[kv_stage]), 0x3);
              wp_end(wpc, WP_MMA_COMMIT);
            }

            // BMM1(next): Q@K -> S
            wp_begin(wpc, WP_MMA_ISSUE);
            if (lead) {
              SmemDescPair da, db;
              da.u64 = desc_q0;  da.w.x += (uint32_t)(i * Q_MTILE_DESC_DELTA);
              db.u64 = desc_kv0; db.w.x += (uint32_t)(kv_stage_next * (int)KV_DESC_DELTA);
              #pragma unroll
              for (int s = 0; s < Q_SUBTILES; ++s) {
                #pragma unroll
                for (int ki = 0; ki < K_ATOMS_PER_TILE; ++ki) {
                  const bool enable_d = (s != 0) || (ki != 0);
                  tcgen05_mma_f16_ss<2>(s_tmem_addr, da.u64, db.u64, idesc_qk, enable_d);
                  da.w.x += 2; db.w.x += 2;
                }
                da.w.x += (uint32_t)(SUB_DESC_DELTA_Q  - 2 * K_ATOMS_PER_TILE);
                db.w.x += (uint32_t)(SUB_DESC_DELTA_KV - 2 * K_ATOMS_PER_TILE);
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
          SmemDescPair dbv;
          dbv.u64 = desc_kv0;
          dbv.w.x += (uint32_t)(kv_stage * (int)KV_DESC_DELTA);
          #pragma unroll
          for (int s = 0; s < V_SUBTILES; ++s) {
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
                tcgen05_mma_f16_ts_2sm(o_tmem_addr, s_tmem_addr + (uint32_t)(a * 8), dbv.u64, idesc_pv, accumulate, 0, 0, 0, 0, 0, 0, 0, 0);
              }
              dbv.w.x += 2;
            }
            dbv.w.x += (uint32_t)(SUB_DESC_DELTA_KV - 2 * K_ATOMS_PER_TILE);
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
      }  // end if (peer == 0)

      if constexpr (USE_CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 2, ClcRasterOrder::AlongN, /*CTA_GROUP=*/2, /*SUSPEND=*/true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, /*do_release=*/elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        if (!next.valid) break;
        cluster_workitem_id = next.n_tile;
      } else {
        cluster_workitem_id += num_clusters;
        if (cluster_workitem_id >= total_workitems) break;
      }
    }
  }
  else if (warp_id == W_EPI) {
    setmaxnreg_dec<72>();

    PhaseTracker<1> full_o_ph;
    // Prime empty_bar_o_epi once: corr's first sO pack must not block (no prior store in flight).
    if (elect_one_sync()) {
      #pragma unroll
      for (int m = 0; m < M_TILES_PER_CTA; ++m)
        mbarrier_arrive(smem_ptr_u32(&empty_bar_o_epi[m]));
    }
    int cluster_workitem_id = cluster_id;
    [[maybe_unused]] int clc_stage = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;
    while (true) {
      int sample, h_kv, q_tile_base, K_TILES;
      decode_workitem<Q_RASTER, IS_CAUSAL>(cluster_workitem_id, peer, seqlen, num_kv_heads, packed_mpairs_per_seq,
          packed_mpairs_per_sample, q_tile_per_cta, magic0, magic1, magic2,
          sample, h_kv, q_tile_base, K_TILES);
      // padding peer waits in lockstep but does NOT store.
      const bool valid = (q_tile_base < seqlen);

      #pragma unroll
      for (int m = 0; m < M_TILES_PER_CTA; ++m) {
        wp_begin(wpc, WP_EPI_WAIT_TMEM);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_o_epi[m]), full_o_ph.get_phase());
        wp_end(wpc, WP_EPI_WAIT_TMEM);

        wp_begin(wpc, WP_EPI_STORE);
        if (valid && elect_one_sync()) {
          const int q_token = q_tile_base + m * q_tile_per_mtile;
          // 4D map (token-in-sample dim): rows past seqlen are clamped away, not
          // written into the next sample's tokens.
          #pragma unroll
          for (int s = 0; s < Q_SUBTILES; ++s) {
            tma_store_4d(&tmap_o, s * SUB_COLS_BF16, h_kv * gqa_group_size, q_token, sample,
                         smem_ptr_u32(reinterpret_cast<const uint8_t*>(sO_bufs[m]) + s * Q_SUB_COLS_BYTES));
          }
          cp_async_bulk_commit_group();
        }
        wp_end(wpc, WP_EPI_STORE);
      }

      wp_begin(wpc, WP_EPI_WAIT_STORE);
      // Drain per-m commit groups; release each sO slot to corr as ITS store completes
      // (without this, corr's next pack races the in-flight store at tiny causal K-loops).
      // Runs on the padding peer too (no groups pending -> waits are no-ops; corr packs in lockstep).
      if (elect_one_sync()) {
        cp_async_bulk_wait_group_read<1>();
        mbarrier_arrive(smem_ptr_u32(&empty_bar_o_epi[0]));
        cp_async_bulk_wait_group_read<0>();
        mbarrier_arrive(smem_ptr_u32(&empty_bar_o_epi[1]));
      }
      wp_end(wpc, WP_EPI_WAIT_STORE);

      full_o_ph.advance();
      if constexpr (USE_CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 2, ClcRasterOrder::AlongN, /*CTA_GROUP=*/2, /*SUSPEND=*/true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, /*do_release=*/elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        if (!next.valid) break;
        cluster_workitem_id = next.n_tile;
      } else {
        cluster_workitem_id += num_clusters;
        if (cluster_workitem_id >= total_workitems) break;
      }
    }
  }
  else if (warp_id == W_SCHED) {
    setmaxnreg_dec<72>();

    if constexpr (USE_CLC) {
      int prod_stage = 0; uint32_t prod_phase = 1;
      int cons_stage = 0; uint32_t cons_phase = 0;
      const bool leader = (peer == 0);
      while (true) {
        if (leader) {
          if (lane == 0)
            mbarrier_wait_parity_suspend(smem_ptr_u32(&clc_empty[prod_stage]), prod_phase);
          __syncwarp();
          clc_arrive_expect_tx_cluster(smem_ptr_u32(&clc_full[prod_stage]), /*tx_bytes=*/16);
          if (lane == 0)
            clc_try_cancel_multicast_all(smem_ptr_u32(&clc_response[prod_stage * 4]),
                                         smem_ptr_u32(&clc_full[prod_stage]));
          advance_stage_phase<CLC_STAGES>(prod_stage, prod_phase);
        }
        ClcTileInfo next = clc_fetch_next_tile<1, 2, ClcRasterOrder::AlongN, /*CTA_GROUP=*/2, /*SUSPEND=*/true>(
            clc_full, clc_empty, clc_response, cons_stage, cons_phase, /*do_release=*/elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(cons_stage, cons_phase);
        if (!next.valid) break;
      }
      // Tail drain (leader only): absorb the in-flight consumer releases before kernel exit.
      if (leader) {
        for (int s = 0; s < CLC_STAGES; ++s) {
          if (lane == 0)
            mbarrier_wait_parity_suspend(smem_ptr_u32(&clc_empty[prod_stage]), prod_phase);
          __syncwarp();
          advance_stage_phase<CLC_STAGES>(prod_stage, prod_phase);
        }
      }
    }
  }
  else if (warp_id >= W_CORR0 && warp_id < W_MMA) {
    setmaxnreg_dec<88>();

    const int corr_warp_id = warp_id - W_CORR0;
    [[maybe_unused]] PhaseTracker<1> alpha_ph;
    PhaseTracker<1> o_acc_ph;
    PhaseTracker<1> o_epi_empty_ph;

    // Prime the return barriers once (first BMM2 / first softmax stat write).
    #pragma unroll
    for (int i = 0; i < M_TILES_PER_CTA; ++i) {
      mbarrier_arrive_cluster_default(mapa_shared_cluster_u32(smem_ptr_u32(&empty_bar_spo[i]), 0));
      mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
    }

    int cluster_workitem_id = cluster_id;
    [[maybe_unused]] int clc_stage = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;
    while (true) {
      int sample, h_kv, q_tile_base, K_TILES;
      decode_workitem<Q_RASTER, IS_CAUSAL>(cluster_workitem_id, peer, seqlen, num_kv_heads, packed_mpairs_per_seq,
          packed_mpairs_per_sample, q_tile_per_cta, magic0, magic1, magic2,
          sample, h_kv, q_tile_base, K_TILES);
      (void)h_kv; (void)q_tile_base;  // 2SM: the unpaired padding peer runs in lockstep (no bail)

      // block 0: no rescale (no prior O); consume alpha + release the scale slot.
      #pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i) {
        wp_begin(wpc, WP_CORR_WAIT);
        if constexpr (FULL_NAMED_BAR) full_bar_wait(i, corr_warp_id);
        else mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_alpha[i]), alpha_ph.get_phase());
        // FA4 cross-stage deferral (fa4_gen correction_loop): prologue releases only slot 0.
        if constexpr (SOFTMAX_THROTTLE) {
          if (i == 0) mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[0]));
        } else {
          mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
        }
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
            const uint32_t o_tmem_addr = tmem_base + (uint32_t)(2 * S_COLS + i * O_COLS) + ((uint32_t)(corr_warp_id * 32) << 16);
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
          // FA4: release the OTHER tile's scale slot (cross-stage deferred release).
          if constexpr (SOFTMAX_THROTTLE) mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[M_TILES_PER_CTA - 1 - i]));
          mbarrier_arrive_cluster_default(mapa_shared_cluster_u32(smem_ptr_u32(&empty_bar_spo[i]), 0));
          wp_end(wpc, WP_CORR_O_SCALE);
        }
        if constexpr (!FULL_NAMED_BAR) alpha_ph.advance();
      }
      // FA4 post-loop rebalance: deferral left slot 1 one release short; pay it so l-publish can proceed.
      if constexpr (SOFTMAX_THROTTLE) mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[M_TILES_PER_CTA - 1]));

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
        // epilogue releases same-stage, early (FA4 fa4_gen.py:2041)
        mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
        float inv_l = (l > 0.f) ? rcp_approx_ftz_f32(l) : 0.f;
        const float2 inv_l2 = f32x2_splat(inv_l);
        const uint32_t o_tmem_addr = tmem_base + (uint32_t)(2 * S_COLS + i * O_COLS) + ((uint32_t)(corr_warp_id * 32) << 16);
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

        mbarrier_arrive_cluster_default(mapa_shared_cluster_u32(smem_ptr_u32(&empty_bar_spo[i]), 0));

        // order this thread's st.shared writes (generic proxy) before TMA store (async proxy)
        fence_proxy_async_shared();

        mbarrier_arrive(smem_ptr_u32(&full_bar_o_epi[i]));
        wp_end(wpc, WP_CORR_EPI);
      }
      o_acc_ph.advance();
      o_epi_empty_ph.advance();

      if constexpr (USE_CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 2, ClcRasterOrder::AlongN, /*CTA_GROUP=*/2, /*SUSPEND=*/true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, /*do_release=*/elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        if (!next.valid) break;
        cluster_workitem_id = next.n_tile;
      } else {
        cluster_workitem_id += num_clusters;
        if (cluster_workitem_id >= total_workitems) break;
      }
    }
  }
  else {
    setmaxnreg_inc<176>();

    // warp-uniform hint (FA4 make_warp_uniform): R2UR.BROADCAST -> promotes m_tile + derived to URs.
    const int warp_id_u = __shfl_sync(0xffffffffu, warp_id, 0);
    const int m_tile = warp_id_u < 4 ? 0 : 1;
    const int warp_in_group = warp_id_u & 3;
    const int row_in_m_tile = warp_in_group * 32 + lane;
    const uint32_t s_tmem_addr = tmem_base + (uint32_t)(m_tile * S_COLS) + ((uint32_t)(warp_in_group * 32) << 16);
    PhaseTracker<1> spo_ph;
    PhaseTracker<1> scale_empty_ph;

    int cluster_workitem_id = cluster_id;
    [[maybe_unused]] int clc_stage = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;
    while (true) {
      int sample, h_kv, q_tile_base, K_TILES;
      decode_workitem<Q_RASTER, IS_CAUSAL>(cluster_workitem_id, peer, seqlen, num_kv_heads, packed_mpairs_per_seq,
          packed_mpairs_per_sample, q_tile_per_cta, magic0, magic1, magic2,
          sample, h_kv, q_tile_base, K_TILES);
      (void)sample; (void)h_kv;   // 2SM: the unpaired padding peer runs in lockstep (no bail)

      // causal: this row's query-token position (pack-GQA qh-inner); unused when IS_CAUSAL=false.
      const int q_pos = q_tile_base + m_tile * q_tile_per_mtile + row_in_m_tile / gqa_group_size;
      float m_run = -INFINITY, l_run = 0.f;
      wp_begin(wpc, WP_SM_WAIT_SCALE);
      mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_alpha_and_l[m_tile]), scale_empty_ph.get_phase());
      scale_empty_ph.advance();
      wp_end(wpc, WP_SM_WAIT_SCALE);
      float* const alpha_slot = &alpha_and_l_smem[m_tile * M_TILE + row_in_m_tile];
      // FA4 softmax_loop: peeled masked steps + unmasked steady loop (compile-time step specializations).
      // CAUSAL: each softmax warp masks every tile from k=0 through its own diagonal (see fmha-fill-degenerate).
      auto softmax_step = [&](int k, auto masked_c, auto is_first_c) {
        constexpr bool MASKED   = decltype(masked_c)::value;
        constexpr bool IS_FIRST = decltype(is_first_c)::value;
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

        // k==0 (last/diagonal K-tile) is the only masked block (mask keys >= seqlen; causal: also keys > q_pos).
        if constexpr (MASKED) mask_s_row_r2p<IS_CAUSAL, K_TILE>(scores, k_offset, q_pos, seqlen);

        float rmax0 = m_run, rmax1 = -INFINITY, rmax2 = -INFINITY, rmax3 = -INFINITY;
        #pragma unroll
        for (int j = 0; j < K_TILE; j += 8) {
          rmax0 = fmaxf(fmaxf(rmax0, scores[j + 0]), scores[j + 1]);
          rmax1 = fmaxf(fmaxf(rmax1, scores[j + 2]), scores[j + 3]);
          rmax2 = fmaxf(fmaxf(rmax2, scores[j + 4]), scores[j + 5]);
          rmax3 = fmaxf(fmaxf(rmax3, scores[j + 6]), scores[j + 7]);
        }
        float new_m = fmaxf(fmaxf(rmax0, rmax1), fmaxf(rmax2, rmax3));
        float row_max_safe = (new_m == -INFINITY) ? 0.0f : new_m;
        float alpha = 0.0f;
        if constexpr (!IS_FIRST) {
          const float acc_scale_ = (m_run - row_max_safe) * scale_log2;
          alpha = ex2_approx_f32(acc_scale_);
          if (acc_scale_ >= -(float)RESCALE_THRESHOLD) {
            new_m = m_run; row_max_safe = m_run; alpha = 1.0f;
          }
          *alpha_slot = alpha;
        }
        if constexpr (FULL_NAMED_BAR) full_bar_arrive(m_tile, warp_in_group);
        else mbarrier_arrive(smem_ptr_u32(&full_bar_alpha[m_tile]));

        // scale_subtract_rowmax + apply_exp2_convert (FA4 form): exp2 IN PLACE on the
        // f32 scores; bf16 pack per 32-elt fragment AFTER that fragment's exp2s.
        // Row-sum is NOT computed here -- FA4 defers it past the P stores AND the
        // wait_scale below, so the S->P critical path carries no row-sum FADD2s.
        const float2 scale2 = f32x2_splat(scale_log2);
        const float2 neg_m_scaled2 = f32x2_splat(-row_max_safe * scale_log2);
        uint32_t p_regs[K_TILE / 2];
        #pragma unroll
        for (int jj = 0; jj < EX2_FRG_CNT; ++jj) {
          #pragma unroll
          for (int cc = 0; cc < EX2_FRG_PAIRS; ++cc) {
            const int c = jj * EX2_FRG_PAIRS + cc;
            const float2 a2 = ffma2(scores2[c], scale2, neg_m_scaled2);
            if constexpr (EX2_EMU) {
              const int kk = 2 * cc;
              const bool use_hw = (kk % EX2_FREQ < EX2_FREQ - EX2_RES) || (jj >= EX2_FRG_CNT - 1) || (jj < EX2_START_FRG);
              scores2[c] = use_hw ? make_float2(ex2_approx_f32(a2.x), ex2_approx_f32(a2.y))
                                  : ex2_emu_f32x2(a2.x, a2.y);
            } else {
              scores2[c] = make_float2(ex2_approx_f32(a2.x), ex2_approx_f32(a2.y));
            }
            p_regs[c] = cvt_f32x2_to_bf16x2(scores2[c].x, scores2[c].y);
          }
        }
        uint32_t p_tmem_addr = s_tmem_addr;
        wp_end(wpc, WP_SM_SOFTMAX);

        wp_begin(wpc, WP_SM_STORE_P);
        if constexpr (SPLIT_P) {
          tcgen05_st_32x32b_x32(p_tmem_addr +  0, *reinterpret_cast<uint32_t(*)[32]>(&p_regs[0]));
          tcgen05_st_32x32b_x16(p_tmem_addr + 32, *reinterpret_cast<uint32_t(*)[16]>(&p_regs[32]));
          // fence orders the async STTM but does NOT wait; wait::st stops the mma reading pre-store garbage.
          tcgen05_wait_st();
          tcgen05_fence_before_thread_sync();
          mbarrier_arrive_cluster_default(mapa_shared_cluster_u32(smem_ptr_u32(&empty_bar_spo[m_tile]), 0));
          tcgen05_st_32x32b_x16(p_tmem_addr + SPLIT_P_COL, *reinterpret_cast<uint32_t(*)[16]>(&p_regs[SPLIT_P_COL]));
          tcgen05_wait_st();
          tcgen05_fence_before_thread_sync();
          mbarrier_arrive_cluster_default(mapa_shared_cluster_u32(smem_ptr_u32(&full_bar_p_last[m_tile]), 0));
        } else {
          tcgen05_st_32x32b_x32(p_tmem_addr, *reinterpret_cast<uint32_t(*)[32]>(&p_regs[0]));
          tcgen05_st_32x32b_x32(p_tmem_addr + 32, *reinterpret_cast<uint32_t(*)[32]>(&p_regs[32]));
          tcgen05_wait_st();
          tcgen05_fence_before_thread_sync();
          mbarrier_arrive_cluster_default(mapa_shared_cluster_u32(smem_ptr_u32(&empty_bar_spo[m_tile]), 0));
        }
        wp_end(wpc, WP_SM_STORE_P);

        spo_ph.advance();
        wp_begin(wpc, WP_SM_WAIT_SCALE);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_alpha_and_l[m_tile]), scale_empty_ph.get_phase());
        scale_empty_ph.advance();
        wp_end(wpc, WP_SM_WAIT_SCALE);
        // deferred update_row_sum. Intent: overlap this chain --
        // corr's O-rescale + wait_scale + this row-sum -- with BMM2 on the MMA warp.
        {
          float2 lt2a = make_float2(IS_FIRST ? 0.0f : l_run * alpha, 0.0f);
          float2 lt2b = make_float2(0.f, 0.f), lt2c = make_float2(0.f, 0.f), lt2d = make_float2(0.f, 0.f);
          #pragma unroll
          for (int c = 0; c < K_TILE / 2; c += 4) {
            lt2a = fadd2(lt2a, scores2[c + 0]);
            lt2b = fadd2(lt2b, scores2[c + 1]);
            lt2c = fadd2(lt2c, scores2[c + 2]);
            lt2d = fadd2(lt2d, scores2[c + 3]);
          }
          const float2 lt2 = fadd2(fadd2(lt2a, lt2b), fadd2(lt2c, lt2d));
          l_run = lt2.x + lt2.y;
        }
        m_run = new_m;
      };
      softmax_step(0, std::true_type{}, std::true_type{});
      int k = 1;
      if constexpr (IS_CAUSAL) {
        const int m_tile_tok0   = q_tile_base + m_tile * q_tile_per_mtile;
        const int masked_steps  = K_TILES - (m_tile_tok0 >> 7);
        for (; k < masked_steps; ++k) softmax_step(k, std::true_type{}, std::false_type{});
      }
      for (; k < K_TILES; ++k) softmax_step(k, std::false_type{}, std::false_type{});
      wp_begin(wpc, WP_SM_READ_L);
      alpha_and_l_smem[m_tile * M_TILE + row_in_m_tile] = l_run;
      if constexpr (FULL_NAMED_BAR) full_bar_arrive(m_tile, warp_in_group);
      else mbarrier_arrive(smem_ptr_u32(&full_bar_l[m_tile]));
      wp_end(wpc, WP_SM_READ_L);

      if constexpr (USE_CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 2, ClcRasterOrder::AlongN, /*CTA_GROUP=*/2, /*SUSPEND=*/true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, /*do_release=*/elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        if (!next.valid) break;
        cluster_workitem_id = next.n_tile;
      } else {
        cluster_workitem_id += num_clusters;
        if (cluster_workitem_id >= total_workitems) break;
      }
    }
  }
  wp_flush(wpc);
  __syncthreads();
  barrier_cluster_arrive_relaxed_aligned();   // control-only join: no release fence
  barrier_cluster_wait_aligned();             // (FA4 has no MEMBAR.ALL pair here)
  if (warp_id == 0) tcgen05_dealloc<2>(tmem_base, TMEM_TOTAL);
}

// ============================== driver ====================================

#include "fmha_cpu_ref.cuh"
#include "fmha_context_bf16_benchmark.cuh"

// Deterministic fill in [-1, 1) so host and device see identical inputs.
static void fillr(__nv_bfloat16* h, long n, unsigned seed) {
  const char* e = getenv("FILL");
  int mode = e ? atoi(e) : 2;
  for (long i = 0; i < n; ++i) {
    uint32_t x = (uint32_t)i * 2654435761u + seed;
    float v;
    switch (mode) {
      case 1: v = ((x % 256) / 256.0f) - 0.5f; break;
      case 3: v = 1.0f; break;
      case 4: v = (float)((int)(x % 7) - 3); break;
      default: v = (x % 2048) / 1024.0f - 1.0f; break;
    }
    h[i] = __float2bfloat16(v);
  }
}

// One benchmark shape. sl[s] = seqlen of sample s (all equal here); nqh/nkh =
// q/kv head counts; hd = head dim; causal toggles the mask; lab is for printing.
struct Sh {
  std::vector<int> sl;
  int  nqh, nkh, hd;
  bool causal;
  const char* lab;
};

template <bool MHA = false, bool IS_CAUSAL = false>
static double run(const Sh& sh, bool verify) {
  const int  ns     = (int)sh.sl.size();
  const int  seqlen = sh.sl[0];
  const long tq     = (long)ns * seqlen;      // total q-tokens
  const long tk     = tq;                     // total k-tokens (== tq here)
  // FA4-form single-TMA V needs 64-aligned sample K/V bases: pad each sample's K/V
  // token span to a K_TILE multiple (pad tokens zero-filled; masked by seqlen anyway).
  const int  seqlen_pad = ((seqlen + 127) / 128) * 128;
  const long tk_pad     = (long)ns * seqlen_pad;

  // ---- device buffers (bf16; V stored transposed as V_T for the BMM2 TMA) ----
  __nv_bfloat16 *dQ, *dK, *dVT, *dO;
  CUDA_CHECK(cudaMalloc(&dQ,  tq * sh.nqh * sh.hd * 2));
  CUDA_CHECK(cudaMalloc(&dK,  tk_pad * sh.nkh * sh.hd * 2));
  CUDA_CHECK(cudaMalloc(&dVT, (long)sh.nkh * sh.hd * tk_pad * 2));
  CUDA_CHECK(cudaMalloc(&dO,  tq * sh.nqh * sh.hd * 2));

  // ---- host inputs + V -> V_T transpose ([tok,head,hd] -> [head,hd,tok]) ----
  std::vector<__nv_bfloat16> hQ(tq * sh.nqh * sh.hd),
                             hK(tk * sh.nkh * sh.hd),
                             hV(tk * sh.nkh * sh.hd);
  fillr(hQ.data(), hQ.size(), 11);
  fillr(hK.data(), hK.size(), 22);
  fillr(hV.data(), hV.size(), 33);

  // padded-layout device images: sample s's tokens at base s*seqlen_pad (pads zeroed)
  std::vector<__nv_bfloat16> hKp((long)tk_pad * sh.nkh * sh.hd, __float2bfloat16(0.f));
  std::vector<__nv_bfloat16> hVT((long)sh.nkh * sh.hd * tk_pad, __float2bfloat16(0.f));
  for (int sm = 0; sm < ns; ++sm)
    for (int t = 0; t < seqlen; ++t) {
      const long src = (long)(sm * seqlen + t), dst = (long)sm * seqlen_pad + t;
      for (int h = 0; h < sh.nkh; ++h)
        for (int d = 0; d < sh.hd; ++d) {
          hKp[(dst * sh.nkh + h) * sh.hd + d]   = hK[(src * sh.nkh + h) * sh.hd + d];
          hVT[(h * sh.hd + d) * tk_pad + dst]   = hV[(src * sh.nkh + h) * sh.hd + d];
        }
    }

  CUDA_CHECK(cudaMemcpy(dQ,  hQ.data(),  hQ.size()  * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dK,  hKp.data(), hKp.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dVT, hVT.data(), hVT.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dO, 0, hQ.size() * 2));

  // ---- TMA tensor maps ----
  // pack-GQA Q/O: 4D TMA over [hd, nqh, token-IN-SAMPLE, sample]; box [hd-subtile x
  // gqa_group_size x q_tile_per_mtile x 1]. The per-sample token dim makes the HW clamp the box
  // at each sample's seqlen boundary (seqlen % q_tile_per_cta != 0: the last packed M-tile's
  // overrun rows would otherwise read/store the NEXT sample's tokens -- cross-sample clobber).
  const int gqa = sh.nqh / sh.nkh;            // q-heads per kv-head
  const int tpi = M_TILE / gqa;               // q-tokens per M-tile
  const int tpc = 2 * tpi;                    // q-tokens per CTA (2 M-tiles)
  CUtensorMap tq_, tk_, tvt_, to_;
  {
    // O store: 4D per-sample [hd, nqh, token-in-sample, sample].
    uint64_t gd[4] = { (uint64_t)sh.hd, (uint64_t)sh.nqh, (uint64_t)seqlen, (uint64_t)ns };
    uint64_t gs[3] = { (uint64_t)sh.hd * 2u, (uint64_t)sh.nqh * sh.hd * 2u,
                       (uint64_t)seqlen * sh.nqh * sh.hd * 2u };
    uint32_t bd[4] = { (uint32_t)SUB_COLS_BF16, (uint32_t)gqa, (uint32_t)tpi, 1u };
    uint32_t es[4] = { 1u, 1u, 1u, 1u };
    CUresult r = cuTensorMapEncodeTiled(&to_, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, dO, gd, gs, bd, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUDA_CHECK(r == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue);
  }
  {
    // Q load: 5D -- fold the 2 hd-atoms into the OUTERMOST box dim so ONE TMA fills the whole
    // head_dim (vs the 2x 4D loop). Dims [hd-col 64, nqh, token-in-sample, sample, hd-atom 2];
    // atom-outer box => smem = atom0 block (16 KB) then atom1, matching the looped layout.
    uint64_t gd[5] = { (uint64_t)SUB_COLS_BF16, (uint64_t)sh.nqh, (uint64_t)seqlen, (uint64_t)ns, (uint64_t)Q_SUBTILES };
    uint64_t gs[4] = { (uint64_t)sh.hd * 2u, (uint64_t)sh.nqh * sh.hd * 2u,
                       (uint64_t)seqlen * sh.nqh * sh.hd * 2u, (uint64_t)SUB_COLS_BF16 * 2u };
    uint32_t bd[5] = { (uint32_t)SUB_COLS_BF16, (uint32_t)gqa, (uint32_t)tpi, 1u, (uint32_t)Q_SUBTILES };
    uint32_t es[5] = { 1u, 1u, 1u, 1u, 1u };
    CUresult r = cuTensorMapEncodeTiled(&tq_, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, dQ, gd, gs, bd, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUDA_CHECK(r == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue);
  }
  // K: ONE 3D TMA copy folds the 2 head-dim swizzle atoms (HEAD_DIM = 2 x SUB_COLS_BF16) into the box
  // (vs looping 2 x 2D copies). dims [atom-col SUB_COLS_BF16, token tk, atom (nkh*hd)/SUB_COLS_BF16]; box
  // [SUB_COLS_BF16, K_TILE, K_SUBTILES]. Box dim order (atom outermost) reproduces the atom-outer smem
  // layout the MMA reads (atom0 then atom1).
  {
    uint64_t gd[3] = { (uint64_t)SUB_COLS_BF16, (uint64_t)tk_pad, (uint64_t)(sh.nkh * sh.hd / SUB_COLS_BF16) };
    uint64_t gs[2] = { (uint64_t)(sh.nkh * sh.hd) * 2u, (uint64_t)SUB_COLS_BF16 * 2u };
    // 2SM: K is N-split along kv-positions -- half-box token dim (K_TILE/2). Each peer loads its half.
    uint32_t bd[3] = { (uint32_t)SUB_COLS_BF16, (uint32_t)(K_TILE / 2), (uint32_t)K_SUBTILES };
    uint32_t es[3] = { 1u, 1u, 1u };
    CUresult r = cuTensorMapEncodeTiled(&tk_, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, dK, gd, gs, bd, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUDA_CHECK(r == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue);
  }
  // 2SM: V_T is N-split along head_dim -- half-box rows (hd/2). FA4 form: ONE 3D TMA per
  // V half-tile (token dim split (chunk, inner); box [inner, hd/2, 2] fills smem
  // [chunk][row][token]). Legal because padded sample bases are K_TILE-aligned.
  {
    uint64_t gd[3] = { (uint64_t)SUB_COLS_BF16, (uint64_t)(sh.nkh * sh.hd), (uint64_t)(tk_pad / SUB_COLS_BF16) };
    uint64_t gs[2] = { (uint64_t)tk_pad * 2u, (uint64_t)SUB_COLS_BF16 * 2u };
    uint32_t bd[3] = { (uint32_t)SUB_COLS_BF16, (uint32_t)(sh.hd / 2), 2u };
    uint32_t es[3] = { 1u, 1u, 1u };
    CUresult r = cuTensorMapEncodeTiled(&tvt_, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, dVT, gd, gs, bd, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUDA_CHECK(r == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue);
  }

  // ---- shared memory budget ----
  const int packed_mtiles_per_seq = (seqlen + tpc - 1) / tpc;   // packed-M tiles per (sample, kv-head)
  // 2SM: a cluster covers one (sample, kv_head) and a PAIR of packed-M tiles (peer 0 = even, peer 1
  // = odd). cluster work-items per sample = ceil(packed_mtiles_per_seq/2) * nkh.
  const int packed_mpairs_per_seq = (packed_mtiles_per_seq + 1) / 2;
  // FastDivmod magics for decode_workitem's divides:
  //   magic0 = mpairs_per_sample (cluster_workitem_id -> sample), magic1 = mpairs_per_seq (rr -> kv_head),
  //   magic2 = num_kv_heads (non-Q_RASTER rr -> pair_index).
  const unsigned long long magic0 = make_magic((unsigned)(packed_mpairs_per_seq * sh.nkh));
  const unsigned long long magic1 = make_magic((unsigned)packed_mpairs_per_seq);
  const unsigned long long magic2 = make_magic((unsigned)sh.nkh);
  // Compile-time kernel config (see the knob docs near the top of this file).
  constexpr bool FULL_NAMED_BAR = true, EX2_EMU = true, SPLIT_P = true,
                 SOFTMAX_THROTTLE = true, USE_CLC = false, Q_RASTER = true;
  const size_t smem =
        (size_t)2 * Q_TILE_BYTES + NUM_KV_STAGES * KV_SLOT_BYTES  // Q (x2) + compact K/V ring
      + (size_t)2 * M_TILE * HEAD_DIM * sizeof(__nv_bfloat16)     // 2 sO bufs for TMA-O
      + (2 * NUM_KV_STAGES + 22) * 8                              // mbarriers (incl full/empty_bar_o_epi)
      + (USE_CLC ? (size_t)CLC_STAGES * (2 * 8 + 16) + 16 : 0)// CLC: clc_full+clc_empty + response (16B aligned)
      + 8                                                         // tmem_slot
      + (size_t)2 * M_TILE * sizeof(float)                        // alpha_and_l_smem [2][M_TILE]
      + 256;                                                      // slack / alignment

  auto kfn = &fmha_context_bf16_gen_kernel<32, FULL_NAMED_BAR, EX2_EMU, SPLIT_P, SOFTMAX_THROTTLE, USE_CLC, Q_RASTER, MHA, IS_CAUSAL>;
  CUDA_CHECK(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  // 2SM: cluster_dims(2,1,1) is a non-portable cluster size (2 CTAs); allow it.
  CUDA_CHECK(cudaFuncSetAttribute(kfn, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

  // ---- launch geometry: persistent, 1 CTA/SM, grid-stride over work tiles ----
  const float scale_log2 = (1.0f / sqrtf((float)sh.hd)) * (float)M_LOG2E;
  int numSM = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, 0));
  const int total_work = ns * packed_mtiles_per_seq * sh.nkh;   // one CTA per (sample, packed-M tile, kv-head)
  const int total_workitems_host = ns * packed_mpairs_per_seq * sh.nkh;   // one cluster per (sample, kv-head, packed-M PAIR)
  int nblk;
  if (USE_CLC) {
    nblk = total_workitems_host * 2;
  } else {
    // FA4 form (ncu LaunchStats on the real run: Grid=152=numSM, Cluster 2, Waves/SM=1):
    // PERSISTENT, one CTA per SM, grid-stride over work items. (An earlier "non-persistent"
    // reading came from trace metadata, not the launch -- corrected here.)
    nblk = std::min(total_work, numSM);
    nblk -= (nblk & 1);                        // 2SM: grid.x must be a multiple of cluster.x = 2
  }
  dim3 grid(nblk, 1, 1), block(N_WARPS * 32, 1, 1);

  // 2SM cluster launch helper: cudaLaunchKernelEx with clusterDim {2,1,1}.
  auto launch = [&](cudaStream_t st = 0) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = smem; cfg.stream = st;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = 2; attr[0].val.clusterDim.y = 1; attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr; cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, kfn, tq_, tk_, tvt_, to_, seqlen, sh.nqh, sh.nkh,
                             scale_log2, packed_mtiles_per_seq, ns, magic0, magic1, magic2);
  };

  const double ms = fmha_context_bf16_benchmark::measure(launch);
#ifdef WARP_PROF
  {
    WpBuffer wp = wp_alloc(grid);
    const unsigned pblk = wp.view_block;
    CUDA_CHECK(launch());
    CUDA_CHECK(cudaDeviceSynchronize());
    wp_readback(wp);
    const char *roles[16] = {"sm0",  "sm0",  "sm0",  "sm0",  "sm1", "sm1", "sm1",  "sm1",
                             "corr", "corr", "corr", "corr", "mma", "epi", "load", "sched"};
    printf("  [%s] WARP_PROF block %u:\n", sh.lab, pblk);
    wp_print_busy(wp, roles, 16, pblk);
    wp_dump_raw(wp, "warp_raw_fmha_pgen.bin.gz", pblk, NUM_KV_STAGES);
    wp_free(wp);
  }
#endif

  const uint64_t eff = (uint64_t)ns * fmha_context_bf16_benchmark::attended_pairs(
      seqlen, seqlen, sh.causal);
  const double tflops = fmha_context_bf16_benchmark::report(
      sh.lab, tq, sh.causal, sh.hd, sh.nqh, eff, ms);

  // ---- correctness check against the fp32 CPU reference ----
  if (verify) {
    std::vector<__nv_bfloat16> ho(hQ.size());
    CUDA_CHECK(cudaMemcpy(ho.data(), dO, ho.size() * 2, cudaMemcpyDeviceToHost));

    std::vector<int> cq(ns + 1, 0);
    for (int i = 0; i < ns; ++i) cq[i + 1] = cq[i] + seqlen;

    std::vector<float> ref(hQ.size(), 0.f), out(hQ.size());
    {   // disk-cached CPU reference (pure function of shape+fill; key bumps if fills change)
      const char* fe = getenv("FILL");
      char key[128];
      snprintf(key, sizeof key, "B%d_S%d_hq%d_hk%d_hd%d_c%d_f%d",
               ns, seqlen, sh.nqh, sh.nkh, sh.hd, (int)sh.causal, fe ? atoi(fe) : 2);
      cached_ref_f32(key, ref.data(), ref.size(), [&] {
        cpu_fmha_ref(hQ.data(), hK.data(), hV.data(), ref.data(),
                     cq, cq, sh.nqh, sh.nkh, sh.hd, sh.causal);
      });
    }
    for (size_t i = 0; i < out.size(); ++i) out[i] = __bfloat162float(ho[i]);

    if (const char* dp = getenv("DUMP_O")) {   // debug: dump ref+out f32 for offline analysis
      std::string base(dp);
      FILE* f1 = fopen((base + ".ref").c_str(), "wb"); fwrite(ref.data(), 4, ref.size(), f1); fclose(f1);
      FILE* f2 = fopen((base + ".out").c_str(), "wb"); fwrite(out.data(), 4, out.size(), f2); fclose(f2);
    }
    const bool ok = check_close_f32(ref.data(), out.data(), (int)out.size(), 0.05f, 0.10f);
    printf("  verify [%s]: %s\n", sh.lab, ok ? "OK" : "FAIL");
  }

  // ---- stress mode: STRESS_N kernel launches, cross-run output consistency ----
  // Catches intermittent races (output drifts from run 0) and per-launch CUDA errors,
  // WITHOUT paying the CPU fp32 verify each time -- the single verify above (deterministic
  // inputs) already certifies run 0; here we just memcmp every run's O against run 0.
  if (const char* s = getenv("STRESS_N")) {
    const int N = atoi(s);
    std::vector<__nv_bfloat16> first(hQ.size()), cur(hQ.size());
    CUDA_CHECK(launch());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(first.data(), dO, first.size() * 2, cudaMemcpyDeviceToHost));
    int mism = 0, errs = 0;
    for (int it = 1; it < N; ++it) {
      CUDA_CHECK(cudaMemset(dO, 0, hQ.size() * 2));
      CUDA_CHECK(launch());
      cudaError_t e = cudaDeviceSynchronize();
      if (e != cudaSuccess) { printf("  STRESS run %d: CUDA ERR %s\n", it, cudaGetErrorString(e)); errs++; continue; }
      CUDA_CHECK(cudaMemcpy(cur.data(), dO, cur.size() * 2, cudaMemcpyDeviceToHost));
      if (memcmp(first.data(), cur.data(), cur.size() * 2) != 0) { printf("  STRESS run %d: OUTPUT MISMATCH vs run0\n", it); mism++; }
    }
    printf("  STRESS [%s] N=%d: mismatches=%d cuda_errs=%d\n", sh.lab, N, mism, errs);
  }

  cudaFree(dQ);
  cudaFree(dK);
  cudaFree(dVT);
  cudaFree(dO);
  return tflops;
}

int main() {
  CUDA_CHECK(cudaFree(0));
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 64 * 1024 * 1024));
  printf("K2 fmha_context_bf16 GEN (warp-spec, 2 M-tiles) sm_100a\n"
         "=====================================\n");

  Sh s{};
  const int B = getenv("BATCH")  ? atoi(getenv("BATCH"))  : 128;
  const int S = getenv("SEQLEN") ? atoi(getenv("SEQLEN")) : 240;
  const int H = getenv("HEADS")  ? atoi(getenv("HEADS"))  : 32;
  // MHA=1 -> HK==HQ (gqa_group=1); default GQA -> HK=4 (gqa_group=HQ/4=8 at HQ=32).
  const bool mha = getenv("MHA") ? (atoi(getenv("MHA")) != 0) : false;
  // CAUSAL=1 -> triangular mask (uniform seqlen; every query still emits a full row, so the
  // O store stays non-ragged and the TMA-O epilogue is unchanged). Varlen lives in its own file.
  const bool causal = getenv("CAUSAL") ? (atoi(getenv("CAUSAL")) != 0) : false;
  s.sl     = std::vector<int>(B, S);
  s.nqh    = H;
  s.nkh    = mha ? H : 4;
  s.hd     = 128;
  s.causal = causal;
  s.lab    = causal ? (mha ? "mha2sm-causal" : "gen2sm-causal") : (mha ? "mha2sm" : "gen2sm");
  const bool verify = getenv("NOVERIFY") ? false : (S <= 1024 || getenv("VERIFY") != nullptr);
  // 2SM uniform keeps Q_RASTER=true for BOTH GQA and MHA (no Q_RASTER=!MHA varlen trick).
  if      (mha && causal) run</*MHA=*/true,  /*IS_CAUSAL=*/true >(s, verify);
  else if (mha)           run</*MHA=*/true,  /*IS_CAUSAL=*/false>(s, verify);
  else if (causal)        run</*MHA=*/false, /*IS_CAUSAL=*/true >(s, verify);
  else                    run</*MHA=*/false, /*IS_CAUSAL=*/false>(s, verify);
  return 0;
}
