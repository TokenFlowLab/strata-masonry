// block_causal_sink_bf16_2sm.cu -- K2 FMHA context BF16, sm_100a.
//
// 2-CTA (cta_group::2) sibling of the 1CTA bcs kernel: same feature set
// (block-causal + sink + sliding window; MHA; USE_CLC / Q_RASTER scheduler), but a 2SM cluster
// pairs two CTAs on one (sample, kv_head) with a joint 256-row cta_group::2 MMA.
//
// ASSUMES all seqlens EQUAL (no varlen) -> uniform K_TILES (varlen lives in
// fmha_context_bf16_varlen.cu). The block-causal-sink mask + K-loop cap are always on;
// the cap comes from the higher (odd / peer1) tile so both peers keep the same K_TILES (lockstep
// for the joint MMA), and each CTA masks its own rows with its own q_pos.
// ALSO ASSUMES (checked host-side in run(), out-of-spec otherwise):
//   num_frames % num_frame_per_block == 0 (no partial last block), and SMALL sink
//   (sink_size - num_frame_per_block) * frame_seqlen <= K_TILE (sink reaches <=1 K_TILE past a block end).
//
// K-LOOP TILE COVERAGE (decode_workitem). Identical tile math to block_causal_sink_bf16.cu: the K-loop runs
// k_tiles = window_tiles + sink_tiles steps over WINDOW tiles [lo_tile, hi_tile) + SINK tiles [0,
// sink_tiles), disjoint (gap-skipped), incl. the RoPE window-dips-into-sink overlap-by-one-tile case (the
// tile holding sink_tokens is done twice: plain-q window step keeps cols >= sink_tokens + q_sink sink step
// keeps cols < sink_tokens) -- see that file's header for the full (a)/(b)/RoPE enumeration. 2SM
// DIFFERENCE: window_tiles / window_hi_tile / K_TILES come from the WHOLE 2-CTA cluster [cluster_first_q,
// cluster_last_q] (cap from the higher peer1 tile), so both peers share one K_TILES and stay in lockstep
// for the joint cta_group::2 MMA; each CTA still masks only its own rows.
//
// MASKING (softmax warp, per row). Keep col c (absolute key token) iff:
//     c < block_end  &&  ( c < sink_tokens  ||  c >= window_start )     [block_end/window_start per row]
//   Each CTA masks its own rows with their OWN block_end/window_start. non-RoPE: one plain-q pass (keep
//   sink UNION window). RoPE: sink tiles use q_sink (keep c < sink_tokens), window tiles use plain q (keep
//   c >= max(window_start, sink_tokens)) -- disjoint.
//
// PEELING (softmax_step). Same peel as bcs.cu (need_mask's two window terms are monotonic in tile, so the
//   masked window tiles form a top prefix + bottom suffix); the loop compiles MASKED per region and runs
//   the clean interior MASKED=false (mask block out of the hot loop). Bands are PER-CTA (not per-cluster):
//   from THIS CTA's cta_min_block_end (its first row) and cta_max_window_start (its last row) --
//     masked_top_tiles    = window_hi_tile - cta_min_block_end/K_TILE       (window tiles crossing block_end)
//     masked_bottom_tiles = ceil(bottom_mask_bound/K_TILE) - lo_tile        (crossing window_start; RoPE
//                                          bottom_mask_bound = max(cta_max_window_start, sink_tokens))
//   A CTA's per-CTA bands also mask the cluster window tiles OUTSIDE its own window (the other peer's), so
//   each CTA keeps exactly its rows. sink tiles: only the first (sink_tiles-1) masked. (The TF 2SM sibling
//   instead computes cluster-wide bands; per-CTA here is tighter when the cluster straddles a block.)
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
//   - Registers: per-warp budgets sum to the SM file (65536 = 2048*32): softmax inc<176> (x8) +
//     correction dec<88> (x4) + four single warps dec<72> (x4) = 2048. Differs from the 1CTA
//     kernel's inc<192>/dec<80>/dec<48> (which also sums to 2048).
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
#include "../../fmha/sm100a/fmha_utils.cuh"
#include "npy_io.cuh"
#include "block_causal_sink_bf16_benchmark.cuh"


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

// Step -> K-tile index, all DESCENDING (see 1CTA kernel). Window steps [0,window_tiles) descend from
// the top/diagonal tile; sink steps [window_tiles,k_tiles) descend from sink_tiles-1 to tile 0
// (= k_tiles-1-step).
__device__ __forceinline__ int tile_for_processing(int step, int window_tiles, int window_hi_tile, int k_tiles) {
  return (step < window_tiles) ? (window_hi_tile - 1 - step) : (k_tiles - 1 - step);
}

// Per-row column mask for block-causal + sink + sliding-window attention (register-only).
// c = absolute key token of a score column. KEEP col c iff:   c < block_end   &&   ( c < sink_tokens  ||  c >= window_start )
//   c < block_end    : block-causal upper bound (no future block).
//   c < sink_tokens  : always-attended sink prefix.
//   c >= window_start: inside the sliding window.
// The caller sets sink_tokens/window_start to select which of the 4 cases this tile is (see the
// call site): non-RoPE keeps sink UNION window in one plain-q pass; RoPE splits at sink_tokens
// into a q_sink pass (sink cols) and a plain-q pass (window cols).
template <int K_TILE, bool USE_R2P_ASM = true>
__device__ __forceinline__ void mask_s_row_bcs(float* scores, int k_tile_offset_in_tokens,
    int block_end, int sink_tokens, int window_start) {
  const int hi  = block_end     - k_tile_offset_in_tokens;
  const int snk = sink_tokens   - k_tile_offset_in_tokens;
  const int lo  = window_start  - k_tile_offset_in_tokens;
  uint32_t* u = reinterpret_cast<uint32_t*>(scores);
  #pragma unroll
  for (int s = 0; s < K_TILE / 32; ++s) {
    const int base = s * 32;
    int hb = hi - base;  hb = hb < 0 ? 0 : (hb > 32 ? 32 : hb);
    int sb = snk - base; sb = sb < 0 ? 0 : (sb > 32 ? 32 : sb);
    int lb = lo - base;  lb = lb < 0 ? 0 : (lb > 32 ? 32 : lb);
    const uint32_t keep_hi   = (hb >= 32) ? 0xFFFFFFFFu : (hb <= 0 ? 0u : ((1u << hb) - 1u));
    const uint32_t keep_sink = (sb >= 32) ? 0xFFFFFFFFu : (sb <= 0 ? 0u : ((1u << sb) - 1u));
    const uint32_t keep_win  = (lb >= 32) ? 0u : (0xFFFFFFFFu << lb);
    const uint32_t keep = keep_hi & (keep_sink | keep_win);
    if constexpr (USE_R2P_ASM) {
      #pragma unroll
      for (int g = 0; g < 32; g += 8) {
        const uint32_t kg = keep >> g;
        asm("{\n\t"
            ".reg .pred p0, p1, p2, p3, p4, p5, p6, p7;\n\t"
            ".reg .b32  t0, t1, t2, t3, t4, t5, t6, t7;\n\t"
            "and.b32 t0, %8, 1;    setp.ne.b32 p0, t0, 0; selp.b32 %0, %0, 0xFF800000, p0;\n\t"
            "and.b32 t1, %8, 2;    setp.ne.b32 p1, t1, 0; selp.b32 %1, %1, 0xFF800000, p1;\n\t"
            "and.b32 t2, %8, 4;    setp.ne.b32 p2, t2, 0; selp.b32 %2, %2, 0xFF800000, p2;\n\t"
            "and.b32 t3, %8, 8;    setp.ne.b32 p3, t3, 0; selp.b32 %3, %3, 0xFF800000, p3;\n\t"
            "and.b32 t4, %8, 16;   setp.ne.b32 p4, t4, 0; selp.b32 %4, %4, 0xFF800000, p4;\n\t"
            "and.b32 t5, %8, 32;   setp.ne.b32 p5, t5, 0; selp.b32 %5, %5, 0xFF800000, p5;\n\t"
            "and.b32 t6, %8, 64;   setp.ne.b32 p6, t6, 0; selp.b32 %6, %6, 0xFF800000, p6;\n\t"
            "and.b32 t7, %8, 128;  setp.ne.b32 p7, t7, 0; selp.b32 %7, %7, 0xFF800000, p7;\n\t"
            "}"
            : "+r"(u[s * 32 + g + 0]), "+r"(u[s * 32 + g + 1]),
              "+r"(u[s * 32 + g + 2]), "+r"(u[s * 32 + g + 3]),
              "+r"(u[s * 32 + g + 4]), "+r"(u[s * 32 + g + 5]),
              "+r"(u[s * 32 + g + 6]), "+r"(u[s * 32 + g + 7])
            : "r"(kg));
      }
    } else {
      #pragma unroll
      for (int i = 0; i < 32; ++i)
        if (!(keep & (1u << i))) scores[s * 32 + i] = -INFINITY;
    }
  }
}


template <bool Q_RASTER, bool HAS_SINK_ROPE_DELTA = false>
__device__ __forceinline__ void decode_workitem(
    int cluster_workitem_id, int peer, int seqlen, int num_kv_heads,
    int packed_mpairs_per_seq, int packed_mpairs_per_sample, int q_tile_per_cta,
    unsigned long long magic0, unsigned long long magic1, unsigned long long magic2,
    int tokens_per_block, int rolling_window_tokens, int sink_tokens,
    int& sample, int& h_kv, int& q_tile_base, int& k_tiles,
    int& window_tiles, int& window_hi_tile) {
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
  // cluster spans q tokens [cluster_first_q, cluster_max_q_base + q_tile_per_cta); both
  // peers share the K-loop, so bounds are conservative over the whole 2-CTA cluster.
  const int cluster_max_q_base = (2 * packed_mpairs_index + 1) * q_tile_per_cta;
  const int first_q = (2 * packed_mpairs_index) * q_tile_per_cta;
  const int last_q  = cluster_max_q_base + q_tile_per_cta - 1;
  int max_block_end = (last_q / tokens_per_block + 1) * tokens_per_block;
  if (max_block_end > seqlen) max_block_end = seqlen;
  // NOMINAL (unclamped) block end for window start: matches the reference's frame-level block_end
  // (blockwise_frame_visible uses no seqlen clamp), so a partial last block's window stays beyond seqlen.
  int min_window_start = ((first_q / tokens_per_block + 1) * tokens_per_block) - rolling_window_tokens;
  if (min_window_start < 0) min_window_start = 0;
  const int hi_tile = (max_block_end + K_TILE - 1) / K_TILE;
  int lo_tile = min_window_start / K_TILE;
  int sink_tiles = (sink_tokens + K_TILE - 1) / K_TILE;
  if constexpr (HAS_SINK_ROPE_DELTA) {
    // sink stays full (q_sink reload is tile-clean). The window band only needs [sink_tiles-1, hi_tile):
    // the boundary tile (holds sink_tokens) is done twice -- window step (plain q, cols >= sink_tokens) +
    // sink step (q_sink, cols < sink_tokens), disjoint -- and tiles below it are pure sink, covered by
    // the sink steps. Clamp lo_tile up so we skip their no-op window step; never past hi_tile (a block
    // fully inside the sink, block_end <= sink_tokens, has an empty window and is all sink).
    if (lo_tile < sink_tiles - 1) lo_tile = sink_tiles - 1;
    if (lo_tile > hi_tile) lo_tile = hi_tile;
  } else {
    if (sink_tiles > lo_tile) sink_tiles = lo_tile;   // don't re-visit window-covered tiles
  }
  window_hi_tile = hi_tile;
  window_tiles   = hi_tile - lo_tile;
  k_tiles     = window_tiles + sink_tiles;
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
template <int S_LD_COLS = 32, bool FULL_NAMED_BAR = false, bool EX2_EMU = false, bool SPLIT_P = true,
          bool SOFTMAX_THROTTLE = false, bool USE_CLC = true, bool Q_RASTER = true, bool MHA = false,
          int RESCALE_THRESHOLD = 8, bool HAS_SINK_ROPE_DELTA = false>
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(N_WARPS * 32, 1)
fmha_context_bf16_bcs_2sm_kernel(const __grid_constant__ CUtensorMap tmap_q,
    const __grid_constant__ CUtensorMap tmap_k, const __grid_constant__ CUtensorMap tmap_v_t,
    const __grid_constant__ CUtensorMap tmap_o, const __grid_constant__ CUtensorMap tmap_q_sink, int seqlen,
    int num_q_heads, int num_kv_heads, float scale_log2,
    int packed_mtiles_per_seq, int num_samples,
    unsigned long long magic0, unsigned long long magic1, unsigned long long magic2,
    int tokens_per_block, int sink_tokens, int rolling_window_tokens) {
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
  WpCtx warp_prof_ctx = wp_ctx_init();

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
      int sample, h_kv, q_tile_base, K_TILES, window_tiles, window_hi_tile;
      decode_workitem<Q_RASTER, HAS_SINK_ROPE_DELTA>(cluster_workitem_id, peer, seqlen, num_kv_heads, packed_mpairs_per_seq,
          packed_mpairs_per_sample, q_tile_per_cta, magic0, magic1, magic2,
          tokens_per_block, rolling_window_tokens, sink_tokens,
          sample, h_kv, q_tile_base, K_TILES, window_tiles, window_hi_tile);
      // clamp padding peer's OOB tile to a dummy in-bounds Q (can't bail).
      const int q_tile_base_safe = (q_tile_base < seqlen) ? q_tile_base : 0;
      // padded K/V token base (driver lays samples at seqlen_pad stride; K_TILE multiple)
      const int k_start = sample * (((seqlen + 127) >> 7) << 7);

      for (int k = 0; k < K_TILES; ++k) {
        const int k_tile_offset_in_tokens = tile_for_processing(k, window_tiles, window_hi_tile, K_TILES) * K_TILE;
        // HAS_SINK_ROPE_DELTA: window/context tiles done -> reload q_sink before the sink tiles. BEFORE
        // this tile's K/V loads (drain the ring; else a post-loop reload deadlocks). Gated empty_bar_q.
        if constexpr (HAS_SINK_ROPE_DELTA) {
          if (k == window_tiles) {
            #pragma unroll
            for (int m = 0; m < M_TILES_PER_CTA; ++m) {
              wp_begin(warp_prof_ctx, WP_LOAD_WAIT);
              mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_q[m]), q_empty_ph.get_phase());
              wp_end(warp_prof_ctx, WP_LOAD_WAIT);
              wp_begin(warp_prof_ctx, WP_LOAD_ISSUE_Q);
              const uint32_t q_sink_bar = tma_peer_bit_mask(smem_ptr_u32(&full_bar_q[m]));
              const int q_token = q_tile_base_safe + m * q_tile_per_mtile;
              const int q_head  = h_kv * gqa_group_size;
              if (elect_one_sync()) {
                if (peer == 0) mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar_q[m]), 2 * Q_TILE_BYTES);
                tma_load_5d_2sm(smem_ptr_u32(sQ[m]), &tmap_q_sink, q_sink_bar, 0, q_head, q_token, sample, 0);
              }
              wp_end(warp_prof_ctx, WP_LOAD_ISSUE_Q);
            }
            q_empty_ph.advance();
          }
        }
        int kv_stage = kv_empty_ph.get_stage();

        wp_begin(warp_prof_ctx, WP_LOAD_WAIT);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar[kv_stage]), kv_empty_ph.get_phase());
        kv_empty_ph.advance();
        wp_end(warp_prof_ctx, WP_LOAD_WAIT);

        wp_begin(warp_prof_ctx, WP_LOAD_ISSUE_K);
        const uint32_t kbar = tma_peer_bit_mask(smem_ptr_u32(&full_bar[kv_stage]));
        const uint32_t kdst = smem_ptr_u32(smem_kv[kv_stage]);
        const int      k_token     = k_start + k_tile_offset_in_tokens + peer * (K_TILE / 2);
        const int      k_head_atom = h_kv * K_SUBTILES;
        if (elect_one_sync()) {
          if (peer == 0) mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar[kv_stage]), K_TILE_BYTES);
          tma_load_3d_2sm(kdst, &tmap_k, kbar, 0, k_token, k_head_atom);
        }
        wp_end(warp_prof_ctx, WP_LOAD_ISSUE_K);

        if (k == 0) {
          #pragma unroll
          for (int m = 0; m < M_TILES_PER_CTA; ++m) {
            wp_begin(warp_prof_ctx, WP_LOAD_WAIT);
            mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_q[m]), q_empty_ph.get_phase());
            wp_end(warp_prof_ctx, WP_LOAD_WAIT);
            wp_begin(warp_prof_ctx, WP_LOAD_ISSUE_Q);
            const uint32_t qbar = tma_peer_bit_mask(smem_ptr_u32(&full_bar_q[m]));
            const int q_token = q_tile_base_safe + m * q_tile_per_mtile;
            const int q_head  = h_kv * gqa_group_size;
            if (elect_one_sync()) {
              if (peer == 0) mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar_q[m]), 2 * Q_TILE_BYTES);
              tma_load_5d_2sm(smem_ptr_u32(sQ[m]), &tmap_q, qbar, 0, q_head, q_token, sample, 0);
            }
            wp_end(warp_prof_ctx, WP_LOAD_ISSUE_Q);
          }
          q_empty_ph.advance();
        }

        kv_stage = kv_empty_ph.get_stage();

        wp_begin(warp_prof_ctx, WP_LOAD_WAIT);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar[kv_stage]), kv_empty_ph.get_phase());
        kv_empty_ph.advance();
        wp_end(warp_prof_ctx, WP_LOAD_WAIT);

        wp_begin(warp_prof_ctx, WP_LOAD_ISSUE_V);
        const uint32_t vbar = tma_peer_bit_mask(smem_ptr_u32(&full_bar[kv_stage]));
        const uint32_t vdst = smem_ptr_u32(smem_kv[kv_stage]);
        const int      v_head_row  = h_kv * HEAD_DIM + peer * (HEAD_DIM / 2);
        const int      v_token_atom = (k_start + k_tile_offset_in_tokens) / SUB_COLS_BF16;
        if (elect_one_sync()) {
          if (peer == 0) mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar[kv_stage]), V_TILE_BYTES);
          tma_load_3d_2sm(vdst, &tmap_v_t, vbar, 0, v_head_row, v_token_atom);
        }
        wp_end(warp_prof_ctx, WP_LOAD_ISSUE_V);
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
      int sample, h_kv, q_tile_base, K_TILES, window_tiles, window_hi_tile;
      decode_workitem<Q_RASTER, HAS_SINK_ROPE_DELTA>(cluster_workitem_id, peer, seqlen, num_kv_heads, packed_mpairs_per_seq,
          packed_mpairs_per_sample, q_tile_per_cta, magic0, magic1, magic2,
          tokens_per_block, rolling_window_tokens, sink_tokens,
          sample, h_kv, q_tile_base, K_TILES, window_tiles, window_hi_tile);
      (void)sample; (void)h_kv;
      // peer 0's even tile is always in range, so the leader never bails.
      (void)q_tile_base;

      if (peer == 0) {
        // prologue: BMM1 of K-block 0 for every M-tile.
        int kv_stage = kv_ph.get_stage();
        wp_begin(warp_prof_ctx, WP_MMA_WAIT_FULL_K);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar[kv_stage]), kv_ph.get_phase());
        kv_ph.advance();
        wp_end(warp_prof_ctx, WP_MMA_WAIT_FULL_K);

        #pragma unroll
        for (int i = 0; i < M_TILES_PER_CTA; ++i) {
          wp_begin(warp_prof_ctx, WP_MMA_WAIT_FULL_Q);
          mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_q[i]), q_ph.get_phase());
          wp_end(warp_prof_ctx, WP_MMA_WAIT_FULL_Q);

          wp_begin(warp_prof_ctx, WP_MMA_ISSUE);
          if (lead) {
            const uint32_t s_tmem_addr = tmem_base + (uint32_t)(i * S_COLS);
            SmemDescPair desc_a, desc_b;
            desc_a.u64 = desc_q0;  desc_a.w.x += (uint32_t)(i * Q_MTILE_DESC_DELTA);
            desc_b.u64 = desc_kv0; desc_b.w.x += (uint32_t)(kv_stage * (int)KV_DESC_DELTA);

            #pragma unroll
            for (int s = 0; s < Q_SUBTILES; ++s) {
              #pragma unroll
              for (int ki = 0; ki < K_ATOMS_PER_TILE; ++ki) {
                const bool enable_d = (s != 0) || (ki != 0);
                tcgen05_mma_f16_ss<2>(s_tmem_addr, desc_a.u64, desc_b.u64, idesc_qk, enable_d);
                desc_a.w.x += 2; desc_b.w.x += 2;
              }
              desc_a.w.x += (uint32_t)(SUB_DESC_DELTA_Q  - 2 * K_ATOMS_PER_TILE);
              desc_b.w.x += (uint32_t)(SUB_DESC_DELTA_KV - 2 * K_ATOMS_PER_TILE);
            }
          }
          wp_end(warp_prof_ctx, WP_MMA_ISSUE);

          wp_begin(warp_prof_ctx, WP_MMA_COMMIT);
          if (lead) tcgen05_commit_multicast<2>(smem_ptr_u32(&full_bar_spo[i]), 0x3);
          wp_end(warp_prof_ctx, WP_MMA_COMMIT);
        }

        wp_begin(warp_prof_ctx, WP_MMA_COMMIT);
        if (lead) tcgen05_commit_multicast<2>(smem_ptr_u32(&empty_bar[kv_stage]), 0x3);
        wp_end(warp_prof_ctx, WP_MMA_COMMIT);

        // main loop: BMM2(tile) then BMM1(next tile)
        for (int k_tile_id = 0; k_tile_id < K_TILES - 1; ++k_tile_id) {
          // ring: kv_stage = V(current) for BMM2, kv_stage_next = K(next) for BMM1-ahead.
          const int kv_stage = kv_ph.get_stage();

          wp_begin(warp_prof_ctx, WP_MMA_WAIT_FULL_V);
          mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar[kv_stage]), kv_ph.get_phase());
          kv_ph.advance();
          wp_end(warp_prof_ctx, WP_MMA_WAIT_FULL_V);

          // HAS_SINK_ROPE_DELTA: at the window->sink boundary, hand the sQ buffer back to the load warp (commit
          // empty_bar_q = "window Q consumed"), then wait for q_sink to land (full_bar_q). The BMM1(next)
          // below (first sink tile) then reads the reloaded q_sink from sQ. The last window BMM1(next)
          // was issued in the previous iteration, so this commit orders after it.
          if constexpr (HAS_SINK_ROPE_DELTA) {
            if (k_tile_id == window_tiles - 1) {
              wp_begin(warp_prof_ctx, WP_MMA_COMMIT);
              if (lead) {
                #pragma unroll
                for (int i = 0; i < M_TILES_PER_CTA; ++i)
                  tcgen05_commit_multicast<2>(smem_ptr_u32(&empty_bar_q[i]), 0x3);
              }
              wp_end(warp_prof_ctx, WP_MMA_COMMIT);
              q_ph.advance();
              #pragma unroll
              for (int i = 0; i < M_TILES_PER_CTA; ++i) {
                wp_begin(warp_prof_ctx, WP_MMA_WAIT_FULL_Q);
                mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_q[i]), q_ph.get_phase());
                wp_end(warp_prof_ctx, WP_MMA_WAIT_FULL_Q);
              }
            }
          }

          int kv_stage_next = 0;
          #pragma unroll
          for (int i = 0; i < M_TILES_PER_CTA; ++i) {
            wp_begin(warp_prof_ctx, WP_MMA_WAIT_P);
            mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_spo[i]), spo_ph.get_phase());
            wp_end(warp_prof_ctx, WP_MMA_WAIT_P);

            wp_begin(warp_prof_ctx, WP_MMA_ISSUE);
            const uint32_t s_tmem_addr = tmem_base + (uint32_t)(i * S_COLS);
            const uint32_t o_tmem_addr = tmem_base + (uint32_t)(2 * S_COLS + i * O_COLS);
            SmemDescPair desc_bv;
            desc_bv.u64 = desc_kv0;
            desc_bv.w.x += (uint32_t)(kv_stage * (int)KV_DESC_DELTA);

            #pragma unroll
            for (int s = 0; s < V_SUBTILES; ++s) {
              #pragma unroll
              for (int ki = 0; ki < K_ATOMS_PER_TILE; ++ki) {
                const int a = s * K_ATOMS_PER_TILE + ki;
                if constexpr (SPLIT_P) if (a == SPLIT_P_ATOM) {
                  wp_end(warp_prof_ctx, WP_MMA_ISSUE);
                  wp_begin(warp_prof_ctx, WP_MMA_WAIT_P);
                  mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_p_last[i]), spo_ph.get_phase());
                  wp_end(warp_prof_ctx, WP_MMA_WAIT_P);
                  wp_begin(warp_prof_ctx, WP_MMA_ISSUE);
                }
                const bool accumulate = (k_tile_id != 0) || (a != 0);
                if (lead) {
                  tcgen05_mma_f16_ts_2sm(o_tmem_addr, s_tmem_addr + (uint32_t)(a * 8), desc_bv.u64, idesc_pv, accumulate, 0, 0, 0, 0, 0, 0, 0, 0);
                }
                desc_bv.w.x += 2;
              }
              desc_bv.w.x += (uint32_t)(SUB_DESC_DELTA_KV - 2 * K_ATOMS_PER_TILE);
            }
            wp_end(warp_prof_ctx, WP_MMA_ISSUE);

            // K(next) is shared by both M-tiles: only i==0 waits + advances the ring.
            if (i == 0) {
              kv_stage_next = kv_ph.get_stage();
              wp_begin(warp_prof_ctx, WP_MMA_WAIT_FULL_K);
              mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar[kv_stage_next]), kv_ph.get_phase());
              wp_end(warp_prof_ctx, WP_MMA_WAIT_FULL_K);
              kv_ph.advance();
            }

            if (lead && i == M_TILES_PER_CTA - 1) {
              wp_begin(warp_prof_ctx, WP_MMA_COMMIT);
              tcgen05_commit_multicast<2>(smem_ptr_u32(&empty_bar[kv_stage]), 0x3);
              wp_end(warp_prof_ctx, WP_MMA_COMMIT);
            }

            // BMM1(next): Q@K -> S
            wp_begin(warp_prof_ctx, WP_MMA_ISSUE);
            if (lead) {
              SmemDescPair desc_a, desc_b;
              desc_a.u64 = desc_q0;  desc_a.w.x += (uint32_t)(i * Q_MTILE_DESC_DELTA);
              desc_b.u64 = desc_kv0; desc_b.w.x += (uint32_t)(kv_stage_next * (int)KV_DESC_DELTA);
              #pragma unroll
              for (int s = 0; s < Q_SUBTILES; ++s) {
                #pragma unroll
                for (int ki = 0; ki < K_ATOMS_PER_TILE; ++ki) {
                  const bool enable_d = (s != 0) || (ki != 0);
                  tcgen05_mma_f16_ss<2>(s_tmem_addr, desc_a.u64, desc_b.u64, idesc_qk, enable_d);
                  desc_a.w.x += 2; desc_b.w.x += 2;
                }
                desc_a.w.x += (uint32_t)(SUB_DESC_DELTA_Q  - 2 * K_ATOMS_PER_TILE);
                desc_b.w.x += (uint32_t)(SUB_DESC_DELTA_KV - 2 * K_ATOMS_PER_TILE);
              }
            }
            wp_end(warp_prof_ctx, WP_MMA_ISSUE);

            wp_begin(warp_prof_ctx, WP_MMA_COMMIT);
            if (lead) tcgen05_commit_multicast<2>(smem_ptr_u32(&full_bar_spo[i]), 0x3);
            wp_end(warp_prof_ctx, WP_MMA_COMMIT);
          }

          wp_begin(warp_prof_ctx, WP_MMA_COMMIT);
          if (lead) tcgen05_commit_multicast<2>(smem_ptr_u32(&empty_bar[kv_stage_next]), 0x3);
          wp_end(warp_prof_ctx, WP_MMA_COMMIT);

          spo_ph.advance();
        }

        wp_begin(warp_prof_ctx, WP_MMA_COMMIT);
        if (lead) {
          #pragma unroll
          for (int i = 0; i < M_TILES_PER_CTA; ++i) {
            tcgen05_commit_multicast<2>(smem_ptr_u32(&empty_bar_q[i]), 0x3);
          }
        }
        wp_end(warp_prof_ctx, WP_MMA_COMMIT);

        // epilogue: BMM2 of the last K-block -> final O
        kv_stage = kv_ph.get_stage();

        wp_begin(warp_prof_ctx, WP_MMA_WAIT_FULL_V);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar[kv_stage]), kv_ph.get_phase());
        kv_ph.advance();
        wp_end(warp_prof_ctx, WP_MMA_WAIT_FULL_V);

        #pragma unroll
        for (int i = 0; i < M_TILES_PER_CTA; ++i) {
          wp_begin(warp_prof_ctx, WP_MMA_WAIT_P);
          mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_spo[i]), spo_ph.get_phase());
          wp_end(warp_prof_ctx, WP_MMA_WAIT_P);

          wp_begin(warp_prof_ctx, WP_MMA_ISSUE);
          const uint32_t s_tmem_addr = tmem_base + (uint32_t)(i * S_COLS);
          const uint32_t o_tmem_addr = tmem_base + (uint32_t)(2 * S_COLS + i * O_COLS);
          SmemDescPair desc_bv;
          desc_bv.u64 = desc_kv0;
          desc_bv.w.x += (uint32_t)(kv_stage * (int)KV_DESC_DELTA);
          #pragma unroll
          for (int s = 0; s < V_SUBTILES; ++s) {
            #pragma unroll
            for (int ki = 0; ki < K_ATOMS_PER_TILE; ++ki) {
              const int a = s * K_ATOMS_PER_TILE + ki;   // flat atom (split-P + P addr)
              if constexpr (SPLIT_P) if (a == SPLIT_P_ATOM) {
                wp_end(warp_prof_ctx, WP_MMA_ISSUE);
                wp_begin(warp_prof_ctx, WP_MMA_WAIT_P);
                mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_p_last[i]), spo_ph.get_phase());
                wp_end(warp_prof_ctx, WP_MMA_WAIT_P);
                wp_begin(warp_prof_ctx, WP_MMA_ISSUE);
              }
              const bool accumulate = (K_TILES != 1) || (a != 0);
              if (lead) {
                tcgen05_mma_f16_ts_2sm(o_tmem_addr, s_tmem_addr + (uint32_t)(a * 8), desc_bv.u64, idesc_pv, accumulate, 0, 0, 0, 0, 0, 0, 0, 0);
              }
              desc_bv.w.x += 2;
            }
            desc_bv.w.x += (uint32_t)(SUB_DESC_DELTA_KV - 2 * K_ATOMS_PER_TILE);
          }
          wp_end(warp_prof_ctx, WP_MMA_ISSUE);

          wp_begin(warp_prof_ctx, WP_MMA_COMMIT);
          if (lead) tcgen05_commit_multicast<2>(smem_ptr_u32(&full_bar_o_acc[i]), 0x3);
          wp_end(warp_prof_ctx, WP_MMA_COMMIT);
        }

        wp_begin(warp_prof_ctx, WP_MMA_COMMIT);
        if (lead) tcgen05_commit_multicast<2>(smem_ptr_u32(&empty_bar[kv_stage]), 0x3);
        wp_end(warp_prof_ctx, WP_MMA_COMMIT);

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
      int sample, h_kv, q_tile_base, K_TILES, window_tiles, window_hi_tile;
      decode_workitem<Q_RASTER, HAS_SINK_ROPE_DELTA>(cluster_workitem_id, peer, seqlen, num_kv_heads, packed_mpairs_per_seq,
          packed_mpairs_per_sample, q_tile_per_cta, magic0, magic1, magic2,
          tokens_per_block, rolling_window_tokens, sink_tokens,
          sample, h_kv, q_tile_base, K_TILES, window_tiles, window_hi_tile);
      // padding peer waits in lockstep but does NOT store.
      const bool valid = (q_tile_base < seqlen);

      #pragma unroll
      for (int m = 0; m < M_TILES_PER_CTA; ++m) {
        wp_begin(warp_prof_ctx, WP_EPI_WAIT_TMEM);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_o_epi[m]), full_o_ph.get_phase());
        wp_end(warp_prof_ctx, WP_EPI_WAIT_TMEM);

        wp_begin(warp_prof_ctx, WP_EPI_STORE);
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
        wp_end(warp_prof_ctx, WP_EPI_STORE);
      }

      wp_begin(warp_prof_ctx, WP_EPI_WAIT_STORE);
      // Drain per-m commit groups; release each sO slot to corr as ITS store completes
      // (without this, corr's next pack races the in-flight store at tiny causal K-loops).
      // Runs on the padding peer too (no groups pending -> waits are no-ops; corr packs in lockstep).
      if (elect_one_sync()) {
        cp_async_bulk_wait_group_read<1>();
        mbarrier_arrive(smem_ptr_u32(&empty_bar_o_epi[0]));
        cp_async_bulk_wait_group_read<0>();
        mbarrier_arrive(smem_ptr_u32(&empty_bar_o_epi[1]));
      }
      wp_end(warp_prof_ctx, WP_EPI_WAIT_STORE);

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
      int sample, h_kv, q_tile_base, K_TILES, window_tiles, window_hi_tile;
      decode_workitem<Q_RASTER, HAS_SINK_ROPE_DELTA>(cluster_workitem_id, peer, seqlen, num_kv_heads, packed_mpairs_per_seq,
          packed_mpairs_per_sample, q_tile_per_cta, magic0, magic1, magic2,
          tokens_per_block, rolling_window_tokens, sink_tokens,
          sample, h_kv, q_tile_base, K_TILES, window_tiles, window_hi_tile);
      (void)h_kv; (void)q_tile_base;  // 2SM: the unpaired padding peer runs in lockstep (no bail)

      // block 0: no rescale (no prior O); consume alpha + release the scale slot.
      #pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i) {
        wp_begin(warp_prof_ctx, WP_CORR_WAIT);
        if constexpr (FULL_NAMED_BAR) full_bar_wait(i, corr_warp_id);
        else mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_alpha[i]), alpha_ph.get_phase());
        // FA4 cross-stage deferral (fa4_gen correction_loop): prologue releases only slot 0.
        if constexpr (SOFTMAX_THROTTLE) {
          if (i == 0) mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[0]));
        } else {
          mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
        }
        wp_end(warp_prof_ctx, WP_CORR_WAIT);
      }
      if constexpr (!FULL_NAMED_BAR) alpha_ph.advance();

      for (int k = 1; k < K_TILES; ++k) {
        #pragma unroll
        for (int i = 0; i < M_TILES_PER_CTA; ++i) {
          wp_begin(warp_prof_ctx, WP_CORR_WAIT);
          if constexpr (FULL_NAMED_BAR) full_bar_wait(i, corr_warp_id);
          else mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_alpha[i]), alpha_ph.get_phase());
          wp_end(warp_prof_ctx, WP_CORR_WAIT);

          wp_begin(warp_prof_ctx, WP_CORR_READ_ALPHA);
          float alpha = alpha_and_l_smem[i * M_TILE + corr_warp_id * 32 + lane];
          if constexpr (!SOFTMAX_THROTTLE) mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
          wp_end(warp_prof_ctx, WP_CORR_READ_ALPHA);

          wp_begin(warp_prof_ctx, WP_CORR_O_SCALE);
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
          wp_end(warp_prof_ctx, WP_CORR_O_SCALE);
        }
        if constexpr (!FULL_NAMED_BAR) alpha_ph.advance();
      }
      // FA4 post-loop rebalance: deferral left slot 1 one release short; pay it so l-publish can proceed.
      if constexpr (SOFTMAX_THROTTLE) mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[M_TILES_PER_CTA - 1]));

      // epilogue: O *= 1/l -> bf16 -> sO[i] -> signal W_EPI (full_bar_o_epi).
      #pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i) {
        wp_begin(warp_prof_ctx, WP_CORR_WAIT);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_o_acc[i]), o_acc_ph.get_phase());
        if constexpr (FULL_NAMED_BAR) full_bar_wait(i, corr_warp_id);
        else mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_l[i]), o_acc_ph.get_phase());
        wp_end(warp_prof_ctx, WP_CORR_WAIT);

        wp_begin(warp_prof_ctx, WP_CORR_EPI);
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
        wp_end(warp_prof_ctx, WP_CORR_EPI);
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
      int sample, h_kv, q_tile_base, K_TILES, window_tiles, window_hi_tile;
      decode_workitem<Q_RASTER, HAS_SINK_ROPE_DELTA>(cluster_workitem_id, peer, seqlen, num_kv_heads, packed_mpairs_per_seq,
          packed_mpairs_per_sample, q_tile_per_cta, magic0, magic1, magic2,
          tokens_per_block, rolling_window_tokens, sink_tokens,
          sample, h_kv, q_tile_base, K_TILES, window_tiles, window_hi_tile);
      (void)sample; (void)h_kv;   // 2SM: the unpaired padding peer runs in lockstep (no bail)

      // this row's query-token position (pack-GQA qh-inner).
      const int q_pos = q_tile_base + m_tile * q_tile_per_mtile + row_in_m_tile / gqa_group_size;
      // window_start from NOMINAL block end; mask UPPER clamped to seqlen (padding exclusion).
      int block_end_in_tokens = (q_pos / tokens_per_block + 1) * tokens_per_block;
      int window_start_in_tokens = block_end_in_tokens - rolling_window_tokens;
      if (window_start_in_tokens < 0) window_start_in_tokens = 0;
      if (block_end_in_tokens > seqlen) block_end_in_tokens = seqlen;
      const int cta_last_q = q_tile_base + q_tile_per_cta - 1;   // CTA-wide bounds for boundary-only masking
      int cta_min_block_end = (q_tile_base / tokens_per_block + 1) * tokens_per_block; if (cta_min_block_end > seqlen) cta_min_block_end = seqlen;
      int cta_max_window_start = ((cta_last_q  / tokens_per_block + 1) * tokens_per_block) - rolling_window_tokens;
      if (cta_max_window_start < 0) cta_max_window_start = 0;

      // Peel bands (per-CTA; the cluster shares K_TILES but each CTA masks its own rows): masked window
      // tiles are a top prefix (crossing this CTA's cta_min_block_end) + bottom suffix (crossing its
      // cta_max_window_start) -- these also cover cluster tiles outside this CTA's own window. The clean
      // interior runs MASKED=false (mask block out of the hot loop). All CTA-uniform.
      const int lo_tile = window_hi_tile - window_tiles;
      int bottom_mask_bound = cta_max_window_start;                    // window tiles starting below this are masked
      if constexpr (HAS_SINK_ROPE_DELTA) bottom_mask_bound = max(bottom_mask_bound, sink_tokens);
      int masked_top_tiles    = window_hi_tile - cta_min_block_end / K_TILE;              // window tiles crossing block_end
      int masked_bottom_tiles = (bottom_mask_bound + K_TILE - 1) / K_TILE - lo_tile;      // window tiles crossing window_start
      masked_top_tiles    = masked_top_tiles    < 0 ? 0 : (masked_top_tiles    > window_tiles ? window_tiles : masked_top_tiles);
      masked_bottom_tiles = masked_bottom_tiles < 0 ? 0 : (masked_bottom_tiles > window_tiles ? window_tiles : masked_bottom_tiles);
      const int interior_begin = masked_top_tiles;
      int interior_end = window_tiles - masked_bottom_tiles;
      if (interior_end < interior_begin) interior_end = interior_begin;

      float m_run = -INFINITY, l_run = 0.f;
      wp_begin(warp_prof_ctx, WP_SM_WAIT_SCALE);
      mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_alpha_and_l[m_tile]), scale_empty_ph.get_phase());
      scale_empty_ph.advance();
      wp_end(warp_prof_ctx, WP_SM_WAIT_SCALE);
      float* const alpha_slot = &alpha_and_l_smem[m_tile * M_TILE + row_in_m_tile];
      // FA4 softmax_loop: peeled masked steps + unmasked steady loop (compile-time step specializations).
      // CAUSAL: each softmax warp masks every tile from k=0 through its own diagonal (see fmha-fill-degenerate).
      auto softmax_step = [&](int k, auto masked_c, auto is_first_c) {
        constexpr bool MASKED   = decltype(masked_c)::value;
        constexpr bool IS_FIRST = decltype(is_first_c)::value;
        const int k_tile_offset_in_tokens = tile_for_processing(k, window_tiles, window_hi_tile, K_TILES) * K_TILE;
        wp_begin(warp_prof_ctx, WP_SM_WAIT_S);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_spo[m_tile]), spo_ph.get_phase());
        wp_end(warp_prof_ctx, WP_SM_WAIT_S);

        wp_begin(warp_prof_ctx, WP_SM_SOFTMAX);
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

        if constexpr (MASKED) {
          // The peel passes MASKED=true only on boundary tiles, and its bands are EXACTLY the tiles need_mask
          // would fire on (they were derived from need_mask's terms), so no need_mask re-check: mask directly.
          const bool is_sink_tile = (k >= window_tiles);   // K-loop visits window tiles then sink tiles
          int effective_sink_tokens = sink_tokens;
          int effective_window_start = window_start_in_tokens;
          if constexpr (HAS_SINK_ROPE_DELTA) {
            // RoPE: sink cols need q_sink, window cols need plain q -> keep the two DISJOINT at sink_tokens.
            if (is_sink_tile) {
              // sink tile (q_sink): keep only c < sink_tokens (window_start -> block_end kills that term).
              effective_window_start = block_end_in_tokens;
            } else {
              // window tile (plain q): keep only c >= max(window_start, sink_tokens); sink_tokens=0 kills the sink term.
              effective_sink_tokens = 0;
              effective_window_start = max(window_start_in_tokens, sink_tokens);
            }
          }
          // non-RoPE: effective_* stay the row's own (one plain-q pass keeps sink UNION window).
          mask_s_row_bcs<K_TILE>(scores, k_tile_offset_in_tokens, block_end_in_tokens,
                                 effective_sink_tokens, effective_window_start);
        }

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
        wp_end(warp_prof_ctx, WP_SM_SOFTMAX);

        wp_begin(warp_prof_ctx, WP_SM_STORE_P);
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
        wp_end(warp_prof_ctx, WP_SM_STORE_P);

        spo_ph.advance();
        wp_begin(warp_prof_ctx, WP_SM_WAIT_SCALE);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_alpha_and_l[m_tile]), scale_empty_ph.get_phase());
        scale_empty_ph.advance();
        wp_end(warp_prof_ctx, WP_SM_WAIT_SCALE);
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
      // Peeled loop (per-CTA): only boundary tiles compile MASKED=true; the clean window interior +
      // interior sink run MASKED=false. tile 0 (k=0) keeps IS_FIRST regardless.
      const bool first_masked = !(window_tiles > 0 && interior_begin == 0 && interior_end > 0);
      if (first_masked) softmax_step(0, std::true_type{},  std::true_type{});
      else              softmax_step(0, std::false_type{}, std::true_type{});
      int k = 1;
      if (window_tiles > 0) {
        for (; k < interior_begin; ++k) softmax_step(k, std::true_type{},  std::false_type{});  // top block_end band
        for (; k < interior_end;   ++k) softmax_step(k, std::false_type{}, std::false_type{});  // clean middle
        for (; k < window_tiles;   ++k) softmax_step(k, std::true_type{},  std::false_type{});  // bottom window_start band
        if (window_tiles < K_TILES) { softmax_step(window_tiles, std::true_type{}, std::false_type{}); ++k; }  // first sink
      }
      for (; k < K_TILES; ++k)          softmax_step(k, std::false_type{}, std::false_type{});   // interior sink
      wp_begin(warp_prof_ctx, WP_SM_READ_L);
      alpha_and_l_smem[m_tile * M_TILE + row_in_m_tile] = l_run;
      if constexpr (FULL_NAMED_BAR) full_bar_arrive(m_tile, warp_in_group);
      else mbarrier_arrive(smem_ptr_u32(&full_bar_l[m_tile]));
      wp_end(warp_prof_ctx, WP_SM_READ_L);

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
  wp_flush(warp_prof_ctx);
  __syncthreads();
  barrier_cluster_arrive_relaxed_aligned();   // control-only join: no release fence
  barrier_cluster_wait_aligned();             // (FA4 has no MEMBAR.ALL pair here)
  if (warp_id == 0) tcgen05_dealloc<2>(tmem_base, TMEM_TOTAL);
}

// ============================== driver ====================================

// CPU reference: per-(sample, q-head) flash-attention in fp32. Q/K/V are the natural
// [token, head, head_dim] layouts (NOT the transposed V_T the kernel uses). q_cumsum/k_cumsum are cumulative-seqlen
// prefix sums (q_cumsum[s]..q_cumsum[s+1] = sample s's tokens). (sample, q-head) pairs write disjoint output
// rows, so the outer pair is parallelized -- a serial reference is ~60 GFLOP and takes ~40 s.
static void cpu_fmha_ref(const __nv_bfloat16* hQ, const __nv_bfloat16* hK,
                         const __nv_bfloat16* hV, float* hO,
                         const std::vector<int>& q_cumsum, const std::vector<int>& k_cumsum,
                         int num_q_heads, int num_kv_heads, int head_dim, bool causal) {
  const long  Nq    = q_cumsum.back();
  const int   g     = num_q_heads / num_kv_heads;              // q-heads per kv-head (GQA group)
  const float scale = 1.0f / sqrtf((float)head_dim);
  const int   num_samples    = (int)q_cumsum.size() - 1;

  for (long i = 0; i < Nq * num_q_heads * head_dim; ++i) hO[i] = 0.f;

  #pragma omp parallel for schedule(dynamic) collapse(2)
  for (int s = 0; s < num_samples; ++s) {
    for (int h = 0; h < num_q_heads; ++h) {
      const int q_lo = q_cumsum[s], k_lo = k_cumsum[s];
      const int sample_q_len   = q_cumsum[s + 1] - q_lo, sample_k_len = k_cumsum[s + 1] - k_lo;
      const int kv_head   = h / g;                 // kv-head this q-head reads

      for (int i = 0; i < sample_q_len; ++i) {
        const int q_pos_abs = q_lo + i;
        std::vector<float> row_scores(sample_k_len);
        float row_max = -INFINITY;

        // scores row_scores[j] = scale * <Q[q_pos_abs], K[k_lo+j]>, with causal/padding mask
        for (int j = 0; j < sample_k_len; ++j) {
          if (causal && j > i) { row_scores[j] = -INFINITY; continue; }
          float dot = 0.f;
          for (int e = 0; e < head_dim; ++e)
            dot += __bfloat162float(hQ[(q_pos_abs * num_q_heads + h) * head_dim + e])
                 * __bfloat162float(hK[((k_lo + j) * num_kv_heads + kv_head) * head_dim + e]);
          row_scores[j] = dot * scale;
          row_max = fmaxf(row_max, row_scores[j]);
        }

        // softmax over the row
        float sum = 0.f;
        for (int j = 0; j < sample_k_len; ++j) {
          if (row_scores[j] == -INFINITY) { row_scores[j] = 0.f; continue; }
          row_scores[j] = expf(row_scores[j] - row_max);
          sum += row_scores[j];
        }
        if (sum == 0.f) continue;
        const float inv_sum = 1.f / sum;

        // O[q_pos_abs] = (row_scores @ V) / sum
        for (int e = 0; e < head_dim; ++e) {
          float acc = 0.f;
          for (int j = 0; j < sample_k_len; ++j)
            acc += row_scores[j] * __bfloat162float(hV[((k_lo + j) * num_kv_heads + kv_head) * head_dim + e]);
          hO[(q_pos_abs * num_q_heads + h) * head_dim + e] = acc * inv_sum;
        }
      }
    }
  }
}

// CPU reference for block-causal + sink + sliding-window (see 1CTA kernel).
static void cpu_fmha_ref_bcs(const __nv_bfloat16* hQ, const __nv_bfloat16* hK,
                             const __nv_bfloat16* hV, float* hO,
                             const std::vector<int>& q_cumsum, const std::vector<int>& k_cumsum,
                             int num_q_heads, int num_kv_heads, int head_dim,
                             int tokens_per_block, int sink_tokens, int rolling_window_tokens,
                             const __nv_bfloat16* hQsink = nullptr) {
  const long  Nq    = q_cumsum.back();
  const int   g     = num_q_heads / num_kv_heads;
  const float scale = 1.0f / sqrtf((float)head_dim);
  const int   num_samples    = (int)q_cumsum.size() - 1;
  for (long i = 0; i < Nq * num_q_heads * head_dim; ++i) hO[i] = 0.f;
  #pragma omp parallel for schedule(dynamic) collapse(2)
  for (int s = 0; s < num_samples; ++s) {
    for (int h = 0; h < num_q_heads; ++h) {
      const int q_lo = q_cumsum[s], k_lo = k_cumsum[s];
      const int sample_q_len   = q_cumsum[s + 1] - q_lo, sample_k_len = k_cumsum[s + 1] - k_lo;
      const int kv_head   = h / g;
      for (int i = 0; i < sample_q_len; ++i) {
        const int q_pos_abs = q_lo + i;
        // NOMINAL block end for window start; upper clamped to sample_k_len (padding). (Matches device.)
        int block_end = (i / tokens_per_block + 1) * tokens_per_block;
        int window_start = block_end - rolling_window_tokens;
        if (window_start < 0) window_start = 0;
        if (block_end > sample_k_len) block_end = sample_k_len;
        std::vector<float> row_scores(sample_k_len);
        float row_max = -INFINITY;
        for (int j = 0; j < sample_k_len; ++j) {
          const bool keep = (j < block_end) && (j < sink_tokens || j >= window_start);
          if (!keep) { row_scores[j] = -INFINITY; continue; }
          // sink columns use the rotated q_sink when HAS_SINK_ROPE_DELTA (relativistic sink correction).
          const __nv_bfloat16* qrow = (hQsink && j < sink_tokens) ? hQsink : hQ;
          float dot = 0.f;
          for (int e = 0; e < head_dim; ++e)
            dot += __bfloat162float(qrow[(q_pos_abs * num_q_heads + h) * head_dim + e])
                 * __bfloat162float(hK[((k_lo + j) * num_kv_heads + kv_head) * head_dim + e]);
          row_scores[j] = dot * scale;
          row_max = fmaxf(row_max, row_scores[j]);
        }
        float sum = 0.f;
        for (int j = 0; j < sample_k_len; ++j) {
          if (row_scores[j] == -INFINITY) { row_scores[j] = 0.f; continue; }
          row_scores[j] = expf(row_scores[j] - row_max); sum += row_scores[j];
        }
        if (sum == 0.f) continue;
        const float inv_sum = 1.f / sum;
        for (int e = 0; e < head_dim; ++e) {
          float acc = 0.f;
          for (int j = 0; j < sample_k_len; ++j)
            acc += row_scores[j] * __bfloat162float(hV[((k_lo + j) * num_kv_heads + kv_head) * head_dim + e]);
          hO[(q_pos_abs * num_q_heads + h) * head_dim + e] = acc * inv_sum;
        }
      }
    }
  }
}


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

// One benchmark shape. seqlens[s] = seqlen of sample s (all equal here); num_q_heads/num_kv_heads =
// q/kv head counts; head_dim = head dim; causal toggles the mask; label is for printing.
struct Shape {
  std::vector<int> seqlens;
  int  num_q_heads, num_kv_heads, head_dim;
  bool causal;
  const char* label;
  bool bcs = false;
  int  tokens_per_block = 0;
  int  sink_tokens = 0;
  int  rolling_window_tokens = 0;
  bool has_delta = false;                                 // ROPE_DELTA: relativistic sink correction
  int  frame_seqlen = 0, num_frame_per_block = 1, num_frames = 1, sink_size = 0, local_attn_size = -1;
};

template <bool MHA = false, bool HAS_SINK_ROPE_DELTA = false>
static double run(const Shape& shape, bool verify) {
  const int  num_samples     = (int)shape.seqlens.size();
  const int  seqlen = shape.seqlens[0];
  // Supported BCS regime (see the file-header ASSUMES). Outside it decode/masking are
  // out-of-spec, so fail loud rather than return silently-wrong results.
  if (shape.bcs) {
    if (shape.num_frames % shape.num_frame_per_block != 0) {
      fprintf(stderr, "BCS: num_frames (%d) must be a multiple of num_frame_per_block (%d) -- no partial last block\n",
              shape.num_frames, shape.num_frame_per_block); exit(1);
    }
    if ((shape.sink_size - shape.num_frame_per_block) * shape.frame_seqlen > K_TILE) {
      fprintf(stderr, "BCS: sink (%d frames) reaches >1 K_TILE past a block end -- unsupported large-sink regime\n",
              shape.sink_size); exit(1);
    }
  }
  const long total_q_tokens     = (long)num_samples * seqlen;      // total q-tokens
  const long total_k_tokens     = total_q_tokens;                     // total k-tokens (== total_q_tokens here)
  // FA4-form single-TMA V needs 64-aligned sample K/V bases: pad each sample's K/V
  // token span to a K_TILE multiple (pad tokens zero-filled; masked by seqlen anyway).
  const int  seqlen_pad = ((seqlen + 127) / 128) * 128;
  const long total_k_tokens_pad     = (long)num_samples * seqlen_pad;

  // ---- device buffers (bf16; V stored transposed as V_T for the BMM2 TMA) ----
  __nv_bfloat16 *dQ, *dK, *dVT, *dO;
  CUDA_CHECK(cudaMalloc(&dQ,  total_q_tokens * shape.num_q_heads * shape.head_dim * 2));
  CUDA_CHECK(cudaMalloc(&dK,  total_k_tokens_pad * shape.num_kv_heads * shape.head_dim * 2));
  CUDA_CHECK(cudaMalloc(&dVT, (long)shape.num_kv_heads * shape.head_dim * total_k_tokens_pad * 2));
  CUDA_CHECK(cudaMalloc(&dO,  total_q_tokens * shape.num_q_heads * shape.head_dim * 2));

  // ---- host inputs + V -> V_T transpose ([tok,head,head_dim] -> [head,head_dim,tok]) ----
  std::vector<__nv_bfloat16> hQ(total_q_tokens * shape.num_q_heads * shape.head_dim),
                             hK(total_k_tokens * shape.num_kv_heads * shape.head_dim),
                             hV(total_k_tokens * shape.num_kv_heads * shape.head_dim);
  const char* load_npy = getenv("LOAD_NPY");   // unified bench: shared Q/K/V from .npy [L,H,D]
  if (load_npy) {
    const std::string d(load_npy);
    auto ld = [&](const char* nm, std::vector<__nv_bfloat16>& h) {
      char p[64]; snprintf(p, sizeof p, "/%s_L%d.npy", nm, seqlen);
      auto bits = npy_load_vec<uint16_t>(d + p);
      if (bits.size() != h.size()) { fprintf(stderr, "LOAD_NPY: %s size %zu != %zu\n", nm, bits.size(), h.size()); exit(1); }
      memcpy(h.data(), bits.data(), h.size() * 2);
    };
    ld("q", hQ); ld("k", hK); ld("v", hV);
  } else {
    fillr(hQ.data(), hQ.size(), 11);
    fillr(hK.data(), hK.size(), 22);
    fillr(hV.data(), hV.size(), 33);
  }

  // ---- RoPE sink-delta: q_sink[r] = R(-delta(block(r))) * q[r] (used for sink columns only).
  // delta(b) = max(0, (b+1)*num_frame_per_block - local_attn_size) FRAMES; closed-form 1D RoPE over
  // head_dim (GPT-J pairs share angle = delta*freq_i, freq_i = 10000^(-2i/D)). The SAME tables are
  // fed to the reference for the gold cross-check. See the 1CTA kernel. ----
  std::vector<__nv_bfloat16> hQsink;
  if (shape.has_delta) {
    hQsink.resize(hQ.size());
    const int head_dim = shape.head_dim, tokens_per_frame = shape.frame_seqlen, num_frame_per_block = shape.num_frame_per_block;
    const int tokens_per_block = num_frame_per_block * tokens_per_frame;
    for (long r = 0; r < total_q_tokens; ++r) {
      const int q_pos = static_cast<int>(r % seqlen);
      const int block = q_pos / tokens_per_block;
      const int block_end_fr = (block + 1) * num_frame_per_block;
      const int delta_fr = (block_end_fr > shape.local_attn_size) ? (block_end_fr - shape.local_attn_size) : 0;
      for (int h = 0; h < shape.num_q_heads; ++h) {
        const long base = (r * shape.num_q_heads + h) * head_dim;
        for (int i = 0; i < head_dim / 2; ++i) {
          const double freq = pow(10000.0, -2.0 * i / head_dim);
          const double ang  = (double)delta_fr * freq;
          const float c = (float)cos(ang), s = (float)sin(ang);
          const float a  = __bfloat162float(hQ[base + 2 * i]);
          const float b  = __bfloat162float(hQ[base + 2 * i + 1]);
          hQsink[base + 2 * i]     = __float2bfloat16(a * c + b * s);
          hQsink[base + 2 * i + 1] = __float2bfloat16(b * c - a * s);
        }
      }
    }
  }

  // padded-layout device images: sample s's tokens at base s*seqlen_pad (pads zeroed)
  std::vector<__nv_bfloat16> hKp((long)total_k_tokens_pad * shape.num_kv_heads * shape.head_dim, __float2bfloat16(0.f));
  std::vector<__nv_bfloat16> hVT((long)shape.num_kv_heads * shape.head_dim * total_k_tokens_pad, __float2bfloat16(0.f));
  for (int sm = 0; sm < num_samples; ++sm)
    for (int t = 0; t < seqlen; ++t) {
      const long src = (long)(sm * seqlen + t), dst = (long)sm * seqlen_pad + t;
      for (int h = 0; h < shape.num_kv_heads; ++h)
        for (int d = 0; d < shape.head_dim; ++d) {
          hKp[(dst * shape.num_kv_heads + h) * shape.head_dim + d]   = hK[(src * shape.num_kv_heads + h) * shape.head_dim + d];
          hVT[(h * shape.head_dim + d) * total_k_tokens_pad + dst]   = hV[(src * shape.num_kv_heads + h) * shape.head_dim + d];
        }
    }

  __nv_bfloat16* dQsink = nullptr;   // rotated queries for the sink columns (HAS_SINK_ROPE_DELTA)
  if (shape.has_delta) {
    CUDA_CHECK(cudaMalloc(&dQsink, total_q_tokens * shape.num_q_heads * shape.head_dim * 2));
    CUDA_CHECK(cudaMemcpy(dQsink, hQsink.data(), hQsink.size() * 2, cudaMemcpyHostToDevice));
  }
  CUDA_CHECK(cudaMemcpy(dQ,  hQ.data(),  hQ.size()  * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dK,  hKp.data(), hKp.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dVT, hVT.data(), hVT.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dO, 0, hQ.size() * 2));

  // ---- TMA tensor maps ----
  // pack-GQA Q/O: 4D TMA over [head_dim, num_q_heads, token-IN-SAMPLE, sample]; box [head_dim-subtile x
  // gqa_group_size x q_tile_per_mtile x 1]. The per-sample token dim makes the HW clamp the box
  // at each sample's seqlen boundary (seqlen % q_tile_per_cta != 0: the last packed M-tile's
  // overrun rows would otherwise read/store the NEXT sample's tokens -- cross-sample clobber).
  const int gqa_group_size = shape.num_q_heads / shape.num_kv_heads;            // q-heads per kv-head
  const int q_tokens_per_mtile = M_TILE / gqa_group_size;               // q-tokens per M-tile
  const int q_tokens_per_cta = 2 * q_tokens_per_mtile;                    // q-tokens per CTA (2 M-tiles)
  CUtensorMap tmap_q, tmap_k, tmap_v_t, tmap_o, tmap_q_sink;
  {
    // O store: 4D per-sample [head_dim, num_q_heads, token-in-sample, sample].
    uint64_t global_dims[4] = { (uint64_t)shape.head_dim, (uint64_t)shape.num_q_heads, (uint64_t)seqlen, (uint64_t)num_samples };
    uint64_t global_strides[3] = { (uint64_t)shape.head_dim * 2u, (uint64_t)shape.num_q_heads * shape.head_dim * 2u,
                       (uint64_t)seqlen * shape.num_q_heads * shape.head_dim * 2u };
    uint32_t box_dims[4] = { (uint32_t)SUB_COLS_BF16, (uint32_t)gqa_group_size, (uint32_t)q_tokens_per_mtile, 1u };
    uint32_t elem_strides[4] = { 1u, 1u, 1u, 1u };
    CUresult r = cuTensorMapEncodeTiled(&tmap_o, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, dO, global_dims, global_strides, box_dims, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUDA_CHECK(r == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue);
  }
  {
    // Q load: 5D -- fold the 2 head_dim-atoms into the OUTERMOST box dim so ONE TMA fills the whole
    // head_dim (vs the 2x 4D loop). Dims [head_dim-col 64, num_q_heads, token-in-sample, sample, head_dim-atom 2];
    // atom-outer box => smem = atom0 block (16 KB) then atom1, matching the looped layout.
    uint64_t global_dims[5] = { (uint64_t)SUB_COLS_BF16, (uint64_t)shape.num_q_heads, (uint64_t)seqlen, (uint64_t)num_samples, (uint64_t)Q_SUBTILES };
    uint64_t global_strides[4] = { (uint64_t)shape.head_dim * 2u, (uint64_t)shape.num_q_heads * shape.head_dim * 2u,
                       (uint64_t)seqlen * shape.num_q_heads * shape.head_dim * 2u, (uint64_t)SUB_COLS_BF16 * 2u };
    uint32_t box_dims[5] = { (uint32_t)SUB_COLS_BF16, (uint32_t)gqa_group_size, (uint32_t)q_tokens_per_mtile, 1u, (uint32_t)Q_SUBTILES };
    uint32_t elem_strides[5] = { 1u, 1u, 1u, 1u, 1u };
    CUresult r = cuTensorMapEncodeTiled(&tmap_q, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, dQ, global_dims, global_strides, box_dims, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUDA_CHECK(r == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue);
  }
  {
    // q_sink map: identical 5D layout to tmap_q, base = dQsink (or dQ when !has_delta -- unused placeholder).
    uint64_t global_dims[5] = { (uint64_t)SUB_COLS_BF16, (uint64_t)shape.num_q_heads, (uint64_t)seqlen, (uint64_t)num_samples, (uint64_t)Q_SUBTILES };
    uint64_t global_strides[4] = { (uint64_t)shape.head_dim * 2u, (uint64_t)shape.num_q_heads * shape.head_dim * 2u,
                       (uint64_t)seqlen * shape.num_q_heads * shape.head_dim * 2u, (uint64_t)SUB_COLS_BF16 * 2u };
    uint32_t box_dims[5] = { (uint32_t)SUB_COLS_BF16, (uint32_t)gqa_group_size, (uint32_t)q_tokens_per_mtile, 1u, (uint32_t)Q_SUBTILES };
    uint32_t elem_strides[5] = { 1u, 1u, 1u, 1u, 1u };
    CUresult r = cuTensorMapEncodeTiled(&tmap_q_sink, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, shape.has_delta ? dQsink : dQ,
        global_dims, global_strides, box_dims, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUDA_CHECK(r == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue);
  }
  // K: ONE 3D TMA copy folds the 2 head-dim swizzle atoms (HEAD_DIM = 2 x SUB_COLS_BF16) into the box
  // (vs looping 2 x 2D copies). dims [atom-col SUB_COLS_BF16, token total_k_tokens, atom (num_kv_heads*head_dim)/SUB_COLS_BF16]; box
  // [SUB_COLS_BF16, K_TILE, K_SUBTILES]. Box dim order (atom outermost) reproduces the atom-outer smem
  // layout the MMA reads (atom0 then atom1).
  {
    uint64_t global_dims[3] = { (uint64_t)SUB_COLS_BF16, (uint64_t)total_k_tokens_pad, (uint64_t)(shape.num_kv_heads * shape.head_dim / SUB_COLS_BF16) };
    uint64_t global_strides[2] = { (uint64_t)(shape.num_kv_heads * shape.head_dim) * 2u, (uint64_t)SUB_COLS_BF16 * 2u };
    // 2SM: K is N-split along kv-positions -- half-box token dim (K_TILE/2). Each peer loads its half.
    uint32_t box_dims[3] = { (uint32_t)SUB_COLS_BF16, (uint32_t)(K_TILE / 2), (uint32_t)K_SUBTILES };
    uint32_t elem_strides[3] = { 1u, 1u, 1u };
    CUresult r = cuTensorMapEncodeTiled(&tmap_k, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, dK, global_dims, global_strides, box_dims, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUDA_CHECK(r == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue);
  }
  // 2SM: V_T is N-split along head_dim -- half-box rows (head_dim/2). FA4 form: ONE 3D TMA per
  // V half-tile (token dim split (chunk, inner); box [inner, head_dim/2, 2] fills smem
  // [chunk][row][token]). Legal because padded sample bases are K_TILE-aligned.
  {
    uint64_t global_dims[3] = { (uint64_t)SUB_COLS_BF16, (uint64_t)(shape.num_kv_heads * shape.head_dim), (uint64_t)(total_k_tokens_pad / SUB_COLS_BF16) };
    uint64_t global_strides[2] = { (uint64_t)total_k_tokens_pad * 2u, (uint64_t)SUB_COLS_BF16 * 2u };
    uint32_t box_dims[3] = { (uint32_t)SUB_COLS_BF16, (uint32_t)(shape.head_dim / 2), 2u };
    uint32_t elem_strides[3] = { 1u, 1u, 1u };
    CUresult r = cuTensorMapEncodeTiled(&tmap_v_t, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, dVT, global_dims, global_strides, box_dims, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUDA_CHECK(r == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue);
  }

  // ---- shared memory budget ----
  const int packed_mtiles_per_seq = (seqlen + q_tokens_per_cta - 1) / q_tokens_per_cta;   // packed-M tiles per (sample, kv-head)
  // 2SM: a cluster covers one (sample, kv_head) and a PAIR of packed-M tiles (peer 0 = even, peer 1
  // = odd). cluster work-items per sample = ceil(packed_mtiles_per_seq/2) * num_kv_heads.
  const int packed_mpairs_per_seq = (packed_mtiles_per_seq + 1) / 2;
  // FastDivmod magics for decode_workitem's divides:
  //   magic0 = mpairs_per_sample (cluster_workitem_id -> sample), magic1 = mpairs_per_seq (rr -> kv_head),
  //   magic2 = num_kv_heads (non-Q_RASTER rr -> pair_index).
  const unsigned long long magic0 = make_magic((unsigned)(packed_mpairs_per_seq * shape.num_kv_heads));
  const unsigned long long magic1 = make_magic((unsigned)packed_mpairs_per_seq);
  const unsigned long long magic2 = make_magic((unsigned)shape.num_kv_heads);
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

  auto kernel_fn = &fmha_context_bf16_bcs_2sm_kernel<32, FULL_NAMED_BAR, EX2_EMU, SPLIT_P, SOFTMAX_THROTTLE, USE_CLC, Q_RASTER, MHA, 8, HAS_SINK_ROPE_DELTA>;
  CUDA_CHECK(cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  // 2SM: cluster_dims(2,1,1) is a non-portable cluster size (2 CTAs); allow it.
  CUDA_CHECK(cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

  // ---- launch geometry: persistent, 1 CTA/SM, grid-stride over work tiles ----
  const float scale_log2 = (1.0f / sqrtf((float)shape.head_dim)) * (float)M_LOG2E;
  int numSM = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, 0));
  const int total_work = num_samples * packed_mtiles_per_seq * shape.num_kv_heads;   // one CTA per (sample, packed-M tile, kv-head)
  const int total_workitems_host = num_samples * packed_mpairs_per_seq * shape.num_kv_heads;   // one cluster per (sample, kv-head, packed-M PAIR)
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

  // block-causal-sink runtime bounds (0 tokens_per_block => plain full/causal path).
  const int tokens_per_block_arg      = shape.bcs ? shape.tokens_per_block : 0;
  const int sink_tokens_arg           = shape.bcs ? shape.sink_tokens : 0;
  const int rolling_window_tokens_arg = shape.bcs ? shape.rolling_window_tokens : 0;
  // 2SM cluster launch helper: cudaLaunchKernelEx with clusterDim {2,1,1}.
  auto launch = [&]() {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = smem; cfg.stream = 0;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = 2; attr[0].val.clusterDim.y = 1; attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr; cfg.numAttrs = 1;
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel_fn, tmap_q, tmap_k, tmap_v_t, tmap_o, tmap_q_sink, seqlen, shape.num_q_heads, shape.num_kv_heads,
                                  scale_log2, packed_mtiles_per_seq, num_samples, magic0, magic1, magic2,
                                  tokens_per_block_arg, sink_tokens_arg, rolling_window_tokens_arg));
  };

  const double ms = block_causal_sink_bf16_benchmark::measure(launch);
#ifdef WARP_PROF
  {
    WpBuffer wp = wp_alloc(grid);
    const unsigned pblk = wp.view_block;
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    wp_readback(wp);
    const char *roles[16] = {"sm0",  "sm0",  "sm0",  "sm0",  "sm1", "sm1", "sm1",  "sm1",
                             "corr", "corr", "corr", "corr", "mma", "epi", "load", "sched"};
    printf("  [%s] WARP_PROF block %u:\n", shape.label, pblk);
    wp_print_busy(wp, roles, 16, pblk);
    wp_dump_raw(wp, "warp_raw_fmha_pgen.bin.gz", pblk, NUM_KV_STAGES);
    wp_free(wp);
  }
#endif

  const auto pairs = block_causal_sink_bf16_benchmark::attended_pairs(
      seqlen, shape.bcs, shape.causal, shape.tokens_per_block,
      shape.sink_tokens, shape.rolling_window_tokens);
  const double tflops = ms > 0.0
      ? fmha_context_bf16_benchmark::report(shape.label, total_q_tokens, shape.causal,
          shape.head_dim, shape.num_q_heads, num_samples * pairs, ms) : 0.0;

  if (const char* dp = getenv("DUMP_O")) {
    std::vector<__nv_bfloat16> ho(hQ.size());
    CUDA_CHECK(cudaMemcpy(ho.data(), dO, ho.size() * 2, cudaMemcpyDeviceToHost));
    std::vector<float> of(ho.size());
    for (size_t i = 0; i < of.size(); ++i) of[i] = __bfloat162float(ho[i]);
    FILE* f = fopen((std::string(dp) + ".out").c_str(), "wb");
    fwrite(of.data(), 4, of.size(), f); fclose(f);
  }

  // ---- correctness check against the fp32 CPU reference ----
  if (verify) {
    std::vector<__nv_bfloat16> ho(hQ.size());
    CUDA_CHECK(cudaMemcpy(ho.data(), dO, ho.size() * 2, cudaMemcpyDeviceToHost));

    std::vector<int> q_cumsum(num_samples + 1, 0);
    for (int i = 0; i < num_samples; ++i) q_cumsum[i + 1] = q_cumsum[i] + seqlen;

    std::vector<float> ref(hQ.size(), 0.f), out(hQ.size());
    {   // disk-cached CPU reference (pure function of shape+fill; key bumps if fills change)
      const char* fe = getenv("FILL");
      char key[192];
      if (shape.bcs)
        snprintf(key, sizeof key, "B%d_S%d_hq%d_hk%d_hd%d_bcs_tpb%d_sink%d_win%d%s_f%d",
                 num_samples, seqlen, shape.num_q_heads, shape.num_kv_heads, shape.head_dim, shape.tokens_per_block, shape.sink_tokens,
                 shape.rolling_window_tokens, shape.has_delta ? "_rope" : "", fe ? atoi(fe) : 2);
      else
        snprintf(key, sizeof key, "B%d_S%d_hq%d_hk%d_hd%d_c%d_f%d",
                 num_samples, seqlen, shape.num_q_heads, shape.num_kv_heads, shape.head_dim, (int)shape.causal, fe ? atoi(fe) : 2);
      auto compute_reference = [&] {
        if (shape.bcs)
          cpu_fmha_ref_bcs(hQ.data(), hK.data(), hV.data(), ref.data(),
                           q_cumsum, q_cumsum, shape.num_q_heads, shape.num_kv_heads, shape.head_dim,
                           shape.tokens_per_block, shape.sink_tokens, shape.rolling_window_tokens,
                           shape.has_delta ? hQsink.data() : nullptr);
        else
          cpu_fmha_ref(hQ.data(), hK.data(), hV.data(), ref.data(),
                       q_cumsum, q_cumsum, shape.num_q_heads, shape.num_kv_heads, shape.head_dim, shape.causal);
      };
      // A shape/fill cache does not identify externally supplied input values.
      if (load_npy) compute_reference();
      else cached_ref_f32((std::string("bcs_ref_v2_") + key).c_str(),
                          ref.data(), ref.size(), compute_reference);
    }
    for (size_t i = 0; i < out.size(); ++i) out[i] = __bfloat162float(ho[i]);

    if (const char* dp = getenv("DUMP_O")) {   // debug: dump ref+out f32 for offline analysis
      std::string base(dp);
      FILE* f1 = fopen((base + ".ref").c_str(), "wb"); fwrite(ref.data(), 4, ref.size(), f1); fclose(f1);
      FILE* f2 = fopen((base + ".out").c_str(), "wb"); fwrite(out.data(), 4, out.size(), f2); fclose(f2);
    }
    const bool ok = check_close_f32(ref.data(), out.data(), (int)out.size(), 0.05f, 0.10f);
    printf("  verify [%s]: %s\n", shape.label, ok ? "OK" : "FAIL");
  }

  // ---- stress mode: STRESS_N kernel launches, cross-run output consistency ----
  // Catches intermittent races (output drifts from run 0) and per-launch CUDA errors,
  // WITHOUT paying the CPU fp32 verify each time -- the single verify above (deterministic
  // inputs) already certifies run 0; here we just memcmp every run's O against run 0.
  if (const char* s = getenv("STRESS_N")) {
    const int N = atoi(s);
    std::vector<__nv_bfloat16> first(hQ.size()), cur(hQ.size());
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(first.data(), dO, first.size() * 2, cudaMemcpyDeviceToHost));
    int mism = 0, errs = 0;
    for (int it = 1; it < N; ++it) {
      CUDA_CHECK(cudaMemset(dO, 0, hQ.size() * 2));
      launch();
      cudaError_t e = cudaDeviceSynchronize();
      if (e != cudaSuccess) { printf("  STRESS run %d: CUDA ERR %s\n", it, cudaGetErrorString(e)); errs++; continue; }
      CUDA_CHECK(cudaMemcpy(cur.data(), dO, cur.size() * 2, cudaMemcpyDeviceToHost));
      if (memcmp(first.data(), cur.data(), cur.size() * 2) != 0) { printf("  STRESS run %d: OUTPUT MISMATCH vs run0\n", it); mism++; }
    }
    printf("  STRESS [%s] N=%d: mismatches=%d cuda_errs=%d\n", shape.label, N, mism, errs);
  }

  if (dQsink) cudaFree(dQsink);
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

  Shape s{};
  const int B = getenv("BATCH")  ? atoi(getenv("BATCH"))  : 128;
  const int H = getenv("HEADS")  ? atoi(getenv("HEADS"))  : 32;
  const bool mha0 = getenv("MHA") ? (atoi(getenv("MHA")) != 0) : false;

  // BCS=1 -> block-causal + attention sink + sliding window (see 1CTA kernel).
  const bool bcs  = getenv("BCS") ? (atoi(getenv("BCS")) != 0) : false;
  const bool rope = getenv("ROPE_DELTA") ? (atoi(getenv("ROPE_DELTA")) != 0) : false;  // relativistic sink RoPE
  const int  tpf  = getenv("TOKENS_PER_FRAME")    ? atoi(getenv("TOKENS_PER_FRAME"))    : 0;
  const int  nfpb = getenv("NUM_FRAME_PER_BLOCK") ? atoi(getenv("NUM_FRAME_PER_BLOCK")) : 1;
  const int  nf   = getenv("NUM_FRAMES")          ? atoi(getenv("NUM_FRAMES"))          : 1;
  const int  sinkf= getenv("SINK_SIZE")           ? atoi(getenv("SINK_SIZE"))           : 0;
  const int  locw = getenv("LOCAL_ATTN_SIZE")     ? atoi(getenv("LOCAL_ATTN_SIZE"))     : -1;
  const int  tokens_per_seq = tpf * nf;

  int S;
  if      (getenv("SEQLEN")) S = atoi(getenv("SEQLEN"));
  else if (bcs)              S = tokens_per_seq;
  else                       S = 240;

  const bool mha    = getenv("MHA") ? mha0 : true;   // BCS defaults to MHA (Wan HK==HQ); MHA=0 -> GQA (orthogonal, HK=4)
  s.seqlens     = std::vector<int>(B, S);
  s.num_q_heads    = H;
  s.num_kv_heads    = mha ? H : 4;
  s.head_dim     = 128;
  s.causal = true;   // block-causal
  if (bcs) {
    s.bcs = true;
    s.tokens_per_block      = nfpb * tpf;
    s.sink_tokens           = sinkf * tpf;
    s.rolling_window_tokens = (locw < 0) ? (S + s.tokens_per_block)
                                         : ((locw - sinkf) > 0 ? (locw - sinkf) : 0) * tpf;
    // RoPE sink-delta only matters when the window scrolls past the sink (locw>=0, sink>0).
    s.has_delta           = rope && locw >= 0 && sinkf > 0;
    s.frame_seqlen        = tpf;
    s.num_frame_per_block = nfpb;
    s.num_frames          = nf;
    s.sink_size           = sinkf;
    s.local_attn_size     = locw;
  }
  s.label    = mha ? "bcs2sm" : "bcs2sm-gqa";
  const bool verify = getenv("NOVERIFY") ? false : (S <= 1024 || getenv("VERIFY") != nullptr);
  // 2SM uniform keeps Q_RASTER=true for BOTH GQA and MHA (no Q_RASTER=!MHA varlen trick).
  // block-causal-sink (masked). GQA is orthogonal: same positional mask, q-heads share a kv-head.
  if (mha) {
    if (s.has_delta) run</*MHA=*/true,  /*HAS_SINK_ROPE_DELTA=*/true >(s, verify);
    else             run</*MHA=*/true,  /*HAS_SINK_ROPE_DELTA=*/false>(s, verify);
  } else {
    if (s.has_delta) run</*MHA=*/false, /*HAS_SINK_ROPE_DELTA=*/true >(s, verify);
    else             run</*MHA=*/false, /*HAS_SINK_ROPE_DELTA=*/false>(s, verify);
  }
  return 0;
}
