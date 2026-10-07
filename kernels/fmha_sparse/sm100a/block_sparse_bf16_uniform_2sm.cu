// block_sparse_bf16_uniform_2sm.cu -- VSA "fine" block-sparse FMHA, bf16, 2SM
// (cta_group::2 cluster), 128-token sparse block, sm_100a.
//
// LINEAGE (this is the point of the file): a WHOLE-FILE FORK of the dense
// fmha_context_bf16_uniform_2sm_inline.cu -- the TUNED fork -- with the VSA features ported ONTO
// it. It was previously built on the BLOCK BASE dense 2SM (fmha_context_bf16_uniform_2sm.cu) and
// called blocks/111 (MMA), 93 (epi), 97 (sched), 99 (corr); base -> _inline measures +8..15% on the
// dense kernels, so the whole file was re-forked rather than patched item by item. It therefore
// inherits, by construction: the compact K/V ring (KV_SLOT_BYTES = 16 KB half-box per peer x
// NUM_KV_STAGES = 6, vs the base's half-wasted 3 x 32 KB -- journey item #0, +88 on dense), the
// walked SmemDescPair descriptors, the FA4 register ladder, the 5D-Q / 3D-V_T single-TMA loads,
// EX2_FREQ=10 + EX2_START_FRG=1, and the inlined corr/epi/sched warps (0 blocks/* calls).
//
// VSA features ported on:
//   - Gather: K/V tiles come from a top-k list, not a seqlen walk (union_id() in the load warp).
//   - UNION: a joint 256-row MMA is TWO adjacent 128-token q-blocks with DIFFERENT top-k lists, so
//     the cluster walks the UNION of the two (union_idx / union_num, k_tiles = union size) and each
//     softmax warp SKIPS the union positions its own q-block did not select (membership bitmask,
//     bit o = q-block 4p+o = peer*2 + m_tile): publish alpha, store an all-zero P, leave m/l alone.
//     Measured union inflation x2.7 at 25% density -- this is why the kernel loses to 1CTA blk128,
//     and it is ALGORITHMIC, not a lineage or tuning problem.
//   - Non-causal, MHA only: IS_CAUSAL / GQA / LPT / varlen from the dense base compile out.
//
// ONE DELIBERATE DEVIATION from the inline base: the softmax row-sum is computed LIVE in the exp2
// loop instead of deferred past the P stores. Deferring it requires s_regs (128 registers of a
// 176-register budget) to stay live across the P stores, and with the membership skip ~63% of union
// positions do no work -- measured as a 40% REGRESSION (0.602x) on this kernel. Keeping the sum in
// the loop bounds s_regs to the member path. Do not "restore" the deferred form here.
//
// GEN-shape variant. Warp-specialized 16-warp body + barrier contract are the SAME as
// fmha_context_bf16_gqa_nonpersistent.cu (see its header for Terminology / Layout / flow / barriers).
//   - PERSISTENT: cluster-stride over work items (phase trackers persist across items). USE_CLC is
//     FALSE here (run() sets it), so the scheduler warp is idle and the stride is gridDim-based.
//   - TMA-store epilogue: valid because every selected block is full and equal-sized (non-ragged).
//
// Barrier contract (additions to gqa_nonpersistent's -- unique to the TMA-sO epilogue):
//   - empty_bar_o_epi[m] (count 1): epi -> corr, "sO[m]'s TMA store drained, slot reusable" --
//     see fmha_context_bf16_uniform.cu's header.
//   NOTE: unlike the dense base there is NO padding peer -- run() rejects nb % 4 != 0, so every
//   cluster owns 4 real q-blocks and both peers always have work. The q_tile_base_safe clamp and
//   the "unpaired padding peer must not bail" comments inherited from the dense base are therefore
//   vestigial here; they are harmless (the clamp never fires) and were left rather than risk a
//   behavioural change in a comment-only cleanup.
//
// Budgets:
//   - Registers: per-warp budgets sum to exactly the SM file (65536 = 32*(8*176 + 4*88 + 4*72)):
//     softmax inc<176> (x8) + correction dec<88> (x4) + four single warps dec<72> (x4). NOTE: the
//     dense base's header (and this file's, before the rewrite) claimed 192/80/48 -- that ladder is
//     self-consistent arithmetically (it also sums to 65536), which is why the stale claim survived,
//     but it does NOT match the code. The values above are what setmaxnreg actually requests.
//     Under WARP_PROF the softmax budget is tight: wp_begin/wp_end in a warp at its cap faults as an
//     illegal instruction, so bump it before profiling.
//   - TMEM: each M-tile needs S + O = K_TILE + HEAD_DIM cols of the 512, so
//     M_TILES_PER_CTA <= 512 / (K_TILE + HEAD_DIM) = 2 for 128/128 (adjacent q-tiles of the
//     SAME sequence).
//
// Work-item math (VSA; replaces the dense base's seqlen-derived packed-M-tile arithmetic):
//   The unit is the 128-token SPARSE BLOCK, not a slice of seqlen. A cluster owns FOUR adjacent
//   q-blocks {4p, 4p+1, 4p+2, 4p+3} for one (sample, head): peer 0 takes {4p, 4p+1}, peer 1 takes
//   {4p+2, 4p+3}, each as its 2 M-tiles. So
//     pairs_per_seq   = nb / 4                        (cluster work-items per (sample, head))
//     total work-items = num_samples * num_heads * pairs_per_seq
//     q_tile_base      = (4*p + 2*peer) * BLOCK       (this CTA's first q-block token)
//   decode_workitem() splits cluster_workitem_id back into (sample, head, p) and additionally
//   returns pair_row -- the row of THIS CLUSTER's union list -- and k_tiles = union_num[pair_row].
//   NOTE k_tiles is the UNION SIZE and therefore VARIES per work item; the dense base derived a
//   constant from seqlen. Every warp must read it from the same decode so the peers stay in
//   lockstep for the joint MMA.
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
#include <string>
#include "npy_io.cuh"
#include "block_sparse_bf16_benchmark.cuh"
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


constexpr int BLOCK  = 128;   // VSA sparse block: the top-k selection granularity (queries AND keys)
constexpr int M_TILE = BLOCK; // one 128-token q-block per M-tile
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

// VSA work-item decode. Same out-params as the dense one it replaces (so every warp's call site is
// unchanged in shape), plus pair_row -- the row of THIS CLUSTER's union list.
//
// A cluster's joint cta_group::2 MMA is 256 rows = TWO adjacent 128-token q-blocks, and those two
// blocks have DIFFERENT top-k lists, so the cluster walks the UNION of the two lists: union_num
// [pair_row] entries, union_idx[pair_row * max_union + j] block ids. k_tiles is therefore the union
// size, NOT a seqlen walk, and each softmax warp skips the union positions its own q-block did not
// select (membership bitmask; see the softmax warp).
template <bool Q_RASTER, bool IS_CAUSAL>
__device__ __forceinline__ void decode_workitem(
    int cluster_workitem_id, int peer, int num_heads,
    int pairs_per_seq, int pairs_per_sample,
    unsigned long long magic0, unsigned long long magic1, unsigned long long magic2,
    const int* __restrict__ union_num,
    int& sample, int& h_kv, int& q_tile_base, int& k_tiles, int& pair_row) {
  sample = (int)fdiv((unsigned)cluster_workitem_id, magic0);
  const int rr = cluster_workitem_id - sample * pairs_per_sample;
  int pidx;
  // magic1 = 1/num_heads, magic2 = 1/pairs_per_seq (see the host's make_magic calls). These are
  // EQUAL when pairs_per_seq == num_heads (e.g. nb=32, H=8), so swapping them only shows up at
  // other shapes -- which is exactly how it hid until nb > 32.
  if constexpr (Q_RASTER) {
    h_kv = (int)fdiv((unsigned)rr, magic2);
    pidx = rr - h_kv * pairs_per_seq;
  } else {
    pidx = (int)fdiv((unsigned)rr, magic1);
    h_kv = rr - pidx * num_heads;
  }
  // this CTA's first q-block token: the cluster covers 128-blocks {4p, 4p+1, 4p+2, 4p+3}, peer p
  // taking the pair {4p + 2*peer, 4p + 2*peer + 1} (2 M-tiles of BLOCK rows each).
  q_tile_base = (4 * pidx + 2 * peer) * BLOCK;
  pair_row    = (sample * num_heads + h_kv) * pairs_per_seq + pidx;
  k_tiles     = union_num[pair_row];   // union size; BOTH peers read the SAME row -> lockstep
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
fmha_context_bf16_vsa_2sm_kernel(const __grid_constant__ CUtensorMap tmap_q,
    const __grid_constant__ CUtensorMap tmap_k, const __grid_constant__ CUtensorMap tmap_v_t,
    const __grid_constant__ CUtensorMap tmap_o, int seqlen, int num_heads, float scale_log2,
    int num_samples, int pairs_per_seq, int max_union,
    unsigned long long magic0, unsigned long long magic1, unsigned long long magic2,
    const int* __restrict__ union_idx, const int* __restrict__ union_num,
    const uint8_t* __restrict__ membership) {
  // VSA is MHA-only and non-causal: gqa_group_size folds to 1, so q_tile_per_mtile == M_TILE == BLOCK
  // and the IS_CAUSAL / GQA paths inherited from the dense base compile out.
  const int gqa_group_size = 1;
  const int q_tile_per_mtile = M_TILE;
  const int q_tile_per_cta   = M_TILES_PER_CTA * q_tile_per_mtile;

  // 2SM: a cluster's 2 CTAs share one (sample, head) and one UNION list; peer 0 takes 128-blocks
  // {4p, 4p+1}, peer 1 {4p+2, 4p+3}; the joint MMA pairs peer0.mtile_i with peer1.mtile_i (256-row
  // M); K/V is N-split per CTA (each peer loads its 64-token half-box).
  const int peer           = blockIdx.x & 1;
  const int cluster_id     = blockIdx.x >> 1;
  const int num_clusters   = gridDim.x >> 1;
  const int pairs_per_sample = pairs_per_seq * num_heads;
  const int total_workitems  = num_samples * pairs_per_sample;
  (void)num_clusters; (void)total_workitems; (void)magic2; (void)gqa_group_size;
  (void)q_tile_per_cta; (void)seqlen;

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
      int sample, h_kv, q_tile_base, K_TILES, pair_row;
      decode_workitem<Q_RASTER, IS_CAUSAL>(cluster_workitem_id, peer, num_heads, pairs_per_seq,
          pairs_per_sample, magic0, magic1, magic2, union_num,
          sample, h_kv, q_tile_base, K_TILES, pair_row);
      // clamp padding peer's OOB tile to a dummy in-bounds Q (can't bail).
      const int q_tile_base_safe = (q_tile_base < seqlen) ? q_tile_base : 0;
      const int k_start = sample * seqlen;   // VSA: uniform seqlen, no K_TILE padding

      // Union-list gather (replaces the dense seqlen walk): 32 block ids per coalesced read held
      // lane-spread in a register chunk and reused by K AND V. Warp-collective (shfl) -> it must be
      // evaluated OUTSIDE elect_one_sync.
      const int* union_row = union_idx + (size_t)pair_row * max_union;
      int id_chunk_base = -(1 << 30);
      int id_chunk_val  = 0;
      auto union_id = [&](int j) -> int {
        if (j < id_chunk_base || j >= id_chunk_base + 32) {
          id_chunk_base = j & ~31;
          const int idx = min(id_chunk_base + lane, K_TILES - 1);   // clamp: tail lanes stay in-row
          id_chunk_val = union_row[idx];
        }
        return __shfl_sync(0xffffffffu, id_chunk_val, j & 31);
      };

      for (int k = 0; k < K_TILES; ++k) {
        const int k_offset = union_id(k) * BLOCK;   // gathered block, not (K_TILES-1-k)*K_TILE
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
      int sample, h_kv, q_tile_base, K_TILES, pair_row;
      decode_workitem<Q_RASTER, IS_CAUSAL>(cluster_workitem_id, peer, num_heads, pairs_per_seq,
          pairs_per_sample, magic0, magic1, magic2, union_num,
          sample, h_kv, q_tile_base, K_TILES, pair_row);
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
      int sample, h_kv, q_tile_base, K_TILES, pair_row;
      decode_workitem<Q_RASTER, IS_CAUSAL>(cluster_workitem_id, peer, num_heads, pairs_per_seq,
          pairs_per_sample, magic0, magic1, magic2, union_num,
          sample, h_kv, q_tile_base, K_TILES, pair_row);
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
      int sample, h_kv, q_tile_base, K_TILES, pair_row;
      decode_workitem<Q_RASTER, IS_CAUSAL>(cluster_workitem_id, peer, num_heads, pairs_per_seq,
          pairs_per_sample, magic0, magic1, magic2, union_num,
          sample, h_kv, q_tile_base, K_TILES, pair_row);
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
    // Which bit of the membership byte belongs to THIS softmax warp's q-block: the cluster owns
    // blocks {4p, 4p+1, 4p+2, 4p+3} and bit o is block 4p+o, so o = peer*2 + m_tile.
    const int membership_bit = peer * 2 + m_tile;
    PhaseTracker<1> spo_ph;
    PhaseTracker<1> scale_empty_ph;

    int cluster_workitem_id = cluster_id;
    [[maybe_unused]] int clc_stage = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;
    while (true) {
      int sample, h_kv, q_tile_base, K_TILES, pair_row;
      decode_workitem<Q_RASTER, IS_CAUSAL>(cluster_workitem_id, peer, num_heads, pairs_per_seq,
          pairs_per_sample, magic0, magic1, magic2, union_num,
          sample, h_kv, q_tile_base, K_TILES, pair_row);
      (void)sample; (void)h_kv;   // 2SM: the unpaired padding peer runs in lockstep (no bail)
      const uint8_t* membership_row = membership + (size_t)pair_row * max_union;   // this cluster's row

      // Inherited from the dense base for its causal mask; VSA is non-causal so nothing reads it.
      const int q_pos = q_tile_base + m_tile * q_tile_per_mtile + row_in_m_tile / gqa_group_size;
      (void)q_pos;
      float m_run = -INFINITY, l_run = 0.f;
      wp_begin(wpc, WP_SM_WAIT_SCALE);
      mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_alpha_and_l[m_tile]), scale_empty_ph.get_phase());
      scale_empty_ph.advance();
      wp_end(wpc, WP_SM_WAIT_SCALE);
      float* const alpha_slot = &alpha_and_l_smem[m_tile * M_TILE + row_in_m_tile];
      // FA4 softmax_loop shape, kept from the dense base: the k==0 step is PEELED (is_first_c) so the
      // steady body has no first-block branch. The MASKED specialization is inert here -- VSA selects
      // whole 128-key blocks and is non-causal, so masked_c is always false_type below.
      auto softmax_step = [&](int k, auto masked_c, auto is_first_c) {
        constexpr bool MASKED   = decltype(masked_c)::value;
        constexpr bool IS_FIRST = decltype(is_first_c)::value;
        const int k_offset = 0; (void)k_offset;   // VSA: gathered block, no seqlen walk; no mask
        // VSA MEMBERSHIP: at union position k this m-tile either selected the block (member: normal
        // online softmax over the full 128-key block) or did not (skip: publish alpha, store an
        // all-zero P, leave m/l alone). Membership is per m-tile, so all 128 rows agree.
        const bool member = ((membership_row[k] >> membership_bit) & 1) != 0;
        wp_begin(wpc, WP_SM_WAIT_S);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_spo[m_tile]), spo_ph.get_phase());
        wp_end(wpc, WP_SM_WAIT_S);

        wp_begin(wpc, WP_SM_SOFTMAX);
        float alpha = 0.0f;
        float lt = 0.0f;
        uint32_t p_regs[K_TILE / 2];
        if (member) {
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

        // VSA: no column mask -- every selected block is a full 128 keys and attention is non-causal.
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

        // scale_subtract_rowmax + apply_exp2_convert (FA4 form): exp2 IN PLACE on the f32 scores;
        // bf16 pack per 32-elt fragment AFTER that fragment's exp2s.
        // ROW-SUM IS LIVE HERE, not deferred past the P store as in the dense inline base. Deferring
        // it requires s_regs (128 registers of a 176-register budget) to stay live across the P
        // stores, and on THIS kernel that measured as a 40% regression: the membership skip means
        // ~63% of union positions (x2.7 inflation) do no work, so a live range spanning every
        // iteration is pure cost. Keeping the sum in the loop bounds s_regs to the member path.
        const float2 scale2 = f32x2_splat(scale_log2);
        const float2 neg_m_scaled2 = f32x2_splat(-row_max_safe * scale_log2);
        float2 lt2 = make_float2(0.f, 0.f);
        #pragma unroll
        for (int jj = 0; jj < EX2_FRG_CNT; ++jj) {
          #pragma unroll
          for (int cc = 0; cc < EX2_FRG_PAIRS; ++cc) {
            const int c = jj * EX2_FRG_PAIRS + cc;
            const float2 a2 = ffma2(scores2[c], scale2, neg_m_scaled2);
            float2 e2;
            if constexpr (EX2_EMU) {
              const int kk = 2 * cc;
              const bool use_hw = (kk % EX2_FREQ < EX2_FREQ - EX2_RES) || (jj >= EX2_FRG_CNT - 1) || (jj < EX2_START_FRG);
              e2 = use_hw ? make_float2(ex2_approx_f32(a2.x), ex2_approx_f32(a2.y))
                          : ex2_emu_f32x2(a2.x, a2.y);
            } else {
              e2 = make_float2(ex2_approx_f32(a2.x), ex2_approx_f32(a2.y));
            }
            lt2 = fadd2(lt2, e2);
            p_regs[c] = cvt_f32x2_to_bf16x2(e2.x, e2.y);
          }
        }
        lt = lt2.x + lt2.y;
        m_run = new_m;
        } else {
          // Non-member union position: alpha = 1 lets corr's __all_sync(alpha == 1.0f) skip the O
          // rescale entirely; k == 0 publishes alpha = 0 so l_run stays 0. P is all zeros, and
          // m_run / the row-sum are untouched.
          alpha = IS_FIRST ? 0.0f : 1.0f;
          *alpha_slot = alpha;
          if constexpr (FULL_NAMED_BAR) full_bar_arrive(m_tile, warp_in_group);
          else mbarrier_arrive(smem_ptr_u32(&full_bar_alpha[m_tile]));
          #pragma unroll
          for (int c = 0; c < K_TILE / 2; ++c) p_regs[c] = 0u;
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
        // Row-sum was accumulated live in the exp2 loop (see the note there); alpha == 0 on the
        // peeled first block makes this l_run = lt, and a non-member step contributes lt == 0.
        l_run = alpha * l_run + lt;
      };
      softmax_step(0, std::false_type{}, std::true_type{});
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

// CPU reference: VSA fine block-sparse attention in fp32, over the PER-Q-BLOCK top-k
// lists (NOT the union -- the union + membership mask must reproduce the per-q-block
// semantics exactly). Same math as the 1CTA VSA's cpu_vsa_ref. Layout: Q/K/V natural
// [token, head, hd]; token = b*S + local. gqb = (b*H+h)*nb + qblk.
static void cpu_vsa_ref(const __nv_bfloat16* hQ, const __nv_bfloat16* hK,
                        const __nv_bfloat16* hV, float* hO,
                        int B, int H, int S, int hd, int nb, int max_kv,
                        const int* q2k_idx, const int* q2k_num) {
  const float scale = 1.0f / sqrtf((float)hd);
  const long Nq = (long)B * S;
  for (long i = 0; i < Nq * H * hd; ++i) hO[i] = 0.f;

  #pragma omp parallel for schedule(dynamic)
  for (int bhq = 0; bhq < B * H * nb; ++bhq) {
    const int qblk = bhq % nb;
    const int bh   = bhq / nb;
    const int h    = bh % H;
    const int b    = bh / H;
    const int gqb  = bhq;                        // (b*H + h)*nb + qblk
    const int nkv  = q2k_num[gqb];

    for (int qi = 0; qi < BLOCK; ++qi) {
      const long qp = (long)b * S + (long)qblk * BLOCK + qi;
      std::vector<float> z((size_t)nkv * BLOCK);
      float row_max = -INFINITY;
      int idx = 0;
      for (int kk = 0; kk < nkv; ++kk) {
        const int blk = q2k_idx[gqb * max_kv + kk];
        for (int kj = 0; kj < BLOCK; ++kj) {
          const long kp = (long)b * S + (long)blk * BLOCK + kj;
          float dot = 0.f;
          for (int e = 0; e < hd; ++e)
            dot += __bfloat162float(hQ[(qp * H + h) * hd + e])
                 * __bfloat162float(hK[(kp * H + h) * hd + e]);
          z[idx++] = dot * scale;
          row_max = fmaxf(row_max, z[idx - 1]);
        }
      }
      float sum = 0.f;
      for (int j = 0; j < nkv * BLOCK; ++j) { z[j] = expf(z[j] - row_max); sum += z[j]; }
      if (sum == 0.f) continue;
      const float inv_sum = 1.f / sum;
      for (int e = 0; e < hd; ++e) {
        float acc = 0.f;
        idx = 0;
        for (int kk = 0; kk < nkv; ++kk) {
          const int blk = q2k_idx[gqb * max_kv + kk];
          for (int kj = 0; kj < BLOCK; ++kj) {
            const long kp = (long)b * S + (long)blk * BLOCK + kj;
            acc += z[idx++] * __bfloat162float(hV[(kp * H + h) * hd + e]);
          }
        }
        hO[(qp * H + h) * hd + e] = acc * inv_sum;
      }
    }
  }
}

// Deterministic fill in [-1, 1). xorshift-mixed hash so the sequence has no short period that
// could alias with H*hd and make every token identical. Same as the 1CTA VSA harness.
static void fillr(__nv_bfloat16* h, long n, unsigned seed) {
  block_sparse_bf16_benchmark::fill(h, n, seed);
}

struct Sh { int B, H, nb, topk, hd; const char* lab; };

static double run(const Sh& sh, bool verify) {
  const int  B = sh.B, H = sh.H, nb = sh.nb, topk = sh.topk, hd = sh.hd;
  const int  S = nb * BLOCK;                       // seqlen (multiple of 128)
  const int  max_kv = topk;                        // tight: exactly topk selected blocks
  const long tq = (long)B * S;                     // total tokens
  const int  pairs_per_seq = nb / 4;               // a CLUSTER does 4 adjacent q-blocks
  const int  total_qblk = B * H * nb;
  const int  npairs = B * H * pairs_per_seq;       // = total clusters
  // LIST_CORR (0..100): percent of each q-block's list drawn from a shared per-pair pool.
  // 0 = independent random lists (EXACTLY the 1CTA harness sequence), 100 = identical lists.
  const int  lc = getenv("LIST_CORR") ? atoi(getenv("LIST_CORR")) : 0;
  const int  n_shared = topk * lc / 100;
  if (nb % 4 != 0) { printf("  [%s] SKIP: nb must be a multiple of 4 (2SM 4-q-block clusters)\n", sh.lab); return 0.0; }
  if (topk < 1 || topk > nb) { printf("  [%s] SKIP: need 1 <= topk <= nb\n", sh.lab); return 0.0; }

  // ---- device buffers (bf16; V stored transposed as V_T for the BMM2 TMA) ----
  __nv_bfloat16 *dQ, *dK, *dVT, *dO;
  CUDA_CHECK(cudaMalloc(&dQ,  tq * H * hd * 2));
  CUDA_CHECK(cudaMalloc(&dK,  tq * H * hd * 2));
  CUDA_CHECK(cudaMalloc(&dVT, (long)H * hd * tq * 2));
  CUDA_CHECK(cudaMalloc(&dO,  tq * H * hd * 2));

  std::vector<__nv_bfloat16> hQ(tq * H * hd), hK(tq * H * hd), hV(tq * H * hd);
  const char* load_npy = getenv("LOAD_NPY");   // unified bench: shared Q/K/V + idx from .npy
  if (load_npy) {
    const std::string d(load_npy);
    auto ld = [&](const char* nm, std::vector<__nv_bfloat16>& h) {
      char p[64]; snprintf(p, sizeof p, "/%s_S%d.npy", nm, S);
      auto bits = npy_load_vec<uint16_t>(d + p);
      if (bits.size() != h.size()) { fprintf(stderr, "LOAD_NPY: %s size %zu != %zu\n", nm, bits.size(), h.size()); exit(1); }
      memcpy(h.data(), bits.data(), h.size() * 2);   // raw bf16 bits
    };
    ld("q", hQ); ld("k", hK); ld("v", hV);
  } else {
    fillr(hQ.data(), hQ.size(), 11);
    fillr(hK.data(), hK.size(), 22);
    fillr(hV.data(), hV.size(), 33);
  }

  // V -> V_T transpose ([tok,head,hd] -> [head,hd,tok])
  std::vector<__nv_bfloat16> hVT((long)H * hd * tq, __float2bfloat16(0.f));
  for (long idx = 0; idx < tq; ++idx)
    for (int h = 0; h < H; ++h)
      for (int d = 0; d < hd; ++d)
        hVT[(h * hd + d) * tq + idx] = hV[(idx * H + h) * hd + d];

  CUDA_CHECK(cudaMemcpy(dQ,  hQ.data(),  hQ.size()  * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dK,  hK.data(),  hK.size()  * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dVT, hVT.data(), hVT.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dO, 0, hQ.size() * 2));

  // ---- q2k index: topk DISTINCT block ids per (b,h,qblk), fixed density ----
  // Same xorshift + partial Fisher-Yates as the 1CTA VSA harness. LIST_CORR pre-places
  // n_shared ids from a per-pair pool at the front of every list of the pair, then the
  // remaining topk - n_shared slots are drawn by the 1CTA Fisher-Yates over the rest.
  std::vector<int> hq2k_idx((size_t)total_qblk * max_kv, 0);
  std::vector<int> hq2k_num(total_qblk, topk);
  if (load_npy) {   // unified bench: head-independent [nb, topk] list, broadcast across heads
    char p[64]; snprintf(p, sizeof p, "/idx_S%d_blk%d.npy", S, BLOCK);
    auto idx = npy_load_vec<int32_t>(std::string(load_npy) + p);
    if (idx.size() != (size_t)nb * topk) { fprintf(stderr, "LOAD_NPY: idx size %zu != %d\n", idx.size(), nb * topk); exit(1); }
    for (int gqb = 0; gqb < total_qblk; ++gqb) {
      const int qblk = gqb % nb;
      for (int i = 0; i < topk; ++i) hq2k_idx[(size_t)gqb * max_kv + i] = idx[(size_t)qblk * topk + i];
    }
  } else if (lc == 0) {
    block_sparse_bf16_benchmark::select_blocks(hq2k_idx, nb, topk);
  } else {
    std::vector<int> perm(nb), pool(topk);
    for (int pr = 0; pr < npairs; ++pr) {
      const int p  = pr % pairs_per_seq;
      const int bh = pr / pairs_per_seq;
      if (n_shared > 0) {   // per-pair shared pool (its own xorshift stream)
        for (int i = 0; i < nb; ++i) perm[i] = i;
        uint32_t st = (uint32_t)pr * 2654435761u ^ 0x9e3779b9u;
        for (int i = 0; i < topk; ++i) {
          st ^= st << 13; st ^= st >> 17; st ^= st << 5;
          const int j = i + (int)(st % (uint32_t)(nb - i));
          const int t = perm[i]; perm[i] = perm[j]; perm[j] = t;
          pool[i] = perm[i];
        }
      }
      for (int o = 0; o < 4; ++o) {
        const int gqb = bh * nb + 4 * p + o;
        for (int i = 0; i < nb; ++i) perm[i] = i;
        for (int i = 0; i < n_shared; ++i) {     // pre-place the shared prefix
          int j = i; while (perm[j] != pool[i]) ++j;
          const int t = perm[i]; perm[i] = perm[j]; perm[j] = t;
          hq2k_idx[(size_t)gqb * max_kv + i] = perm[i];
        }
        uint32_t st = (uint32_t)gqb * 2654435761u + 12345u;   // 1CTA seed
        for (int i = n_shared; i < topk; ++i) {               // partial Fisher-Yates
          st ^= st << 13; st ^= st >> 17; st ^= st << 5;      // xorshift32
          const int j = i + (int)(st % (uint32_t)(nb - i));
          const int t = perm[i]; perm[i] = perm[j]; perm[j] = t;
          hq2k_idx[(size_t)gqb * max_kv + i] = perm[i];
        }
      }
    }
  }

  // ---- union list + membership per m-pair (cluster work-item) ----
  // union_idx [npairs, max_union] = sorted distinct ids of the pair's 4 lists;
  // union_num [npairs]; membership [npairs, max_union] u8, bit o = q-block 4p + o
  // (o = peer*2 + m_tile in the kernel).
  const int max_union = (4 * topk < nb) ? 4 * topk : nb;
  std::vector<int>     hunion_idx((size_t)npairs * max_union, 0);
  std::vector<int>     hunion_num(npairs, 0);
  std::vector<uint8_t> hmember((size_t)npairs * max_union, 0);
  long sum_union = 0;
  {
    std::vector<uint8_t> bits(nb);
    for (int pr = 0; pr < npairs; ++pr) {
      const int p  = pr % pairs_per_seq;
      const int bh = pr / pairs_per_seq;
      std::fill(bits.begin(), bits.end(), (uint8_t)0);
      for (int o = 0; o < 4; ++o) {
        const int gqb = bh * nb + 4 * p + o;
        for (int i = 0; i < topk; ++i)
          bits[hq2k_idx[(size_t)gqb * max_kv + i]] |= (uint8_t)(1u << o);
      }
      int u = 0;
      for (int id = 0; id < nb; ++id) {
        if (bits[id]) {
          hunion_idx[(size_t)pr * max_union + u] = id;
          hmember[(size_t)pr * max_union + u] = bits[id];
          ++u;
        }
      }
      hunion_num[pr] = u;
      sum_union += u;
    }
  }
  const double inflation = (double)sum_union / ((double)npairs * topk);

  // The kernel consumes ONLY the union arrays; the per-q-block lists stay host-side
  // (union build above + CPU reference below).
  int *dunion_idx, *dunion_num;
  uint8_t* dmember;
  CUDA_CHECK(cudaMalloc(&dunion_idx, hunion_idx.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dunion_num, hunion_num.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dmember,    hmember.size()));
  CUDA_CHECK(cudaMemcpy(dunion_idx, hunion_idx.data(), hunion_idx.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dunion_num, hunion_num.data(), hunion_num.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dmember,    hmember.data(),    hmember.size(),                  cudaMemcpyHostToDevice));

  // ---- TMA tensor maps ----
  // Q/O: 4D per-sample map over [hd, H, token-IN-SAMPLE, sample]; box [64, 1, 128, 1]
  // (MHA: head box dim 1). The per-sample token dim mirrors the dense 2SM kernel.
  CUtensorMap tq_, tk_, tvt_, to_;
  // O store: 4D per-sample [hd, H, token-in-sample, sample] (epi warp issues tma_store_4d).
  {
    uint64_t gd[4] = { (uint64_t)hd, (uint64_t)H, (uint64_t)S, (uint64_t)B };
    uint64_t gs[3] = { (uint64_t)hd * 2u, (uint64_t)H * hd * 2u, (uint64_t)S * H * hd * 2u };
    uint32_t bd[4] = { (uint32_t)SUB_COLS_BF16, 1u, (uint32_t)M_TILE, 1u };
    uint32_t es[4] = { 1u, 1u, 1u, 1u };
    CUresult r = cuTensorMapEncodeTiled(&to_, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, dO, gd, gs, bd, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUDA_CHECK(r == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue);
  }
  // Q load: 5D -- the inline load warp folds BOTH hd-atoms into the OUTERMOST box dim so ONE TMA
  // fills the whole head_dim (tma_load_5d_2sm), instead of the 2x 4D loop the base VSA kernel used.
  {
    uint64_t gd[5] = { (uint64_t)SUB_COLS_BF16, (uint64_t)H, (uint64_t)S, (uint64_t)B, (uint64_t)Q_SUBTILES };
    uint64_t gs[4] = { (uint64_t)hd * 2u, (uint64_t)H * hd * 2u, (uint64_t)S * H * hd * 2u,
                       (uint64_t)SUB_COLS_BF16 * 2u };
    uint32_t bd[5] = { (uint32_t)SUB_COLS_BF16, 1u, (uint32_t)M_TILE, 1u, (uint32_t)Q_SUBTILES };
    uint32_t es[5] = { 1u, 1u, 1u, 1u, 1u };
    CUresult r = cuTensorMapEncodeTiled(&tq_, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, dQ, gd, gs, bd, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUDA_CHECK(r == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue);
  }
  // K: dims [atom-col 64, token tq, atom (H*hd)/64]; 2SM half-box token dim (K_TILE/2 = 64):
  // each peer loads its 64-token half of the SAME selected block, both hd-atoms in one TMA.
  {
    uint64_t gd[3] = { (uint64_t)SUB_COLS_BF16, (uint64_t)tq, (uint64_t)((long)H * hd / SUB_COLS_BF16) };
    uint64_t gs[2] = { (uint64_t)((long)H * hd) * 2u, (uint64_t)SUB_COLS_BF16 * 2u };
    uint32_t bd[3] = { (uint32_t)SUB_COLS_BF16, (uint32_t)(K_TILE / 2), (uint32_t)K_SUBTILES };
    uint32_t es[3] = { 1u, 1u, 1u };
    CUresult r = cuTensorMapEncodeTiled(&tk_, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, dK, gd, gs, bd, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUDA_CHECK(r == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue);
  }
  // V_T: 3D, the inline FA4 form -- the token dim is split (chunk, inner) so ONE TMA per V half-tile
  // fills smem [chunk][row][token]; box [inner 64, hd/2 rows, 2 chunks] (D-split half per peer).
  // Legal because every gathered block base is a multiple of BLOCK = 128, hence of SUB_COLS_BF16.
  {
    uint64_t gd[3] = { (uint64_t)SUB_COLS_BF16, (uint64_t)((long)H * hd), (uint64_t)(tq / SUB_COLS_BF16) };
    uint64_t gs[2] = { (uint64_t)tq * 2u, (uint64_t)SUB_COLS_BF16 * 2u };
    uint32_t bd[3] = { (uint32_t)SUB_COLS_BF16, (uint32_t)(hd / 2), 2u };
    uint32_t es[3] = { 1u, 1u, 1u };
    CUresult r = cuTensorMapEncodeTiled(&tvt_, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, dVT, gd, gs, bd, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUDA_CHECK(r == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue);
  }

  // ---- shared memory budget (same shape as the dense 2SM kernel) ----
  constexpr bool FULL_NAMED_BAR = false, EX2_EMU = true, SPLIT_P = true,
                 SOFTMAX_THROTTLE = false, USE_CLC = false, Q_RASTER = true;
  const size_t smem =
        (size_t)2 * Q_TILE_BYTES + NUM_KV_STAGES * KV_SLOT_BYTES  // Q (x2) + K/V ring (16 KB half-box
                                                                  // per peer x 6 stages = 96 KB)
      + (size_t)2 * M_TILE * HEAD_DIM * sizeof(__nv_bfloat16)     // 2 sO bufs for TMA-O
      + (2 * NUM_KV_STAGES + 26) * 8                              // mbarriers (incl o_epi + throttle x4)
      + (USE_CLC ? (size_t)CLC_STAGES * (2 * 8 + 16) + 16 : 0)    // CLC: full+empty + response (16B aligned)
      + 8                                                         // tmem_slot
      + 16                                                        // tmem_dealloc_bar [1] + pad
      + (size_t)2 * M_TILE * sizeof(float)                        // alpha_and_l_smem [2][M_TILE]
      + 256;                                                      // slack / alignment

  auto kfn = &fmha_context_bf16_vsa_2sm_kernel<32, FULL_NAMED_BAR, EX2_EMU, SPLIT_P, SOFTMAX_THROTTLE, USE_CLC, Q_RASTER>;
  CUDA_CHECK(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  CUDA_CHECK(cudaFuncSetAttribute(kfn, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

  const float scale_log2 = (1.0f / sqrtf((float)hd)) * (float)M_LOG2E;
  int numSM = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, 0));
  int nblk;
  if (USE_CLC) {
    nblk = npairs * 2;                       // full problem: one cluster per m-pair
  } else {
    nblk = std::min(npairs * 2, numSM);
    nblk -= (nblk & 1);                      // grid.x must be a multiple of cluster.x = 2
  }
  dim3 grid(nblk, 1, 1), block(N_WARPS * 32, 1, 1);

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = smem; cfg.stream = 0;
  cudaLaunchAttribute cfgAttr[1];
  cfgAttr[0].id = cudaLaunchAttributeClusterDimension;
  cfgAttr[0].val.clusterDim.x = 2; cfgAttr[0].val.clusterDim.y = 1; cfgAttr[0].val.clusterDim.z = 1;
  cfg.attrs = cfgAttr; cfg.numAttrs = 1;
  const unsigned long long magic0 = make_magic((unsigned)(H * pairs_per_seq));
  const unsigned long long magic1 = make_magic((unsigned)H);
  const unsigned long long magic2 = make_magic((unsigned)pairs_per_seq);
  auto launch = [&](cudaStream_t stream = nullptr) {
    cfg.stream = stream;
    return cudaLaunchKernelEx(&cfg, kfn, tq_, tk_, tvt_, to_, S, H, scale_log2,
                              B, pairs_per_seq, max_union,
                              magic0, magic1, magic2,
                              (const int*)dunion_idx, (const int*)dunion_num, (const uint8_t*)dmember);
  };

  const double ms = block_sparse_bf16_benchmark::measure(launch);
#ifdef WARP_PROF
  {
    WpBuffer wp = wp_alloc(grid);
    const unsigned pblk = wp.view_block;
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    wp_readback(wp);
    const char *roles[16] = {"sm0",  "sm0",  "sm0",  "sm0",  "sm1", "sm1", "sm1",  "sm1",
                             "corr", "corr", "corr", "corr", "mma", "epi", "load", "sched"};
    printf("  [%s] WARP_PROF block %u:\n", sh.lab, pblk);
    wp_print_busy(wp, roles, 16, pblk);
    wp_dump_raw(wp, "warp_raw_vsa_2sm.bin.gz", pblk, NUM_KV_STAGES);
    wp_free(wp);
  }
#endif

  // selected: true useful FLOPs from the per-q-block lists.
  // visited: union-visited work (every m-tile of the cluster rides all union positions).
  const double sel_pairs = (double)B * H * nb * BLOCK * (double)topk * BLOCK;
  const double vis_pairs = 4.0 * (double)sum_union * BLOCK * (double)BLOCK;
  const double tflops_sel = block_sparse_bf16_benchmark::tflops(sel_pairs, hd, ms);
  const double tflops_vis = block_sparse_bf16_benchmark::tflops(vis_pairs, hd, ms);
  if (block_sparse_bf16_benchmark::benchmark_enabled()) printf("  [%-9s H%-2d nb%-3d k%-3d S%d lc%d] N_q=%ld  %.4f ms  %.1f TFLOPS(sel)  %.1f TFLOPS(vis)  union x%.3f (avg %.1f blk)\n",
         sh.lab, H, nb, topk, S, lc, tq, ms, tflops_sel, tflops_vis,
         inflation, (double)sum_union / npairs);

  block_sparse_bf16_benchmark::dump_output(dO, hQ.size());

  if (verify) {
    std::vector<__nv_bfloat16> ho(hQ.size());
    CUDA_CHECK(cudaMemcpy(ho.data(), dO, ho.size() * 2, cudaMemcpyDeviceToHost));
    std::vector<float> ref(hQ.size(), 0.f), out(hQ.size());
    cpu_vsa_ref(hQ.data(), hK.data(), hV.data(), ref.data(), B, H, S, hd, nb, max_kv,
                hq2k_idx.data(), hq2k_num.data());
    for (size_t i = 0; i < out.size(); ++i) out[i] = __bfloat162float(ho[i]);
    const bool ok = check_close_f32(ref.data(), out.data(), (int)out.size(), 0.05f, 0.10f);
    printf("  verify [%s]: %s\n", sh.lab, ok ? "OK" : "FAIL");
    if (!ok) std::exit(EXIT_FAILURE);
  }

  // ---- stress mode: STRESS_N launches, cross-run output consistency ----
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
    printf("  STRESS [%s] N=%d: mismatches=%d cuda_errs=%d\n", sh.lab, N, mism, errs);
  }

  cudaFree(dQ); cudaFree(dK); cudaFree(dVT); cudaFree(dO);
  cudaFree(dunion_idx); cudaFree(dunion_num); cudaFree(dmember);
  return tflops_sel;
}

int main() {
  CUDA_CHECK(cudaFree(0));
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 64 * 1024 * 1024));
  printf("VSA fine block-sparse FMHA bf16 2SM (cta_group::2, blk128, union list + membership mask) sm_100a\n"
         "=====================================\n");

  // shapes: {B, H, nb, topk, hd, label}. nb must be a multiple of 4.
  Sh shapes[] = {
    {1,  4,  8,  4, 128, "small"},
    {1, 16, 32,  8, 128, "fastvideo"},
    {1,  8, 64, 16, 128, "25pct"},
  };
  const bool verify = getenv("NOVERIFY") ? false : true;

  // single-shape override: SHAPE=0..2 plus BATCH/HEADS/NB/TOPK/LIST_CORR env knobs.
  if (const char* s = getenv("SHAPE")) {
    Sh sh = shapes[atoi(s) % 3];
    if (getenv("BATCH")) sh.B    = atoi(getenv("BATCH"));
    if (getenv("HEADS")) sh.H    = atoi(getenv("HEADS"));
    if (getenv("NB"))    sh.nb   = atoi(getenv("NB"));
    if (getenv("TOPK"))  sh.topk = atoi(getenv("TOPK"));
    sh.lab = "custom";
    run(sh, verify);
    return 0;
  }
  const int B = getenv("BATCH") ? atoi(getenv("BATCH")) : 1;
  for (Sh sh : shapes) { sh.B = B; run(sh, verify); }
  return 0;
}
