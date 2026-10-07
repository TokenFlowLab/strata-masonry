// block_sparse_bf16_varlen.cu -- VSA "fine" block-sparse FMHA with VARIABLE BLOCK SIZES
// (the ragged/varlen VSA), bf16, sm_100a.
//
// Same kernel as block_sparse_bf16_uniform.cu (see its header for the full lineage: 16-warp warp-spec body,
// BMM1-ahead pipeline, split-P, blk64 tcgen05.mma.ws Layout-E dual-pack / blk128 plain m128 via
// the VSA_BLK128 toggle, CLC scheduling, TMA-store epilogue) PLUS variable_block_sizes:
//
//   FastVideo's vbs semantics (from its triton fine kernel _attn_fwd_sparse): the layout is
//   PADDED-STRIDED -- block i occupies the fixed token slice [i*BLOCK, (i+1)*BLOCK) and only its
//   first variable_block_sizes[i] tokens are VALID. The ONLY ragged mechanism is a KEY mask:
//   scores for key j >= vbs[block] are -inf before the online softmax. Query rows are NOT
//   masked: all BLOCK rows of a query block compute and store (full non-ragged tiles), so the
//   TMA-store epilogue is unchanged.
//
// The key mask reuses fmha_context_bf16_varlen.cu's padding-mask primitive mask_s_row_r2p
// (composites/112_fmha_softmax_utils.cuh) per 64-token score segment with threshold vbs[block_id] (it masks keys
// >= seqlen there; keys >= vbs here). Loads are unchanged: full fixed-stride boxes; the invalid
// tail tokens are loaded but masked (exp2(-inf) = 0 -> P cols 0 -> V garbage never reaches O).
// vbs[i] >= 1 is assumed (a fully-empty block would produce NaN rows; FastVideo never selects one).
//
// Barrier contract (unique to the TMA-sO epilogue):
//   - empty_bar_o_epi[m] (count 1): epi -> corr, "sO[m]'s TMA store drained, slot reusable" --
//     see fmha_context_bf16_uniform_inline.cu's header.
//
// Budgets:
//   - Registers: per-warp budgets sum to exactly the SM file (65536 = 128*512): softmax inc<192>
//     (x8) + correction dec<80> (x4) + the four single warps dec<48> (x4). WARP_PROF's
//     wp_begin/wp_end markers fit as-is (single warps ~R26; corr ~R77; softmax ~R186).
//   - TMEM: M_TILES_PER_CTA = 2 = the two ADJACENT q-blocks (2p,2p+1), each with its OWN top-k
//     list; each M-tile needs S + O = 128 + 128 cols; 2*(128+128) = 512.
//
// EX2_EMU hybrid exp2 (FA4 apply_exp2_convert):
//   - Per 32-elt fragment, a fraction of pairs use the f32x2 ALU emulation (ex2_emu_f32x2)
//     instead of MUFU.EX2 to relieve the EX2 pipe.
//   - Gate: HW unless (k%EX2_FREQ >= EX2_FREQ-EX2_RES) AND (fragment < last), where for
//     softmax pair c: fragment j = c/EX2_FRG_PAIRS, in-fragment elt k = 2*(c%EX2_FRG_PAIRS).
//
// MHA (HQ==HK), Sq==Skv, non-causal. d=128, bf16.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include "npy_io.cuh"
#include "block_sparse_bf16_benchmark.cuh"
#include "../../../tests/test_utils.cuh"
#include "../../../primitives/0_tcgen05_alloc.cuh"
#include "../../../primitives/1_tcgen05_dealloc.cuh"
#include "../../../primitives/2_tcgen05_relinquish.cuh"
#include "../../../primitives/3_tcgen05_mma_f16.cuh"
#include "../../../primitives/79_tcgen05_mma_ws_f16.cuh"
#include "../../../primitives/8_tcgen05_mma_idesc.cuh"
#include "../../../primitives/9_tcgen05_ld.cuh"
#include "../../../primitives/10_tcgen05_st.cuh"
#include "../../../primitives/11_tcgen05_commit.cuh"
#include "../../../primitives/12_tcgen05_wait.cuh"
#include "../../../primitives/15_tcgen05_fence.cuh"
#include "../../../primitives/18_tma_load.cuh"
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
#include "../../../primitives/63_cvt_f32_to_f16_bf16.cuh"
#include "../../../composites/118_mbarrier_phase_tracking.cuh"
#include "../../../composites/106_clc_fetch_next_tile.cuh"
#include "../../../primitives/46_setmaxnreg.cuh"
#include "../../../primitives/76_packed_f32x2.cuh"
#include "../../../primitives/77_ex2_approx.cuh"
#include "../../../primitives/78_rcp_approx.cuh"
#include "../../../primitives/_warp_prof_noop.cuh"
#include "../../../composites/109_fastdivmod.cuh"
#include "../../../composites/112_fmha_softmax_utils.cuh"

// ---- blk64 / blk128 toggle -------------------------------------------------------------------
// VSA_BLK128=false (default, "blk64"): 64-token blocks. M=64 MMA needs tcgen05.mma.ws Layout E
//   (2x2 datapath): K_TILE=256 (4 blocks/GEMM, the M=64 ISA max), S(64x256) DUAL-PACKED into 128
//   TMEM cols (lanes 0-63 = kv-half 0 = blocks {0,1}, lanes 64-127 = half 1 = blocks {2,3}). Each
//   half runs an independent online softmax; the two O partials merge in the correction epilogue.
// VSA_BLK128=true ("blk128"): 128-token blocks. M=128 is the NATIVE full-datapath tcgen05.mma --
//   plain instruction, K_TILE=128 (1 block/GEMM), no dual-pack, no half-merge: exactly the dense
//   fmha_context_bf16_uniform_inline.cu tiling (S/O = 4x128 TMEM cols) with the VSA block-id gather.
#ifndef VSA_BLK128
#define VSA_BLK128 false
#endif
constexpr bool BLK128 = VSA_BLK128;
// Hot-spin softmax waits (inline-base uplift): the mma waits are hot-spin
// unconditionally (f22fb2e-era finding: suspend-hint wake latency gates the
// pipeline); softmax hot-spin is a knob -- the dense kernel gates it
// !MHA && !IS_CAUSAL, VSA's regime differs (short per-item loops), so bench it.
#ifndef VSA_SM_HOT
#define VSA_SM_HOT true
#endif
// Deferred row-sum (uplift #6) helps blk64 (dual-half softmax) but costs ~2% on
// blk128 (plain path): default per block mode, override with -DVSA_DEFER_ROWSUM.
#ifndef VSA_DEFER_ROWSUM
#define VSA_DEFER_ROWSUM (!VSA_BLK128)
#endif
constexpr bool DEFER_ROWSUM = VSA_DEFER_ROWSUM;
// ----------------------------------------------------------------------------------------------
constexpr int BLOCK  = BLK128 ? 128 : 64;        // VSA block size (queries AND keys)
constexpr int M_TILE = BLOCK;                    // one query block per M-tile
constexpr int M_TILES_PER_CTA = 2;
constexpr int K_TILE = 256 / (BLK128 ? 2 : 1);      // QK MMA N: blk64 256 (ws), blk128 128 (plain)
constexpr int BLOCKS_PER_KTILE = K_TILE / BLOCK;    // blk64: 4, blk128: 1 blocks per K/V tile
constexpr int KV_HALF = K_TILE / 2;                 // blk64: tokens per datapath half
constexpr int HEAD_DIM = 128;
// B128 swizzle atom = 128 bytes = 64 bf16: all SMEM tiles are laid out in
// 64-wide sub-tiles along the contiguous dim.
constexpr int SUB_COLS_BF16 = 64;
constexpr int SUB_COLS_BYTES = SUB_COLS_BF16 * (int)sizeof(__nv_bfloat16);   // 128 B (one swizzle atom)
constexpr int Q_SUBTILES = HEAD_DIM / SUB_COLS_BF16;     // 2 (Q tile: M_TILE rows x HEAD_DIM)
constexpr int K_SUBTILES = HEAD_DIM / SUB_COLS_BF16;     // 2 (K tile hd-atoms)
constexpr int V_SUBTILES = (BLK128 ? K_TILE : KV_HALF) / SUB_COLS_BF16;  // 2 (V token-atoms)
constexpr int Q_SUB_COLS_BYTES = M_TILE * SUB_COLS_BYTES;         // Q subtile stride: blk64 8 KB, blk128 16 KB
// The K/V ring is a ring of 32 KB slots.
//   blk64: a K or V group = TWO PLANE slots with independent full/empty barriers (K = 2 hd-atom
//     planes of all 4 blocks; V = 2 within-half token planes), so the MMA starts on plane 0 while
//     plane 1's TMA is still landing.
//   blk128: a K or V tile = ONE slot (K: 1 block x 128 hd = 2 hd-atoms in-slot; V: 2 token-atoms
//     in-slot) -- the dense uniform_inline.cu tile shape.
constexpr int SLOT_BYTES = 32 * 1024;
constexpr int SLOTS_PER_TILE = BLK128 ? 1 : 2;      // ring slots consumed per K (or V) group
constexpr int BLK_SUB_BYTES = BLOCK * SUB_COLS_BYTES;    // one block's tokens within a K hd-atom
constexpr int V_BLK_BYTES = HEAD_DIM * SUB_COLS_BYTES;   // 16 KB: one 64-token V^T atom (128 hd x 64 tok)
constexpr int K_SUB_COLS_BYTES = K_TILE * SUB_COLS_BYTES;            // blk128: hd-atom stride inside the K slot (16 KB)
constexpr int Q_TILE_BYTES = Q_SUBTILES * Q_SUB_COLS_BYTES;     // blk64 16 KB, blk128 32 KB
// matmul-K atoms in one 64-wide (sub)tile: tcgen05 contracts K=16 per atom, so 64/16 = 4.
constexpr int K_ATOMS_PER_TILE = SUB_COLS_BF16 / 16;    // 4
constexpr int S_COLS = 128;                      // FP32 S per M-tile: 128 TMEM cols (blk64 dual-packed)
constexpr int SPLIT_P_N    = S_COLS / 4 * 3;            // 96
constexpr int SPLIT_P_ATOM = SPLIT_P_N / 16;            // 6 (BMM2 atom at the split)
constexpr int SPLIT_P_COL  = SPLIT_P_N / 2;     // 48 (u32 P cols written before the empty_bar_spo signal)
constexpr int EX2_FRG_PAIRS = 16;          // 32 elts / fragment = 16 pairs
constexpr int EX2_FRG_CNT   = S_COLS / 32; // = 4 (per-thread 128 score cols)
constexpr int EX2_FREQ      = 16;          // FA4 ex2_emu_freq
constexpr int EX2_RES       = 4;           // FA4 ex2_emu_res
// Ring depth in 32 KB slots. SMEM cap 227K: blk64 2*Q(32K) + 4*32K + 2*sO(32K); blk128
// 2*Q(64K) + 3*32K + 2*sO(64K) (the dense kernel also ran 3 stages at this tile size).
constexpr int NUM_KV_STAGES = BLK128 ? 3 : 4;
constexpr int O_COLS = HEAD_DIM;                 // blk64: O dual (per-half partials); blk128: plain
constexpr int TMEM_TOTAL = 512;                     // S0,S1(128*2)+O0,O1(128*2)=512
// Softmax->correction stats: 128 per M-tile (blk64: 64 rows x 2 halves; blk128: 128 rows).
// Regions: [0] alpha, [1] l, and for blk64 only [2] m (the half-merge needs it).
constexpr int STATS = BLK128 ? M_TILE : 2 * M_TILE;   // 128 either way
constexpr int STAT_REGIONS = BLK128 ? 2 : 3;
// f32 exchange buffer for the blk64 half-merge (one 64x16 chunk); absent for blk128.
constexpr int XCHG_FLOATS = BLK128 ? 0 : M_TILE * 16;
constexpr int W_CORR0 = 8, W_MMA = 12, W_EPI = 13, W_LOAD = 14, W_SCHED = 15;
constexpr int N_WARPS = 16;
constexpr int CLC_STAGES = 2;

extern __shared__ __align__(1024) uint8_t fmha_smem[];

// mbarrier_wait_parity: now in primitives/33_mbarrier_try_wait.cuh.

template <bool HOT>
__device__ __forceinline__
void mbarrier_wait_parity_sel(uint32_t mbar_smem, uint32_t phase_parity) {
  if constexpr (HOT) mbarrier_wait_parity(mbar_smem, phase_parity);
  else               mbarrier_wait_parity_suspend(mbar_smem, phase_parity);
}

// Lead-predicated tcgen05 issue forms (inline-base uplift, FA4 shape): the elect
// predicate guards the INSTRUCTION instead of an if(lead) block -- straight-line
// SASS, no BSSY/BSYNC reconvergence region per issue site on the pacing warp.
// ws (Layout E dual-pack) predicated forms -- same pattern over the mma.ws asm
// (primitives/79); the trailing 0 is the zero-column-mask operand.
__device__ __forceinline__ void tcgen05_mma_ws_f16_ss_1sm_lead(uint32_t lead,
    uint32_t tmem_d, uint64_t desc_a, uint64_t desc_b, uint32_t idesc,
    bool enable_input_d) {
  asm volatile(
    "{\n\t"
    ".reg .pred p, q;\n\t"
    "setp.ne.b32 q, %0, 0;\n\t"
    "setp.ne.b32 p, %4, 0;\n\t"
    "@q tcgen05.mma.ws.cta_group::1.kind::f16 [%1], %2, %3, %5, p, 0;\n\t"
    "}\n"
    :: "r"(lead), "r"(tmem_d), "l"(desc_a), "l"(desc_b),
       "r"(enable_input_d ? 1u : 0u), "r"(idesc));
}
__device__ __forceinline__ void tcgen05_mma_ws_f16_ts_1sm_lead(uint32_t lead,
    uint32_t tmem_d, uint32_t tmem_a, uint64_t desc_b, uint32_t idesc,
    bool enable_input_d) {
  asm volatile(
    "{\n\t"
    ".reg .pred p, q;\n\t"
    "setp.ne.b32 q, %0, 0;\n\t"
    "setp.ne.b32 p, %4, 0;\n\t"
    "@q tcgen05.mma.ws.cta_group::1.kind::f16 [%1], [%2], %3, %5, p, 0;\n\t"
    "}\n"
    :: "r"(lead), "r"(tmem_d), "r"(tmem_a), "l"(desc_b),
       "r"(enable_input_d ? 1u : 0u), "r"(idesc));
}


// VSA work-item decode. tile_id -> (sample, head, PAIR of adjacent q-blocks (2p, 2p+1)).
// gqb = global q-block index (sample*H + head)*nb + qblk, the row of q2k_idx / q2k_num.
// nkv (selected KV blocks) is the same for both q-blocks (fixed density).
// Q_RASTER=true: q-block pair INNERMOST (concurrent CTAs sit within one head -> its KV/topk
// blocks stay hot in L2). false: head innermost (concurrent CTAs spread across all heads).
struct VsaItem { int sample, head, qblk0, qblk1, gqb0, gqb1, nkv; };
template <bool Q_RASTER>
__device__ __forceinline__ VsaItem vsa_decode(
    int tile_id, int num_heads, int nb, int packed_mtiles_per_seq, const int* q2k_num) {
  VsaItem it;
  const int per_sample = num_heads * packed_mtiles_per_seq;
  it.sample = tile_id / per_sample;
  const int rem = tile_id - it.sample * per_sample;
  int p;
  if constexpr (Q_RASTER) {
    it.head = rem / packed_mtiles_per_seq;
    p       = rem - it.head * packed_mtiles_per_seq;
  } else {
    p       = rem / num_heads;
    it.head = rem - p * num_heads;
  }
  it.qblk0 = 2 * p;
  it.qblk1 = 2 * p + 1;
  it.gqb0  = (it.sample * num_heads + it.head) * nb + it.qblk0;
  it.gqb1  = it.gqb0 + 1;
  it.nkv   = q2k_num[it.gqb0];
  return it;
}

// Compile-time kernel config (template args, set in run()'s `constexpr` block):
//   S_LD_COLS        : cols per softmax tcgen05.ld of the S row (32/64 compile; 128 aborts ptxas).
//   FULL_NAMED_BAR   : softmax->corr "scale ready": true = HW named barrier (per-band), false =
//                      mbarrier (full_bar_alpha/full_bar_l). Both use alpha_and_l_smem.
//   EX2_EMU          : route a fraction of softmax exp2 through FFMA f32x2 emulation (vs MUFU.EX2).
//   SPLIT_P          : softmax publishes P in two chunks (96 + 32 cols); BMM2 starts on the
//                      first chunk, full_bar_p_last gates the tail atoms.
//   SOFTMAX_THROTTLE : FA4 pacing -- corr defers releasing the alpha/l slot until after it consumes,
//                      holding softmax ~1 stage behind correction.
//   USE_CLC          : true = CLC work-stealing sched (w15 sched warp, grid=full problem); false =
//                      static grid-stride loop (w15 idle, grid=#SMs). Same pipe.
//   Q_RASTER         : true = q-block pair innermost (see vsa_decode above); false = head innermost.
//   MHA              : HQ==HK (VSA MHA: q2k_idx is per (sample, head, q-block)). GQA is future work.
// (uniform's IS_CAUSAL / LPT do not apply: block-sparse, non-causal.)
template <int S_LD_COLS = 32, bool FULL_NAMED_BAR = false, bool EX2_EMU = false, bool SPLIT_P = false,
          bool SOFTMAX_THROTTLE = false, bool USE_CLC = true, bool Q_RASTER = true, bool MHA = true>
__global__ void __cluster_dims__(1, 1, 1) __launch_bounds__(N_WARPS * 32, 1)
fmha_context_bf16_gen_kernel(const __grid_constant__ CUtensorMap tmap_q,
    const __grid_constant__ CUtensorMap tmap_k, const __grid_constant__ CUtensorMap tmap_v_t,
    const __grid_constant__ CUtensorMap tmap_o, int seqlen,
    int num_heads, float scale_log2, int num_samples, int nb,
    int packed_mtiles_per_seq, int max_kv,
    const int* __restrict__ q2k_idx, const int* __restrict__ q2k_num,
    const int* __restrict__ variable_block_sizes) {
  const int total_tiles = num_samples * num_heads * packed_mtiles_per_seq;

  uint8_t* sQ0 = fmha_smem;
  uint8_t* sQ1 = sQ0 + Q_TILE_BYTES;
  uint8_t* sQ[2] = { sQ0, sQ1 };
  uint8_t* sKV = sQ1 + Q_TILE_BYTES;
  __nv_bfloat16* sO0 = reinterpret_cast<__nv_bfloat16*>(sKV + NUM_KV_STAGES * SLOT_BYTES);
  __nv_bfloat16* sO1 = sO0 + M_TILE * HEAD_DIM;
  __nv_bfloat16* sO_bufs[2] = { sO0, sO1 };
  float* o_xchg = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(sO1) + M_TILE * HEAD_DIM * sizeof(__nv_bfloat16));
  uint64_t* full_bar = reinterpret_cast<uint64_t*>(o_xchg + XCHG_FLOATS);
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
      (reinterpret_cast<uintptr_t>(clc_empty + CLC_STAGES) + 15u) & ~uintptr_t(15u)); // [CLC_STAGES*4]
  uint32_t* tmem_slot = clc_response + CLC_STAGES * 4;
  float* alpha_and_l_smem = reinterpret_cast<float*>(tmem_slot + 2);   // [2][3][STATS]

  uint8_t* smem_kv[NUM_KV_STAGES];
  for (int s = 0; s < NUM_KV_STAGES; ++s) smem_kv[s] = sKV + s * SLOT_BYTES;

  const int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;
  WpCtx wpc = wp_ctx_init();

  if (warp_id == 0) {
    tcgen05_alloc<1>(smem_ptr_u32(tmem_slot), TMEM_TOTAL);
    tcgen05_relinquish_alloc_permit<1>();
  }
  __syncthreads();
  const uint32_t tmem_base = *tmem_slot;

  if (tid == 0) {
    #pragma unroll
    for (int s = 0; s < NUM_KV_STAGES; ++s) {
      mbarrier_init(smem_ptr_u32(&full_bar[s]), 1);
      mbarrier_init(smem_ptr_u32(&empty_bar[s]), 1);
    }
    for (int i = 0; i < 2; ++i) {
      mbarrier_init(smem_ptr_u32(&full_bar_q[i]), 1);
      mbarrier_init(smem_ptr_u32(&empty_bar_q[i]), 1);
      mbarrier_init(smem_ptr_u32(&full_bar_l[i]), 128);
      mbarrier_init(smem_ptr_u32(&full_bar_spo[i]), 1);
      mbarrier_init(smem_ptr_u32(&empty_bar_spo[i]), 256);
      mbarrier_init(smem_ptr_u32(&full_bar_o_acc[i]), 1);
      mbarrier_init(smem_ptr_u32(&full_bar_alpha[i]), 128);
      mbarrier_init(smem_ptr_u32(&empty_bar_alpha_and_l[i]), 128);
      mbarrier_init(smem_ptr_u32(&full_bar_p_last[i]), 128);
      mbarrier_init(smem_ptr_u32(&full_bar_o_epi[i]), 128);
      mbarrier_init(smem_ptr_u32(&empty_bar_o_epi[i]), 1);
    }
    if constexpr (USE_CLC) {
      #pragma unroll
      for (int s = 0; s < CLC_STAGES; ++s) {
        mbarrier_init(smem_ptr_u32(&clc_full[s]), 1);
        mbarrier_init(smem_ptr_u32(&clc_empty[s]), N_WARPS);
      }
      #pragma unroll
      for (int i = 0; i < CLC_STAGES * 4; ++i)
        clc_response[i] = 0;
    }
  }
  fence_mbarrier_init_release_cluster();
  __syncthreads();

  if (warp_id == W_LOAD) {
    setmaxnreg_dec<48>();

    EmptyPhaseTracker<NUM_KV_STAGES> kv_empty_ph;
    EmptyPhaseTracker<1> q_empty_ph;
    [[maybe_unused]] int clc_stage = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;
    int tile_id = (int)blockIdx.x;
    while (true) {
      const VsaItem it = vsa_decode<Q_RASTER>(tile_id, num_heads, nb, packed_mtiles_per_seq, q2k_num);
      const int q_start = it.sample * seqlen;
      const int k_start = it.sample * seqlen;
      const int gqb[2]  = { it.gqb0, it.gqb1 };
      const int qblk[2] = { it.qblk0, it.qblk1 };
      const int ngroups = it.nkv / BLOCKS_PER_KTILE;   // KV tiles; nkv % BLOCKS_PER_KTILE == 0

      // Block-id prefetch: 32 ids per M-tile live in a lane-spread register chunk (one coalesced
      // 128 B read per 16 groups, reused by K AND V). get_id is warp-collective (refill load +
      // shfl broadcast) -> call OUTSIDE elect_one_sync.
      int id_chunk_base0 = -(1 << 30), id_chunk_base1 = -(1 << 30);
      int id_chunk_val0 = 0, id_chunk_val1 = 0;
      auto get_id = [&](int& chunk_base, int& chunk_val, int gqb_mt, int j) -> int {
        if (j < chunk_base || j >= chunk_base + 32) {
          chunk_base = j & ~31;
          const int idx = min(chunk_base + lane, it.nkv - 1);   // clamp: tail lanes stay in-row
          chunk_val = q2k_idx[gqb_mt * max_kv + idx];
        }
        return __shfl_sync(0xffffffffu, chunk_val, j & 31);
      };
      auto block_id = [&](int mt, int j) -> int {
        return mt == 0 ? get_id(id_chunk_base0, id_chunk_val0, gqb[0], j)
                       : get_id(id_chunk_base1, id_chunk_val1, gqb[1], j);
      };
      // Gather one 32 KB PLANE per ring slot. K plane s: all 4 blocks' hd-atom s, block j at
      // j*BLK_SUB_BYTES (blocks {0,1} = kv-half 0, {2,3} = half 1). V plane p: kv-half h's
      // block (2h + p) at h*V_BLK_BYTES.
      auto load_k_plane = [&](int mt, int g, int s) {
        const int kv_stage = kv_empty_ph.get_stage();

        wp_begin(wpc, WP_LOAD_WAIT);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar[kv_stage]), kv_empty_ph.get_phase());
        kv_empty_ph.advance();
        wp_end(wpc, WP_LOAD_WAIT);

        wp_begin(wpc, WP_LOAD_ISSUE_K);
        int kv_tok[BLOCKS_PER_KTILE];
        #pragma unroll
        for (int blk = 0; blk < BLOCKS_PER_KTILE; ++blk)
          kv_tok[blk] = k_start + block_id(mt, g * BLOCKS_PER_KTILE + blk) * BLOCK;   // warp-collective
        if (elect_one_sync()) {
          mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar[kv_stage]), SLOT_BYTES);
          if constexpr (BLK128) {
            // one 3D TMA folds both hd-atoms of the single 128-token block: box [SUB_COLS_BF16, BLOCK, 2]
            tma_load_3d(smem_ptr_u32(smem_kv[kv_stage]), &tmap_k, smem_ptr_u32(&full_bar[kv_stage]),
                        0, kv_tok[0], it.head * K_SUBTILES);
          } else {
            #pragma unroll
            for (int blk = 0; blk < BLOCKS_PER_KTILE; ++blk)
              tma_load_3d(smem_ptr_u32(smem_kv[kv_stage] + blk * BLK_SUB_BYTES),
                          &tmap_k, smem_ptr_u32(&full_bar[kv_stage]),
                          0, kv_tok[blk], it.head * K_SUBTILES + s);   // box [SUB_COLS_BF16, BLOCK, 1]
          }
        }
        wp_end(wpc, WP_LOAD_ISSUE_K);
      };
      auto load_v_plane = [&](int mt, int g, int p) {
        const int kv_stage = kv_empty_ph.get_stage();

        wp_begin(wpc, WP_LOAD_WAIT);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar[kv_stage]), kv_empty_ph.get_phase());
        kv_empty_ph.advance();
        wp_end(wpc, WP_LOAD_WAIT);

        wp_begin(wpc, WP_LOAD_ISSUE_V);
        if constexpr (BLK128) {
          // the single block's V^T (128 hd x 128 tok) = 2 token-atoms in ONE slot
          const int kv_tok0 = k_start + block_id(mt, g) * BLOCK;
          if (elect_one_sync()) {
            mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar[kv_stage]), SLOT_BYTES);
            #pragma unroll
            for (int s = 0; s < V_SUBTILES; ++s)
              tma_load_2d(smem_ptr_u32(smem_kv[kv_stage] + s * V_BLK_BYTES), &tmap_v_t,
                          smem_ptr_u32(&full_bar[kv_stage]), kv_tok0 + s * SUB_COLS_BF16, it.head * HEAD_DIM);
          }
        } else {
          int kv_tok[2];
          #pragma unroll
          for (int h = 0; h < 2; ++h)   // kv-half h's block (2h + p)
            kv_tok[h] = k_start + block_id(mt, g * BLOCKS_PER_KTILE + 2 * h + p) * BLOCK;
          if (elect_one_sync()) {
            mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar[kv_stage]), SLOT_BYTES);
            #pragma unroll
            for (int h = 0; h < 2; ++h)
              tma_load_2d(smem_ptr_u32(smem_kv[kv_stage] + h * V_BLK_BYTES), &tmap_v_t,
                          smem_ptr_u32(&full_bar[kv_stage]), kv_tok[h], it.head * HEAD_DIM);
          }
        }
        wp_end(wpc, WP_LOAD_ISSUE_V);
      };
      auto load_k = [&](int mt, int g) {
        load_k_plane(mt, g, 0);
        if constexpr (!BLK128) load_k_plane(mt, g, 1);
      };
      auto load_v = [&](int mt, int g) {
        load_v_plane(mt, g, 0);
        if constexpr (!BLK128) load_v_plane(mt, g, 1);
      };

      // Q for both M-tiles, once (elected lane). MHA: box middle dim = 1, head coord = it.head.
      if (elect_one_sync()) {
        #pragma unroll
        for (int m = 0; m < M_TILES_PER_CTA; ++m) {
          mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_q[m]), q_empty_ph.get_phase());
          mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar_q[m]), Q_TILE_BYTES);
          const int tok0 = q_start + qblk[m] * BLOCK;
          #pragma unroll
          for (int s = 0; s < Q_SUBTILES; ++s)
            tma_load_3d(smem_ptr_u32(sQ[m] + s * Q_SUB_COLS_BYTES), &tmap_q, smem_ptr_u32(&full_bar_q[m]),
                        s * SUB_COLS_BF16, it.head, tok0);
        }
        q_empty_ph.advance();
      }

      // Produce KV tiles in the BMM1-ahead consume order (single ring):
      //   K[m0](0),K[m1](0) ; {V[m0](g),K[m0](g+1),V[m1](g),K[m1](g+1)} ; V[m0](last),V[m1](last)
      load_k(0, 0); load_k(1, 0);
      for (int g = 0; g + 1 < ngroups; ++g) {
        load_v(0, g); load_k(0, g + 1);
        load_v(1, g); load_k(1, g + 1);
      }
      load_v(0, ngroups - 1); load_v(1, ngroups - 1);
      if constexpr (USE_CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, /*CTA_GROUP=*/1, /*SUSPEND=*/true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, /*do_release=*/elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        if (!next.valid) break;
        tile_id = next.n_tile;
      } else {
        tile_id += gridDim.x;
        if (tile_id >= total_tiles) break;
      }
    }
  }
  else if (warp_id == W_MMA) {
    setmaxnreg_dec<48>();

    const uint32_t lead = elect_one_sync() ? 1u : 0u;
    constexpr uint32_t DESC_SBO = 1024, DESC_LBO = 16;
    // TMEM layout: S[i] at i*128, O[i] at 256 + i*128 (512 cols total). S is
    // single-buffered; the 2 M-tiles provide the cross-tile overlap instead.
    // blk64: both MMAs are tcgen05.mma.ws m64n256k16 (Layout E dual-pack; PV's N = 256 = the two
    // per-half 128-hd O partials). blk128: plain full-datapath m128.
    const uint32_t idesc_qk = make_idesc_bf16_f32(M_TILE, K_TILE, false, false);
    const uint32_t idesc_pv = make_idesc_bf16_f32(M_TILE, BLK128 ? HEAD_DIM : 2 * HEAD_DIM, false, false);
    const uint64_t desc_q0  = build_smem_desc_blackwell(smem_ptr_u32(sQ0), DESC_SBO, DESC_LBO, SmemSwizzleBlackwell::B128);
    const uint64_t desc_kv0 = build_smem_desc_blackwell(smem_ptr_u32(sKV), DESC_SBO, DESC_LBO, SmemSwizzleBlackwell::B128);
    // >> 4: the SMEM descriptor address field is in 16-byte units (addr >> 4).
    constexpr uint64_t SLOT_DESC_DELTA    = SLOT_BYTES >> 4;
    constexpr uint64_t Q_SUB_DELTA        = Q_SUB_COLS_BYTES >> 4;
    constexpr uint64_t Q_MTILE_DESC_DELTA = Q_TILE_BYTES >> 4;

    PhaseTracker<NUM_KV_STAGES> kv_ph;
    PhaseTracker<1> q_ph;
    PhaseTracker<1> spo_ph;
    [[maybe_unused]] int clc_stage = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;
    int tile_id = (int)blockIdx.x;
    while (true) {
      const VsaItem it = vsa_decode<Q_RASTER>(tile_id, num_heads, nb, packed_mtiles_per_seq, q2k_num);
      const int ngroups = it.nkv / BLOCKS_PER_KTILE;

      // BMM1: S[i] = Q[i] @ K (blk64: mma.ws m64n256k16 dual-pack; blk128: plain m128).
      auto bmm1 = [&](int i) {
        const uint32_t s_tmem_addr = tmem_base + (uint32_t)(i * S_COLS);
        const uint64_t da_base = desc_q0 + (uint64_t)i * Q_MTILE_DESC_DELTA;
        // blk64: each hd-atom is its own ring slot (wait per atom -> atom 0 computes under atom 1's
        // TMA). blk128: one slot holds both hd-atoms (atom s at s*K_SUB_COLS_BYTES in-slot), waited once.
        int slot = 0;
        #pragma unroll
        for (int s = 0; s < Q_SUBTILES; ++s) {
          if (!BLK128 || s == 0) {
            slot = kv_ph.get_stage();
            wp_begin(wpc, WP_MMA_WAIT_FULL_K);
            mbarrier_wait_parity(smem_ptr_u32(&full_bar[slot]), kv_ph.get_phase());
            wp_end(wpc, WP_MMA_WAIT_FULL_K);
            kv_ph.advance();
          }
          wp_begin(wpc, WP_MMA_ISSUE);
          // straight-line issue: address math unguarded (warp-uniform -> URs),
          // the elect predicate rides ON the instruction (no BSSY/BSYNC).
          const uint64_t da = da_base + (uint64_t)s * Q_SUB_DELTA;
          const uint64_t db = desc_kv0 + (uint64_t)slot * SLOT_DESC_DELTA
                            + (BLK128 ? (uint64_t)s * (K_SUB_COLS_BYTES >> 4) : 0u);
          #pragma unroll
          for (int ki = 0; ki < K_ATOMS_PER_TILE; ++ki) {
            const bool enable_d = (s != 0) || (ki != 0);
            if constexpr (BLK128)
              tcgen05_mma_f16_ss_lead(lead, s_tmem_addr, da + 2 * ki, db + 2 * ki, idesc_qk, enable_d);
            else
              tcgen05_mma_ws_f16_ss_1sm_lead(lead, s_tmem_addr, da + 2 * ki, db + 2 * ki, idesc_qk, enable_d);
          }
          if (!BLK128 || s == Q_SUBTILES - 1)
            tcgen05_commit1_lead(lead, smem_ptr_u32(&empty_bar[slot]));
          wp_end(wpc, WP_MMA_ISSUE);
        }

        wp_begin(wpc, WP_MMA_COMMIT);
        tcgen05_commit1_lead(lead, smem_ptr_u32(&full_bar_spo[i]));
        wp_end(wpc, WP_MMA_COMMIT);
      };
      // BMM2: O[i] += P[i] @ V (blk64: datapath half h reads its P from lanes 64h+ and
      // contracts its OWN 128 tokens).
      auto bmm2 = [&](int i, bool first_group, bool last_group) {
        const uint32_t s_tmem_addr = tmem_base + (uint32_t)(i * S_COLS);
        const uint32_t o_tmem_addr = tmem_base + (uint32_t)(2 * S_COLS + i * O_COLS);
        // blk64: each V token-plane is its own ring slot. blk128: one slot holds both token-atoms
        // (atom p at p*V_BLK_BYTES in-slot), waited once.
        int slot = 0;
        #pragma unroll
        for (int p = 0; p < V_SUBTILES; ++p) {
          if (!BLK128 || p == 0) {
            slot = kv_ph.get_stage();
            wp_begin(wpc, WP_MMA_WAIT_FULL_V);
            mbarrier_wait_parity(smem_ptr_u32(&full_bar[slot]), kv_ph.get_phase());
            wp_end(wpc, WP_MMA_WAIT_FULL_V);
            kv_ph.advance();
          }
          const uint64_t dbV = desc_kv0 + (uint64_t)slot * SLOT_DESC_DELTA
                             + (BLK128 ? (uint64_t)p * (V_BLK_BYTES >> 4) : 0u);
          #pragma unroll
          for (int ki = 0; ki < K_ATOMS_PER_TILE; ++ki) {
            const int a = p * K_ATOMS_PER_TILE + ki;              // flat P/O atom 0..7
            if constexpr (SPLIT_P) if (a == SPLIT_P_ATOM) {
              wp_begin(wpc, WP_MMA_WAIT_P);
              mbarrier_wait_parity(smem_ptr_u32(&full_bar_p_last[i]), spo_ph.get_phase());
              wp_end(wpc, WP_MMA_WAIT_P);
            }
            const bool accumulate = (!first_group) || (a != 0);
            wp_begin(wpc, WP_MMA_ISSUE);
            if constexpr (BLK128)
              tcgen05_mma_f16_ts_1sm_lead(lead, o_tmem_addr, s_tmem_addr + (uint32_t)(a * 8), dbV + 2 * ki, idesc_pv, accumulate);
            else
              tcgen05_mma_ws_f16_ts_1sm_lead(lead, o_tmem_addr, s_tmem_addr + (uint32_t)(a * 8), dbV + 2 * ki, idesc_pv, accumulate);
            wp_end(wpc, WP_MMA_ISSUE);
          }
          if (!BLK128 || p == V_SUBTILES - 1) {
            wp_begin(wpc, WP_MMA_COMMIT);
            tcgen05_commit1_lead(lead, smem_ptr_u32(&empty_bar[slot]));
            wp_end(wpc, WP_MMA_COMMIT);
          }
        }

        wp_begin(wpc, WP_MMA_COMMIT);
        tcgen05_commit1_lead(lead & (last_group ? 1u : 0u), smem_ptr_u32(&full_bar_o_acc[i]));
        wp_end(wpc, WP_MMA_COMMIT);
      };

      // prologue: BMM1 of group 0 for every M-tile.
      #pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i) {
        wp_begin(wpc, WP_MMA_WAIT_FULL_Q);
        mbarrier_wait_parity(smem_ptr_u32(&full_bar_q[i]), q_ph.get_phase());
        wp_end(wpc, WP_MMA_WAIT_FULL_Q);

        bmm1(i);
      }

      // main loop: BMM2(group) then BMM1(next group)
      for (int k = 0; k + 1 < ngroups; ++k) {
        #pragma unroll
        for (int i = 0; i < M_TILES_PER_CTA; ++i) {
          wp_begin(wpc, WP_MMA_WAIT_P);
          mbarrier_wait_parity(smem_ptr_u32(&empty_bar_spo[i]), spo_ph.get_phase());
          wp_end(wpc, WP_MMA_WAIT_P);

          bmm2(i, /*first_group=*/(k == 0), /*last_group=*/false);

          // BMM1(next): Q@K -> S
          bmm1(i);
        }

        spo_ph.advance();
      }

      // epilogue: BMM2 of the last group -> final O
      #pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i) {
        wp_begin(wpc, WP_MMA_WAIT_P);
        mbarrier_wait_parity(smem_ptr_u32(&empty_bar_spo[i]), spo_ph.get_phase());
        wp_end(wpc, WP_MMA_WAIT_P);

        bmm2(i, /*first_group=*/(ngroups == 1), /*last_group=*/true);
      }

      spo_ph.advance();

      wp_begin(wpc, WP_MMA_COMMIT);
      {
        #pragma unroll
        for (int i = 0; i < M_TILES_PER_CTA; ++i)
          tcgen05_commit1_lead(lead, smem_ptr_u32(&empty_bar_q[i]));
      }
      wp_end(wpc, WP_MMA_COMMIT);

      q_ph.advance();

      if constexpr (USE_CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, /*CTA_GROUP=*/1, /*SUSPEND=*/true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, /*do_release=*/elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        if (!next.valid) break;
        tile_id = next.n_tile;
      } else {
        tile_id += gridDim.x;
        if (tile_id >= total_tiles) break;
      }
    }
  }
  else if (warp_id == W_EPI) {
    setmaxnreg_dec<48>();

    PhaseTracker<1> full_o_ph;
    // Prime empty_bar_o_epi once: corr's first sO pack must not block (no prior store in flight).
    if (elect_one_sync()) {
      #pragma unroll
      for (int m = 0; m < M_TILES_PER_CTA; ++m)
        mbarrier_arrive(smem_ptr_u32(&empty_bar_o_epi[m]));
    }
    [[maybe_unused]] int clc_stage = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;
    int tile_id = (int)blockIdx.x;
    while (true) {
      const VsaItem it = vsa_decode<Q_RASTER>(tile_id, num_heads, nb, packed_mtiles_per_seq, q2k_num);
      const int q_start = it.sample * seqlen;
      const int qblk[2] = { it.qblk0, it.qblk1 };

      #pragma unroll
      for (int m = 0; m < M_TILES_PER_CTA; ++m) {
        wp_begin(wpc, WP_EPI_WAIT_TMEM);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_o_epi[m]), full_o_ph.get_phase());
        wp_end(wpc, WP_EPI_WAIT_TMEM);

        wp_begin(wpc, WP_EPI_STORE);
        if (elect_one_sync()) {
          const int tok0 = q_start + qblk[m] * BLOCK;
          #pragma unroll
          for (int s = 0; s < Q_SUBTILES; ++s) {
            tma_store_3d(&tmap_o, s * SUB_COLS_BF16, it.head, tok0,
                         smem_ptr_u32(reinterpret_cast<const uint8_t*>(sO_bufs[m]) + s * Q_SUB_COLS_BYTES));
          }
          cp_async_bulk_commit_group();
        }
        wp_end(wpc, WP_EPI_STORE);
      }

      wp_begin(wpc, WP_EPI_WAIT_STORE);
      // Drain per-m commit groups; release each sO slot to corr as ITS store completes
      // (without this, corr's next pack races the in-flight store at tiny K-loops).
      if (elect_one_sync()) {
        cp_async_bulk_wait_group_read<1>();
        mbarrier_arrive(smem_ptr_u32(&empty_bar_o_epi[0]));
        cp_async_bulk_wait_group_read<0>();
        mbarrier_arrive(smem_ptr_u32(&empty_bar_o_epi[1]));
      }
      wp_end(wpc, WP_EPI_WAIT_STORE);

      full_o_ph.advance();
      if constexpr (USE_CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, /*CTA_GROUP=*/1, /*SUSPEND=*/true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, /*do_release=*/elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        if (!next.valid) break;
        tile_id = next.n_tile;
      } else {
        tile_id += gridDim.x;
        if (tile_id >= total_tiles) break;
      }
    }
  }
  else if (warp_id == W_SCHED) {
    setmaxnreg_dec<48>();

    if constexpr (USE_CLC) {
      int prod_stage = 0; uint32_t prod_phase = 1;
      int cons_stage = 0; uint32_t cons_phase = 0;
      while (true) {
        if (lane == 0)
          mbarrier_wait_parity_suspend(smem_ptr_u32(&clc_empty[prod_stage]), prod_phase);
        __syncwarp();
        clc_arrive_expect_tx_cta(smem_ptr_u32(&clc_full[prod_stage]), /*tx_bytes=*/16);
        if (lane == 0)
          clc_try_cancel_async(smem_ptr_u32(&clc_response[prod_stage * 4]),
                               smem_ptr_u32(&clc_full[prod_stage]));
        advance_stage_phase<CLC_STAGES>(prod_stage, prod_phase);
        ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, /*CTA_GROUP=*/1, /*SUSPEND=*/true>(
            clc_full, clc_empty, clc_response, cons_stage, cons_phase, /*do_release=*/elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(cons_stage, cons_phase);
        if (!next.valid) break;
      }
      // Tail drain: absorb the in-flight consumer releases before kernel exit.
      for (int s = 0; s < CLC_STAGES; ++s) {
        if (lane == 0)
          mbarrier_wait_parity_suspend(smem_ptr_u32(&clc_empty[prod_stage]), prod_phase);
        __syncwarp();
        advance_stage_phase<CLC_STAGES>(prod_stage, prod_phase);
      }
    }
  }
  else if (warp_id >= W_CORR0 && warp_id < W_MMA) {
    setmaxnreg_dec<80>();

    // Layout E: corr thread t = corr_warp_id*32 + lane -> TMEM lane t -> O-partial row (t & 63)
    // of kv-half (t >> 6).
    const int corr_warp_id = warp_id - W_CORR0;
    const int corr_tid = corr_warp_id * 32 + lane;
    const int corr_row = BLK128 ? corr_tid : (corr_tid & 63);
    const bool kv_half0 = BLK128 || corr_tid < 64;   // blk128: every thread packs its own row
    [[maybe_unused]] PhaseTracker<1> alpha_ph;
    PhaseTracker<1> o_acc_ph;
    PhaseTracker<1> o_epi_empty_ph;

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
      const VsaItem it = vsa_decode<Q_RASTER>(tile_id, num_heads, nb, packed_mtiles_per_seq, q2k_num);
      const int ngroups = it.nkv / BLOCKS_PER_KTILE;

      // group 0: no rescale (no prior O); consume alpha + release the scale slot.
      #pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i) {
        wp_begin(wpc, WP_CORR_WAIT);
        if constexpr (FULL_NAMED_BAR) full_bar_wait(i, corr_warp_id);
        else mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_alpha[i]), alpha_ph.get_phase());
        mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
        wp_end(wpc, WP_CORR_WAIT);
      }
      if constexpr (!FULL_NAMED_BAR) alpha_ph.advance();

      for (int k = 1; k < ngroups; ++k) {
        #pragma unroll
        for (int i = 0; i < M_TILES_PER_CTA; ++i) {
          wp_begin(wpc, WP_CORR_WAIT);
          if constexpr (FULL_NAMED_BAR) full_bar_wait(i, corr_warp_id);
          else mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_alpha[i]), alpha_ph.get_phase());
          wp_end(wpc, WP_CORR_WAIT);

          wp_begin(wpc, WP_CORR_READ_ALPHA);
          const float alpha = alpha_and_l_smem[(i * STAT_REGIONS + 0) * STATS + corr_tid];
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
          if constexpr (SOFTMAX_THROTTLE) mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
          mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[i]));
          wp_end(wpc, WP_CORR_O_SCALE);
        }
        if constexpr (!FULL_NAMED_BAR) alpha_ph.advance();
      }

      // epilogue: half-merge + O *= 1/l -> bf16 -> sO[i] -> signal W_EPI (full_bar_o_epi).
      #pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i) {
        wp_begin(wpc, WP_CORR_WAIT);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_o_acc[i]), o_acc_ph.get_phase());
        if constexpr (FULL_NAMED_BAR) full_bar_wait(i, corr_warp_id);
        else mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_l[i]), o_acc_ph.get_phase());
        wp_end(wpc, WP_CORR_WAIT);

        wp_begin(wpc, WP_CORR_EPI);
        // blk64 HALF-MERGE: partner thread t^64 holds the other half's O partial. 2-way split-KV
        // combine: m_tot = max(m0, m1); beta_h = exp2((m_h - m_tot)*scale_log2);
        // out = (beta_0*O_0 + beta_1*O_1) / (beta_0*l_0 + beta_1*l_1). Only O is exchanged
        // (16-col f32 chunks via o_xchg); both partners compute l_tot from the SMEM stats.
        float scale_own;
        if constexpr (BLK128) {
          // plain M=128 accumulator: no halves, no merge -- scale = 1/l (classic dense epilogue).
          const float l = alpha_and_l_smem[(i * STAT_REGIONS + 1) * STATS + corr_tid];
          if constexpr (!SOFTMAX_THROTTLE) mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
          scale_own = (l > 0.f) ? rcp_approx_ftz_f32(l) : 0.f;
        } else {
          const float l_own = alpha_and_l_smem[(i * STAT_REGIONS + 1) * STATS + corr_tid];
          const float m_own = alpha_and_l_smem[(i * STAT_REGIONS + 2) * STATS + corr_tid];
          const float l_par = alpha_and_l_smem[(i * STAT_REGIONS + 1) * STATS + (corr_tid ^ 64)];
          const float m_par = alpha_and_l_smem[(i * STAT_REGIONS + 2) * STATS + (corr_tid ^ 64)];
          if constexpr (!SOFTMAX_THROTTLE) mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
          const float m_tot = fmaxf(m_own, m_par);
          const float beta_own = ex2_approx_f32((m_own - m_tot) * scale_log2);
          const float beta_par = ex2_approx_f32((m_par - m_tot) * scale_log2);
          const float l_tot = beta_own * l_own + beta_par * l_par;
          scale_own = (l_tot > 0.f) ? beta_own * rcp_approx_ftz_f32(l_tot) : 0.f;
        }
        const float2 scale2 = f32x2_splat(scale_own);
        const uint32_t o_tmem_addr = tmem_base + (uint32_t)(2 * S_COLS + i * O_COLS) + ((uint32_t)(corr_warp_id * 32) << 16);
        #pragma unroll
        for (int c0 = 0; c0 < HEAD_DIM; c0 += 16) {
          uint32_t o_regs[16];
          tcgen05_ld_32x32b_x16(o_tmem_addr + (uint32_t)c0, o_regs);
          if (c0 == 0) mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_o_epi[i]), o_epi_empty_ph.get_phase());
          float2* o2 = reinterpret_cast<float2*>(o_regs);
          #pragma unroll
          for (int e = 0; e < 8; ++e) o2[e] = fmul2(o2[e], scale2);
          if constexpr (!BLK128) {
            if (!kv_half0) {
              #pragma unroll
              for (int e = 0; e < 16; ++e) o_xchg[corr_row * 16 + e] = reinterpret_cast<float*>(o_regs)[e];
            }
            bar_sync<9>(128);   // corr warps 8-11: half-1 chunk visible
          }
          if (kv_half0) {
            // half 0: add the partner's chunk, then pack bf16 -> sO[i].
            if constexpr (!BLK128) {
              #pragma unroll
              for (int e = 0; e < 8; ++e) {
                const float2 x = *reinterpret_cast<const float2*>(&o_xchg[corr_row * 16 + 2 * e]);
                o2[e] = fadd2(o2[e], x);
              }
            }
            const int s = c0 / SUB_COLS_BF16;
            const int v_base = (c0 % SUB_COLS_BF16) / 8;
            __nv_bfloat16* so_sub = sO_bufs[i] + s * (M_TILE * SUB_COLS_BF16);
            #pragma unroll
            for (int vv = 0; vv < 2; ++vv) {
              const int v = v_base + vv;
              uint4 packed;
              packed.x = cvt_f32x2_to_bf16x2(o2[vv * 4 + 0].x, o2[vv * 4 + 0].y);
              packed.y = cvt_f32x2_to_bf16x2(o2[vv * 4 + 1].x, o2[vv * 4 + 1].y);
              packed.z = cvt_f32x2_to_bf16x2(o2[vv * 4 + 2].x, o2[vv * 4 + 2].y);
              packed.w = cvt_f32x2_to_bf16x2(o2[vv * 4 + 3].x, o2[vv * 4 + 3].y);
              *reinterpret_cast<uint4*>(&so_sub[corr_row * SUB_COLS_BF16 + (v ^ (corr_row & 7)) * 8]) = packed;
            }
          }
          if constexpr (!BLK128)
            bar_sync<9>(128);   // half-0 consumed the chunk -> half-1 may overwrite o_xchg
        }
        // O ld done -> MMA may reuse the O slot
        tcgen05_fence_before_thread_sync();

        if constexpr (SOFTMAX_THROTTLE) mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
        mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[i]));

        // order this thread's st.shared writes (generic proxy) before TMA store (async proxy)
        fence_proxy_async_shared();

        mbarrier_arrive(smem_ptr_u32(&full_bar_o_epi[i]));
        wp_end(wpc, WP_CORR_EPI);
      }
      o_acc_ph.advance();
      o_epi_empty_ph.advance();

      if constexpr (USE_CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, /*CTA_GROUP=*/1, /*SUSPEND=*/true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, /*do_release=*/elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        if (!next.valid) break;
        tile_id = next.n_tile;
      } else {
        tile_id += gridDim.x;
        if (tile_id >= total_tiles) break;
      }
    }
  }
  else {
    setmaxnreg_inc<192>();

    const int m_tile = warp_id < 4 ? 0 : 1;
    const int warp_in_group = warp_id & 3;
    // Layout E: thread t = warp_in_group*32 + lane -> TMEM lane t -> S row (t & 63) of kv-half
    // (t >> 6); each (row, half) runs an INDEPENDENT online softmax (halves merge in corr).
    const int sm_tid = warp_in_group * 32 + lane;
    const uint32_t s_tmem_addr = tmem_base + (uint32_t)(m_tile * S_COLS) + ((uint32_t)(warp_in_group * 32) << 16);
    PhaseTracker<1> spo_ph;
    PhaseTracker<1> scale_empty_ph;
    [[maybe_unused]] int clc_stage = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;

    int tile_id = (int)blockIdx.x;
    while (true) {
      const VsaItem it = vsa_decode<Q_RASTER>(tile_id, num_heads, nb, packed_mtiles_per_seq, q2k_num);
      const int ngroups = it.nkv / BLOCKS_PER_KTILE;

      float m_run = -INFINITY, l_run = 0.f;
      wp_begin(wpc, WP_SM_WAIT_SCALE);
      mbarrier_wait_parity_sel<VSA_SM_HOT>(smem_ptr_u32(&empty_bar_alpha_and_l[m_tile]), scale_empty_ph.get_phase());
      scale_empty_ph.advance();
      wp_end(wpc, WP_SM_WAIT_SCALE);
      for (int k = 0; k < ngroups; ++k) {
        wp_begin(wpc, WP_SM_WAIT_S);
        mbarrier_wait_parity_sel<VSA_SM_HOT>(smem_ptr_u32(&full_bar_spo[m_tile]), spo_ph.get_phase());
        wp_end(wpc, WP_SM_WAIT_S);

        wp_begin(wpc, WP_SM_SOFTMAX);
        uint32_t s_regs[S_COLS];
        #pragma unroll
        for (int c0 = 0; c0 < S_COLS; c0 += S_LD_COLS) {
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

        // vbs KEY MASK (the one ragged mechanism): within each 64-token segment, keys >=
        // vbs[block_id] -> -inf. blk64: cols 0-63 = block g*4 + 2*half, 64-127 = its neighbor;
        // blk128: all 128 cols = the group's 1 block. vbs/idx reads are warp-converged.
        {
          const int gqb_mt = (m_tile == 0) ? it.gqb0 : it.gqb1;
          if constexpr (BLK128) {
            const int bid = q2k_idx[gqb_mt * max_kv + k];
            mask_s_row_r2p<false, S_COLS>(scores, 0, 0, variable_block_sizes[bid]);
          } else {
            const int half = warp_in_group >> 1;
            const int jbase = gqb_mt * max_kv + k * BLOCKS_PER_KTILE + 2 * half;
            const int bid0 = q2k_idx[jbase], bid1 = q2k_idx[jbase + 1];
            mask_s_row_r2p<false, 64>(scores,      0, 0, variable_block_sizes[bid0]);
            mask_s_row_r2p<false, 64>(scores + 64, 0, 0, variable_block_sizes[bid1]);
          }
        }

        // rmax via 4 independent FMNMX3 accumulators (4-way ILP). S_COLS % 8 == 0.
        float rmax0 = -INFINITY, rmax1 = -INFINITY, rmax2 = -INFINITY, rmax3 = -INFINITY;
        #pragma unroll
        for (int j = 0; j < S_COLS; j += 8) {
          rmax0 = fmaxf(fmaxf(rmax0, scores[j + 0]), scores[j + 1]);
          rmax1 = fmaxf(fmaxf(rmax1, scores[j + 2]), scores[j + 3]);
          rmax2 = fmaxf(fmaxf(rmax2, scores[j + 4]), scores[j + 5]);
          rmax3 = fmaxf(fmaxf(rmax3, scores[j + 6]), scores[j + 7]);
        }
        float rmax = fmaxf(fmaxf(rmax0, rmax1), fmaxf(rmax2, rmax3));
        float new_m = fmaxf(m_run, rmax);
        float alpha = 0.0f;
        if (k != 0) {
          // FA4 sticky max (uplift #7) + volatile STS publish (#8); pairs corr's
          // __all_sync(alpha == 1.0f) skip vote. No first-step slot write (#11).
          const float acc_scale_ = (m_run - new_m) * scale_log2;
          if (acc_scale_ >= -8.0f) { new_m = m_run; alpha = 1.0f; }
          else                     { alpha = ex2_approx_f32(acc_scale_); }
          sts_f32(smem_ptr_u32(&alpha_and_l_smem[(m_tile * STAT_REGIONS + 0) * STATS + sm_tid]), alpha);
        }
        if constexpr (FULL_NAMED_BAR) full_bar_arrive(m_tile, warp_in_group);
        else mbarrier_arrive(smem_ptr_u32(&full_bar_alpha[m_tile]));

        // fused ffma2(scale) + exp2 + row-sum + bf16 pack: 128 per-half keys -> 64 u32 P cols.
        const float2 scale2 = f32x2_splat(scale_log2);
        const float2 neg_m_scaled2 = f32x2_splat(-new_m * scale_log2);
        uint32_t p_regs[S_COLS / 2];
        [[maybe_unused]] float2 lt2_live = make_float2(0.f, 0.f);
        #pragma unroll
        for (int c = 0; c < S_COLS / 2; ++c) {
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
          if constexpr (DEFER_ROWSUM) scores2[c] = e2;  // rowsum after publish
          else                        lt2_live = fadd2(lt2_live, e2);
          p_regs[c] = cvt_f32x2_to_bf16x2(e2.x, e2.y);
        }
        const uint32_t p_tmem_addr = s_tmem_addr;
        wp_end(wpc, WP_SM_SOFTMAX);

        wp_begin(wpc, WP_SM_STORE_P);
        if constexpr (SPLIT_P) {
          tcgen05_st_32x32b_x32(p_tmem_addr,      *reinterpret_cast<uint32_t(*)[32]>(&p_regs[0]));
          tcgen05_st_32x32b_x16(p_tmem_addr + 32, *reinterpret_cast<uint32_t(*)[16]>(&p_regs[32]));
          tcgen05_wait_st();   // RACE FIX f22fb2e: fence orders but does NOT complete the async STTM
          tcgen05_fence_before_thread_sync();
          mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[m_tile]));
          tcgen05_st_32x32b_x16(p_tmem_addr + SPLIT_P_COL, *reinterpret_cast<uint32_t(*)[16]>(&p_regs[SPLIT_P_COL]));
          tcgen05_wait_st();   // RACE FIX f22fb2e
          tcgen05_fence_before_thread_sync();
          mbarrier_arrive(smem_ptr_u32(&full_bar_p_last[m_tile]));
        } else {
          tcgen05_st_32x32b_x32(p_tmem_addr,      *reinterpret_cast<uint32_t(*)[32]>(&p_regs[0]));
          tcgen05_st_32x32b_x32(p_tmem_addr + 32, *reinterpret_cast<uint32_t(*)[32]>(&p_regs[32]));
          tcgen05_wait_st();   // RACE FIX f22fb2e
          tcgen05_fence_before_thread_sync();
          mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[m_tile]));
        }
        wp_end(wpc, WP_SM_STORE_P);

        spo_ph.advance();
        float2 lt2 = lt2_live;
        if constexpr (DEFER_ROWSUM) {
          // P is already published; this FADD2 chain runs in the shadow of the
          // next wait_scale instead of gating BMM2.
          float2 lt2a = make_float2(0.f, 0.f), lt2b = make_float2(0.f, 0.f);
          float2 lt2c = make_float2(0.f, 0.f), lt2d = make_float2(0.f, 0.f);
          #pragma unroll
          for (int c = 0; c < S_COLS / 2; c += 4) {
            lt2a = fadd2(lt2a, scores2[c + 0]);
            lt2b = fadd2(lt2b, scores2[c + 1]);
            lt2c = fadd2(lt2c, scores2[c + 2]);
            lt2d = fadd2(lt2d, scores2[c + 3]);
          }
          lt2 = fadd2(fadd2(lt2a, lt2b), fadd2(lt2c, lt2d));
        }
        float lt = lt2.x + lt2.y;
        l_run = alpha * l_run + lt; m_run = new_m;
        wp_begin(wpc, WP_SM_WAIT_SCALE);
        mbarrier_wait_parity_sel<VSA_SM_HOT>(smem_ptr_u32(&empty_bar_alpha_and_l[m_tile]), scale_empty_ph.get_phase());
        scale_empty_ph.advance();
        wp_end(wpc, WP_SM_WAIT_SCALE);
      }

      wp_begin(wpc, WP_SM_READ_L);
      alpha_and_l_smem[(m_tile * STAT_REGIONS + 1) * STATS + sm_tid] = l_run;
      if constexpr (!BLK128)
        alpha_and_l_smem[(m_tile * STAT_REGIONS + 2) * STATS + sm_tid] = m_run;
      if constexpr (FULL_NAMED_BAR) full_bar_arrive(m_tile, warp_in_group);
      else mbarrier_arrive(smem_ptr_u32(&full_bar_l[m_tile]));
      wp_end(wpc, WP_SM_READ_L);

      if constexpr (USE_CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, /*CTA_GROUP=*/1, /*SUSPEND=*/true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, /*do_release=*/elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        if (!next.valid) break;
        tile_id = next.n_tile;
      } else {
        tile_id += gridDim.x;
        if (tile_id >= total_tiles) break;
      }
    }
  }
  wp_flush(wpc);
  __syncthreads();
  if (warp_id == 0) tcgen05_dealloc<1>(tmem_base, TMEM_TOTAL);
}

// ============================== driver ====================================

// CPU reference: VSA fine block-sparse attention with variable_block_sizes, fp32. For EVERY
// query row of a q-block (padding rows too -- the triton kernel does not mask q rows), attend
// to the VALID tokens (kj < vbs[blk]) of the SELECTED KV blocks only, softmax, @V.
// Layout: padded-strided [token, head, hd]; block i at tokens [i*BLOCK, (i+1)*BLOCK).
static void cpu_vsa_ref(const __nv_bfloat16* hQ, const __nv_bfloat16* hK,
                        const __nv_bfloat16* hV, float* hO,
                        int B, int H, int S, int hd, int nb, int max_kv,
                        const int* q2k_idx, const int* q2k_num, const int* vbs) {
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
        const int nvalid = vbs[blk];
        for (int kj = 0; kj < BLOCK; ++kj) {
          if (kj >= nvalid) { z[idx++] = -INFINITY; continue; }   // vbs key mask
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
      for (int j = 0; j < nkv * BLOCK; ++j) {
        if (z[j] == -INFINITY) { z[j] = 0.f; continue; }
        z[j] = expf(z[j] - row_max); sum += z[j];
      }
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
// could alias with H*hd and make every token identical.
static void fillr(__nv_bfloat16* h, long n, unsigned seed) {
  block_sparse_bf16_benchmark::fill(h, n, seed);
}

struct Sh { int B, H, nb, topk, hd; const char* lab; };

static double run(const Sh& sh, bool verify) {
  const int  B = sh.B, H = sh.H, nb = sh.nb, topk = sh.topk, hd = sh.hd;
  const int  S = nb * BLOCK;                       // seqlen (multiple of 64)
  const int  max_kv = topk;                        // tight: exactly topk selected blocks
  const long tq = (long)B * S;                     // total tokens
  const int  packed_mtiles_per_seq = nb / 2;   // a CTA does a PAIR of adjacent q-blocks
  const int  total_qblk = B * H * nb;
  const int  total_work = B * H * packed_mtiles_per_seq;
  if (nb % 2 != 0) { printf("  [%s] SKIP: nb must be even (adjacent-pair tiling)\n", sh.lab); return 0.0; }
  if (topk % BLOCKS_PER_KTILE != 0) { printf("  [%s] SKIP: topk must be a multiple of %d\n", sh.lab, BLOCKS_PER_KTILE); return 0.0; }

  // ---- device buffers (bf16; V stored transposed as V_T for the BMM2 TMA) ----
  __nv_bfloat16 *dQ, *dK, *dVT, *dO;
  CUDA_CHECK(cudaMalloc(&dQ,  tq * H * hd * 2));
  CUDA_CHECK(cudaMalloc(&dK,  tq * H * hd * 2));
  CUDA_CHECK(cudaMalloc(&dVT, (long)H * hd * tq * 2));
  CUDA_CHECK(cudaMalloc(&dO,  tq * H * hd * 2));

  std::vector<__nv_bfloat16> hQ(tq * H * hd), hK(tq * H * hd), hV(tq * H * hd);
  const char* load_npy = getenv("LOAD_NPY");
  if (load_npy) {
    auto load = [&](const char* name, std::vector<__nv_bfloat16>& values) {
      const std::string path = std::string(load_npy) + "/" + name + "_S" + std::to_string(S) + ".npy";
      auto bits = npy_load_vec<uint16_t>(path);
      if (bits.size() != values.size()) {
        fprintf(stderr, "LOAD_NPY: wrong tensor size in %s\n", path.c_str());
        std::exit(EXIT_FAILURE);
      }
      memcpy(values.data(), bits.data(), values.size() * 2);
    };
    load("q", hQ); load("k", hK); load("v", hV);
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

  // ---- q2k index: topk DISTINCT block ids per (b,h,qblk), fixed density (num == topk) ----
  std::vector<int> hq2k_idx((size_t)total_qblk * max_kv, 0);
  std::vector<int> hq2k_num(total_qblk, topk);
  if (load_npy) {
    const std::string path = std::string(load_npy) + "/idx_S" + std::to_string(S)
                           + "_blk" + std::to_string(BLOCK) + ".npy";
    auto idx = npy_load_vec<int32_t>(path);
    if (idx.size() != (size_t)nb * topk) {
      fprintf(stderr, "LOAD_NPY: wrong index count in %s\n", path.c_str());
      std::exit(EXIT_FAILURE);
    }
    for (int gqb = 0; gqb < total_qblk; ++gqb)
      for (int i = 0; i < topk; ++i)
        hq2k_idx[(size_t)gqb * max_kv + i] = idx[(size_t)(gqb % nb) * topk + i];
  } else {
    block_sparse_bf16_benchmark::select_blocks(hq2k_idx, nb, topk);
  }
  int *dq2k_idx, *dq2k_num;
  CUDA_CHECK(cudaMalloc(&dq2k_idx, hq2k_idx.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dq2k_num, hq2k_num.size() * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(dq2k_idx, hq2k_idx.data(), hq2k_idx.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dq2k_num, hq2k_num.data(), hq2k_num.size() * sizeof(int), cudaMemcpyHostToDevice));

  // ---- variable_block_sizes: deterministic random in [VBS_MIN, BLOCK] (the FastVideo test uses
  // 16..64), shared across batch and heads (indexed by block id only, like the triton kernel).
  // VBS_MIN=BLOCK (env VBS_MIN=64/128) -> all blocks full = the uniform case (regression). ----
  // Shared-input grid is full-block; retain the historical ragged default otherwise.
  const int vbs_min = block_sparse_bf16_benchmark::env_count("VBS_MIN", load_npy ? BLOCK : 16, 1, BLOCK);
  std::vector<int> hvbs(nb);
  for (int i = 0; i < nb; ++i) {
    uint32_t x = (uint32_t)i * 2654435761u + 777u;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    hvbs[i] = vbs_min >= BLOCK ? BLOCK : vbs_min + (int)(x % (uint32_t)(BLOCK - vbs_min + 1));
  }
  int* dvbs;
  CUDA_CHECK(cudaMalloc(&dvbs, hvbs.size() * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(dvbs, hvbs.data(), hvbs.size() * sizeof(int), cudaMemcpyHostToDevice));

  // ---- TMA tensor maps ----
  // Q/O: 3D over [hd, H, tq]; box [hd-subtile x 1 (MHA) x M_TILE]. Matches the load's 3D box.
  CUtensorMap tq_, tk_, tvt_, to_;
  CUDA_CHECK(make_tma_3d_tiled(&tq_, dQ, hd, H, (int)tq, SUB_COLS_BF16, 1, M_TILE, 2,
                               CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B));
  CUDA_CHECK(make_tma_3d_tiled(&to_, dO, hd, H, (int)tq, SUB_COLS_BF16, 1, M_TILE, 2,
                               CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B));
  // K: dims [atom-col SUB_COLS_BF16, token tq, atom (H*hd)/SUB_COLS_BF16]. Box = ONE (block x hd-atom) =
  // [SUB_COLS_BF16, BLOCK, 1]; the load issues BLOCKS_PER_KTILE x K_SUBTILES of these per KV tile, placing
  // each gathered block into its token-half of the 128-token 2-hd-subtile K tile.
  {
    uint64_t gd[3] = { (uint64_t)SUB_COLS_BF16, (uint64_t)tq, (uint64_t)((long)H * hd / SUB_COLS_BF16) };
    uint64_t gs[2] = { (uint64_t)((long)H * hd) * 2u, (uint64_t)SUB_COLS_BF16 * 2u };
    // blk64 box = one (block x hd-atom) [SUB_COLS_BF16, BLOCK, 1]; blk128 box folds both hd-atoms of
    // the single block per K tile [SUB_COLS_BF16, BLOCK, K_SUBTILES] (one TMA per tile).
    uint32_t bd[3] = { (uint32_t)SUB_COLS_BF16, (uint32_t)BLOCK, BLK128 ? (uint32_t)K_SUBTILES : 1u };
    uint32_t es[3] = { 1u, 1u, 1u };
    CUresult r = cuTensorMapEncodeTiled(&tk_, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, dK, gd, gs, bd, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUDA_CHECK(r == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue);
  }
  // V_T: 2D [H*hd rows, tq cols]; box [hd rows, SUB_COLS_BF16 cols] = one 64-token block (V subtile);
  // the load issues BLOCKS_PER_KTILE of these per KV tile (each fills one V subtile).
  CUDA_CHECK(make_tma_2d_tiled(&tvt_, dVT, (long)H * hd, (int)tq, hd, SUB_COLS_BF16, 2,
                               CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B));

  // ---- shared memory budget ----
  const size_t smem =
        (size_t)2 * Q_TILE_BYTES + NUM_KV_STAGES * SLOT_BYTES     // Q (x2) + K/V plane-slot ring
      + (size_t)2 * M_TILE * HEAD_DIM * sizeof(__nv_bfloat16)     // 2 sO bufs for TMA-O
      + (size_t)XCHG_FLOATS * sizeof(float)                      // o_xchg (blk64 half-merge chunk)
      + (2 * NUM_KV_STAGES + 22) * 8                              // mbarriers
      + (size_t)CLC_STAGES * (2 * 8 + 16) + 16                    // CLC: clc_full+clc_empty + response
      + 8                                                         // tmem_slot
      + (size_t)2 * STAT_REGIONS * STATS * sizeof(float)         // alpha/l(/m) stats [2][REGIONS][STATS]
      + 256;                                                      // slack / alignment

  // Compile-time kernel config (see the knob docs near the top of this file).
#ifndef VSA_NAMED_BAR
#define VSA_NAMED_BAR false
#endif
#ifndef VSA_THROTTLE
#define VSA_THROTTLE false
#endif
#ifndef VSA_USE_CLC
#define VSA_USE_CLC true
#endif
  // NAMED_BAR and THROTTLE are an interlocked pair in the inline dense kernel
  // (throttle-off + named-bar desyncs the alpha/l pairing) -- flip them together.
  constexpr bool FULL_NAMED_BAR = VSA_NAMED_BAR, EX2_EMU = true, SPLIT_P = true,
                 SOFTMAX_THROTTLE = VSA_THROTTLE, USE_CLC = VSA_USE_CLC,
                 Q_RASTER = true, MHA = true;
  auto kfn = &fmha_context_bf16_gen_kernel<32, FULL_NAMED_BAR, EX2_EMU, SPLIT_P, SOFTMAX_THROTTLE, USE_CLC, Q_RASTER, MHA>;
  CUDA_CHECK(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));

  const float scale_log2 = (1.0f / sqrtf((float)hd)) * (float)M_LOG2E;
  int numSM = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, 0));
  const int nblk = USE_CLC ? total_work : std::min(total_work, numSM);
  dim3 grid(nblk, 1, 1), block(N_WARPS * 32, 1, 1);

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = smem; cfg.stream = 0;
  cudaLaunchAttribute cfgAttr[1];
  cfgAttr[0].id = cudaLaunchAttributeClusterDimension;
  cfgAttr[0].val.clusterDim.x = 1; cfgAttr[0].val.clusterDim.y = 1; cfgAttr[0].val.clusterDim.z = 1;
  cfg.attrs = cfgAttr; cfg.numAttrs = 1;
  auto launch = [&](cudaStream_t stream = nullptr) {
    cfg.stream = stream;
    if (USE_CLC)
      return cudaLaunchKernelEx(&cfg, kfn, tq_, tk_, tvt_, to_, S, H, scale_log2,
                                B, nb, packed_mtiles_per_seq, max_kv, (const int*)dq2k_idx, (const int*)dq2k_num,
                                (const int*)dvbs);
    kfn<<<grid, block, smem, stream>>>(tq_, tk_, tvt_, to_, S, H, scale_log2,
                               B, nb, packed_mtiles_per_seq, max_kv, (const int*)dq2k_idx, (const int*)dq2k_num,
                               (const int*)dvbs);
    return cudaGetLastError();
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
    wp_dump_raw(wp, "warp_raw_vsa.bin.gz", pblk, NUM_KV_STAGES);
    wp_free(wp);
  }
#endif

  // FLOPs over SELECTED blocks with FULL-block counting (the kernel computes full BLOCKxBLOCK
  // tiles and masks; same convention as the triton bench -> runtimes are directly comparable).
  const double pairs = (double)B * H * nb * BLOCK * (double)topk * BLOCK;
  const double tflops = block_sparse_bf16_benchmark::tflops(pairs, hd, ms);
  if (block_sparse_bf16_benchmark::benchmark_enabled()) printf("  [%-9s H%-2d nb%-3d k%-3d S%d vbs>=%d] N_q=%ld topk=%d  %.4f ms  %.1f TFLOPS (selected, full-block)\n",
         sh.lab, H, nb, topk, S, vbs_min, tq, topk, ms, tflops);

  block_sparse_bf16_benchmark::dump_output(dO, hQ.size());

  if (verify) {
    std::vector<__nv_bfloat16> ho(hQ.size());
    CUDA_CHECK(cudaMemcpy(ho.data(), dO, ho.size() * 2, cudaMemcpyDeviceToHost));
    std::vector<float> ref(hQ.size(), 0.f), out(hQ.size());
    cpu_vsa_ref(hQ.data(), hK.data(), hV.data(), ref.data(), B, H, S, hd, nb, max_kv,
                hq2k_idx.data(), hq2k_num.data(), hvbs.data());
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
  cudaFree(dq2k_idx); cudaFree(dq2k_num); cudaFree(dvbs);
  return tflops;
}

int main() {
  CUDA_CHECK(cudaFree(0));
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 64 * 1024 * 1024));
  printf("VSA fine block-sparse FMHA bf16 VARLEN (variable_block_sizes; %s, block=%d) sm_100a\n"
         "=====================================\n", BLK128 ? "blk128" : "blk64", BLOCK);

  // shapes: {B, H, nb, topk, hd, label}. nb even.
  Sh shapes[] = {
    {1,  4,  8,  4, 128, "small"},
    {1, 16, 32,  8, 128, "fastvideo"},
    {1,  8, 64, 16, 128, "25pct"},
  };
  const bool verify = getenv("NOVERIFY") ? false : true;

  // single-shape override: SHAPE=0..2 plus BATCH/HEADS/NB/TOPK env knobs.
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
