// block_sparse_bf16_uniform_blk256.cu -- VSA "fine" block-sparse FMHA, bf16, 1CTA,
// 256-token sparse block, sm_100a.
//
// Forked from block_sparse_bf16_uniform.cu (which selects a 64- or 128-token sparse block via
// the VSA_BLK128 toggle). Here the sparse block is FIXED at 256 tokens, the like-for-like match to
// FA4's vsa256 fast path (its minimum sparse block is 256). The 16-warp warp-specialized body, the
// BMM1-ahead software pipeline, the softmax<->correction<->MMA barrier contract, persistent
// scheduling (USE_CLC / grid-stride), split-P and the TMA-store epilogue are inherited unchanged
// from the dense fmha_context_bf16_uniform_inline.cu.
//
// DESIGN (what makes this file different from its parent):
//   - SPARSE_BLOCK = 256 is the top-k selection granularity AND the query-block granularity
//     (square: R = C = 256; one top-k list per 256-token query block, shared by all 256 of its rows).
//     The MMA datapath stays at 128 -- BLOCK = M_TILE = K_TILE = 128, plain full-datapath
//     tcgen05.mma (the parent's blk64 mma.ws dual-pack path is NOT used; BLK128 is forced true).
//   - A selected 256-block is gathered as KTILES_PER_BLOCK = 2 consecutive 128-token K-tiles:
//       K-tile kt covers tokens block_id*256 + (kt % 2)*128, and num_k_tiles = num_kv_blocks * 2.
//     K_TILE cannot be 256: TMEM holds S[0],S[1],O[0],O[1] = 4 x 128 = the full 512 columns, so a
//     256-wide S would need 768.
//   - ONE SHARED K/V STREAM, exactly the dense fmha_context_bf16_uniform_inline.cu discipline. A CTA's
//     2 M-tiles are the two 128-query HALVES of ONE 256-block and SHARE its top-k list, so they consume
//     IDENTICAL K/V bytes: every tile is gathered ONCE into a single ring slot that the caller waits,
//     both M-tiles' MMAs issue against, and the caller frees (bmm1/bmm2 do no ring wait/commit):
//       K(0) | { V(k), K(k+1) } per step | V(last)
//     i.e. the load warp issues 2*num_k_tiles tiles, half the per-M-tile version's 4*num_k_tiles.
//   - GOTCHA THAT MADE THIS LOOK IMPOSSIBLE (cost a full debug cycle -- do not undo it): the shared
//     stream is only correct if the MMA warp WALKS its SMEM descriptors with desc_add_lo (asm volatile,
//     32-bit add on the low word). With plain uint64 descriptor arithmetic the compiler CSEs the
//     identical descriptor across the two M-tiles' MMA groups into base+imm forms that rematerialize
//     the hi word, and the second group's B operand comes out wrong -- every output column then reads
//     row 0 of the tile, so that tile contributes nothing to a data-varying V. It only bit the tiles
//     whose two groups are issued BACK TO BACK (K-tile 0 in the prologue, V of the last K-tile in the
//     epilogue); interior tiles have a bmm1/bmm2 against a different slot in between, which breaks the
//     CSE. Proof: with both M-tiles waiting their own slot but pointed at the SAME descriptor value it
//     is wrong (max|err| 0.77), and with the two slots SWAPPED -- identical bytes, different descriptor
//     values -- it is correct (0.0017). See SmemDescPair below.
//     Gate any change here with an oracle diff (DUMP_O + external reference): the built-in
//     check_close_f32 (atol 0.05) is loose enough to hide this for topk >= 4.
//   - No causal mask and no padding mask: every selected block is FULL (all 256 tokens valid) and
//     the attention is non-causal (Sq == Skv, MHA, D = 128, bf16).
//
// Math / assumptions:
//   Shapes:   B samples, H heads (MHA: HQ == HK), S tokens/sample, D = 128.
//   Blocks:   num_blocks = S / SPARSE_BLOCK (S % SPARSE_BLOCK == 0, no token padding).
//   Density:  uniform -- num_kv_blocks = q2k_num[global_mtile] = round(d * num_blocks) (bench uses
//             d = 25%), the SAME for every query block, which keeps the 2 M-tiles in lockstep.
//   Work item = one CTA = ONE 256-token query block (its 2 M-tiles are that block's two halves):
//             num_work_items = B*H*num_blocks, packed_mtiles_per_seq = num_blocks.
//   Coords:   q_tok  = sample*S + mtile*M_TILE + row;
//             kv_tok = sample*S + q2k_idx[global_mtile*max_kv + j]*SPARSE_BLOCK + (kt % 2)*K_TILE.
//   FLOPs:    2*(QK + PV) = B*H*num_blocks*num_kv_blocks * SPARSE_BLOCK^2 * D * 2*2 (= "selected",
//             the same visited-pair convention the blk64/blk128 kernels report, so columns compare).
//   TMEM:     per M-tile S_COLS + O_COLS = 128 + 128; 2 M-tiles => TMEM_TOTAL = 512 cols.
//
// Data layout (D = 128; one K-tile = 128 gathered keys = half a selected 256-block):
//   name     where  dtype  shape                          written by -> read by
//   Q        SMEM   bf16   128 tok x 128 D (per M-tile)    TMA          -> BMM1 A
//   K        SMEM   bf16   128 keys x 128 D (gathered)     TMA gather   -> BMM1 B
//   V^T      SMEM   bf16   128 D x 128 keys (gathered)     TMA gather   -> BMM2 B
//   S=Q@K^T  TMEM   fp32   128 lanes x 128 cols            BMM1         -> softmax
//   P=exp(S) TMEM   bf16   128 lanes x  64 cols (in S)     softmax      -> BMM2 A
//   O=P@V    TMEM   fp32   128 lanes x 128 cols            BMM2         -> corr -> sO -> TMA
//   BMM1 = tcgen05.mma SS (Q,K in SMEM -> S in TMEM); BMM2 = tcgen05.mma TS (P in TMEM, V in SMEM).
//
// Barrier contract: identical to fmha_context_bf16_uniform_inline.cu (including the TMA-sO
// epilogue's empty_bar_o_epi) -- see that file's header.
//
// Budgets:
//   - Registers: softmax inc<192> x8 + correction dec<80> x4 + the four single warps dec<48> x4.
//   - SMEM: Q 2x32 KB + sO 2x32 KB + a NUM_KV_STAGES(3) x 32 KB K/V ring = 224 KB of the 227 KB cap.
//   - TMEM: the 512-col layout above; here the 2 M-tiles are the two halves of the SAME 256-block.
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
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include "npy_io.cuh"
#include "block_sparse_bf16_benchmark.cuh"
#include "../../../../tests/test_utils.cuh"
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

// blk256 variant: a 256-token SPARSE block (top-k selection granularity) is gathered as 2 x 128-token
// K-tiles. The MMA/TMEM/SMEM datapath is the blk128 plain-m128 path VERBATIM (BLK128 forced true, so every
// `BLK128 ? a : b` below picks the blk128 branch); M_TILE and K_TILE stay 128. The only structural change
// vs blk128: a CTA's 2 M-tiles are the two 128-query halves of ONE 256 sparse block, SHARING one top-k
// list (blk128's 2 M-tiles are 2 different blocks with 2 lists). See gather/decode below.
#define VSA_BLK128 true
constexpr bool BLK128 = VSA_BLK128;
constexpr bool DEFER_ROWSUM = false;   // blk128 plain path (no dual-half half-merge)
// ----------------------------------------------------------------------------------------------
constexpr int SPARSE_BLOCK = 256;                   // top-k selection granularity (the "block" the idx lists index)
constexpr int BLOCK  = 128;                         // = M_TILE, the MMA datapath tile (NOT the sparse block)
constexpr int M_TILE = BLOCK;
constexpr int M_TILES_PER_CTA = 2;                  // the two 128-query halves of one 256 sparse block
constexpr int K_TILE = 128;
constexpr int KTILES_PER_BLOCK = SPARSE_BLOCK / K_TILE;   // 2 K-tiles per gathered 256-block
constexpr int BLOCKS_PER_KTILE = 1;                 // kept for code shared with blk128 (1 K-tile <= 1 block)
constexpr int KV_HALF = K_TILE / 2;
constexpr int HEAD_DIM = 128;

// "B128 unit" (throughout) = one 128 B swizzle tile = SUB_COLS_BF16 (64) elements wide.
// hd B128-unit = 64 head-dims; token B128-unit = 64 tokens. HEAD_DIM=128 = 2 hd B128-units; a
// 128-token block = 2 token B128-units. TMAs and ring slots are sized in these units.
constexpr int SUB_COLS_BF16 = 64;
constexpr int SUB_COLS_BYTES = SUB_COLS_BF16 * (int)sizeof(__nv_bfloat16);   // 128 B (one 128B swizzle unit)
constexpr int Q_SUBTILES = HEAD_DIM / SUB_COLS_BF16;     // 2
constexpr int K_SUBTILES = HEAD_DIM / SUB_COLS_BF16;     // 2
constexpr int V_SUBTILES = (BLK128 ? K_TILE : KV_HALF) / SUB_COLS_BF16;  // 2
constexpr int Q_SUB_COLS_BYTES = M_TILE * SUB_COLS_BYTES;
constexpr int K_SUB_COLS_BYTES = K_TILE * SUB_COLS_BYTES;
constexpr int Q_TILE_BYTES = Q_SUBTILES * Q_SUB_COLS_BYTES;     // blk64 16 KB, blk128 32 KB
// The K/V ring is a ring of 32 KB slots.
//   blk64: a K-tile or V-tile is 64 KB -> loaded into TWO slots, each with its own full/empty
//     barrier, so the MMA can consume slot 0 while slot 1's TMA is still landing.
//       K-tile (256 keys x 128 hd): split by head-dim -- each slot = all 256 keys x one 64-wide
//         head-dim half. (256 keys = 4 KV blocks of 64 tokens.)
//       V-tile (128 hd x 256 tok): split by tokens -- each slot = full 128 hd x one 128-token
//         half (= 2 KV blocks of 64 tok, one per kv-half).
//   blk128: a K-tile or V-tile is exactly 32 KB -> ONE slot, waited once -- the dense uniform_inline.cu
//     tile shape.
//       K-tile: 1 KV block (128 tok) x 128 hd, both head-dim halves in one slot.
//       V-tile: 128 hd x 128 tok, both token halves in one slot.
constexpr int KV_RING_SLOT_BYTES = 32 * 1024;
constexpr int SLOTS_PER_KV_TILE = BLK128 ? 1 : 2;
// Ring depth in 32 KB slots. SMEM cap 227K: blk64 2*Q(32K) + 4*32K + 2*sO(32K); blk128
// 2*Q(64K) + 3*32K + 2*sO(64K) (the dense kernel also ran 3 stages at this tile size).
constexpr int NUM_KV_STAGES = BLK128 ? 3 : 4;

constexpr int V_BLK_BYTES = HEAD_DIM * SUB_COLS_BYTES;   // 16 KB: one 64-token V^T B128 unit (128 hd x 64 tok)

constexpr int S_COLS = 128;
constexpr int K_ATOMS_PER_TILE = SUB_COLS_BF16 / 16;    // 4
constexpr int SPLIT_P_N    = S_COLS / 4 * 3;            // 96
constexpr int SPLIT_P_ATOM = SPLIT_P_N / 16;            // 6 (BMM2 atom at the split)
constexpr int SPLIT_P_COL  = SPLIT_P_N / 2;     // 48 (u32 P cols written before the empty_bar_spo signal)
constexpr int EX2_FRG_PAIRS = 16;          // 32 elts / fragment = 16 pairs
constexpr int EX2_FRG_CNT   = S_COLS / 32; // = 4 (per-thread 128 score cols)
constexpr int EX2_FREQ      = 16;          // FA4 ex2_emu_freq
constexpr int EX2_RES       = 4;           // FA4 ex2_emu_res

constexpr int O_COLS = HEAD_DIM;
constexpr int TMEM_TOTAL = 512;

// Softmax->correction stats: 128 per M-tile (blk64: 64 rows x 2 halves; blk128: 128 rows).
// Regions: [0] alpha, [1] l, and for blk64 only [2] m (the half-merge needs it).
constexpr int STATS = BLK128 ? M_TILE : 2 * M_TILE;   // 128 either way
constexpr int STAT_REGIONS = BLK128 ? 2 : 3;
// f32 exchange buffer for the blk64 half-merge (one 64x16 chunk); absent for blk128.

// 64-bit SMEM descriptor; the atom walk adds to the LOW word only (the hi/swizzle word is
// walk-invariant) -> one 32-bit add per step, not a 64-bit carry pair. Copied from the dense
// fmha_context_bf16_uniform_inline.cu, and it is LOAD-BEARING FOR CORRECTNESS here, not just
// codegen: with plain uint64 descriptor arithmetic the compiler CSEs the identical descriptor
// across the two M-tiles' MMA groups into base+imm forms that rematerialize the hi word, and
// when both M-tiles issue against the SAME ring slot the second group's operand comes out
// wrong (every output column reads row 0 of the tile). asm volatile pins the serial chain and
// blocks that. Symptom if you "simplify" this back: max|err| 0.80 vs 0.0017, only for the
// K-tiles whose two groups are issued back to back (prologue K-tile 0, epilogue last V).
union SmemDescPair { uint64_t u64; uint2 w; };

__device__ __forceinline__ void desc_add_lo(SmemDescPair& d, uint32_t inc) {
  asm volatile("{\n\t"
      ".reg .b32 lo, hi;\n\t"
      "mov.b64 {lo, hi}, %0;\n\t"
      "add.u32 lo, lo, %1;\n\t"
      "mov.b64 %0, {lo, hi};\n\t"
      "}" : "+l"(d.u64) : "r"(inc));
}

constexpr int W_CORR0 = 8, W_MMA = 12, W_EPI = 13, W_LOAD = 14, W_SCHED = 15;
constexpr int N_WARPS = 16;
constexpr int CLC_STAGES = 2;

extern __shared__ __align__(1024) uint8_t fmha_smem[];


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


struct WorkItem {
  int sample;
  int head;
  int mtile0;
  int mtile1;
  int global_mtile0;
  int global_mtile1;
  int num_kv_blocks;
};
template <bool Q_RASTER>
__device__ __forceinline__ WorkItem decode_workitem(
    int workitem_id, int num_heads, int num_blocks, int packed_mtiles_per_seq,
    unsigned long long magic0, unsigned long long magic1, unsigned long long magic2, const int* q2k_num) {
  WorkItem it;
  const int per_sample = num_heads * packed_mtiles_per_seq;
  it.sample = (int)fdiv((unsigned)workitem_id, magic0);
  const int rem = workitem_id - it.sample * per_sample;
  int p;
  if constexpr (Q_RASTER) {
    it.head = (int)fdiv((unsigned)rem, magic2);
    p       = rem - it.head * packed_mtiles_per_seq;
  } else {
    p       = (int)fdiv((unsigned)rem, magic1);
    it.head = rem - p * num_heads;
  }
  it.mtile0 = 2 * p;       // the 256-block's two 128-query halves (q_base = mtile * M_TILE = mtile*128)
  it.mtile1 = 2 * p + 1;
  // blk256: both M-tiles are the two halves of ONE 256 sparse block -> ONE shared top-k list, indexed by
  // the 256-block p (num_blocks = #256-blocks per (sample,head); packed_mtiles_per_seq == num_blocks).
  it.global_mtile0  = (it.sample * num_heads + it.head) * num_blocks + p;
  it.global_mtile1  = it.global_mtile0;
  it.num_kv_blocks   = q2k_num[it.global_mtile0];
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
//   Q_RASTER         : true = q-block pair innermost (see decode_workitem above); false = head innermost.
//   MHA              : HQ==HK (VSA MHA: q2k_idx is per (sample, head, q-block)). GQA is future work.
// (uniform's IS_CAUSAL / LPT do not apply: block-sparse, non-causal.)
template <int S_LD_COLS = 32, bool FULL_NAMED_BAR = false, bool EX2_EMU = false, bool SPLIT_P = false,
          bool SOFTMAX_THROTTLE = false, bool USE_CLC = true, bool Q_RASTER = true, bool MHA = true,
          int RESCALE_THRESHOLD = 8>
__global__ void __cluster_dims__(1, 1, 1) __launch_bounds__(N_WARPS * 32, 1)
fmha_context_bf16_gen_kernel(const __grid_constant__ CUtensorMap tmap_q,
    const __grid_constant__ CUtensorMap tmap_k, const __grid_constant__ CUtensorMap tmap_v_t,
    const __grid_constant__ CUtensorMap tmap_o, int seqlen,
    int num_heads, float scale_log2, int num_samples, int num_blocks,
    int packed_mtiles_per_seq, int max_kv,
    unsigned long long magic0, unsigned long long magic1, unsigned long long magic2,
    const int* __restrict__ q2k_idx, const int* __restrict__ q2k_num) {

  const int total_workitems = num_samples * num_heads * packed_mtiles_per_seq;

  uint8_t* sQ0 = fmha_smem;
  uint8_t* sQ1 = sQ0 + Q_TILE_BYTES;
  uint8_t* sQ[2] = { sQ0, sQ1 };
  uint8_t* sKV = sQ1 + Q_TILE_BYTES;
  __nv_bfloat16* sO0 = reinterpret_cast<__nv_bfloat16*>(sKV + NUM_KV_STAGES * KV_RING_SLOT_BYTES);
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
  float* alpha_and_l_smem = reinterpret_cast<float*>(tmem_slot + 2);   // [2][3][STATS]


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
    int workitem_id = (int)blockIdx.x;
    while (true) {
      const WorkItem it = decode_workitem<Q_RASTER>(workitem_id, num_heads, num_blocks, packed_mtiles_per_seq, magic0, magic1, magic2, q2k_num);
      const int q_start = it.sample * seqlen;
      const int k_start = it.sample * seqlen;
      const int global_mtile[2]  = { it.global_mtile0, it.global_mtile1 };
      const int mtile[2] = { it.mtile0, it.mtile1 };
      const int num_k_tiles = it.num_kv_blocks * KTILES_PER_BLOCK;   // blk256: 2 K-tiles per selected 256-block

      // KV-block-id cache (warp-collective). q2k_idx[global_mtile*max_kv + k] is
      // this q-block's list of selected KV block ids, k in [0, num_kv_blocks).
      // The warp caches 32 ids at a time in registers -- lane L holds list index
      // window_start+L, filled by ONE coalesced 32-wide load. get_kv_block_id(j) returns
      // id j via __shfl_sync from lane (j & 31): no memory read unless j has left
      // the cached 32-window (then reload). min(window_start+lane, num_kv_blocks
      // -1) clamps tail lanes when < 32 ids remain so they don't read OOB. One
      // cache feeds both K and V; all 32 lanes must run it -> call OUTSIDE
      // elect_one_sync.
      int window_start[2] = { -(1 << 30), -(1 << 30) };
      int kv_block_id_cache[2] = { 0, 0 };
      auto get_kv_block_id = [&](int mtile_idx, int j) -> int {
        if (j >= window_start[mtile_idx] + 32) {   // j only ascends per M-tile
          window_start[mtile_idx] = j & ~31;
          const int idx = min(window_start[mtile_idx] + lane, it.num_kv_blocks - 1);
          kv_block_id_cache[mtile_idx] = q2k_idx[global_mtile[mtile_idx] * max_kv + idx];
        }
        return __shfl_sync(0xffffffffu, kv_block_id_cache[mtile_idx], j & 31);
      };
      // ktile_idx = the K-tile index: 128 keys that BMM1 consumes in one K-step -- here the
      // (ktile_idx % 2)-th 128-token HALF of selected 256-block (ktile_idx / 2).
      // load_k/v_oneslot fills ONE fresh 32 KB ring slot (a whole K-tile or V-tile fits in one slot):
      //   K: one 3D TMA folds both hd B128-units of the 128-token tile, box [SUB_COLS_BF16, BLOCK, 2].
      //   V: one 3D TMA folds both token B128-units of V^T (128 hd x 128 tok).
      auto load_k_oneslot = [&](int mtile_idx, int ktile_idx, int s) {
        const int kv_stage = kv_empty_ph.get_stage();

        wp_begin(wpc, WP_LOAD_WAIT);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar[kv_stage]), kv_empty_ph.get_phase());
        kv_empty_ph.advance();
        wp_end(wpc, WP_LOAD_WAIT);

        wp_begin(wpc, WP_LOAD_ISSUE_K);
        int kv_tok[BLOCKS_PER_KTILE];
        #pragma unroll
        for (int blk = 0; blk < BLOCKS_PER_KTILE; ++blk)
          // blk256: K-tile ktile_idx = the (ktile_idx%2)-th 128-token half of 256-block (ktile_idx/2).
          kv_tok[blk] = k_start + get_kv_block_id(mtile_idx, ktile_idx / KTILES_PER_BLOCK) * SPARSE_BLOCK
                        + (ktile_idx % KTILES_PER_BLOCK) * K_TILE;   // warp-collective
        if (elect_one_sync()) {
          mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar[kv_stage]), KV_RING_SLOT_BYTES);
          if constexpr (BLK128) {
            // one 3D TMA folds both hd B128-units of the single 128-token block: box [SUB_COLS_BF16, BLOCK, 2]
            tma_load_3d(smem_ptr_u32((sKV + kv_stage * KV_RING_SLOT_BYTES)), &tmap_k, smem_ptr_u32(&full_bar[kv_stage]),
                        0, kv_tok[0], it.head * K_SUBTILES);
          } else {
            #pragma unroll
            for (int blk = 0; blk < BLOCKS_PER_KTILE; ++blk)
              tma_load_3d(smem_ptr_u32((sKV + kv_stage * KV_RING_SLOT_BYTES) + blk * BLOCK * SUB_COLS_BYTES),
                          &tmap_k, smem_ptr_u32(&full_bar[kv_stage]),
                          0, kv_tok[blk], it.head * K_SUBTILES + s);   // box [SUB_COLS_BF16, BLOCK, 1]
          }
        }
        wp_end(wpc, WP_LOAD_ISSUE_K);
      };
      auto load_v_oneslot = [&](int mtile_idx, int ktile_idx, int p) {
        const int kv_stage = kv_empty_ph.get_stage();

        wp_begin(wpc, WP_LOAD_WAIT);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar[kv_stage]), kv_empty_ph.get_phase());
        kv_empty_ph.advance();
        wp_end(wpc, WP_LOAD_WAIT);

        wp_begin(wpc, WP_LOAD_ISSUE_V);
        if constexpr (BLK128) {
          // the single 128-token block's V^T (128 hd x 128 tok) = 2 token B128-units folded into ONE 3D TMA
          const int kv_tok0 = k_start + get_kv_block_id(mtile_idx, ktile_idx / KTILES_PER_BLOCK) * SPARSE_BLOCK
                              + (ktile_idx % KTILES_PER_BLOCK) * K_TILE;   // blk256: 128-half of a 256-block
          if (elect_one_sync()) {
            mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar[kv_stage]), KV_RING_SLOT_BYTES);
            // coord [within-unit-tok=0, head*hd row, token B128-unit]; box [SUB_COLS_BF16, hd, V_SUBTILES]
            tma_load_3d(smem_ptr_u32((sKV + kv_stage * KV_RING_SLOT_BYTES)), &tmap_v_t, smem_ptr_u32(&full_bar[kv_stage]),
                        0, it.head * HEAD_DIM, kv_tok0 / SUB_COLS_BF16);
          }
        } else {
          int kv_tok[2];
          #pragma unroll
          for (int h = 0; h < 2; ++h)   // kv-half h's block (2h + p)
            kv_tok[h] = k_start + get_kv_block_id(mtile_idx, ktile_idx * BLOCKS_PER_KTILE + 2 * h + p) * BLOCK;
          if (elect_one_sync()) {
            mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar[kv_stage]), KV_RING_SLOT_BYTES);
            #pragma unroll
            for (int h = 0; h < 2; ++h)
              tma_load_2d(smem_ptr_u32((sKV + kv_stage * KV_RING_SLOT_BYTES) + h * V_BLK_BYTES), &tmap_v_t,
                          smem_ptr_u32(&full_bar[kv_stage]), kv_tok[h], it.head * HEAD_DIM);
          }
        }
        wp_end(wpc, WP_LOAD_ISSUE_V);
      };
      auto load_k = [&](int mtile_idx, int ktile_idx) {
        load_k_oneslot(mtile_idx, ktile_idx, 0);
        if constexpr (!BLK128) load_k_oneslot(mtile_idx, ktile_idx, 1);
      };
      auto load_v = [&](int mtile_idx, int ktile_idx) {
        load_v_oneslot(mtile_idx, ktile_idx, 0);
        if constexpr (!BLK128) load_v_oneslot(mtile_idx, ktile_idx, 1);
      };

      // Produce KV tiles in the BMM1-ahead consume order (single ring; see header). K-first.
      // The CTA's 2 M-tiles are the two halves of ONE 256 sparse block and SHARE its top-k list, so
      // they consume identical K/V bytes: every tile is gathered ONCE and both M-tiles' MMAs issue
      // against that single slot.
      load_k(0, 0);

      // Q for both M-tiles, once (elected lane). MHA: box middle dim = 1, head coord = it.head.
      #pragma unroll
      for (int m = 0; m < M_TILES_PER_CTA; ++m) {
        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_q[m]), q_empty_ph.get_phase());
        const int tok0 = q_start + mtile[m] * BLOCK;
        if (elect_one_sync()) {
          mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar_q[m]), Q_TILE_BYTES);
          // fold both hd B128-units into one 4D TMA (coord [within-unit-subcol=0, head, tok, hd B128-unit=0])
          tma_load_4d(smem_ptr_u32(sQ[m]), &tmap_q, smem_ptr_u32(&full_bar_q[m]),
                      0, it.head, tok0, 0);
        }
      }
      q_empty_ph.advance();
      for (int ktile_idx = 0; ktile_idx + 1 < num_k_tiles; ++ktile_idx) {
        load_v(0, ktile_idx); load_k(0, ktile_idx + 1);
      }
      load_v(0, num_k_tiles - 1);

      if constexpr (USE_CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, /*CTA_GROUP=*/1, /*SUSPEND=*/true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, /*do_release=*/elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        if (!next.valid) break;
        workitem_id = next.n_tile;
      } else {
        workitem_id += gridDim.x;
        if (workitem_id >= total_workitems) break;
      }
    }
  }
  else if (warp_id == W_MMA) {
    setmaxnreg_dec<48>();

    const uint32_t lead = elect_one_sync() ? 1u : 0u;
    constexpr uint32_t DESC_SBO = 1024, DESC_LBO = 16;
    // TMEM layout: S[i] at i*128, O[i] at 256 + i*128 (512 cols total). S is
    // single-buffered; the 2 M-tiles provide the cross-tile overlap instead.
    // blk256 MMAs are the plain full-datapath form: BMM1 m128 n K_TILE(128) k16, BMM2 m128 n HEAD_DIM.
    const uint32_t idesc_qk = make_idesc_bf16_f32(M_TILE, K_TILE, false, false);
    const uint32_t idesc_pv = make_idesc_bf16_f32(M_TILE, BLK128 ? HEAD_DIM : 2 * HEAD_DIM, false, false);
    const uint64_t desc_q0  = build_smem_desc_blackwell(smem_ptr_u32(sQ0), DESC_SBO, DESC_LBO, SmemSwizzleBlackwell::B128);
    const uint64_t desc_kv0 = build_smem_desc_blackwell(smem_ptr_u32(sKV), DESC_SBO, DESC_LBO, SmemSwizzleBlackwell::B128);
    // >> 4: the SMEM descriptor address field is in 16-byte units (addr >> 4).
    constexpr uint64_t KV_DESC_DELTA    = KV_RING_SLOT_BYTES >> 4;
    constexpr uint64_t Q_SUB_DELTA        = Q_SUB_COLS_BYTES >> 4;
    constexpr uint64_t Q_MTILE_DESC_DELTA = Q_TILE_BYTES >> 4;

    PhaseTracker<NUM_KV_STAGES> kv_ph;
    PhaseTracker<1> q_ph;
    PhaseTracker<1> spo_ph;
    [[maybe_unused]] int clc_stage = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;
    int workitem_id = (int)blockIdx.x;
    while (true) {
      const WorkItem it = decode_workitem<Q_RASTER>(workitem_id, num_heads, num_blocks, packed_mtiles_per_seq, magic0, magic1, magic2, q2k_num);
      const int num_k_tiles = it.num_kv_blocks * KTILES_PER_BLOCK;   // blk256: 2 K-tiles per selected 256-block

      // BMM1: S[i] = Q[i] @ K (plain m128, one MMA per hd B128-unit step).
      // blk256 SHARED KV STREAM (the dense fmha_context_bf16_uniform_inline.cu discipline): both M-tiles are
      // the two 128-query halves of ONE 256 sparse block, so they consume the SAME K/V tiles. The ring slot
      // is waited and freed by the CALLER -- bmm1/bmm2 do no ring wait/commit themselves -- so one gather
      // can feed both M-tiles' MMAs.
      auto bmm1 = [&](int i, int slot) {
        const uint32_t s_tmem_addr = tmem_base + (uint32_t)(i * S_COLS);
        // one slot holds both hd B128-units (unit s at s*K_SUB_COLS_BYTES in-slot); the caller waits it.
        // Descriptors are WALKED (desc_add_lo), never recomputed as base+imm -- see SmemDescPair.
        SmemDescPair da, db;
        da.u64 = desc_q0;  da.w.x += (uint32_t)(i * (int)Q_MTILE_DESC_DELTA);
        db.u64 = desc_kv0; db.w.x += (uint32_t)(slot * (int)KV_DESC_DELTA);
        #pragma unroll
        for (int s = 0; s < Q_SUBTILES; ++s) {
          wp_begin(wpc, WP_MMA_ISSUE);
          // straight-line issue: address math unguarded (warp-uniform -> URs),
          // the elect predicate rides ON the instruction (no BSSY/BSYNC).
          #pragma unroll
          for (int ki = 0; ki < K_ATOMS_PER_TILE; ++ki) {
            const bool enable_d = (s != 0) || (ki != 0);
            tcgen05_mma_f16_ss_lead(lead, s_tmem_addr, da.u64, db.u64, idesc_qk, enable_d);
            desc_add_lo(da, 2); desc_add_lo(db, 2);
          }
          desc_add_lo(da, (uint32_t)(Q_SUB_DELTA - 2 * K_ATOMS_PER_TILE));
          desc_add_lo(db, (uint32_t)((K_SUB_COLS_BYTES >> 4) - 2 * K_ATOMS_PER_TILE));
          wp_end(wpc, WP_MMA_ISSUE);
        }

        wp_begin(wpc, WP_MMA_COMMIT);
        tcgen05_commit1_lead(lead, smem_ptr_u32(&full_bar_spo[i]));
        wp_end(wpc, WP_MMA_COMMIT);
      };
      // BMM2: O[i] += P[i] @ V -- plain tcgen05.mma TS, one 128x128 accumulator per M-tile (P is read
      // from the same TMEM columns the softmax wrote it into). V ring-slot layout: see the p-loop below.
      auto bmm2 = [&](int i, int slot, bool first_ktile, auto last_c) {
        constexpr bool last_ktile = decltype(last_c)::value;
        const uint32_t s_tmem_addr = tmem_base + (uint32_t)(i * S_COLS);
        const uint32_t o_tmem_addr = tmem_base + (uint32_t)(2 * S_COLS + i * O_COLS);
        // V is read in V_SUBTILES=2 pieces; both token B128-units share ONE ring slot (unit p at
        // p*V_BLK_BYTES in-slot). The caller waits the slot.
        // Descriptors are WALKED (desc_add_lo), never recomputed as base+imm -- see SmemDescPair.
        SmemDescPair dbV;
        dbV.u64 = desc_kv0; dbV.w.x += (uint32_t)(slot * (int)KV_DESC_DELTA);
        #pragma unroll
        for (int p = 0; p < V_SUBTILES; ++p) {
          #pragma unroll
          for (int ki = 0; ki < K_ATOMS_PER_TILE; ++ki) {
            const int a = p * K_ATOMS_PER_TILE + ki;              // flat P/O atom 0..7
            if constexpr (SPLIT_P) if (a == SPLIT_P_ATOM) {
              wp_begin(wpc, WP_MMA_WAIT_P);
              mbarrier_wait_parity(smem_ptr_u32(&full_bar_p_last[i]), spo_ph.get_phase());
              wp_end(wpc, WP_MMA_WAIT_P);
            }
            const bool accumulate = (!first_ktile) || (a != 0);
            wp_begin(wpc, WP_MMA_ISSUE);
            tcgen05_mma_f16_ts_1sm_lead(lead, o_tmem_addr, s_tmem_addr + (uint32_t)(a * 8), dbV.u64, idesc_pv, accumulate);
            desc_add_lo(dbV, 2);
            wp_end(wpc, WP_MMA_ISSUE);
          }
          desc_add_lo(dbV, (uint32_t)((V_BLK_BYTES >> 4) - 2 * K_ATOMS_PER_TILE));
        }

        if constexpr (last_ktile) {
          wp_begin(wpc, WP_MMA_COMMIT);
          tcgen05_commit1_lead(lead, smem_ptr_u32(&full_bar_o_acc[i]));
          wp_end(wpc, WP_MMA_COMMIT);
        }
      };

      // prologue: BMM1 of K-tile 0 -- ONE slot shared by both M-tiles, waited once, freed after both.
      int kv_stage = kv_ph.get_stage();
      wp_begin(wpc, WP_MMA_WAIT_FULL_K);
      mbarrier_wait_parity(smem_ptr_u32(&full_bar[kv_stage]), kv_ph.get_phase());
      kv_ph.advance();
      wp_end(wpc, WP_MMA_WAIT_FULL_K);
      #pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i) {
        wp_begin(wpc, WP_MMA_WAIT_FULL_Q);
        mbarrier_wait_parity(smem_ptr_u32(&full_bar_q[i]), q_ph.get_phase());
        wp_end(wpc, WP_MMA_WAIT_FULL_Q);
        bmm1(i, kv_stage);
      }
      wp_begin(wpc, WP_MMA_COMMIT);
      tcgen05_commit1_lead(lead, smem_ptr_u32(&empty_bar[kv_stage]));
      wp_end(wpc, WP_MMA_COMMIT);

      // main loop: BMM2(k) then BMM1-ahead(k+1). ONE shared K/V stream: V(k) and K(k+1) are each a
      // SINGLE ring slot that both M-tiles issue against -- the CTA's 2 M-tiles are the two 128-query
      // halves of one 256 sparse block, so they consume identical K/V tiles. V(k) is freed after the
      // last M-tile's BMM2, K(k+1) is waited by i==0 and freed after both M-tiles' BMM1.
      for (int k = 0; k + 1 < num_k_tiles; ++k) {
        const int kv_stage_v = kv_ph.get_stage();
        wp_begin(wpc, WP_MMA_WAIT_FULL_V);
        mbarrier_wait_parity(smem_ptr_u32(&full_bar[kv_stage_v]), kv_ph.get_phase());
        kv_ph.advance();
        wp_end(wpc, WP_MMA_WAIT_FULL_V);

        int kv_stage_next = 0;
        #pragma unroll
        for (int i = 0; i < M_TILES_PER_CTA; ++i) {
          wp_begin(wpc, WP_MMA_WAIT_P);
          mbarrier_wait_parity(smem_ptr_u32(&empty_bar_spo[i]), spo_ph.get_phase());
          wp_end(wpc, WP_MMA_WAIT_P);
          bmm2(i, kv_stage_v, /*first_ktile=*/(k == 0), std::false_type{});

          if (i == 0) {
            kv_stage_next = kv_ph.get_stage();
            wp_begin(wpc, WP_MMA_WAIT_FULL_K);
            mbarrier_wait_parity(smem_ptr_u32(&full_bar[kv_stage_next]), kv_ph.get_phase());
            kv_ph.advance();
            wp_end(wpc, WP_MMA_WAIT_FULL_K);
          }
          if (i == M_TILES_PER_CTA - 1) {
            wp_begin(wpc, WP_MMA_COMMIT);
            tcgen05_commit1_lead(lead, smem_ptr_u32(&empty_bar[kv_stage_v]));
            wp_end(wpc, WP_MMA_COMMIT);
          }
          bmm1(i, kv_stage_next);
        }
        wp_begin(wpc, WP_MMA_COMMIT);
        tcgen05_commit1_lead(lead, smem_ptr_u32(&empty_bar[kv_stage_next]));
        wp_end(wpc, WP_MMA_COMMIT);

        spo_ph.advance();
      }

      wp_begin(wpc, WP_MMA_COMMIT);
      #pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i)
        tcgen05_commit1_lead(lead, smem_ptr_u32(&empty_bar_q[i]));
      wp_end(wpc, WP_MMA_COMMIT);
      q_ph.advance();

      // epilogue: BMM2 of the last K-tile -> final O. One shared V slot, as in the prologue.
      kv_stage = kv_ph.get_stage();
      wp_begin(wpc, WP_MMA_WAIT_FULL_V);
      mbarrier_wait_parity(smem_ptr_u32(&full_bar[kv_stage]), kv_ph.get_phase());
      kv_ph.advance();
      wp_end(wpc, WP_MMA_WAIT_FULL_V);
      #pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i) {
        wp_begin(wpc, WP_MMA_WAIT_P);
        mbarrier_wait_parity(smem_ptr_u32(&empty_bar_spo[i]), spo_ph.get_phase());
        wp_end(wpc, WP_MMA_WAIT_P);
        bmm2(i, kv_stage, /*first_ktile=*/(num_k_tiles == 1), std::true_type{});
      }
      wp_begin(wpc, WP_MMA_COMMIT);
      tcgen05_commit1_lead(lead, smem_ptr_u32(&empty_bar[kv_stage]));
      wp_end(wpc, WP_MMA_COMMIT);

      spo_ph.advance();

      if constexpr (USE_CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, /*CTA_GROUP=*/1, /*SUSPEND=*/true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, /*do_release=*/elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        if (!next.valid) break;
        workitem_id = next.n_tile;
      } else {
        workitem_id += gridDim.x;
        if (workitem_id >= total_workitems) break;
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
    int workitem_id = (int)blockIdx.x;
    while (true) {
      const WorkItem it = decode_workitem<Q_RASTER>(workitem_id, num_heads, num_blocks, packed_mtiles_per_seq, magic0, magic1, magic2, q2k_num);
      const int q_start = it.sample * seqlen;
      const int mtile[2] = { it.mtile0, it.mtile1 };

      #pragma unroll
      for (int m = 0; m < M_TILES_PER_CTA; ++m) {
        wp_begin(wpc, WP_EPI_WAIT_TMEM);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_o_epi[m]), full_o_ph.get_phase());
        wp_end(wpc, WP_EPI_WAIT_TMEM);

        wp_begin(wpc, WP_EPI_STORE);
        if (elect_one_sync()) {
          const int tok0 = q_start + mtile[m] * BLOCK;
          // fold both hd B128-units into one 4D store (coord [within-unit-subcol=0, head, tok, hd B128-unit=0])
          tma_store_4d(&tmap_o, 0, it.head, tok0, 0,
                       smem_ptr_u32(reinterpret_cast<const uint8_t*>(sO_bufs[m])));
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
        workitem_id = next.n_tile;
      } else {
        workitem_id += gridDim.x;
        if (workitem_id >= total_workitems) break;
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

    // corr thread t = corr_warp_id*32 + lane owns TMEM lane t = O row t of the M-tile (1 lane per
    // row, no half-merge: BLK128 is frozen true here, so corr_row == corr_tid and kv_half0 == true).
    const int corr_warp_id = warp_id - W_CORR0;
    const int corr_tid = corr_warp_id * 32 + lane;
    const int corr_row = BLK128 ? corr_tid : (corr_tid & 63);   // O row within the M-tile
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
    int workitem_id = (int)blockIdx.x;
    while (true) {
      const WorkItem it = decode_workitem<Q_RASTER>(workitem_id, num_heads, num_blocks, packed_mtiles_per_seq, magic0, magic1, magic2, q2k_num);
      const int num_k_tiles = it.num_kv_blocks * KTILES_PER_BLOCK;   // blk256: 2 K-tiles per selected 256-block

      // ktile 0: no rescale (no prior O); consume alpha + release the scale slot.
      #pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i) {
        wp_begin(wpc, WP_CORR_WAIT);
        if constexpr (FULL_NAMED_BAR) full_bar_wait(i, corr_warp_id);
        else mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_alpha[i]), alpha_ph.get_phase());
        mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
        wp_end(wpc, WP_CORR_WAIT);
      }
      if constexpr (!FULL_NAMED_BAR) alpha_ph.advance();

      for (int k = 1; k < num_k_tiles; ++k) {
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
            // O(ktile_idx-1) is done: BMM1(ktile_idx) trails BMM2(ktile_idx-1) in the in-order tcgen05 pipe.
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

      // epilogue: O *= 1/l -> bf16 -> sO[i] -> signal W_EPI (full_bar_o_epi).
      #pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i) {
        wp_begin(wpc, WP_CORR_WAIT);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_o_acc[i]), o_acc_ph.get_phase());
        if constexpr (FULL_NAMED_BAR) full_bar_wait(i, corr_warp_id);
        else mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_l[i]), o_acc_ph.get_phase());
        wp_end(wpc, WP_CORR_WAIT);

        wp_begin(wpc, WP_CORR_EPI);
        // blk64 HALF-MERGE: partner threads t / t^64 hold the two per-half O partials of row
        // (t & 63); 2-way split-KV combine out = (beta_0*O_0 + beta_1*O_1) / (beta_0*l_0 + beta_1*l_1),
        // beta_h = exp2((m_h - m_tot)*scale_log2). Both partners compute l_tot from the SMEM
        // stats; only the O values are exchanged, in 16-col f32 chunks via o_xchg.
        float scale_own;
        if constexpr (BLK128) {
          // plain M=128 accumulator: no halves, no merge -- scale = 1/l (classic dense epilogue).
          const float l = alpha_and_l_smem[(i * STAT_REGIONS + 1) * STATS + corr_tid];
          mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
          scale_own = (l > 0.f) ? rcp_approx_ftz_f32(l) : 0.f;
        } else {
          const float l_own = alpha_and_l_smem[(i * STAT_REGIONS + 1) * STATS + corr_tid];
          const float m_own = alpha_and_l_smem[(i * STAT_REGIONS + 2) * STATS + corr_tid];
          const float l_par = alpha_and_l_smem[(i * STAT_REGIONS + 1) * STATS + (corr_tid ^ 64)];
          const float m_par = alpha_and_l_smem[(i * STAT_REGIONS + 2) * STATS + (corr_tid ^ 64)];
          mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
          const float d = (m_own - m_par) * scale_log2;
          const float beta_lo = ex2_approx_f32(-fabsf(d));
          const float beta_own = (d >= 0.f) ? 1.f : beta_lo;
          const float beta_par = (d >= 0.f) ? beta_lo : 1.f;
          const float l_tot = beta_own * l_own + beta_par * l_par;
          scale_own = (l_tot > 0.f) ? beta_own * rcp_approx_ftz_f32(l_tot) : 0.f;
        }
        const float2 scale2 = f32x2_splat(scale_own);
        const uint32_t o_tmem_addr = tmem_base + (uint32_t)(2 * S_COLS + i * O_COLS) + ((uint32_t)(corr_warp_id * 32) << 16);
        if constexpr (BLK128) {
          // plain M=128: no halves. Fuse the O*=1/l scale into the per-vv bf16 pack (overlaps FMUL2+CVT).
          #pragma unroll
          for (int c0 = 0; c0 < HEAD_DIM; c0 += 16) {
            uint32_t o_regs[16];
            tcgen05_ld_32x32b_x16(o_tmem_addr + (uint32_t)c0, o_regs);
            if (c0 == 0) mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_o_epi[i]), o_epi_empty_ph.get_phase());
            const float2* o2 = reinterpret_cast<const float2*>(o_regs);
            const int s = c0 / SUB_COLS_BF16;
            const int v_base = (c0 % SUB_COLS_BF16) / 8;
            __nv_bfloat16* so_sub = sO_bufs[i] + s * (M_TILE * SUB_COLS_BF16);
            #pragma unroll
            for (int vv = 0; vv < 2; ++vv) {
              const int v = v_base + vv;
              const float2 r0 = fmul2(o2[vv * 4 + 0], scale2);
              const float2 r1 = fmul2(o2[vv * 4 + 1], scale2);
              const float2 r2 = fmul2(o2[vv * 4 + 2], scale2);
              const float2 r3 = fmul2(o2[vv * 4 + 3], scale2);
              uint4 packed;
              packed.x = cvt_f32x2_to_bf16x2(r0.x, r0.y);
              packed.y = cvt_f32x2_to_bf16x2(r1.x, r1.y);
              packed.z = cvt_f32x2_to_bf16x2(r2.x, r2.y);
              packed.w = cvt_f32x2_to_bf16x2(r3.x, r3.y);
              *reinterpret_cast<uint4*>(&so_sub[corr_row * SUB_COLS_BF16 + (v ^ (corr_row & 7)) * 8]) = packed;
            }
          }
        } else {
          // blk64 dual-pack half-merge IN PLACE through sO_bufs[i] (corr owns it here: empty_bar_o_epi
          // gates W_EPI until the full_bar_o_epi arrive, no dedicated o_xchg). half-1 packs its scaled
          // O partial to bf16 in so_sub; half-0 reads it back, adds its own scaled partial, overwrites
          // so_sub with the merged row. One produce + one drain bar_sync replace the per-chunk pair.
          mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_o_epi[i]), o_epi_empty_ph.get_phase());
          if (!kv_half0) {
            #pragma unroll
            for (int c0 = 0; c0 < HEAD_DIM; c0 += 16) {
              uint32_t o_regs[16];
              tcgen05_ld_32x32b_x16(o_tmem_addr + (uint32_t)c0, o_regs);
              float2* o2 = reinterpret_cast<float2*>(o_regs);
              #pragma unroll
              for (int e = 0; e < 8; ++e) o2[e] = fmul2(o2[e], scale2);
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
          }
          bar_sync<9>(128);   // half-1's bf16 partials now visible in sO
          if (kv_half0) {
            #pragma unroll
            for (int c0 = 0; c0 < HEAD_DIM; c0 += 16) {
              uint32_t o_regs[16];
              tcgen05_ld_32x32b_x16(o_tmem_addr + (uint32_t)c0, o_regs);
              float2* o2 = reinterpret_cast<float2*>(o_regs);
              #pragma unroll
              for (int e = 0; e < 8; ++e) o2[e] = fmul2(o2[e], scale2);
              const int s = c0 / SUB_COLS_BF16;
              const int v_base = (c0 % SUB_COLS_BF16) / 8;
              __nv_bfloat16* so_sub = sO_bufs[i] + s * (M_TILE * SUB_COLS_BF16);
              #pragma unroll
              for (int vv = 0; vv < 2; ++vv) {
                const int v = v_base + vv;
                __nv_bfloat16* dst = &so_sub[corr_row * SUB_COLS_BF16 + (v ^ (corr_row & 7)) * 8];
                uint4 h1 = *reinterpret_cast<uint4*>(dst);
                o2[vv * 4 + 0] = fadd2(o2[vv * 4 + 0], __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&h1.x)));
                o2[vv * 4 + 1] = fadd2(o2[vv * 4 + 1], __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&h1.y)));
                o2[vv * 4 + 2] = fadd2(o2[vv * 4 + 2], __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&h1.z)));
                o2[vv * 4 + 3] = fadd2(o2[vv * 4 + 3], __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&h1.w)));
                uint4 packed;
                packed.x = cvt_f32x2_to_bf16x2(o2[vv * 4 + 0].x, o2[vv * 4 + 0].y);
                packed.y = cvt_f32x2_to_bf16x2(o2[vv * 4 + 1].x, o2[vv * 4 + 1].y);
                packed.z = cvt_f32x2_to_bf16x2(o2[vv * 4 + 2].x, o2[vv * 4 + 2].y);
                packed.w = cvt_f32x2_to_bf16x2(o2[vv * 4 + 3].x, o2[vv * 4 + 3].y);
                *reinterpret_cast<uint4*>(dst) = packed;
              }
            }
          }
          bar_sync<9>(128);   // half-0 done reading O from TMEM -> O slot may be released below
        }
        // O ld done -> MMA may reuse the O slot
        tcgen05_fence_before_thread_sync();

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
        workitem_id = next.n_tile;
      } else {
        workitem_id += gridDim.x;
        if (workitem_id >= total_workitems) break;
      }
    }
  }
  else {
    setmaxnreg_inc<192>();

    // warp-uniform hint (make_warp_uniform): shfl-broadcast warp_id so m_tile/warp_in_group and the
    // TMEM address math promote to uniform registers (R2UR.BROADCAST), freeing vector regs.
    const int warp_id_u = __shfl_sync(0xffffffffu, warp_id, 0);
    const int m_tile = warp_id_u < 4 ? 0 : 1;
    const int warp_in_group = warp_id_u & 3;
    // band thread t = warp_in_group*32 + lane owns TMEM lane t = S row t of the M-tile, and runs one
    // online softmax over that row's full S_COLS(128) keys (1 lane per row, no per-half partials).
    const int sm_tid = warp_in_group * 32 + lane;
    const uint32_t alpha_slot_u32 = smem_ptr_u32(&alpha_and_l_smem[(m_tile * STAT_REGIONS + 0) * STATS + sm_tid]);
    const uint32_t s_tmem_addr = tmem_base + (uint32_t)(m_tile * S_COLS) + ((uint32_t)(warp_in_group * 32) << 16);
    PhaseTracker<1> spo_ph;
    PhaseTracker<1> scale_empty_ph;
    [[maybe_unused]] int clc_stage = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;

    int workitem_id = (int)blockIdx.x;
    while (true) {
      const WorkItem it = decode_workitem<Q_RASTER>(workitem_id, num_heads, num_blocks, packed_mtiles_per_seq, magic0, magic1, magic2, q2k_num);
      const int num_k_tiles = it.num_kv_blocks * KTILES_PER_BLOCK;   // blk256: 2 K-tiles per selected 256-block

      float m_run = -INFINITY, l_run = 0.f;
      wp_begin(wpc, WP_SM_WAIT_SCALE);
      mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_alpha_and_l[m_tile]), scale_empty_ph.get_phase());
      scale_empty_ph.advance();
      wp_end(wpc, WP_SM_WAIT_SCALE);

      // k==0 (alpha-less first block, no running max to correct against) PEELED via
      // is_first_c = std::true_type; the steady body drops the k==0 branches. Mirrors
      // uniform_inline's softmax_step, minus MASKED (VSA is non-causal, no mask).
      auto softmax_step = [&](auto is_first_c) {
        constexpr bool IS_FIRST = decltype(is_first_c)::value;
        wp_begin(wpc, WP_SM_WAIT_S);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_spo[m_tile]), spo_ph.get_phase());
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

        // VSA: no mask (every selected block is full, non-causal).

        // rmax via 4 independent FMNMX3 accumulators (4-way ILP), OLD MAX FOLDED
        // into accumulator 0 (inline-base uplift #7). S_COLS % 8 == 0.
        float rmax0 = m_run, rmax1 = -INFINITY, rmax2 = -INFINITY, rmax3 = -INFINITY;
        #pragma unroll
        for (int j = 0; j < S_COLS; j += 8) {
          rmax0 = fmaxf(fmaxf(rmax0, scores[j + 0]), scores[j + 1]);
          rmax1 = fmaxf(fmaxf(rmax1, scores[j + 2]), scores[j + 3]);
          rmax2 = fmaxf(fmaxf(rmax2, scores[j + 4]), scores[j + 5]);
          rmax3 = fmaxf(fmaxf(rmax3, scores[j + 6]), scores[j + 7]);
        }
        float new_m = fmaxf(fmaxf(rmax0, rmax1), fmaxf(rmax2, rmax3));
        float alpha = 0.0f;
        if constexpr (!IS_FIRST) {
          // FA4 sticky max (uplift #7): a small max drift costs a full O rescale
          // downstream; below the threshold keep the old max EXACTLY and publish
          // alpha == 1.0 -- pairs corr's __all_sync(alpha == 1.0f) skip vote.
          const float acc_scale_ = (m_run - new_m) * scale_log2;
          if (acc_scale_ >= -(float)RESCALE_THRESHOLD) { new_m = m_run; alpha = 1.0f; }
          else                     { alpha = ex2_approx_f32(acc_scale_); }
          // volatile STS publish (uplift #8, MHA form): keeps ptxas from sinking the store past the bar arrive.
          sts_f32(alpha_slot_u32, alpha);
        }
        if constexpr (FULL_NAMED_BAR) full_bar_arrive(m_tile, warp_in_group);
        else mbarrier_arrive(smem_ptr_u32(&full_bar_alpha[m_tile]));

        // fused ffma2(scale) + exp2 + row-sum + bf16 pack: this row's 128 keys -> 64 u32 P cols.
        const float2 scale2 = f32x2_splat(scale_log2);
        const float2 neg_m_scaled2 = f32x2_splat(-new_m * scale_log2);
        uint32_t p_regs[S_COLS / 2];
        [[maybe_unused]] float2 lt2_live = make_float2(IS_FIRST ? 0.0f : l_run * alpha, 0.0f);  // seed rescale (blk128 live)
        #pragma unroll
        for (int c = 0; c < S_COLS / 2; ++c) {
          const float2 a2 = ffma2(scores2[c], scale2, neg_m_scaled2);
          // exp2 written in place into scores2[c] (aliases s_regs; free read-back for row-sum + pack)
          if constexpr (EX2_EMU) {
            const int jj = c / EX2_FRG_PAIRS;
            const int kk = 2 * (c % EX2_FRG_PAIRS);
            const bool use_hw = (kk % EX2_FREQ < EX2_FREQ - EX2_RES) || (jj >= EX2_FRG_CNT - 1);
            scores2[c] = use_hw ? make_float2(ex2_approx_f32(a2.x), ex2_approx_f32(a2.y))
                                : ex2_emu_f32x2(a2.x, a2.y);
          } else {
            scores2[c] = make_float2(ex2_approx_f32(a2.x), ex2_approx_f32(a2.y));
          }
          if constexpr (!DEFER_ROWSUM) lt2_live = fadd2(lt2_live, scores2[c]);  // rowsum live (blk128)
          p_regs[c] = cvt_f32x2_to_bf16x2(scores2[c].x, scores2[c].y);
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
          tcgen05_wait_st();
          tcgen05_fence_before_thread_sync();
          mbarrier_arrive(smem_ptr_u32(&full_bar_p_last[m_tile]));
        } else {
          tcgen05_st_32x32b_x32(p_tmem_addr,      *reinterpret_cast<uint32_t(*)[32]>(&p_regs[0]));
          tcgen05_st_32x32b_x32(p_tmem_addr + 32, *reinterpret_cast<uint32_t(*)[32]>(&p_regs[32]));
          tcgen05_wait_st();
          tcgen05_fence_before_thread_sync();
          mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[m_tile]));
        }
        wp_end(wpc, WP_SM_STORE_P);

        spo_ph.advance();
        float2 lt2 = lt2_live;
        if constexpr (DEFER_ROWSUM) {
          // deferred row-sum: P is already published, so this FADD2 chain overlaps
          // the MMA warp's BMM2 instead of gating it.
          float2 lt2a = make_float2(IS_FIRST ? 0.0f : l_run * alpha, 0.0f), lt2b = make_float2(0.f, 0.f);  // seed rescale (blk64 tree)
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
        l_run = lt2.x + lt2.y; m_run = new_m;   // rescale already folded into the reduction seed
        wp_begin(wpc, WP_SM_WAIT_SCALE);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_alpha_and_l[m_tile]), scale_empty_ph.get_phase());
        scale_empty_ph.advance();
        wp_end(wpc, WP_SM_WAIT_SCALE);
      };
      if (num_k_tiles > 0) softmax_step(std::true_type{});
      for (int k = 1; k < num_k_tiles; ++k) softmax_step(std::false_type{});

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
        workitem_id = next.n_tile;
      } else {
        workitem_id += gridDim.x;
        if (workitem_id >= total_workitems) break;
      }
    }
  }
  wp_flush(wpc);
  __syncthreads();
  if (warp_id == 0) tcgen05_dealloc<1>(tmem_base, TMEM_TOTAL);
}

// ============================== driver ====================================

// CPU reference: VSA fine block-sparse attention in fp32. For each query row of a q-block,
// attend to ALL tokens of the SELECTED KV blocks only, softmax, @V, bf16 round on read-back.
// Layout: Q/K/V natural [token, head, hd]; token = b*S + local. global_mtile = (b*H+h)*num_blocks + mtile.
static void cpu_vsa_ref(const __nv_bfloat16* hQ, const __nv_bfloat16* hK,
                        const __nv_bfloat16* hV, float* hO,
                        int B, int H, int S, int hd, int num_blocks, int max_kv,
                        const int* q2k_idx, const int* q2k_num) {
  const float scale = 1.0f / sqrtf((float)hd);
  const long Nq = (long)B * S;
  for (long i = 0; i < Nq * H * hd; ++i) hO[i] = 0.f;

  #pragma omp parallel for schedule(dynamic)
  for (int bhq = 0; bhq < B * H * num_blocks; ++bhq) {
    const int mtile = bhq % num_blocks;
    const int bh   = bhq / num_blocks;
    const int h    = bh % H;
    const int b    = bh / H;
    const int global_mtile  = bhq;                        // (b*H + h)*num_blocks + mtile
    const int num_kv_blocks  = q2k_num[global_mtile];

    for (int qi = 0; qi < SPARSE_BLOCK; ++qi) {
      const long qp = (long)b * S + (long)mtile * SPARSE_BLOCK + qi;
      std::vector<float> z((size_t)num_kv_blocks * SPARSE_BLOCK);
      float row_max = -INFINITY;
      int idx = 0;
      for (int kk = 0; kk < num_kv_blocks; ++kk) {
        const int blk = q2k_idx[global_mtile * max_kv + kk];
        for (int kj = 0; kj < SPARSE_BLOCK; ++kj) {
          const long kp = (long)b * S + (long)blk * SPARSE_BLOCK + kj;
          float dot = 0.f;
          for (int e = 0; e < hd; ++e)
            dot += __bfloat162float(hQ[(qp * H + h) * hd + e])
                 * __bfloat162float(hK[(kp * H + h) * hd + e]);
          z[idx++] = dot * scale;
          row_max = fmaxf(row_max, z[idx - 1]);
        }
      }
      float sum = 0.f;
      for (int j = 0; j < num_kv_blocks * SPARSE_BLOCK; ++j) { z[j] = expf(z[j] - row_max); sum += z[j]; }
      if (sum == 0.f) continue;
      const float inv_sum = 1.f / sum;
      for (int e = 0; e < hd; ++e) {
        float acc = 0.f;
        idx = 0;
        for (int kk = 0; kk < num_kv_blocks; ++kk) {
          const int blk = q2k_idx[global_mtile * max_kv + kk];
          for (int kj = 0; kj < SPARSE_BLOCK; ++kj) {
            const long kp = (long)b * S + (long)blk * SPARSE_BLOCK + kj;
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

// N(0,1) Gaussian fill via Box-Muller (matches flashinfer's torch.randn input
// distribution). Env-guarded test path only (VSA_GAUSS) -- default fill is uniform.
static void fillg(__nv_bfloat16* h, long n, unsigned seed) {
  for (long i = 0; i < n; ++i) {
    uint32_t x = (uint32_t)i * 2654435761u + seed * 40503u + 0x9e3779b9u;
    x ^= x >> 15; x *= 2246822519u; x ^= x >> 13; x *= 3266489917u; x ^= x >> 16;
    uint32_t y = x * 2654435761u + 0x85ebca6bu; y ^= y >> 13; y *= 3266489917u; y ^= y >> 16;
    float u1 = (float)((x % 1000003u) + 1u) / 1000004.0f;   // (0,1]
    float u2 = (float)(y % 1000003u) / 1000003.0f;          // [0,1)
    float z = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
    h[i] = __float2bfloat16(z);
  }
}

struct Sh { int B, H, num_blocks, topk, hd; const char* lab; };

static double run(const Sh& sh, bool verify) {
  const int  B = sh.B, H = sh.H, num_blocks = sh.num_blocks, topk = sh.topk, hd = sh.hd;
  const int  S = num_blocks * SPARSE_BLOCK;        // seqlen; num_blocks = #256 sparse blocks
  const int  max_kv = topk;                        // tight: exactly topk selected blocks
  const long tq = (long)B * S;                     // total tokens
  const int  packed_mtiles_per_seq = num_blocks;   // blk256: a CTA does ONE 256-block (= its 2 M-tiles)
  const int  num_global_q_blocks = B * H * num_blocks;
  const int  total_work = B * H * packed_mtiles_per_seq;
  // blk256: one work-item = one 256-block (no adjacent-pair tiling), so num_blocks need not be even.

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
    auto FILL = getenv("VSA_GAUSS") ? fillg : fillr;   // VSA_GAUSS: match FI's randn N(0,1)
    FILL(hQ.data(), hQ.size(), 11);
    FILL(hK.data(), hK.size(), 22);
    FILL(hV.data(), hV.size(), 33);
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

  // ---- q2k index: topk DISTINCT block ids per (b,h,mtile), fixed density (num == topk) ----
  std::vector<int> hq2k_idx((size_t)num_global_q_blocks * max_kv, 0);
  std::vector<int> hq2k_num(num_global_q_blocks, topk);
  if (load_npy) {   // unified bench: head-independent [num_blocks, topk] list, broadcast across heads
    char p[64]; snprintf(p, sizeof p, "/idx_S%d_blk%d.npy", S, SPARSE_BLOCK);
    auto idx = npy_load_vec<int32_t>(std::string(load_npy) + p);
    if (idx.size() != (size_t)num_blocks * topk) { fprintf(stderr, "LOAD_NPY: idx size %zu != %d\n", idx.size(), num_blocks * topk); exit(1); }
    for (int global_mtile = 0; global_mtile < num_global_q_blocks; ++global_mtile) {
      const int mtile = global_mtile % num_blocks;
      for (int i = 0; i < topk; ++i) hq2k_idx[(size_t)global_mtile * max_kv + i] = idx[(size_t)mtile * topk + i];
    }
  } else {
    block_sparse_bf16_benchmark::select_blocks(hq2k_idx, num_blocks, topk);
  }
  int *dq2k_idx, *dq2k_num;
  CUDA_CHECK(cudaMalloc(&dq2k_idx, hq2k_idx.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dq2k_num, hq2k_num.size() * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(dq2k_idx, hq2k_idx.data(), hq2k_idx.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dq2k_num, hq2k_num.data(), hq2k_num.size() * sizeof(int), cudaMemcpyHostToDevice));

  // ---- TMA tensor maps ----
  // O: 3D over [hd, H, tq]; box [hd-subtile x 1 (MHA) x M_TILE]. Matches the store's 3D box.
  // Q: 4D [hd-subcol, H, tq, hd B128-unit]; box [SUB_COLS_BF16, 1, M_TILE, Q_SUBTILES] folds both hd B128-units
  //    of HEAD_DIM into ONE TMA (hd B128-unit slowest -> matches per-subtile SMEM stride Q_SUB_COLS_BYTES).
  CUtensorMap tq_, tk_, tvt_, to_;
  {
    uint64_t gd[4] = { (uint64_t)SUB_COLS_BF16, (uint64_t)H, (uint64_t)tq, (uint64_t)Q_SUBTILES };
    uint64_t gs[3] = { (uint64_t)hd * 2u, (uint64_t)((long)H * hd) * 2u, (uint64_t)SUB_COLS_BF16 * 2u };
    uint32_t bd[4] = { (uint32_t)SUB_COLS_BF16, 1u, (uint32_t)M_TILE, (uint32_t)Q_SUBTILES };
    uint32_t es[4] = { 1u, 1u, 1u, 1u };
    CUresult r = cuTensorMapEncodeTiled(&tq_, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, dQ, gd, gs, bd, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUDA_CHECK(r == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue);
  }
  // O: 4D [hd-subcol, H, tq, hd B128-unit] (same as Q); box [SUB_COLS_BF16, 1, M_TILE, Q_SUBTILES] folds
  //    both hd B128-units into ONE store TMA (hd B128-unit slowest -> matches per-subtile Q_SUB_COLS_BYTES stride).
  {
    uint64_t gd[4] = { (uint64_t)SUB_COLS_BF16, (uint64_t)H, (uint64_t)tq, (uint64_t)Q_SUBTILES };
    uint64_t gs[3] = { (uint64_t)hd * 2u, (uint64_t)((long)H * hd) * 2u, (uint64_t)SUB_COLS_BF16 * 2u };
    uint32_t bd[4] = { (uint32_t)SUB_COLS_BF16, 1u, (uint32_t)M_TILE, (uint32_t)Q_SUBTILES };
    uint32_t es[4] = { 1u, 1u, 1u, 1u };
    CUresult r = cuTensorMapEncodeTiled(&to_, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, dO, gd, gs, bd, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUDA_CHECK(r == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue);
  }
  // K: dims [B128-unit-col SUB_COLS_BF16, token tq, B128-unit (H*hd)/SUB_COLS_BF16]; each gathered block
  // lands in its token-half of the K tile (box per blk64/blk128 below).
  {
    uint64_t gd[3] = { (uint64_t)SUB_COLS_BF16, (uint64_t)tq, (uint64_t)((long)H * hd / SUB_COLS_BF16) };
    uint64_t gs[2] = { (uint64_t)((long)H * hd) * 2u, (uint64_t)SUB_COLS_BF16 * 2u };
    // blk64 box = one (block x hd B128-unit) [SUB_COLS_BF16, BLOCK, 1]; blk128 box folds both hd B128-units of
    // the single block per K tile [SUB_COLS_BF16, BLOCK, K_SUBTILES] (one TMA per tile).
    uint32_t bd[3] = { (uint32_t)SUB_COLS_BF16, (uint32_t)BLOCK, BLK128 ? (uint32_t)K_SUBTILES : 1u };
    uint32_t es[3] = { 1u, 1u, 1u };
    CUresult r = cuTensorMapEncodeTiled(&tk_, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, dK, gd, gs, bd, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUDA_CHECK(r == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue);
  }
  // V_T: [H*hd rows, tq cols] (token contiguous).
  //   blk64:  2D map, box [hd rows, SUB_COLS_BF16 tok cols] = one 64-token block (one TMA per block).
  //   blk128: 3D map [within-unit-tok, H*hd, token B128-unit]; box [SUB_COLS_BF16, hd, V_SUBTILES] folds the
  //           two 64-token B128-units of the single 128-token block into ONE TMA.
  if constexpr (BLK128) {
    uint64_t gd[3] = { (uint64_t)SUB_COLS_BF16, (uint64_t)((long)H * hd), (uint64_t)((long)tq / SUB_COLS_BF16) };
    uint64_t gs[2] = { (uint64_t)tq * 2u, (uint64_t)SUB_COLS_BF16 * 2u };
    uint32_t bd[3] = { (uint32_t)SUB_COLS_BF16, (uint32_t)hd, (uint32_t)V_SUBTILES };
    uint32_t es[3] = { 1u, 1u, 1u };
    CUresult r = cuTensorMapEncodeTiled(&tvt_, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, dVT, gd, gs, bd, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUDA_CHECK(r == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue);
  } else {
    CUDA_CHECK(make_tma_2d_tiled(&tvt_, dVT, (long)H * hd, (int)tq, hd, SUB_COLS_BF16, 2,
                                 CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B));
  }

  // ---- shared memory budget ----
  const size_t smem =
        (size_t)2 * Q_TILE_BYTES + NUM_KV_STAGES * KV_RING_SLOT_BYTES     // Q (x2) + K/V slot ring
      + (size_t)2 * M_TILE * HEAD_DIM * sizeof(__nv_bfloat16)     // 2 sO bufs for TMA-O (also blk64 half-merge scratch)
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

  const unsigned long long magic0 = make_magic((unsigned)(H * packed_mtiles_per_seq));
  const unsigned long long magic1 = make_magic((unsigned)H);
  const unsigned long long magic2 = make_magic((unsigned)packed_mtiles_per_seq);

  const float scale_log2 = (1.0f / sqrtf((float)hd)) * (float)M_LOG2E;
  int numSM = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, 0));
  const int num_ctas = USE_CLC ? total_work : std::min(total_work, numSM);
  dim3 grid(num_ctas, 1, 1), block(N_WARPS * 32, 1, 1);

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
                                B, num_blocks, packed_mtiles_per_seq, max_kv, magic0, magic1, magic2, (const int*)dq2k_idx, (const int*)dq2k_num);
    kfn<<<grid, block, smem, stream>>>(tq_, tk_, tvt_, to_, S, H, scale_log2,
                               B, num_blocks, packed_mtiles_per_seq, max_kv, magic0, magic1, magic2, (const int*)dq2k_idx, (const int*)dq2k_num);
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

  // FLOPs over SELECTED blocks only: (256-blocks) * 256 q-rows * (topk * 256 keys) * hd * 4.
  const double pairs = (double)B * H * num_blocks * SPARSE_BLOCK * (double)topk * SPARSE_BLOCK;
  const double tflops = block_sparse_bf16_benchmark::tflops(pairs, hd, ms);
  if (block_sparse_bf16_benchmark::benchmark_enabled()) printf("  [%-9s H%-2d num_blocks%-3d k%-3d S%d] N_q=%ld topk=%d  %.4f ms  %.1f TFLOPS (selected)\n",
         sh.lab, H, num_blocks, topk, S, tq, topk, ms, tflops);

  block_sparse_bf16_benchmark::dump_output(dO, hQ.size());

  if (verify) {
    std::vector<__nv_bfloat16> ho(hQ.size());
    CUDA_CHECK(cudaMemcpy(ho.data(), dO, ho.size() * 2, cudaMemcpyDeviceToHost));
    std::vector<float> ref(hQ.size(), 0.f), out(hQ.size());
    cpu_vsa_ref(hQ.data(), hK.data(), hV.data(), ref.data(), B, H, S, hd, num_blocks, max_kv,
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
  cudaFree(dq2k_idx); cudaFree(dq2k_num);
  return tflops;
}

int main() {
  CUDA_CHECK(cudaFree(0));
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 64 * 1024 * 1024));
  printf("VSA fine block-sparse FMHA bf16 (blk256: %d sparse block = 2x%d K-tiles, shared per 256-block) sm_100a\n"
         "=====================================\n", SPARSE_BLOCK, K_TILE);

  // shapes: {B, H, num_blocks, topk, hd, label}; num_blocks counts 256-token sparse blocks
  // (one work-item each, so no evenness requirement -- see run()).
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
    if (getenv("NB"))    sh.num_blocks   = atoi(getenv("NB"));
    if (getenv("TOPK"))  sh.topk = atoi(getenv("TOPK"));
    sh.lab = "custom";
    run(sh, verify);
    return 0;
  }
  const int B = getenv("BATCH") ? atoi(getenv("BATCH")) : 1;
  for (Sh sh : shapes) { sh.B = B; run(sh, verify); }
  return 0;
}
