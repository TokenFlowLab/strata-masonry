// block_sparse_bwd_bf16_blk128_2sm.cu -- two-CTA (cta_group::2, 2-SM cluster) form of
// blk128: VSA block-sparse BACKWARD, bf16, sm_100a, 128-token blocks, KV-stationary, FA4-form
// (flash_bwd_sm100.py @ 0251105, 2cta, hdim128).
//
// A cluster pair owns one kv128 block and walks its q-list, one q128 block per step; CTA rank r
// holds kv rows [64r, +64) of the block and q rows [64r, +64) of every visited q block. Per step
// the pair recomputes P^T, forms dS^T, accumulates dK/dV in TMEM and reduce-adds its dQ half tile
// into a global fp32 accumulator; a postprocess unscrambles it, scales and stores bf16 dQ. Tile ==
// block: no q padding, union or kv masking. Grid = (2 * kv blocks, B*H), cluster (2, 1, 1): pair
// (2k, 2k+1) owns kv block k, rank 0 leads. Activations [B, S, H, 128] (VSA_BHSD: [B, H, S, 128]).
//
// Structure:
//   1. Non-persistent: one (kv block, b*h) per pair, no work-item loop, exit after the dK/dV store.
//      16 warps: w0-3 epilogue, w4-11 softmax, w12 mma, w13 load, w14 dS^T relay, w15 donor.
//   2. Operands. Each CTA TMA-loads (cta_group::2) its own 64-row halves; K, V (16 KB each) and
//      K^T (all 128 kv rows x own hd half, 16 KB) ride step 0's Q and dO loads; Q is double-
//      buffered (2 x 16 KB), dO single. Every completion lands on the LEADER's full barrier, armed
//      by the leader for both CTAs' bytes. LSE rides the Q stage index, Delta the dO stage (local).
//   3. Layout B. A cta_group::2 atom folds a [64 x 128] per-CTA accumulator into 128 lanes x 64
//      columns: lane l holds row l & 63, columns [64 * (l >> 6), +64). A TS operand in TMEM
//      follows the same fold: lanes 0-63 feed the n < 64 half of D, lanes 64-127 the n >= 64
//      half, each lane holding all k of an atom.
//   4. dS^T exchange. dQ = dS @ K contracts over all 128 kv rows: a softmax warp writes its own
//      q-half piece into sDST and the peer's into sXCHG, which one thread bulk-copies (8 KB) into
//      the peer's sDST, arming its dsx_full; the relay forwards that to the leader's dsx_leader.
//
// Data layout (D = 128, BLOCK = 128; per CTA unless stated):
//   name       where dtype shape                                   written by    -> read by
//   Q,K,V,dO,O gmem  bf16  [B, S, H, 128] (VSA_BHSD: [B, H, S, 128]) caller      -> TMA / pre
//   LSE, Delta gmem  fp32  [B*H, S] (log2 form)                    fwd / pre     -> load (bulk)
//   K, V       SMEM  bf16  2 hd-halves x [64 kv x 64 hd]           TMA (once)    -> GEMM 1, 2
//   K^T        SMEM  bf16  [128 kv x 64 hd], own hd half           TMA (once)    -> GEMM 4
//   Q[2], dO   SMEM  bf16  2 hd-halves x [64 q x 64 hd]            TMA per step  -> GEMM 1,5 / 2,3
//   S^T        TMEM  fp32  128 lanes x 64 (cols 0-63)              GEMM 1        -> softmax
//   P^T        TMEM  bf16  bf16x2 overlay at cols 0-15, 32-47      softmax       -> GEMM 3
//   dP^T       TMEM  fp32  128 lanes x 64 (cols 64-127)            GEMM 2        -> softmax
//   dS^T       TMEM  bf16  bf16x2 overlay at cols 64-79, 96-111    softmax       -> GEMM 5
//   dS^T(sDST) SMEM  bf16  2 kv-halves x [64 kv x 64 own q]        softmax, peer -> GEMM 4
//   dV, dK     TMEM  fp32  2 x 64 cols (hd halves a, b)            GEMM 3, 5     -> softmax
//   dQ         TMEM  fp32  128 lanes x 64 (cols 384-447)           GEMM 4        -> epilogue
//   dqaccum    gmem  fp32  per (b*h, q128 block) 16384 f32, rank halves  epilogue    -> post
//
// The 5 GEMMs per step (cta_group::2 m128n128k16 x 8, leader-issued, commits multicast), in order:
//   1. S^T = K @ Q^T     SS: A = own K rows, B = own Q[stage] rows (K-major); commit full_bar_st.
//   2. dP^T = V @ dO^T   SS, as GEMM 1 with V and dO; commit full_bar_dpt.
//   3. dV += P^T @ dO    TS: A = P^T overlay, B = own dO rows MN-major (tb = 1), hd subtile p x 4
//                        k16 into dV_p; commit empty_bar_do.
//   4. dQ = dS @ K       SS: A = own q rows x 128 kv from sDST (ta = 1), B = K^T (tb = 1); commits
//                        full_bar_dq and empty_bar_dst.
//   5. dK += dS^T @ Q    TS as GEMM 3 with dS^T, Q[stage], into dK_p; commit empty_bar_q[stage].
//   Prologue: S^T(0), dP^T(0), dV(0). Step j: S^T(j+1) | dK(j) | dP^T(j+1) | dQ(j) | dV(j+1);
//   full_bar_dv is committed entering the last step, full_bar_dk after the tail dK. Scaling per
//   FA4: scale_log2 = sm_scale * log2(e) inside the exp2; sm_scale on dK at the store, dQ in post.
//
// Warp roles:
//   load (w13)       Per step: TMA of its Q half into the free stage (+K half and K^T on step 0),
//                    bulk LSE row; its dO half (+V half on step 0), bulk Delta row.
//   mma (w12)        Both CTAs allocate 512 TMEM columns (cta_group::2) and publish them via
//                    bar_sync<10>(416); the leader issues GEMMs 1-5 and the commits; tmem_dealloc
//                    handshake with the peer before the dealloc.
//   softmax (w4-11)  Warp i = w - 4 owns lanes [32 (i & 3), +32) x columns [32 (i >> 2), +32) of
//                    the folded tiles. Per step: P^T = exp2(S^T * scale_log2 - LSE) -> bf16x2 over
//                    its own columns; dS^T = P^T * (dP^T - Delta) packed the same way, written to
//                    sDST / sXCHG, shipped by one w4 lane after bar_sync<14>(256). Tile end: dV,
//                    then dK (x sm_scale): lanes 64-127 publish fp32 partials into the dead sK, sV;
//                    lanes 0-63 add, pack and stage bf16 (sDO / Q stage 0); one w4 lane TMA-stores.
//   epilogue (w0-3)  Per step: warp w reads lanes [32w, +32) of the dQ tile (q row (w & 1) * 32 +
//                    lane, hd half w >> 1) as two x32 loads, arrives the leader's empty_bar_dq,
//                    then stages 2 rounds x 2 chunks of 32 hd columns (8 KB) through 4 stage
//                    buffers; one w0 lane issues one cp.reduce.async.bulk per chunk.
//   relay (w14)      Per step: local dsx_full -> one arrive on the leader's dsx_leader.
//   donor (w15)      No work; donates its registers.
//
// Barrier contract. arv = arrival count as initialised; copy = whose mbarrier the consumer waits.
//   barrier          ring  arv  producer -> consumer  copy    meaning
//   ---------------  ----  ---  --------------------  ------  ------------------------------------
//   full_bar_q        [2]   1   load     -> mma       leader  Q halves; tx 32 KB, 96 KB step 0.
//   empty_bar_q       [2]   1   mma      -> load      local   Q stage read: commit after GEMM 5.
//   full_bar_do       [1]   1   load     -> mma       leader  dO halves; tx 32 KB, 64 KB step 0.
//   empty_bar_do      [1]   1   mma      -> load      local   dO read: commit after GEMM 3.
//   full_bar_lse      [2]   1   load     -> softmax   local   step's 128 LSE values; tx 512 B.
//   empty_bar_lse     [2]   8   softmax  -> load      local   LSE consumed (one lane per warp).
//   full_bar_delta    [1]   1   load     -> softmax   local   step's 128 Delta values; tx 512 B.
//   empty_bar_delta   [1]   8   softmax  -> load      local   Delta consumed.
//   full_bar_st       [1]   1   mma      -> softmax   local   S^T in TMEM: commit after GEMM 1.
//   full_bar_dpt      [1]   1   mma      -> softmax   local   dP^T in TMEM: commit after GEMM 2.
//   full_bar_pt       [1]  16   softmax  -> mma       leader  P^T overlays stored, both CTAs.
//   full_bar_dst      [1]  16   softmax  -> mma       leader  dS^T overlays + sDST written, both.
//   full_bar_dq       [1]   1   mma      -> epilogue  local   dQ tile in TMEM: commit after GEMM 4.
//   empty_bar_dq      [1]   8   epilogue -> mma       leader  dQ read out, both CTAs' 4 warps.
//   full_bar_dv       [1]   1   mma      -> softmax   local   dV complete: commit at the last step.
//   full_bar_dk       [1]   1   mma      -> softmax   local   dK complete: commit after tail dK.
//   dsx_full          [1]   1   peer w4  -> relay     local   peer's dS^T half landed; tx 8 KB.
//   dsx_leader        [1]   2   relays   -> mma       leader  both halves exchanged; gates dQ(j).
//   empty_bar_dst     [1]   1   mma      -> softmax   local   both sDST read: commit after GEMM 4.
//   tmem_dealloc      [1]  32   peer mma -> mma       local   peer's MMA warp done; dealloc may go.
//   full_* waits are PhaseTrackers; the load warp's empty_* and the softmax warps' empty_bar_dst
//   waits EmptyPhaseTrackers (seeded to parity 1); the MMA's empty_bar_dq is first waited at dQ(1).
//
// Budgets:
//   TMEM (one cta_group::2 allocation of 512 columns): 0-63 S^T (P^T overlay 0-15, 32-47) | 64-127
//     dP^T (dS^T overlay 64-79, 96-111) | 128-255 dV_a, dV_b | 256-383 dK_a, dK_b | 384-447 dQ.
//   SMEM (~154 KB): K 16K | V 16K | K^T 16K | Q[2] 32K | dO 16K | sDST 16K | sXCHG 8K | dQ stage
//     4 x 8K | LSE 2 x 512 B | Delta 512 B | 24 mbarriers | tmem_slot. The dV / dK stores bounce
//     through the then-free sDO and Q stage 0, their fp32 merge through sK + sV.
//   Registers (setmaxnreg from the 128/thread base; 4*152 + 8*136 + 3*88 + 24 = 1984 <= 2048):
//     w0-3 152 | w4-11 136 | w12-14 88 | w15 24.
//
// Index contract (FastVideo's padded k2q form, the forward's q2k_idx/q2k_num transposed): k2q_idx
// int32 [B*H*nb, max_q_blocks] holds per (b*h, kv128) row the LOCAL q128 ids that select it, in
// entries [0, k2q_num[row]). Timed = pre + main + post; TFLOPS = 2.5*4*D*(B*H*nb*topk*128^2) / t.
//
// Env knobs: LOAD_NPY=<dir> (q/k/v/do_S{S}.npy, idx_S{S}_blk128.npy), SHAPE=0..5 + BATCH/HEADS/NB/
// TOPK, CPU_REF=0|1, DUMP_BWD=<prefix> (fp32 npy of the reference dq/dk/dv [B*S, H, D], M/delta
// [B*H, S]), DUMP_BWD_GPU=<prefix>, VERIFY_ARGMAX, STRESS_N, BENCH_WARMUP, BENCH_ITERS.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <cmath>
#include <chrono>
#include <climits>
#include <vector>
#include <algorithm>
#include <string>
#include "../npy_io.cuh"
#include "block_sparse_bwd_bf16_benchmark.cuh"
#include "../../../../tests/test_utils.cuh"
#include "../../../../primitives/0_tcgen05_alloc.cuh"
#include "../../../../primitives/1_tcgen05_dealloc.cuh"
#include "../../../../primitives/2_tcgen05_relinquish.cuh"
#include "../../../../primitives/8_tcgen05_mma_idesc.cuh"
#include "../../../../primitives/9_tcgen05_ld.cuh"
#include "../../../../primitives/10_tcgen05_st.cuh"
#include "../../../../primitives/12_tcgen05_wait.cuh"
#include "../../../../primitives/15_tcgen05_fence.cuh"
#include "../../../../primitives/18_tma_load.cuh"
#include "../../../../primitives/19_tma_load_2sm.cuh"
#include "../../../../primitives/22_tma_store.cuh"
#include "../../../../primitives/25_tma_async_group.cuh"
#include "../../../../primitives/27_cp_async_cg.cuh"
#include "../../../../primitives/28_cp_async_commit_wait.cuh"
#include "../../../../primitives/29_mbarrier_init.cuh"
#include "../../../../primitives/30_mbarrier_arrive.cuh"
#include "../../../../primitives/31_mbarrier_arrive_tx.cuh"
#include "../../../../primitives/33_mbarrier_try_wait.cuh"
#include "../../../../primitives/34_fence_proxy_async.cuh"
#include "../../../../primitives/35_fence_mbarrier_init.cuh"
#include "../../../../primitives/37_bar_sync.cuh"
#include "../../../../primitives/38_barrier_cluster.cuh"
#include "../../../../primitives/42_smem_desc_blackwell.cuh"
#include "../../../../primitives/44_elect_sync.cuh"
#include "../../../../primitives/46_setmaxnreg.cuh"
#include "../../../../primitives/63_cvt_f32_to_f16_bf16.cuh"
#include "../../../../primitives/67_mapa.cuh"
#include "../../../../primitives/69_griddepcontrol.cuh"
#include "../../../../primitives/76_packed_f32x2.cuh"
#include "../../../../primitives/77_ex2_approx.cuh"
#include "../../../../composites/118_mbarrier_phase_tracking.cuh"
#include "../../../../primitives/_warp_prof_noop.cuh"

#ifndef VSA_BHSD
#define VSA_BHSD false  // false: [B, S, H, 128]; true: FastVideo's [B, H, S, 128]
#endif
// Programmatic dependent launch (KERNEL_PDL): preprocess -> main -> postprocess each start while
// the predecessor drains; griddepcontrol.wait sits in front of every read of the predecessor's data.
#ifndef KERNEL_PDL
#define KERNEL_PDL true
#endif

constexpr int BLOCK           = 128;         // q and kv block size (tokens); tile == block
constexpr int KV_TILE         = BLOCK;       // kv rows per kv block (a CTA pair)
constexpr int Q_TILE          = BLOCK;       // q rows per step
constexpr int HEAD_DIM        = 128;
constexpr int SUB_COLS_BF16   = 64;  // one 128B-swizzle unit
constexpr int SUB_COLS_BYTES  = SUB_COLS_BF16 * (int)sizeof(__nv_bfloat16);  // 128 B
constexpr int HD_SUBTILES     = HEAD_DIM / SUB_COLS_BF16;  // 2 hd-halves of a K, V, Q or dO tile
constexpr int KV_SUB_COLS_BYTES = KV_TILE * SUB_COLS_BYTES;  // 16 KB: K or V, one 64-hd subtile
constexpr int NUM_Q_STAGES      = 2;                         // Q double-buffered; dO single
constexpr int MMA_K               = 16;                      // bf16 tcgen05.mma K per issue
constexpr int K_ATOMS_PER_SUBTILE = SUB_COLS_BF16 / MMA_K;   // 4: GEMM 1/2, hd within a subtile
constexpr int K_ATOMS_PER_Q_HALF  = SUB_COLS_BF16 / MMA_K;   // 4: the 64 q of one q-half
constexpr int K_ATOMS_PER_KV_TILE = KV_TILE / MMA_K;         // 8: GEMM 4, the 128 kv rows
constexpr int BF16X2_COLS_PER_K16 = MMA_K / 2;               // TMEM columns per k16, bf16x2 tile

// dQ drain: the dQ tile leaves TMEM whole, then goes to dqaccum in chunks of DQ_CHUNK_COLS hd
// columns, each staged in an SMEM buffer and pushed as one cp.reduce.async.bulk.
constexpr int DQ_CHUNK_COLS     = 32;
constexpr int DQ_CHUNKS         = HEAD_DIM / DQ_CHUNK_COLS;                  // 4 per tile
constexpr int DQ_BLOCK_ELEMS    = Q_TILE * HEAD_DIM;  // accumulator elems per q128 block

constexpr int N_WARPS = 16;
constexpr int W_EPI0 = 0, W_SOFTMAX0 = 4, W_MMA = 12, W_LOAD = 13;
constexpr int W_RELAY = 14;   // the dS^T relay warp

// Per CTA: 64 kv rows, 64 q rows of each q block; layout B folds a [64 x 128] accumulator into
// 128 lanes x 64 columns (lane l = row l & 63, columns [64 * (l >> 6), +64)); header, Structure 3.
constexpr int KV_ROWS_2CTA        = KV_TILE / 2;                      // 64 kv rows per CTA
constexpr int Q_ROWS_2CTA         = Q_TILE / 2;                       // 64 q rows per CTA
constexpr int HALF_SUB_BYTES      = KV_ROWS_2CTA * SUB_COLS_BYTES;    // 8 KB: [64 rows x 64 hd]
constexpr int HALF_TILE_BYTES     = HD_SUBTILES * HALF_SUB_BYTES;     // 16 KB: [64 rows x 128 hd]
constexpr int KT_TILE_BYTES       = KV_SUB_COLS_BYTES;                // 16 KB: [128 kv x 64 hd], own hd half
constexpr int DST_REGION_BYTES    = KV_ROWS_2CTA * SUB_COLS_BYTES;    // 8 KB: dS^T [64 kv x 64 q]
constexpr int DST_BYTES_2CTA      = 2 * DST_REGION_BYTES;             // 16 KB: kv-half 0, then 1
constexpr int DQ_CHUNK_BYTES_2CTA = Q_ROWS_2CTA * DQ_CHUNK_COLS * (int)sizeof(float);  // 8 KB
constexpr int DQ_STAGE_BUFFERS_2CTA = 4;
constexpr int DQ_RANK_ELEMS       = DQ_BLOCK_ELEMS / 2;               // one CTA's half of a block
constexpr int NUM_BARS_2CTA       = 4 * NUM_Q_STAGES + 12 + 4;  // + dsx_full, dsx_leader, tmem_dealloc, empty_bar_dst
constexpr int SMEM_TOTAL_2CTA = 3 * HALF_TILE_BYTES + NUM_Q_STAGES * HALF_TILE_BYTES + HALF_TILE_BYTES +
                                DST_BYTES_2CTA + DST_REGION_BYTES +
                                DQ_STAGE_BUFFERS_2CTA * DQ_CHUNK_BYTES_2CTA +
                                (NUM_Q_STAGES + 1) * Q_TILE * (int)sizeof(float) + NUM_BARS_2CTA * 8 +
                                16;  // align + tmem_slot
// TMEM (2CTA): S^T | dP^T | dV_a dV_b | dK_a dK_b | dQ, 64 columns each. Accumulator pair X_a / X_b
// = hd [0, 64) / [64, 128); the two lane halves of each are the two q-half partials of its rows.
constexpr int COLS_2CTA        = 64;
constexpr int TMEM_ST_2CTA     = 0;
constexpr int TMEM_DPT_2CTA    = TMEM_ST_2CTA + COLS_2CTA;
constexpr int TMEM_DV_2CTA     = TMEM_DPT_2CTA + COLS_2CTA;
constexpr int TMEM_DK_2CTA     = TMEM_DV_2CTA + 2 * COLS_2CTA;
constexpr int TMEM_DQ_2CTA     = TMEM_DK_2CTA + 2 * COLS_2CTA;
constexpr int TMEM_ALLOC_2CTA  = 512;
static_assert(TMEM_DQ_2CTA + COLS_2CTA <= TMEM_ALLOC_2CTA, "2CTA TMEM map exceeds the allocation");

extern __shared__ __align__(1024) uint8_t bwd_smem[];

// Lead-lane-predicated cta_group::2 forms of primitives 3 / 11: issued by the leader CTA's
// elected MMA lane, D and A in TMEM, the commit multicast to both CTAs' barriers.
__device__ __forceinline__ void tcgen05_mma_f16_ss_2sm_lead(uint32_t lead, uint32_t tmem_c,
                                                            uint64_t desc_a, uint64_t desc_b,
                                                            uint32_t idesc, bool enable_input_d) {
  asm volatile(
      "{\n\t"
      ".reg .pred p, q;\n\t"
      "setp.ne.b32 q, %0, 0;\n\t"
      "setp.ne.b32 p, %5, 0;\n\t"
      "@q tcgen05.mma.cta_group::2.kind::f16 [%1], %2, %3, %4, {%6, %6, %6, %6, %6, %6, %6, %6}, p;\n\t"
      "}\n"
      :: "r"(lead), "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc),
         "r"(enable_input_d ? 1u : 0u), "r"(0u));
}
__device__ __forceinline__ void tcgen05_mma_f16_ts_2sm_lead(uint32_t lead, uint32_t tmem_c,
                                                            uint32_t tmem_a, uint64_t desc_b,
                                                            uint32_t idesc, bool enable_input_d) {
  asm volatile(
      "{\n\t"
      ".reg .pred p, q;\n\t"
      "setp.ne.b32 q, %0, 0;\n\t"
      "setp.ne.b32 p, %5, 0;\n\t"
      "@q tcgen05.mma.cta_group::2.kind::f16 [%1], [%2], %3, %4, {%6, %6, %6, %6, %6, %6, %6, %6}, p;\n\t"
      "}\n"
      :: "r"(lead), "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(idesc),
         "r"(enable_input_d ? 1u : 0u), "r"(0u));
}
__device__ __forceinline__ void tcgen05_commit_multicast2_lead(uint32_t lead, uint32_t mbar_smem) {
  asm volatile(
      "{\n\t"
      ".reg .pred q;\n\t"
      "setp.ne.b32 q, %0, 0;\n\t"
      "@q tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 "
      "[%1], %2;\n\t"
      "}\n"
      :: "r"(lead), "r"(mbar_smem), "h"((uint16_t)3) : "memory");
}

// Element offset of token t of (batch, head) in the activation layout: BHSD = [B, H, S, 128],
// else [B, S, H, 128].
template <bool BHSD>
__device__ __forceinline__ size_t token_offset(int batch, int head, int num_heads, int seqlen,
                                               int t) {
  if constexpr (BHSD)
    return ((size_t)(batch * num_heads + head) * seqlen + t) * HEAD_DIM;
  else
    return ((size_t)(batch * seqlen + t) * num_heads + head) * HEAD_DIM;
}

struct WorkItem {
  int batch, head, kv_block_id_in_seq, batch_head;
  const int* local_k2q_idx;
  int local_k2q_num;
};

// Item (batch*head, kv block) -> its q-list row. Item row = batch_head * nb + kv block (size_t
// offset: with max_q_blocks == nb, B*H*nb*nb can exceed 2^31).
__device__ __forceinline__ WorkItem decode_workitem(int batch_head, int kv_block_id_in_seq,
                                                  const int* __restrict__ k2q_idx,
                                                  const int* __restrict__ k2q_num,
                                                  int max_q_blocks, int num_heads,
                                                  int num_kv_blocks_per_seq) {
  WorkItem it;
  it.batch_head         = batch_head;
  it.batch              = batch_head / num_heads;
  it.head               = batch_head % num_heads;
  it.kv_block_id_in_seq = kv_block_id_in_seq;
  const int item        = batch_head * num_kv_blocks_per_seq + kv_block_id_in_seq;
  it.local_k2q_idx      = k2q_idx + (size_t)item * (size_t)max_q_blocks;
  it.local_k2q_num      = k2q_num[item];
  return it;
}

// Compile-time kernel config:
//   BHSD    : activation layout of q/k/v/o/dO and dq/dk/dv: false = [B, S, H, 128], true =
//             FastVideo's [B, H, S, 128] (K/V/Q/dO/dK/dV as 4D tensor maps, token offsets through
//             token_offset<BHSD>). Fixed per build by VSA_BHSD.
// Kernel arguments (activations bf16 in the BHSD-selected layout unless stated):
//   tmap_k             : K as a 3D map [64 hd, B*S tokens, H*2 hd units], box (64, 128, 1) (BHSD:
//                        4D [64 hd, S, 2 hd units, B*H], box (64, 128, 1, 1)): the K^T tile.
//   tmap_*64           : Q, K, V, dO, dK, dV, same geometry with a 64-token box (a CTA's half).
//   dqaccum            : fp32 [B*H, nb, 128*128] drain-native dQ accumulator; zeroed by the
//                        preprocess, reduce-added here, unscrambled by the postprocess.
//   lse_rows           : fp32 [B*H, S], the forward's LSE in log2 form (M = max + log2(l)).
//   delta_rows         : fp32 [B*H, S], Delta = rowsum(bf16(O) * dO) from the preprocess.
//   k2q_idx, k2q_num   : padded k2q lists (Index contract above): row item = (b*H + h)*nb + kv
//                        holds k2q_num[item] q128 block ids at k2q_idx[item * max_q_blocks + i].
//   max_q_blocks       : k2q_idx row stride.
//   num_samples, num_heads, seqlen : B, H, S; nb = num_kv_blocks_per_seq = S/128 is derived.
//   scale_log2         : sm_scale * log2(e), applied to S^T before exp2.
//   sm_scale           : applied to dK in the tile-end store (dQ gets it in the postprocess).
template <bool BHSD = false>
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(N_WARPS * 32, 1)
    vsa_bwd_main_kernel(const __grid_constant__ CUtensorMap tmap_k,
                        const __grid_constant__ CUtensorMap tmap_q64,
                        const __grid_constant__ CUtensorMap tmap_k64,
                        const __grid_constant__ CUtensorMap tmap_v64,
                        const __grid_constant__ CUtensorMap tmap_do64,
                        const __grid_constant__ CUtensorMap tmap_dk64,
                        const __grid_constant__ CUtensorMap tmap_dv64, float* __restrict__ dqaccum,
                        const float* __restrict__ lse_rows, const float* __restrict__ delta_rows,
                        const int* __restrict__ k2q_idx, const int* __restrict__ k2q_num,
                        int max_q_blocks, int num_samples, int num_heads, int seqlen,
                        float scale_log2, float sm_scale) {
  const int num_kv_blocks_per_seq = seqlen / BLOCK;
  // SMEM carve: K, V, Q stages, dO are this CTA's 64-row halves (16 KB); K^T (128 kv x own hd half)
  // feeds the dQ GEMM; sDST holds dS^T [kv-half 0 | kv-half 1] x own q rows; sXCHG the peer's half.
  constexpr int K_BYTES        = HALF_TILE_BYTES;
  constexpr int KT_BYTES       = KT_TILE_BYTES;
  constexpr int QO_BYTES       = HALF_TILE_BYTES;
  constexpr int DST_TOTAL      = DST_BYTES_2CTA;
  constexpr int XCHG_BYTES     = DST_REGION_BYTES;
  constexpr int DQ_STAGE_TOTAL = DQ_STAGE_BUFFERS_2CTA * DQ_CHUNK_BYTES_2CTA;
  uint8_t* sK                = bwd_smem;
  uint8_t* sV                = sK + K_BYTES;
  uint8_t* sKT               = sV + K_BYTES;
  uint8_t* sQ[NUM_Q_STAGES]  = {sKT + KT_BYTES, sKT + KT_BYTES + QO_BYTES};
  uint8_t* sDO               = sQ[0] + NUM_Q_STAGES * QO_BYTES;
  __nv_bfloat16* sDST        = reinterpret_cast<__nv_bfloat16*>(sDO + QO_BYTES);
  uint8_t* sXCHG             = sDO + QO_BYTES + DST_TOTAL;
  uint8_t* sDQ_STAGE_bytes   = sXCHG + XCHG_BYTES;
  float* sLSE   = reinterpret_cast<float*>(sDQ_STAGE_bytes + DQ_STAGE_TOTAL);
  float* sDelta = sLSE + NUM_Q_STAGES * Q_TILE;  // sLSE: [stage][128]; sDelta: [128]

  // mbarriers (arrival counts at the init below)
  uint64_t* full_bar_q      = reinterpret_cast<uint64_t*>(sDelta + Q_TILE);  // [stage] TMA tx
  uint64_t* empty_bar_q     = full_bar_q + NUM_Q_STAGES;     // [stage] GEMM 5 commit
  uint64_t* full_bar_do     = empty_bar_q + NUM_Q_STAGES;    // TMA tx
  uint64_t* empty_bar_do    = full_bar_do + 1;               // GEMM 3 commit
  uint64_t* full_bar_lse    = empty_bar_do + 1;              // [stage] load -> softmax (128 f32)
  uint64_t* empty_bar_lse   = full_bar_lse + NUM_Q_STAGES;   // [stage] lane 0 of each softmax warp
  uint64_t* full_bar_delta  = empty_bar_lse + NUM_Q_STAGES;  // load -> softmax (128 f32)
  uint64_t* empty_bar_delta = full_bar_delta + 1;            // lane 0 of each softmax warp
  uint64_t* full_bar_st     = empty_bar_delta + 1;           // commit after the S^T atoms
  uint64_t* full_bar_dpt    = full_bar_st + 1;               // commit after the dP^T atoms
  uint64_t* full_bar_pt     = full_bar_dpt + 1;              // P^T overlay stored (gates dV)
  uint64_t* full_bar_dst    = full_bar_pt + 1;               // dS^T overlay + sDST written
  uint64_t* full_bar_dq     = full_bar_dst + 1;              // commit after the dQ GEMM
  uint64_t* empty_bar_dq    = full_bar_dq + 1;               // epilogue warps read the dQ tile
  uint64_t* full_bar_dv     = empty_bar_dq + 1;              // all dV accumulation issued
  uint64_t* full_bar_dk     = full_bar_dv + 1;               // last dK issued
  // dsx_full: the peer's dS^T half landed in my sDST (1 arrive + 8 KB tx, per step); dsx_leader: both
  // relays certified the exchange (leader CTA, 2); tmem_dealloc: the peer's MMA warp is done (32).
  uint64_t* dsx_full        = full_bar_dk + 1;
  uint64_t* dsx_leader      = dsx_full + 1;
  uint64_t* tmem_dealloc    = dsx_leader + 1;
  uint64_t* empty_bar_dst   = tmem_dealloc + 1;  // dQ(j) has read both CTAs' sDST (multicast commit)
  uint32_t* tmem_slot       = reinterpret_cast<uint32_t*>(
      (reinterpret_cast<uintptr_t>(empty_bar_dst + 1) + 15u) & ~uintptr_t(15u));

  const int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;
  // The pair is the two x-adjacent CTAs (cluster (2, 1, 1)); rank 0 issues the MMAs.
  const int cta_rank = (int)blockIdx.x & 1;
  const bool is_leader = cta_rank == 0;
  // Cluster-window addresses of a barrier in the leader / in this CTA (cta_group::2 TMA and remote
  // arrives take cluster addresses).
  auto leader_addr = [&](const uint64_t* bar) {
    return mapa_shared_cluster_u32(smem_ptr_u32(bar), 0u);
  };
  auto own_cluster_addr = [&](const void* smem) {
    return mapa_shared_cluster_u32(smem_ptr_u32(smem), (uint32_t)cta_rank);
  };

  if (tid == 0) {
    #pragma unroll
    for (int s = 0; s < NUM_Q_STAGES; ++s) {
      mbarrier_init(smem_ptr_u32(&full_bar_q[s]), 1);
      mbarrier_init(smem_ptr_u32(&empty_bar_q[s]), 1);
    }
    mbarrier_init(smem_ptr_u32(full_bar_do), 1);
    mbarrier_init(smem_ptr_u32(empty_bar_do), 1);
    #pragma unroll
    for (int s = 0; s < NUM_Q_STAGES; ++s) {
      mbarrier_init(smem_ptr_u32(&full_bar_lse[s]), 1);
      mbarrier_init(smem_ptr_u32(&empty_bar_lse[s]), 8);
    }
    mbarrier_init(smem_ptr_u32(full_bar_delta), 1);
    mbarrier_init(smem_ptr_u32(empty_bar_delta), 8);
    mbarrier_init(smem_ptr_u32(full_bar_st), 1);
    mbarrier_init(smem_ptr_u32(full_bar_dpt), 1);
    // The softmax / epilogue warps of BOTH CTAs arrive on the leader's copy.
    mbarrier_init(smem_ptr_u32(full_bar_pt), 16);
    mbarrier_init(smem_ptr_u32(full_bar_dst), 16);
    mbarrier_init(smem_ptr_u32(full_bar_dq), 1);
    mbarrier_init(smem_ptr_u32(empty_bar_dq), 8);
    mbarrier_init(smem_ptr_u32(full_bar_dv), 1);
    mbarrier_init(smem_ptr_u32(full_bar_dk), 1);
    mbarrier_init(smem_ptr_u32(dsx_full), 1);
    mbarrier_init(smem_ptr_u32(dsx_leader), 2);
    mbarrier_init(smem_ptr_u32(tmem_dealloc), 32);
    mbarrier_init(smem_ptr_u32(empty_bar_dst), 1);
  }
  fence_mbarrier_init_release_cluster();
  __syncthreads();
  // The peer's barriers must be initialised before any remote arrive or cta_group::2 TMA.
  barrier_cluster_arrive();
  barrier_cluster_wait();

  // The 2D grid is the item list: the pair shares the item of kv block blockIdx.x >> 1.
  auto current_workitem = [&]() {
    return decode_workitem((int)blockIdx.y, (int)blockIdx.x >> 1, k2q_idx, k2q_num, max_q_blocks,
                         num_heads, num_kv_blocks_per_seq);
  };

  // Relay warp (w14, FA4 warp 14): per step, "the peer's dS^T half landed in my sDST" (local
  // dsx_full) becomes one arrive on the leader's dsx_leader, which gates that step's dQ GEMM.
  if (warp_id == W_RELAY) {
    setmaxnreg_dec<88>();
    const WorkItem it = current_workitem();
    if (it.local_k2q_num != 0) {
      const uint32_t dsx_leader_addr = leader_addr(dsx_leader);
      for (int j = 0; j < it.local_k2q_num; ++j) {
        mbarrier_wait_parity_suspend(smem_ptr_u32(dsx_full), (uint32_t)(j & 1));
        if (elect_one_sync()) mbarrier_arrive_cluster_default(dsx_leader_addr);
      }
    }
    return;
  }
  if (warp_id > W_RELAY) {
    setmaxnreg_dec<24>();
    return;
  }

  WpCtx wpc = wp_ctx_init();

  if (warp_id == W_LOAD) {
    // Both CTAs issue their own 64-row slices with cta_group::2 TMA; every completion lands on
    // the LEADER's full barrier, which only the leader arms (for both CTAs' bytes). Each CTA waits
    // its LOCAL empty barriers (the leader's multicast commits). LSE / Delta stay CTA-local.
    setmaxnreg_dec<88>();
    if constexpr (KERNEL_PDL) griddepcontrol_wait();  // Delta and the zeroed dqaccum
    const WorkItem it = current_workitem();
    if (it.local_k2q_num != 0) {
      EmptyPhaseTracker<NUM_Q_STAGES> q_empty_ph, lse_empty_ph;
      EmptyPhaseTracker<1> do_empty_ph, delta_empty_ph;
      const int kv_begin = it.kv_block_id_in_seq * KV_TILE;
      // This CTA's 64 rows at token_begin, both hd subtiles, of a 64-token-box map.
      auto load_half_tile = [&](uint8_t* dst, const CUtensorMap* map64, uint64_t* full_bar,
                                int token_begin) {
        #pragma unroll
        for (int s = 0; s < HD_SUBTILES; ++s) {
          if constexpr (BHSD)
            tma_load_4d_2sm(own_cluster_addr(dst + s * HALF_SUB_BYTES), map64, leader_addr(full_bar),
                            0, token_begin, s, it.batch_head);
          else
            tma_load_3d_2sm(own_cluster_addr(dst + s * HALF_SUB_BYTES), map64, leader_addr(full_bar),
                            0, it.batch * seqlen + token_begin, it.head * HD_SUBTILES + s);
        }
      };
      // K^T for the dQ GEMM: all 128 kv rows, this CTA's hd subtile (the 128-token K map).
      auto load_kt = [&](uint64_t* full_bar) {
        if constexpr (BHSD)
          tma_load_4d_2sm(own_cluster_addr(sKT), &tmap_k, leader_addr(full_bar), 0, kv_begin,
                          cta_rank, it.batch_head);
        else
          tma_load_3d_2sm(own_cluster_addr(sKT), &tmap_k, leader_addr(full_bar), 0,
                          it.batch * seqlen + kv_begin, it.head * HD_SUBTILES + cta_rank);
      };
      auto load_row = [&](float* dst, const float* rows, int qblock, uint64_t* full_bar) {
        cpasync_bulk_load_mbarrier(
            smem_ptr_u32(dst),
            rows + (size_t)it.batch_head * seqlen + (size_t)qblock * Q_TILE,
            Q_TILE * sizeof(float), smem_ptr_u32(full_bar));
      };

      for (int j = 0; j < it.local_k2q_num; ++j) {
        wp_marker(wpc, WP_ITER, j);
        const int qblock  = it.local_k2q_idx[j];
        const int q_begin = qblock * Q_TILE + cta_rank * Q_ROWS_2CTA;
        const int stage   = q_empty_ph.get_stage();
        const bool first  = j == 0;

        wp_begin(wpc, WP_LOAD_WAIT);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_q[stage]), q_empty_ph.get_phase());
        q_empty_ph.advance();
        wp_end(wpc, WP_LOAD_WAIT);
        wp_begin(wpc, WP_LOAD_ISSUE_Q);
        if (elect_one_sync()) {
          if (is_leader)
            mbarrier_arrive_expect_tx(
                smem_ptr_u32(&full_bar_q[stage]),
                2 * (HALF_TILE_BYTES + (first ? HALF_TILE_BYTES + KT_TILE_BYTES : 0)));
          load_half_tile(sQ[stage], &tmap_q64, &full_bar_q[stage], q_begin);
          if (first) {
            load_half_tile(sK, &tmap_k64, &full_bar_q[stage], kv_begin + cta_rank * KV_ROWS_2CTA);
            load_kt(&full_bar_q[stage]);
          }
        }
        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_lse[stage]),
                                     lse_empty_ph.get_phase());
        lse_empty_ph.advance();
        if (elect_one_sync()) {
          mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar_lse[stage]),
                                    Q_TILE * (int)sizeof(float));
          load_row(sLSE + stage * Q_TILE, lse_rows, qblock, &full_bar_lse[stage]);
        }
        wp_end(wpc, WP_LOAD_ISSUE_Q);

        wp_begin(wpc, WP_LOAD_WAIT_THROTTLE);
        mbarrier_wait_parity_suspend(smem_ptr_u32(empty_bar_do), do_empty_ph.get_phase());
        do_empty_ph.advance();
        wp_end(wpc, WP_LOAD_WAIT_THROTTLE);
        wp_begin(wpc, WP_LOAD_ISSUE_V);
        if (elect_one_sync()) {
          if (is_leader)
            mbarrier_arrive_expect_tx(smem_ptr_u32(full_bar_do),
                                      2 * (HALF_TILE_BYTES + (first ? HALF_TILE_BYTES : 0)));
          load_half_tile(sDO, &tmap_do64, full_bar_do, q_begin);
          if (first)
            load_half_tile(sV, &tmap_v64, full_bar_do, kv_begin + cta_rank * KV_ROWS_2CTA);
        }
        mbarrier_wait_parity_suspend(smem_ptr_u32(empty_bar_delta), delta_empty_ph.get_phase());
        delta_empty_ph.advance();
        if (elect_one_sync()) {
          mbarrier_arrive_expect_tx(smem_ptr_u32(full_bar_delta), Q_TILE * (int)sizeof(float));
          load_row(sDelta, delta_rows, qblock, full_bar_delta);
        }
        wp_end(wpc, WP_LOAD_ISSUE_V);
      }
    }
    wp_flush(wpc);
    return;
  }
  else if (warp_id == W_MMA) {
    // Both MMA warps allocate (cta_group::2) and publish; only the leader issues. Every commit is
    // multicast to both CTAs' local barriers; every wait is on the leader's local copy.
    setmaxnreg_dec<88>();
    tcgen05_alloc<2>(smem_ptr_u32(tmem_slot), TMEM_ALLOC_2CTA);
    bar_sync<10>(416);
    const uint32_t tmem_base = *tmem_slot;
    // Teardown handshake (FA4): all 32 lanes arrive on the PEER's tmem_dealloc, wait for the
    // peer's 32, then deallocate; neither CTA frees TMEM the pair still uses.
    auto teardown = [&]() {
      tcgen05_relinquish_alloc_permit<2>();
      bar_sync<10>(416);
      mbarrier_arrive_cluster_default(
          mapa_shared_cluster_u32(smem_ptr_u32(tmem_dealloc), (uint32_t)(cta_rank ^ 1)));
      mbarrier_wait_parity(smem_ptr_u32(tmem_dealloc), 0);
      tcgen05_dealloc<2>(tmem_base, TMEM_ALLOC_2CTA);
    };
    const WorkItem it = current_workitem();
    if (it.local_k2q_num == 0 || !is_leader) {
      wp_flush(wpc);
      teardown();
      return;
    }
    const uint32_t tmem_st       = tmem_base + TMEM_ST_2CTA;   // S^T; P^T bf16 overlay
    const uint32_t tmem_dpt      = tmem_base + TMEM_DPT_2CTA;  // dP^T; dS^T overlay
    const uint32_t tmem_dv       = tmem_base + TMEM_DV_2CTA;   // dV_a, dV_b
    const uint32_t tmem_dk       = tmem_base + TMEM_DK_2CTA;   // dK_a, dK_b
    const uint32_t tmem_dq       = tmem_base + TMEM_DQ_2CTA;
    const uint32_t tmem_pt_bf16  = tmem_st, tmem_dst_bf16 = tmem_dpt;
    const uint32_t lead          = elect_one_sync() ? 1u : 0u;

    constexpr uint32_t DESC_SBO = 1024, DESC_LBO = 16;
    auto make_smem_desc = [](const void* smem, uint32_t leading_byte_offset) {
      return build_smem_desc_blackwell(smem_ptr_u32(smem), DESC_SBO, leading_byte_offset,
                                       SmemSwizzleBlackwell::B128);
    };
    // K-major (GEMMs 1, 2): A = own 64 kv rows of K / V, B = own 64 q rows of Q / dO. MN-major
    // (GEMMs 3, 4, 5): one 64-column subtile per CTA, so the leading offset is unused.
    const uint64_t desc_k      = make_smem_desc(sK, DESC_LBO);
    const uint64_t desc_v      = make_smem_desc(sV, DESC_LBO);
    const uint64_t desc_q0     = make_smem_desc(sQ[0], DESC_LBO);
    const uint64_t desc_do     = make_smem_desc(sDO, DESC_LBO);
    const uint64_t desc_q0_mn  = make_smem_desc(sQ[0], HALF_SUB_BYTES);
    const uint64_t desc_do_mn  = make_smem_desc(sDO, HALF_SUB_BYTES);
    const uint64_t desc_kt_mn  = make_smem_desc(sKT, KT_TILE_BYTES);
    const uint64_t desc_dst_mn = make_smem_desc(sDST, DST_REGION_BYTES);
    constexpr uint64_t K16_COLS_DELTA = (MMA_K * (int)sizeof(__nv_bfloat16)) >> 4;
    constexpr uint32_t K16_ROWS_DELTA = (MMA_K * SUB_COLS_BYTES) >> 4;
    constexpr uint64_t HALF_SUB_DELTA = HALF_SUB_BYTES >> 4;
    constexpr uint64_t Q_STAGE_DELTA  = HALF_TILE_BYTES >> 4;
    // idesc M / N are the cluster totals (128 x 128).
    const uint32_t idesc_st_dpt = make_idesc_bf16_f32(KV_TILE, Q_TILE, false, false);
    const uint32_t idesc_dv_dk  = make_idesc_bf16_f32(KV_TILE, HEAD_DIM, false, true);
    const uint32_t idesc_dq     = make_idesc_bf16_f32(Q_TILE, HEAD_DIM, true, true);

    PhaseTracker<NUM_Q_STAGES> q_full_ph;
    PhaseTracker<1> do_full_ph, pt_ph, dst_ph, dq_empty_ph, dsx_ph;

    auto commit_mc = [&](uint64_t* bar) { tcgen05_commit_multicast2_lead(lead, smem_ptr_u32(bar)); };

    // GEMM 1 (S^T = K @ Q^T) or 2 (dP^T = V @ dO^T): 2 hd subtiles x 4 k16; D lanes 0-63 = kv rows
    // x q 0-63 (CTA 0's Q half), lanes 64-127 = kv rows x q 64-127 (CTA 1's).
    auto gemm12_st_dpt = [&](auto is_st_const, int stage) {
      constexpr bool is_st    = decltype(is_st_const)::value;
      const uint32_t tmem_acc = is_st ? tmem_st : tmem_dpt;
      const uint64_t da_base  = is_st ? desc_k : desc_v;
      const uint64_t db_base  = is_st ? desc_q0 + (uint64_t)stage * Q_STAGE_DELTA : desc_do;
      #pragma unroll
      for (int s = 0; s < HD_SUBTILES; ++s) {
        #pragma unroll
        for (int ki = 0; ki < K_ATOMS_PER_SUBTILE; ++ki) {
          const bool accumulate = (s | ki) != 0;
          tcgen05_mma_f16_ss_2sm_lead(lead, tmem_acc, da_base + s * HALF_SUB_DELTA + ki * K16_COLS_DELTA,
                                      db_base + s * HALF_SUB_DELTA + ki * K16_COLS_DELTA, idesc_st_dpt,
                                      accumulate);
        }
      }
      commit_mc(is_st ? full_bar_st : full_bar_dpt);
    };

    // GEMM 3 (dV += P^T @ dO) or 5 (dK += dS^T @ Q[stage]): A = the bf16x2 overlay (lane half h
    // holds q-half h, packed at the start of each 32-column half: words [0, 16) and [32, 48)),
    // B = this CTA's own 64 q rows of dO / Q, MN-major. Pass p over the hd subtiles accumulates
    // into X_p; its lane halves are the two q-half partials of hd [64p, +64). The commit frees B.
    auto gemm35_dv_dk = [&](auto is_dv_const, int stage, bool first) {
      constexpr bool is_dv       = decltype(is_dv_const)::value;
      const uint32_t tmem_a_base = is_dv ? tmem_pt_bf16 : tmem_dst_bf16;
      const uint32_t tmem_acc0   = is_dv ? tmem_dv : tmem_dk;
      const uint64_t db_base = is_dv ? desc_do_mn : desc_q0_mn + (uint64_t)stage * Q_STAGE_DELTA;
      #pragma unroll
      for (int p = 0; p < HD_SUBTILES; ++p) {
        uint64_t db = db_base + p * HALF_SUB_DELTA;
        #pragma unroll
        for (int ki = 0; ki < K_ATOMS_PER_Q_HALF; ++ki) {
          const int col_half    = ki / (K_ATOMS_PER_Q_HALF / 2);
          const int k_in_half   = ki % (K_ATOMS_PER_Q_HALF / 2);
          const uint32_t tmem_a = tmem_a_base + (uint32_t)(col_half * (COLS_2CTA / 2) +
                                                           k_in_half * BF16X2_COLS_PER_K16);
          const bool accumulate = !(first && ki == 0);
          tcgen05_mma_f16_ts_2sm_lead(lead, tmem_acc0 + (uint32_t)(p * COLS_2CTA), tmem_a, db,
                                      idesc_dv_dk, accumulate);
          smem_desc_add_lo(db, K16_ROWS_DELTA);
        }
      }
      commit_mc(is_dv ? empty_bar_do : &empty_bar_q[stage]);
    };

    // GEMM 4 (dQ = dS @ K): A = own 64 q rows x 128 kv from sDST (kv-half 0 then 1, one walk),
    // B = K^T (all kv rows, own hd half) MN-major. D lanes 0-63 = hd 0-63, lanes 64-127 = hd 64-127.
    auto gemm4_dq = [&]() {
      uint64_t adst = desc_dst_mn;
      uint64_t bkt  = desc_kt_mn;
      #pragma unroll
      for (int ki = 0; ki < K_ATOMS_PER_KV_TILE; ++ki) {
        tcgen05_mma_f16_ss_2sm_lead(lead, tmem_dq, adst, bkt, idesc_dq, ki != 0);
        smem_desc_add_lo(adst, K16_ROWS_DELTA);
        smem_desc_add_lo(bkt, K16_ROWS_DELTA);
      }
      commit_mc(full_bar_dq);
      commit_mc(empty_bar_dst);  // both CTAs' sDST read: the softmax warps may write dS^T(j+1)
    };

    // Prologue: S^T(0), dP^T(0), dV(0). dP^T has its own columns here, so only dQ(j) waits for
    // the epilogue's release of dQ(j-1).
    {
      const int stage0 = q_full_ph.get_stage();
      wp_begin(wpc, WP_MMA_WAIT_FULL_Q);
      mbarrier_wait_parity(smem_ptr_u32(&full_bar_q[stage0]), q_full_ph.get_phase());
      wp_end(wpc, WP_MMA_WAIT_FULL_Q);
      gemm12_st_dpt(std::true_type{}, stage0);
      wp_begin(wpc, WP_MMA_WAIT_FULL_V);
      mbarrier_wait_parity(smem_ptr_u32(full_bar_do), do_full_ph.get_phase());
      do_full_ph.advance();
      wp_end(wpc, WP_MMA_WAIT_FULL_V);
      gemm12_st_dpt(std::false_type{}, 0);
      wp_begin(wpc, WP_MMA_WAIT_FULL_K);
      mbarrier_wait_parity(smem_ptr_u32(full_bar_pt), pt_ph.get_phase());
      pt_ph.advance();
      wp_end(wpc, WP_MMA_WAIT_FULL_K);
      gemm35_dv_dk(std::true_type{}, 0, true);
    }
    // Step j (FA4's 2cta order): S^T(j+1) | dK(j) | dP^T(j+1) | dQ(j) | dV(j+1). dQ has its own
    // columns, so dP^T(j+1) can go first and the dS^T exchange lands while it runs.
    for (int j = 0; j < it.local_k2q_num; ++j) {
      wp_marker(wpc, WP_ITER, j);
      const int stage = q_full_ph.get_stage();
      q_full_ph.advance();
      const int next_stage = q_full_ph.get_stage();
      const bool last      = j + 1 == it.local_k2q_num;

      if (!last) {
        wp_begin(wpc, WP_MMA_WAIT_FULL_Q);
        mbarrier_wait_parity(smem_ptr_u32(&full_bar_q[next_stage]), q_full_ph.get_phase());
        wp_end(wpc, WP_MMA_WAIT_FULL_Q);
        gemm12_st_dpt(std::true_type{}, next_stage);
      }
      if (last) commit_mc(full_bar_dv);
      wp_begin(wpc, WP_MMA_WAIT_P);
      mbarrier_wait_parity(smem_ptr_u32(full_bar_dst), dst_ph.get_phase());
      dst_ph.advance();
      wp_end(wpc, WP_MMA_WAIT_P);
      wp_begin(wpc, WP_MMA_ISSUE);
      gemm35_dv_dk(std::false_type{}, stage, j == 0);
      if (last) commit_mc(full_bar_dk);
      wp_end(wpc, WP_MMA_ISSUE);
      if (!last) {
        wp_begin(wpc, WP_MMA_WAIT_FULL_V);
        mbarrier_wait_parity(smem_ptr_u32(full_bar_do), do_full_ph.get_phase());
        do_full_ph.advance();
        wp_end(wpc, WP_MMA_WAIT_FULL_V);
        gemm12_st_dpt(std::false_type{}, 0);
      }
      // dQ(j) needs both CTAs' dS^T halves in every sDST (the relays' dsx_leader) and, from the
      // second step on, the epilogue warps' release of dQ(j-1).
      wp_begin(wpc, WP_MMA_WAIT_ACC);
      mbarrier_wait_parity(smem_ptr_u32(dsx_leader), dsx_ph.get_phase());
      dsx_ph.advance();
      if (j != 0) {
        mbarrier_wait_parity(smem_ptr_u32(empty_bar_dq), dq_empty_ph.get_phase());
        dq_empty_ph.advance();
      }
      wp_end(wpc, WP_MMA_WAIT_ACC);
      wp_begin(wpc, WP_MMA_ISSUE);
      gemm4_dq();
      wp_end(wpc, WP_MMA_ISSUE);
      if (!last) {
        wp_begin(wpc, WP_MMA_WAIT_FULL_K);
        mbarrier_wait_parity(smem_ptr_u32(full_bar_pt), pt_ph.get_phase());
        pt_ph.advance();
        wp_end(wpc, WP_MMA_WAIT_FULL_K);
        gemm35_dv_dk(std::true_type{}, 0, false);
      }
    }
    wp_flush(wpc);
    teardown();
    return;
  }
  else if (warp_id >= W_SOFTMAX0) {
    setmaxnreg_inc<136>();
    bar_sync<10>(416);
    const uint32_t tmem_base    = *tmem_slot;
    const uint32_t tmem_st      = tmem_base + TMEM_ST_2CTA;
    const uint32_t tmem_dpt     = tmem_base + TMEM_DPT_2CTA;
    const uint32_t tmem_dv      = tmem_base + TMEM_DV_2CTA;
    const uint32_t tmem_dk      = tmem_base + TMEM_DK_2CTA;
    const uint32_t tmem_pt_bf16 = tmem_st, tmem_dst_bf16 = tmem_dpt;

    // Softmax warp -> its piece of the folded 128-lane x 64-column S^T / dP^T tiles: lanes
    // [32 * lane_group, +32) = kv rows (lane_group & 1) * 32 + lane of q-half lane_group >> 1, the
    // 32 fp32 columns of col_half. The bf16x2 overlay (16 words) starts at the warp's own first
    // column, so it only overwrites columns the warp itself has consumed.
    const int softmax_warp_id = warp_id - W_SOFTMAX0;
    const int lane_group      = softmax_warp_id & 3;
    const int col_half        = softmax_warp_id >> 2;
    const int q_half          = lane_group >> 1;
    const int kv_row          = (lane_group & 1) * 32 + lane;  // within this CTA's 64 kv rows
    constexpr int WARP_COLS   = COLS_2CTA / 2;                  // 32 fp32 columns per warp
    const int q_offset        = q_half * Q_ROWS_2CTA + col_half * WARP_COLS;  // first q of the warp
    const uint32_t tmem_lane_base     = (uint32_t)(lane_group * 32) << 16;
    const uint32_t tmem_f32_offset    = tmem_lane_base + (uint32_t)(col_half * WARP_COLS);
    const uint32_t tmem_bf16x2_offset = tmem_f32_offset;
    constexpr int CHUNK_BF16     = 16 / (int)sizeof(__nv_bfloat16);
    constexpr int WARP_CHUNKS    = WARP_COLS / CHUNK_BF16;       // 4 of the row's 8 chunks
    // dS^T destination: the q-half this CTA reduces (q_half == cta_rank) goes to its own kv-half
    // region of sDST, the other q-half is staged in sXCHG for the peer's sDST.
    __nv_bfloat16* dst_tile = (q_half == cta_rank)
        ? sDST + (size_t)cta_rank * (DST_REGION_BYTES / (int)sizeof(__nv_bfloat16))
        : reinterpret_cast<__nv_bfloat16*>(sXCHG);
    __nv_bfloat16* sdst_row = dst_tile + kv_row * SUB_COLS_BF16;
    const uint32_t full_bar_pt_leader  = leader_addr(full_bar_pt);
    const uint32_t full_bar_dst_leader = leader_addr(full_bar_dst);

    PhaseTracker<1> st_ph, dpt_ph, delta_ph, dv_ph, dk_ph;
    PhaseTracker<NUM_Q_STAGES> lse_ph;
    EmptyPhaseTracker<1> dst_empty_ph;
    const WorkItem it = current_workitem();

    for (int j = 0; j < it.local_k2q_num; ++j) {
      wp_marker(wpc, WP_ITER, j);
      const int lse_stage     = lse_ph.get_stage();
      const float* lse_part   = sLSE + lse_stage * Q_TILE + q_offset;
      const float* delta_part = sDelta + q_offset;

      mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_lse[lse_stage]), lse_ph.get_phase());
      lse_ph.advance();
      wp_begin(wpc, WP_SM_WAIT_S);
      mbarrier_wait_parity_suspend(smem_ptr_u32(full_bar_st), st_ph.get_phase());
      st_ph.advance();
      wp_end(wpc, WP_SM_WAIT_S);

      wp_begin(wpc, WP_SM_SOFTMAX);
      uint32_t st_regs[WARP_COLS];
      uint32_t pt_bf16x2[WARP_COLS / 2];
      float* pt_fp32 = reinterpret_cast<float*>(st_regs);
      tcgen05_ld_32x32b_x32(tmem_st + tmem_f32_offset, st_regs);
      tcgen05_fence_before_thread_sync();
      {
        const float2 scale2 = f32x2_splat(scale_log2);
        #pragma unroll
        for (int c0 = 0; c0 < WARP_COLS; c0 += 4) {
          const float4 lse4           = *reinterpret_cast<const float4*>(lse_part + c0);
          const float negative_lse[4] = {-lse4.x, -lse4.y, -lse4.z, -lse4.w};
          #pragma unroll
          for (int c = c0; c < c0 + 4; c += 2) {
            const float2 z = ffma2(make_float2(pt_fp32[c], pt_fp32[c + 1]), scale2,
                                   make_float2(negative_lse[c - c0], negative_lse[c - c0 + 1]));
            const float p0 = ex2_approx_f32(z.x);
            const float p1 = ex2_approx_f32(z.y);
            pt_fp32[c]     = p0;
            pt_fp32[c + 1] = p1;
            pt_bf16x2[c / 2] = cvt_f32x2_to_bf16x2(p0, p1);
          }
        }
      }
      tcgen05_st_32x32b_x16(tmem_pt_bf16 + tmem_bf16x2_offset, pt_bf16x2);
      tcgen05_wait_st();
      tcgen05_fence_before_thread_sync();
      if (elect_one_sync()) {
        mbarrier_arrive_cluster_default(full_bar_pt_leader);
        mbarrier_arrive(smem_ptr_u32(&empty_bar_lse[lse_stage]));
      }
      wp_end(wpc, WP_SM_SOFTMAX);

      wp_begin(wpc, WP_CORR_WAIT);
      mbarrier_wait_parity_suspend(smem_ptr_u32(full_bar_delta), delta_ph.get_phase());
      delta_ph.advance();
      mbarrier_wait_parity_suspend(smem_ptr_u32(full_bar_dpt), dpt_ph.get_phase());
      dpt_ph.advance();
      wp_end(wpc, WP_CORR_WAIT);

      wp_begin(wpc, WP_SM_STORE_P);
      {
        uint32_t dpt_regs[WARP_COLS];
        tcgen05_ld_32x32b_x32(tmem_dpt + tmem_f32_offset, dpt_regs);
        tcgen05_fence_before_thread_sync();
        const float* dpt = reinterpret_cast<const float*>(dpt_regs);
        #pragma unroll
        for (int c0 = 0; c0 < WARP_COLS; c0 += 4) {
          const float4 delta4           = *reinterpret_cast<const float4*>(delta_part + c0);
          const float negative_delta[4] = {-delta4.x, -delta4.y, -delta4.z, -delta4.w};
          #pragma unroll
          for (int c = c0; c < c0 + 4; c += 2) {
            const float2 ds = fmul2(
                make_float2(pt_fp32[c], pt_fp32[c + 1]),
                fadd2(make_float2(dpt[c], dpt[c + 1]),
                      make_float2(negative_delta[c - c0], negative_delta[c - c0 + 1])));
            st_regs[c / 2] = cvt_f32x2_to_bf16x2(ds.x, ds.y);
          }
        }
      }
      uint32_t (&dst_bf16x2)[WARP_COLS / 2] = reinterpret_cast<uint32_t (&)[WARP_COLS / 2]>(st_regs);
      tcgen05_st_32x32b_x16(tmem_dst_bf16 + tmem_bf16x2_offset, dst_bf16x2);
      // sDST / sXCHG (and the peer's region this CTA's copy lands in) are free once dQ(j-1) has
      // completed: dP^T(j) is issued before dQ(j-1), so full_bar_dpt does not imply it.
      mbarrier_wait_parity_suspend(smem_ptr_u32(empty_bar_dst), dst_empty_ph.get_phase());
      dst_empty_ph.advance();
      const uint4* dst_chunks = reinterpret_cast<const uint4*>(dst_bf16x2);
      #pragma unroll
      for (int v = 0; v < WARP_CHUNKS; ++v)
        *reinterpret_cast<uint4*>(sdst_row + ((col_half * WARP_CHUNKS + v) ^ (kv_row & 7)) * CHUNK_BF16) =
            dst_chunks[v];
      tcgen05_wait_st();
      tcgen05_fence_before_thread_sync();
      fence_proxy_async_shared_cta();
      if (elect_one_sync()) {
        mbarrier_arrive_cluster_default(full_bar_dst_leader);
        mbarrier_arrive(smem_ptr_u32(empty_bar_delta));
      }
      // Exchange: once all 8 warps have staged, one thread ships the peer's q-half (8 KB) into
      // the peer's sDST region for this CTA's kv half and arms the peer's dsx_full with its tx.
      // The previous step's copy has completed (dQ(j-1) read it before dP^T(j) could commit).
      bar_sync<14>(256);
      if (softmax_warp_id == 0 && elect_one_sync()) {
        const uint32_t peer      = (uint32_t)(cta_rank ^ 1);
        const uint32_t peer_full = mapa_shared_cluster_u32(smem_ptr_u32(dsx_full), peer);
        mbarrier_arrive_expect_tx_cluster(peer_full, DST_REGION_BYTES);
        cpasync_bulk_s2cluster(
            mapa_shared_cluster_u32(smem_ptr_u32(sDST) + (uint32_t)(cta_rank * DST_REGION_BYTES), peer),
            smem_ptr_u32(sXCHG), DST_REGION_BYTES, peer_full);
      }
      wp_end(wpc, WP_SM_STORE_P);
    }

    // Tile end: dV, then dK. Accumulator pair X_a (hd 0-63) / X_b (hd 64-127); lanes m and 64 + m
    // hold the two q-half partials of kv row m. Lanes 64-127 publish theirs (fp32) into the dead
    // sK / sV area, lanes 0-63 add, scale, pack and write the bf16 bounce tile, one warp stores it.
    if (it.local_k2q_num != 0) {
      float* merge_scratch = reinterpret_cast<float*>(sK);  // [64 kv rows][128 hd] fp32 = 32 KB
      static_assert(2 * HALF_TILE_BYTES == KV_ROWS_2CTA * HEAD_DIM * (int)sizeof(float),
                    "sK + sV hold one fp32 [64 x 128] merge tile");
      auto store_dv_dk_tile = [&](uint32_t tmem_acc_pair, auto apply_sm_scale_const,
                                  uint8_t* bounce, const CUtensorMap* map64) {
        constexpr bool apply_sm_scale = decltype(apply_sm_scale_const)::value;
        // This warp: 64 hd columns [64 * col_half, +64) of X_{col_half} for its 32 kv rows.
        const uint32_t acc = tmem_acc_pair + (uint32_t)(col_half * COLS_2CTA) + tmem_lane_base;
        uint32_t acc_regs[COLS_2CTA];
        tcgen05_ld_32x32b_x32(acc, reinterpret_cast<uint32_t (&)[32]>(acc_regs[0]));
        tcgen05_ld_32x32b_x32(acc + 32, reinterpret_cast<uint32_t (&)[32]>(acc_regs[32]));
        tcgen05_fence_before_thread_sync();
        float* own = reinterpret_cast<float*>(acc_regs);
        // float4 slot v of row r at column group v ^ (r & 15): conflict-free strided rows.
        float* scratch_row = merge_scratch + kv_row * HEAD_DIM + col_half * COLS_2CTA;
        if (q_half == 1) {
          #pragma unroll
          for (int v = 0; v < COLS_2CTA / 4; ++v)
            *reinterpret_cast<float4*>(scratch_row + ((v ^ (kv_row & 15)) * 4)) =
                *reinterpret_cast<const float4*>(own + v * 4);
        }
        bar_sync<12>(256);
        if (q_half == 0) {
          uint8_t* bounce_subtile = bounce + col_half * HALF_SUB_BYTES;
          __nv_bfloat16* stage_row =
              reinterpret_cast<__nv_bfloat16*>(bounce_subtile) + kv_row * SUB_COLS_BF16;
          #pragma unroll
          for (int v = 0; v < COLS_2CTA / CHUNK_BF16; ++v) {
            float value[CHUNK_BF16];
            #pragma unroll
            for (int f4 = 0; f4 < 2; ++f4) {
              const int slot   = v * 2 + f4;
              const float4 peer = *reinterpret_cast<const float4*>(
                  scratch_row + ((slot ^ (kv_row & 15)) * 4));
              const float peer_v[4] = {peer.x, peer.y, peer.z, peer.w};
              #pragma unroll
              for (int e = 0; e < 4; ++e) {
                const float sum = own[slot * 4 + e] + peer_v[e];
                value[f4 * 4 + e] = apply_sm_scale ? sum * sm_scale : sum;
              }
            }
            uint4 packed;
            packed.x = cvt_f32x2_to_bf16x2(value[0], value[1]);
            packed.y = cvt_f32x2_to_bf16x2(value[2], value[3]);
            packed.z = cvt_f32x2_to_bf16x2(value[4], value[5]);
            packed.w = cvt_f32x2_to_bf16x2(value[6], value[7]);
            *reinterpret_cast<uint4*>(stage_row + ((v ^ (kv_row & 7)) * CHUNK_BF16)) = packed;
          }
        }
        fence_proxy_async_shared_cta();
        bar_sync<12>(256);
        if (softmax_warp_id == 0 && elect_one_sync()) {
          const int kv_begin = it.kv_block_id_in_seq * KV_TILE + cta_rank * KV_ROWS_2CTA;
          #pragma unroll
          for (int s = 0; s < HD_SUBTILES; ++s) {
            if constexpr (BHSD)
              tma_store_4d(map64, 0, kv_begin, s, it.batch_head,
                           smem_ptr_u32(bounce + s * HALF_SUB_BYTES));
            else
              tma_store_3d(map64, 0, it.batch * seqlen + kv_begin, it.head * HD_SUBTILES + s,
                           smem_ptr_u32(bounce + s * HALF_SUB_BYTES));
          }
          cp_async_bulk_commit_group();
        }
      };
      wp_begin(wpc, WP_EPI_WAIT_STORE);
      mbarrier_wait_parity_suspend(smem_ptr_u32(full_bar_dv), dv_ph.get_phase());
      dv_ph.advance();
      wp_end(wpc, WP_EPI_WAIT_STORE);
      wp_begin(wpc, WP_CORR_EPI);
      store_dv_dk_tile(tmem_dv, std::false_type{}, sDO, &tmap_dv64);
      mbarrier_wait_parity_suspend(smem_ptr_u32(full_bar_dk), dk_ph.get_phase());
      dk_ph.advance();
      store_dv_dk_tile(tmem_dk, std::true_type{}, sQ[0], &tmap_dk64);
      wp_end(wpc, WP_CORR_EPI);
    }
    wp_flush(wpc);
    bar_sync<10>(416);
    return;
  }
  else {
    setmaxnreg_inc<152>();
    bar_sync<10>(416);
    const uint32_t tmem_base = *tmem_slot;
    const uint32_t tmem_dq   = tmem_base + TMEM_DQ_2CTA;

    // This CTA's dQ tile: its 64 q rows x 128 hd, folded: lane l = q row l & 63, hd half l >> 6.
    // Epilogue warp w reads lanes [32w, +32): q_loc = (w & 1) * 32 + lane, hd [64 * (w >> 1), +64).
    // Drain: 2 rounds x 2 chunks of 32 hd columns; round p, hd half h -> chunk 2h + p, stage buffer
    // 2p + h; each round's two 8 KB chunks are one bulk group.
    const int epi_warp_id         = warp_id - W_EPI0;
    const uint32_t tmem_lane_base = (uint32_t)(epi_warp_id * 32) << 16;
    const int q_loc               = (epi_warp_id & 1) * 32 + lane;
    const int hd_half             = epi_warp_id >> 1;
    float* stage_buf[DQ_STAGE_BUFFERS_2CTA];
    #pragma unroll
    for (int b = 0; b < DQ_STAGE_BUFFERS_2CTA; ++b)
      stage_buf[b] = reinterpret_cast<float*>(sDQ_STAGE_bytes + b * DQ_CHUNK_BYTES_2CTA);
    const uint32_t empty_bar_dq_leader = leader_addr(empty_bar_dq);

    PhaseTracker<1> dq_full_ph;
    const WorkItem it = current_workitem();
    // Rank r's half of a q block's accumulator: 4 chunks x [8 float4 groups][64 q rows][4].
    float* dqaccum_head = dqaccum + (size_t)it.batch_head * num_kv_blocks_per_seq * DQ_BLOCK_ELEMS +
                          (size_t)cta_rank * DQ_RANK_ELEMS;

    for (int j = 0; j < it.local_k2q_num; ++j) {
      wp_marker(wpc, WP_ITER, j);
      float* dqaccum_block = dqaccum_head + (size_t)it.local_k2q_idx[j] * DQ_BLOCK_ELEMS;

      wp_begin(wpc, WP_EPI_WAIT_ACC);
      mbarrier_wait_parity(smem_ptr_u32(full_bar_dq), dq_full_ph.get_phase());
      dq_full_ph.advance();
      wp_end(wpc, WP_EPI_WAIT_ACC);

      wp_begin(wpc, WP_EPI_TMEM_LD);
      uint32_t dq_regs[COLS_2CTA];
      tcgen05_ld_32x32b_x32(tmem_dq + tmem_lane_base, reinterpret_cast<uint32_t (&)[32]>(dq_regs[0]));
      tcgen05_ld_32x32b_x32(tmem_dq + tmem_lane_base + 32,
                            reinterpret_cast<uint32_t (&)[32]>(dq_regs[32]));
      tcgen05_fence_before_thread_sync();
      if (elect_one_sync()) mbarrier_arrive_cluster_default(empty_bar_dq_leader);
      wp_end(wpc, WP_EPI_TMEM_LD);

      wp_begin(wpc, WP_EPI_STORE);
      #pragma unroll
      for (int p = 0; p < 2; ++p) {
        // The buffers of round p were last read two groups ago; at most the previous round's
        // group may still be in flight.
        if (epi_warp_id == 0 && elect_one_sync()) cp_async_bulk_wait_group_read<1>();
        bar_sync<11>(128);
        const float4* dq_row4 =
            reinterpret_cast<const float4*>(dq_regs + p * DQ_CHUNK_COLS);
        float* stage = stage_buf[2 * p + hd_half];
        #pragma unroll
        for (int v4 = 0; v4 < DQ_CHUNK_COLS / 4; ++v4)
          *reinterpret_cast<float4*>(stage + v4 * Q_ROWS_2CTA * 4 + q_loc * 4) = dq_row4[v4];
        fence_proxy_async_shared_cta();
        bar_sync<11>(128);
        if (epi_warp_id == 0 && elect_one_sync()) {
          #pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int chunk = 2 * h + p;
            cpasync_reduce_bulk_add_f32(dqaccum_block + (size_t)chunk * Q_ROWS_2CTA * DQ_CHUNK_COLS,
                                        smem_ptr_u32(stage_buf[2 * p + h]), DQ_CHUNK_BYTES_2CTA);
          }
          cp_async_bulk_commit_group();
        }
      }
      wp_end(wpc, WP_EPI_STORE);
    }
    if (it.local_k2q_num == 0) {
      // Zero tiles: this CTA's 64 kv rows of dK and dV from stage buffers 0 and 1 (2 x 8 KB = one
      // bf16 [64 x 128] tile as two hd subtiles).
      static_assert(2 * DQ_CHUNK_BYTES_2CTA == HALF_TILE_BYTES,
                    "two dQ stage buffers hold one 64 kv x 128 hd bf16 tile");
      constexpr int ZERO_UINT4_PER_THREAD = HALF_TILE_BYTES / (128 * 16);
      uint4* zero_tile = reinterpret_cast<uint4*>(sDQ_STAGE_bytes) + epi_warp_id * 32 + lane;
      #pragma unroll
      for (int v = 0; v < ZERO_UINT4_PER_THREAD; ++v)
        zero_tile[v * 128] = make_uint4(0u, 0u, 0u, 0u);
      fence_proxy_async_shared_cta();
      bar_sync<11>(128);
      if (epi_warp_id == 0 && elect_one_sync()) {
        const int kv_begin = it.kv_block_id_in_seq * KV_TILE + cta_rank * KV_ROWS_2CTA;
        #pragma unroll
        for (int which = 0; which < 2; ++which) {
          const CUtensorMap* map64 = (which == 0) ? &tmap_dk64 : &tmap_dv64;
          #pragma unroll
          for (int s = 0; s < HD_SUBTILES; ++s) {
            const uint32_t src = smem_ptr_u32(sDQ_STAGE_bytes + (size_t)s * HALF_SUB_BYTES);
            if constexpr (BHSD)
              tma_store_4d(map64, 0, kv_begin, s, it.batch_head, src);
            else
              tma_store_3d(map64, 0, it.batch * seqlen + kv_begin, it.head * HD_SUBTILES + s, src);
          }
        }
        cp_async_bulk_commit_group();
        cp_async_bulk_wait_group_read<0>();
      }
      bar_sync<11>(128);
    }
    if (epi_warp_id == 0 && elect_one_sync()) cp_async_bulk_wait_group_read<0>();
    bar_sync<11>(128);
    wp_flush(wpc);
    // All dQ pushes issued: the postprocess may launch (its wait still covers their completion).
    if constexpr (KERNEL_PDL) {
      if (epi_warp_id == 0 && elect_one_sync()) griddepcontrol_launch_dependents();
    }
    bar_sync<10>(416);
    return;
  }
}

// Preprocess: one CTA per (q128 block, batch*head), 256 threads. Zeroes the block's dqaccum slice
// (coalesced float4 stores), then Delta = rowsum(bf16(O) * dO): 16 threads per row x 8 columns
// each, shuffle-reduced.
template <bool BHSD = false>
__global__ void __launch_bounds__(256, 1)
    vsa_bwd_preprocess_kernel(const __nv_bfloat16* __restrict__ o,
                              const __nv_bfloat16* __restrict__ dout,
                              float* __restrict__ delta_rows, float* __restrict__ dqaccum,
                              int num_heads, int seqlen) {
  const int q_block_id  = (int)blockIdx.x;
  const int batch_head  = (int)blockIdx.y;
  const int batch = batch_head / num_heads, head = batch_head % num_heads;
  const int token_begin = q_block_id * Q_TILE;

  // The previous postprocess still reads dqaccum: wait before zeroing it.
  if constexpr (KERNEL_PDL) griddepcontrol_wait();
  // Zero this block's dqaccum slice: 16384 f32 = 4096 float4 by 256 threads.
  float4* dqaccum_zero_destination = reinterpret_cast<float4*>(
      dqaccum + (size_t)batch_head * seqlen * HEAD_DIM + (size_t)q_block_id * DQ_BLOCK_ELEMS);
  const float4 zero_float4 = make_float4(0.f, 0.f, 0.f, 0.f);
  #pragma unroll
  for (int i = 0; i < (DQ_BLOCK_ELEMS / 4) / 256; ++i)
    dqaccum_zero_destination[i * 256 + threadIdx.x] = zero_float4;
  // dqaccum is zeroed: the main grid may set up while Delta forms.
  if constexpr (KERNEL_PDL) griddepcontrol_launch_dependents();

  // Delta: 16 threads per row, 8 columns each (one uint4 of bf16 from O and one from dO).
  const int row_in_pass     = (int)threadIdx.x / 16;        // 16 rows per pass, 8 passes
  const int dimension_begin = ((int)threadIdx.x % 16) * 8;  // 8 bf16 per thread
  #pragma unroll
  for (int row_pass = 0; row_pass < Q_TILE / 16; ++row_pass) {
    const int token      = token_begin + row_pass * 16 + row_in_pass;
    const size_t element =
        token_offset<BHSD>(batch, head, num_heads, seqlen, token) + dimension_begin;
    const uint4 o_vector    = *reinterpret_cast<const uint4*>(o + element);
    const uint4 dout_vector = *reinterpret_cast<const uint4*>(dout + element);
    const __nv_bfloat162* o_pairs    = reinterpret_cast<const __nv_bfloat162*>(&o_vector);
    const __nv_bfloat162* dout_pairs = reinterpret_cast<const __nv_bfloat162*>(&dout_vector);
    float delta_accumulator = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 o_pair_as_float    = __bfloat1622float2(o_pairs[i]);
      const float2 dout_pair_as_float = __bfloat1622float2(dout_pairs[i]);
      delta_accumulator += o_pair_as_float.x * dout_pair_as_float.x +
                           o_pair_as_float.y * dout_pair_as_float.y;
    }
    #pragma unroll
    for (int shuffle_offset = 8; shuffle_offset > 0; shuffle_offset >>= 1)
      delta_accumulator += __shfl_down_sync(0xffffffffu, delta_accumulator, shuffle_offset);
    if (dimension_begin == 0) delta_rows[(size_t)batch_head * seqlen + token] = delta_accumulator;
  }
}

// Postprocess (FA4's scheme, fa4_bwd_postprocess.py): one CTA per (q128 block, batch*head),
// 128 threads, 64 KB SMEM.
//   A. The whole 16384-f32 drain-native block is loaded contiguously with cp.async.cg (32 x 16 B
//      per thread).
//   B. Thread t unscrambles q row t: rank r = t >> 6, chunk c, float4 group v4 sits at smem[r * 8192
//      + c * 2048 + v4 * 256 + (t & 63) * 4] (epilogue staging order); x sm_scale, packed to bf16.
//   C. Row t goes back as 16 x 16 B into a bf16 overlay of the same SMEM, chunk slots xor-swizzled
//      by t & 7 so the strided row stores are conflict-free.
//   D. Store: 16 threads x 8 bf16 per row, 8 rows per pass, every gmem row segment 256 B
//      contiguous.
template <bool BHSD = false>
__global__ void __launch_bounds__(128, 1)
    vsa_bwd_postprocess_kernel(const float* __restrict__ dqaccum, __nv_bfloat16* __restrict__ dq,
                               int num_heads, int seqlen, float sm_scale) {
  extern __shared__ __align__(16) float post_smem[];  // 16384 f32, later the bf16 overlay
  const int q_block_id  = (int)blockIdx.x;
  const int batch_head  = (int)blockIdx.y;
  const int batch = batch_head / num_heads, head = batch_head % num_heads;
  const int token_begin = q_block_id * Q_TILE;
  const int thread      = (int)threadIdx.x;
  constexpr int CHUNK_BF16 = 16 / (int)sizeof(__nv_bfloat16);  // 8 bf16 per 16-byte chunk

  // A
  const float4* dqaccum_block = reinterpret_cast<const float4*>(
      dqaccum + (size_t)batch_head * seqlen * HEAD_DIM + (size_t)q_block_id * DQ_BLOCK_ELEMS);
  if constexpr (KERNEL_PDL) griddepcontrol_wait();  // the main grid's pushes are complete
  #pragma unroll
  for (int i = 0; i < (DQ_BLOCK_ELEMS / 4) / 128; ++i)
    cp_async_cg_16(smem_ptr_u32(post_smem + (i * 128 + thread) * 4),
                   dqaccum_block + i * 128 + thread);
  cp_async_commit_group();
  cp_async_wait_group<0>();
  __syncthreads();
  // dqaccum read out: the next preprocess may launch (it waits before re-zeroing).
  if constexpr (KERNEL_PDL) griddepcontrol_launch_dependents();

  // B
  uint32_t dq_packed_low[HEAD_DIM / 4];   // hd 0..63 as 32 bf16x2
  uint32_t dq_packed_high[HEAD_DIM / 4];  // hd 64..127
  #pragma unroll
  for (int chunk = 0; chunk < DQ_CHUNKS; ++chunk) {
    #pragma unroll
    for (int v4 = 0; v4 < DQ_CHUNK_COLS / 4; ++v4) {
      const float4 value = *reinterpret_cast<const float4*>(
          post_smem + (thread >> 6) * DQ_RANK_ELEMS + chunk * Q_ROWS_2CTA * DQ_CHUNK_COLS +
          v4 * Q_ROWS_2CTA * 4 + (thread & 63) * 4);
      const int dimension  = chunk * DQ_CHUNK_COLS + v4 * 4;  // ascending hd
      uint32_t* dq_packed  = (dimension < 64) ? dq_packed_low : dq_packed_high;
      const int pair_index = (dimension & 63) / 2;
      dq_packed[pair_index + 0] = cvt_f32x2_to_bf16x2(value.x * sm_scale, value.y * sm_scale);
      dq_packed[pair_index + 1] = cvt_f32x2_to_bf16x2(value.z * sm_scale, value.w * sm_scale);
    }
  }
  __syncthreads();

  // C
  __nv_bfloat16* dq_tile_bf16 = reinterpret_cast<__nv_bfloat16*>(post_smem);
  {
    const uint4* low_chunks  = reinterpret_cast<const uint4*>(dq_packed_low);
    const uint4* high_chunks = reinterpret_cast<const uint4*>(dq_packed_high);
    __nv_bfloat16* row = dq_tile_bf16 + thread * HEAD_DIM;
    #pragma unroll
    for (int v = 0; v < 8; ++v)
      *reinterpret_cast<uint4*>(row + (v ^ (thread & 7)) * CHUNK_BF16) = low_chunks[v];
    #pragma unroll
    for (int v = 0; v < 8; ++v)
      *reinterpret_cast<uint4*>(row + 64 + (v ^ (thread & 7)) * CHUNK_BF16) = high_chunks[v];
  }
  __syncthreads();

  // D
  const int row_in_pass     = thread / 16;        // 8 rows per pass
  const int dimension_begin = (thread % 16) * 8;  // 8 bf16 per thread
  #pragma unroll
  for (int row_pass = 0; row_pass < Q_TILE / 8; ++row_pass) {
    const int row           = row_pass * 8 + row_in_pass;
    const int half          = dimension_begin / 64;
    const int chunk         = ((dimension_begin % 64) / CHUNK_BF16) ^ (row & 7);
    const uint4 value       = *reinterpret_cast<const uint4*>(
        dq_tile_bf16 + row * HEAD_DIM + half * 64 + chunk * CHUNK_BF16);
    const size_t element =
        token_offset<BHSD>(batch, head, num_heads, seqlen, token_begin + row) + dimension_begin;
    *reinterpret_cast<uint4*>(dq + element) = value;
  }
}

// ---------------------------------------------------------------------------
// Host launchers (stream-chained: pre -> main -> post).
// ---------------------------------------------------------------------------

struct VsaBwdArgs {
  const __nv_bfloat16 *q, *k, *v, *dout;  // bf16, VSA_BHSD ? [B, H, S, 128] : [B, S, H, 128]
  float* dqaccum;                          // drain-native fp32 scratch (pre zeroes it)
  __nv_bfloat16 *dk, *dv, *dq;             // bf16 outputs, same layout as the inputs
  const float *lse_rows, *delta_rows;      // [B*H, S]
  const int* k2q_idx;                      // [B*H*nb, max_q_blocks] local q128 ids, padded rows
  const int* k2q_num;                      // [B*H*nb] entries valid per row
  int num_samples, num_heads, seqlen, num_kv_blocks_per_seq;  // kv128 blocks per sequence
  int max_q_blocks;                        // k2q_idx row stride
  float sm_scale;
};

// Q, K, V, dO, dK, dV tensor maps, one 128-token x 64-hd box per TMA (two per tile):
//   BSHD: 3D [SUB_COLS_BF16 hd, B*S tokens, H*2 hd units], strides {H*128*2, 128} bytes.
//   BHSD: 4D [SUB_COLS_BF16 hd, S tokens, 2 hd units, B*H], strides {128*2, 128, S*128*2} bytes.
inline cudaError_t make_tma_tile_units(CUtensorMap* map, const __nv_bfloat16* ptr, int B, int H,
                                       int S, int box_tokens) {
  CUresult r;
  if (VSA_BHSD) {
    uint64_t gd[4] = {(uint64_t)SUB_COLS_BF16, (uint64_t)S, (uint64_t)HD_SUBTILES,
                      (uint64_t)B * H};
    uint64_t gs[3] = {(uint64_t)HEAD_DIM * 2, (uint64_t)SUB_COLS_BYTES,
                      (uint64_t)S * HEAD_DIM * 2};
    uint32_t bd[4] = {(uint32_t)SUB_COLS_BF16, (uint32_t)box_tokens, 1u, 1u};
    uint32_t es[4] = {1u, 1u, 1u, 1u};
    r = cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
                               const_cast<__nv_bfloat16*>(ptr), gd, gs, bd, es,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                               CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  } else {
    uint64_t gd[3] = {(uint64_t)SUB_COLS_BF16, (uint64_t)B * S, (uint64_t)H * HD_SUBTILES};
    uint64_t gs[2] = {(uint64_t)H * HEAD_DIM * 2, (uint64_t)SUB_COLS_BYTES};
    uint32_t bd[3] = {(uint32_t)SUB_COLS_BF16, (uint32_t)box_tokens, 1u};
    uint32_t es[3] = {1u, 1u, 1u};
    r = cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3,
                               const_cast<__nv_bfloat16*>(ptr), gd, gs, bd, es,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                               CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  }
  return (r == CUDA_SUCCESS) ? cudaSuccess : cudaErrorInvalidValue;
}

inline cudaError_t launch_vsa_bwd_sm100a(const VsaBwdArgs& a, cudaStream_t stream) {
  const int B = a.num_samples, H = a.num_heads, S = a.seqlen;

  // Tensor maps are pure functions of (pointer, shape): encode once per config (re-encoding per
  // launch costs ~100+ us of driver time per call).
  constexpr int SMEM_BYTES = SMEM_TOTAL_2CTA;
  auto kernel = vsa_bwd_main_kernel<VSA_BHSD>;
  static CUtensorMap tk_;                                          // 128-token box (K^T)
  static CUtensorMap tq64_, tk64_, tv64_, tdo64_, tdk64_, tdv64_;  // 64-token boxes
  static const void* cached_q = nullptr;
  static int cached_B = 0, cached_S = 0, cached_H = 0;
  if (cached_q != (const void*)a.q || cached_B != B || cached_S != S || cached_H != H) {
    const int half = KV_ROWS_2CTA;
    const bool ok = make_tma_tile_units(&tk_, a.k, B, H, S, KV_TILE) == cudaSuccess &&
                    make_tma_tile_units(&tq64_, a.q, B, H, S, half) == cudaSuccess &&
                    make_tma_tile_units(&tk64_, a.k, B, H, S, half) == cudaSuccess &&
                    make_tma_tile_units(&tv64_, a.v, B, H, S, half) == cudaSuccess &&
                    make_tma_tile_units(&tdo64_, a.dout, B, H, S, half) == cudaSuccess &&
                    make_tma_tile_units(&tdk64_, a.dk, B, H, S, half) == cudaSuccess &&
                    make_tma_tile_units(&tdv64_, a.dv, B, H, S, half) == cudaSuccess;
    if (!ok) return cudaErrorInvalidValue;
    cached_q = (const void*)a.q;
    cached_B = B;
    cached_S = S;
    cached_H = H;
  }
  static bool smem_set  = false;
  if (!smem_set) {
    cudaError_t e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         SMEM_BYTES);
    if (e != cudaSuccess) return e;
    smem_set = true;
  }

  const float scale_log2 = a.sm_scale * 1.4426950408889634f;
  // Grid: cluster pairs along x, (2 * kv blocks, B*H), the pair (2k, 2k+1) owns kv block k.
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim            = dim3((unsigned)(2 * a.num_kv_blocks_per_seq), (unsigned)(B * H), 1);
  cfg.blockDim           = dim3(N_WARPS * 32, 1, 1);
  cfg.dynamicSmemBytes   = SMEM_BYTES;
  cfg.stream             = stream;
  cudaLaunchAttribute at[2];
  at[0].id               = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = 2;
  at[0].val.clusterDim.y = 1;
  at[0].val.clusterDim.z = 1;
  at[1].id               = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[1].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs              = at;
  cfg.numAttrs           = KERNEL_PDL ? 2 : 1;
  return cudaLaunchKernelEx(&cfg, kernel, tk_, tq64_, tk64_, tv64_, tdo64_, tdk64_, tdv64_,
                            a.dqaccum, a.lse_rows, a.delta_rows, a.k2q_idx, a.k2q_num,
                            a.max_q_blocks, B, H, S, scale_log2, a.sm_scale);
}

// Preprocess / postprocess launches carry the PDL attribute so each may start while the previous
// kernel in the stream drains (their griddepcontrol.wait guards the data).
inline cudaLaunchConfig_t pdl_launch_config(dim3 grid, dim3 block, size_t smem,
                                            cudaStream_t stream, cudaLaunchAttribute* at) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim          = grid;
  cfg.blockDim         = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream           = stream;
  at->id               = cudaLaunchAttributeProgrammaticStreamSerialization;
  at->val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs            = at;
  cfg.numAttrs         = KERNEL_PDL ? 1 : 0;
  return cfg;
}

inline cudaError_t launch_vsa_bwd_preprocess(const __nv_bfloat16* o, const __nv_bfloat16* dout,
                                             float* delta_rows, float* dqaccum, int num_samples,
                                             int num_heads, int seqlen, cudaStream_t stream) {
  cudaLaunchAttribute at[1];
  cudaLaunchConfig_t cfg = pdl_launch_config(
      dim3((unsigned)(seqlen / Q_TILE), (unsigned)(num_samples * num_heads), 1), dim3(256, 1, 1),
      0, stream, at);
  return cudaLaunchKernelEx(&cfg, vsa_bwd_preprocess_kernel<VSA_BHSD>, o, dout, delta_rows,
                            dqaccum, num_heads, seqlen);
}

inline cudaError_t launch_vsa_bwd_postprocess(const float* dqaccum, __nv_bfloat16* dq,
                                              int num_samples, int num_heads, int seqlen,
                                              float sm_scale, cudaStream_t stream) {
  constexpr int POST_SMEM_BYTES = DQ_BLOCK_ELEMS * (int)sizeof(float);
  static bool post_smem_set     = false;
  if (!post_smem_set) {
    cudaError_t e = cudaFuncSetAttribute(vsa_bwd_postprocess_kernel<VSA_BHSD>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         POST_SMEM_BYTES);
    if (e != cudaSuccess) return e;
    post_smem_set = true;
  }
  cudaLaunchAttribute at[1];
  cudaLaunchConfig_t cfg = pdl_launch_config(
      dim3((unsigned)(seqlen / Q_TILE), (unsigned)(num_samples * num_heads), 1), dim3(128, 1, 1),
      POST_SMEM_BYTES, stream, at);
  return cudaLaunchKernelEx(&cfg, vsa_bwd_postprocess_kernel<VSA_BHSD>, dqaccum,
                            dq, num_heads, seqlen, sm_scale);
}

// Harness: the host reference tensors are [B*S, H, hd]; under VSA_BHSD the device copies are
// permuted to [B, H, S, hd] on upload and back on download.
static std::vector<__nv_bfloat16> to_device_layout(const std::vector<__nv_bfloat16>& src, int B,
                                                   int H, int S, int head_dim) {
  if (!VSA_BHSD) return src;
  std::vector<__nv_bfloat16> dst(src.size());
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h)
      for (int t = 0; t < S; ++t)
        memcpy(&dst[(((long)b * H + h) * S + t) * head_dim],
               &src[(((long)b * S + t) * H + h) * head_dim], (size_t)head_dim * 2);
  return dst;
}

static std::vector<__nv_bfloat16> from_device_layout(const std::vector<__nv_bfloat16>& src,
                                                     int B, int H, int S, int head_dim) {
  if (!VSA_BHSD) return src;
  std::vector<__nv_bfloat16> dst(src.size());
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h)
      for (int t = 0; t < S; ++t)
        memcpy(&dst[(((long)b * S + t) * H + h) * head_dim],
               &src[(((long)b * H + h) * S + t) * head_dim], (size_t)head_dim * 2);
  return dst;
}

// ---------------------------------------------------------------------------
// Bench harness (CPU reference + verify + timing).
// ---------------------------------------------------------------------------

static const float LOG2E = 1.4426950408889634f;

// Minimal fp32 .npy v1.0 writer (npy_io.cuh only reads). C order, 64-byte-aligned header.
static void npy_save_f32(const std::string& path, const float* data,
                         std::initializer_list<long> shape) {
  std::string dims;
  long n   = 1;
  size_t i = 0;
  for (long d : shape) {
    dims += std::to_string(d);
    n *= d;
    if (++i < shape.size()) dims += ", ";
  }
  if (shape.size() == 1) dims += ",";
  std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + dims + "), }";
  const size_t pad   = (64 - (10 + header.size() + 1) % 64) % 64;
  header.append(pad, ' ');
  header += '\n';
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) {
    fprintf(stderr, "npy_save: cannot open %s\n", path.c_str());
    exit(1);
  }
  const unsigned char magic[8] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
  fwrite(magic, 1, 8, f);
  const uint16_t header_len = (uint16_t)header.size();
  fwrite(&header_len, 2, 1, f);
  fwrite(header.data(), 1, header.size(), f);
  fwrite(data, sizeof(float), (size_t)n, f);
  fclose(f);
}

// Deterministic sorted k2q inversion: count per kv block + concatenated q-block lists,
// sorted ascending (built by ascending q-block walk); feeds the CPU reference and
// build_k2q_padded -> the kernel's k2q_idx/k2q_num.
struct KvToQ {
  std::vector<int> count;     // per global kv block, size B*H*num_blocks
  std::vector<int> offset;    // prefix sum, size B*H*num_blocks + 1
  std::vector<int> q_blocks;  // concatenated sorted local q-block ids
};

static KvToQ invert_q2k(const int* q2k_idx, const int* q2k_num, int B, int H, int num_blocks,
                        int max_kv) {
  const int total = B * H * num_blocks;
  KvToQ inv;
  inv.count.assign(total, 0);
  for (int global_mtile = 0; global_mtile < total; ++global_mtile) {
    const int bh = global_mtile / num_blocks;
    for (int i = 0; i < q2k_num[global_mtile]; ++i)
      inv.count[bh * num_blocks + q2k_idx[(size_t)global_mtile * max_kv + i]]++;
  }
  inv.offset.assign(total + 1, 0);
  for (int i = 0; i < total; ++i) inv.offset[i + 1] = inv.offset[i] + inv.count[i];
  inv.q_blocks.assign(inv.offset[total], 0);
  std::vector<int> cursor(inv.offset.begin(), inv.offset.end() - 1);
  for (int global_mtile = 0; global_mtile < total; ++global_mtile) {
    const int bh    = global_mtile / num_blocks;
    const int mtile = global_mtile % num_blocks;
    for (int i = 0; i < q2k_num[global_mtile]; ++i) {
      const int gkb               = bh * num_blocks + q2k_idx[(size_t)global_mtile * max_kv + i];
      inv.q_blocks[cursor[gkb]++] = mtile;
    }
  }
  return inv;
}

// The kernel's index form (FastVideo's invert_indices layout): one padded row of max_q_blocks
// plain q-block ids per (batch*head, kv block) plus a count; entries past the count are never
// read (poisoned with -1 so a stray read shows). Rows are the widest list here; FastVideo pads
// to num_blocks.
struct K2qPadded {
  int max_q_blocks;
  std::vector<int> idx;  // [B*H*nb, max_q_blocks]
  std::vector<int> num;  // [B*H*nb]
};

static K2qPadded build_k2q_padded(const KvToQ& k2q, int B, int H, int nb) {
  const int items = B * H * nb;
  K2qPadded out;
  int widest = 0;
  for (int c : k2q.count) widest = std::max(widest, c);
  out.max_q_blocks = std::max(widest, 1);
  out.idx.assign((size_t)items * out.max_q_blocks, -1);
  out.num = k2q.count;
  for (int item = 0; item < items; ++item)
    for (int i = k2q.offset[item]; i < k2q.offset[item + 1]; ++i)
      out.idx[(size_t)item * out.max_q_blocks + (i - k2q.offset[item])] = k2q.q_blocks[i];
  return out;
}

// CPU reference. Pass 1 (parallel over q-blocks): forward O + M, Delta, dQ.
// Pass 2 (parallel over kv blocks, sorted k2q walk): dK, dV -- race-free and
// deterministic because each kv block owns its dK/dV rows.
static void cpu_vsa_bwd_ref(const __nv_bfloat16* hQ, const __nv_bfloat16* hK,
                            const __nv_bfloat16* hV, const __nv_bfloat16* hdO, float* hO, float* hM,
                            float* hDelta, float* hdQ, float* hdK, float* hdV, int B, int H, int S,
                            int hd, int num_blocks, int max_kv, const int* q2k_idx,
                            const int* q2k_num, const KvToQ& k2q, const float* file_O,
                            const float* file_M, float* file_err) {
  const float sm_scale   = 1.0f / sqrtf((float)hd);
  const long total_elems = (long)B * S * H * hd;
  for (long i = 0; i < total_elems; ++i) {
    hO[i]  = 0.f;
    hdQ[i] = 0.f;
    hdK[i] = 0.f;
    hdV[i] = 0.f;
  }

  float lse_err = 0.f, o_err = 0.f;
  #pragma omp parallel for schedule(dynamic) reduction(max : lse_err, o_err)
  for (int bhq = 0; bhq < B * H * num_blocks; ++bhq) {
    const int mtile         = bhq % num_blocks;
    const int bh            = bhq / num_blocks;
    const int h             = bh % H;
    const int b             = bh / H;
    const int num_kv_blocks = q2k_num[bhq];

    for (int qi = 0; qi < BLOCK; ++qi) {
      const long qp  = (long)b * S + (long)mtile * BLOCK + qi;
      const long row = (long)bh * S + (long)mtile * BLOCK + qi;
      std::vector<float> z((size_t)num_kv_blocks * BLOCK);
      float m = -INFINITY;
      int idx = 0;
      for (int kk = 0; kk < num_kv_blocks; ++kk) {
        const int blk = q2k_idx[(size_t)bhq * max_kv + kk];
        for (int kj = 0; kj < BLOCK; ++kj) {
          const long kp = (long)b * S + (long)blk * BLOCK + kj;
          float dot     = 0.f;
          for (int e = 0; e < hd; ++e)
            dot += __bfloat162float(hQ[(qp * H + h) * hd + e]) *
                   __bfloat162float(hK[(kp * H + h) * hd + e]);
          z[idx] = dot * sm_scale * LOG2E;
          m      = fmaxf(m, z[idx]);
          ++idx;
        }
      }
      float l = 0.f;
      for (int j = 0; j < num_kv_blocks * BLOCK; ++j) l += exp2f(z[j] - m);
      const float inv_l = 1.f / l;
      idx               = 0;
      for (int kk = 0; kk < num_kv_blocks; ++kk) {
        const int blk = q2k_idx[(size_t)bhq * max_kv + kk];
        for (int kj = 0; kj < BLOCK; ++kj) {
          const long kp = (long)b * S + (long)blk * BLOCK + kj;
          const float p = exp2f(z[idx] - m) * inv_l;
          for (int e = 0; e < hd; ++e)
            hO[(qp * H + h) * hd + e] += p * __bfloat162float(hV[(kp * H + h) * hd + e]);
          ++idx;
        }
      }
      float M_row = m + log2f(l);
      if (file_M) {
        lse_err = fmaxf(lse_err, fabsf(file_M[row] - M_row));
        for (int e = 0; e < hd; ++e) {
          const long o_index = (qp * H + h) * hd + e;
          o_err = fmaxf(o_err, fabsf(file_O[o_index] - hO[o_index]) /
                                   (fabsf(hO[o_index]) / 128.f + 1e-4f));
          hO[o_index] = file_O[o_index];
        }
        M_row = file_M[row];
      }
      hM[row] = M_row;

      // Delta from bf16-ROUNDED O: production backward consumes the forward's
      // saved bf16 O, so the reference must quantize O before the rowsum or
      // the GPU comparison inherits a spurious ~4e-3 quantization skew.
      float delta = 0.f;
      for (int e = 0; e < hd; ++e)
        delta += __bfloat162float(hdO[(qp * H + h) * hd + e]) *
                 __bfloat162float(__float2bfloat16(hO[(qp * H + h) * hd + e]));
      hDelta[row] = delta;

      idx = 0;
      for (int kk = 0; kk < num_kv_blocks; ++kk) {
        const int blk = q2k_idx[(size_t)bhq * max_kv + kk];
        for (int kj = 0; kj < BLOCK; ++kj) {
          const long kp = (long)b * S + (long)blk * BLOCK + kj;
          const float P = exp2f(z[idx] - M_row);
          float dP      = 0.f;
          for (int e = 0; e < hd; ++e)
            dP += __bfloat162float(hdO[(qp * H + h) * hd + e]) *
                  __bfloat162float(hV[(kp * H + h) * hd + e]);
          // dS is cast to bf16 before the dQ/dK dots on both the Triton and
          // the sm100a GPU paths; the reference matches that quantization.
          const float coef = __bfloat162float(__float2bfloat16(P * (dP - delta)));
          for (int e = 0; e < hd; ++e)
            hdQ[(qp * H + h) * hd + e] += coef * __bfloat162float(hK[(kp * H + h) * hd + e]);
          ++idx;
        }
      }
      for (int e = 0; e < hd; ++e) hdQ[(qp * H + h) * hd + e] *= sm_scale;
    }
  }

  if (file_err) {
    file_err[0] = lse_err;
    file_err[1] = o_err;
  }

  #pragma omp parallel for schedule(dynamic)
  for (int gkb = 0; gkb < B * H * num_blocks; ++gkb) {
    const int kb = gkb % num_blocks;
    const int bh = gkb / num_blocks;
    const int h  = bh % H;
    const int b  = bh / H;
    for (int qi_list = k2q.offset[gkb]; qi_list < k2q.offset[gkb + 1]; ++qi_list) {
      const int mtile = k2q.q_blocks[qi_list];
      for (int qi = 0; qi < BLOCK; ++qi) {
        const long qp     = (long)b * S + (long)mtile * BLOCK + qi;
        const long row    = (long)bh * S + (long)mtile * BLOCK + qi;
        const float M_row = hM[row];
        const float delta = hDelta[row];
        for (int kj = 0; kj < BLOCK; ++kj) {
          const long kp = (long)b * S + (long)kb * BLOCK + kj;
          float dot = 0.f, dP = 0.f;
          for (int e = 0; e < hd; ++e) {
            dot += __bfloat162float(hQ[(qp * H + h) * hd + e]) *
                   __bfloat162float(hK[(kp * H + h) * hd + e]);
            dP += __bfloat162float(hdO[(qp * H + h) * hd + e]) *
                  __bfloat162float(hV[(kp * H + h) * hd + e]);
          }
          const float P = exp2f(dot * sm_scale * LOG2E - M_row);
          // P feeds dV as bf16 (MMA operand) and dS feeds dK as bf16, matching
          // the Triton and sm100a GPU quantization points.
          const float Pq   = __bfloat162float(__float2bfloat16(P));
          const float coef = __bfloat162float(__float2bfloat16(P * (dP - delta)));
          for (int e = 0; e < hd; ++e) {
            hdV[(kp * H + h) * hd + e] += Pq * __bfloat162float(hdO[(qp * H + h) * hd + e]);
            hdK[(kp * H + h) * hd + e] += coef * __bfloat162float(hQ[(qp * H + h) * hd + e]);
          }
        }
      }
    }
    const float sm_scale2 = 1.0f / sqrtf((float)hd);
    for (int kj = 0; kj < BLOCK; ++kj) {
      const long kp = (long)b * S + (long)kb * BLOCK + kj;
      for (int e = 0; e < hd; ++e) hdK[(kp * H + h) * hd + e] *= sm_scale2;
    }
  }
}

// Deterministic fill in [-1, 1) (same hash as the forward bench).
static void fillr(__nv_bfloat16* h, long n, unsigned seed) {
  for (long i = 0; i < n; ++i) {
    uint32_t x = (uint32_t)i * 2654435761u + seed * 40503u + 0x9e3779b9u;
    x ^= x >> 15;
    x *= 2246822519u;
    x ^= x >> 13;
    x *= 3266489917u;
    x ^= x >> 16;
    h[i] = __float2bfloat16((float)(x % 2039u) / 1019.5f - 1.0f);
  }
}

struct Sh {
  int B, H, num_blocks, topk, hd;
  const char* lab;
};

static void run(const Sh& sh) {
  const int B = sh.B, H = sh.H, num_blocks = sh.num_blocks, topk = sh.topk, hd = sh.hd;
  const int S                   = num_blocks * BLOCK;
  const int max_kv              = topk;
  const long tq                 = (long)B * S;
  const int num_global_q_blocks = B * H * num_blocks;
  const size_t q2k_entries      = (size_t)num_global_q_blocks * (size_t)max_kv;
  if (q2k_entries > (size_t)INT_MAX) {
    fprintf(stderr, "metadata overflow: %zu q2k entries exceed int32 offsets\n", q2k_entries);
    exit(EXIT_FAILURE);
  }

  printf("  [%-9s H%-2d num_blocks%-3d topk%-3d S%d blk%d] N_q=%ld\n", sh.lab, H, num_blocks, topk,
         S, BLOCK, tq);

  std::vector<__nv_bfloat16> hQ(tq * H * hd), hK(tq * H * hd), hV(tq * H * hd), hdO(tq * H * hd);
  const char* load_npy = getenv("LOAD_NPY");
  if (load_npy) {
    const std::string d(load_npy);
    auto ld = [&](const char* nm, std::vector<__nv_bfloat16>& h) {
      char p[64];
      snprintf(p, sizeof p, "/%s_S%d.npy", nm, S);
      auto bits = npy_load_vec<uint16_t>(d + p);
      if (bits.size() != h.size()) {
        fprintf(stderr, "LOAD_NPY: %s size %zu != %zu\n", nm, bits.size(), h.size());
        exit(1);
      }
      memcpy(h.data(), bits.data(), h.size() * 2);
    };
    ld("q", hQ);
    ld("k", hK);
    ld("v", hV);
    ld("do", hdO);
  } else {
    auto FILL = fillr;
    FILL(hQ.data(), hQ.size(), 11);
    FILL(hK.data(), hK.size(), 22);
    FILL(hV.data(), hV.size(), 33);
    FILL(hdO.data(), hdO.size(), 44);
  }

  // q2k index: LOAD_NPY head-independent [num_blocks, topk] broadcast, or topk DISTINCT
  // block ids per (b,h,mtile) via partial Fisher-Yates (same knobs as the forward bench).
  std::vector<int> hq2k_idx(q2k_entries, 0);
  std::vector<int> hq2k_num(num_global_q_blocks, topk);
  if (load_npy) {
    char p[64];
    snprintf(p, sizeof p, "/idx_S%d_blk%d.npy", S, BLOCK);
    auto idx = npy_load_vec<int32_t>(std::string(load_npy) + p);
    if (idx.size() != (size_t)num_blocks * topk) {
      fprintf(stderr, "LOAD_NPY: idx size %zu != %d\n", idx.size(), num_blocks * topk);
      exit(1);
    }
    for (int global_mtile = 0; global_mtile < num_global_q_blocks; ++global_mtile) {
      const int mtile = global_mtile % num_blocks;
      for (int i = 0; i < topk; ++i)
        hq2k_idx[(size_t)global_mtile * max_kv + i] = idx[(size_t)mtile * topk + i];
    }
  } else {
    std::vector<int> perm(num_blocks);
    for (int global_mtile = 0; global_mtile < num_global_q_blocks; ++global_mtile) {
      for (int i = 0; i < num_blocks; ++i) perm[i] = i;
      uint32_t st = (uint32_t)global_mtile * 2654435761u + 12345u;
      for (int i = 0; i < topk; ++i) {
        st ^= st << 13;
        st ^= st >> 17;
        st ^= st << 5;
        const int j                                 = i + (int)(st % (uint32_t)(num_blocks - i));
        const int t                                 = perm[i];
        perm[i]                                     = perm[j];
        perm[j]                                     = t;
        hq2k_idx[(size_t)global_mtile * max_kv + i] = perm[i];
      }
    }
  }

  const KvToQ k2q = invert_q2k(hq2k_idx.data(), hq2k_num.data(), B, H, num_blocks, max_kv);
  const K2qPadded k2q_padded = build_k2q_padded(k2q, B, H, num_blocks);
  {
    const long usum = k2q.offset[B * H * num_blocks];
    printf("  q-lists: items=%d entries=%ld steps/item=%.1f max_q_blocks=%d\n", B * H * num_blocks,
           usum, (double)usum / (B * H * num_blocks), k2q_padded.max_q_blocks);
  }
  {
    int cmin = INT_MAX, cmax = 0, zeros = 0;
    long csum = 0;
    for (int c : k2q.count) {
      cmin = std::min(cmin, c);
      cmax = std::max(cmax, c);
      csum += c;
      if (c == 0) ++zeros;
    }
    // kv-block count == q-block count here (square S x S block grid).
    printf("  k2q: kv_blocks=%d count min=%d max=%d mean=%.2f zero-count=%d\n", num_global_q_blocks,
           cmin, cmax, (double)csum / num_global_q_blocks, zeros);
  }

  const char* dump_prefix     = getenv("DUMP_BWD");
  const char* cpu_env         = getenv("CPU_REF");
  const double selected_pairs = (double)B * H * num_blocks * topk * (double)BLOCK * BLOCK;
  bool run_cpu = cpu_env ? (atoi(cpu_env) != 0) : (dump_prefix != nullptr || selected_pairs <= 4e6);
  if (dump_prefix && !run_cpu) printf("  DUMP_BWD set but CPU_REF=0: no reference to dump\n");

  std::vector<float> hO(tq * H * hd), hdQ(tq * H * hd), hdK(tq * H * hd), hdV(tq * H * hd);
  std::vector<float> hM((size_t)B * H * S), hDelta((size_t)B * H * S);
  std::vector<float> file_O, file_M;
  if (load_npy) {
    char p[64];
    snprintf(p, sizeof p, "/o_S%d_blk%d.npy", S, BLOCK);
    const auto o_bits = npy_load_vec<uint16_t>(std::string(load_npy) + p);
    snprintf(p, sizeof p, "/lse_S%d_blk%d.npy", S, BLOCK);
    file_M = npy_load_vec<float>(std::string(load_npy) + p);
    if (o_bits.size() != hO.size() || file_M.size() != hM.size()) {
      fprintf(stderr, "LOAD_NPY: forward state size mismatch; rerun gen_inputs.py\n");
      exit(1);
    }
    file_O.resize(o_bits.size());
    for (size_t i = 0; i < o_bits.size(); ++i) {
      __nv_bfloat16 o;
      memcpy(&o, &o_bits[i], 2);
      file_O[i] = __bfloat162float(o);
    }
    printf("  fwd state: o/lse_S%d_blk%d.npy\n", S, BLOCK);
  } else if (!run_cpu) {
    if (cpu_env) {
      fprintf(stderr, "CPU_REF=0 needs LOAD_NPY: perf runs use the real forward O/LSE\n");
      exit(1);
    }
    printf("  skipped: above the CPU_REF size limit and no LOAD_NPY forward state\n");
    return;
  }
  if (!run_cpu) {
    hO = std::move(file_O);
    hM = std::move(file_M);
  }
  if (run_cpu) {
    const auto t0 = std::chrono::steady_clock::now();
    float file_err[2] = {0.f, 0.f};
    cpu_vsa_bwd_ref(hQ.data(), hK.data(), hV.data(), hdO.data(), hO.data(), hM.data(),
                    hDelta.data(), hdQ.data(), hdK.data(), hdV.data(), B, H, S, hd, num_blocks,
                    max_kv, hq2k_idx.data(), hq2k_num.data(), k2q,
                    load_npy ? file_O.data() : nullptr, load_npy ? file_M.data() : nullptr,
                    file_err);
    const auto t1       = std::chrono::steady_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  cpu ref: %.1f ms (fwd O/M + Delta + dQ + k2q dK/dV, fp32)\n", cpu_ms);
    if (load_npy) {
      const bool fwd_ok = file_err[0] < 1e-4f && file_err[1] <= 1.f;
      printf("  fwd state vs cpu: lse max|diff|=%.2e  o max|diff|/(|o|/128+1e-4)=%.2f  %s\n",
             file_err[0], file_err[1], fwd_ok ? "OK" : "FAIL");
      if (!fwd_ok) exit(1);
    }

    if (dump_prefix) {
      const std::string prefix(dump_prefix);
      npy_save_f32(prefix + "_dq.npy", hdQ.data(), {tq, (long)H, (long)hd});
      npy_save_f32(prefix + "_dk.npy", hdK.data(), {tq, (long)H, (long)hd});
      npy_save_f32(prefix + "_dv.npy", hdV.data(), {tq, (long)H, (long)hd});
      npy_save_f32(prefix + "_M.npy", hM.data(), {(long)B * H, (long)S});
      npy_save_f32(prefix + "_delta.npy", hDelta.data(), {(long)B * H, (long)S});
      printf(
          "  dump: %s_{dq,dk,dv}.npy [%ld,%d,%d] f32; %s_{M,delta}.npy [%d,%d] f32"
          " ([H,S] at B=1; M log2-domain)\n",
          dump_prefix, tq, H, hd, dump_prefix, B * H, S);
    }
  } else {
    printf("  cpu ref: skipped (%.0f selected pairs; CPU_REF=1 to force)\n", selected_pairs);
  }

  // GPU backward: stream-chained preprocess (Delta + dqaccum zeroing) -> non-persistent main
  // kernel -> postprocess (bf16 dQ).
  {
    const long elems = tq * H * hd;
    __nv_bfloat16 *dQg, *dKg, *dVg, *dDOg, *dOg, *dDKout, *dDVout, *dDQout;
    float *dMg, *dDeltag, *dDQA;
    int *dK2qIdx, *dK2qNum;
    CUDA_CHECK(cudaMalloc(&dQg, elems * 2));
    CUDA_CHECK(cudaMalloc(&dKg, elems * 2));
    CUDA_CHECK(cudaMalloc(&dVg, elems * 2));
    CUDA_CHECK(cudaMalloc(&dDOg, elems * 2));
    CUDA_CHECK(cudaMalloc(&dOg, elems * 2));
    CUDA_CHECK(cudaMalloc(&dDKout, elems * 2));
    CUDA_CHECK(cudaMalloc(&dDVout, elems * 2));
    CUDA_CHECK(cudaMalloc(&dDQout, elems * 2));
    CUDA_CHECK(cudaMalloc(&dMg, (size_t)B * H * S * 4));
    CUDA_CHECK(cudaMalloc(&dDeltag, (size_t)B * H * S * 4));
    CUDA_CHECK(cudaMalloc(&dDQA, (size_t)B * H * S * hd * 4));
    CUDA_CHECK(cudaMalloc(&dK2qIdx, k2q_padded.idx.size() * 4));
    CUDA_CHECK(cudaMalloc(&dK2qNum, k2q_padded.num.size() * 4));
    auto upload_act = [&](__nv_bfloat16* dst, const std::vector<__nv_bfloat16>& host) {
      const std::vector<__nv_bfloat16> permuted = to_device_layout(host, B, H, S, hd);
      CUDA_CHECK(cudaMemcpy(dst, permuted.data(), elems * 2, cudaMemcpyHostToDevice));
    };
    upload_act(dQg, hQ);
    upload_act(dKg, hK);
    upload_act(dVg, hV);
    upload_act(dDOg, hdO);
    {
      std::vector<__nv_bfloat16> hObf(elems);
      for (long i = 0; i < elems; ++i) hObf[i] = __float2bfloat16(hO[i]);
      upload_act(dOg, hObf);
    }
    CUDA_CHECK(cudaMemcpy(dMg, hM.data(), (size_t)B * H * S * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK2qIdx, k2q_padded.idx.data(), k2q_padded.idx.size() * 4,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK2qNum, k2q_padded.num.data(), k2q_padded.num.size() * 4,
                          cudaMemcpyHostToDevice));

    VsaBwdArgs args;
    args.q                     = dQg;
    args.k                     = dKg;
    args.v                     = dVg;
    args.dout                  = dDOg;
    args.dqaccum               = dDQA;
    args.dk                    = dDKout;
    args.dv                    = dDVout;
    args.dq                    = dDQout;
    args.lse_rows              = dMg;
    args.delta_rows            = dDeltag;
    args.k2q_idx               = dK2qIdx;
    args.k2q_num               = dK2qNum;
    args.max_q_blocks          = k2q_padded.max_q_blocks;
    args.num_samples           = B;
    args.num_heads             = H;
    args.seqlen                = S;
    args.num_kv_blocks_per_seq = num_blocks;
    args.sm_scale              = 1.0f / sqrtf((float)hd);

    auto run_once = [&]() {
      CUDA_CHECK(launch_vsa_bwd_preprocess(dOg, dDOg, dDeltag, dDQA, B, H, S, 0));
      CUDA_CHECK(launch_vsa_bwd_sm100a(args, 0));
      CUDA_CHECK(launch_vsa_bwd_postprocess(dDQA, dDQout, B, H, S, args.sm_scale, 0));
    };
    run_once();
    CUDA_CHECK(cudaDeviceSynchronize());

    if (run_cpu || getenv("DUMP_BWD_GPU")) {
      std::vector<float> gDelta((size_t)B * H * S);
      std::vector<__nv_bfloat16> gDK(elems), gDV(elems), gDQ(elems);
      CUDA_CHECK(cudaMemcpy(gDelta.data(), dDeltag, gDelta.size() * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(gDQ.data(), dDQout, elems * 2, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(gDK.data(), dDKout, elems * 2, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(gDV.data(), dDVout, elems * 2, cudaMemcpyDeviceToHost));
      gDQ = from_device_layout(gDQ, B, H, S, hd);
      gDK = from_device_layout(gDK, B, H, S, hd);
      gDV = from_device_layout(gDV, B, H, S, hd);

      auto rel_norm = [](const float* ref, const float* got, long n) {
        double mref = 0, mdiff = 0;
        for (long i = 0; i < n; ++i) {
          const double r = fabs((double)ref[i]);
          const double d = fabs((double)ref[i] - got[i]);
          if (r > mref) mref = r;
          if (d > mdiff) mdiff = d;
        }
        return mref > 0 ? mdiff / mref : mdiff;
      };
      double dmax = 0;
      for (long i = 0; i < (long)gDelta.size(); ++i)
        dmax = std::max(dmax, fabs((double)gDelta[i] - hDelta[i]));
      // dq: the postprocess already unscrambled + scaled + rounded to bf16.
      std::vector<float> gdq(elems);
      for (long i = 0; i < elems; ++i) gdq[i] = __bfloat162float(gDQ[i]);
      std::vector<float> gdk(elems), gdv(elems);
      for (long i = 0; i < elems; ++i) {
        gdk[i] = __bfloat162float(gDK[i]);
        gdv[i] = __bfloat162float(gDV[i]);
      }
      const double rq = rel_norm(hdQ.data(), gdq.data(), elems);
      const double rk = rel_norm(hdK.data(), gdk.data(), elems);
      const double rv = rel_norm(hdV.data(), gdv.data(), elems);
      if (run_cpu && getenv("VERIFY_ARGMAX")) {
        long am     = 0;
        double best = -1;
        for (long i = 0; i < elems; ++i) {
          const double d = fabs((double)hdQ[i] - gdq[i]);
          if (d > best) {
            best = d;
            am   = i;
          }
        }
        const int d_  = (int)(am % hd);
        const int h_  = (int)((am / hd) % H);
        const long t_ = am / hd / H;
        printf("  dq argmax: t=%ld (blk %ld, row %ld) h=%d d=%d ref=%.6f got=%.6f\n", t_,
               t_ / Q_TILE, t_ % Q_TILE, h_, d_, hdQ[am], gdq[am]);
        long over1e3 = 0, over3e4 = 0;
        double sum = 0;
        for (long i = 0; i < elems; ++i) {
          const double d = fabs((double)hdQ[i] - gdq[i]);
          sum += d;
          if (d > 1e-3) ++over1e3;
          if (d > 3e-4) ++over3e4;
        }
        printf("  dq diff: mean=%.2e  >3e-4: %ld/%ld  >1e-3: %ld\n", sum / elems, over3e4, elems,
               over1e3);
        int shown = 0;
        for (long i = 0; i < elems && shown < 12; ++i) {
          const double d = fabs((double)hdQ[i] - gdq[i]);
          if (d > 3e-4) {
            printf("    t=%ld h=%ld d=%ld ref=%.6f got=%.6f\n", i / hd / H, (i / hd) % H, i % hd,
                   hdQ[i], gdq[i]);
            ++shown;
          }
        }
      }
      if (const char* gp = getenv("DUMP_BWD_GPU")) {
        const std::string prefix(gp);
        npy_save_f32(prefix + "_dq.npy", gdq.data(), {tq, (long)H, (long)hd});
        npy_save_f32(prefix + "_dk.npy", gdk.data(), {tq, (long)H, (long)hd});
        npy_save_f32(prefix + "_dv.npy", gdv.data(), {tq, (long)H, (long)hd});
      }
      // dq gate 8e-3: the CPU-ref-vs-torch-fp32 noise floor from the
      // production bf16 quantization points is ~1.9-2.7e-3 (oracle_bwd.py,
      // 2026-08-25), GPU-vs-CPU can legitimately reach ~2x that, and the
      // bf16-rounded dq output adds its own rounding on top.
      if (run_cpu) {
        const double rq_gate = 8e-3;
        const bool pass      = dmax < 1e-4 && rq < rq_gate && rk < 8e-3 && rv < 8e-3;
        printf("  gpu verify: delta max|diff|=%.2e  dq rel=%.2e  dk rel=%.2e  dv rel=%.2e  %s\n",
               dmax, rq, rk, rv, pass ? "OK" : "FAIL");
        if (!pass) exit(1);
      }

      if (const char* sn = getenv("STRESS_N")) {
        const int n = atoi(sn);
        std::vector<__nv_bfloat16> rDK(elems), rDV(elems), rDQ(elems);
        bool ok = true;
        for (int it = 0; it < n && ok; ++it) {
          run_once();
          CUDA_CHECK(cudaDeviceSynchronize());
          CUDA_CHECK(cudaMemcpy(rDK.data(), dDKout, elems * 2, cudaMemcpyDeviceToHost));
          CUDA_CHECK(cudaMemcpy(rDV.data(), dDVout, elems * 2, cudaMemcpyDeviceToHost));
          CUDA_CHECK(cudaMemcpy(rDQ.data(), dDQout, elems * 2, cudaMemcpyDeviceToHost));
          rDK       = from_device_layout(rDK, B, H, S, hd);
          rDV       = from_device_layout(rDV, B, H, S, hd);
          rDQ       = from_device_layout(rDQ, B, H, S, hd);
          ok        = memcmp(rDK.data(), gDK.data(), elems * 2) == 0 &&
                      memcmp(rDV.data(), gDV.data(), elems * 2) == 0;
          double mq = 0;
          for (long i = 0; i < elems; ++i)
            mq = std::max(mq, fabs((double)__bfloat162float(rDQ[i]) - __bfloat162float(gDQ[i])));
          if (mq > 1e-2) ok = false;  // reduce-add order x bf16 rounding
        }
        printf("  stress x%d: %s\n", n, ok ? "OK (dk/dv bitwise, dq stable)" : "FAIL");
        if (!ok) exit(1);
      }
    }

    if (block_sparse_bwd_bf16_benchmark::enabled()) {
      const auto options = block_sparse_bwd_bf16_benchmark::options_from_env();
      const double ms = block_sparse_bwd_bf16_benchmark::measure(run_once, options);
      const double tflops = block_sparse_bwd_bf16_benchmark::tflops(hd, selected_pairs, ms);
      printf("  gpu bwd: %.4f ms  %.1f TFLOPS (bwd 2.5x sel; pre+main+post)\n", ms, tflops);
    }

#ifdef WARP_PROF
    {
      WpBuffer wp = wp_alloc(dim3((unsigned)num_blocks, (unsigned)H, 1));
      run_once();
      CUDA_CHECK(cudaDeviceSynchronize());
      wp_readback(wp);
      const char* roles[16] = {"red", "red", "red", "red", "cmp", "cmp",  "cmp", "cmp",
                               "cmp", "cmp", "cmp", "cmp", "mma", "load", "rly", "emp"};
      printf("  WARP_PROF block %u:\n", wp.view_block);
      wp_print_busy(wp, roles, 16, wp.view_block);
      wp_dump_raw(wp, "warp_raw_vsa_bwd_blk128_2sm.bin.gz", wp.view_block, 2);
      wp_free(wp);
    }
#endif

    cudaFree(dQg);
    cudaFree(dKg);
    cudaFree(dVg);
    cudaFree(dDOg);
    cudaFree(dOg);
    cudaFree(dDKout);
    cudaFree(dDVout);
    cudaFree(dDQout);
    cudaFree(dMg);
    cudaFree(dDeltag);
    cudaFree(dDQA);
    cudaFree(dK2qIdx);
    cudaFree(dK2qNum);
  }
}

int main() {
  CUDA_CHECK(cudaFree(0));
  printf(
      "VSA block-sparse BACKWARD bench bf16 (blk128-native, FA4-aligned) sm_100a\n"
      "2CTA non-persistent 16-warp kernel; pre+main+post (block=%d)\n"
      "=====================================\n",
      BLOCK);

  // shapes: {B, H, num_blocks, topk, hd, label} (as the forward bench).
  Sh shapes[] = {
      {1, 4, 8, 4, 128, "small"},
      {1, 16, 32, 8, 128, "fastvideo"},
      {1, 8, 64, 16, 128, "25pct"},
      {1, 8, 2048, 512, 128, "25pct-262k"},
      {1, 8, 4096, 1024, 128, "25pct-524k"},
  };
  constexpr int default_shapes = 3;
  constexpr int num_shapes     = sizeof(shapes) / sizeof(shapes[0]);

  if (const char* s = getenv("SHAPE")) {
    const int shape_index = atoi(s);
    if (shape_index < 0 || shape_index >= num_shapes) {
      fprintf(stderr, "SHAPE=%d out of range [0, %d)\n", shape_index, num_shapes);
      return 1;
    }
    Sh sh = shapes[shape_index];
    if (getenv("BATCH")) sh.B = atoi(getenv("BATCH"));
    if (getenv("HEADS")) sh.H = atoi(getenv("HEADS"));
    if (getenv("NB")) sh.num_blocks = atoi(getenv("NB"));
    if (getenv("TOPK")) sh.topk = atoi(getenv("TOPK"));
    sh.lab = "custom";
    run(sh);
    return 0;
  }
  const int B = getenv("BATCH") ? atoi(getenv("BATCH")) : 1;
  for (int i = 0; i < default_shapes; ++i) {
    Sh sh = shapes[i];
    sh.B  = B;
    run(sh);
  }
  return 0;
}
