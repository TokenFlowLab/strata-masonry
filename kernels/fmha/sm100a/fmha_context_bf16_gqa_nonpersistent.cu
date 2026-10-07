// fmha_context_bf16_gqa_nonpersistent.cu -- K2 FMHA context BF16, sm_100a.
// 
// ** Terminology **
//   D (HEAD_DIM=128)   head dim: contracted in BMM1 (Q@K^T), output width of BMM2 (P@V).
//   gqa_group_size=8   q-heads sharing one kv-head (= HQ/HK).
//   K-tile = 128 keys  one K-block of the context dim; BMM1 -> 128 score cols S.
//   V-tile = 128 vals  V^T, paired with a K-tile, feeds BMM2. One ring slot = 1 K- or 1 V-tile.
//   M-tile = 128 rows  the MMA M dim; pack-GQA folds 8 q-heads x 16 q-tokens (row = qh + 8*q_pos).
//                      Owns its S/O in TMEM + softmax state. M_TILES_PER_CTA=2 (overlap to pipeline).
//   Q-tile = 32 q-tokens of one sequence/kv-head = the 2 M-tiles (q_tile_per_mtile=16 each).
//   work tile = one CTA's job = (kv_head, Q-tile). Non-persistent: 1 CTA per work tile.
//
// ** Data Layout **
// pack-GQA: each M-tile is 128 packed rows = gqa_group_size(8) q-heads x 16 q-tokens of one
// kv-head (row = qh + 8*q_pos), so the 8 q-heads sharing a kv-head reuse every K/V tile.
// One CTA runs 2 M-tiles (A,B) = the two adjacent 16-q-token halves of one Q-tile (SAME
// sequence, SAME kv-head); they share the K/V ring but keep independent online-softmax state
// and O accumulators.
//
// ** Execution flow **
//
// wait[B] = consume mbarrier B (block until ready); arrive[B] = signal/produce B. MMA's arrives
// are async tcgen05_commit on the in-order pipe; load's are arrive_expect_tx (TMA completes them).
//
// Load warp -- TMA producer. Descending KV (g=0 = diagonal, first).
//   mainloop (per K-tile g):
//     K(g):
//       wait[empty_bar(slot)]
//       arrive_tx[full_bar(slot)] + TMA K            (2 sub-tiles)
//     Q (g==0 only), per i:
//       wait[empty_bar_q(i)]
//       arrive_tx[full_bar_q(i)] + 3D TMA pack-GQA Q
//     V(g):
//       wait[empty_bar(slot)]
//       arrive_tx[full_bar(slot)] + TMA V^T          (ring order K0,V0,K1,V1,..)
//
// MMA warp drives every tcgen05 MMA. The 2 M-tiles (i=0,1) share the K/V ring but each has its
// own S and O in TMEM. For one work-tile it loops over K-tiles g (BMM1 = Q@K -> S, BMM2 = P@V ->
// O), handling both M-tiles i=0,1:
//   prologue (BMM1 of K-tile 0):
//     wait[full_bar(K0)]
//     per i:
//       wait[full_bar_q(i)]
//       BMM1(g=0,i) -> S
//       arrive[full_bar_spo(i)]
//     arrive[empty_bar(K0)]                          (free K0 ring slot)
//   mainloop (g = 0 .. last-1):
//     wait[full_bar(V g)]
//     per i:
//       wait[empty_bar_spo(i)]
//       BMM2(g) P@V += O                             (split-P: mid-BMM2 wait[full_bar_p_last(i)])
//       i==0: wait[full_bar(K g+1)]
//       BMM1(g+1) -> S                               (BMM1-ahead)
//       arrive[full_bar_spo(i)]
//       i==last: arrive[empty_bar(V g)]
//     arrive[empty_bar(K g+1)]                       (free K g+1 after BMM1-ahead)
//   epilogue (BMM2 of last K-tile):
//     wait[full_bar(V last)]
//     per i:
//       wait[empty_bar_spo(i)]
//       BMM2(last) P@V += O
//       arrive[full_bar_o_acc(i)]                    (final O)
//     arrive[empty_bar(V last)]
//     post-loop: arrive[empty_bar_q(i)]              (free Q; non-persistent, no consumer)
//
// Softmax warps (group 0 = M-tile 0, group 1 = M-tile 1; each warp owns a 32-row band).
// reverse-K online softmax.
//   prologue:
//     wait[empty_bar_alpha_and_l]                    (acquire scale slot)
//   mainloop (per K-tile g):
//     wait[full_bar_spo]
//     tcgen05.ld S row
//     write alpha;
//     arrive[full_bar_alpha]            (-> correction rescale)
//     compute P (exp2)
//     store P to TMEM (P aliases S); 
//     arrive[empty_bar_spo]   (-> MMA BMM2)
//     l_run update
//     wait[empty_bar_alpha_and_l]                    (re-acquire for next write)
//   epilogue:
//     write final l; arrive[full_bar_l]              (-> correction normalize)
//
// Correction warps (the O epilogue runs here -- FA4 use_correction_warps_for_epi).
//   prologue:
//     arrive[empty_bar_spo]
//     arrive[empty_bar_alpha_and_l]   (prime the return barriers)
//     g==0, per i: 
//	   wait[full_bar_alpha] -> arrive[empty_bar_alpha_and_l]   (no prior O; no rescale)
//   mainloop (g = 1 .. last):
//     per i:
//       wait[full_bar_alpha]
//       read alpha; arrive[empty_bar_alpha_and_l]
//       O*=alpha in TMEM
//       arrive[empty_bar_spo]                        (-> MMA BMM2 g)
//   epilogue (per i):
//     wait[full_bar_l]
//     read l; arrive[empty_bar_alpha_and_l]; inv_l = rcp(l)
//     wait[full_bar_o_acc]                           (final O)
//     O*=inv_l; 
//     arrive[empty_bar_spo]
//     Store O to GMEM.
//
// Epilogue/sched warps: reserved-idle -- setmaxnreg_dec<56> then return (frees registers
//   for the softmax warps' inc<192>). No TMA-store-O / CLC scheduler implemented yet.
//
// ** Memory Layout **
// SMEM: sQ0, sQ1 | shared K/V ring (NUM_KV_STAGES slots; entry 2g = K(g), 2g+1 = V(g),
//       one slot holds one K- or one V-tile) | sO (single [M_TILE][HEAD_DIM] bf16, REUSED by
//       both M-tiles for epilogue O re-tile staging) | mbarriers | tmem_slot | alpha_and_l_smem[2][M_TILE].
// TMEM (512 cols): S0, S1 (128 each, FP32, single-buffered; P aliases the S slot) |
//       O0, O1 (128 each). The 2 M-tiles provide the overlap a double-buffered S would.
//
// ** Barrier Contract **
//
// Swimlane: 4 warp columns, time flows DOWN, each row is one mbarrier handoff. The arrow tail (+)
// is the arrive; the head (>/<) is the warp whose wait it unblocks. Example = 2 M-tiles (Q0,Q1)
// x 3 K-tiles (K0,K1,K2). empty_bar_al = empty_bar_alpha_and_l. empty_bar_spo is arrived by BOTH
// softmax (after P stored) and correction (after O read/rescaled).
//
// LOAD                       MMA                     SOFTMAX                    CORR
//   prime:
//   |                         <-------------- empty_bar_spo[Q0,Q1] ---------------+
//   |                         |                         <-- empty_bar_al[Q0,Q1] --+
//   K0:
//   +----- full_bar[K0] ------>                         |                         |
//   +--- full_bar_q[Q0,Q1] --->                         |                         |
//   |                         +-- full_bar_spo[Q0,Q1] -->                         |
//   <----- empty_bar[K0] -----+                         |                         |
//   |                         |                         +- full_bar_alpha[Q0,Q1] ->
//   |                         |                         <-- empty_bar_al[Q0,Q1] --+
//   |                         <- empty_bar_spo[Q0,Q1] --+                         |
//   K1:
//   +----- full_bar[V0] ------>                         |                         |
//   <----- empty_bar[V0] -----+                         |                         |
//   +----- full_bar[K1] ------>                         |                         |
//   |                         +-- full_bar_spo[Q0,Q1] -->                         |
//   <----- empty_bar[K1] -----+                         |                         |
//   |                         |                         +- full_bar_alpha[Q0,Q1] ->
//   |                         <-------------- empty_bar_spo[Q0,Q1] ---------------+
//   |                         |                         <-- empty_bar_al[Q0,Q1] --+
//   |                         <- empty_bar_spo[Q0,Q1] --+                         |
//   K2:
//   +----- full_bar[V1] ------>                         |                         |
//   <----- empty_bar[V1] -----+                         |                         |
//   +----- full_bar[K2] ------>                         |                         |
//   |                         +-- full_bar_spo[Q0,Q1] -->                         |
//   <----- empty_bar[K2] -----+                         |                         |
//   |                         |                         +- full_bar_alpha[Q0,Q1] ->
//   |                         <-------------- empty_bar_spo[Q0,Q1] ---------------+
//   |                         |                         <-- empty_bar_al[Q0,Q1] --+
//   |                         <- empty_bar_spo[Q0,Q1] --+                         |
//   store O:
//   +----- full_bar[V2] ------>                         |                         |
//   <----- empty_bar[V2] -----+                         |                         |
//   |                         +-------------- full_bar_o_acc[Q0,Q1] -------------->
//   |                         |                         +--- full_bar_l[Q0,Q1] --->
//   |                         <-------------- empty_bar_spo[Q0,Q1] ---------------+
//   |                         |                         <-- empty_bar_al[Q0,Q1] --+
//   tail drain:
//   <----- empty_bar[x NUM_KV_STAGES] -----+                         |                         |
//   <--- empty_bar_q[Q0,Q1] (elect lane) --+                         |                         |
//   |                         <---------- empty_bar_spo[Q0,Q1] prime -+ (softmax +128)          |
//   |                         |                         <-- empty_bar_al[Q0,Q1] --+
// (the "store O" empty_bar_spo/empty_bar_al arrives close the contract: correction frees the O
//  TMEM slot (after PASS1 reads O) and the alpha/l slot (after reading l). Non-persistent: dead
//  tail -- warps just exit, leftover arrives harmless. Persistent: they ARE the next work-tile's
//  "prime:", so the chart loops closed -- OR a self-draining variant waits them out at the tail:
//  load drains NUM_KV_STAGES empty_bar (all lanes) + M_TILES empty_bar_q (elect lane only, since
//  one lane advances the Q tracker); softmax drains empty_bar_al and posts a +128 closing prime on
//  empty_bar_spo so correction's dangling 128 completes the 256 the MMA warp drains (prime MUST be
//  after the TMEM-teardown bar.sync, else it races MMA's last BMM2 acquire). The sO/TMA-store
//  epilogue variant adds correction draining empty_bar_o_epi <- EPI; this varlen file has no o_epi.)
//
// Barriers (producer -> consumer; *_full = data ready, *_empty = buffer free):
//   full_bar / empty_bar  : load <-> MMA       K/V tile loaded / ring slot free
//   full_bar_q / empty_bar_q : load <-> MMA    Q loaded / Q read out
//   full_bar_spo          : MMA -> softmax      S ready (after BMM1)
//   empty_bar_spo         : softmax + correction -> MMA   S read + P[0:96] written + prev O read out
//   full_bar_p_last       : softmax -> MMA      P[96:128] written (split-P at k=96)
//   full_bar_o_acc        : MMA -> correction   final O ready
//   full_bar_alpha        : softmax -> correction   per-block rescale alpha
//   full_bar_l            : softmax -> correction   final row-sum l
//   empty_bar_alpha_and_l : correction -> softmax   alpha/l slot free. Bounded stats
//      pipe: without this return path softmax laps correction and the 1-bit phase
//      desyncs -> hang at speed (serialized runs stay in lockstep and hide it).
//
// Order is guaranteed by the single in-order tcgen05 pipe + these acquires:
//   * BMM1(g+1) is issued after BMM2(g), so it needs no explicit S barrier:
//     S(g-1) is freed by the empty_bar_spo acquire that precedes BMM2(g-1) on the pipe.
//   * BMM2(g) acquires empty_bar_spo (P first 96 cols ready AND prev-tile O read out,
//     so the O slot is safe to reuse), then waits full_bar_p_last for the last 32 P cols.
//   * correction's per-block O rescale needs no signal: softmax's full_bar_alpha
//     implies BMM1(g) issued -> in-order pipe -> BMM2(g-1)'s O write done. Only
//     the final O is signaled, once per tile, via full_bar_o_acc.
//
// ** Work decomposition (NON-PERSISTENT) **
// One CTA = one work tile = (kv_head, Q-tile)  (FA4 SingleTileVarlenScheduler).
// Varlen means the host can't know the exact #Q-tiles without a per-sample loop, so it launches
// a tight UPPER BOUND of CTAs and lets the extras exit:
//     grid.x = total_blocks_max * num_kv_heads      (total_blocks_max = upper bound on the
//                                                    #Q-tiles summed over samples; ~2264 for UND)
// The EXACT count is num_qtiles_real = qtile_prefix[num_samples], where qtile_prefix is a
// host-built prefix sum: qtile_prefix[s] = sum over samples < s of ceil(seqlen_q[s] / q_tokens_per_cta).
// Each CTA decodes its work tile from blockIdx.x (kv_head is the innermost flatten dim):
//     h_kv = blockIdx.x % num_kv_heads
//     gq   = blockIdx.x / num_kv_heads                       (global Q-tile index, all samples)
//     if (gq >= num_qtiles_real) return                      (slack CTA -- nothing to do)
//     sample      = largest s with qtile_prefix[s] <= gq     (binary search the prefix sum)
//     q_tile_id   = (#Q-tiles in sample - 1) - (gq - qtile_prefix[sample])   (LPT: heaviest first)
//     q_tile_base = q_tile_id * q_tokens_per_cta             (first q-token this CTA owns)
// LPT (FA4 lpt=True): within a sample the last Q-tile (most causal keys = heaviest) runs first,
// for HW-scheduler load balance. FA4's L2 head-swizzle is a no-op here (it degenerates to
// kv_head-innermost, which the flatten already is).
//
// ** Q&A **
// Q: alpha_and_l_smem already has a "ready" signal (full_bar_alpha / full_bar_l). Why does
//    correction ALSO arrive empty_bar_alpha_and_l back to softmax?
// A: alpha_and_l_smem is ONE slot (depth 1), reused for every K-tile's alpha (and the final l).
//    full_bar_* tells correction "alpha is ready to read"; empty_bar_alpha_and_l is the reverse
//    credit: "I've read it, you may overwrite." Softmax must wait on that credit before writing
//    the next alpha. It is backpressure -- it bounds softmax to at most ONE K-tile ahead of
//    correction (a depth-1 producer/consumer pipe).
//
// Q: What breaks if I drop that return arrive -- a wrong number, or a crash?
// A: A HANG, and only at full speed. An mbarrier's phase is a single toggling bit (0,1,0,1,..);
//    a depth-1 pipe assumes producer and consumer stay within one step of each other. Without
//    the credit, softmax runs free and "laps" correction (gets 2+ K-tiles ahead); the phase bit
//    each side expects stops matching the barrier's real phase, so a wait_parity blocks forever
//    -> deadlock. Serialized execution (debugger / single-step) keeps them in lockstep so they
//    never lap and the hang vanishes -- a heisenbug that only reproduces under real concurrency.
//
// Q: Is sO one buffer per M-tile?
// A: No -- a SINGLE [M_TILE][HEAD_DIM] buffer reused by both M-tiles (the epilogue runs them
//    serially; the trailing bar_sync<9> guards the reuse). Double-buffering it regressed und
//    144->135 TFLOPS: the extra 32 KB SMEM pushed the SMEM/L1 carveout to the GB200 max and
//    halved the L1 hit rate. Staying under that ceiling is the win.
//
// Q: Why load K-tiles in descending order (g=0 = the diagonal / last keys)?
// A: For causal, the diagonal block is the only masked one. Doing it first hides its mask cost
//    in the prologue and matches FA4's KV order.
//
// TMEM budget: each M-tile needs S + O = K_TILE + HEAD_DIM cols of the 512, so
// M_TILES_PER_CTA <= 512 / (K_TILE + HEAD_DIM) = 2 for 128/128 (adjacent q-tiles of the
// SAME sequence).

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
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
#include "../../../primitives/76_packed_f32x2.cuh"
#include "../../../primitives/77_ex2_approx.cuh"
#include "../../../primitives/78_rcp_approx.cuh"
#include "../../../composites/118_mbarrier_phase_tracking.cuh"
#include "../../../primitives/46_setmaxnreg.cuh"
#include "../../../primitives/_warp_prof_noop.cuh"
#include "../../../composites/109_fastdivmod.cuh"
#include "../../../composites/112_fmha_softmax_utils.cuh"

constexpr int M_TILE = 128;
constexpr int M_TILES_PER_CTA = 2;
constexpr int K_TILE = 128;
constexpr int HEAD_DIM = 128;
// FA4 partial ex2 emulation (apply_exp2_convert): per 32-elt fragment, a fraction of
// pairs use the f32x2 ALU emulation (ex2_emu_f32x2) instead of MUFU.EX2 to relieve the
// EX2 pipe. Gate: HW unless (k%EX2_FREQ >= EX2_FREQ-EX2_RES) AND (fragment < last), where
// for softmax pair c: fragment j=c/EX2_FRG_PAIRS, in-fragment elt k=2*(c%EX2_FRG_PAIRS).
constexpr int EX2_FRG_PAIRS = 16;          // 32 elts / fragment = 16 pairs
constexpr int EX2_FRG_CNT   = K_TILE / 32; // = 4 for K_TILE=128
constexpr int EX2_FREQ      = 16;          // FA4 ex2_emu_freq (UND)
constexpr int EX2_RES       = 4;           // FA4 ex2_emu_res
// B128 swizzle atom = 128 bytes = 64 bf16: all SMEM tiles are laid out in
// 64-wide sub-tiles along the contiguous dim.
constexpr int SUB_COLS_BF16 = 64;
constexpr int SUB_COLS_BYTES = SUB_COLS_BF16 * (int)sizeof(__nv_bfloat16);
constexpr int Q_SUBTILES = HEAD_DIM / SUB_COLS_BF16;
constexpr int K_SUBTILES = HEAD_DIM / SUB_COLS_BF16;
constexpr int V_SUBTILES = K_TILE / SUB_COLS_BF16;
constexpr int P_SUBTILES = K_TILE / SUB_COLS_BF16;
constexpr int Q_SUB_COLS_BYTES = M_TILE * SUB_COLS_BYTES;
constexpr int K_SUB_COLS_BYTES = K_TILE * SUB_COLS_BYTES;
constexpr int Q_TILE_BYTES = Q_SUBTILES * Q_SUB_COLS_BYTES;
constexpr int K_TILE_BYTES = K_SUBTILES * K_SUB_COLS_BYTES;
constexpr int V_TILE_BYTES = V_SUBTILES * Q_SUB_COLS_BYTES;
constexpr int P_TILE_BYTES = P_SUBTILES * Q_SUB_COLS_BYTES;
// matmul-K atoms in one 64-wide (sub)tile: tcgen05 contracts K=16 per atom, so 64/16 = 4.
constexpr int K_ATOMS_PER_TILE = SUB_COLS_BF16 / 16;
constexpr int SPLIT_P_N    = K_TILE / 4 * 3;
constexpr int SPLIT_P_ATOM = SPLIT_P_N / 16;
constexpr int SPLIT_P_COL  = SPLIT_P_N / 2;
constexpr int NUM_KV_STAGES = 3;
constexpr int S_COLS = K_TILE;
constexpr int O_COLS = HEAD_DIM;
constexpr int TMEM_TOTAL = 512;
constexpr int W_CORR0 = 8;
constexpr int W_MMA = 12, W_EPI = 13, W_LOAD = 14, W_SCHED = 15;
constexpr int N_WARPS = 16;

extern __shared__ __align__(1024) uint8_t dyn_smem[];

#ifndef KERNEL_SPLIT_P
#define KERNEL_SPLIT_P true
#endif
#ifndef KERNEL_WARP_SCHED
#define KERNEL_WARP_SCHED false
#endif

// FA4-style varlen tile decode (flash-attn SingleTileVarlenScheduler): map global q-tile index gq
// -> (sample, #q-tiles in sample, local q-tile index) via a warp prefix-sum over cu_seqlens -- no
// host prefix array, no binary search. Warp-uniform result; scans 31 batches per round (lane 31 is
// the cu[b+1] boundary). gq < num_qtiles_real is guaranteed by the caller's slack check.
__device__ __forceinline__ void fa4_decode_qtile(int gq, int num_samples, const int* cu_seqlens_q,
                                                  int q_tokens_per_cta, int& sample,
                                                  int& num_m_blocks, int& qtile_in_sample) {
  const int lane = threadIdx.x & 31;
  int bidb_start = 0, group_start = 0;
  sample = 0; num_m_blocks = 1; qtile_in_sample = 0;
  while (bidb_start < num_samples) {
    const int batch  = bidb_start + lane;
    const int cu_cur = (batch <= num_samples) ? cu_seqlens_q[batch] : 0;
    const int cu_nxt = __shfl_down_sync(0xffffffffu, cu_cur, 1);   // cu[batch+1] for lanes 0..30
    const int nmb = (batch < num_samples && lane < 31)
                  ? (cu_nxt - cu_cur + q_tokens_per_cta - 1) / q_tokens_per_cta : 0;
    int cum = nmb;                                                 // inclusive warp prefix sum
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, cum, o);
      if (lane >= o) cum += v;
    }
    const int group_total = __shfl_sync(0xffffffffu, cum, 31);     // q-tiles in these 31 batches
    if (group_start + group_total > gq) {                          // gq falls in this group
      const int target   = gq - group_start;
      const int in_group = __popc(__ballot_sync(0xffffffffu, cum <= target));  // owning lane offset
      sample          = bidb_start + in_group;
      const int prev  = (in_group == 0) ? 0 : __shfl_sync(0xffffffffu, cum, in_group - 1);
      qtile_in_sample = target - prev;
      num_m_blocks    = __shfl_sync(0xffffffffu, nmb, in_group);
      return;
    }
    bidb_start += 31; group_start += group_total;
  }
}

// S_LD_COLS: cols per softmax tcgen05.ld of the S row (32/64 compile; 128 aborts ptxas).
// FULL_NAMED_BAR: softmax->correction "scale ready" signal -- true = HW named barrier (per-band),
//   false = mbarrier (full_bar_alpha/full_bar_l); both use alpha_and_l_smem. (EX2_EMU/SPLIT_P above.)
// WARP_SCHED: tile decode -- false = binary search on host qtile_prefix; true = FA4 warp prefix-sum.
template <bool IS_CAUSAL, int S_LD_COLS = 32, bool FULL_NAMED_BAR = false, bool EX2_EMU = false, bool SPLIT_P = true, bool WARP_SCHED = false>
__global__ void __launch_bounds__(N_WARPS * 32, 1)
fmha_context_bf16_kernel(const __grid_constant__ CUtensorMap tmap_q,
    const __grid_constant__ CUtensorMap tmap_k, const __grid_constant__ CUtensorMap tmap_v_t,
    __nv_bfloat16* __restrict__ O_out, const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ k_base, const int* __restrict__ seqlens_kv,
    int num_q_heads, int num_kv_heads, float scale_log2,
    const int* __restrict__ qtile_prefix, int num_qtiles_real, int num_samples) {
  const int gqa_group_size   = num_q_heads / num_kv_heads;
  const int q_tile_per_mtile = M_TILE / gqa_group_size;
  const int q_tokens_per_cta   = M_TILES_PER_CTA * q_tile_per_mtile;

  const int h_kv = blockIdx.x % num_kv_heads;
  const int gq   = blockIdx.x / num_kv_heads;
  if (gq >= num_qtiles_real) return;
  int sample, q_tile_id;
  if constexpr (WARP_SCHED) {
    int num_m_blocks, qtile_in_sample;
    fa4_decode_qtile(gq, num_samples, cu_seqlens_q, q_tokens_per_cta, sample, num_m_blocks, qtile_in_sample);
    q_tile_id = (num_m_blocks - 1) - qtile_in_sample;   // LPT (heaviest q-tile first)
  } else {
    int s_lo = 0, s_hi = num_samples;
    while (s_hi - s_lo > 1) {
      const int mid = (s_lo + s_hi) >> 1;
      if (qtile_prefix[mid] <= gq) s_lo = mid;
      else                         s_hi = mid;
    }
    sample = s_lo;
    const int num_m_blocks = qtile_prefix[sample + 1] - qtile_prefix[sample];
    q_tile_id = (num_m_blocks - 1) - (gq - qtile_prefix[sample]);   // LPT (heaviest q-tile first)
  }
  const int q_tile_base  = q_tile_id * q_tokens_per_cta;
  const int q_start      = cu_seqlens_q[sample];
  const int seqlen_q     = cu_seqlens_q[sample + 1] - q_start;
  const int k_start      = k_base[sample];
  const int seqlen_k     = seqlens_kv[sample];
  int K_TILES = (seqlen_k + K_TILE - 1) / K_TILE;
  if constexpr (IS_CAUSAL) {
    const int causal_k_tiles_cap = (q_tile_base + q_tokens_per_cta - 1) / K_TILE + 1;
    if (causal_k_tiles_cap < K_TILES) K_TILES = causal_k_tiles_cap;
  }
#ifdef WP_DUMP_TILE_MAP
  if (threadIdx.x == 0)
    printf("OURS-MAP cta=%d sample=%d q_tile_id=%d h_kv=%d seqlen_q=%d seqlen_k=%d K_TILES=%d\n",
           blockIdx.x, sample, q_tile_id, h_kv, seqlen_q, seqlen_k, K_TILES);
#endif

  uint8_t* sQ0 = dyn_smem;
  uint8_t* sQ1 = sQ0 + Q_TILE_BYTES;
  uint8_t* sQ[2] = { sQ0, sQ1 };
  uint8_t* sKV = sQ1 + Q_TILE_BYTES;
  __nv_bfloat16* sO = reinterpret_cast<__nv_bfloat16*>(sKV + NUM_KV_STAGES * K_TILE_BYTES);
  uint64_t* bar = reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(sO) + M_TILE * HEAD_DIM * sizeof(__nv_bfloat16));
  uint8_t* smem_kv[NUM_KV_STAGES];
  for (int s = 0; s < NUM_KV_STAGES; ++s) smem_kv[s] = sKV + s * K_TILE_BYTES;
  uint64_t* full_bar    = bar;
  uint64_t* empty_bar   = full_bar + NUM_KV_STAGES;
  uint64_t* full_bar_q  = empty_bar + NUM_KV_STAGES;
  uint64_t* empty_bar_q   = full_bar_q + 2;
  uint64_t* full_bar_spo  = empty_bar_q + 2;
  uint64_t* empty_bar_spo = full_bar_spo + 2;
  uint64_t* full_bar_o_acc = empty_bar_spo + 2;
  uint64_t* full_bar_alpha = full_bar_o_acc + 2;
  uint64_t* full_bar_l     = full_bar_alpha + 2;
  uint64_t* full_bar_p_last = full_bar_l + 2;
  uint64_t* empty_bar_alpha_and_l = full_bar_p_last + 2;
  uint32_t* tmem_slot = reinterpret_cast<uint32_t*>(empty_bar_alpha_and_l + 2);
  float* alpha_and_l_smem = reinterpret_cast<float*>(tmem_slot + 2);

  const int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;

  if (warp_id == 0) { tcgen05_alloc<1>(smem_ptr_u32(tmem_slot), TMEM_TOTAL); tcgen05_relinquish_alloc_permit<1>(); }
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
    }
  }
  fence_mbarrier_init_release_cluster();
  __syncthreads();
  WpCtx wpc = wp_ctx_init();

  if (warp_id == W_LOAD) {
    setmaxnreg_dec<56>();

    EmptyPhaseTracker<NUM_KV_STAGES> kv_empty_ph;
    EmptyPhaseTracker<1> q_empty_ph;

    for (int k = 0; k < K_TILES; ++k) {
      const int k_offset = (K_TILES - 1 - k) * K_TILE;
      int kv_stage = kv_empty_ph.get_stage();

      wp_begin(wpc, WP_LOAD_WAIT);
      mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar[kv_stage]), kv_empty_ph.get_phase());
      kv_empty_ph.advance();
      wp_end(wpc, WP_LOAD_WAIT);

      wp_begin(wpc, WP_LOAD_ISSUE_K);
      if (lane == 0) {
        mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar[kv_stage]), K_TILE_BYTES);
        // ONE 3D copy folds both head-dim swizzle atoms (vs 2 x 2D). coords
        // {atom-col 0, token, atom h_kv*K_SUBTILES}; box [SUB_COLS_BF16, K_TILE, K_SUBTILES].
        tma_load_3d(smem_ptr_u32(smem_kv[kv_stage]), &tmap_k, smem_ptr_u32(&full_bar[kv_stage]),
                    0, k_start + k_offset, h_kv * K_SUBTILES);
      }
      wp_end(wpc, WP_LOAD_ISSUE_K);

      if (k == 0 && lane == 0) {
        // Q M-tile m: all gqa_group_size q-heads packed (qh-inner) via a 3D TMA box
        #pragma unroll
        for (int m = 0; m < M_TILES_PER_CTA; ++m) {
          wp_begin(wpc, WP_LOAD_WAIT);
          mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_q[m]), q_empty_ph.get_phase());
          wp_end(wpc, WP_LOAD_WAIT);

          wp_begin(wpc, WP_LOAD_ISSUE_Q);
          mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar_q[m]), Q_TILE_BYTES);
          const int tok0 = q_start + q_tile_base + m * q_tile_per_mtile;
          #pragma unroll
          for (int s = 0; s < Q_SUBTILES; ++s) {
            tma_load_3d(smem_ptr_u32(sQ[m] + s * Q_SUB_COLS_BYTES), &tmap_q, smem_ptr_u32(&full_bar_q[m]),
                        s * SUB_COLS_BF16, h_kv * gqa_group_size, tok0);
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
      if (lane == 0) {
        mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar[kv_stage]), V_TILE_BYTES);
        #pragma unroll
        for (int s = 0; s < V_SUBTILES; ++s) {
          tma_load_2d(smem_ptr_u32(smem_kv[kv_stage] + s * Q_SUB_COLS_BYTES), &tmap_v_t, smem_ptr_u32(&full_bar[kv_stage]),
                      k_start + k_offset + s * SUB_COLS_BF16, h_kv * HEAD_DIM);
        }
      }
      wp_end(wpc, WP_LOAD_ISSUE_V);
    }
  }
  else if (warp_id == W_MMA) {
    setmaxnreg_dec<56>();

    const bool lead = (lane == 0);
    constexpr uint32_t DESC_SBO = 1024, DESC_LBO = 16;
    const uint32_t idesc_qk = make_idesc_bf16_f32(M_TILE, K_TILE, false, false);
    const uint32_t idesc_pv = make_idesc_bf16_f32(M_TILE, HEAD_DIM,  false, false);
    const uint64_t desc_q0  = build_smem_desc_blackwell(smem_ptr_u32(sQ0), DESC_SBO, DESC_LBO, SmemSwizzleBlackwell::B128);
    const uint64_t desc_kv0 = build_smem_desc_blackwell(smem_ptr_u32(sKV), DESC_SBO, DESC_LBO, SmemSwizzleBlackwell::B128);
    // >> 4: the SMEM descriptor address field is in 16-byte units (addr >> 4).
    constexpr uint64_t KV_DESC_DELTA      = K_TILE_BYTES >> 4;
    constexpr uint64_t SUB_DESC_DELTA     = Q_SUB_COLS_BYTES >> 4;
    constexpr uint64_t Q_MTILE_DESC_DELTA = Q_TILE_BYTES >> 4;

    PhaseTracker<NUM_KV_STAGES> kv_ph;
    PhaseTracker<1> q_ph;
    PhaseTracker<1> spo_ph;

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
          const uint64_t da = da_base + (uint64_t)s * SUB_DESC_DELTA;
          const uint64_t db = db_base + (uint64_t)s * SUB_DESC_DELTA;
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
        tcgen05_fence_after_thread_sync();
        const uint32_t s_tmem_addr = tmem_base + (uint32_t)(i * S_COLS);
        const uint32_t o_tmem_addr = tmem_base + (uint32_t)(2 * S_COLS + i * O_COLS);
        const uint64_t dbV = desc_kv0 + (uint64_t)kv_stage * KV_DESC_DELTA;
        #pragma unroll
        for (int s = 0; s < V_SUBTILES; ++s) {
          const uint64_t db_s = dbV + (uint64_t)s * SUB_DESC_DELTA;
          #pragma unroll
          for (int ki = 0; ki < K_ATOMS_PER_TILE; ++ki) {
            const int a = s * K_ATOMS_PER_TILE + ki;
            if constexpr (SPLIT_P) if (a == SPLIT_P_ATOM) {
              wp_end(wpc, WP_MMA_ISSUE);
              wp_begin(wpc, WP_MMA_WAIT_P);
              mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_p_last[i]), spo_ph.get_phase());
              wp_end(wpc, WP_MMA_WAIT_P);
              wp_begin(wpc, WP_MMA_ISSUE);
              tcgen05_fence_after_thread_sync();
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
          const uint64_t da_base = desc_q0 + (uint64_t)i * Q_MTILE_DESC_DELTA;
          const uint64_t db_base = desc_kv0 + (uint64_t)kv_stage_next * KV_DESC_DELTA;
          #pragma unroll
          for (int s = 0; s < Q_SUBTILES; ++s) {
            const uint64_t da = da_base + (uint64_t)s * SUB_DESC_DELTA;
            const uint64_t db = db_base + (uint64_t)s * SUB_DESC_DELTA;
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
      tcgen05_fence_after_thread_sync();
      const uint32_t s_tmem_addr = tmem_base + (uint32_t)(i * S_COLS);
      const uint32_t o_tmem_addr = tmem_base + (uint32_t)(2 * S_COLS + i * O_COLS);
      const uint64_t dbV = desc_kv0 + (uint64_t)kv_stage * KV_DESC_DELTA;
      #pragma unroll
      for (int s = 0; s < V_SUBTILES; ++s) {
        const uint64_t db_s = dbV + (uint64_t)s * SUB_DESC_DELTA;
        #pragma unroll
        for (int ki = 0; ki < K_ATOMS_PER_TILE; ++ki) {
          const int a = s * K_ATOMS_PER_TILE + ki;   // flat atom (split-P + P addr)
          if constexpr (SPLIT_P) if (a == SPLIT_P_ATOM) {
            wp_end(wpc, WP_MMA_ISSUE);
            wp_begin(wpc, WP_MMA_WAIT_P);
            mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_p_last[i]), spo_ph.get_phase());
            wp_end(wpc, WP_MMA_WAIT_P);
            wp_begin(wpc, WP_MMA_ISSUE);
            tcgen05_fence_after_thread_sync();
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
  else if (warp_id == W_EPI || warp_id == W_SCHED) {
    setmaxnreg_dec<56>();
  }
  else if (warp_id >= W_CORR0 && warp_id < W_MMA) {
    setmaxnreg_dec<72>();

    const int corr_warp_id = warp_id - W_CORR0;
    [[maybe_unused]] PhaseTracker<1> alpha_ph;
    PhaseTracker<1> o_acc_ph;

    #pragma unroll
    for (int i = 0; i < M_TILES_PER_CTA; ++i) {
      mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[i]));
      mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
    }

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
        mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
        wp_end(wpc, WP_CORR_READ_ALPHA);

        wp_begin(wpc, WP_CORR_O_SCALE);
        bool skip = __all_sync(0xffffffffu, alpha == 1.0f);
        if (!skip) {
          tcgen05_fence_after_thread_sync();
          const uint32_t o_tmem_addr = tmem_base + (uint32_t)(2 * S_COLS + i * O_COLS) + ((uint32_t)(corr_warp_id * 32) << 16);
          // SERIAL x64 chunks: our inline-asm tcgen05.ld/st are opaque to ptxas, so each
          // chunk drains fully via tcgen05.wait (no per-store scoreboard overlap).
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
          tcgen05_fence_before_thread_sync();
        }
        mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[i]));
        wp_end(wpc, WP_CORR_O_SCALE);
      }
      if constexpr (!FULL_NAMED_BAR) alpha_ph.advance();
    }

    // epilogue: O *= 1/l, then store to gmem
    #pragma unroll
    for (int i = 0; i < M_TILES_PER_CTA; ++i) {
      // order: wait l -> read l + rcp.approx -> wait O, so the RCP latency hides under the O-acc wait
      const int corr_tid = corr_warp_id * 32 + lane;
      wp_begin(wpc, WP_CORR_WAIT);
      if constexpr (FULL_NAMED_BAR) full_bar_wait(i, corr_warp_id);
      else mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_l[i]), o_acc_ph.get_phase());
      wp_end(wpc, WP_CORR_WAIT);
      float l = alpha_and_l_smem[i * M_TILE + corr_tid];
      // release stats slot; dead tail (non-persistent) but kept for wait/release pairing
      mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
      float inv_l = (l > 0.f) ? rcp_approx_ftz_f32(l) : 0.f;
      wp_begin(wpc, WP_CORR_WAIT);
      mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_o_acc[i]), o_acc_ph.get_phase());
      wp_end(wpc, WP_CORR_WAIT);

      wp_begin(wpc, WP_CORR_EPI);
      // PASS 1: read this row's O from TMEM, O *= 1/l, FP32 -> bf16 -> sO[row]
      tcgen05_fence_after_thread_sync();
      const uint32_t o_tmem_addr = tmem_base + (uint32_t)(2 * S_COLS + i * O_COLS) + ((uint32_t)(corr_warp_id * 32) << 16);
      __nv_bfloat16* sOi = sO;
      // swizzle chunk c -> c ^ (row&7) (avoids the 4-bank collision); same map on read.
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
          *reinterpret_cast<uint4*>(&sOi[corr_tid * HEAD_DIM + (chunk ^ (corr_tid & 7)) * 8]) = packed;
        }
      }
      tcgen05_fence_before_thread_sync();
      mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[i]));
      bar_sync<9>(128);
      // PASS 2: re-tile sO -> gmem (16 threads/row, 8 bf16 each): r = (corr_tid>>4) + 8*iter,
      // so q_head + swizzle are per-thread constants; both addresses reduce to base + stride/iter.
      const int qhi = corr_tid >> 4;                       // r % gqa_group_size  and  r & 7
      const int c = corr_tid & 15;
      const int q_head = h_kv * gqa_group_size + qhi;
      const int tok0 = q_tile_base + i * q_tile_per_mtile;
      const long gstride = (long)num_q_heads * HEAD_DIM;   // gmem bytes-in-elems per token row
      long ob = (long)(q_start + tok0) * gstride + (long)q_head * HEAD_DIM + (long)(c * 8);
      const uint4* srow = reinterpret_cast<const uint4*>(&sOi[qhi * HEAD_DIM + (c ^ qhi) * 8]);
      #pragma unroll
      for (int iter = 0; iter < M_TILE / 8; ++iter) {
        if (tok0 + iter < seqlen_q)
          *reinterpret_cast<uint4*>(&O_out[ob]) = *srow;
        ob   += gstride;        // token += 1
        srow += HEAD_DIM;       // r += 8 -> sO row += 8*HEAD_DIM bf16 = HEAD_DIM uint4
      }
      // 2nd bar_sync: guards the sO WAR reuse (see header Q&A)
      bar_sync<9>(128);
      wp_end(wpc, WP_CORR_EPI);
    }
    o_acc_ph.advance();
  }
  else {
    setmaxnreg_inc<192>();

    const int m_tile = warp_id < 4 ? 0 : 1;
    const int warp_in_group = warp_id & 3;
    const int row_in_m_tile = warp_in_group * 32 + lane;
    const uint32_t s_tmem_addr = tmem_base + (uint32_t)(m_tile * S_COLS) + ((uint32_t)(warp_in_group * 32) << 16);
    PhaseTracker<1> spo_ph;
    PhaseTracker<1> scale_empty_ph;

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
      tcgen05_fence_after_thread_sync();
      uint32_t s_regs[K_TILE];
      #pragma unroll
      for (int c0 = 0; c0 < K_TILE; c0 += S_LD_COLS) {
        const uint32_t taddr = s_tmem_addr + (uint32_t)c0;
        if      constexpr (S_LD_COLS == 32)  tcgen05_ld_32x32b_x32 (taddr, *reinterpret_cast<uint32_t(*)[32]>(&s_regs[c0]));
        else if constexpr (S_LD_COLS == 64)  tcgen05_ld_32x32b_x64 (taddr, *reinterpret_cast<uint32_t(*)[64]>(&s_regs[c0]));
        else if constexpr (S_LD_COLS == 128) tcgen05_ld_32x32b_x128(taddr, *reinterpret_cast<uint32_t(*)[128]>(&s_regs[c0]));
      }
      tcgen05_wait_ld();

      float* scores = reinterpret_cast<float*>(s_regs);
      float2* scores2 = reinterpret_cast<float2*>(s_regs);
      // order this S read before the P st that overwrites the slot
      tcgen05_fence_before_thread_sync();

      // k==0 (diagonal block) is the only masked block
      if (k == 0) mask_s_row_r2p<IS_CAUSAL, K_TILE>(scores, k_offset, q_pos, seqlen_k);

      // rmax via 4 independent FMNMX3 accumulators: breaks the serial max chain (4-way ILP)
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
      float alpha = 0.0f;
      if (k != 0) alpha = ex2_approx_f32((m_run - new_m) * scale_log2);
      alpha_and_l_smem[m_tile * M_TILE + row_in_m_tile] = alpha;
      if constexpr (FULL_NAMED_BAR) full_bar_arrive(m_tile, warp_in_group);
      else mbarrier_arrive(smem_ptr_u32(&full_bar_alpha[m_tile]));

      // fused ffma2(scale) + exp2 + row-sum + bf16 pack: 128 keys -> 64 u32 P cols.
      const float2 scale2 = f32x2_splat(scale_log2);
      const float2 neg_m_scaled2 = f32x2_splat(-new_m * scale_log2);
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
        tcgen05_fence_before_thread_sync();
        mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[m_tile]));
        tcgen05_st_32x32b_x16(p_tmem_addr + SPLIT_P_COL, *reinterpret_cast<uint32_t(*)[16]>(&p_regs[SPLIT_P_COL]));
        tcgen05_fence_before_thread_sync();
        mbarrier_arrive(smem_ptr_u32(&full_bar_p_last[m_tile]));
      } else {
        tcgen05_st_32x32b_x32(p_tmem_addr, *reinterpret_cast<uint32_t(*)[32]>(&p_regs[0]));
        tcgen05_st_32x32b_x32(p_tmem_addr + 32, *reinterpret_cast<uint32_t(*)[32]>(&p_regs[32]));
        tcgen05_fence_before_thread_sync();
        mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[m_tile]));
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
  wp_flush(wpc);
  __syncthreads();
  if (warp_id == 0) tcgen05_dealloc<1>(tmem_base, TMEM_TOTAL);
}

// ====================== driver ============================
#include "fmha_cpu_ref.cuh"
#include "fmha_context_bf16_benchmark.cuh"

static void fillr(__nv_bfloat16 *h, long n, unsigned s) {
  for (long i = 0; i < n; ++i)
    h[i] = __float2bfloat16((float)((i * 2654435761u + s) % 2048) / 1024.0f - 1.0f);
}
struct Sh {
  std::vector<int> sl;
  int nqh, nkh, hd;
  bool causal;
  const char *lab;
};
static double run(const Sh &sh, bool verify) {
  int num_samples = (int)sh.sl.size();
  std::vector<int> cq(num_samples + 1, 0);
  for (int i = 0; i < num_samples; ++i)
    cq[i + 1] = cq[i] + sh.sl[i];
  std::vector<int> ck = cq;
  long tq = cq.back();
  std::vector<int> kb(num_samples), sk(num_samples);
  long acc = 0;
  for (int i = 0; i < num_samples; ++i) {
    kb[i] = (int)acc;
    sk[i] = sh.sl[i];
    acc += ((long)sh.sl[i] + 7) / 8 * 8;
  }
  long tka = acc > 0 ? acc : 8;
  __nv_bfloat16 *dQ, *dK, *dVT, *dO;
  CUDA_CHECK(cudaMalloc(&dQ, tq * sh.nqh * sh.hd * 2));
  CUDA_CHECK(cudaMalloc(&dK, tka * sh.nkh * sh.hd * 2));
  CUDA_CHECK(cudaMalloc(&dVT, (long)sh.nkh * sh.hd * tka * 2));
  CUDA_CHECK(cudaMalloc(&dO, tq * sh.nqh * sh.hd * 2));
  std::vector<__nv_bfloat16> hQ(tq * sh.nqh * sh.hd), hK(tq * sh.nkh * sh.hd),
      hV(tq * sh.nkh * sh.hd);
  fillr(hQ.data(), hQ.size(), 11);
  fillr(hK.data(), hK.size(), 22);
  fillr(hV.data(), hV.size(), 33);
  std::vector<__nv_bfloat16> hKa(tka * sh.nkh * sh.hd, __float2bfloat16(0.f)),
      hVT((long)sh.nkh * sh.hd * tka, __float2bfloat16(0.f));
  for (int s = 0; s < num_samples; ++s)
    for (int j = 0; j < sk[s]; ++j) {
      long sr = ck[s] + j, ds = kb[s] + j;
      for (int h = 0; h < sh.nkh; ++h)
        for (int d = 0; d < sh.hd; ++d) {
          hKa[(ds * sh.nkh + h) * sh.hd + d] = hK[(sr * sh.nkh + h) * sh.hd + d];
          hVT[(h * sh.hd + d) * tka + ds] = hV[(sr * sh.nkh + h) * sh.hd + d];
        }
    }
  CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), hQ.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dK, hKa.data(), hKa.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dVT, hVT.data(), hVT.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dO, 0, hQ.size() * 2));
  int *dCq, *dKb, *dSk;
  CUDA_CHECK(cudaMalloc(&dCq, (num_samples + 1) * 4));
  CUDA_CHECK(cudaMalloc(&dKb, num_samples * 4));
  CUDA_CHECK(cudaMalloc(&dSk, num_samples * 4));
  CUDA_CHECK(cudaMemcpy(dCq, cq.data(), (num_samples + 1) * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dKb, kb.data(), num_samples * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dSk, sk.data(), num_samples * 4, cudaMemcpyHostToDevice));
  CUtensorMap tq_, tk_, tvt_;
  // pack-GQA Q: 3D TMA over [hd, nqh, total_q]; box [hd-subtile x gqa_group_size x
  // q_tile_per_mtile] -> 128 packed rows (qh-inner).
  int gqa_group_size = sh.nqh / sh.nkh, q_tile_per_mtile = M_TILE / gqa_group_size,
      q_tokens_per_cta = 2 * q_tile_per_mtile;
  CUDA_CHECK(make_tma_3d_tiled(&tq_, dQ, sh.hd, sh.nqh, (int)tq, SUB_COLS_BF16, gqa_group_size,
                               q_tile_per_mtile, 2, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                               CU_TENSOR_MAP_SWIZZLE_128B));
  // K: ONE 3D TMA copy folds the 2 head-dim swizzle atoms (HEAD_DIM = 2 x SUB_COLS_BF16) into the box
  // (vs looping 2 x 2D copies). dims [atom-col SUB_COLS_BF16, token tka, atom (nkh*hd)/SUB_COLS_BF16]; box
  // [SUB_COLS_BF16, K_TILE, K_SUBTILES]; strides token=(nkh*hd)*2B, atom=SUB_COLS_BF16*2B. The box dim order
  // (atom outermost) reproduces the atom-outer smem layout the MMA reads (atom0 then atom1).
  {
    uint64_t gd[3] = { (uint64_t)SUB_COLS_BF16, (uint64_t)tka, (uint64_t)(sh.nkh * sh.hd / SUB_COLS_BF16) };
    uint64_t gs[2] = { (uint64_t)(sh.nkh * sh.hd) * 2u, (uint64_t)SUB_COLS_BF16 * 2u };
    uint32_t bd[3] = { (uint32_t)SUB_COLS_BF16, (uint32_t)K_TILE, (uint32_t)K_SUBTILES };
    uint32_t es[3] = { 1u, 1u, 1u };
    CUresult r = cuTensorMapEncodeTiled(&tk_, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, dK, gd, gs, bd, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUDA_CHECK(r == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue);
  }
  CUDA_CHECK(make_tma_2d_tiled(&tvt_, dVT, (long)sh.nkh * sh.hd, tka, sh.hd, SUB_COLS_BF16, 2,
                               CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B));
  // host q-tile prefix sum for the kernel's tile decode (formula in header ** Work decomposition **)
  std::vector<int> qtile_prefix(num_samples + 1, 0);
  for (int i = 0; i < num_samples; ++i)
    qtile_prefix[i + 1] = qtile_prefix[i] + (sh.sl[i] + q_tokens_per_cta - 1) / q_tokens_per_cta;
  const int num_qtiles_real = qtile_prefix[num_samples];
  // SMEM: 2 Q tiles + K/V ring + sO[M_TILE][HEAD_DIM] + mbarriers (2*NS+18 u64) +
  // tmem_slot + alpha_and_l_smem[2][M_TILE] + alignment slack.
  size_t smem = (size_t)2 * Q_TILE_BYTES + NUM_KV_STAGES * K_TILE_BYTES +
                (size_t)M_TILE * HEAD_DIM * sizeof(__nv_bfloat16) + (2 * NUM_KV_STAGES + 18) * 8 +
                8 + (size_t)2 * M_TILE * sizeof(float) + 256;
  auto kfn = sh.causal ? &fmha_context_bf16_kernel<true, 32, false, false, KERNEL_SPLIT_P, KERNEL_WARP_SCHED>
                       : &fmha_context_bf16_kernel<false, 32, false, false, KERNEL_SPLIT_P, KERNEL_WARP_SCHED>;
  CUDA_CHECK(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  float sl2 = (1.0f / sqrtf((float)sh.hd)) * (float)M_LOG2E;
  // NON-PERSISTENT grid = tight upper bound x num_kv_heads, one CTA/tile (see header);
  // the bound is computed in packed-row units (total_q and tile_M x gqa_group_size).
  const long packed_total_q = tq * gqa_group_size;
  const int packed_tile_m = q_tokens_per_cta * gqa_group_size;
  const int total_blocks_max =
      (int)((packed_total_q + (long)num_samples * (packed_tile_m - 1)) / packed_tile_m);
  int *d_qtile_prefix;
  CUDA_CHECK(cudaMalloc(&d_qtile_prefix, (num_samples + 1) * 4));
  CUDA_CHECK(cudaMemcpy(d_qtile_prefix, qtile_prefix.data(), (num_samples + 1) * 4,
                        cudaMemcpyHostToDevice));
  dim3 grid(total_blocks_max * sh.nkh, 1, 1), block(N_WARPS * 32, 1, 1);
  printf("  [%s] non-persistent grid=%d (valid tiles=%d, slack=%d)\n", sh.lab, grid.x,
         num_qtiles_real * sh.nkh, grid.x - num_qtiles_real * sh.nkh);
  auto launch = [&](cudaStream_t st = 0) {
    kfn<<<grid, block, smem, st>>>(tq_, tk_, tvt_, dO, dCq, dKb, dSk, sh.nqh, sh.nkh, sl2,
                                 d_qtile_prefix, num_qtiles_real, num_samples);
    return cudaGetLastError();
  };
  const double ms = fmha_context_bf16_benchmark::measure(launch);
#ifdef WARP_PROF
  {
    const int rep_block = (grid.x > 1636u) ? 1636 : 0; // representative block (max K-tiles)
    WpBuffer wp = wp_alloc(grid, rep_block);
    const unsigned pblk = wp.view_block;
    CUDA_CHECK(launch());
    CUDA_CHECK(cudaDeviceSynchronize());
    wp_readback(wp);
    const char *roles[16] = {"sm0",  "sm0",  "sm0",  "sm0",  "sm1", "sm1", "sm1",  "sm1",
                             "corr", "corr", "corr", "corr", "mma", "epi", "load", "sched"};
    printf("  [%s] WARP_PROF block %u:\n", sh.lab, pblk);
    wp_print_busy(wp, roles, 16, pblk);
    if (grid.x > 1636u) {
      wp_dump_raw(wp, "warp_raw_fmha_np_und.bin.gz", pblk, NUM_KV_STAGES);
    }
    wp_free(wp);
  }
#endif
  uint64_t eff = 0;
  for (int i = 0; i < num_samples; ++i) {
    uint64_t L = sh.sl[i];
    eff += fmha_context_bf16_benchmark::attended_pairs(L, L, sh.causal);
  }
  const double tf = fmha_context_bf16_benchmark::report(
      sh.lab, tq, sh.causal, sh.hd, sh.nqh, eff, ms);
  if (verify) {
    std::vector<__nv_bfloat16> ho(hQ.size());
    CUDA_CHECK(cudaMemcpy(ho.data(), dO, ho.size() * 2, cudaMemcpyDeviceToHost));
    std::vector<float> rf(hQ.size(), 0.f), ou(hQ.size());
    cpu_fmha_ref(hQ.data(), hK.data(), hV.data(), rf.data(), cq, ck, sh.nqh, sh.nkh, sh.hd,
                 sh.causal);
    for (size_t i = 0; i < ou.size(); ++i)
      ou[i] = __bfloat162float(ho[i]);
    bool ok = check_close_f32(rf.data(), ou.data(), (int)ou.size(), 0.05f, 0.10f);
    printf("  verify [%s]: %s\n", sh.lab, ok ? "OK" : "FAIL");
  }
  cudaFree(dQ);
  cudaFree(dK);
  cudaFree(dVT);
  cudaFree(dO);
  cudaFree(dCq);
  cudaFree(dKb);
  cudaFree(dSk);
  cudaFree(d_qtile_prefix);
  return tf;
}
static std::vector<int> und() {
  return {29,  101, 29,  53,  135, 85,  106, 94,  29,  95,  104, 69,  156, 214, 95,  164,
          159, 29,  170, 60,  118, 58,  93,  151, 50,  163, 133, 58,  61,  134, 203, 182,
          62,  56,  29,  29,  67,  29,  92,  118, 153, 108, 202, 62,  108, 87,  29,  29,
          52,  68,  175, 99,  64,  160, 29,  47,  179, 59,  55,  220, 49,  131, 95,  29,
          88,  153, 171, 89,  195, 96,  108, 53,  193, 60,  105, 110, 113, 67,  92,  113,
          60,  53,  175, 177, 91,  175, 128, 133, 207, 141, 186, 68,  192, 56,  94,  76,
          105, 159, 64,  199, 192, 128, 104, 102, 110, 54,  95,  195, 55,  29,  98,  58,
          291, 61,  89,  177, 221, 179, 187, 61,  102, 107, 100, 206, 102, 104, 189, 154};
}
int main() {
  CUDA_CHECK(cudaFree(0));
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 64 * 1024 * 1024));
  printf("K2 fmha_context_bf16 gqa non-persistent (warp-spec, 2 M-tiles) sm_100a\n");
  printf("=====================================================================\n");

  // Causal shapes only: tiny-causal = fast correctness check, und = the target shape.
  {
    Sh s{};
    s.sl = std::vector<int>{29, 101, 240};
    s.nqh = 8;
    s.nkh = 1;
    s.hd = 128;
    s.causal = true;
    s.lab = "tiny-causal";
    run(s, true);
  }
  {
    Sh s{};
    s.sl = und();
    s.nqh = 32;
    s.nkh = 4;
    s.hd = 128;
    s.causal = true;
    s.lab = "und";
    run(s, true);
  }
  return 0;
}
