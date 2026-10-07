// block_sparse_bwd_bf16_blk128.cu -- VSA block-sparse BACKWARD, bf16, sm_100a,
// 128-token blocks, KV-stationary, FA4-form (flash_bwd_sm100.py, 1cta, hdim128).
//
// One CTA owns one kv128 block and walks that block's q-list, one q128 block per step. Per step it
// recomputes P^T, forms dS^T, accumulates dK/dV in TMEM, and pushes the step's dQ tile to a global
// fp32 accumulator with cp.reduce.async.bulk. A postprocess kernel unscrambles the accumulator,
// applies sm_scale and stores bf16 dQ. Tile == block, so there is no q padding, no union and no kv
// row masking (every kv block is full). Grid = (kv blocks, B*H). Activations are [B, S, H, 128]
// or, with VSA_BHSD, FastVideo's [B, H, S, 128].
//
// Warps: 16 -- w0-3 epilogue (dQ drain), w4-11 softmax (P^T + dS^T; also the dK/dV store at the
// end of the tile), w12 mma, w13 load, w14 CLC scheduler (a second donor otherwise), w15 register
// donor. All GEMMs are plain m128 tcgen05.mma atoms.
//
// Structure:
//   1. Non-persistent. One (kv block, head) per CTA, no work-item loop, no CLC; the CTA exits after
//      the dK/dV store. Zero-count kv blocks (no q block selects them) take the zero-fill path: the
//      softmax warps TMA-store zero dK/dV tiles through sDST, everyone else exits.
//   2. Operands. K and V (32 KB each) ride the first step's Q and dO loads as extra TMA bytes. Q is
//      double-buffered (two 32 KB stages), dO single-buffered (32 KB); both are natural [token, hd]
//      tiles read K-major by GEMMs 1/2 and MN-major (transpose-B) by GEMMs 3/5. LSE rides the Q
//      stage index (two 512 B slots), Delta rides the dO stage (one 512 B slot).
//   3. Stage release. A stage is freed by the tcgen05 commit of its LAST-reading GEMM: dK(j) frees
//      Q[stage j], dV(j) frees dO.
//   4. dQ. GEMM 4 leaves the step's dQ tile [128 q x 128 hd] fp32 in TMEM; the epilogue warps read
//      the whole tile (empty_bar_dq fires right after, which gates the next dP^T into the same
//      columns), then stage and reduce-add it in 4 chunks of 32 hd columns through 2 SMEM buffers.
//
// Data layout (D = 128, BLOCK = 128):
//   name       where dtype shape                                   written by    -> read by
//   Q,K,V,dO,O gmem  bf16  [B, S, H, 128] (VSA_BHSD: [B, H, S, 128]) caller      -> TMA / pre
//   LSE, Delta gmem  fp32  [B*H, S] (log2 form)                    fwd / pre     -> load (bulk)
//   K, V       SMEM  bf16  2 hd-halves x [128 kv x 64 hd]          TMA (once)    -> GEMM 1,2,4
//   Q[2], dO   SMEM  bf16  2 hd-halves x [128 q x 64 hd]           TMA per step  -> GEMM 1,5 / 2,3
//   S^T        TMEM  fp32  128 lanes x 128 q (cols 0-127)          GEMM 1        -> softmax
//   P^T        TMEM  bf16  128 lanes x 128 q (bf16x2, cols 0-31, 64-95)  softmax -> GEMM 3
//   dP^T       TMEM  fp32  128 lanes x 128 q (cols 256-383)        GEMM 2        -> softmax
//   dS^T       TMEM  bf16  128 lanes x 128 q (bf16x2, cols 256-287, 320-351)  softmax -> GEMM 5
//   dS (sDST)  SMEM  bf16  2 q-halves x [128 kv x 64 q]            softmax       -> GEMM 4
//   dV, dK     TMEM  fp32  128 lanes x 128 hd (cols 128-255 / 384-511)  GEMM 3, 5 -> softmax
//   dQ         TMEM  fp32  128 lanes x 128 hd (cols 256-383)       GEMM 4        -> epilogue
//   dqaccum    gmem  fp32  per (b*h, q128 block) 16384 elems, drain-native  epilogue -> post
//   dK,dV,dQ   gmem  bf16  same layout as Q                        softmax / post
//   (bf16x2) = two bf16 values packed in one 32-bit word, low half first. A softmax warp rewrites
//            the 64 fp32 columns of its q-half as 32 columns of bf16 pairs, in place at the start
//            of those same 64 columns (P^T over S^T, dS^T over dP^T): q-half 0 at cols 0-31,
//            q-half 1 at cols 64-95. Staying inside its own fp32 range means a warp never
//            overwrites columns the other warp of its lane group still has to load. GEMMs 3/5 read
//            the two 32-column runs as their bf16 A operand.
//
// The 5 GEMMs per step, in issue order (S(j+1) -> dK(j) -> dQ(j) -> dP^T(j+1) -> dV(j+1)):
//   1. S^T = K @ Q^T       SS m128n128k16 x 8: A = K (K-major over hd), B = Q[stage] (K-major),
//                          2 hd subtiles x 4 k16 each; commit -> full_bar_st.
//   2. dP^T = V @ dO^T     SS m128n128k16 x 8, as GEMM 1 with V and dO; commit -> full_bar_dpt.
//   3. dV += P^T @ dO      TS m128n128k16 x 8: A = P^T bf16x2 overlay (TMEM), B = dO MN-major
//                          (tb = 1), 8 k16 over the 128 q; commit -> empty_bar_do (last dO reader).
//   4. dQ = dS @ K         SS m128n128k16 x 8: A = dS from sDST (MN-major, ta = 1), B = K MN-major
//                          (tb = 1), 8 k16 over the 128 kv; commit -> full_bar_dq.
//   5. dK += dS^T @ Q      TS m128n128k16 x 8: A = dS^T bf16x2 overlay, B = Q[stage] MN-major;
//                          commit -> empty_bar_q[stage] (last Q reader).
//   Prologue: S^T(0), dP^T(0), dV(0). Step j: S^T(j+1) | dK(j) | dQ(j) | dP^T(j+1) (after
//   empty_bar_dq: dQ(j) has left cols 256-383) | dV(j+1). full_bar_dv is committed when the loop
//   enters its last step (every dV accumulation issued), full_bar_dk after the tail dK.
//
// Warp roles:
//   load (w13)       Per step: TMA Q tile into the free Q stage (+K on step 0), bulk LSE row
//                    (512 B) into the LSE slot of that stage, TMA dO tile (+V on step 0), bulk
//                    Delta row.
//   mma (w12)        Allocates TMEM after the init sync and publishes it via bar_sync<10>(416);
//                    issues GEMMs 1-5 in the order above and the commits that publish S^T, dP^T,
//                    dQ, dV, dK and free the Q / dO stages.
//   softmax (w4-11)  8 warps split the 128-lane x 128-column S^T / dP^T tiles into four 32-lane
//                    groups x two 64-column q-halves: w4-7 take columns 0-63, w8-11 columns 64-127;
//                    within each four, warp i takes lanes 32i..32i+31 (TMEM lane access is fixed
//                    by warp_id % 4). Per step: P^T = exp2(S^T*scale_log2 - LSE) packed to bf16
//                    pairs over S^T's own columns (GEMM 3 reads it); dS^T = P^T*(dP^T - Delta)
//                    packed the same way over dP^T's columns (GEMM 5) and written to its sDST
//                    q-half tile (GEMM 4). At tile end: dV, then dK (x sm_scale), each bounced
//                    through a then-free SMEM buffer (sDO, Q stage 0) and TMA-stored per q-half.
//   epilogue (w0-3)  Per step: warp w reads lanes [32w, +32) of the dQ tile (all 128 hd columns,
//                    one q row per thread) as four x32 loads, arrives empty_bar_dq, then stages
//                    4 chunks of 32 hd columns through the 2 dQ stage buffers; w0 lane 0 issues
//                    one 16 KB cp.reduce.async.bulk per chunk with wait_group_read<1>.
//   sched (w14)      CLC scheduler: fetches the next work item for all warps (Sched::CLC);
//                    otherwise a second donor.
//   donor (w15)      No work; donates its registers.
//
// Barrier contract. arv = arrive count: 1 is one elected thread, a TMA completion or a tcgen05
// commit; 8 is one elected lane per softmax warp; 4 one per epilogue warp. A tcgen05-commit arrive
// fires when the tensor core has finished the GEMMs issued before it.
//   barrier          ring  arv  producer -> consumer  meaning
//   ---------------  ----  ---  --------------------  ---------------------------------------
//   full_bar_q        [2]   1   load     -> mma       Q tile resident; expect_tx 32 KB, +32 KB
//                                                     when step 0 also carries K.
//   empty_bar_q       [2]   1   mma      -> load      Q stage read: commit after GEMM 5.
//   full_bar_do       [1]   1   load     -> mma       dO tile resident; +32 KB V on step 0.
//   empty_bar_do      [1]   1   mma      -> load      dO read: commit after GEMM 3.
//   full_bar_lse      [2]   1   load     -> softmax   step's 128 LSE values; expect_tx 512 B.
//   empty_bar_lse     [2]   8   softmax  -> load      LSE consumed (after the P^T loop).
//   full_bar_delta     [1]  1   TMA      -> load/softmax  128 FP32 Delta values, expect_tx 512 B.
//   full_bar_delta_cvt [1]  1   load     -> softmax       packed only: BF16 Delta conversion done.
//   empty_bar_delta   [1]   8   softmax  -> load      Delta consumed (after the dS^T loop).
//   full_bar_st       [1]   1   mma      -> softmax   S^T in TMEM: commit after GEMM 1.
//   full_bar_dpt      [1]   1   mma      -> softmax   dP^T in TMEM: commit after GEMM 2.
//   full_bar_pt       [1]   8   softmax  -> mma       P^T overlay stored; GEMM 3 may read it.
//   full_bar_dst      [1]   8   softmax  -> mma       dS^T overlay and sDST written; GEMMs 5, 4.
//   full_bar_dq       [1]   1   mma      -> epilogue  dQ tile in TMEM: commit after GEMM 4.
//   empty_bar_dq      [1]   4   epilogue -> mma       dQ tile read out; the next GEMM 2 may reuse
//                                                     cols 256-383.
//   full_bar_dv       [1]   1   mma      -> softmax   tile's dV complete (commit entering the last
//                                                     step).
//   full_bar_dk       [1]   1   mma      -> softmax   tile's dK complete: commit after the tail
//                                                     GEMM 5.
//   All full_* waits are PhaseTrackers; every empty_* wait on the load warp is an
//   EmptyPhaseTracker (seeded to parity 1: the first wait passes with nobody arrived). The MMA's
//   empty_bar_dq is a PhaseTracker: dP^T(0) needs no gate (cols 256-383 start free), the first
//   wait is before dP^T(1).
//
// Named barriers and async-proxy waits:
//   bar_sync<10>(416)  w0-12     TMEM publish after the MMA's alloc; teardown before the dealloc.
//   bar_sync<11>(128)  epilogue  two per dQ chunk: after staging + fence.proxy.async (the leader
//                                then pushes), after the leader's wait_group_read<1>; once at exit.
//   bar_sync<12/13>    softmax   per q-half warpgroup, once per dV / dK tile: after staging +
//   (128)                        fence.proxy.async, before the warpgroup's TMA store.
//   bar_sync<14>(256)  softmax   zero-fill path only: around the zero dK/dV TMA stores.
//   wait_group_read    epilogue  <1> by the leader per chunk (the other stage buffer is free),
//                                <0> at exit (SMEM dies with the CTA).
//
// Budgets:
//   TMEM (128 lanes x 512 cols, fp32):
//     cols     holds   written by  also holds
//     0-127    S^T     GEMM 1      P^T bf16x2 overlay at 0-31, 64-95
//     128-255  dV      GEMM 3      -
//     256-383  dP^T    GEMM 2      dS^T bf16x2 overlay at 256-287, 320-351; dQ tile (GEMM 4)
//     384-511  dK      GEMM 5      -
//   SMEM (~226 KB): K 32K | V 32K | Q[2] 64K | dO 32K | sDST 32K | dQ stage 2x16K | LSE 2x512 B |
//     Delta 512 B | barriers. The dV / dK stores bounce through the then-free sDO and Q stage 0.
//   Registers (setmaxnreg from the 128/thread base; 4*152 + 8*136 + 2*88 + 2*24 = 1920 <= 2048):
//     warps    role        regs  does
//     w0-3     epilogue    152   drain dQ tiles from TMEM into dqaccum (cp.reduce.async.bulk)
//     w4-11    softmax     136   form P^T and dS^T; store dV, dK at tile end
//     w12      mma          88   issue every tcgen05.mma and commit
//     w13      load         88   TMA Q / dO tiles, K, V once; bulk LSE / Delta rows
//     w14      sched        88   CLC tile scheduler (Sched::CLC); 24 as a second donor otherwise
//     w15      donor        24   no work
//
// Scaling per FA4: K never pre-scaled; scale_log2 = sm_scale * log2(e) in fp32 inside the exp2;
// sm_scale on dK in the tile-end store, on dQ in the postprocess.
//
// Index contract (the padded k2q form FastVideo's invert_indices produces, and the transpose of
// the forward kernel's q2k_idx/q2k_num): k2q_idx int32 [B*H*nb, max_q_blocks] holds, per
// (batch*head, kv128) row, the LOCAL q128 block ids that select this kv block in entries
// [0, k2q_num[row]); entries past the count are never read. The harness inverts its q2k index
// into this form on the host.
// Timed path = preprocess (including device remap sort) + main + postprocess;
// TFLOPS = 2.5 * 4 * D * (B*H*nb*topk*128^2) / t.
//
// CPU reference math (fp32 throughout, inputs read bf16 -> fp32):
//   forward (log2 domain): score2_j = (q.k_j) * sm_scale * log2(e); m = max_j score2_j;
//     l = sum_j exp2(score2_j - m); O_row = sum_j exp2(score2_j - m) * v_j / l;
//     M_row = m + log2(l).
//   Delta_row = rowsum(dO_row * O_row).
//   backward: P_j = exp2(score2_j - M_row); dP_j = dO_row . v_j;
//     dQ_row = sm_scale * sum_j P_j * (dP_j - Delta_row) * k_j
//     dK_j  += sm_scale * P_j * (dP_j - Delta_row) * q_row   (kv-stationary pass, sorted k2q)
//     dV_j  += P_j * dO_row
//
// Env knobs: LOAD_NPY=<dir> (q/k/v/do_S{S}.npy + idx_S{S}_blk128.npy), SHAPE=0..5 +
// BATCH/HEADS/NB/TOPK, CPU_REF=0|1, DUMP_BWD=<prefix> (fp32 npy
// of the reference dq/dk/dv [B*S, H, D] and M/delta [B*H, S]), DUMP_BWD_GPU=<prefix>,
// VERIFY_ARGMAX, STRESS_N, BENCH_WARMUP, BENCH_ITERS, K2Q_ORDER, K2Q_COUNT_BIN,
// K2Q_ORDER_MODE (0=median, 1=quartile lexicographic, 2=quartile Morton),
// K2Q_TRAVERSAL_SNAKE, CLC_PER_HEAD, K2Q_WAVES, K2Q_WAVE_SNAKE, COOPERATIVE_GRID_SYNC.
// Build knob: KERNEL_LPT_MODE=LptMode::{OFF,PER_HEAD,GLOBAL,AUTO}; OFF by default.
// Experimental build knob: KERNEL_EXP_PACKED_DS=1; off by default due to numerical error.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
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
#include "../../../../../tests/test_utils.cuh"
#include "../../../../primitives/0_tcgen05_alloc.cuh"
#include "../../../../primitives/1_tcgen05_dealloc.cuh"
#include "../../../../primitives/2_tcgen05_relinquish.cuh"
#include "../../../../primitives/3_tcgen05_mma_f16.cuh"
#include "../../../../primitives/8_tcgen05_mma_idesc.cuh"
#include "../../../../primitives/9_tcgen05_ld.cuh"
#include "../../../../primitives/10_tcgen05_st.cuh"
#include "../../../../primitives/11_tcgen05_commit.cuh"
#include "../../../../primitives/12_tcgen05_wait.cuh"
#include "../../../../primitives/15_tcgen05_fence.cuh"
#include "../../../../primitives/18_tma_load.cuh"
#include "../../../../primitives/22_tma_store.cuh"
#include "../../../../primitives/25_tma_async_group.cuh"
#include "../../../../primitives/27_cp_async_cg.cuh"
#include "../../../../primitives/28_cp_async_commit_wait.cuh"
#include "../../../../primitives/29_mbarrier_init.cuh"
#include "../../../../primitives/30_mbarrier_arrive.cuh"
#include "../../../../primitives/31_mbarrier_arrive_tx.cuh"
#include "../../../../primitives/33_mbarrier_try_wait.cuh"
#include "../../../../primitives/34_fence_proxy_async.cuh"
#include "../../../../primitives/114_fence_acq_rel.cuh"
#include "../../../../primitives/35_fence_mbarrier_init.cuh"
#include "../../../../primitives/37_bar_sync.cuh"
#include "../../../../primitives/42_smem_desc_blackwell.cuh"
#include "../../../../primitives/44_elect_sync.cuh"
#include "../../../../primitives/46_setmaxnreg.cuh"
#include "../../../../primitives/50_atom_global.cuh"
#include "../../../../primitives/63_cvt_f32_to_f16_bf16.cuh"
#include "../../../../primitives/69_griddepcontrol.cuh"
#include "../../../../primitives/76_packed_f32x2.cuh"
#include "../../../../primitives/77_ex2_approx.cuh"
#include "../../../../composites/118_mbarrier_phase_tracking.cuh"
#include "../../../../composites/106_clc_fetch_next_tile.cuh"
#include "../../../../primitives/_warp_prof_noop.cuh"

#ifndef VSA_BHSD
#define VSA_BHSD false  // false: [B, S, H, 128]; true: FastVideo's [B, H, S, 128]
#endif
// Work scheduling (kernel template parameter SCHED, fixed per build by KERNEL_SCHED):
//   NON_PERSISTENT    1D grid over the flattened head-major item list, one work item per CTA.
//   STATIC_PERSISTENT 1D grid of min(items, #SMs) CTAs, each strides the item list by gridDim.x.
//   CLC               1D grid over all items; cancelled CTA coordinates are the next work id.
//   CLC_HEAD_MAJOR    One global CLC grid; cancelled CTAs are continuation tokens and a global
//                     logical counter assigns all KV blocks of head 0, then head 1, and so on.
//   DEFENSIVE         One cooperative grid of <= #SMs CTAs per head. Each CTA executes one
//                     ordered item per round; a grid barrier preserves hard cache-wave bounds.
// A CLC_HEAD_MAJOR build may instead use the host-only CLC_PER_HEAD=1 launch policy: one grid,
// head base, and logical counter per batch-head. It is not a device scheduling mode.
enum class Sched { NON_PERSISTENT, STATIC_PERSISTENT, CLC, CLC_HEAD_MAJOR, DEFENSIVE };
#ifndef KERNEL_SCHED
#define KERNEL_SCHED Sched::NON_PERSISTENT
#endif
// Longest-item-first ordering applies only to the non-persistent grid. AUTO selects global
// ordering for small total sequences, per-head ordering through 16K, and otherwise disables it.
enum class LptMode { OFF, PER_HEAD, GLOBAL, AUTO };
#ifndef KERNEL_LPT_MODE
#define KERNEL_LPT_MODE LptMode::OFF
#endif
static_assert(KERNEL_LPT_MODE == LptMode::OFF || KERNEL_SCHED == Sched::NON_PERSISTENT,
              "LPT ordering requires NON_PERSISTENT scheduling");
#ifndef KERNEL_EXP_PACKED_DS
#define KERNEL_EXP_PACKED_DS false
#endif
// Programmatic dependent launch (KERNEL_PDL): preprocess -> main -> postprocess each start while
// the predecessor drains; griddepcontrol.wait sits in front of every read of the predecessor's data.
#ifndef KERNEL_PDL
#define KERNEL_PDL true
#endif

constexpr int BLOCK           = 128;         // q and kv block size (tokens); tile == block
constexpr int KV_TILE         = BLOCK;       // kv rows owned by one CTA
constexpr int Q_TILE          = BLOCK;       // q rows per step
constexpr int HEAD_DIM        = 128;
constexpr int SUB_COLS_BF16   = 64;  // one 128B-swizzle unit
constexpr int SUB_COLS_BYTES  = SUB_COLS_BF16 * (int)sizeof(__nv_bfloat16);  // 128 B
constexpr int HD_SUBTILES     = HEAD_DIM / SUB_COLS_BF16;  // 2 hd-halves of a K, V, Q or dO tile
constexpr int KV_SUB_COLS_BYTES = KV_TILE * SUB_COLS_BYTES;  // 16 KB: K or V, one 64-hd subtile
constexpr int KV_TILE_BYTES     = HD_SUBTILES * KV_SUB_COLS_BYTES;  // 32 KB: 128 kv x 128 hd
constexpr int Q_SUB_COLS_BYTES  = Q_TILE * SUB_COLS_BYTES;   // 16 KB: Q or dO, one 64-hd subtile
constexpr int Q_TILE_BYTES      = HD_SUBTILES * Q_SUB_COLS_BYTES;  // 32 KB: 128 q x 128 hd
constexpr int NUM_Q_STAGES      = 2;                         // Q double-buffered; dO single
constexpr int DST_TILE_BYTES    = KV_TILE * SUB_COLS_BYTES;  // 16 KB: dS^T of one q-half
constexpr int DST_BYTES         = 2 * DST_TILE_BYTES;        // 32 KB sDST, q-half 0 then 1
constexpr int MMA_K               = 16;                      // bf16 tcgen05.mma K per issue
constexpr int K_ATOMS_PER_SUBTILE = SUB_COLS_BF16 / MMA_K;   // 4: GEMM 1/2, hd within a subtile
constexpr int K_ATOMS_PER_Q_HALF  = SUB_COLS_BF16 / MMA_K;   // 4: the 64 q of one q-half
constexpr int K_ATOMS_PER_KV_TILE = KV_TILE / MMA_K;         // 8: GEMM 4, the 128 kv rows
constexpr int BF16X2_COLS_PER_K16 = MMA_K / 2;               // TMEM columns per k16, bf16x2 tile

// dQ drain: the dQ tile leaves TMEM whole (128 hd fp32 per thread), then goes to dqaccum in
// slices of COLS hd columns, each staged in one of two SMEM buffers and pushed as one
// cp.reduce.async.bulk.
struct DQConfig {
  static constexpr int COLS = 32;
  // One push = the tile's 128 rows of one slice.
  static constexpr int DQ_ONE_PUSH_BYTES = Q_TILE * COLS * (int)sizeof(float);  // 16 KB
  static constexpr int DQ_STAGE_BYTES    = DQ_ONE_PUSH_BYTES;
  static constexpr int DQ_STAGE_BUFFERS  = 2;                  // double-buffered stage
  static constexpr int DQ_BLOCK_ELEMS    = Q_TILE * HEAD_DIM;  // accumulator elems per q128 block
};
using DQ = DQConfig;

constexpr int N_WARPS = 16;
constexpr int W_EPI0 = 0, W_SOFTMAX0 = 4, W_MMA = 12, W_LOAD = 13;
constexpr int W_SCHED = 14;   // CLC scheduler warp (Sched::CLC); a second donor otherwise
constexpr int CLC_STAGES   = 2;
constexpr int CLC_ARRIVALS = 15;  // worker warps 0-13 + sched consumer fetch

// SMEM (~226 KB): K 32K | V 32K | Q[2] 64K | dO 32K | sDST 32K | dQ stage 2x16K | LSE 2x512 B |
// Delta 512 B | barriers (incl. CLC + logical-id publication) | CLC responses | scalar slots.
// The dV / dK stores bounce through the then-free sDO and Q stage 0. 48 covers alignment/slots.
constexpr int NUM_BARS   = 4 * NUM_Q_STAGES + 14 + 3 * CLC_STAGES;
constexpr int SMEM_TOTAL = 2 * KV_TILE_BYTES + NUM_Q_STAGES * Q_TILE_BYTES + Q_TILE_BYTES +
                           DST_BYTES + DQ::DQ_STAGE_BUFFERS * DQ::DQ_STAGE_BYTES +
                           (NUM_Q_STAGES + 1) * Q_TILE * (int)sizeof(float) + NUM_BARS * 8 +
                           CLC_STAGES * 16 + 48;

constexpr int ST_COLS      = Q_TILE;         // S^T / dP^T tile: 128 q columns
constexpr int ST_HALF_COLS = SUB_COLS_BF16;  // fp32 columns of one q-half (one softmax warp)
constexpr int DV_COLS      = HEAD_DIM;
constexpr int DK_COLS      = HEAD_DIM;
constexpr int TMEM_TOTAL   = ST_COLS + DV_COLS + ST_COLS + DK_COLS;  // S^T | dV | dP^T | dK
static_assert(ST_COLS == 2 * ST_HALF_COLS, "two q-halves per tile");
static_assert(TMEM_TOTAL == 512, "TMEM map must fill exactly 512 columns");

extern __shared__ __align__(1024) uint8_t bwd_smem[];

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
  int batch;
  int head;
  int kv_block_id_in_seq;    // kv128 block within the sequence (token row kv_block_id_in_seq * 128)
  int batch_head;            // batch * num_heads + head: the [B*H, ...] row of LSE/Delta/dqaccum
  const int* local_k2q_idx;  // this row's q-list: k2q_idx + item * max_q_blocks (size_t offset:
                             // with max_q_blocks == nb, B*H*nb*nb can exceed 2^31)
  int local_k2q_num;         // q-list length in q128 blocks (= k2q_num[item])
};

template <bool FIRST_DESCENDING, bool TRAVERSAL_SNAKE>
__device__ __forceinline__ int k2q_block_at(const WorkItem& it, int step, int item_ordinal) {
  const bool descending =
      FIRST_DESCENDING ^ (TRAVERSAL_SNAKE && ((item_ordinal & 1) != 0));
  const int index = descending ? it.local_k2q_num - 1 - step : step;
  return it.local_k2q_idx[index];
}

// Work id -> item, one formula for every schedule: global id = batch_head_base * nb + workitem_id,
// head = global id / nb, kv block = ORDERED ? remap[global id] : global id % nb. Global LPT
// entries encode (batch_head + 1) << 16 | kv, allowing items to move across heads. Per schedule:
//   schedule                 grid          workitem_id (first / next)        base  remap  ORDERED
//   NON_PERSISTENT           B*H*nb        blockIdx.x / one item            0     LPT?   LPT?
//   NON_PERSISTENT waves     <= #SMs each  wave_begin + blockIdx.x / one    0     perm   true
//   STATIC_PERSISTENT        min(items,SM) blockIdx.x / += gridDim.x        0     null   false
//   CLC                      B*H*nb        blockIdx.x / cancelled CTA's x   0     null   false
//   CLC_HEAD_MAJOR           B*H*nb        counter / counter per token      0     perm?  K2Q_ORDER
//   CLC_HEAD_MAJOR per head  nb per head   head counter / per token         head  perm?  K2Q_ORDER
//   DEFENSIVE                <= #SMs/head  blockIdx.x / += gridDim.x        head  perm?  K2Q_ORDER
// perm = the head-major KV permutation [B*H*nb]; item row = head * nb + kv (size_t offset:
// B*H*nb*nb can exceed 2^31).
template <bool ORDERED_KV_BLOCKS, bool PADDED_DEFENSIVE_WAVE = false>
__device__ __forceinline__ WorkItem decode_workitem(int workitem_id, int batch_head_base,
                                                    const int* __restrict__ workitem_remap,
                                                    const int* __restrict__ k2q_idx,
                                                    const int* __restrict__ k2q_num,
                                                    int max_q_blocks, int num_heads,
                                                    int num_kv_blocks_per_seq) {
  // A cooperative grid must execute the same number of barriers in every CTA. The final round is
  // padded to gridDim.x; its inactive lanes use -1 as a private sentinel and touch no data.
  if constexpr (PADDED_DEFENSIVE_WAVE) {
    if (workitem_id >= num_kv_blocks_per_seq) return {0, 0, 0, 0, nullptr, -1};
  }
  const int global_workitem_id = batch_head_base * num_kv_blocks_per_seq + workitem_id;
  int batch_head               = global_workitem_id / num_kv_blocks_per_seq;
  int kv_block_id_in_seq       = global_workitem_id % num_kv_blocks_per_seq;
  if constexpr (ORDERED_KV_BLOCKS) {
    const int entry = workitem_remap[global_workitem_id];
    if (entry >= 0x10000) {
      batch_head         = (entry >> 16) - 1;
      kv_block_id_in_seq = entry & 0xffff;
    } else {
      kv_block_id_in_seq = entry;
    }
  }
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
//   SCHED   : work scheduling, see enum Sched. NON_PERSISTENT runs the work-item loop once and
//             touches no cross-item barrier; the two persistent forms hand K/V, the store bounce
//             and the dV / dK / dQ TMEM columns from one item to the next (empty_bar_kv,
//             empty_bar_epi, the prologue empty_bar_dq wait with its one-time prime); CLC adds
//             the try_cancel ring and the w14 scheduler warp (otherwise w14 is a second donor).
//             An item with an empty q-list touches none of these barriers in any form: the
//             epilogue warps store its zero dK / dV tiles from their own stage buffers. Fixed
//             per build by KERNEL_SCHED.
//             DEFENSIVE uses the persistent hand-offs and a cooperative grid barrier after every
//             item, so no CTA can enter the next ordered cache wave early.
//   K2Q_FIRST_DESCENDING: direction for CTA-local work item 0: false = low-to-high,
//                         true = high-to-low. Without traversal snake, every item uses it.
//   ORDERED_KV_BLOCKS: false maps logical id directly to the head-local KV id; true reads the
//                      head-local KV permutation supplied in workitem_remap.
//   K2Q_TRAVERSAL_SNAKE: when true, XOR K2Q_FIRST_DESCENDING with each persistent CTA's local
//                        work-item parity, alternating direction between consecutive items.
//   USE_COOPERATIVE_GRID_SYNC: DEFENSIVE only; true uses cooperative_groups::this_grid().sync(),
//                              false uses the faster one-atomic-per-CTA barrier.
//   Ordered NON_PERSISTENT is the SM-count-wave specialization: each CTA executes exactly one
//   remapped WI at workitem_begin + blockIdx.x. It contains no CLC or persistent hand-off path.
// Kernel arguments (activations bf16 in the BHSD-selected layout unless stated):
//   tmap_q, tmap_do    : Q, dO as 3D maps [64 hd, B*S tokens, H*2 hd units], box (64, 128, 1)
//                        (BHSD: 4D [64 hd, S, 2 hd units, B*H], box (64, 128, 1, 1)): one q128
//                        block = 2 TMAs (one per hd subtile).
//   tmap_k, tmap_v     : K, V, same geometry; the CTA loads its kv128 block once (2 TMAs each).
//   tmap_dk, tmap_dv   : dK, dV output maps, same geometry (TMA stores, bf16).
//   dqaccum            : fp32 [B*H, nb, 128*128] drain-native dQ accumulator; zeroed by the
//                        preprocess, reduce-added here, unscrambled by the postprocess.
//   lse_rows           : fp32 [B*H, S], the forward's LSE in log2 form (M = max + log2(l)).
//   delta_rows         : fp32 [B*H, S], Delta = rowsum(bf16(O) * dO) from preprocess.
//   k2q_idx, k2q_num   : padded k2q lists (Index contract above): row item = (b*H + h)*nb + kv
//                        holds k2q_num[item] q128 block ids at k2q_idx[item * max_q_blocks + i].
//   max_q_blocks       : k2q_idx row stride.
//   workitem_remap     : logical CLC receives logical-id -> local-KV-id entries. Ordered
//                        NON_PERSISTENT waves receive the global head-major array plus an offset.
//                        Ordinary NON_PERSISTENT receives nullptr.
//   clc_work_counter   : logical-CLC launch-local counter, reset before launch.
//   batch_head_base    : global batch-head offset for the launch (0 for the global grid).
//   num_samples, num_heads, seqlen : B, H, S; nb = num_kv_blocks_per_seq = S/128 is derived.
//   scale_log2         : sm_scale * log2(e), applied to S^T before exp2.
//   sm_scale           : applied to dK in the tile-end store (dQ gets it in the postprocess).
//   DK_FROM_SMEM       : keep dS^T only in sDST, feed dK with SS MMA, and issue dQ before dK.
//   EXP_PACKED_DS      : round P, dP, and Delta to BF16 before packed BF16x2 dS arithmetic.
template <bool BHSD = false, Sched SCHED = Sched::NON_PERSISTENT,
          bool K2Q_FIRST_DESCENDING = false, bool ORDERED_KV_BLOCKS = false,
          bool K2Q_TRAVERSAL_SNAKE = false, bool USE_COOPERATIVE_GRID_SYNC = false,
          bool DK_FROM_SMEM = false, bool EXP_PACKED_DS = false>
__global__ void __cluster_dims__(1, 1, 1) __launch_bounds__(N_WARPS * 32, 1)
    vsa_bwd_main_kernel(const __grid_constant__ CUtensorMap tmap_q,
                        const __grid_constant__ CUtensorMap tmap_k,
                        const __grid_constant__ CUtensorMap tmap_v,
                        const __grid_constant__ CUtensorMap tmap_do,
                        const __grid_constant__ CUtensorMap tmap_dk,
                        const __grid_constant__ CUtensorMap tmap_dv, float* __restrict__ dqaccum,
                        const float* __restrict__ lse_rows, const float* __restrict__ delta_rows,
                        const int* __restrict__ k2q_idx, const int* __restrict__ k2q_num,
                        int max_q_blocks, const int* __restrict__ workitem_remap,
                        uint32_t* __restrict__ clc_work_counter, int workitem_begin,
                        int batch_head_base, int num_samples, int num_heads, int seqlen,
                        float scale_log2, float sm_scale) {
  constexpr bool PERSISTENT    = SCHED != Sched::NON_PERSISTENT;
  constexpr bool CLC           = SCHED == Sched::CLC || SCHED == Sched::CLC_HEAD_MAJOR;
  constexpr bool LOGICAL_CLC   = SCHED == Sched::CLC_HEAD_MAJOR;
  constexpr bool DEFENSIVE     = SCHED == Sched::DEFENSIVE;

  const int num_kv_blocks_per_seq = seqlen / BLOCK;
  [[maybe_unused]] const int total_workitems = num_samples * num_heads * num_kv_blocks_per_seq;
  [[maybe_unused]] const int defensive_padded_workitems =
      ((num_kv_blocks_per_seq + (int)gridDim.x - 1) / (int)gridDim.x) * (int)gridDim.x;
  uint8_t* sK                = bwd_smem;
  uint8_t* sV                = sK + KV_TILE_BYTES;
  uint8_t* sQ[NUM_Q_STAGES]  = {sV + KV_TILE_BYTES, sV + KV_TILE_BYTES + Q_TILE_BYTES};
  uint8_t* sDO               = sQ[0] + NUM_Q_STAGES * Q_TILE_BYTES;
  __nv_bfloat16* sDST        = reinterpret_cast<__nv_bfloat16*>(sDO + Q_TILE_BYTES);
  uint8_t* sDQ_STAGE_bytes   = sDO + Q_TILE_BYTES + DST_BYTES;
  float* sDQ_STAGE[DQ::DQ_STAGE_BUFFERS] = {
      reinterpret_cast<float*>(sDQ_STAGE_bytes),
      reinterpret_cast<float*>(sDQ_STAGE_bytes + DQ::DQ_STAGE_BYTES)};
  float* sLSE =
      reinterpret_cast<float*>(sDQ_STAGE_bytes + DQ::DQ_STAGE_BUFFERS * DQ::DQ_STAGE_BYTES);
  float* sDelta = sLSE + NUM_Q_STAGES * Q_TILE;  // sLSE: [stage][128]; sDelta: [128]

  // mbarriers (arrival counts at the init below)
  uint64_t* full_bar_q      = reinterpret_cast<uint64_t*>(sDelta + Q_TILE);  // [stage] TMA tx
  uint64_t* empty_bar_q     = full_bar_q + NUM_Q_STAGES;     // [stage] GEMM 5 commit
  uint64_t* full_bar_do     = empty_bar_q + NUM_Q_STAGES;    // TMA tx
  uint64_t* empty_bar_do    = full_bar_do + 1;               // GEMM 3 commit
  uint64_t* full_bar_lse    = empty_bar_do + 1;              // [stage] load -> softmax (128 f32)
  uint64_t* empty_bar_lse   = full_bar_lse + NUM_Q_STAGES;   // [stage] lane 0 of each softmax warp
  uint64_t* full_bar_delta     = empty_bar_lse + NUM_Q_STAGES;  // TMA -> load / softmax
  uint64_t* full_bar_delta_cvt = full_bar_delta + 1;            // load -> softmax (packed only)
  uint64_t* empty_bar_delta    = full_bar_delta + (EXP_PACKED_DS ? 2 : 1);
  uint64_t* full_bar_st     = empty_bar_delta + 1;           // commit after the S^T atoms
  uint64_t* full_bar_dpt    = full_bar_st + 1;               // commit after the dP^T atoms
  uint64_t* full_bar_pt     = full_bar_dpt + 1;              // P^T overlay stored (gates dV)
  uint64_t* full_bar_dst    = full_bar_pt + 1;               // dS^T overlay + sDST written
  uint64_t* full_bar_dq     = full_bar_dst + 1;              // commit after the dQ GEMM
  uint64_t* empty_bar_dq    = full_bar_dq + 1;               // epilogue warps read the dQ tile
  uint64_t* full_bar_dv     = empty_bar_dq + 1;              // all dV accumulation issued
  uint64_t* full_bar_dk     = full_bar_dv + 1;               // last dK issued
  // Persistent only, one hand-off per work item: empty_bar_kv is the MMA's commit after the item's
  // last dQ GEMM (K, V and sDST are read out); empty_bar_epi is the softmax warps' arrive after
  // their dV / dK stores have read SMEM (the sDO / sQ[0] bounce and the dV / dK TMEM columns).
  uint64_t* empty_bar_kv    = full_bar_dk + 1;
  uint64_t* empty_bar_epi   = empty_bar_kv + 1;
  uint64_t* clc_full        = empty_bar_epi + 1;             // [CLC_STAGES]
  uint64_t* clc_empty       = clc_full + CLC_STAGES;          // [CLC_STAGES]
  uint64_t* clc_logical_full = clc_empty + CLC_STAGES;        // [CLC_STAGES], load -> workers
  uint32_t* clc_response    = reinterpret_cast<uint32_t*>(
      (reinterpret_cast<uintptr_t>(clc_logical_full + CLC_STAGES) + 15u) & ~uintptr_t(15u));
  uint32_t* tmem_slot       = clc_response + CLC_STAGES * 4;
  uint32_t* clc_initial_work = tmem_slot + 1;
  uint32_t* clc_logical_work = clc_initial_work + 1;          // [CLC_STAGES]

  const int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;

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
    if constexpr (EXP_PACKED_DS) mbarrier_init(smem_ptr_u32(full_bar_delta_cvt), 1);
    mbarrier_init(smem_ptr_u32(empty_bar_delta), 8);
    mbarrier_init(smem_ptr_u32(full_bar_st), 1);
    mbarrier_init(smem_ptr_u32(full_bar_dpt), 1);
    mbarrier_init(smem_ptr_u32(full_bar_pt), 8);
    mbarrier_init(smem_ptr_u32(full_bar_dst), 8);
    mbarrier_init(smem_ptr_u32(full_bar_dq), 1);
    mbarrier_init(smem_ptr_u32(empty_bar_dq), 4);
    mbarrier_init(smem_ptr_u32(full_bar_dv), 1);
    mbarrier_init(smem_ptr_u32(full_bar_dk), 1);
    if constexpr (PERSISTENT) {
      mbarrier_init(smem_ptr_u32(empty_bar_kv), 1);
      mbarrier_init(smem_ptr_u32(empty_bar_epi), 8 * 32);  // every softmax thread arrives
    }
    if constexpr (CLC) {
      #pragma unroll
      for (int st = 0; st < CLC_STAGES; ++st) {
        mbarrier_init(smem_ptr_u32(&clc_full[st]), 1);
        mbarrier_init(smem_ptr_u32(&clc_empty[st]), CLC_ARRIVALS);
        mbarrier_init(smem_ptr_u32(&clc_logical_full[st]), 1);
      }
      #pragma unroll
      for (int i = 0; i < CLC_STAGES * 4; ++i) clc_response[i] = 0;
      if constexpr (LOGICAL_CLC) {
        // Each launched CTA owns one logical item; a cancellation is only a continuation token.
        *clc_initial_work = atom_global_add_u32(clc_work_counter, 1u);
      }
    }
  }
  fence_mbarrier_init_release_cluster();
  __syncthreads();
  // Delta, the zeroed dqaccum and the k2q metadata come from earlier grids: every warp waits.
  if constexpr (KERNEL_PDL) griddepcontrol_wait();

  WpCtx wpc = wp_ctx_init();

  // Cooperative residency makes this CTA-leader barrier deadlock-free. The functionally correct
  // cooperative_groups::this_grid().sync() form reduces S524K throughput from
  // 1187.6 to 822.8 TFLOP/s. One atomic arrival per CTA avoids its all-thread grid-barrier cost.
  // A monotonic counter avoids a separate sense word: round r releases at
  // (r + 1) * gridDim.x arrivals. The surrounding CTA barriers make all 16 warps finish the
  // current item before the leader arrives and keep them parked until every CTA is ready.
  auto defensive_wave_barrier = [&](int workitem_id) {
    if constexpr (DEFENSIVE) {
      if constexpr (USE_COOPERATIVE_GRID_SYNC) {
        cooperative_groups::this_grid().sync();
      } else {
        __syncthreads();
        if (tid == 0) {
          atom_global_add_u32(clc_work_counter, 1u);
          const uint32_t target =
              (uint32_t)(workitem_id / (int)gridDim.x + 1) * (uint32_t)gridDim.x;
          while (atom_global_add_u32(clc_work_counter, 0u) < target) __nanosleep(64);
        }
        __syncthreads();
      }
    }
  };

  // Next work id (-1 = done): LOGICAL_CLC reads the sched warp's id, CLC the token, STATIC strides.
  auto get_next_workitem_id = [&](int workitem_id, int& clc_stage, uint32_t& clc_phase) {
    if constexpr (LOGICAL_CLC) {
      mbarrier_wait_parity_suspend(smem_ptr_u32(&clc_logical_full[clc_stage]), clc_phase);
      const uint32_t logical_workitem_id = clc_logical_work[clc_stage];
      // Complete the read before the release hands the stage back to the sched warp.
      fence_acq_rel_cta();
      if (elect_one_sync()) mbarrier_arrive(smem_ptr_u32(&clc_empty[clc_stage]));
      clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
      workitem_id = logical_workitem_id == ~0u ? -1 : (int)logical_workitem_id;
    } else if constexpr (CLC) {
      ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, 1, true>(
          clc_full, clc_empty, clc_response, clc_stage, clc_phase, elect_one_sync());
      clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
      workitem_id = next.valid ? (int)next.n_tile : -1;
    } else if constexpr (DEFENSIVE) {
      defensive_wave_barrier(workitem_id);
      workitem_id += (int)gridDim.x;
      if (workitem_id >= defensive_padded_workitems) workitem_id = -1;
    } else if constexpr (PERSISTENT) {
      workitem_id += (int)gridDim.x;
      if (workitem_id >= total_workitems) workitem_id = -1;
    }
    return workitem_id;
  };

  if (warp_id == W_LOAD) {
    setmaxnreg_dec<88>();

    EmptyPhaseTracker<NUM_Q_STAGES> q_empty_ph, lse_empty_ph;
    EmptyPhaseTracker<1> do_empty_ph, delta_empty_ph;
    [[maybe_unused]] PhaseTracker<1> delta_load_ph;
    [[maybe_unused]] EmptyPhaseTracker<1> epi_empty_ph, kv_empty_ph;
    int clc_stage = 0;
    uint32_t clc_phase = 0;

    int workitem_id = !PERSISTENT ? workitem_begin + (int)blockIdx.x
                                  : (LOGICAL_CLC ? (int)*clc_initial_work
                                                 : (int)blockIdx.x);
    int item_ordinal = 0;
    do {
      wp_marker(wpc, WP_ITEM, workitem_id);
      const WorkItem it = decode_workitem<ORDERED_KV_BLOCKS, DEFENSIVE>(
          workitem_id, batch_head_base, workitem_remap, k2q_idx, k2q_num, max_q_blocks,
          num_heads, num_kv_blocks_per_seq);

      if (it.local_k2q_num > 0) {
        auto load_tile = [&](uint8_t* dst, const CUtensorMap* map, uint64_t* full_bar,
                             int token_begin) {
          static_assert(Q_SUB_COLS_BYTES == KV_SUB_COLS_BYTES,
                        "one subtile stride for Q/dO and K/V");
          #pragma unroll
          for (int s = 0; s < HD_SUBTILES; ++s) {
            if constexpr (BHSD)
              tma_load_4d(smem_ptr_u32(dst + s * Q_SUB_COLS_BYTES), map, smem_ptr_u32(full_bar),
                          0, token_begin, s, it.batch_head);
            else
              tma_load_3d(smem_ptr_u32(dst + s * Q_SUB_COLS_BYTES), map, smem_ptr_u32(full_bar),
                          0, it.batch * seqlen + token_begin, it.head * HD_SUBTILES + s);
          }
        };
        auto load_lse = [&](int qblock) {
          const int stage = lse_empty_ph.get_stage();
          wp_begin(wpc, WP_LOAD_WAIT_EMPTY_LSE);
          mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_lse[stage]),
                                       lse_empty_ph.get_phase());
          lse_empty_ph.advance();
          wp_end(wpc, WP_LOAD_WAIT_EMPTY_LSE);

          wp_begin(wpc, WP_LOAD_ISSUE_LSE);
          if (elect_one_sync()) {
            mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar_lse[stage]),
                                      Q_TILE * (int)sizeof(float));
            cpasync_bulk_load_mbarrier(
                smem_ptr_u32(sLSE + stage * Q_TILE),
                lse_rows + (size_t)it.batch_head * seqlen + (size_t)qblock * Q_TILE,
                Q_TILE * sizeof(float), smem_ptr_u32(&full_bar_lse[stage]));
          }
          wp_end(wpc, WP_LOAD_ISSUE_LSE);
        };
        auto load_delta = [&](int qblock) {
          wp_begin(wpc, WP_LOAD_WAIT_EMPTY_DELTA);
          mbarrier_wait_parity_suspend(smem_ptr_u32(empty_bar_delta), delta_empty_ph.get_phase());
          delta_empty_ph.advance();
          wp_end(wpc, WP_LOAD_WAIT_EMPTY_DELTA);

          wp_begin(wpc, WP_LOAD_ISSUE_DELTA);
          if (elect_one_sync()) {
            mbarrier_arrive_expect_tx(smem_ptr_u32(full_bar_delta), Q_TILE * (int)sizeof(float));
            cpasync_bulk_load_mbarrier(
                smem_ptr_u32(sDelta),
                delta_rows + (size_t)it.batch_head * seqlen + (size_t)qblock * Q_TILE,
                Q_TILE * sizeof(float), smem_ptr_u32(full_bar_delta));
          }
          if constexpr (EXP_PACKED_DS) {
            mbarrier_wait_parity_suspend(smem_ptr_u32(full_bar_delta),
                                         delta_load_ph.get_phase());
            delta_load_ph.advance();
            // Capture all 512 FP32 bytes before overwriting the first 256 bytes with BF16.
            const float4 delta4 = reinterpret_cast<const float4*>(sDelta)[lane];
            __syncwarp();
            uint2 packed;
            packed.x = cvt_f32x2_to_bf16x2(delta4.x, delta4.y);
            packed.y = cvt_f32x2_to_bf16x2(delta4.z, delta4.w);
            reinterpret_cast<uint2*>(sDelta)[lane] = packed;
            __syncwarp();
            if (elect_one_sync()) mbarrier_arrive(smem_ptr_u32(full_bar_delta_cvt));
          }
          wp_end(wpc, WP_LOAD_ISSUE_DELTA);
        };
        // Q[stage] (+ K on the item's first step).
        auto load_q = [&](int qblock, auto with_kv_const) {
          constexpr bool with_kv = decltype(with_kv_const)::value;
          const int stage = q_empty_ph.get_stage();
          wp_begin(wpc, WP_LOAD_WAIT_EMPTY_Q);
          mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_q[stage]), q_empty_ph.get_phase());
          q_empty_ph.advance();
          wp_end(wpc, WP_LOAD_WAIT_EMPTY_Q);

          wp_begin(wpc, WP_LOAD_ISSUE_Q);
          if (elect_one_sync()) {
            mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar_q[stage]),
                                      Q_TILE_BYTES + (with_kv ? KV_TILE_BYTES : 0));
            load_tile(sQ[stage], &tmap_q, &full_bar_q[stage], qblock * Q_TILE);
            if constexpr (with_kv)
              load_tile(sK, &tmap_k, &full_bar_q[stage], it.kv_block_id_in_seq * KV_TILE);
          }
          wp_end(wpc, WP_LOAD_ISSUE_Q);
        };
        // dO (+ V on the item's first step).
        auto load_do = [&](int qblock, auto with_kv_const) {
          constexpr bool with_kv = decltype(with_kv_const)::value;
          wp_begin(wpc, WP_LOAD_WAIT_EMPTY_DO);
          mbarrier_wait_parity_suspend(smem_ptr_u32(empty_bar_do), do_empty_ph.get_phase());
          do_empty_ph.advance();
          wp_end(wpc, WP_LOAD_WAIT_EMPTY_DO);

          wp_begin(wpc, WP_LOAD_ISSUE_V);
          if (elect_one_sync()) {
            mbarrier_arrive_expect_tx(smem_ptr_u32(full_bar_do),
                                      Q_TILE_BYTES + (with_kv ? KV_TILE_BYTES : 0));
            load_tile(sDO, &tmap_do, full_bar_do, qblock * Q_TILE);
            if constexpr (with_kv)
              load_tile(sV, &tmap_v, full_bar_do, it.kv_block_id_in_seq * KV_TILE);
          }
          wp_end(wpc, WP_LOAD_ISSUE_V);
        };

        // Persistent: sDO and sQ[0] are the previous item's dV / dK store bounce; sK and sV are
        // still read by its last dQ GEMM. Only items with work take part in these hand-offs:
        // each wait is answered by a producer whose work depends on this item's loads, so the
        // producer can never run a phase ahead of a late consumer (an empty item has no such
        // chain, so it must touch no barrier).
        if constexpr (PERSISTENT) {
          wp_begin(wpc, WP_LOAD_WAIT_EMPTY_EPI);
          mbarrier_wait_parity_suspend(smem_ptr_u32(empty_bar_epi), epi_empty_ph.get_phase());
          epi_empty_ph.advance();
          wp_end(wpc, WP_LOAD_WAIT_EMPTY_EPI);

          wp_begin(wpc, WP_LOAD_WAIT_EMPTY_KV);
          mbarrier_wait_parity_suspend(smem_ptr_u32(empty_bar_kv), kv_empty_ph.get_phase());
          kv_empty_ph.advance();
          wp_end(wpc, WP_LOAD_WAIT_EMPTY_KV);
        }

        // Step 0 is peeled because it also brings in the item-resident K/V tiles. Mark it like
        // the loop below; the four helpers retain their per-operation regions.
        wp_marker(wpc, WP_ITER, 0);
        const int first_qblock =
            k2q_block_at<K2Q_FIRST_DESCENDING, K2Q_TRAVERSAL_SNAKE>(it, 0, item_ordinal);
        load_q(first_qblock, std::true_type{});
        load_lse(first_qblock);
        load_do(first_qblock, std::true_type{});
        load_delta(first_qblock);
        for (int j = 1; j < it.local_k2q_num; ++j) {
          wp_marker(wpc, WP_ITER, j);

          const int qblock =
              k2q_block_at<K2Q_FIRST_DESCENDING, K2Q_TRAVERSAL_SNAKE>(it, j, item_ordinal);
          load_q(qblock, std::false_type{});
          load_lse(qblock);
          load_do(qblock, std::false_type{});
          load_delta(qblock);
        }
      }

      ++item_ordinal;
      workitem_id = get_next_workitem_id(workitem_id, clc_stage, clc_phase);
    } while (PERSISTENT && workitem_id >= 0);
    wp_flush(wpc);
    return;
  }
  else if (warp_id == W_MMA) {
    setmaxnreg_dec<88>();

    // TMEM alloc protocol (FA4): the MMA warp allocates after the init sync and publishes via a
    // 13-warp named barrier (w0-12); the load warp never waits on it and starts TMA immediately.
    tcgen05_alloc<1>(smem_ptr_u32(tmem_slot), TMEM_TOTAL);
    bar_sync<10>(416);
    const uint32_t tmem_base    = *tmem_slot;
    const uint32_t tmem_st      = tmem_base;           // S^T; P^T bf16 overlay
    const uint32_t tmem_dv      = tmem_st + ST_COLS;   // dV
    const uint32_t tmem_dpt     = tmem_dv + DV_COLS;   // dP^T; dS^T overlay; dQ tile reuses it
    const uint32_t tmem_dk      = tmem_dpt + ST_COLS;  // dK
    const uint32_t tmem_pt_bf16 = tmem_st, tmem_dst_bf16 = tmem_dpt, tmem_dq = tmem_dpt;
    const uint32_t lead         = elect_one_sync() ? 1u : 0u;

    // SMEM descriptors: 128B swizzle, SBO 1024 B (8 rows of 128 B). K-major reads (GEMMs 1, 2:
    // contraction over hd, the contiguous dim) take LBO 16; MN-major reads (GEMMs 3, 4, 5:
    // contraction over tokens) take LBO = the 64-column subtile stride.
    constexpr uint32_t DESC_SBO = 1024, DESC_LBO = 16;
    auto make_smem_desc = [](const void* smem, uint32_t leading_byte_offset) {
      return build_smem_desc_blackwell(smem_ptr_u32(smem), DESC_SBO, leading_byte_offset,
                                       SmemSwizzleBlackwell::B128);
    };
    const uint64_t desc_k      = make_smem_desc(sK, DESC_LBO);
    const uint64_t desc_v      = make_smem_desc(sV, DESC_LBO);
    const uint64_t desc_q0     = make_smem_desc(sQ[0], DESC_LBO);
    const uint64_t desc_do     = make_smem_desc(sDO, DESC_LBO);
    const uint64_t desc_q0_mn  = make_smem_desc(sQ[0], Q_SUB_COLS_BYTES);
    const uint64_t desc_do_mn  = make_smem_desc(sDO, Q_SUB_COLS_BYTES);
    const uint64_t desc_k_mn   = make_smem_desc(sK, KV_SUB_COLS_BYTES);
    const uint64_t desc_dst_mn = make_smem_desc(sDST, DST_TILE_BYTES);
    const uint64_t desc_dst_k  = make_smem_desc(sDST, DESC_LBO);
    // Descriptor address deltas (16-byte units): one k16 along the contiguous hd dim, one k16
    // along tokens (16 rows of 128 B), one 64-hd subtile of Q / dO and of K / V, one Q stage.
    constexpr uint64_t K16_COLS_DELTA = (MMA_K * (int)sizeof(__nv_bfloat16)) >> 4;
    constexpr uint32_t K16_ROWS_DELTA = (MMA_K * SUB_COLS_BYTES) >> 4;
    constexpr uint64_t Q_SUB_DELTA    = Q_SUB_COLS_BYTES >> 4;
    constexpr uint64_t KV_SUB_DELTA   = KV_SUB_COLS_BYTES >> 4;
    constexpr uint64_t DST_SUB_DELTA  = DST_TILE_BYTES >> 4;
    constexpr uint64_t Q_STAGE_DELTA  = Q_TILE_BYTES >> 4;

    const uint32_t idesc_st_dpt = make_idesc_bf16_f32(KV_TILE, Q_TILE, false, false);
    const uint32_t idesc_dv_dk  = make_idesc_bf16_f32(KV_TILE, HEAD_DIM, false, true);
    const uint32_t idesc_dq     = make_idesc_bf16_f32(Q_TILE, HEAD_DIM, true, true);

    PhaseTracker<NUM_Q_STAGES> q_full_ph;
    PhaseTracker<1> do_full_ph, pt_ph, dst_ph, dq_empty_ph;
    [[maybe_unused]] EmptyPhaseTracker<1> epi_empty_ph;
    int clc_stage = 0;
    uint32_t clc_phase = 0;
    int workitem_id = !PERSISTENT ? workitem_begin + (int)blockIdx.x
                                  : (LOGICAL_CLC ? (int)*clc_initial_work
                                                 : (int)blockIdx.x);

    // GEMM 1 (S^T = K @ Q^T) or GEMM 2 (dP^T = V @ dO^T): A = K or V, B = Q[stage] or dO, both
    // K-major; 2 hd subtiles x 4 k16. Waits for B to land; the commit publishes the tile to the
    // softmax warps.
    auto gemm12_st_dpt = [&](auto is_st_const, int stage) {
      constexpr bool is_st    = decltype(is_st_const)::value;
      const uint32_t tmem_acc = is_st ? tmem_st : tmem_dpt;
      const uint64_t da_base  = is_st ? desc_k : desc_v;
      const uint64_t db_base  = is_st ? desc_q0 + (uint64_t)stage * Q_STAGE_DELTA : desc_do;
      uint64_t* commit_bar    = is_st ? full_bar_st : full_bar_dpt;
      if constexpr (is_st) {
        wp_begin(wpc, WP_MMA_WAIT_FULL_Q);
        mbarrier_wait_parity(smem_ptr_u32(&full_bar_q[stage]), q_full_ph.get_phase());
        wp_end(wpc, WP_MMA_WAIT_FULL_Q);
      } else {
        wp_begin(wpc, WP_MMA_WAIT_FULL_DO);
        mbarrier_wait_parity(smem_ptr_u32(full_bar_do), do_full_ph.get_phase());
        do_full_ph.advance();
        wp_end(wpc, WP_MMA_WAIT_FULL_DO);
      }

      wp_begin(wpc, WP_MMA_ISSUE);
      #pragma unroll
      for (int s = 0; s < HD_SUBTILES; ++s) {
        #pragma unroll
        for (int ki = 0; ki < K_ATOMS_PER_SUBTILE; ++ki) {
          const bool accumulate = (s | ki) != 0;
          tcgen05_mma_f16_ss_lead(lead, tmem_acc, da_base + s * KV_SUB_DELTA + ki * K16_COLS_DELTA,
                                  db_base + s * Q_SUB_DELTA + ki * K16_COLS_DELTA, idesc_st_dpt,
                                  accumulate);
        }
      }
      wp_end(wpc, WP_MMA_ISSUE);

      wp_begin(wpc, WP_MMA_COMMIT);
      tcgen05_commit1_lead(lead, smem_ptr_u32(commit_bar));
      wp_end(wpc, WP_MMA_COMMIT);
    };

    // GEMM 3 (dV += P^T @ dO) or GEMM 5 (dK += dS^T @ Q[stage]): A = the bf16x2 overlay in TMEM,
    // B = dO or Q MN-major (tb = 1), 2 q-halves x 4 k16 over the 128 q. q-half h's bf16x2 atoms sit
    // in the first 32 columns of its 64 fp32 columns (see the softmax warps). The commit frees B's stage.
    auto gemm35_dv_dk = [&](auto is_dv_const, int stage, bool first) {
      constexpr bool is_dv       = decltype(is_dv_const)::value;
      const uint32_t tmem_acc    = is_dv ? tmem_dv : tmem_dk;
      const uint32_t tmem_a_base = is_dv ? tmem_pt_bf16 : tmem_dst_bf16;
      uint64_t db          = is_dv ? desc_do_mn : desc_q0_mn + (uint64_t)stage * Q_STAGE_DELTA;
      uint64_t* commit_bar = is_dv ? empty_bar_do : &empty_bar_q[stage];
      wp_begin(wpc, WP_MMA_ISSUE);
      if constexpr (!is_dv && DK_FROM_SMEM) {
        #pragma unroll
        for (int q_half = 0; q_half < 2; ++q_half) {
          #pragma unroll
          for (int ki = 0; ki < K_ATOMS_PER_Q_HALF; ++ki) {
            const uint64_t da = desc_dst_k + q_half * DST_SUB_DELTA + ki * K16_COLS_DELTA;
            const bool accumulate = !(first && (q_half | ki) == 0);
            tcgen05_mma_f16_ss_lead(lead, tmem_acc, da, db, idesc_dv_dk, accumulate);
            smem_desc_add_lo(db, K16_ROWS_DELTA);
          }
        }
      } else {
        #pragma unroll
        for (int q_half = 0; q_half < 2; ++q_half) {
          #pragma unroll
          for (int ki = 0; ki < K_ATOMS_PER_Q_HALF; ++ki) {
            const uint32_t tmem_a = tmem_a_base + (uint32_t)(q_half * ST_HALF_COLS +
                                                             ki * BF16X2_COLS_PER_K16);
            const bool accumulate = !(first && (q_half | ki) == 0);
            tcgen05_mma_f16_ts_1sm_lead(lead, tmem_acc, tmem_a, db, idesc_dv_dk, accumulate);
            smem_desc_add_lo(db, K16_ROWS_DELTA);
          }
        }
      }
      wp_end(wpc, WP_MMA_ISSUE);

      wp_begin(wpc, WP_MMA_COMMIT);
      tcgen05_commit1_lead(lead, smem_ptr_u32(commit_bar));
      wp_end(wpc, WP_MMA_COMMIT);
    };

    // GEMM 4 (dQ = dS @ K): A = dS from sDST (MN-major, ta = 1), B = K MN-major (tb = 1), 8 k16
    // over the 128 kv rows. Reuses dP^T's TMEM columns; the epilogue warps drain the tile.
    auto gemm4_dq = [&]() {
      uint64_t adst = desc_dst_mn;
      uint64_t bk   = desc_k_mn;
      wp_begin(wpc, WP_MMA_ISSUE);
      #pragma unroll
      for (int ki = 0; ki < K_ATOMS_PER_KV_TILE; ++ki) {
        tcgen05_mma_f16_ss_lead(lead, tmem_dq, adst, bk, idesc_dq, ki != 0);
        smem_desc_add_lo(adst, K16_ROWS_DELTA);
        smem_desc_add_lo(bk, K16_ROWS_DELTA);
      }
      wp_end(wpc, WP_MMA_ISSUE);

      wp_begin(wpc, WP_MMA_COMMIT);
      tcgen05_commit1_lead(lead, smem_ptr_u32(full_bar_dq));
      wp_end(wpc, WP_MMA_COMMIT);
    };

    do {
      wp_marker(wpc, WP_ITEM, workitem_id);
      const WorkItem it = decode_workitem<ORDERED_KV_BLOCKS, DEFENSIVE>(
          workitem_id, batch_head_base, workitem_remap, k2q_idx, k2q_num, max_q_blocks,
          num_heads, num_kv_blocks_per_seq);

      if (it.local_k2q_num > 0) {
        // Steps 1..N-1 start here (each also issues S^T(j+1)); step 0 is the peeled prologue below.
        wp_marker(wpc, WP_ITER, 0);

        // Prologue: S^T(0), dP^T(0), dV(0). Only a persistent CTA gates dP^T(0) on empty_bar_dq:
        // its dP^T / dQ columns hold the previous item's dQ tile until the epilogue drains it.
        gemm12_st_dpt(std::true_type{}, q_full_ph.get_stage());

        if constexpr (PERSISTENT) {
          wp_begin(wpc, WP_MMA_WAIT_EMPTY_DQ);
          mbarrier_wait_parity(smem_ptr_u32(empty_bar_dq), dq_empty_ph.get_phase());
          dq_empty_ph.advance();
          wp_end(wpc, WP_MMA_WAIT_EMPTY_DQ);
        }

        gemm12_st_dpt(std::false_type{}, 0);

        wp_begin(wpc, WP_MMA_WAIT_FULL_PT);
        mbarrier_wait_parity(smem_ptr_u32(full_bar_pt), pt_ph.get_phase());
        pt_ph.advance();
        wp_end(wpc, WP_MMA_WAIT_FULL_PT);

        // Persistent: dV(0) zero-inits tmem_dv (dK(0) tmem_dk right after) -- wait until the
        // softmax warps have stored the previous item's dV and dK.
        if constexpr (PERSISTENT) {
          wp_begin(wpc, WP_MMA_WAIT_EMPTY_EPI);
          mbarrier_wait_parity(smem_ptr_u32(empty_bar_epi), epi_empty_ph.get_phase());
          epi_empty_ph.advance();
          wp_end(wpc, WP_MMA_WAIT_EMPTY_EPI);
        }

        gemm35_dv_dk(std::true_type{}, 0, true);

        // Step j: S^T(j+1) | dK(j) | dQ(j) | dP^T(j+1) | dV(j+1).
        for (int j = 0; j < it.local_k2q_num; ++j) {
          if (j != 0) wp_marker(wpc, WP_ITER, j);

          const int stage = q_full_ph.get_stage();
          q_full_ph.advance();
          const int next_stage = q_full_ph.get_stage();
          const bool last      = j + 1 == it.local_k2q_num;

          if (!last) gemm12_st_dpt(std::true_type{}, next_stage);
          // Entering the last step every dV accumulation has been issued: publish dV now so the
          // softmax warps store it while the tail dK / dQ still run.
          if (last) {
            wp_begin(wpc, WP_MMA_COMMIT);
            tcgen05_commit1_lead(lead, smem_ptr_u32(full_bar_dv));
            wp_end(wpc, WP_MMA_COMMIT);
          }
          wp_begin(wpc, WP_MMA_WAIT_FULL_DST);
          mbarrier_wait_parity(smem_ptr_u32(full_bar_dst), dst_ph.get_phase());
          dst_ph.advance();
          wp_end(wpc, WP_MMA_WAIT_FULL_DST);
          if constexpr (DK_FROM_SMEM) {
            // dQ consumes the transposed view of sDST and publishes its epilogue early. dK then
            // consumes the same physical dS^T tile directly; it no longer needs the TMEM overlay.
            gemm4_dq();
            gemm35_dv_dk(std::false_type{}, stage, j == 0);
            if (last) {
              wp_begin(wpc, WP_MMA_COMMIT);
              tcgen05_commit1_lead(lead, smem_ptr_u32(full_bar_dk));
              wp_end(wpc, WP_MMA_COMMIT);
            }
          } else {
            gemm35_dv_dk(std::false_type{}, stage, j == 0);
            if (last) {
              wp_begin(wpc, WP_MMA_COMMIT);
              tcgen05_commit1_lead(lead, smem_ptr_u32(full_bar_dk));
              wp_end(wpc, WP_MMA_COMMIT);
            }
            gemm4_dq();
          }
          if (!last) {
            wp_begin(wpc, WP_MMA_WAIT_EMPTY_DQ);
            mbarrier_wait_parity(smem_ptr_u32(empty_bar_dq), dq_empty_ph.get_phase());
            dq_empty_ph.advance();
            wp_end(wpc, WP_MMA_WAIT_EMPTY_DQ);
            gemm12_st_dpt(std::false_type{}, 0);
            wp_begin(wpc, WP_MMA_WAIT_FULL_PT);
            mbarrier_wait_parity(smem_ptr_u32(full_bar_pt), pt_ph.get_phase());
            pt_ph.advance();
            wp_end(wpc, WP_MMA_WAIT_FULL_PT);
            gemm35_dv_dk(std::true_type{}, 0, false);
          }
        }

        // Persistent: the item's last dQ GEMM has read K, so K and V may be replaced.
        if constexpr (PERSISTENT) {
          wp_begin(wpc, WP_MMA_COMMIT);
          tcgen05_commit1_lead(lead, smem_ptr_u32(empty_bar_kv));
          wp_end(wpc, WP_MMA_COMMIT);
        }
      }

      workitem_id = get_next_workitem_id(workitem_id, clc_stage, clc_phase);
    } while (PERSISTENT && workitem_id >= 0);
    wp_flush(wpc);
    tcgen05_relinquish_alloc_permit<1>();
    bar_sync<10>(416);
    tcgen05_dealloc<1>(tmem_base, TMEM_TOTAL);
    return;
  }
  else if (warp_id == W_SCHED) {
    // Scheduler warp (w14): the CLC producer loop, or (CLC = false) a donor like w15.
    if constexpr (CLC) {
      setmaxnreg_dec<88>();
      int prod_stage = 0; uint32_t prod_phase = 1;
      int cons_stage = 0; uint32_t cons_phase = 0;
      while (true) {
        if (lane == 0)
          mbarrier_wait_parity_suspend(smem_ptr_u32(&clc_empty[prod_stage]), prod_phase);
        __syncwarp();
        clc_arrive_expect_tx_cta(smem_ptr_u32(&clc_full[prod_stage]), 16);
        if (lane == 0)
          clc_try_cancel_async(smem_ptr_u32(&clc_response[prod_stage * 4]),
                               smem_ptr_u32(&clc_full[prod_stage]));
        advance_stage_phase<CLC_STAGES>(prod_stage, prod_phase);
        ClcTileInfo n = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, 1, true>(
            clc_full, clc_empty, clc_response, cons_stage, cons_phase, elect_one_sync());
        if constexpr (LOGICAL_CLC) {
          // The token is only a continuation: claim the next head-major id and publish it.
          if (lane == 0) {
            clc_logical_work[cons_stage] =
                n.valid ? atom_global_add_u32(clc_work_counter, 1u) : ~0u;
            mbarrier_arrive(smem_ptr_u32(&clc_logical_full[cons_stage]));
          }
        }
        clc_fetch_next_tile_advance<CLC_STAGES>(cons_stage, cons_phase);
        if (!n.valid) break;
      }

      #pragma unroll
      for (int st = 0; st < CLC_STAGES; ++st) {
        if (lane == 0)
          mbarrier_wait_parity_suspend(smem_ptr_u32(&clc_empty[prod_stage]), prod_phase);
        __syncwarp();
        advance_stage_phase<CLC_STAGES>(prod_stage, prod_phase);
      }
    } else if constexpr (DEFENSIVE) {
      setmaxnreg_dec<24>();
      for (int workitem_id = (int)blockIdx.x; workitem_id < defensive_padded_workitems;
           workitem_id += (int)gridDim.x)
        defensive_wave_barrier(workitem_id);
    } else {
      setmaxnreg_dec<24>();
    }
    return;
  }
  else if (warp_id >= W_SOFTMAX0 && warp_id < W_MMA) {
    setmaxnreg_inc<136>();
    bar_sync<10>(416);
    const uint32_t tmem_base    = *tmem_slot;
    const uint32_t tmem_st      = tmem_base;
    const uint32_t tmem_dv      = tmem_st + ST_COLS;
    const uint32_t tmem_dpt     = tmem_dv + DV_COLS;
    const uint32_t tmem_dk      = tmem_dpt + ST_COLS;
    const uint32_t tmem_pt_bf16 = tmem_st, tmem_dst_bf16 = tmem_dpt;

    // Softmax warp -> its piece of the 128-lane x 128-column S^T / dP^T tiles: TMEM lanes
    // [32 * lane_group, +32) (fixed by warp_id % 4) and the 64 fp32 columns of q-half col_half.
    const int softmax_warp_id = warp_id - W_SOFTMAX0;
    const int lane_group      = softmax_warp_id & 3;
    const int col_half        = softmax_warp_id >> 2;
    const int kv_row          = lane_group * 32 + lane;
    // TMEM address offsets (lane << 16 | column) added to a tile base. A warp's bf16x2 overlay
    // (32 columns) starts at the same column as the 64 fp32 columns it loads, so it only ever
    // overwrites columns it has itself consumed: the other warp of the lane group reads the other
    // q-half's columns, and no cross-warp ordering is needed between its ld and this st.
    const uint32_t tmem_lane_base     = (uint32_t)(lane_group * 32) << 16;
    const uint32_t tmem_f32_offset    = tmem_lane_base + (uint32_t)(col_half * ST_HALF_COLS);
    const uint32_t tmem_bf16x2_offset = tmem_f32_offset;
    // 128B-swizzled bf16 SMEM tiles (sDST and the dV / dK bounce subtiles): each 128 B row is 8
    // chunks of 16 B; chunk v of row r is stored at chunk slot v ^ (r & 7).
    constexpr int CHUNK_BF16     = 16 / (int)sizeof(__nv_bfloat16);  // 8 bf16 per 16-byte chunk
    constexpr int CHUNKS_PER_ROW = SUB_COLS_BF16 / CHUNK_BF16;        // 8 chunks per 128 B row
    // This thread's row of its q-half's dS^T tile in sDST: 64 bf16 = 128 B.
    __nv_bfloat16* sdst_row =
        sDST + (size_t)col_half * KV_TILE * SUB_COLS_BF16 + kv_row * SUB_COLS_BF16;

    PhaseTracker<1> st_ph, dpt_ph, delta_ph, dv_ph, dk_ph;
    PhaseTracker<NUM_Q_STAGES> lse_ph;
    int clc_stage = 0;
    uint32_t clc_phase = 0;
    int workitem_id = !PERSISTENT ? workitem_begin + (int)blockIdx.x
                                  : (LOGICAL_CLC ? (int)*clc_initial_work
                                                 : (int)blockIdx.x);

    do {
      wp_marker(wpc, WP_ITEM, workitem_id);
      const WorkItem it = decode_workitem<ORDERED_KV_BLOCKS, DEFENSIVE>(
          workitem_id, batch_head_base, workitem_remap, k2q_idx, k2q_num, max_q_blocks,
          num_heads, num_kv_blocks_per_seq);

      // Tile end: this warp's 32 kv rows x 64 hd columns (its col_half is also its hd subtile) of
      // dV or dK leave TMEM in two x32 loads, go to the bounce buffer's hd subtile as bf16
      // (swizzled like sDST), and the warpgroup's first warp TMA-stores that subtile. dK is
      // multiplied by sm_scale on the way out (dS was formed on the scaled scores); dV is not.
      auto store_dv_dk_tile = [&](uint32_t tmem_acc, auto apply_sm_scale_const, uint8_t* bounce,
                                  const CUtensorMap* map) {
        constexpr bool apply_sm_scale = decltype(apply_sm_scale_const)::value;
        uint8_t* bounce_subtile       = bounce + col_half * KV_SUB_COLS_BYTES;
        __nv_bfloat16* stage_row =
            reinterpret_cast<__nv_bfloat16*>(bounce_subtile) + kv_row * SUB_COLS_BF16;
        #pragma unroll
        for (int c0 = 0; c0 < ST_HALF_COLS; c0 += 32) {
          uint32_t acc_regs[32];
          tcgen05_ld_32x32b_x32(tmem_acc + tmem_f32_offset + (uint32_t)c0, acc_regs);
          tcgen05_fence_before_thread_sync();
          const float* acc = reinterpret_cast<const float*>(acc_regs);
          #pragma unroll
          for (int v = 0; v < 32 / CHUNK_BF16; ++v) {
            float value[CHUNK_BF16];
            #pragma unroll
            for (int e = 0; e < CHUNK_BF16; ++e)
              value[e] = apply_sm_scale ? acc[v * CHUNK_BF16 + e] * sm_scale
                                        : acc[v * CHUNK_BF16 + e];
            uint4 packed;
            packed.x = cvt_f32x2_to_bf16x2(value[0], value[1]);
            packed.y = cvt_f32x2_to_bf16x2(value[2], value[3]);
            packed.z = cvt_f32x2_to_bf16x2(value[4], value[5]);
            packed.w = cvt_f32x2_to_bf16x2(value[6], value[7]);
            const int chunk = c0 / CHUNK_BF16 + v;
            *reinterpret_cast<uint4*>(stage_row + ((chunk ^ (kv_row & 7)) * CHUNK_BF16)) = packed;
          }
        }
        fence_proxy_async_shared_cta();
        if (col_half == 0) bar_sync<12>(128); else bar_sync<13>(128);
        if (lane_group == 0 && elect_one_sync()) {
          if constexpr (BHSD)
            tma_store_4d(map, 0, it.kv_block_id_in_seq * KV_TILE, col_half, it.batch_head,
                         smem_ptr_u32(bounce_subtile));
          else
            tma_store_3d(map, 0, it.batch * seqlen + it.kv_block_id_in_seq * KV_TILE,
                         it.head * HD_SUBTILES + col_half, smem_ptr_u32(bounce_subtile));
          cp_async_bulk_commit_group();
        }
      };

      // Step j: P^T(j) for GEMM 3, then dS^T(j) for GEMMs 4 and 5. An empty q list runs zero
      // iterations, so it touches none of these barriers.
      for (int j = 0; j < it.local_k2q_num; ++j) {
        wp_marker(wpc, WP_ITER, j);
        const int lse_stage     = lse_ph.get_stage();  // the LSE slot rides the Q stage index
        const float2* lse2   = reinterpret_cast<const float2*>(sLSE + lse_stage * Q_TILE +
                                                               col_half * ST_HALF_COLS);
        const float2* delta2 = reinterpret_cast<const float2*>(sDelta + col_half * ST_HALF_COLS);
        const uint32_t* delta_bf16x2 =
            reinterpret_cast<const uint32_t*>(sDelta) + col_half * (ST_HALF_COLS / 2);

        wp_begin(wpc, WP_SM_WAIT_FULL_LSE);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_lse[lse_stage]), lse_ph.get_phase());
        lse_ph.advance();
        wp_end(wpc, WP_SM_WAIT_FULL_LSE);

        wp_begin(wpc, WP_SM_WAIT_FULL_ST);
        mbarrier_wait_parity_suspend(smem_ptr_u32(full_bar_st), st_ph.get_phase());
        st_ph.advance();
        wp_end(wpc, WP_SM_WAIT_FULL_ST);

        // P^T = exp2(S^T * scale_log2 - LSE), kept in fp32 for dS^T and packed to bf16 pairs
        // for GEMM 3; the affine step is one ffma2 per column pair, exp2 stays on the SFU.
        wp_begin(wpc, WP_SM_STORE_PT);
        uint32_t st_regs[ST_HALF_COLS];
        uint32_t pt_bf16x2[ST_HALF_COLS / 2];
        float2* pt_fp32 = reinterpret_cast<float2*>(st_regs);  // P^T overwrites S^T in registers
        tcgen05_ld_32x32b_x64(tmem_st + tmem_f32_offset, st_regs);
        tcgen05_fence_before_thread_sync();
        const float2 scale2 = f32x2_splat(scale_log2);
        #pragma unroll
        for (int c = 0; c < ST_HALF_COLS / 2; ++c) {
          const float2 z = ffma2(pt_fp32[c], scale2, make_float2(-lse2[c].x, -lse2[c].y));
          const float2 p = make_float2(ex2_approx_f32(z.x), ex2_approx_f32(z.y));
          pt_fp32[c]     = p;
          pt_bf16x2[c]   = cvt_f32x2_to_bf16x2(p.x, p.y);
        }
        tcgen05_st_32x32b_x32(tmem_pt_bf16 + tmem_bf16x2_offset, pt_bf16x2);
        tcgen05_wait_st();
        tcgen05_fence_before_thread_sync();
        if (elect_one_sync()) {
          mbarrier_arrive(smem_ptr_u32(full_bar_pt));
          mbarrier_arrive(smem_ptr_u32(&empty_bar_lse[lse_stage]));
        }
        wp_end(wpc, WP_SM_STORE_PT);

        wp_begin(wpc, WP_SM_WAIT_FULL_DELTA);
        uint64_t* delta_ready = EXP_PACKED_DS ? full_bar_delta_cvt : full_bar_delta;
        mbarrier_wait_parity_suspend(smem_ptr_u32(delta_ready), delta_ph.get_phase());
        delta_ph.advance();
        wp_end(wpc, WP_SM_WAIT_FULL_DELTA);

        wp_begin(wpc, WP_SM_WAIT_FULL_DPT);
        mbarrier_wait_parity_suspend(smem_ptr_u32(full_bar_dpt), dpt_ph.get_phase());
        dpt_ph.advance();
        wp_end(wpc, WP_SM_WAIT_FULL_DPT);

        // dS^T = P^T * (dP^T - Delta), packed to bf16 pairs over dP^T's columns for GEMM 5 and
        // written to this thread's sDST row for GEMM 4.
        wp_begin(wpc, WP_SM_STORE_DST);
        uint32_t dpt_regs[ST_HALF_COLS];
        tcgen05_ld_32x32b_x64(tmem_dpt + tmem_f32_offset, dpt_regs);
        tcgen05_fence_before_thread_sync();
        const float2* dpt2 = reinterpret_cast<const float2*>(dpt_regs);
        #pragma unroll
        for (int c = 0; c < ST_HALF_COLS / 2; ++c) {
          if constexpr (EXP_PACKED_DS) {
            // Gotcha: rounding dP and Delta to BF16 before subtracting can lose a small
            // dP-Delta residual. At S=4K, H=8, topk=8, dQ rel=8.02e-3 versus 2.31e-3 for
            // FP32 dS arithmetic, failing the 8e-3 gate. Keep this experimental.
            const uint32_t dpt_bf16x2 = cvt_f32x2_to_bf16x2(dpt2[c].x, dpt2[c].y);
            const __nv_bfloat162 p = *reinterpret_cast<const __nv_bfloat162*>(&pt_bf16x2[c]);
            const __nv_bfloat162 dp = *reinterpret_cast<const __nv_bfloat162*>(&dpt_bf16x2);
            const __nv_bfloat162 delta =
                *reinterpret_cast<const __nv_bfloat162*>(&delta_bf16x2[c]);
            const __nv_bfloat162 ds = __hmul2(p, __hsub2(dp, delta));
            st_regs[c] = *reinterpret_cast<const uint32_t*>(&ds);
          } else {
            const float2 ds =
                fmul2(pt_fp32[c], fadd2(dpt2[c], make_float2(-delta2[c].x, -delta2[c].y)));
            st_regs[c] = cvt_f32x2_to_bf16x2(ds.x, ds.y);
          }
        }
        uint32_t (&dst_bf16x2)[ST_HALF_COLS / 2] =
            reinterpret_cast<uint32_t (&)[ST_HALF_COLS / 2]>(st_regs);
        if constexpr (!DK_FROM_SMEM)
          tcgen05_st_32x32b_x32(tmem_dst_bf16 + tmem_bf16x2_offset, dst_bf16x2);
        const uint4* dst_chunks = reinterpret_cast<const uint4*>(dst_bf16x2);
        #pragma unroll
        for (int v = 0; v < CHUNKS_PER_ROW; ++v)
          *reinterpret_cast<uint4*>(sdst_row + (v ^ (kv_row & 7)) * CHUNK_BF16) = dst_chunks[v];
        if constexpr (!DK_FROM_SMEM) {
          tcgen05_wait_st();
          tcgen05_fence_before_thread_sync();
        }
        fence_proxy_async_shared_cta();
        if (elect_one_sync()) {
          mbarrier_arrive(smem_ptr_u32(full_bar_dst));
          mbarrier_arrive(smem_ptr_u32(empty_bar_delta));
        }
        wp_end(wpc, WP_SM_STORE_DST);
      }

      // Tile end: dV first (it completes while the tail dK / dQ still run), then dK. An empty
      // item has no dV / dK (the epilogue warps store its zero tiles) and touches no barrier.
      if (it.local_k2q_num > 0) {
        wp_begin(wpc, WP_SM_WAIT_FULL_DV);
        mbarrier_wait_parity_suspend(smem_ptr_u32(full_bar_dv), dv_ph.get_phase());
        dv_ph.advance();
        wp_end(wpc, WP_SM_WAIT_FULL_DV);

        wp_begin(wpc, WP_SM_STORE_DV);
        store_dv_dk_tile(tmem_dv, std::false_type{}, sDO, &tmap_dv);
        wp_end(wpc, WP_SM_STORE_DV);

        wp_begin(wpc, WP_SM_WAIT_FULL_DK);
        mbarrier_wait_parity_suspend(smem_ptr_u32(full_bar_dk), dk_ph.get_phase());
        dk_ph.advance();
        wp_end(wpc, WP_SM_WAIT_FULL_DK);

        wp_begin(wpc, WP_SM_STORE_DK);
        store_dv_dk_tile(tmem_dk, std::true_type{}, sQ[0], &tmap_dk);
        // Persistent: the dV / dK stores have read the sDO / sQ[0] bounce and their TMEM columns
        // are drained, so the next item may refill both.
        if constexpr (PERSISTENT) {
          if (lane_group == 0 && elect_one_sync()) cp_async_bulk_wait_group_read<0>();
          bar_sync<14>(256);
          mbarrier_arrive(smem_ptr_u32(empty_bar_epi));
        }
        wp_end(wpc, WP_SM_STORE_DK);
      }

      workitem_id = get_next_workitem_id(workitem_id, clc_stage, clc_phase);
    } while (PERSISTENT && workitem_id >= 0);
    wp_flush(wpc);
    bar_sync<10>(416);
    return;
  }
  else if (warp_id < W_SOFTMAX0) {
    setmaxnreg_inc<152>();

    bar_sync<10>(416);
    const uint32_t tmem_base = *tmem_slot;
    const uint32_t tmem_dq = tmem_base + ST_COLS + DV_COLS;  // the dQ tile reuses dP^T's columns

    // Epilogue warp w owns TMEM lanes [32w, +32) of the dQ tile: 32 q rows x 128 hd columns, one
    // row per thread. Warp 0's elected lane issues the bulk reduce-adds.
    const int epi_warp_id         = warp_id - W_EPI0;
    const uint32_t tmem_lane_base = (uint32_t)(epi_warp_id * 32) << 16;
    const int q_row               = epi_warp_id * 32 + lane;
    const bool is_leader          = epi_warp_id == 0 && lane == 0;

    PhaseTracker<1> dq_full_ph;
    int clc_stage = 0;
    uint32_t clc_phase = 0;
    int workitem_id = !PERSISTENT ? workitem_begin + (int)blockIdx.x
                                  : (LOGICAL_CLC ? (int)*clc_initial_work
                                                 : (int)blockIdx.x);
    int item_ordinal = 0;
    // Persistent: prime empty_bar_dq once. The MMA's item prologue waits on it before dP^T(0),
    // which shares TMEM with the previous item's dQ tile; on the first item nothing precedes it.
    if constexpr (PERSISTENT) {
      if (elect_one_sync()) mbarrier_arrive(smem_ptr_u32(empty_bar_dq));
    }

    do {
      wp_marker(wpc, WP_ITEM, workitem_id);

      const WorkItem it = decode_workitem<ORDERED_KV_BLOCKS, DEFENSIVE>(
          workitem_id, batch_head_base, workitem_remap, k2q_idx, k2q_num, max_q_blocks,
          num_heads, num_kv_blocks_per_seq);

      float* dqaccum_head =
          dqaccum + (size_t)it.batch_head * num_kv_blocks_per_seq * DQ::DQ_BLOCK_ELEMS;

      for (int j = 0; j < it.local_k2q_num; ++j) {
        wp_marker(wpc, WP_ITER, j);
        const int qblock =
            k2q_block_at<K2Q_FIRST_DESCENDING, K2Q_TRAVERSAL_SNAKE>(it, j, item_ordinal);
        float* dqaccum_block = dqaccum_head + (size_t)qblock * DQ::DQ_BLOCK_ELEMS;

        wp_begin(wpc, WP_EPI_WAIT_FULL_DQ);
        mbarrier_wait_parity(smem_ptr_u32(full_bar_dq), dq_full_ph.get_phase());
        dq_full_ph.advance();
        wp_end(wpc, WP_EPI_WAIT_FULL_DQ);

        // This thread's q row: 128 hd fp32 columns, loaded 64 at a time (a single x128 load
        // needs 146 registers at that instruction and does not fit the kernel's 128-register
        // compile cap; setmaxnreg only raises the budget at run time). The whole tile is read
        // before any staging so empty_bar_dq fires early and the MMA's next dP^T overlaps the
        // drain below.
        wp_begin(wpc, WP_EPI_LOAD_DQ);
        uint32_t dq_regs[HEAD_DIM];
        #pragma unroll
        for (int c = 0; c < HEAD_DIM / 64; ++c)
          tcgen05_ld_32x32b_x64(tmem_dq + tmem_lane_base + (uint32_t)(c * 64),
                                reinterpret_cast<uint32_t (&)[64]>(dq_regs[c * 64]));
        // The arrive hands tmem_dq back to the MMA; only the loads' register consumers are
        // scoreboarded, so wait::ld keeps it behind them.
        tcgen05_wait_ld();
        tcgen05_fence_before_thread_sync();
        if (elect_one_sync()) mbarrier_arrive(smem_ptr_u32(empty_bar_dq));
        wp_end(wpc, WP_EPI_LOAD_DQ);

        // Drain in slices of DQ::COLS hd columns through the two stage buffers. Stage layout:
        // float4 v4 of q row t at [v4 * Q_TILE * 4 + t * 4] (a warp writes 512 B contiguous
        // per v4); dqaccum keeps this order and the postprocess kernel undoes it.
        wp_begin(wpc, WP_EPI_STORE_DQ);
        #pragma unroll
        for (int hd_slice = 0; hd_slice < HEAD_DIM / DQ::COLS; ++hd_slice) {
          wp_begin(wpc, WP_EPI_STAGE_DQ);
          const int stage_buf   = hd_slice & 1;  // slice s+1 stages while s's push still reads
          const float4* dq_row4 = reinterpret_cast<const float4*>(dq_regs + hd_slice * DQ::COLS);
          #pragma unroll
          for (int v4 = 0; v4 < DQ::COLS / 4; ++v4)
            *reinterpret_cast<float4*>(sDQ_STAGE[stage_buf] + v4 * Q_TILE * 4 + q_row * 4) =
                dq_row4[v4];
          fence_proxy_async_shared_cta();
          wp_end(wpc, WP_EPI_STAGE_DQ);
          wp_begin(wpc, WP_EPI_SYNC_DQ_READY);
          bar_sync<11>(128);
          wp_end(wpc, WP_EPI_SYNC_DQ_READY);
          if (is_leader) {
            const size_t slice_offset = (size_t)hd_slice * Q_TILE * DQ::COLS;
            wp_begin(wpc, WP_EPI_REDUCE_DQ_ISSUE);
            cpasync_reduce_bulk_add_f32(dqaccum_block + slice_offset,
                                        smem_ptr_u32(sDQ_STAGE[stage_buf]), DQ::DQ_ONE_PUSH_BYTES);
            cp_async_bulk_commit_group();
            wp_end(wpc, WP_EPI_REDUCE_DQ_ISSUE);
            wp_begin(wpc, WP_EPI_REDUCE_DQ_WAIT);
            cp_async_bulk_wait_group_read<1>();  // the other stage buffer is free again
            wp_end(wpc, WP_EPI_REDUCE_DQ_WAIT);
          }
          wp_begin(wpc, WP_EPI_SYNC_DQ_REUSE);
          bar_sync<11>(128);
          wp_end(wpc, WP_EPI_SYNC_DQ_REUSE);
        }
        wp_end(wpc, WP_EPI_STORE_DQ);
      }

      if (it.local_k2q_num == 0) {
        // Zero tiles (FA4): no q block selects this kv block, so dK = dV = 0. The stage buffers
        // (2 x 16 KB = one bf16 tile as two 64-hd subtiles) are these warps' own, so no other
        // role is involved: drain the previous item's pushes, zero them, and the leader
        // TMA-stores the pair as dK and again as dV; wait until the stores have read them.
        if (is_leader) cp_async_bulk_wait_group_read<0>();
        bar_sync<11>(128);
        static_assert(DQ::DQ_STAGE_BUFFERS * DQ::DQ_STAGE_BYTES == HD_SUBTILES * KV_SUB_COLS_BYTES,
                      "the dQ stage buffers hold one 128 kv x 128 hd bf16 tile");
        constexpr int ZERO_UINT4_PER_THREAD =
            DQ::DQ_STAGE_BUFFERS * DQ::DQ_STAGE_BYTES / (128 * 16);
        uint4* zero_tile = reinterpret_cast<uint4*>(sDQ_STAGE_bytes) + epi_warp_id * 32 + lane;
        #pragma unroll
        for (int v = 0; v < ZERO_UINT4_PER_THREAD; ++v)
          zero_tile[v * 128] = make_uint4(0u, 0u, 0u, 0u);
        fence_proxy_async_shared_cta();
        bar_sync<11>(128);
        if (is_leader) {
          #pragma unroll
          for (int which = 0; which < 2; ++which) {
            const CUtensorMap* map = (which == 0) ? &tmap_dk : &tmap_dv;
            #pragma unroll
            for (int s = 0; s < HD_SUBTILES; ++s) {
              const uint32_t src = smem_ptr_u32(sDQ_STAGE_bytes + (size_t)s * KV_SUB_COLS_BYTES);
              if constexpr (BHSD)
                tma_store_4d(map, 0, it.kv_block_id_in_seq * KV_TILE, s, it.batch_head, src);
              else
                tma_store_3d(map, 0, it.batch * seqlen + it.kv_block_id_in_seq * KV_TILE,
                             it.head * HD_SUBTILES + s, src);
            }
          }
          cp_async_bulk_commit_group();
          cp_async_bulk_wait_group_read<0>();
        }
        bar_sync<11>(128);
      }

      ++item_ordinal;
      workitem_id = get_next_workitem_id(workitem_id, clc_stage, clc_phase);
    } while (PERSISTENT && workitem_id >= 0);

    // Bulk reads must complete before SMEM dies with the CTA.
    if (is_leader) cp_async_bulk_wait_group_read<0>();
    bar_sync<11>(128);
    wp_flush(wpc);
    bar_sync<10>(416);
    // CTA done: the postprocess may launch (its wait still covers completion).
    if constexpr (KERNEL_PDL) {
      if (is_leader) griddepcontrol_launch_dependents();
    }
    return;
  }
  else {  // idle warp: no work, donates its registers
    setmaxnreg_dec<24>();
    if constexpr (DEFENSIVE) {
      for (int workitem_id = (int)blockIdx.x; workitem_id < defensive_padded_workitems;
           workitem_id += (int)gridDim.x)
        defensive_wave_barrier(workitem_id);
    }
    return;
  }
}

// Stable counting sort by decreasing k2q-list length. One CTA handles one head, or the whole
// small global list; the encoded global entries retain both the batch-head and KV block IDs.
template <bool ENCODE_GLOBAL>
__device__ __forceinline__ void build_lpt_remap(const int* __restrict__ k2q_num,
                                                int* __restrict__ remap, int item_begin,
                                                int item_count, int max_q_blocks,
                                                int num_kv_blocks, int* smem) {
  int* counts        = smem;
  int* bins          = smem + item_count;
  const int num_bins = max_q_blocks + 1;
  for (int b = (int)threadIdx.x; b < num_bins; b += (int)blockDim.x) bins[b] = 0;
  for (int i = (int)threadIdx.x; i < item_count; i += (int)blockDim.x)
    counts[i] = k2q_num[item_begin + i];
  __syncthreads();
  for (int i = (int)threadIdx.x; i < item_count; i += (int)blockDim.x)
    atomicAdd(&bins[counts[i]], 1);
  __syncthreads();
  const int lane = (int)threadIdx.x & 31, warp = (int)threadIdx.x >> 5;
  if (warp == 0) {
    int carry = 0;
    for (int top = num_bins - 1; top >= 0; top -= 32) {
      const int c = top - lane;
      const int h = c >= 0 ? bins[c] : 0;
      int inclusive = h;
      #pragma unroll
      for (int d = 1; d < 32; d <<= 1) {
        const int up = __shfl_up_sync(0xffffffffu, inclusive, d);
        if (lane >= d) inclusive += up;
      }
      if (c >= 0) bins[c] = carry + inclusive - h;
      carry += __shfl_sync(0xffffffffu, inclusive, 31);
    }
  }
  __syncthreads();
  // Warp turns preserve ascending item IDs for equal counts. Keep the CTA barrier outside the
  // warp-conditional branch so every thread reaches it.
  const unsigned lanes_below = (1u << lane) - 1u;
  for (int chunk = 0; chunk < item_count; chunk += (int)blockDim.x) {
    const int i = chunk + (int)threadIdx.x;
    const bool valid = i < item_count;
    const int c = valid ? counts[i] : -1;
    for (int w = 0; w < (int)(blockDim.x >> 5); ++w) {
      if (warp == w) {
        const unsigned peers = __match_any_sync(0xffffffffu, c);
        const int leader = __ffs(peers) - 1;
        int base = 0;
        if (valid && lane == leader) base = atomicAdd(&bins[c], __popc(peers));
        base = __shfl_sync(0xffffffffu, base, leader);
        if (valid) {
          const int pos = base + __popc(peers & lanes_below);
          if constexpr (ENCODE_GLOBAL)
            remap[item_begin + pos] = (((i / num_kv_blocks + 1) << 16) | (i % num_kv_blocks));
          else
            remap[item_begin + pos] = i;
        }
      }
      __syncthreads();
    }
  }
}

// Count-bin plus Q-trajectory snake ordering. Mode 0 compares medians; mode 1 compares three
// quartiles lexicographically; mode 2 compares their Morton-interleaved key.
__device__ __forceinline__ bool kv_snake_less(int a, int b, int batch_head, int num_kv_blocks,
                                              int max_q_blocks, int count_bin, int order_mode,
                                              const int* __restrict__ k2q_idx,
                                              const int* __restrict__ k2q_num) {
  if (a >= num_kv_blocks || b >= num_kv_blocks) return a < b;
  const int item_a = batch_head * num_kv_blocks + a;
  const int item_b = batch_head * num_kv_blocks + b;
  const int ca = k2q_num[item_a], cb = k2q_num[item_b];
  const int ba = ca / count_bin, bb = cb / count_bin;
  if (ba != bb) return ba < bb;
  const int* qa = k2q_idx + (size_t)item_a * max_q_blocks;
  const int* qb = k2q_idx + (size_t)item_b * max_q_blocks;
  if (order_mode == 1 && ca && cb) {
    for (int quantile = 1; quantile <= 3; ++quantile) {
      const int va = qa[quantile * ca / 4], vb = qb[quantile * cb / 4];
      if (va != vb) return (ba & 1) ? va > vb : va < vb;
    }
  }
  if (order_mode == 2) {
    uint64_t ma = 0, mb = 0;
    if (ca) {
      const int q1 = qa[ca / 4], q2 = qa[ca / 2], q3 = qa[3 * ca / 4];
      for (int bit = 0; bit < 16; ++bit) {
        ma |= (uint64_t)((q1 >> bit) & 1) << (3 * bit);
        ma |= (uint64_t)((q2 >> bit) & 1) << (3 * bit + 1);
        ma |= (uint64_t)((q3 >> bit) & 1) << (3 * bit + 2);
      }
    }
    if (cb) {
      const int q1 = qb[cb / 4], q2 = qb[cb / 2], q3 = qb[3 * cb / 4];
      for (int bit = 0; bit < 16; ++bit) {
        mb |= (uint64_t)((q1 >> bit) & 1) << (3 * bit);
        mb |= (uint64_t)((q2 >> bit) & 1) << (3 * bit + 1);
        mb |= (uint64_t)((q3 >> bit) & 1) << (3 * bit + 2);
      }
    }
    if (ma != mb) return (ba & 1) ? ma > mb : ma < mb;
  }
  const int median_a = ca ? qa[ca / 2] : -1;
  const int median_b = cb ? qb[cb / 2] : -1;
  if (median_a != median_b) return (ba & 1) ? median_a > median_b : median_a < median_b;
  return a < b;
}

__device__ __forceinline__ void build_kv_snake_remap(
    const int* __restrict__ k2q_idx, const int* __restrict__ k2q_num,
    int* __restrict__ remap, int batch_head, int num_kv_blocks, int max_q_blocks,
    int count_bin, int order_mode, int* smem) {
  int padded = 1;
  while (padded < num_kv_blocks) padded <<= 1;
  for (int i = (int)threadIdx.x; i < padded; i += (int)blockDim.x)
    smem[i] = i < num_kv_blocks ? i : num_kv_blocks;
  __syncthreads();
  for (int span = 2; span <= padded; span <<= 1) {
    for (int stride = span >> 1; stride > 0; stride >>= 1) {
      for (int pair = (int)threadIdx.x; pair < padded / 2; pair += (int)blockDim.x) {
        const int left = (pair / stride) * (stride << 1) + pair % stride;
        const int right = left + stride;
        const int a = smem[left], b = smem[right];
        const bool ascending = (left & span) == 0;
        const bool swap = ascending
                              ? kv_snake_less(b, a, batch_head, num_kv_blocks, max_q_blocks,
                                              count_bin, order_mode, k2q_idx, k2q_num)
                              : kv_snake_less(a, b, batch_head, num_kv_blocks, max_q_blocks,
                                              count_bin, order_mode, k2q_idx, k2q_num);
        if (swap) {
          smem[left] = b;
          smem[right] = a;
        }
      }
      __syncthreads();
    }
  }
  const int base = batch_head * num_kv_blocks;
  for (int i = (int)threadIdx.x; i < num_kv_blocks; i += (int)blockDim.x)
    remap[base + i] = smem[i];
}

// Preprocess: one CTA per (q128 block, batch*head), 256 threads. Zeroes the block's dqaccum slice
// (coalesced float4 stores), then Delta = rowsum(bf16(O) * dO): 16 threads per row x 8 columns
// each, shuffle-reduced. An extra CTA per batch-head builds the LPT or KV-snake remap.
template <bool BHSD = false, LptMode MODE = LptMode::OFF, bool KV_SNAKE = false>
__global__ void __launch_bounds__(256, 1)
    vsa_bwd_preprocess_kernel(const __nv_bfloat16* __restrict__ o,
                              const __nv_bfloat16* __restrict__ dout,
                              float* __restrict__ delta_rows, float* __restrict__ dqaccum,
                              int num_heads, int seqlen, const int* __restrict__ k2q_idx,
                              const int* __restrict__ k2q_num,
                              int* __restrict__ workitem_remap, int num_kv_blocks,
                              int max_q_blocks, int count_bin, int order_mode) {
  static_assert(MODE != LptMode::AUTO, "resolve AUTO before instantiating the device kernel");
  static_assert(MODE == LptMode::OFF || !KV_SNAKE, "LPT and KV snake are exclusive");
  const int q_block_id  = (int)blockIdx.x;
  const int batch_head  = (int)blockIdx.y;
  const int batch = batch_head / num_heads, head = batch_head % num_heads;
  const int token_begin = q_block_id * Q_TILE;

  if constexpr (MODE != LptMode::OFF || KV_SNAKE) {
    if (q_block_id >= seqlen / Q_TILE) {
      extern __shared__ int remap_smem[];
      if constexpr (MODE == LptMode::PER_HEAD)
        build_lpt_remap<false>(k2q_num, workitem_remap, batch_head * num_kv_blocks,
                              num_kv_blocks, max_q_blocks, num_kv_blocks, remap_smem);
      else if constexpr (MODE == LptMode::GLOBAL) {
        if (batch_head == 0)
          build_lpt_remap<true>(k2q_num, workitem_remap, 0,
                                (int)gridDim.y * num_kv_blocks, max_q_blocks, num_kv_blocks,
                                remap_smem);
      }
      if constexpr (KV_SNAKE)
        build_kv_snake_remap(k2q_idx, k2q_num, workitem_remap, batch_head,
                             num_kv_blocks, max_q_blocks, count_bin, order_mode, remap_smem);
      return;
    }
  }

  // The previous postprocess still reads dqaccum: wait before zeroing it.
  if constexpr (KERNEL_PDL) griddepcontrol_wait();
  // Zero this block's dqaccum slice: 16384 f32 = 4096 float4 by 256 threads.
  float4* dqaccum_zero_destination = reinterpret_cast<float4*>(
      dqaccum + (size_t)batch_head * seqlen * HEAD_DIM + (size_t)q_block_id * DQ::DQ_BLOCK_ELEMS);
  const float4 zero_float4 = make_float4(0.f, 0.f, 0.f, 0.f);
  #pragma unroll
  for (int i = 0; i < (DQ::DQ_BLOCK_ELEMS / 4) / 256; ++i)
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

// Postprocess (FA4's scheme, 1CTA path): one CTA per (q128 block,
// batch*head), 128 threads, 64 KB SMEM.
//   A. The whole 16384-f32 drain-native block is loaded contiguously with cp.async.cg (32 x 16 B
//      per thread).
//   B. Thread t unscrambles q row t into registers: chunk c, float4 group v4 sits at
//      smem[c * 4096 + v4 * 512 + t * 4] (the epilogue warps' staging order; a warp reads 512 B
//      contiguous), times sm_scale, packed to bf16.
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
      dqaccum + (size_t)batch_head * seqlen * HEAD_DIM + (size_t)q_block_id * DQ::DQ_BLOCK_ELEMS);
  if constexpr (KERNEL_PDL) griddepcontrol_wait();  // the main grid's pushes are complete
  #pragma unroll
  for (int i = 0; i < (DQ::DQ_BLOCK_ELEMS / 4) / 128; ++i)
    cp_async_cg_16(smem_ptr_u32(post_smem + (i * 128 + thread) * 4),
                   dqaccum_block + i * 128 + thread);
  cp_async_commit_group();
  cp_async_wait_group<0>();
  __syncthreads();
  // dqaccum read out: the next preprocess may launch (it waits before re-zeroing).
  if constexpr (KERNEL_PDL) griddepcontrol_launch_dependents();

  // B
  uint32_t dq_packed[HEAD_DIM / 2];  // this row's 128 hd as bf16x2, ascending
  #pragma unroll
  for (int hd_slice = 0; hd_slice < HEAD_DIM / DQ::COLS; ++hd_slice) {
    #pragma unroll
    for (int v4 = 0; v4 < DQ::COLS / 4; ++v4) {
      const float4 value = *reinterpret_cast<const float4*>(
          post_smem + hd_slice * Q_TILE * DQ::COLS + v4 * Q_TILE * 4 + thread * 4);
      const int pair_index = (hd_slice * DQ::COLS + v4 * 4) / 2;
      dq_packed[pair_index + 0] = cvt_f32x2_to_bf16x2(value.x * sm_scale, value.y * sm_scale);
      dq_packed[pair_index + 1] = cvt_f32x2_to_bf16x2(value.z * sm_scale, value.w * sm_scale);
    }
  }
  __syncthreads();

  // C: chunk v of the row goes to hd half v / 8, chunk slot (v % 8) ^ (thread & 7).
  __nv_bfloat16* dq_tile_bf16 = reinterpret_cast<__nv_bfloat16*>(post_smem);
  const uint4* dq_chunks      = reinterpret_cast<const uint4*>(dq_packed);
  __nv_bfloat16* dq_row_bf16  = dq_tile_bf16 + thread * HEAD_DIM;
  #pragma unroll
  for (int v = 0; v < HEAD_DIM / CHUNK_BF16; ++v)
    *reinterpret_cast<uint4*>(dq_row_bf16 + (v / 8) * 64 + ((v % 8) ^ (thread & 7)) * CHUNK_BF16) =
        dq_chunks[v];
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
  const float* lse_rows;                    // [B*H, S]
  const float* delta_rows;                  // [B*H, S] FP32
  const int* k2q_idx;                      // [B*H*nb, max_q_blocks] local q128 ids, padded rows
  const int* k2q_num;                      // [B*H*nb] entries valid per row
  const int* workitem_remap;               // [B*H*nb] head-local logical id -> local KV id
  uint32_t* clc_work_counter;              // [B*H], one head-local counter per CLC grid
  bool ordered_kv_blocks;                  // selects ORDERED_KV_BLOCKS kernel specialization
  bool k2q_traversal_snake;                // selects K2Q_TRAVERSAL_SNAKE specialization
  bool clc_per_head;                        // host-only CLC_HEAD_MAJOR per-head launch policy
  bool cta_waves;                          // <= #SM ordered NON_PERSISTENT one-WI/CTA grids
  bool alternate_wave_direction;           // wave 0 asc, wave 1 desc, then alternate
  bool cooperative_grid_sync;              // select this_grid().sync() defensive specialization
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

template <Sched SCHED = KERNEL_SCHED, bool DK_FROM_SMEM = false,
          bool EXP_PACKED_DS = KERNEL_EXP_PACKED_DS>
inline cudaError_t launch_vsa_bwd_sm100a(const VsaBwdArgs& a, cudaStream_t stream) {
  const int B = a.num_samples, H = a.num_heads, S = a.seqlen;

  // Tensor maps are pure functions of (pointer, shape): encode once per config (re-encoding per
  // launch costs ~100+ us of driver time per call).
  constexpr int SMEM_BYTES = SMEM_TOTAL + (EXP_PACKED_DS ? 8 : 0);
  auto kernel_ascending_unordered =
      vsa_bwd_main_kernel<VSA_BHSD, SCHED, false, false, false, false, DK_FROM_SMEM,
                          EXP_PACKED_DS>;
  auto kernel_snake_unordered =
      vsa_bwd_main_kernel<VSA_BHSD, SCHED, false, false, true, false, DK_FROM_SMEM,
                          EXP_PACKED_DS>;
  auto kernel_ascending_ordered =
      vsa_bwd_main_kernel<VSA_BHSD, SCHED, false, true, false, false, DK_FROM_SMEM,
                          EXP_PACKED_DS>;
  auto kernel_snake_ordered =
      vsa_bwd_main_kernel<VSA_BHSD, SCHED, false, true, true, false, DK_FROM_SMEM,
                          EXP_PACKED_DS>;
  auto kernel_wave_ascending =
      vsa_bwd_main_kernel<VSA_BHSD, Sched::NON_PERSISTENT, false, true, false, false,
                          DK_FROM_SMEM, EXP_PACKED_DS>;
  auto kernel_wave_descending =
      vsa_bwd_main_kernel<VSA_BHSD, Sched::NON_PERSISTENT, true, true, false, false,
                          DK_FROM_SMEM, EXP_PACKED_DS>;
  auto kernel = a.ordered_kv_blocks
                    ? (a.k2q_traversal_snake ? kernel_snake_ordered : kernel_ascending_ordered)
                    : (a.k2q_traversal_snake ? kernel_snake_unordered
                                             : kernel_ascending_unordered);
  if constexpr (SCHED == Sched::DEFENSIVE) {
    if (a.cooperative_grid_sync) {
      auto kernel_grid_ascending_unordered =
          vsa_bwd_main_kernel<VSA_BHSD, SCHED, false, false, false, true, DK_FROM_SMEM,
                              EXP_PACKED_DS>;
      auto kernel_grid_snake_unordered =
          vsa_bwd_main_kernel<VSA_BHSD, SCHED, false, false, true, true, DK_FROM_SMEM,
                              EXP_PACKED_DS>;
      auto kernel_grid_ascending_ordered =
          vsa_bwd_main_kernel<VSA_BHSD, SCHED, false, true, false, true, DK_FROM_SMEM,
                              EXP_PACKED_DS>;
      auto kernel_grid_snake_ordered =
          vsa_bwd_main_kernel<VSA_BHSD, SCHED, false, true, true, true, DK_FROM_SMEM,
                              EXP_PACKED_DS>;
      kernel = a.ordered_kv_blocks
                   ? (a.k2q_traversal_snake ? kernel_grid_snake_ordered
                                            : kernel_grid_ascending_ordered)
                   : (a.k2q_traversal_snake ? kernel_grid_snake_unordered
                                            : kernel_grid_ascending_unordered);
    }
  }
  static CUtensorMap tq_, tk_, tv_, tdo_, tdk_, tdv_;
  static const void* cached_q = nullptr;
  static int cached_B = 0, cached_S = 0, cached_H = 0;
  if (cached_q != (const void*)a.q || cached_B != B || cached_S != S || cached_H != H) {
    const bool ok = make_tma_tile_units(&tq_, a.q, B, H, S, Q_TILE) == cudaSuccess &&
                    make_tma_tile_units(&tk_, a.k, B, H, S, KV_TILE) == cudaSuccess &&
                    make_tma_tile_units(&tv_, a.v, B, H, S, KV_TILE) == cudaSuccess &&
                    make_tma_tile_units(&tdo_, a.dout, B, H, S, Q_TILE) == cudaSuccess &&
                    make_tma_tile_units(&tdk_, a.dk, B, H, S, KV_TILE) == cudaSuccess &&
                    make_tma_tile_units(&tdv_, a.dv, B, H, S, KV_TILE) == cudaSuccess;
    if (!ok) return cudaErrorInvalidValue;
    cached_q = (const void*)a.q;
    cached_B = B;
    cached_S = S;
    cached_H = H;
  }
  static bool smem_set  = false;
  static int sms        = 0;
  if (!smem_set) {
    cudaError_t e = cudaFuncSetAttribute(kernel_ascending_unordered,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    if (e != cudaSuccess) return e;
    e = cudaFuncSetAttribute(kernel_snake_unordered,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    if (e != cudaSuccess) return e;
    e = cudaFuncSetAttribute(kernel_ascending_ordered,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    if (e != cudaSuccess) return e;
    e = cudaFuncSetAttribute(kernel_snake_ordered,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             SMEM_BYTES);
    if (e != cudaSuccess) return e;
    e = cudaFuncSetAttribute(kernel_wave_ascending,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    if (e != cudaSuccess) return e;
    e = cudaFuncSetAttribute(kernel_wave_descending,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    if (e != cudaSuccess) return e;
    if constexpr (SCHED == Sched::DEFENSIVE) {
      e = cudaFuncSetAttribute(
          vsa_bwd_main_kernel<VSA_BHSD, SCHED, false, false, false, true, DK_FROM_SMEM,
                              EXP_PACKED_DS>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
      if (e != cudaSuccess) return e;
      e = cudaFuncSetAttribute(
          vsa_bwd_main_kernel<VSA_BHSD, SCHED, false, false, true, true, DK_FROM_SMEM,
                              EXP_PACKED_DS>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
      if (e != cudaSuccess) return e;
      e = cudaFuncSetAttribute(
          vsa_bwd_main_kernel<VSA_BHSD, SCHED, false, true, false, true, DK_FROM_SMEM,
                              EXP_PACKED_DS>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
      if (e != cudaSuccess) return e;
      e = cudaFuncSetAttribute(
          vsa_bwd_main_kernel<VSA_BHSD, SCHED, false, true, true, true, DK_FROM_SMEM,
                              EXP_PACKED_DS>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
      if (e != cudaSuccess) return e;
    }
    e = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    if (e != cudaSuccess) return e;
    smem_set = true;
  }

  const float scale_log2 = a.sm_scale * 1.4426950408889634f;
  const int total_items  = B * H * a.num_kv_blocks_per_seq;
  if (a.ordered_kv_blocks != (a.workitem_remap != nullptr)) return cudaErrorInvalidValue;
  // Grid: CLC = one CTA per work item (the resident wave runs first, try_cancel hands each
  // finisher the next uncancelled item); STATIC = min(items, #SMs); NON_PERSISTENT = all items;
  // DEFENSIVE = one cooperative <= #SM grid per head.
  cudaLaunchConfig_t cfg = {};
  if constexpr (SCHED == Sched::CLC || SCHED == Sched::CLC_HEAD_MAJOR)
    cfg.gridDim = dim3((unsigned)total_items, 1, 1);
  else if constexpr (SCHED == Sched::STATIC_PERSISTENT)
    cfg.gridDim = dim3((unsigned)std::min(total_items, sms), 1, 1);
  else if constexpr (SCHED == Sched::DEFENSIVE)
    cfg.gridDim = dim3((unsigned)std::min(a.num_kv_blocks_per_seq, sms), 1, 1);
  else
    cfg.gridDim = dim3((unsigned)total_items, 1, 1);
  cfg.blockDim           = dim3(N_WARPS * 32, 1, 1);
  cfg.dynamicSmemBytes   = SMEM_BYTES;
  cfg.stream             = stream;
  cudaLaunchAttribute at[3] = {};
  int num_attrs             = 0;
  at[num_attrs].id               = cudaLaunchAttributeClusterDimension;
  at[num_attrs].val.clusterDim.x = 1;
  at[num_attrs].val.clusterDim.y = 1;
  at[num_attrs].val.clusterDim.z = 1;
  ++num_attrs;
  if constexpr (KERNEL_PDL) {
    at[num_attrs].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[num_attrs].val.programmaticStreamSerializationAllowed = 1;
    ++num_attrs;
  }
  if constexpr (SCHED == Sched::DEFENSIVE) {
    at[num_attrs].id              = cudaLaunchAttributeCooperative;
    at[num_attrs].val.cooperative = 1;
    ++num_attrs;
  }
  cfg.attrs              = at;
  cfg.numAttrs           = num_attrs;
  auto launch = [&](auto selected_kernel, int workitem_begin, int batch_head_base,
                    const int* workitem_remap, uint32_t* clc_work_counter) {
    return cudaLaunchKernelEx(&cfg, selected_kernel, tq_, tk_, tv_, tdo_, tdk_, tdv_, a.dqaccum,
                              a.lse_rows, a.delta_rows, a.k2q_idx, a.k2q_num, a.max_q_blocks,
                              workitem_remap, clc_work_counter, workitem_begin, batch_head_base,
                              B, H, S, scale_log2, a.sm_scale);
  };
  if (a.cta_waves) {
    if constexpr (SCHED != Sched::CLC_HEAD_MAJOR) return cudaErrorInvalidValue;
    if (a.clc_per_head || !a.ordered_kv_blocks || a.k2q_traversal_snake)
      return cudaErrorInvalidValue;
    for (int wave_begin = 0, wave = 0; wave_begin < total_items; wave_begin += sms, ++wave) {
      cfg.gridDim = dim3((unsigned)std::min(sms, total_items - wave_begin), 1, 1);
      auto wave_kernel = a.alternate_wave_direction && (wave & 1)
                             ? kernel_wave_descending
                             : kernel_wave_ascending;
      cudaError_t e = launch(wave_kernel, wave_begin, 0, a.workitem_remap,
                             a.clc_work_counter);
      if (e != cudaSuccess) return e;
    }
    return cudaSuccess;
  }
  if constexpr (SCHED == Sched::DEFENSIVE) {
    if (a.clc_per_head || a.cta_waves) return cudaErrorInvalidValue;
    // A cooperative launch guarantees that every CTA is resident, making the grid barriers
    // between defensive waves deadlock-free. One grid owns one head and reuses its L2 footprint.
    for (int batch_head = 0; batch_head < B * H; ++batch_head) {
      cudaError_t e = launch(kernel, 0, batch_head, a.workitem_remap,
                             a.clc_work_counter + batch_head);
      if (e != cudaSuccess) return e;
    }
    return cudaSuccess;
  }
  if constexpr (SCHED == Sched::CLC_HEAD_MAJOR) {
    if (a.clc_per_head) {
      // One grid per head: head-local logical ids under batch_head_base and a separate counter;
      // the kernel indexes the global remap at batch_head_base * nb + logical id.
      cfg.gridDim = dim3((unsigned)a.num_kv_blocks_per_seq, 1, 1);
      for (int batch_head = 0; batch_head < B * H; ++batch_head) {
        cudaError_t e = launch(kernel, 0, batch_head, a.workitem_remap,
                               a.clc_work_counter + batch_head);
        if (e != cudaSuccess) return e;
      }
      return cudaSuccess;
    }
  } else if (a.clc_per_head) return cudaErrorInvalidValue;
  return launch(kernel, 0, 0, a.workitem_remap, a.clc_work_counter);
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

template <LptMode MODE = LptMode::OFF, bool KV_SNAKE = false>
inline cudaError_t launch_vsa_bwd_preprocess(const __nv_bfloat16* o, const __nv_bfloat16* dout,
                                             float* delta_rows, float* dqaccum, int num_samples,
                                             int num_heads, int seqlen, const int* k2q_idx,
                                             const int* k2q_num, int* workitem_remap,
                                             int num_kv_blocks, int max_q_blocks, int count_bin,
                                             int order_mode, cudaStream_t stream) {
  static_assert(MODE == LptMode::OFF || !KV_SNAKE, "LPT and KV snake are exclusive");
  if constexpr (MODE == LptMode::AUTO) {
    if ((long)num_samples * num_heads * seqlen < 65536)
      return launch_vsa_bwd_preprocess<LptMode::GLOBAL, false>(
          o, dout, delta_rows, dqaccum, num_samples, num_heads, seqlen, k2q_idx, k2q_num,
          workitem_remap, num_kv_blocks, max_q_blocks, count_bin, order_mode, stream);
    if (seqlen <= 16384)
      return launch_vsa_bwd_preprocess<LptMode::PER_HEAD, false>(
          o, dout, delta_rows, dqaccum, num_samples, num_heads, seqlen, k2q_idx, k2q_num,
          workitem_remap, num_kv_blocks, max_q_blocks, count_bin, order_mode, stream);
    return launch_vsa_bwd_preprocess<LptMode::OFF, false>(
        o, dout, delta_rows, dqaccum, num_samples, num_heads, seqlen, k2q_idx, k2q_num,
        workitem_remap, num_kv_blocks, max_q_blocks, count_bin, order_mode, stream);
  } else {
    constexpr bool LPT = MODE != LptMode::OFF;
    if constexpr (MODE == LptMode::GLOBAL) {
      if (num_kv_blocks > 65535 || (long)num_samples * num_heads > 32766)
        return cudaErrorInvalidValue;
    }
    const int lpt_items = MODE == LptMode::PER_HEAD
                              ? num_kv_blocks : num_samples * num_heads * num_kv_blocks;
    const size_t lpt_smem = LPT ? (size_t)(lpt_items + max_q_blocks + 1) * sizeof(int) : 0;
    int snake_items = 1;
    while (snake_items < num_kv_blocks) snake_items <<= 1;
    const size_t snake_smem = KV_SNAKE ? (size_t)snake_items * sizeof(int) : 0;
    const size_t remap_smem = LPT ? lpt_smem : snake_smem;
    if (remap_smem > 48 * 1024 ||
        ((LPT || KV_SNAKE) && (workitem_remap == nullptr || k2q_num == nullptr)) ||
        (KV_SNAKE && (k2q_idx == nullptr || count_bin <= 0)))
      return cudaErrorInvalidValue;
    cudaLaunchAttribute at[1];
    cudaLaunchConfig_t cfg = pdl_launch_config(
        dim3((unsigned)(seqlen / Q_TILE + ((LPT || KV_SNAKE) ? 1 : 0)),
             (unsigned)(num_samples * num_heads), 1),
        dim3(256, 1, 1), remap_smem, stream, at);
    return cudaLaunchKernelEx(&cfg, vsa_bwd_preprocess_kernel<VSA_BHSD, MODE, KV_SNAKE>, o,
                              dout, delta_rows, dqaccum, num_heads, seqlen, k2q_idx, k2q_num,
                              workitem_remap, num_kv_blocks, max_q_blocks, count_bin,
                              order_mode);
  }
}

inline cudaError_t launch_vsa_bwd_postprocess(const float* dqaccum, __nv_bfloat16* dq,
                                              int num_samples, int num_heads, int seqlen,
                                              float sm_scale, cudaStream_t stream) {
  constexpr int POST_SMEM_BYTES = DQ::DQ_BLOCK_ELEMS * (int)sizeof(float);
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
  return cudaLaunchKernelEx(&cfg, vsa_bwd_postprocess_kernel<VSA_BHSD>, dqaccum, dq, num_heads,
                            seqlen, sm_scale);
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

  // LPT is a compile-time NON_PERSISTENT ordering policy. AUTO selects the measured small-S
  // policy and disables LPT where its last-wave benefit disappears.
  const LptMode lpt_mode = KERNEL_LPT_MODE == LptMode::AUTO
                               ? ((long)B * H * S < 65536 ? LptMode::GLOBAL
                                  : S <= 16384           ? LptMode::PER_HEAD
                                                         : LptMode::OFF)
                               : KERNEL_LPT_MODE;
  const bool lpt = lpt_mode != LptMode::OFF;
  // Cheap within-head locality order. Reverse lists are already sorted Q ids, so list length
  // aligns iteration quantiles and the midpoint approximates the Q neighborhood visited.
  const bool can_order = KERNEL_SCHED == Sched::CLC_HEAD_MAJOR ||
                         KERNEL_SCHED == Sched::DEFENSIVE || lpt;
  const bool clc_per_head =
      KERNEL_SCHED == Sched::CLC_HEAD_MAJOR && getenv("CLC_PER_HEAD") &&
      atoi(getenv("CLC_PER_HEAD")) != 0;
  const bool cta_waves =
      KERNEL_SCHED == Sched::CLC_HEAD_MAJOR && !clc_per_head && getenv("K2Q_WAVES") &&
      atoi(getenv("K2Q_WAVES")) != 0;
  const bool alternate_wave_direction =
      cta_waves && getenv("K2Q_WAVE_SNAKE") && atoi(getenv("K2Q_WAVE_SNAKE")) != 0;
  const bool order_k2q =
      lpt || (can_order && (!getenv("K2Q_ORDER") || atoi(getenv("K2Q_ORDER")) != 0));
  const char* traversal_snake_env = getenv("K2Q_TRAVERSAL_SNAKE");
  const bool traversal_snake =
      !lpt && can_order && !cta_waves &&
      (!traversal_snake_env || atoi(traversal_snake_env) != 0);
  const bool cooperative_grid_sync =
      KERNEL_SCHED == Sched::DEFENSIVE && getenv("COOPERATIVE_GRID_SYNC") &&
      atoi(getenv("COOPERATIVE_GRID_SYNC")) != 0;
  if (cta_waves && !order_k2q) {
    fprintf(stderr, "K2Q_WAVES requires ordered KV blocks (K2Q_ORDER=1)\n");
    exit(1);
  }
  std::vector<int> workitem_remap;
  double order_ms = 0.0;
  int count_bin = 0, order_mode = 0;
  if (order_k2q) {
    const auto order_t0 = std::chrono::steady_clock::now();
    workitem_remap.resize((size_t)B * H * num_blocks);
    if (lpt) {
      if (lpt_mode == LptMode::GLOBAL) {
        std::vector<int> items(workitem_remap.size());
        for (size_t i = 0; i < items.size(); ++i) items[i] = (int)i;
        std::stable_sort(items.begin(), items.end(), [&](int a, int b) {
          return k2q.count[a] != k2q.count[b] ? k2q.count[a] > k2q.count[b] : a < b;
        });
        for (size_t p = 0; p < items.size(); ++p) {
          const int item = items[p];
          workitem_remap[p] = (((item / num_blocks + 1) << 16) | (item % num_blocks));
        }
      } else {
        for (int batch_head = 0; batch_head < B * H; ++batch_head) {
          int* first = workitem_remap.data() + (size_t)batch_head * num_blocks;
          for (int n = 0; n < num_blocks; ++n) first[n] = n;
          std::stable_sort(first, first + num_blocks, [&](int a, int b) {
            const int ca = k2q.count[batch_head * num_blocks + a];
            const int cb = k2q.count[batch_head * num_blocks + b];
            return ca != cb ? ca > cb : a < b;
          });
        }
      }
    } else {
      // Explicit waves use tighter length alignment; other ordered schedulers retain bin 12.
      const int default_count_bin = cta_waves && S >= 131072 ? 4 : 12;
      count_bin = getenv("K2Q_COUNT_BIN") ? std::max(1, atoi(getenv("K2Q_COUNT_BIN")))
                                             : default_count_bin;
      order_mode = getenv("K2Q_ORDER_MODE") ? atoi(getenv("K2Q_ORDER_MODE")) : 0;
      std::vector<uint64_t> trajectory_key;
      if (order_mode == 2) {
        trajectory_key.resize((size_t)B * H * num_blocks);
        for (int item = 0; item < B * H * num_blocks; ++item) {
          const int count = k2q.count[item];
          if (count == 0) continue;
          const int q1 = k2q_padded.idx[(size_t)item * k2q_padded.max_q_blocks + count / 4];
          const int q2 = k2q_padded.idx[(size_t)item * k2q_padded.max_q_blocks + count / 2];
          const int q3 = k2q_padded.idx[(size_t)item * k2q_padded.max_q_blocks + 3 * count / 4];
          uint64_t morton = 0;
          for (int bit = 0; bit < 16; ++bit) {
            morton |= (uint64_t)((q1 >> bit) & 1) << (3 * bit + 0);
            morton |= (uint64_t)((q2 >> bit) & 1) << (3 * bit + 1);
            morton |= (uint64_t)((q3 >> bit) & 1) << (3 * bit + 2);
          }
          trajectory_key[item] = morton;
        }
      }
      for (int batch_head = 0; batch_head < B * H; ++batch_head) {
        int* first = workitem_remap.data() + (size_t)batch_head * num_blocks;
        for (int n = 0; n < num_blocks; ++n) first[n] = n;
        std::stable_sort(first, first + num_blocks, [&](int local_a, int local_b) {
          const int a = batch_head * num_blocks + local_a;
          const int b = batch_head * num_blocks + local_b;
          const int ca = k2q.count[a], cb = k2q.count[b];
          const int ba = ca / count_bin, bb = cb / count_bin;
          if (ba != bb) return ba < bb;
          if (order_mode == 1 && ca && cb) {
            for (int quantile = 1; quantile <= 3; ++quantile) {
              const int qa = k2q_padded.idx[(size_t)a * k2q_padded.max_q_blocks +
                                            quantile * ca / 4];
              const int qb = k2q_padded.idx[(size_t)b * k2q_padded.max_q_blocks +
                                            quantile * cb / 4];
              if (qa != qb) return (ba & 1) ? qa > qb : qa < qb;
            }
          }
          if (order_mode == 2 && trajectory_key[a] != trajectory_key[b])
            return (ba & 1) ? trajectory_key[a] > trajectory_key[b]
                            : trajectory_key[a] < trajectory_key[b];
          const int ma = ca ? k2q_padded.idx[(size_t)a * k2q_padded.max_q_blocks + ca / 2] : -1;
          const int mb = cb ? k2q_padded.idx[(size_t)b * k2q_padded.max_q_blocks + cb / 2] : -1;
          if (ma != mb) return (ba & 1) ? ma > mb : ma < mb;
          return local_a < local_b;
        });
      }
    }
    const auto order_t1 = std::chrono::steady_clock::now();
    order_ms = std::chrono::duration<double, std::milli>(order_t1 - order_t0).count();
    if (lpt) {
      printf("  lpt oracle: %s order, %.3f ms, %.1f KiB (device sort is timed)\n",
             lpt_mode == LptMode::GLOBAL ? "global" : "per-head", order_ms,
             workitem_remap.size() * sizeof(int) / 1024.0);
    } else {
      const char* order_name = order_mode == 1 ? "quartile lexicographic"
                               : order_mode == 2 ? "quartile Morton"
                                                 : "midpoint";
      printf("  k2q oracle: count-bin%d + %s snake, %.3f ms, %.1f KiB (device sort is timed)\n",
             count_bin, order_name, order_ms,
             workitem_remap.size() * sizeof(int) / 1024.0);
    }
  }
  if (traversal_snake) printf("  k2q traversal: per-CTA ascending/descending snake\n");
  if (KERNEL_SCHED == Sched::DEFENSIVE)
    printf("  defensive barrier: %s\n", cooperative_grid_sync
                                               ? "cooperative_groups::this_grid().sync()"
                                               : "one atomic arrival per CTA");
  if (cta_waves)
    printf("  k2q waves: <= one CTA/SM, %s direction\n",
           alternate_wave_direction ? "alternating" : "ascending");

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
      fprintf(stderr, "LOAD_NPY: forward state size mismatch; rerun block_sparse_bf16_gen_inputs.py\n");
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
    int *dK2qIdx, *dK2qNum, *dOrder = nullptr;
    uint32_t* dClcWorkCounter;
    CUDA_CHECK(cudaMalloc(&dQg, elems * 2));
    CUDA_CHECK(cudaMalloc(&dKg, elems * 2));
    CUDA_CHECK(cudaMalloc(&dVg, elems * 2));
    CUDA_CHECK(cudaMalloc(&dDOg, elems * 2));
    CUDA_CHECK(cudaMalloc(&dOg, elems * 2));
    CUDA_CHECK(cudaMalloc(&dDKout, elems * 2));
    CUDA_CHECK(cudaMalloc(&dDVout, elems * 2));
    CUDA_CHECK(cudaMalloc(&dDQout, elems * 2));
    CUDA_CHECK(cudaMalloc(&dMg, (size_t)B * H * S * 4));
    CUDA_CHECK(cudaMalloc(&dDeltag, (size_t)B * H * S * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dDQA, (size_t)B * H * S * hd * 4));
    CUDA_CHECK(cudaMalloc(&dK2qIdx, k2q_padded.idx.size() * 4));
    CUDA_CHECK(cudaMalloc(&dK2qNum, k2q_padded.num.size() * 4));
    if (order_k2q) CUDA_CHECK(cudaMalloc(&dOrder, workitem_remap.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dClcWorkCounter, (size_t)B * H * sizeof(uint32_t)));
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
    args.workitem_remap        = dOrder;
    args.clc_work_counter      = dClcWorkCounter;
    args.ordered_kv_blocks     = order_k2q;
    args.k2q_traversal_snake   = traversal_snake;
    args.clc_per_head          = clc_per_head;
    args.cta_waves             = cta_waves;
    args.alternate_wave_direction = alternate_wave_direction;
    args.cooperative_grid_sync    = cooperative_grid_sync;
    args.max_q_blocks          = k2q_padded.max_q_blocks;
    args.num_samples           = B;
    args.num_heads             = H;
    args.seqlen                = S;
    args.num_kv_blocks_per_seq = num_blocks;
    args.sm_scale              = 1.0f / sqrtf((float)hd);

    auto run_once = [&]() {
      if constexpr (KERNEL_SCHED == Sched::CLC_HEAD_MAJOR ||
                    KERNEL_SCHED == Sched::DEFENSIVE)
        CUDA_CHECK(cudaMemsetAsync(dClcWorkCounter, 0, (size_t)B * H * sizeof(uint32_t), 0));
      if (order_k2q && !lpt) {
        CUDA_CHECK((launch_vsa_bwd_preprocess<LptMode::OFF, true>(
            dOg, dDOg, dDeltag, dDQA, B, H, S, dK2qIdx, dK2qNum, dOrder, num_blocks,
            k2q_padded.max_q_blocks, count_bin, order_mode, 0)));
      } else {
        CUDA_CHECK((launch_vsa_bwd_preprocess<KERNEL_LPT_MODE, false>(
            dOg, dDOg, dDeltag, dDQA, B, H, S, dK2qIdx, dK2qNum, dOrder, num_blocks,
            k2q_padded.max_q_blocks, count_bin, order_mode, 0)));
      }
      CUDA_CHECK(launch_vsa_bwd_sm100a(args, 0));
      CUDA_CHECK(launch_vsa_bwd_postprocess(dDQA, dDQout, B, H, S, args.sm_scale, 0));
    };
    run_once();
    CUDA_CHECK(cudaDeviceSynchronize());

    if (order_k2q) {
      std::vector<int> device_order(workitem_remap.size());
      CUDA_CHECK(cudaMemcpy(device_order.data(), dOrder, device_order.size() * sizeof(int),
                            cudaMemcpyDeviceToHost));
      size_t mismatches = 0;
      for (size_t i = 0; i < device_order.size(); ++i) {
        if (device_order[i] != workitem_remap[i]) {
          if (mismatches < 12)
            printf("    remap[%zu]: device=%d expected=%d\n", i, device_order[i],
                   workitem_remap[i]);
          ++mismatches;
        }
      }
      printf("  device %s order: %zu entries, %zu mismatches vs host oracle\n",
             lpt ? "LPT" : "KV snake", device_order.size(), mismatches);
      if (mismatches) exit(1);
    }

    if (run_cpu || getenv("DUMP_BWD_GPU")) {
      std::vector<float> gDelta((size_t)B * H * S);
      std::vector<__nv_bfloat16> gDK(elems), gDV(elems), gDQ(elems);
      CUDA_CHECK(cudaMemcpy(gDelta.data(), dDeltag, gDelta.size() * sizeof(float),
                            cudaMemcpyDeviceToHost));
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
      // production bf16 quantization points is ~1.9-2.7e-3, GPU-vs-CPU can
      // legitimately reach ~2x that, and the
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
      wp_dump_raw(wp, "warp_raw_vsa_bwd_blk128.bin.gz", wp.view_block, 2);
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
    if (dOrder) cudaFree(dOrder);
    cudaFree(dClcWorkCounter);
  }
}

int main() {
  CUDA_CHECK(cudaFree(0));
  printf(
      "VSA block-sparse BACKWARD bench bf16 (blk128-native, FA4-aligned) sm_100a\n"
      "%s 16-warp kernel; pre+main+post (block=%d)\n"
      "=====================================\n",
      KERNEL_SCHED == Sched::CLC_HEAD_MAJOR && getenv("CLC_PER_HEAD") &&
              atoi(getenv("CLC_PER_HEAD")) != 0  ? "CLC_PER_HEAD host policy"
      : KERNEL_SCHED == Sched::DEFENSIVE         ? "defensive persistent"
      : KERNEL_SCHED == Sched::CLC_HEAD_MAJOR    ? "CLC_HEAD_MAJOR persistent"
      : KERNEL_SCHED == Sched::CLC               ? "CLC persistent"
      : KERNEL_SCHED == Sched::STATIC_PERSISTENT ? "static persistent"
                                                 : "non-persistent",
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
