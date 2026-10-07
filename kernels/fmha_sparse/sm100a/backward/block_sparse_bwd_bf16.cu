// block_sparse_bwd_bf16.cu -- VSA block-sparse BACKWARD, sm_100a.
// Single-file basis kernel + bench harness (counterpart of
// ../block_sparse_bf16_uniform.cu).
//
// Backward of the SPARSE BASIS forward (../block_sparse_bf16_uniform.cu semantics):
//   - uniform per-row q2k counts (every q-block selects the same topk >= 1)
//   - every KV block is full (no variable_block_sizes)
//   - B=1-any, H heads, D=128, bf16 inputs, natural layout [token, head, hd]
//     (= [S,H,D] at B=1, matching the unified bench .npy files)
//
// M1 contents: CPU fp32 reference (forward O + log2-domain M, Delta, dQ/dK/dV),
// deterministic sorted k2q inversion (kept for the M3 GPU kernel; histogram printed),
// DUMP_BWD npy dumps. GPU section is a TODO (M3) placeholder.
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
// DUMP_BWD=<prefix> writes fp32 C-order npy:
//   <prefix>_dq.npy / _dk.npy / _dv.npy   shape [B*S, H, D]  (= [S,H,D] at B=1)
//   <prefix>_M.npy  / _delta.npy          shape [B*H, S]     (= [H,S]   at B=1)
//   M is log2-domain: M = max(score2) + log2(l).
//
// Env knobs (mirrors the forward bench):
//   LOAD_NPY=<dir>   q_S{S}.npy k_S{S}.npy v_S{S}.npy do_S{S}.npy (uint16 raw bf16 bits,
//                    [S,H,D]) + idx_S{S}_blk{BLOCK}.npy (int32 [nb,topk], head-independent)
//   BLOCK=64|128     sparse block size (runtime here; compile-time in the forward benches)
//   SHAPE=0..2 + BATCH/HEADS/NB/TOPK   single-shape override
//   VSA_GAUSS / VSA_SORT_SEL / VSA_SEED_QBLK   built-in fill / index knobs (as the forward bench)
//   CPU_REF=0|1      skip / force the CPU reference (default: small shapes only,
//                    forced when DUMP_BWD is set)
//   DUMP_BWD=<prefix>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <chrono>
#include <climits>
#include <vector>
#include <algorithm>
#include <string>
#include "../npy_io.cuh"
#include "block_sparse_bwd_bf16_benchmark.cuh"
#include "../../../../tests/test_helpers.cuh"
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
#include "../../../../primitives/35_fence_mbarrier_init.cuh"
#include "../../../../primitives/37_bar_sync.cuh"
#include "../../../../primitives/42_smem_desc_blackwell.cuh"
#include "../../../../primitives/44_elect_sync.cuh"
#include "../../../../primitives/46_setmaxnreg.cuh"
#include "../../../../primitives/50_atom_global.cuh"
#include "../../../../primitives/61_clc_try_cancel.cuh"
#include "../../../../primitives/63_cvt_f32_to_f16_bf16.cuh"
#include "../../../../primitives/70_smem_ptr.cuh"
#include "../../../../primitives/76_packed_f32x2.cuh"
#include "../../../../primitives/77_ex2_approx.cuh"
#include "../../../../composites/118_mbarrier_phase_tracking.cuh"
#include "../../../../composites/106_clc_fetch_next_tile.cuh"
#include "../../../../primitives/_warp_prof_noop.cuh"

// sm_100a VSA block-sparse BACKWARD, blk64 v5 -- FA4's exact tile and flow.
//
// KV-stationary main kernel on FA4's 128x128 tile: each CTA owns an ADJACENT
// PAIR of 64-token KV blocks (kv rows 0-63 = even block, 64-127 = odd) and
// walks the pair's sorted union of Q-BLOCK PAIRS (128 q rows per step).
// Entries pack q_pair (bits 0-27) + quadrant membership flags (bits 28-31,
// kv-half x q-half); non-member quadrants are zeroed in the compute warps.
//
// Five GEMMs per q-step j, FA4's rotation, with in-order tcgen05 execution
// as the TMEM-aliasing hazard barrier:
//   S(j+1) -> dK(j) -> dQc(j) -> dP(j+1) -> dV(j+1)
//   ST  = Kpair . Q^T         SS  M128 N128 K128 -> TMEM 0-127 (raw dot)
//   dPT = Vpair . dO^T        SS  M128 N128 K128 -> TMEM 256-383
//   PT  = exp2(ST*scale_log2 - M[q]) masked; STTM bf16 over ST cols
//         {0-31, 64-95} (each q-half packs INSIDE its own fp32 read range,
//         so the overlay is self-ordered per warp -- no cross-warp race)
//   dST = PT * (dPT - Delta[q]);     STTM bf16 over dPT cols {256-287,
//                                    320-351} + RF->SMEM (MN view for dQc)
//   dV += PT . dO             TS  M128 N128 K128  (A = PT TMEM)
//   dK += dST . Q             TS  M128 N128 K128  (A = dST TMEM)
//   dQc = dST^T . Kpair       SS  M128 N128 K128  (A = SMEM dST, MN-major)
//                             -> TMEM 256-383 (dPT slot; dK-before-dQc order)
//   dQaccum[q blk] += dQc     via PLAIN cp.reduce.async.bulk.add.f32 into a
//                             drain-native per-q-BLOCK layout; a q-half with
//                             no membership in either kv half is exactly
//                             zero and skipped end to end
// dP(j+1) waits for the dQc(j) whole-tile t2r (shared TMEM cols, FA4 gate);
// the t2r is two x64 LDTMs (col-split lo/hi in the same regs) with no gmem
// work before DQC_FREE; the bulk pushes trail after FREE. dO is single-stage
// (dV(j) is its last reader); K/V bytes piggyback on the j==0 Q/dO
// expect_tx (no dedicated K/V barrier). Fully-masked compute quadrants skip
// their LDTMs and math (zero STTMs only). Item end: dK *= sm_scale, dV
// unscaled, packed bf16, TMA-stored per pair; the dK/dV epilogue bounces
// through the then-free dST SMEM.
//
// TMEM (512 cols, FA4 map): ST=PT 0-127 | dV 128-255 | dPT=dST=dQc 256-383 |
// dK 384-511. All tiles are M=128 (lane = row); no m64 folding.
//
// Warps (15): 0-7 compute (col-split: warp w = kv subpartition w&3, q half
// w>>2, 64 fp32 per lane per tile), 8-11 dQ reduce (warps 8,9 = q-half 0,
// 10,11 = q-half 1; per-half sync domains and bulk chains), 12 MMA, 13 load,
// 14 sched (CLC).
//
// dQaccum drain-native layout: per (bh, q_blk): 4 chunks of [64 q rows x
// 32 d cols] fp32, contiguous 8KB each, float4 slots xor-swizzled by row&7
// inside each 128B row (bank-conflict-free STS); the verify / postprocess
// unscrambles.
//
// Scaling follows the #1730 rule: K never pre-scaled; log2-domain scale in
// fp32 inside the exp2; sm_scale on dK in the drain, on dQ in postprocess.




namespace vsa_bwd_blk64 {

constexpr int BLOCK = 64;
constexpr int KV_PAIR = 2 * BLOCK;                 // 128 kv rows per item
constexpr int Q_PAIR  = 2 * BLOCK;                 // 128 q rows per step
constexpr int HEAD_DIM = 128;
constexpr int SUB = 64;
constexpr int Q_SUBTILE_BYTES  = Q_PAIR * SUB * 2;      // 16 KB
constexpr int Q_TILE_BYTES     = 2 * Q_SUBTILE_BYTES;   // 32 KB
constexpr int KV_SUBTILE_BYTES = KV_PAIR * SUB * 2;     // 16 KB
constexpr int KV_TILE_BYTES    = 2 * KV_SUBTILE_BYTES;  // 32 KB
constexpr int DST_SUBTILE_BYTES = KV_PAIR * SUB * 2;    // 16 KB (128 kv x 64 q bf16)
constexpr int DST_TILE_BYTES   = 2 * DST_SUBTILE_BYTES; // 32 KB
constexpr int DQC_CHUNK_COLS   = 32;
constexpr int DQC_CHUNK_BYTES  = BLOCK * DQC_CHUNK_COLS * 4;   // 8 KB (per q-half)
#ifndef VSA_BWD_SPIN_COMPUTE
#define VSA_BWD_SPIN_COMPUTE 0
#endif
#ifndef VSA_BWD_SKIP_REDUCE     // probe: skip only the dQ gmem bulk push
#define VSA_BWD_SKIP_REDUCE 0
#endif
#ifndef VSA_BWD_SKIP_DQ_T2R     // probe: skip the whole per-step dQ drain
#define VSA_BWD_SKIP_DQ_T2R 0
#endif
#ifndef VSA_BWD_T2R_ONLY        // probe: keep the t2r + FREE, skip r2s/pushes
#define VSA_BWD_T2R_ONLY 0
#endif
#ifndef VSA_BWD_PROBE_HALF_DRAIN // probe (wrong dq, perf-only): drain chunks 0,1 only
#define VSA_BWD_PROBE_HALF_DRAIN 0
#endif
#ifndef VSA_BWD_FREE_FIRST      // t2r both halves (2x64 regs) before any SMEM work
#define VSA_BWD_FREE_FIRST 0
#endif
#ifndef VSA_BWD_PROBE_NO_SYNC   // probe (RACY, perf-only): drop drain bar_syncs
#define VSA_BWD_PROBE_NO_SYNC 0
#endif
#ifndef VSA_BWD_PROBE_NO_R2S    // probe (garbage dq, perf-only): drop drain STS
#define VSA_BWD_PROBE_NO_R2S 0
#endif
#ifndef VSA_BWD_CSKIP           // compute warps skip LDTM+math in masked quadrants
#define VSA_BWD_CSKIP 1
#endif
#ifndef VSA_BWD_HSKIP           // reduce warps skip dead q-halves entirely
#define VSA_BWD_HSKIP 1
#endif
#ifndef VSA_BWD_CLC
#define VSA_BWD_CLC 0
#endif
constexpr bool USE_CLC = VSA_BWD_CLC;
constexpr int CLC_STAGES = 2;
constexpr int N_WARPS = 15;                        // 0-7 compute, 8-11 reduce, 12 MMA, 13 load, 14 sched
constexpr int W_REDUCE0 = 8, W_MMA = 12, W_LOAD = 13, W_SCHED = 14;

constexpr uint32_t UQ_MASK = (1u << 28) - 1u;      // quadrant flags in bits 28-31

union SmemDescPair {
  uint64_t u64;
  uint2 w;
};

__device__ __forceinline__ void desc_add_lo(SmemDescPair& d, uint32_t inc) {
  asm volatile(
      "{\n\t"
      ".reg .b32 lo, hi;\n\t"
      "mov.b64 {lo, hi}, %0;\n\t"
      "add.u32 lo, lo, %1;\n\t"
      "mov.b64 %0, {lo, hi};\n\t"
      "}"
      : "+l"(d.u64)
      : "r"(inc));
}

__device__ __forceinline__
void bulk_reduce_add_f32(const float* gmem_dst, uint32_t smem_src, int bytes) {
  asm volatile(
    "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
    " [%0], [%1], %2;\n"
    :: "l"(gmem_dst), "r"(smem_src), "r"(bytes)
    : "memory");
}

// SMEM (~226 KB): K 32K | V 32K | Q[2] 64K | dO 32K | dST 32K |
// dQc chunks [2] 32K | M/Delta rings 2K | barriers. The dK/dV epilogue
// bounces through sDST (free at item end); no dedicated buffer.
constexpr int SMEM_K    = 0;
constexpr int SMEM_V    = SMEM_K + KV_TILE_BYTES;
constexpr int SMEM_Q    = SMEM_V + KV_TILE_BYTES;
constexpr int SMEM_DO   = SMEM_Q + 2 * Q_TILE_BYTES;
constexpr int SMEM_DST  = SMEM_DO + Q_TILE_BYTES;
constexpr int SMEM_DQC  = SMEM_DST + DST_TILE_BYTES;
constexpr int SMEM_MD   = SMEM_DQC + 4 * DQC_CHUNK_BYTES;
constexpr int SMEM_BARS = SMEM_MD + 2 * 2 * Q_PAIR * 4;
constexpr int NUM_BARS  = 14 + 2 * CLC_STAGES;     // + clc_full[2], clc_empty[2]
constexpr int SMEM_TOTAL = SMEM_BARS + NUM_BARS * 8 + 16 + CLC_STAGES * 16 + 16;

enum {
  BAR_Q_FULL0 = 0, BAR_Q_FULL1,   // load -> mma; j==0 stage also carries K tx
  BAR_Q_EMPTY0, BAR_Q_EMPTY1,     // commit after dK(j) (last Q reader)
  BAR_DO_FULL,                    // load -> mma; j==0 also carries V tx
  BAR_DO_EMPTY,                   // commit after dV(j) (last dO reader)
  BAR_ST_READY,     // commit after the ST atoms
  BAR_DPT_READY,    // commit after the dPT atoms
  BAR_PT_STTMD,     // compute: PT STTM'd (gates the dV issue)
  BAR_DST_READY,    // compute: dST STTM'd + SMEM'd (gates dK/dQc issue)
  BAR_DQC_FULL,     // commit after the dQc atoms
  BAR_DQC_FREE,     // reduce: whole-tile t2r done (gates dP(j+1), shared cols)
  BAR_DKV_FULL,     // commit at item end
  BAR_DKV_DONE,
};
enum { BAR_CLC_FULL0 = BAR_DKV_DONE + 1, BAR_CLC_EMPTY0 = BAR_CLC_FULL0 + CLC_STAGES };

// TMEM (512 cols, FA4 map): ST=PT 0-127 | dV 128-255 | dPT=dST=dQc 256-383 |
// dK 384-511. PT bf16 packs at cols {0-31, 64-95} (q-half h at h*64, inside
// its OWN fp32 read range -> the overlay is self-ordered per warp); dST
// bf16 likewise at 256 + {0-31, 64-95}; dQc fp32 reuses 256-383 after dK
// read dST (in-order tcgen05 pipe).
constexpr uint32_t T_ST = 0, T_DV = 128, T_DPT = 256, T_DK = 384;
constexpr uint32_t T_PT_BF16  = T_ST;
constexpr uint32_t T_DST_BF16 = T_DPT;
constexpr uint32_t T_DQC      = T_DPT;

extern __shared__ __align__(1024) uint8_t bwd_smem[];

// Per-warp work-item source: CLC work stealing or grid stride.
struct BwdItemSource {
  uint64_t* clc_full;
  uint64_t* clc_empty;
  uint32_t* clc_response;
  int total;
  int clc_stage = 0;
  uint32_t clc_phase = 0;
  __device__ __forceinline__ int next(int item) {
    if constexpr (USE_CLC) {
      ClcTileInfo n = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, 1, true>(
          clc_full, clc_empty, clc_response, clc_stage, clc_phase, elect_one_sync());
      clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
      return n.valid ? (int)n.n_tile : -1;
    } else {
      const int nx = item + (int)gridDim.x;
      return nx < total ? nx : -1;
    }
  }
};

__global__ void __launch_bounds__(N_WARPS * 32, 1)
vsa_bwd_main_kernel(const __grid_constant__ CUtensorMap tmap_q,
                    const __grid_constant__ CUtensorMap tmap_k,
                    const __grid_constant__ CUtensorMap tmap_v,
                    const __grid_constant__ CUtensorMap tmap_do,
                    const __grid_constant__ CUtensorMap tmap_dk,
                    const __grid_constant__ CUtensorMap tmap_dv,
                    float* __restrict__ dqaccum,
                    const __grid_constant__ CUtensorMap tmap_m,
                    const __grid_constant__ CUtensorMap tmap_delta,
                    const int* __restrict__ pair_offset,
                    const unsigned* __restrict__ pair_union,
                    int num_heads, int seqlen, int num_blocks,
                    float scale_log2, float sm_scale) {
  uint8_t* base = bwd_smem;
  __nv_bfloat16* sK  = reinterpret_cast<__nv_bfloat16*>(base + SMEM_K);
  __nv_bfloat16* sV  = reinterpret_cast<__nv_bfloat16*>(base + SMEM_V);
  __nv_bfloat16* sQ[2]  = { reinterpret_cast<__nv_bfloat16*>(base + SMEM_Q),
                            reinterpret_cast<__nv_bfloat16*>(base + SMEM_Q + Q_TILE_BYTES) };
  __nv_bfloat16* sDO  = reinterpret_cast<__nv_bfloat16*>(base + SMEM_DO);
  __nv_bfloat16* sDST = reinterpret_cast<__nv_bfloat16*>(base + SMEM_DST);
  float* sMD[2] = { reinterpret_cast<float*>(base + SMEM_MD),
                    reinterpret_cast<float*>(base + SMEM_MD + 2 * Q_PAIR * 4) };
  uint64_t* bars = reinterpret_cast<uint64_t*>(base + SMEM_BARS);
  uint64_t* clc_full  = &bars[BAR_CLC_FULL0];
  uint64_t* clc_empty = &bars[BAR_CLC_EMPTY0];
  uint32_t* clc_response = reinterpret_cast<uint32_t*>(
      (reinterpret_cast<uintptr_t>(bars + NUM_BARS) + 15u) & ~uintptr_t(15u));
  uint32_t* tmem_slot = clc_response + CLC_STAGES * 4;

  const int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;

  if (warp_id == 0) {
    tcgen05_alloc<1>(smem_ptr_u32(tmem_slot), 512);
    tcgen05_relinquish_alloc_permit<1>();
  }
  if (tid == 0) {
    mbarrier_init(smem_ptr_u32(&bars[BAR_Q_FULL0]),    1);
    mbarrier_init(smem_ptr_u32(&bars[BAR_Q_FULL1]),    1);
    mbarrier_init(smem_ptr_u32(&bars[BAR_Q_EMPTY0]),   1);
    mbarrier_init(smem_ptr_u32(&bars[BAR_Q_EMPTY1]),   1);
    mbarrier_init(smem_ptr_u32(&bars[BAR_DO_FULL]),    1);
    mbarrier_init(smem_ptr_u32(&bars[BAR_DO_EMPTY]),   1);
    mbarrier_init(smem_ptr_u32(&bars[BAR_ST_READY]),   1);
    mbarrier_init(smem_ptr_u32(&bars[BAR_DPT_READY]),  1);
    mbarrier_init(smem_ptr_u32(&bars[BAR_PT_STTMD]),   256);
    mbarrier_init(smem_ptr_u32(&bars[BAR_DST_READY]),  256);
    mbarrier_init(smem_ptr_u32(&bars[BAR_DQC_FULL]),   1);
    mbarrier_init(smem_ptr_u32(&bars[BAR_DQC_FREE]),   4);
    mbarrier_init(smem_ptr_u32(&bars[BAR_DKV_FULL]),   1);
    mbarrier_init(smem_ptr_u32(&bars[BAR_DKV_DONE]),   128);
    if constexpr (USE_CLC) {
      #pragma unroll
      for (int st = 0; st < CLC_STAGES; ++st) {
        mbarrier_init(smem_ptr_u32(&clc_full[st]), 1);
        mbarrier_init(smem_ptr_u32(&clc_empty[st]), N_WARPS);
      }
      #pragma unroll
      for (int i = 0; i < CLC_STAGES * 4; ++i) clc_response[i] = 0;
    }
  }
  fence_mbarrier_init_release_cluster();
  __syncthreads();
  const uint32_t tmem_base = *tmem_slot;
  WpCtx wpc = wp_ctx_init();

  const int nb = num_blocks;
  const int total = num_heads * (nb / 2);

  if (warp_id == W_LOAD) {
    EmptyPhaseTracker<2> q_empty_ph;
    EmptyPhaseTracker<1> do_empty_ph;
    EmptyPhaseTracker<1> kv_free_ph;
    BwdItemSource src{clc_full, clc_empty, clc_response, total};
    for (int item = blockIdx.x; item >= 0; item = src.next(item)) {
      const int bh   = item / (nb / 2);
      const int pair = item % (nb / 2);
      const int beg = pair_offset[item], cnt = pair_offset[item + 1] - beg;
      const int steps = cnt > 0 ? cnt : 1;

      wp_marker(wpc, WP_ITEM, item);
      wp_begin(wpc, WP_USER0);
      mbarrier_wait_parity_suspend(smem_ptr_u32(&bars[BAR_DKV_DONE]), kv_free_ph.get_phase());
      kv_free_ph.advance();
      wp_end(wpc, WP_USER0);

      for (int j = 0; j < steps; ++j) {
        wp_marker(wpc, WP_ITER, j);
        const int q_pair_id = (j < cnt) ? (int)(pair_union[beg + j] & UQ_MASK) : 0;
        const int sd = q_empty_ph.get_stage() & 1;
        wp_begin(wpc, WP_LOAD_WAIT);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&bars[BAR_Q_EMPTY0 + sd]),
                                     q_empty_ph.get_phase());
        q_empty_ph.advance();
        wp_end(wpc, WP_LOAD_WAIT);
        wp_begin(wpc, WP_LOAD_ISSUE_Q);
        if (elect_one_sync()) {
          mbarrier_arrive_expect_tx(smem_ptr_u32(&bars[BAR_Q_FULL0 + sd]),
                                    Q_TILE_BYTES + 2 * Q_PAIR * 4 +
                                    (j == 0 ? KV_TILE_BYTES : 0));
          #pragma unroll
          for (int s = 0; s < 2; ++s)
            tma_load_3d(smem_ptr_u32(reinterpret_cast<uint8_t*>(sQ[sd]) + s * Q_SUBTILE_BYTES),
                        &tmap_q, smem_ptr_u32(&bars[BAR_Q_FULL0 + sd]),
                        0, q_pair_id * Q_PAIR, bh * 2 + s);
          tma_load_2d(smem_ptr_u32(sMD[sd]), &tmap_m,
                      smem_ptr_u32(&bars[BAR_Q_FULL0 + sd]), q_pair_id * Q_PAIR, bh);
          tma_load_2d(smem_ptr_u32(sMD[sd] + Q_PAIR), &tmap_delta,
                      smem_ptr_u32(&bars[BAR_Q_FULL0 + sd]), q_pair_id * Q_PAIR, bh);
          if (j == 0) {
            #pragma unroll
            for (int s = 0; s < 2; ++s)
              tma_load_3d(smem_ptr_u32(reinterpret_cast<uint8_t*>(sK) + s * KV_SUBTILE_BYTES),
                          &tmap_k, smem_ptr_u32(&bars[BAR_Q_FULL0 + sd]),
                          0, pair * KV_PAIR, bh * 2 + s);
          }
        }
        wp_end(wpc, WP_LOAD_ISSUE_Q);
        wp_begin(wpc, WP_LOAD_WAIT_THROTTLE);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&bars[BAR_DO_EMPTY]),
                                     do_empty_ph.get_phase());
        do_empty_ph.advance();
        wp_end(wpc, WP_LOAD_WAIT_THROTTLE);
        wp_begin(wpc, WP_LOAD_ISSUE_V);
        if (elect_one_sync()) {
          mbarrier_arrive_expect_tx(smem_ptr_u32(&bars[BAR_DO_FULL]),
                                    Q_TILE_BYTES + (j == 0 ? KV_TILE_BYTES : 0));
          #pragma unroll
          for (int s = 0; s < 2; ++s)
            tma_load_3d(smem_ptr_u32(reinterpret_cast<uint8_t*>(sDO) + s * Q_SUBTILE_BYTES),
                        &tmap_do, smem_ptr_u32(&bars[BAR_DO_FULL]),
                        0, q_pair_id * Q_PAIR, bh * 2 + s);
          if (j == 0) {
            #pragma unroll
            for (int s = 0; s < 2; ++s)
              tma_load_3d(smem_ptr_u32(reinterpret_cast<uint8_t*>(sV) + s * KV_SUBTILE_BYTES),
                          &tmap_v, smem_ptr_u32(&bars[BAR_DO_FULL]),
                          0, pair * KV_PAIR, bh * 2 + s);
          }
        }
        wp_end(wpc, WP_LOAD_ISSUE_V);
      }
    }
  }
  else if (warp_id == W_MMA) {
    const uint32_t lead = elect_one_sync() ? 1u : 0u;
    constexpr uint32_t DESC_SBO = 1024, DESC_LBO = 16;
    constexpr uint32_t Q_LBO_MN   = (uint32_t)Q_SUBTILE_BYTES;
    constexpr uint32_t KV_LBO_MN  = (uint32_t)KV_SUBTILE_BYTES;
    constexpr uint32_t DST_LBO_MN = (uint32_t)DST_SUBTILE_BYTES;
    constexpr uint32_t K16_MN = (uint32_t)((16 * SUB * 2) >> 4);
    constexpr uint64_t Q_SUB_DELTA  = Q_SUBTILE_BYTES >> 4;
    constexpr uint64_t KV_SUB_DELTA = KV_SUBTILE_BYTES >> 4;
    constexpr uint64_t Q_SLOT_DELTA = Q_TILE_BYTES >> 4;

    const uint32_t idesc_st  = make_idesc_bf16_f32(KV_PAIR, Q_PAIR, false, false);
    const uint32_t idesc_acc = make_idesc_bf16_f32(KV_PAIR, HEAD_DIM, false, true);
    const uint32_t idesc_dqc = make_idesc_bf16_f32(Q_PAIR, HEAD_DIM, true, true);

    const uint64_t desc_k  = build_smem_desc_blackwell(smem_ptr_u32(sK),  DESC_SBO, DESC_LBO, SmemSwizzleBlackwell::B128);
    const uint64_t desc_v  = build_smem_desc_blackwell(smem_ptr_u32(sV),  DESC_SBO, DESC_LBO, SmemSwizzleBlackwell::B128);
    const uint64_t desc_q0 = build_smem_desc_blackwell(smem_ptr_u32(sQ[0]), DESC_SBO, DESC_LBO, SmemSwizzleBlackwell::B128);
    const uint64_t desc_do = build_smem_desc_blackwell(smem_ptr_u32(sDO), DESC_SBO, DESC_LBO, SmemSwizzleBlackwell::B128);
    const uint64_t desc_q0_mn = build_smem_desc_blackwell(smem_ptr_u32(sQ[0]), DESC_SBO, Q_LBO_MN, SmemSwizzleBlackwell::B128);
    const uint64_t desc_do_mn = build_smem_desc_blackwell(smem_ptr_u32(sDO), DESC_SBO, Q_LBO_MN, SmemSwizzleBlackwell::B128);
    const uint64_t desc_k_mn   = build_smem_desc_blackwell(smem_ptr_u32(sK),  DESC_SBO, KV_LBO_MN, SmemSwizzleBlackwell::B128);
    const uint64_t desc_dst_mn = build_smem_desc_blackwell(smem_ptr_u32(sDST), DESC_SBO, DST_LBO_MN, SmemSwizzleBlackwell::B128);

    PhaseTracker<2> q_ph;           // Q_FULL waits
    PhaseTracker<1> do_ph;          // DO_FULL waits
    EmptyPhaseTracker<1> dqcf_ph, dkvd_ph;
    PhaseTracker<1> ptst_ph, dstr_ph;

    BwdItemSource src{clc_full, clc_empty, clc_response, total};
    for (int item = blockIdx.x; item >= 0; item = src.next(item)) {
      const int beg = pair_offset[item], cnt = pair_offset[item + 1] - beg;
      const int steps = cnt > 0 ? cnt : 1;
      (void)beg;

      wp_marker(wpc, WP_ITEM, item);
      wp_begin(wpc, WP_MMA_WAIT_FULL);
      mbarrier_wait_parity(smem_ptr_u32(&bars[BAR_DKV_DONE]), dkvd_ph.get_phase());
      dkvd_ph.advance();
      wp_end(wpc, WP_MMA_WAIT_FULL);

      auto issue_st = [&](int sd) {
        const uint64_t dq = desc_q0 + (uint64_t)sd * Q_SLOT_DELTA;
        #pragma unroll
        for (int s = 0; s < 2; ++s)
          #pragma unroll
          for (int ki = 0; ki < 4; ++ki)
            tcgen05_mma_f16_ss_lead(lead, tmem_base + T_ST,
                                    desc_k + s * KV_SUB_DELTA + 2 * ki,
                                    dq + s * Q_SUB_DELTA + 2 * ki,
                                    idesc_st, (s | ki) != 0);
        tcgen05_commit1_lead(lead, smem_ptr_u32(&bars[BAR_ST_READY]));
      };
      auto issue_dpt = [&]() {
        #pragma unroll
        for (int s = 0; s < 2; ++s)
          #pragma unroll
          for (int ki = 0; ki < 4; ++ki)
            tcgen05_mma_f16_ss_lead(lead, tmem_base + T_DPT,
                                    desc_v + s * KV_SUB_DELTA + 2 * ki,
                                    desc_do + s * Q_SUB_DELTA + 2 * ki,
                                    idesc_st, (s | ki) != 0);
        tcgen05_commit1_lead(lead, smem_ptr_u32(&bars[BAR_DPT_READY]));
      };
      auto issue_dv = [&](bool first) {
        SmemDescPair bdo; bdo.u64 = desc_do_mn;
        #pragma unroll
        for (int ki = 0; ki < 8; ++ki) {
          tcgen05_mma_f16_ts_1sm_lead(lead, tmem_base + T_DV,
                                      tmem_base + T_PT_BF16 + (uint32_t)(ki * 8 + (ki >= 4 ? 32 : 0)),
                                      bdo.u64, idesc_acc, !(first && ki == 0));
          desc_add_lo(bdo, K16_MN);
        }
        tcgen05_commit1_lead(lead, smem_ptr_u32(&bars[BAR_DO_EMPTY]));
      };
      auto issue_dk = [&](int sd, bool first) {
        SmemDescPair bq; bq.u64 = desc_q0_mn + (uint64_t)sd * Q_SLOT_DELTA;
        #pragma unroll
        for (int ki = 0; ki < 8; ++ki) {
          tcgen05_mma_f16_ts_1sm_lead(lead, tmem_base + T_DK,
                                      tmem_base + T_DST_BF16 + (uint32_t)(ki * 8 + (ki >= 4 ? 32 : 0)),
                                      bq.u64, idesc_acc, !(first && ki == 0));
          desc_add_lo(bq, K16_MN);
        }
        tcgen05_commit1_lead(lead, smem_ptr_u32(&bars[BAR_Q_EMPTY0 + sd]));
      };
      auto issue_dqc = [&]() {
        SmemDescPair adst; adst.u64 = desc_dst_mn;
        SmemDescPair bk;   bk.u64 = desc_k_mn;
        #pragma unroll
        for (int ki = 0; ki < 8; ++ki) {
          tcgen05_mma_f16_ss_lead(lead, tmem_base + T_DQC,
                                  adst.u64, bk.u64, idesc_dqc, ki != 0);
          desc_add_lo(adst, K16_MN);
          desc_add_lo(bk, K16_MN);
        }
        tcgen05_commit1_lead(lead, smem_ptr_u32(&bars[BAR_DQC_FULL]));
      };

      // Prologue: S(0), dP(0), dV(0). dP writes the shared 256-383 slot, so
      // it waits for the previous item's last dQc drain (DQC_FREE).
      {
        const int sd0 = q_ph.get_stage() & 1;
        wp_begin(wpc, WP_MMA_WAIT_FULL_Q);
        mbarrier_wait_parity(smem_ptr_u32(&bars[BAR_Q_FULL0 + sd0]), q_ph.get_phase());
        wp_end(wpc, WP_MMA_WAIT_FULL_Q);
        issue_st(sd0);
        wp_begin(wpc, WP_MMA_WAIT_ACC);
        mbarrier_wait_parity(smem_ptr_u32(&bars[BAR_DQC_FREE]), dqcf_ph.get_phase());
        dqcf_ph.advance();
        wp_end(wpc, WP_MMA_WAIT_ACC);
        wp_begin(wpc, WP_MMA_WAIT_FULL_V);
        mbarrier_wait_parity(smem_ptr_u32(&bars[BAR_DO_FULL]), do_ph.get_phase());
        do_ph.advance();
        wp_end(wpc, WP_MMA_WAIT_FULL_V);
        issue_dpt();
        wp_begin(wpc, WP_MMA_WAIT_FULL_K);
        mbarrier_wait_parity(smem_ptr_u32(&bars[BAR_PT_STTMD]), ptst_ph.get_phase());
        ptst_ph.advance();
        wp_end(wpc, WP_MMA_WAIT_FULL_K);
        issue_dv(true);
      }
      for (int j = 0; j < steps; ++j) {
        wp_marker(wpc, WP_ITER, j);
        const int sd = q_ph.get_stage() & 1;
        q_ph.advance();
        const int sn = q_ph.get_stage() & 1;

        if (j + 1 < steps) {
          wp_begin(wpc, WP_MMA_WAIT_FULL_Q);
          mbarrier_wait_parity(smem_ptr_u32(&bars[BAR_Q_FULL0 + sn]), q_ph.get_phase());
          wp_end(wpc, WP_MMA_WAIT_FULL_Q);
          issue_st(sn);
        }
        wp_begin(wpc, WP_MMA_WAIT_P);
        mbarrier_wait_parity(smem_ptr_u32(&bars[BAR_DST_READY]), dstr_ph.get_phase());
        dstr_ph.advance();
        wp_end(wpc, WP_MMA_WAIT_P);
        wp_begin(wpc, WP_MMA_ISSUE);
        issue_dk(sd, j == 0);
        issue_dqc();
        wp_end(wpc, WP_MMA_ISSUE);
        if (j + 1 < steps) {
          wp_begin(wpc, WP_MMA_WAIT_ACC);
          mbarrier_wait_parity(smem_ptr_u32(&bars[BAR_DQC_FREE]), dqcf_ph.get_phase());
          dqcf_ph.advance();
          wp_end(wpc, WP_MMA_WAIT_ACC);
          wp_begin(wpc, WP_MMA_WAIT_FULL_V);
          mbarrier_wait_parity(smem_ptr_u32(&bars[BAR_DO_FULL]), do_ph.get_phase());
          do_ph.advance();
          wp_end(wpc, WP_MMA_WAIT_FULL_V);
          issue_dpt();
          wp_begin(wpc, WP_MMA_WAIT_FULL_K);
          mbarrier_wait_parity(smem_ptr_u32(&bars[BAR_PT_STTMD]), ptst_ph.get_phase());
          ptst_ph.advance();
          wp_end(wpc, WP_MMA_WAIT_FULL_K);
          issue_dv(false);
        }
      }
      tcgen05_commit1_lead(lead, smem_ptr_u32(&bars[BAR_DKV_FULL]));
    }
  }
  else if (warp_id < W_REDUCE0) {
    // Compute warps (0-7), col-split over the 128x128 tile: warp w covers kv
    // subpartition w&3 (lanes = kv rows) and q half w>>2 (64 of 128 cols).
    const int subp = warp_id & 3;
    const int q_half = warp_id >> 2;
    const uint32_t lane_base = (uint32_t)((subp * 32) << 16);
    const int row = subp * 32 + lane;
    const int kv_half = subp >> 1;
    const uint32_t quad_bit = 1u << (28 + 2 * kv_half + q_half);
    const uint32_t half_mask = (1u << (28 + q_half)) | (1u << (30 + q_half));
    const uint32_t col_off = (uint32_t)(q_half * SUB);
    __nv_bfloat16* sDSTsub = sDST + (size_t)q_half * KV_PAIR * SUB;

    PhaseTracker<1> str_ph, dptr_ph;
    int md_slot = 0;

    BwdItemSource src{clc_full, clc_empty, clc_response, total};
    for (int item = blockIdx.x; item >= 0; item = src.next(item)) {
      const int beg = pair_offset[item], cnt = pair_offset[item + 1] - beg;
      const int steps = cnt > 0 ? cnt : 1;

      for (int j = 0; j < steps; ++j) {
        wp_marker(wpc, WP_ITER, j);
        const unsigned entry = (j < cnt) ? pair_union[beg + j] : 0u;
        const bool quad_on = (entry & quad_bit) != 0;
        const float* mg = sMD[md_slot] + col_off;
        const float* dg = sMD[md_slot] + Q_PAIR + col_off;
        md_slot ^= 1;

        wp_begin(wpc, WP_SM_WAIT_S);
        if (VSA_BWD_SPIN_COMPUTE)
          mbarrier_wait_parity(smem_ptr_u32(&bars[BAR_ST_READY]), str_ph.get_phase());
        else
          mbarrier_wait_parity_suspend(smem_ptr_u32(&bars[BAR_ST_READY]), str_ph.get_phase());
        str_ph.advance();
        wp_end(wpc, WP_SM_WAIT_S);
        wp_begin(wpc, WP_SM_SOFTMAX);

        // Fully-masked quadrant: skip both LDTMs and the math; only the zero
        // STTMs (dV/dK read the full PT/dST tiles) and barriers remain.
        // The bf16 STTM overlays fp32 columns the OTHER col-half warp reads,
        // so a named barrier must separate ALL the LDTMs from the first STTM
        // (FA4 does the same before its first P STTM).
        uint32_t st_regs[64];
        uint32_t pt_pack[32];
        float* pt = reinterpret_cast<float*>(st_regs);   // PT overwrites ST in place
        if (quad_on || !VSA_BWD_CSKIP) {
          tcgen05_ld_32x32b_x64(tmem_base + T_ST + col_off + lane_base, st_regs);
          tcgen05_fence_before_thread_sync();
        }
        if (quad_on || !VSA_BWD_CSKIP) {
          // Packed f32x2 affine (FA4 flash_bwd_sm100.py:3187-3258 form):
          // z2 = st2 * scale - m2 in one ffma2; exp2 stays on the HW SFU.
          const float2 scale2 = f32x2_splat(scale_log2);
          #pragma unroll
          for (int c0 = 0; c0 < 64; c0 += 4) {
            const float4 m4 = *reinterpret_cast<const float4*>(mg + c0);
            const float mv[4] = { -m4.x, -m4.y, -m4.z, -m4.w };
            #pragma unroll
            for (int c = c0; c < c0 + 4; c += 2) {
              const float2 z2 = ffma2(make_float2(pt[c], pt[c + 1]), scale2,
                                      make_float2(mv[c - c0], mv[c - c0 + 1]));
              const float p0 = quad_on ? ex2_approx_f32(z2.x) : 0.f;
              const float p1 = quad_on ? ex2_approx_f32(z2.y) : 0.f;
              pt[c] = p0; pt[c + 1] = p1;
              pt_pack[c / 2] = cvt_f32x2_to_bf16x2(p0, p1);
            }
          }
        } else {
          #pragma unroll
          for (int c = 0; c < 32; ++c) pt_pack[c] = 0u;
        }
        tcgen05_st_32x32b_x32(tmem_base + T_PT_BF16 + (uint32_t)(q_half * 64) + lane_base, pt_pack);
        tcgen05_wait_st();
        tcgen05_fence_before_thread_sync();
        mbarrier_arrive(smem_ptr_u32(&bars[BAR_PT_STTMD]));
        wp_end(wpc, WP_SM_SOFTMAX);

        wp_begin(wpc, WP_CORR_WAIT);
        if (VSA_BWD_SPIN_COMPUTE)
          mbarrier_wait_parity(smem_ptr_u32(&bars[BAR_DPT_READY]), dptr_ph.get_phase());
        else
          mbarrier_wait_parity_suspend(smem_ptr_u32(&bars[BAR_DPT_READY]), dptr_ph.get_phase());
        dptr_ph.advance();
        wp_end(wpc, WP_CORR_WAIT);
        wp_begin(wpc, WP_SM_STORE_P);
        if (quad_on || !VSA_BWD_CSKIP) {
          // dPT consumed in two x32 chunks: peak live regs = pt[64] +
          // dpt_chunk[32] (~96) instead of pt[64] + dpt[64] (~128+, spills
          // at the 480-thread cap). Packed dST lands in the dead low half
          // of pt (slot c/2 was consumed before iteration c writes it).
          #pragma unroll
          for (int half32 = 0; half32 < 2; ++half32) {
            uint32_t dpt_regs[32];
            tcgen05_ld_32x32b_x32(tmem_base + T_DPT + col_off
                                  + (uint32_t)(half32 * 32) + lane_base, dpt_regs);
            tcgen05_fence_before_thread_sync();
            const float* dpt = reinterpret_cast<const float*>(dpt_regs);
            // Packed f32x2 dS: s2 = pt2 * (dpt2 - delta2) as fadd2 + fmul2.
            #pragma unroll
            for (int c0 = 0; c0 < 32; c0 += 4) {
              const float4 d4 = *reinterpret_cast<const float4*>(dg + half32 * 32 + c0);
              const float dv4[4] = { -d4.x, -d4.y, -d4.z, -d4.w };
              #pragma unroll
              for (int c = c0; c < c0 + 4; c += 2) {
                const int cc = half32 * 32 + c;
                const float2 s2 = fmul2(make_float2(pt[cc], pt[cc + 1]),
                                        fadd2(make_float2(dpt[c], dpt[c + 1]),
                                              make_float2(dv4[c - c0], dv4[c - c0 + 1])));
                st_regs[cc / 2] = cvt_f32x2_to_bf16x2(s2.x, s2.y);
              }
            }
          }
        } else {
          #pragma unroll
          for (int c = 0; c < 32; ++c) st_regs[c] = 0u;
        }
        tcgen05_st_32x32b_x32(tmem_base + T_DST_BF16 + (uint32_t)(q_half * 64) + lane_base,
                              reinterpret_cast<const uint32_t(&)[32]>(st_regs));
        // sDST feeds only dQc; write it if this q-half is live in EITHER kv
        // half (a live half needs true zeros from masked quadrants). A fully
        // dead q-half's dQc rows are never pushed, so stale sDST is fine.
        if ((entry & half_mask) != 0) {
          const uint4* src4 = reinterpret_cast<const uint4*>(st_regs);
          #pragma unroll
          for (int v = 0; v < 8; ++v)
            *reinterpret_cast<uint4*>(&sDSTsub[row * SUB + ((v ^ (row & 7)) * 8)]) = src4[v];
        }
        tcgen05_wait_st();
        tcgen05_fence_before_thread_sync();
        fence_proxy_async_shared_cta();
        mbarrier_arrive(smem_ptr_u32(&bars[BAR_DST_READY]));
        wp_end(wpc, WP_SM_STORE_P);
      }

    }
  }
  else if (warp_id < W_MMA) {
    // dQ reduce warps (8-11): warp = TMEM subpartition (lane = q row, M=128
    // dQc tile). Warps 8,9 own q-half 0 (rows 0-63), warps 10,11 q-half 1;
    // a q-half with no membership in either kv half is exactly zero and is
    // SKIPPED end to end (no t2r, no STS, no push). FREE path holds no gmem:
    // ld x64 lo -> STS lo -> ld x64 hi (same regs) -> DQC_FREE; the bulk
    // pushes follow after FREE, overlapping the MMA. Sync domains and bulk
    // chains are per half (barriers 11/12, elect warps 8/10). dK/dV epilogue
    // at item end bounces through the then-free sDST (barrier 13).
    const int dwarp = warp_id - W_REDUCE0;
    const int subp = warp_id & 3;
    const int half = dwarp >> 1;
    const uint32_t lane_base = (uint32_t)((subp * 32) << 16);
    const int row = subp * 32 + lane;
    const int hrow = (dwarp & 1) * 32 + lane;
    const uint32_t half_mask = (1u << (28 + half)) | (1u << (30 + half));
    float* hbuf[2] = {
      reinterpret_cast<float*>(base + SMEM_DQC + half * 2 * DQC_CHUNK_BYTES),
      reinterpret_cast<float*>(base + SMEM_DQC + half * 2 * DQC_CHUNK_BYTES + DQC_CHUNK_BYTES) };
    __nv_bfloat16* sDKV = sDST;

    PhaseTracker<1> dqcp_ph, dkvf_ph;

    BwdItemSource src{clc_full, clc_empty, clc_response, total};
    for (int item = blockIdx.x; item >= 0; item = src.next(item)) {
      const int bh   = item / (nb / 2);
      const int pair = item % (nb / 2);
      const int beg = pair_offset[item], cnt = pair_offset[item + 1] - beg;
      const int steps = cnt > 0 ? cnt : 1;

      for (int j = 0; j < steps; ++j) {
        const unsigned entry = (j < cnt) ? pair_union[beg + j] : 0u;
        const bool active = (entry & half_mask) != 0;
        const int q_blk = 2 * (int)(entry & UQ_MASK) + half;
        const float* gq = dqaccum + (size_t)bh * seqlen * HEAD_DIM
                        + (size_t)q_blk * BLOCK * HEAD_DIM;

        wp_marker(wpc, WP_ITER, j);
        // Spin (not suspend): this wait sits in the per-step critical path --
        // DQC_FULL -> t2r -> DQC_FREE gates dP(j+1) under the TMEM aliasing.
        wp_begin(wpc, WP_EPI_WAIT_ACC);
        mbarrier_wait_parity(smem_ptr_u32(&bars[BAR_DQC_FULL]), dqcp_ph.get_phase());
        dqcp_ph.advance();
        wp_end(wpc, WP_EPI_WAIT_ACC);

        if (VSA_BWD_SKIP_DQ_T2R || (VSA_BWD_HSKIP && !active)) {
          if (elect_one_sync())
            mbarrier_arrive(smem_ptr_u32(&bars[BAR_DQC_FREE]));
          continue;
        }

        uint32_t qc_regs[64];
        const bool lead = (dwarp & 1) == 0 && elect_one_sync();
        auto hsync = [&]() {
          if (VSA_BWD_PROBE_NO_SYNC) return;
          if (half == 0) bar_sync<11>(64); else bar_sync<12>(64);
        };
        auto r2s_one = [&](int c, const uint32_t* regs32) {
          if (VSA_BWD_PROBE_NO_R2S) return;
          // Bank-conflict-free: float4 slot v of row r stored at v ^ (r & 7)
          // (128B rows alias all lanes onto one bank otherwise). dQaccum is
          // drain-native, so the xor is part of its layout; the verify /
          // postprocess unscrambles it.
          const float4* qc4 = reinterpret_cast<const float4*>(regs32);
          float4* dst4 = reinterpret_cast<float4*>(hbuf[c & 1] + hrow * DQC_CHUNK_COLS);
          #pragma unroll
          for (int v = 0; v < DQC_CHUNK_COLS / 4; ++v) dst4[v ^ (hrow & 7)] = qc4[v];
        };
        auto push_one = [&](int c) {
          if (!VSA_BWD_SKIP_REDUCE)
            bulk_reduce_add_f32(gq + (size_t)c * BLOCK * DQC_CHUNK_COLS,
                                smem_ptr_u32(hbuf[c & 1]), DQC_CHUNK_BYTES);
          cp_async_bulk_commit_group();
        };
        // FA4's dQ drain contract: 2 rotating buffers, per-chunk pushes with
        // wait_group(1, read); each r2s overlaps the previous push's SMEM
        // read; the trailing pushes roll into the next step's guard (which
        // hides behind the lo t2r).
        wp_begin(wpc, WP_EPI_TMEM_LD);
#if VSA_BWD_FREE_FIRST
        uint32_t qc_hi[64];
        tcgen05_ld_32x32b_x64(tmem_base + T_DQC + lane_base, qc_regs);
        tcgen05_ld_32x32b_x64(tmem_base + T_DQC + 64 + lane_base, qc_hi);
        tcgen05_fence_before_thread_sync();
        if (elect_one_sync())
          mbarrier_arrive(smem_ptr_u32(&bars[BAR_DQC_FREE]));
        if (!VSA_BWD_T2R_ONLY) {
          if (lead) cp_async_bulk_wait_group_read<0>();   // prev step's tail
          hsync();
          r2s_one(0, qc_regs);
          r2s_one(1, qc_regs + DQC_CHUNK_COLS);
          #pragma unroll
          for (int c = 0; c < 64; ++c) qc_regs[c] = qc_hi[c];
        }
#else
        tcgen05_ld_32x32b_x64(tmem_base + T_DQC + lane_base, qc_regs);
        tcgen05_fence_before_thread_sync();
        if (!VSA_BWD_T2R_ONLY) {
          if (lead) cp_async_bulk_wait_group_read<0>();   // prev step's tail
          hsync();
          r2s_one(0, qc_regs);
          r2s_one(1, qc_regs + DQC_CHUNK_COLS);
        }
        tcgen05_ld_32x32b_x64(tmem_base + T_DQC + 64 + lane_base, qc_regs);
        tcgen05_fence_before_thread_sync();
        if (elect_one_sync())
          mbarrier_arrive(smem_ptr_u32(&bars[BAR_DQC_FREE]));
#endif
        wp_end(wpc, WP_EPI_TMEM_LD);
        wp_begin(wpc, WP_EPI_STORE);
        if (!VSA_BWD_T2R_ONLY) {
          fence_proxy_async_shared_cta();
          hsync();
          if (lead) { push_one(0); push_one(1); cp_async_bulk_wait_group_read<1>(); }
          if (!VSA_BWD_PROBE_HALF_DRAIN) {
            hsync();
            r2s_one(2, qc_regs);
            fence_proxy_async_shared_cta();
            hsync();
            if (lead) { push_one(2); cp_async_bulk_wait_group_read<1>(); }
            hsync();
            r2s_one(3, qc_regs + DQC_CHUNK_COLS);
            fence_proxy_async_shared_cta();
            hsync();
            if (lead) push_one(3);
          }
        }
      }

      wp_begin(wpc, WP_EPI_WAIT_STORE);
      mbarrier_wait_parity_suspend(smem_ptr_u32(&bars[BAR_DKV_FULL]), dkvf_ph.get_phase());
      dkvf_ph.advance();
      wp_end(wpc, WP_EPI_WAIT_STORE);
      wp_begin(wpc, WP_CORR_EPI);
      #pragma unroll
      for (int which = 0; which < 2; ++which) {
        const uint32_t t_addr = tmem_base + (which == 0 ? T_DK : T_DV);
        const float scale = (which == 0) ? sm_scale : 1.0f;
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
          #pragma unroll
          for (int c0 = 0; c0 < SUB; c0 += 32) {
            uint32_t acc_regs[32];
            tcgen05_ld_32x32b_x32(t_addr + lane_base + (uint32_t)(s * SUB + c0), acc_regs);
            tcgen05_fence_before_thread_sync();
            const float* acc = reinterpret_cast<const float*>(acc_regs);
            #pragma unroll
            for (int v = 0; v < 4; ++v) {
              uint4 packed;
              packed.x = cvt_f32x2_to_bf16x2(acc[v * 8 + 0] * scale, acc[v * 8 + 1] * scale);
              packed.y = cvt_f32x2_to_bf16x2(acc[v * 8 + 2] * scale, acc[v * 8 + 3] * scale);
              packed.z = cvt_f32x2_to_bf16x2(acc[v * 8 + 4] * scale, acc[v * 8 + 5] * scale);
              packed.w = cvt_f32x2_to_bf16x2(acc[v * 8 + 6] * scale, acc[v * 8 + 7] * scale);
              const int vv = c0 / 8 + v;
              *reinterpret_cast<uint4*>(&sDKV[row * SUB + ((vv ^ (row & 7)) * 8)]) = packed;
            }
          }
          fence_proxy_async_shared_cta();
          bar_sync<13>(128);
          if (dwarp == 1 && elect_one_sync()) {
            const CUtensorMap* map = (which == 0) ? &tmap_dk : &tmap_dv;
            tma_store_3d(map, 0, pair * KV_PAIR, bh * 2 + s, smem_ptr_u32(sDKV));
            cp_async_bulk_commit_group();
            cp_async_bulk_wait_group_read<0>();
          }
          bar_sync<13>(128);
        }
      }
      wp_end(wpc, WP_CORR_EPI);
      mbarrier_arrive(smem_ptr_u32(&bars[BAR_DKV_DONE]));
    }
  }

  else {
    // Scheduler warp (CLC skeleton): produces try_cancel results into the
    // clc ring; also runs its own consumer fetch to know when to stop.
    if constexpr (USE_CLC) {
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
        clc_fetch_next_tile_advance<CLC_STAGES>(cons_stage, cons_phase);
        if (!n.valid) break;
      }
      for (int st = 0; st < CLC_STAGES; ++st) {
        if (lane == 0)
          mbarrier_wait_parity_suspend(smem_ptr_u32(&clc_empty[prod_stage]), prod_phase);
        __syncwarp();
        advance_stage_phase<CLC_STAGES>(prod_stage, prod_phase);
      }
    }
  }

  wp_flush(wpc);
  __syncthreads();
  if (warp_id == 0) tcgen05_dealloc<1>(tmem_base, 512);
}

// Preprocess: Delta[bh, t] = sum_d O[t*H+h, d] * dO[t*H+h, d], one warp per row.
__global__ void vsa_bwd_preprocess_kernel(const __nv_bfloat16* __restrict__ o,
                                          const __nv_bfloat16* __restrict__ dout,
                                          float* __restrict__ delta_rows,
                                          int num_heads, int seqlen) {
  const int warps_per_block = blockDim.x >> 5;
  const long row = (long)blockIdx.x * warps_per_block + (threadIdx.x >> 5);
  const long total = (long)gridDim.x * warps_per_block;
  const int lane = threadIdx.x & 31;
  for (long r = row; r < (long)num_heads * seqlen; r += total) {
    const int bh = (int)(r / seqlen);
    const int t  = (int)(r % seqlen);
    const long base = ((long)t * num_heads + bh) * HEAD_DIM;
    float acc = 0.f;
    #pragma unroll
    for (int c = lane; c < HEAD_DIM; c += 32)
      acc += __bfloat162float(o[base + c]) * __bfloat162float(dout[base + c]);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) delta_rows[r] = acc;
  }
}

// ---------------------------------------------------------------------------
// Host launcher.
// ---------------------------------------------------------------------------

struct VsaBwdArgs {
  const __nv_bfloat16 *q, *k, *v, *dout;
  float* dqaccum;                 // [BH*S, 128] fp32, pre-zeroed; RAW sums
  __nv_bfloat16 *dk, *dv;         // [S, H, 128] bf16 (token-major, B folded)
  const float *m_rows, *delta_rows;   // [BH, S]
  const int* pair_offset;             // [BH*nb/2 + 1]
  const unsigned* pair_union;         // flagged union entries
  int num_heads, seqlen, num_blocks;
  float sm_scale;
};

inline cudaError_t vsa_bwd_encode_bf16_3d(CUtensorMap* map, const __nv_bfloat16* ptr,
                                          int H, long tq, int box_tokens) {
  uint64_t gd[3] = { (uint64_t)SUB, (uint64_t)tq, (uint64_t)H * (HEAD_DIM / SUB) };
  uint64_t gs[2] = { (uint64_t)H * HEAD_DIM * 2, (uint64_t)SUB * 2 };
  uint32_t bd[3] = { (uint32_t)SUB, (uint32_t)box_tokens, 1u };
  uint32_t es[3] = { 1u, 1u, 1u };
  if (cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3,
                             const_cast<__nv_bfloat16*>(ptr), gd, gs, bd, es,
                             CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                             CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
    return cudaErrorInvalidValue;
  return cudaSuccess;
}

inline cudaError_t launch_vsa_bwd_sm100a(const VsaBwdArgs& a, cudaStream_t stream) {
  const int H = a.num_heads, S = a.seqlen;
  const long tq = (long)S;

  CUtensorMap tq_, tk_, tv_, tdo_, tdk_, tdv_, tm_, tdelta_;
  if (vsa_bwd_encode_bf16_3d(&tq_, a.q, H, tq, Q_PAIR) != cudaSuccess) return cudaErrorInvalidValue;
  if (vsa_bwd_encode_bf16_3d(&tk_, a.k, H, tq, KV_PAIR) != cudaSuccess) return cudaErrorInvalidValue;
  if (vsa_bwd_encode_bf16_3d(&tv_, a.v, H, tq, KV_PAIR) != cudaSuccess) return cudaErrorInvalidValue;
  if (vsa_bwd_encode_bf16_3d(&tdo_, a.dout, H, tq, Q_PAIR) != cudaSuccess) return cudaErrorInvalidValue;
  if (vsa_bwd_encode_bf16_3d(&tdk_, a.dk, H, tq, KV_PAIR) != cudaSuccess) return cudaErrorInvalidValue;
  if (vsa_bwd_encode_bf16_3d(&tdv_, a.dv, H, tq, KV_PAIR) != cudaSuccess) return cudaErrorInvalidValue;
  {
    uint64_t gd[2] = { (uint64_t)S, (uint64_t)H };
    uint64_t gs[1] = { (uint64_t)S * 4 };
    uint32_t bd[2] = { (uint32_t)Q_PAIR, 1u };
    uint32_t es[2] = { 1u, 1u };
    if (cuTensorMapEncodeTiled(&tm_, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2,
                               const_cast<float*>(a.m_rows), gd, gs, bd, es,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                               CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
      return cudaErrorInvalidValue;
    if (cuTensorMapEncodeTiled(&tdelta_, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2,
                               const_cast<float*>(a.delta_rows), gd, gs, bd, es,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                               CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
      return cudaErrorInvalidValue;
  }

  static bool smem_set = false;
  if (!smem_set) {
    cudaError_t e = cudaFuncSetAttribute(vsa_bwd_main_kernel,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         SMEM_TOTAL);
    if (e != cudaSuccess) return e;
    smem_set = true;
  }

  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  const int total = H * (a.num_blocks / 2);
  const int grid = USE_CLC ? total : (total < sms ? total : sms);
  const float scale_log2 = a.sm_scale * 1.4426950408889634f;

  if constexpr (USE_CLC) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)grid, 1, 1);
    cfg.blockDim = dim3(N_WARPS * 32, 1, 1);
    cfg.dynamicSmemBytes = SMEM_TOTAL;
    cfg.stream = stream;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 1; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cfg.attrs = at; cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, vsa_bwd_main_kernel,
        tq_, tk_, tv_, tdo_, tdk_, tdv_, a.dqaccum, tm_, tdelta_,
        a.pair_offset, a.pair_union,
        H, S, a.num_blocks, scale_log2, a.sm_scale);
  }
  vsa_bwd_main_kernel<<<grid, N_WARPS * 32, SMEM_TOTAL, stream>>>(
      tq_, tk_, tv_, tdo_, tdk_, tdv_, a.dqaccum, tm_, tdelta_,
      a.pair_offset, a.pair_union,
      H, S, a.num_blocks, scale_log2, a.sm_scale);
  return cudaGetLastError();
}

inline cudaError_t launch_vsa_bwd_preprocess(const __nv_bfloat16* o, const __nv_bfloat16* dout,
                                             float* delta_rows, int num_heads, int seqlen,
                                             cudaStream_t stream) {
  const long rows = (long)num_heads * seqlen;
  const int warps_per_block = 8;
  const long blocks_needed = (rows + warps_per_block - 1) / warps_per_block;
  const int grid = (int)(blocks_needed < 65535 ? blocks_needed : 65535);
  vsa_bwd_preprocess_kernel<<<grid, warps_per_block * 32, 0, stream>>>(
      o, dout, delta_rows, num_heads, seqlen);
  return cudaGetLastError();
}

}  // namespace vsa_bwd_blk64



static int BLOCK = 64;

static const float LOG2E = 1.4426950408889634f;

// Minimal fp32 .npy v1.0 writer (npy_io.cuh only reads). C order, 64-byte-aligned header.
static void npy_save_f32(const std::string& path, const float* data,
                         std::initializer_list<long> shape) {
  std::string dims;
  long n = 1;
  size_t i = 0;
  for (long d : shape) { dims += std::to_string(d); n *= d; if (++i < shape.size()) dims += ", "; }
  if (shape.size() == 1) dims += ",";
  std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + dims + "), }";
  const size_t pad = (64 - (10 + header.size() + 1) % 64) % 64;
  header.append(pad, ' ');
  header += '\n';
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) { fprintf(stderr, "npy_save: cannot open %s\n", path.c_str()); exit(1); }
  const unsigned char magic[8] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
  fwrite(magic, 1, 8, f);
  const uint16_t header_len = (uint16_t)header.size();
  fwrite(&header_len, 2, 1, f);
  fwrite(header.data(), 1, header.size(), f);
  fwrite(data, sizeof(float), (size_t)n, f);
  fclose(f);
}

// Deterministic sorted k2q inversion: count per kv block + concatenated q-block lists,
// sorted ascending (built by ascending q-block walk). Kept for the M3 GPU kernel.
struct KvToQ {
  std::vector<int> count;      // per global kv block, size B*H*num_blocks
  std::vector<int> offset;     // prefix sum, size B*H*num_blocks + 1
  std::vector<int> q_blocks;   // concatenated sorted local q-block ids
};

static KvToQ invert_q2k(const int* q2k_idx, const int* q2k_num,
                        int B, int H, int num_blocks, int max_kv) {
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
      const int gkb = bh * num_blocks + q2k_idx[(size_t)global_mtile * max_kv + i];
      inv.q_blocks[cursor[gkb]++] = mtile;
    }
  }
  return inv;
}

// Quad-union metadata for the v5 kernel: one work item per ADJACENT KV-BLOCK
// PAIR walking Q-BLOCK-PAIR tiles; entry = q_pair id (bits 0-27) + quadrant
// membership flags: bit 28 = (kv even, q even), 29 = (kv even, q odd),
// 30 = (kv odd, q even), 31 = (kv odd, q odd).
struct PairUnion {
  std::vector<int> offset;          // [B*H*nb/2 + 1]
  std::vector<unsigned> entries;    // flagged q-pair ids
};

// blk128-native (M4): a 128-token block IS the 128x128 tile -- kv block b =
// kv-pair b, q block q = q-pair q, all four quadrants on. Zero union
// inflation; the kernel body runs FA4's dense per-step form.
static PairUnion build_pair_union_blk128(const KvToQ& k2q, int B, int H, int nb128) {
  const int pairs = B * H * nb128;
  PairUnion pu;
  pu.offset.assign(pairs + 1, 0);
  for (int p = 0; p < pairs; ++p) {
    for (int i = k2q.offset[p]; i < k2q.offset[p + 1]; ++i)
      pu.entries.push_back((unsigned)k2q.q_blocks[i] | 0xF0000000u);
    pu.offset[p + 1] = (int)pu.entries.size();
  }
  return pu;
}

static PairUnion build_pair_union(const KvToQ& k2q, int B, int H, int num_blocks) {
  const int pairs = B * H * (num_blocks / 2);
  PairUnion pu;
  pu.offset.assign(pairs + 1, 0);
  std::vector<unsigned> flags(num_blocks / 2);
  for (int p = 0; p < pairs; ++p) {
    const int bh = p / (num_blocks / 2);
    const int kb0 = bh * num_blocks + 2 * (p % (num_blocks / 2));
    std::fill(flags.begin(), flags.end(), 0u);
    for (int half = 0; half < 2; ++half)
      for (int i = k2q.offset[kb0 + half]; i < k2q.offset[kb0 + half + 1]; ++i) {
        const int qb = k2q.q_blocks[i];
        flags[qb >> 1] |= 1u << (28 + 2 * half + (qb & 1));
      }
    for (int qp = 0; qp < num_blocks / 2; ++qp)
      if (flags[qp]) pu.entries.push_back((unsigned)qp | flags[qp]);
    pu.offset[p + 1] = (int)pu.entries.size();
  }
  return pu;
}

// CPU reference. Pass 1 (parallel over q-blocks): forward O + M, Delta, dQ.
// Pass 2 (parallel over kv blocks, sorted k2q walk): dK, dV -- race-free and
// deterministic because each kv block owns its dK/dV rows.
static void cpu_vsa_bwd_ref(const __nv_bfloat16* hQ, const __nv_bfloat16* hK,
                            const __nv_bfloat16* hV, const __nv_bfloat16* hdO,
                            float* hO, float* hM, float* hDelta,
                            float* hdQ, float* hdK, float* hdV,
                            int B, int H, int S, int hd, int num_blocks, int max_kv,
                            const int* q2k_idx, const int* q2k_num, const KvToQ& k2q,
                            const float* file_O, const float* file_M, float* file_err) {
  const float sm_scale = 1.0f / sqrtf((float)hd);
  const long total_elems = (long)B * S * H * hd;
  for (long i = 0; i < total_elems; ++i) { hO[i] = 0.f; hdQ[i] = 0.f; hdK[i] = 0.f; hdV[i] = 0.f; }

  float lse_err = 0.f, o_err = 0.f;
  #pragma omp parallel for schedule(dynamic) reduction(max : lse_err, o_err)
  for (int bhq = 0; bhq < B * H * num_blocks; ++bhq) {
    const int mtile = bhq % num_blocks;
    const int bh    = bhq / num_blocks;
    const int h     = bh % H;
    const int b     = bh / H;
    const int num_kv_blocks = q2k_num[bhq];

    for (int qi = 0; qi < BLOCK; ++qi) {
      const long qp   = (long)b * S + (long)mtile * BLOCK + qi;
      const long row  = (long)bh * S + (long)mtile * BLOCK + qi;
      std::vector<float> z((size_t)num_kv_blocks * BLOCK);
      float m = -INFINITY;
      int idx = 0;
      for (int kk = 0; kk < num_kv_blocks; ++kk) {
        const int blk = q2k_idx[(size_t)bhq * max_kv + kk];
        for (int kj = 0; kj < BLOCK; ++kj) {
          const long kp = (long)b * S + (long)blk * BLOCK + kj;
          float dot = 0.f;
          for (int e = 0; e < hd; ++e)
            dot += __bfloat162float(hQ[(qp * H + h) * hd + e])
                 * __bfloat162float(hK[(kp * H + h) * hd + e]);
          z[idx] = dot * sm_scale * LOG2E;
          m = fmaxf(m, z[idx]);
          ++idx;
        }
      }
      float l = 0.f;
      for (int j = 0; j < num_kv_blocks * BLOCK; ++j) l += exp2f(z[j] - m);
      const float inv_l = 1.f / l;
      idx = 0;
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
        delta += __bfloat162float(hdO[(qp * H + h) * hd + e])
               * __bfloat162float(__float2bfloat16(hO[(qp * H + h) * hd + e]));
      hDelta[row] = delta;

      idx = 0;
      for (int kk = 0; kk < num_kv_blocks; ++kk) {
        const int blk = q2k_idx[(size_t)bhq * max_kv + kk];
        for (int kj = 0; kj < BLOCK; ++kj) {
          const long kp = (long)b * S + (long)blk * BLOCK + kj;
          const float P = exp2f(z[idx] - M_row);
          float dP = 0.f;
          for (int e = 0; e < hd; ++e)
            dP += __bfloat162float(hdO[(qp * H + h) * hd + e])
                * __bfloat162float(hV[(kp * H + h) * hd + e]);
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
        const long qp    = (long)b * S + (long)mtile * BLOCK + qi;
        const long row   = (long)bh * S + (long)mtile * BLOCK + qi;
        const float M_row = hM[row];
        const float delta = hDelta[row];
        for (int kj = 0; kj < BLOCK; ++kj) {
          const long kp = (long)b * S + (long)kb * BLOCK + kj;
          float dot = 0.f, dP = 0.f;
          for (int e = 0; e < hd; ++e) {
            dot += __bfloat162float(hQ[(qp * H + h) * hd + e])
                 * __bfloat162float(hK[(kp * H + h) * hd + e]);
            dP  += __bfloat162float(hdO[(qp * H + h) * hd + e])
                 * __bfloat162float(hV[(kp * H + h) * hd + e]);
          }
          const float P = exp2f(dot * sm_scale * LOG2E - M_row);
          // P feeds dV as bf16 (MMA operand) and dS feeds dK as bf16, matching
          // the Triton and sm100a GPU quantization points.
          const float Pq = __bfloat162float(__float2bfloat16(P));
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
    x ^= x >> 15; x *= 2246822519u; x ^= x >> 13; x *= 3266489917u; x ^= x >> 16;
    h[i] = __float2bfloat16((float)(x % 2039u) / 1019.5f - 1.0f);
  }
}

// N(0,1) Gaussian fill via Box-Muller (VSA_GAUSS; same hash as the forward bench).
static void fillg(__nv_bfloat16* h, long n, unsigned seed) {
  for (long i = 0; i < n; ++i) {
    uint32_t x = (uint32_t)i * 2654435761u + seed * 40503u + 0x9e3779b9u;
    x ^= x >> 15; x *= 2246822519u; x ^= x >> 13; x *= 3266489917u; x ^= x >> 16;
    uint32_t y = x * 2654435761u + 0x85ebca6bu; y ^= y >> 13; y *= 3266489917u; y ^= y >> 16;
    float u1 = (float)((x % 1000003u) + 1u) / 1000004.0f;
    float u2 = (float)(y % 1000003u) / 1000003.0f;
    float g = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
    h[i] = __float2bfloat16(g);
  }
}

struct Sh { int B, H, num_blocks, topk, hd; const char* lab; };

static void run(const Sh& sh) {
  const int  B = sh.B, H = sh.H, num_blocks = sh.num_blocks, topk = sh.topk, hd = sh.hd;
  const int  S = num_blocks * BLOCK;
  const int  max_kv = topk;
  const long tq = (long)B * S;
  const int  num_global_q_blocks = B * H * num_blocks;

  printf("  [%-9s H%-2d num_blocks%-3d k%-3d S%d blk%d] N_q=%ld topk=%d\n",
         sh.lab, H, num_blocks, topk, S, BLOCK, tq, topk);

  std::vector<__nv_bfloat16> hQ(tq * H * hd), hK(tq * H * hd), hV(tq * H * hd), hdO(tq * H * hd);
  const char* load_npy = getenv("LOAD_NPY");
  if (load_npy) {
    const std::string d(load_npy);
    auto ld = [&](const char* nm, std::vector<__nv_bfloat16>& h) {
      char p[64]; snprintf(p, sizeof p, "/%s_S%d.npy", nm, S);
      auto bits = npy_load_vec<uint16_t>(d + p);
      if (bits.size() != h.size()) { fprintf(stderr, "LOAD_NPY: %s size %zu != %zu\n", nm, bits.size(), h.size()); exit(1); }
      memcpy(h.data(), bits.data(), h.size() * 2);
    };
    ld("q", hQ); ld("k", hK); ld("v", hV); ld("do", hdO);
  } else {
    auto FILL = getenv("VSA_GAUSS") ? fillg : fillr;
    FILL(hQ.data(),  hQ.size(),  11);
    FILL(hK.data(),  hK.size(),  22);
    FILL(hV.data(),  hV.size(),  33);
    FILL(hdO.data(), hdO.size(), 44);
  }

  // q2k index: LOAD_NPY head-independent [num_blocks, topk] broadcast, or topk DISTINCT
  // block ids per (b,h,mtile) via partial Fisher-Yates (same knobs as the forward bench).
  std::vector<int> hq2k_idx((size_t)num_global_q_blocks * max_kv, 0);
  std::vector<int> hq2k_num(num_global_q_blocks, topk);
  const bool sort_sel = getenv("VSA_SORT_SEL") != nullptr;
  const bool seed_by_local_q_block = getenv("VSA_SEED_QBLK") != nullptr;
  if (load_npy) {
    char p[64]; snprintf(p, sizeof p, "/idx_S%d_blk%d.npy", S, BLOCK);
    auto idx = npy_load_vec<int32_t>(std::string(load_npy) + p);
    if (idx.size() != (size_t)num_blocks * topk) { fprintf(stderr, "LOAD_NPY: idx size %zu != %d\n", idx.size(), num_blocks * topk); exit(1); }
    for (int global_mtile = 0; global_mtile < num_global_q_blocks; ++global_mtile) {
      const int mtile = global_mtile % num_blocks;
      for (int i = 0; i < topk; ++i) hq2k_idx[(size_t)global_mtile * max_kv + i] = idx[(size_t)mtile * topk + i];
    }
  } else {
    std::vector<int> perm(num_blocks);
    for (int global_mtile = 0; global_mtile < num_global_q_blocks; ++global_mtile) {
      for (int i = 0; i < num_blocks; ++i) perm[i] = i;
      uint32_t st = (uint32_t)(seed_by_local_q_block ? (global_mtile % num_blocks) : global_mtile) * 2654435761u + 12345u;
      for (int i = 0; i < topk; ++i) {
        st ^= st << 13; st ^= st >> 17; st ^= st << 5;
        const int j = i + (int)(st % (uint32_t)(num_blocks - i));
        const int t = perm[i]; perm[i] = perm[j]; perm[j] = t;
        hq2k_idx[(size_t)global_mtile * max_kv + i] = perm[i];
      }
      if (sort_sel) std::sort(&hq2k_idx[(size_t)global_mtile * max_kv], &hq2k_idx[(size_t)global_mtile * max_kv] + topk);
    }
  }

  const KvToQ k2q = invert_q2k(hq2k_idx.data(), hq2k_num.data(), B, H, num_blocks, max_kv);
  const PairUnion pu = (BLOCK == 128) ? build_pair_union_blk128(k2q, B, H, num_blocks)
                                      : build_pair_union(k2q, B, H, num_blocks);
  {
    long usum = pu.entries.size();
    long csum = 0;
    for (int c : k2q.count) csum += c;
    printf("  pair-union: pairs=%d entries=%ld inflation=%.3fx steps/pair=%.1f\n",
           B * H * (num_blocks / 2), usum, csum ? 2.0 * usum / csum : 0.0,
           (double)usum / (B * H * (num_blocks / 2)));
  }
  {
    int cmin = INT_MAX, cmax = 0, zeros = 0;
    long csum = 0;
    for (int c : k2q.count) { cmin = std::min(cmin, c); cmax = std::max(cmax, c); csum += c; if (c == 0) ++zeros; }
    printf("  k2q: kv_blocks=%d count min=%d max=%d mean=%.2f zero-count=%d\n",
           num_global_q_blocks, cmin, cmax, (double)csum / num_global_q_blocks, zeros);
  }

  const char* dump_prefix = getenv("DUMP_BWD");
  const char* cpu_env = getenv("CPU_REF");
  const double selected_pairs = (double)B * H * num_blocks * topk * (double)BLOCK * BLOCK;
  bool run_cpu = cpu_env ? (atoi(cpu_env) != 0)
                         : (dump_prefix != nullptr || selected_pairs <= 4e6);
  if (dump_prefix && !run_cpu)
    printf("  DUMP_BWD set but CPU_REF=0: no reference to dump\n");

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
    cpu_vsa_bwd_ref(hQ.data(), hK.data(), hV.data(), hdO.data(),
                    hO.data(), hM.data(), hDelta.data(),
                    hdQ.data(), hdK.data(), hdV.data(),
                    B, H, S, hd, num_blocks, max_kv,
                    hq2k_idx.data(), hq2k_num.data(), k2q,
                    load_npy ? file_O.data() : nullptr, load_npy ? file_M.data() : nullptr,
                    file_err);
    const auto t1 = std::chrono::steady_clock::now();
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
      npy_save_f32(prefix + "_dq.npy",    hdQ.data(),    {tq, (long)H, (long)hd});
      npy_save_f32(prefix + "_dk.npy",    hdK.data(),    {tq, (long)H, (long)hd});
      npy_save_f32(prefix + "_dv.npy",    hdV.data(),    {tq, (long)H, (long)hd});
      npy_save_f32(prefix + "_M.npy",     hM.data(),     {(long)B * H, (long)S});
      npy_save_f32(prefix + "_delta.npy", hDelta.data(), {(long)B * H, (long)S});
      printf("  dump: %s_{dq,dk,dv}.npy [%ld,%d,%d] f32; %s_{M,delta}.npy [%d,%d] f32"
             " ([H,S] at B=1; M log2-domain)\n",
             dump_prefix, tq, H, hd, dump_prefix, B * H, S);
    }
  } else {
    printf("  cpu ref: skipped (%.0f selected pairs; CPU_REF=1 to force)\n", selected_pairs);
  }

  // GPU backward: Delta preprocess + kv-stationary main kernel
  // (dK/dV private, dQ via plain bulk reduce-add into fp32 dQaccum).
  // blk64 = quad-union metadata; blk128 = native (all quadrants on).
  if ((BLOCK != 64 && BLOCK != 128) || B != 1) {
    printf("  gpu: skipped (blk64/blk128, B=1 only)\n");
    return;
  }
  const int num_blocks64 = num_blocks * (BLOCK / 64);
  {
    using namespace vsa_bwd_blk64;
    const long elems = tq * H * hd;
    __nv_bfloat16 *dQg, *dKg, *dVg, *dDOg, *dOg, *dDKout, *dDVout;
    float *dMg, *dDeltag, *dDQA;
    int *dOff, *dBlk;
    CUDA_CHECK(cudaMalloc(&dQg,  elems * 2));
    CUDA_CHECK(cudaMalloc(&dKg,  elems * 2));
    CUDA_CHECK(cudaMalloc(&dVg,  elems * 2));
    CUDA_CHECK(cudaMalloc(&dDOg, elems * 2));
    CUDA_CHECK(cudaMalloc(&dOg,  elems * 2));
    CUDA_CHECK(cudaMalloc(&dDKout, elems * 2));
    CUDA_CHECK(cudaMalloc(&dDVout, elems * 2));
    CUDA_CHECK(cudaMalloc(&dMg,     (size_t)B * H * S * 4));
    CUDA_CHECK(cudaMalloc(&dDeltag, (size_t)B * H * S * 4));
    CUDA_CHECK(cudaMalloc(&dDQA, (size_t)B * H * S * hd * 4));
    CUDA_CHECK(cudaMalloc(&dOff, (pu.offset.size()) * 4));
    CUDA_CHECK(cudaMalloc(&dBlk, (pu.entries.size() ? pu.entries.size() : 1) * 4));
    CUDA_CHECK(cudaMemcpy(dQg,  hQ.data(),  elems * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dKg,  hK.data(),  elems * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVg,  hV.data(),  elems * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dDOg, hdO.data(), elems * 2, cudaMemcpyHostToDevice));
    {
      std::vector<__nv_bfloat16> hObf(elems);
      for (long i = 0; i < elems; ++i) hObf[i] = __float2bfloat16(hO[i]);
      CUDA_CHECK(cudaMemcpy(dOg, hObf.data(), elems * 2, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(dMg, hM.data(), (size_t)B * H * S * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dOff, pu.offset.data(), pu.offset.size() * 4, cudaMemcpyHostToDevice));
    if (!pu.entries.empty())
      CUDA_CHECK(cudaMemcpy(dBlk, pu.entries.data(), pu.entries.size() * 4, cudaMemcpyHostToDevice));

    VsaBwdArgs args;
    args.q = dQg; args.k = dKg; args.v = dVg; args.dout = dDOg;
    args.dqaccum = dDQA; args.dk = dDKout; args.dv = dDVout;
    args.m_rows = dMg; args.delta_rows = dDeltag;
    args.pair_offset = dOff; args.pair_union = reinterpret_cast<const unsigned*>(dBlk);
    args.num_heads = H; args.seqlen = S; args.num_blocks = num_blocks64;
    args.sm_scale = 1.0f / sqrtf((float)hd);

    auto run_once = [&]() {
      CUDA_CHECK(cudaMemsetAsync(dDQA, 0, (size_t)B * H * S * hd * 4));
      CUDA_CHECK(launch_vsa_bwd_preprocess(dOg, dDOg, dDeltag, H, S, 0));
      CUDA_CHECK(launch_vsa_bwd_sm100a(args, 0));
    };
    run_once();
    CUDA_CHECK(cudaDeviceSynchronize());

    if (run_cpu || getenv("DUMP_BWD_GPU")) {
      std::vector<float> gDelta((size_t)B * H * S), gDQA((size_t)B * H * S * hd);
      std::vector<__nv_bfloat16> gDK(elems), gDV(elems);
      CUDA_CHECK(cudaMemcpy(gDelta.data(), dDeltag, gDelta.size() * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(gDQA.data(), dDQA, gDQA.size() * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(gDK.data(), dDKout, elems * 2, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(gDV.data(), dDVout, elems * 2, cudaMemcpyDeviceToHost));

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
      // dq: gpu dQaccum is drain-native (per (bh, q_blk): 4 chunks of
      // [64 rows x 32 cols] fp32, contiguous, float4 slots xor-swizzled by
      // row&7 within each 128B row) and holds RAW dQc sums -- unscramble to
      // [t*H + h, d] and apply sm_scale (the postprocess's job).
      const float dq_scale = 1.0f / sqrtf((float)hd);
      std::vector<float> gdq(elems);
      for (int h = 0; h < H; ++h)
        for (int t = 0; t < S; ++t)
          for (int d = 0; d < hd; ++d) {
            const int r = t % 64;
            const int slot = ((d % 32) / 4) ^ (r & 7);
            gdq[((size_t)t * H + h) * hd + d] =
                gDQA[(size_t)h * S * hd + (size_t)(t / 64) * 64 * hd
                     + (size_t)(d / 32) * 64 * 32 + (size_t)r * 32 + slot * 4 + (d % 4)]
                * dq_scale;
          }
      std::vector<float> gdk(elems), gdv(elems);
      for (long i = 0; i < elems; ++i) { gdk[i] = __bfloat162float(gDK[i]); gdv[i] = __bfloat162float(gDV[i]); }
      const double rq = rel_norm(hdQ.data(), gdq.data(), elems);
      const double rk = rel_norm(hdK.data(), gdk.data(), elems);
      const double rv = rel_norm(hdV.data(), gdv.data(), elems);
      if (run_cpu && getenv("VERIFY_ARGMAX")) {
        long am = 0; double best = -1;
        for (long i = 0; i < elems; ++i) {
          const double d = fabs((double)hdQ[i] - gdq[i]);
          if (d > best) { best = d; am = i; }
        }
        const int d_ = (int)(am % hd);
        const int h_ = (int)((am / hd) % H);
        const long t_ = am / hd / H;
        printf("  dq argmax: t=%ld (blk64 %ld, row64 %ld) h=%d d=%d ref=%.6f got=%.6f\n",
               t_, t_ / 64, t_ % 64, h_, d_, hdQ[am], gdq[am]);
        long over1e3 = 0, over3e4 = 0; double sum = 0;
        for (long i = 0; i < elems; ++i) {
          const double d = fabs((double)hdQ[i] - gdq[i]);
          sum += d;
          if (d > 1e-3) ++over1e3;
          if (d > 3e-4) ++over3e4;
        }
        printf("  dq diff: mean=%.2e  >3e-4: %ld/%ld  >1e-3: %ld\n",
               sum / elems, over3e4, elems, over1e3);
        int shown = 0;
        for (long i = 0; i < elems && shown < 12; ++i) {
          const double d = fabs((double)hdQ[i] - gdq[i]);
          if (d > 3e-4) {
            printf("    t=%ld h=%ld d=%ld ref=%.6f got=%.6f\n",
                   i / hd / H, (i / hd) % H, i % hd, hdQ[i], gdq[i]);
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
      // dq gate: 2e-3 at blk64; 4e-3 at blk128 -- the CPU-ref-vs-torch-fp32
      // noise floor from the production bf16 quantization points is
      // ~1.9-2.7e-3 (oracle_bwd.py, 2026-08-25), so GPU-vs-CPU can
      // legitimately reach ~2x that; GPU-vs-torch measured at the same
      // distance as CPU-vs-torch.
      if (run_cpu) {
        const double rq_gate = (::BLOCK == 128) ? 4e-3 : 2e-3;
        const bool pass = dmax < 1e-4 && rq < rq_gate && rk < 8e-3 && rv < 8e-3;
        printf("  gpu verify: delta max|diff|=%.2e  dq rel=%.2e  dk rel=%.2e  dv rel=%.2e  %s\n",
               dmax, rq, rk, rv, pass ? "OK" : "FAIL");
        if (!pass) exit(1);
      }

      if (const char* sn = getenv("STRESS_N")) {
        const int n = atoi(sn);
        std::vector<__nv_bfloat16> rDK(elems), rDV(elems);
        std::vector<float> rDQ(gDQA.size());
        bool ok = true;
        for (int it = 0; it < n && ok; ++it) {
          run_once();
          CUDA_CHECK(cudaDeviceSynchronize());
          CUDA_CHECK(cudaMemcpy(rDK.data(), dDKout, elems * 2, cudaMemcpyDeviceToHost));
          CUDA_CHECK(cudaMemcpy(rDV.data(), dDVout, elems * 2, cudaMemcpyDeviceToHost));
          CUDA_CHECK(cudaMemcpy(rDQ.data(), dDQA, rDQ.size() * 4, cudaMemcpyDeviceToHost));
          ok = memcmp(rDK.data(), gDK.data(), elems * 2) == 0 &&
               memcmp(rDV.data(), gDV.data(), elems * 2) == 0;
          double mq = 0;
          for (long i = 0; i < (long)rDQ.size(); ++i)
            mq = std::max(mq, fabs((double)rDQ[i] - gDQA[i]));
          if (mq > 1e-3) ok = false;
        }
        printf("  stress x%d: %s\n", n, ok ? "OK (dk/dv bitwise, dq stable)" : "FAIL");
        if (!ok) exit(1);
      }
    }

    if (block_sparse_bwd_bf16_benchmark::enabled()) {
      const auto options = block_sparse_bwd_bf16_benchmark::options_from_env();
      const double ms = block_sparse_bwd_bf16_benchmark::measure(run_once, options);
      const double tflops = block_sparse_bwd_bf16_benchmark::tflops(hd, selected_pairs, ms);
      printf("  gpu bwd: %.4f ms  %.1f TFLOPS(bwd 2.5x sel; incl. memset+preprocess)\n",
             ms, tflops);
    }

#ifdef WARP_PROF
    {
      int sms = 0;
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
      const int wp_total = H * (num_blocks64 / 2);
      const int wp_grid = USE_CLC ? wp_total : (wp_total < sms ? wp_total : sms);
      WpBuffer wp = wp_alloc(dim3((unsigned)wp_grid, 1, 1));
      run_once();
      CUDA_CHECK(cudaDeviceSynchronize());
      wp_readback(wp);
      const char* roles[15] = {"cmp","cmp","cmp","cmp","cmp","cmp","cmp","cmp",
                               "red","red","red","red","mma","load","sched"};
      printf("  WARP_PROF block %u:\n", wp.view_block);
      wp_print_busy(wp, roles, 15, wp.view_block);
      wp_dump_raw(wp, "warp_raw_vsa_bwd.bin.gz", wp.view_block, 2);
      wp_free(wp);
    }
#endif

    cudaFree(dQg); cudaFree(dKg); cudaFree(dVg); cudaFree(dDOg); cudaFree(dOg);
    cudaFree(dDKout); cudaFree(dDVout); cudaFree(dMg); cudaFree(dDeltag);
    cudaFree(dDQA); cudaFree(dOff); cudaFree(dBlk);
  }
}

int main() {
  if (const char* b = getenv("BLOCK")) BLOCK = atoi(b);
  CUDA_CHECK(cudaFree(0));
  printf("VSA block-sparse BACKWARD bench bf16 (block=%d, uniform top-k) sm_100a\n"
         "M1 scaffold: CPU fp32 reference + sorted k2q inversion; GPU TODO (M3)\n"
         "=====================================\n", BLOCK);

  // shapes: {B, H, num_blocks, topk, hd, label} (as the forward bench).
  Sh shapes[] = {
    {1,  4,  8,  4, 128, "small"},
    {1, 16, 32,  8, 128, "fastvideo"},
    {1,  8, 64, 16, 128, "25pct"},
  };

  if (const char* s = getenv("SHAPE")) {
    Sh sh = shapes[atoi(s) % 3];
    if (getenv("BATCH")) sh.B          = atoi(getenv("BATCH"));
    if (getenv("HEADS")) sh.H          = atoi(getenv("HEADS"));
    if (getenv("NB"))    sh.num_blocks = atoi(getenv("NB"));
    if (getenv("TOPK"))  sh.topk       = atoi(getenv("TOPK"));
    sh.lab = "custom";
    run(sh);
    return 0;
  }
  const int B = getenv("BATCH") ? atoi(getenv("BATCH")) : 1;
  for (Sh sh : shapes) { sh.B = B; run(sh); }
  return 0;
}
