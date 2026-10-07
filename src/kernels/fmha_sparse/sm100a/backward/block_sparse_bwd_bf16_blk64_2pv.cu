// block_sparse_bwd_bf16_blk64_2pv.cu -- two-pass blk64 kernel for long
// sequences: pass 2 is an
// occupancy-preserving ws-TS ITEM-PAIRED kernel. Pairing two ranks per step
// with a double-buffered D would need 224KB SMEM / 384
// TMEM columns and therefore one CTA/SM. This kernel uses one rank per
// step, a two-stage ring, one persistent D, and two 32-column A buffers:
// 112KB SMEM / 192 live TMEM columns (256 allocated), hence two CTAs/SM.
// Its 16KB output image is split between the two items; both item stores
// run concurrently for each 64-column dimension half. Long-path specifics:
//   - the long-path preprocess emits a THIRD transposed slab, K^T
//     ([H*hd, tokens]), which paired pass 2 reads (tmap_kt);
//   - pass 2: 7 warps, two q64 items per CTA iteration in lockstep
//     (uniform topk), 4 ws-TS m64n256 atoms per one-rank step -- see the
//     header above vsa_bwd_pass2_kernel for the full design.
// This translation unit also has a plain-m128 single-item pass2 and dispatches
// to it below S=32768, where item pairing is ~2-3% slower. The specialized
// short preprocess skips the otherwise-unused K^T slab.
// The two-pass rationale:
// the measured cp.reduce wall (48 GB/s/SM, 6.5 TB/s aggregate) makes any
// fp32 dqaccum RMW drain alone slower than the whole blk128 kernel --
// so dQ is computed with ZERO RMW anywhere:
//   Pass 1 (kv-stationary main kernel without the dQ path): dK/dV,
//   plus each (q64, kv64) dS^T tile (8KB bf16, xor-swizzled MMA-A
//   image) is plain-STORED to ds_buf[(bh*nq + q64)*topk + rank] (rank =
//   the kv block's position in the q-block's own topk list, carried in
//   bits 16+ of the k2q entries).
//   Pass 2 gathers those tiles and reduces dQ per q64 item pair; no
//   dqaccum, no postprocess. ds_buf = pairs x 8KB (67 GB at S=131k).
//
// Pass 1 design:
//   - CLC-persistent (sched warp 14 + per-warp BwdItemSource); item = flat
//     bh * nb64 + kv_block, chunkable via item_base/item_count.
//   - Ring stage-pair roles ALTERNATE per global quad G: QT pair =
//     (G&1)?{2,3}:{0,1}, dOT the other. QT(G) reuses the pair dV(G-1)
//     freed EARLY, taking that TMA off the dK -> S^T critical cycle.
//     Per-stage parity counters on both
//     sides; do not reintroduce a rotating stage tracker here.
//   - Item hand-off gates: KV_FREE (commit after the item's last quad --
//     S^T is the last K reader) before the next item's K/V piggyback;
//     EPI_DONE (256)
//     before the next item's first fills (the epilogue bounce = the LAST
//     quad's dOT pair) and before the MMA's next dV zero-init.
//   - dS-store warp 0: per quad, waits DST_READY, one
//     8KB bulk S2G per REAL entry (pads skipped: their (id, rank) would
//     alias the clamped last entry and overwrite its tile), then
//     wait_group_read<0>; the compute lane never touches the images.
//     Warp 1 loads LSE/Delta; warps 2-3 only ride the item stream.
//   - TMEM (512 cols): S^T=P^T 0-127 (dual) | dV 128-255 | dP^T 256-383
//     (dS^T bf16 overlay 256-319) | dK 384-511.
//   - 16 warps: w0-3 store/stats (152), w4-11 compute (136), w12 MMA (88),
//     w13 load (88), w14 sched (88), w15 donor (24).
//
// Same harness contract as the sibling files: BLOCK=64 native,
// pair_offset/pair_union = per-kv64 q-lists (rank<<16 | q64 id), timed
// path = pre + pass1 + pass2, one stream, in head chunks: the dS buffer
// holds one chunk (the fewest chunks that fit device memory),
// TFLOPS = 2.5 * 4 * D * (B*H*nb*topk*64^2) / t.
//
// Env knobs: LOAD_NPY, BLOCK=64 (GPU path requires 64), SHAPE/BATCH/HEADS/
// NB/TOPK, VSA_GAUSS/VSA_SORT_SEL/VSA_SEED_QBLK, CPU_REF, DUMP_BWD,
// DUMP_BWD_GPU, VERIFY_ARGMAX, STRESS_N, BENCH_ITERS, BENCH_WARMUP.

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
#include "../../../../primitives/0_tcgen05_alloc.cuh"
#include "../../../../primitives/1_tcgen05_dealloc.cuh"
#include "../../../../primitives/2_tcgen05_relinquish.cuh"
#include "../../../../primitives/3_tcgen05_mma_f16.cuh"
#include "../../../../primitives/79_tcgen05_mma_ws_f16.cuh"
#include "../../../../primitives/69_griddepcontrol.cuh"
// Programmatic dependent launch across the preprocess -> pass1 -> pass2 -> pass1 ... chain: each
// launch carries the PDL attribute so the next kernel's CTAs set up (TMEM alloc, barrier init)
// while the predecessor drains; griddepcontrol.wait sits in front of every read of its data.
#ifndef KERNEL_PDL
#define KERNEL_PDL true
#endif
#include "../../../../primitives/8_tcgen05_mma_idesc.cuh"
#include "../../../../primitives/9_tcgen05_ld.cuh"
#include "../../../../primitives/10_tcgen05_st.cuh"
#include "../../../../primitives/11_tcgen05_commit.cuh"
#include "../../../../primitives/12_tcgen05_wait.cuh"
#include "../../../../primitives/15_tcgen05_fence.cuh"
#include "../../../../primitives/18_tma_load.cuh"
#include "../../../../primitives/22_tma_store.cuh"
#include "../../../../primitives/25_tma_async_group.cuh"
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
#include "../../../../primitives/63_cvt_f32_to_f16_bf16.cuh"
#include "../../../../primitives/70_smem_ptr.cuh"
#include "../../../../primitives/76_packed_f32x2.cuh"
#include "../../../../primitives/77_ex2_approx.cuh"
#include "../../../../primitives/61_clc_try_cancel.cuh"
#include "../../../../composites/118_mbarrier_phase_tracking.cuh"
#include "../../../../composites/106_clc_fetch_next_tile.cuh"
#include "../../../../../tests/test_helpers.cuh"
#include "../../../../primitives/_warp_prof_noop.cuh"

namespace vsa_bwd_blk64 {

constexpr int KV_ROWS  = 64;                        // kv block = tile rows
constexpr int QUAD     = 256;                       // gathered q tokens/step
constexpr int HEAD_DIM = 128;
constexpr int SUB      = 64;                        // one 128B-swizzle unit
constexpr int KV_UNIT_BYTES  = KV_ROWS * SUB * 2;   // 8 KB (K/V per hd unit)
constexpr int RING_SLOT_BYTES = 32 * 1024;
constexpr int RING_STAGES     = 4;
constexpr int DST_HALF_BYTES  = KV_ROWS * 2 * SUB * 2;  // 16 KB (64 kv x 128 q)
// FA4 warp map (as blk128): 0-3 reduce, 4-11 compute, 12 MMA, 13 load,
// 14 CLC scheduler, 15 empty (register donor).
constexpr int N_WARPS = 16;
constexpr int W_COMPUTE0 = 4, W_MMA = 12, W_LOAD = 13, W_SCHED = 14;

// Persistent CLC work stealing (default on; VSA_BWD_CLC=0 for the
// grid-stride fallback). Item = flat bh * nb64 + kv_block.
#ifndef VSA_BWD_CLC
#define VSA_BWD_CLC 1
#endif
constexpr bool USE_CLC = VSA_BWD_CLC;
constexpr int CLC_STAGES = 2;
constexpr int CLC_ARRIVALS = 15;   // worker warps 0-13 + the sched fetch

// blk128's per-warp register budgets via __maxnreg__(128) + setmaxnreg:
// reduce 152, compute 136, mma/load/sched 88, empty 24.
// Pool 4*152 + 8*136 + 3*88 + 24 = 1984 <= 2048.
__device__ __forceinline__ void warp_regs_inc_152() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 152;");
}
__device__ __forceinline__ void warp_regs_inc_136() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 136;");
}
__device__ __forceinline__ void warp_regs_dec_88() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
}
__device__ __forceinline__ void warp_regs_dec_24() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 24;");
}

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
void bulk_g2s(uint32_t smem_dst, const void* gmem_src, int bytes, uint32_t mbar_smem) {
  asm volatile(
    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
    " [%0], [%1], %2, [%3];\n"
    :: "r"(smem_dst), "l"(gmem_src), "r"(bytes), "r"(mbar_smem)
    : "memory");
}

__device__ __forceinline__
void bulk_s2g(const void* gmem_dst, uint32_t smem_src, int bytes) {
  asm volatile(
    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
    :: "l"(gmem_dst), "r"(smem_src), "r"(bytes)
    : "memory");
}

// ws-atom lead wrappers (fwd kernel's form: elect predicate rides ON the
// instruction). Trailing operands: idesc, enable-input-d pred, zero-column
// mask (0 = disabled).
__device__ __forceinline__ void tcgen05_mma_ws_f16_ss_lead(uint32_t lead,
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
__device__ __forceinline__ void tcgen05_mma_ws_f16_ts_lead(uint32_t lead,
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
__device__ __forceinline__ void tcgen05_mma_f16_ss_lead1(uint32_t lead,
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc,
    bool enable_input_d) {
  asm volatile(
    "{\n\t"
    ".reg .pred p, q;\n\t"
    "setp.ne.b32 q, %0, 0;\n\t"
    "setp.ne.b32 p, %5, 0;\n\t"
    "@q tcgen05.mma.cta_group::1.kind::f16 [%1], %2, %3, %4, {%6, %7, %8, %9}, p;\n\t"
    "}\n"
    :: "r"(lead), "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc),
       "r"(enable_input_d ? 1u : 0u), "r"(0u), "r"(0u), "r"(0u), "r"(0u));
}
__device__ __forceinline__ void tcgen05_commit1_lead(uint32_t lead, uint32_t mbar) {
  asm volatile(
    "{\n\t"
    ".reg .pred q;\n\t"
    "setp.ne.b32 q, %0, 0;\n\t"
    "@q tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%1];\n\t"
    "}\n"
    :: "r"(lead), "r"(mbar) : "memory");
}

// SMEM (~232 KB): K 16K | V 16K | ring 4x32K | dS^T half images 2x16K |
// dV/dK epilogue bounce 2x16K | LSE 1K + dPsum 1K (256 f32, 1-stage) |
// barriers. The bounce has its OWN buffer (SMEM_BOUNCE); it does not
// alias the ring.
constexpr int SMEM_K    = 0;
constexpr int SMEM_V    = SMEM_K + 2 * KV_UNIT_BYTES;
constexpr int SMEM_RING = SMEM_V + 2 * KV_UNIT_BYTES;
constexpr int SMEM_DST  = SMEM_RING + RING_STAGES * RING_SLOT_BYTES;
constexpr int SMEM_BOUNCE = SMEM_DST + 2 * DST_HALF_BYTES;  // dV+dK epilogue
constexpr int SMEM_LSE  = SMEM_BOUNCE + 2 * DST_HALF_BYTES;
constexpr int SMEM_DPS  = SMEM_LSE + QUAD * 4;
constexpr int SMEM_BARS = SMEM_DPS + QUAD * 4;
constexpr int NUM_BARS  = 21 + 2 * CLC_STAGES;
constexpr int SMEM_TOTAL = SMEM_BARS + NUM_BARS * 8 + CLC_STAGES * 16 + 48;

enum {
  BAR_RING_FULL0 = 0,             // +stage; TMA tx per slot (K/V piggyback on
  //                                 the prologue's Qq slots)
  BAR_RING_EMPTY0 = RING_STAGES,  // +stage; tcgen05 commit after the slot's MMAs
  BAR_LSE_FULL = 2 * RING_STAGES, // load -> compute, 1-stage (256 f32 / quad)
  BAR_LSE_EMPTY,                  // 8 arrivals: lane 0 of each compute warp
  BAR_DPSUM_FULL,                 // load -> compute, 1-stage
  BAR_DPSUM_EMPTY,                // 8 arrivals
  BAR_ST_READY,     // commit after the S^T atoms
  BAR_DPT_READY,    // commit after the dP^T atoms
  BAR_PT_STTMD,     // P^T STTM'd; 8 arrivals (gates dV(j))
  BAR_DST_READY,    // dS^T STTM'd (TMEM overlay); 8 arrivals (gates dK +
  //                   the store warps' t2r image production)
  BAR_DV_FULL,      // all dV accumulation issued (epilogue stage 0)
  BAR_DK_FULL,      // tail dK issued (epilogue stage 1)
  BAR_KV_FREE,      // commit after the item's last quad: K/V SMEM reusable
  BAR_EPI_DONE,     // 256: epilogue done (bounce free, dV/dK TMEM dead)
  BAR_CLC_FULL0,    // +stage
  BAR_CLC_EMPTY0 = BAR_CLC_FULL0 + CLC_STAGES,
};

// TMEM (512 cols): S^T=P^T 0-127 (dual) | dV 128-255 (dual partials) |
// dP^T=dS^T 256-383 | dK 384-511 (dual partials). P^T bf16 overlays
// words 0-63 (col-half ch at words 32ch, inside its own fp32 read range ->
// self-ordered per warp); dS^T likewise at 256 + 32ch (dK's TS A operand;
// dP^T(j+1)'s overwrite rides the in-order tcgen05 pipe behind dK(j)).
constexpr uint32_t T_ST = 0, T_DV = 128, T_DPT = 256, T_DK = 384;
constexpr uint32_t T_PT_BF16  = T_ST;
constexpr uint32_t T_DST_BF16 = T_DPT;

extern __shared__ __align__(1024) uint8_t bwd_smem[];

// Per-warp work-item source: CLC work stealing or grid stride. All worker
// warps fetch the same response ring in lockstep (clc_empty counts 15).
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

__global__ void __maxnreg__(128)
vsa_bwd_main_kernel(const __grid_constant__ CUtensorMap tmap_k64,   // box (64,64)
                    const __grid_constant__ CUtensorMap tmap_v64,
                    const __grid_constant__ CUtensorMap tmap_qt,    // [H*hd, tok], box (64,128)
                    const __grid_constant__ CUtensorMap tmap_dot,
                    const __grid_constant__ CUtensorMap tmap_dk,
                    const __grid_constant__ CUtensorMap tmap_dv,
                    __nv_bfloat16* __restrict__ ds_buf,
                    const float* __restrict__ m_rows,
                    const float* __restrict__ delta_rows,
                    const int* __restrict__ pair_offset,
                    const unsigned* __restrict__ pair_union,
                    int num_heads, int seqlen, int num_blocks, int topk,
                    int item_base, int item_count, int ds_head_base,
                    float scale_log2, float sm_scale) {
  uint8_t* base = bwd_smem;
  __nv_bfloat16* sK    = reinterpret_cast<__nv_bfloat16*>(base + SMEM_K);
  __nv_bfloat16* sV    = reinterpret_cast<__nv_bfloat16*>(base + SMEM_V);
  uint8_t*       sRING = base + SMEM_RING;
  __nv_bfloat16* sDST  = reinterpret_cast<__nv_bfloat16*>(base + SMEM_DST);
  float* sLSE = reinterpret_cast<float*>(base + SMEM_LSE);   // (256,) quad
  float* sDPS = reinterpret_cast<float*>(base + SMEM_DPS);
  uint64_t* bars = reinterpret_cast<uint64_t*>(base + SMEM_BARS);
  uint32_t* clc_response = reinterpret_cast<uint32_t*>(
      (reinterpret_cast<uintptr_t>(bars + NUM_BARS) + 15u) & ~uintptr_t(15u));
  uint32_t* tmem_slot = clc_response + CLC_STAGES * 4;
  uint64_t* clc_full  = &bars[BAR_CLC_FULL0];
  uint64_t* clc_empty = &bars[BAR_CLC_EMPTY0];

  const int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;

  if (warp_id == 0) {
    tcgen05_alloc<1>(smem_ptr_u32(tmem_slot), 512);
    tcgen05_relinquish_alloc_permit<1>();
  }
  if (tid == 0) {
    #pragma unroll
    for (int s = 0; s < RING_STAGES; ++s) {
      mbarrier_init(smem_ptr_u32(&bars[BAR_RING_FULL0 + s]),  1);
      mbarrier_init(smem_ptr_u32(&bars[BAR_RING_EMPTY0 + s]), 1);
    }
    mbarrier_init(smem_ptr_u32(&bars[BAR_LSE_FULL]),   1);
    mbarrier_init(smem_ptr_u32(&bars[BAR_LSE_EMPTY]),  8);
    mbarrier_init(smem_ptr_u32(&bars[BAR_DPSUM_FULL]), 1);
    mbarrier_init(smem_ptr_u32(&bars[BAR_DPSUM_EMPTY]),8);
    mbarrier_init(smem_ptr_u32(&bars[BAR_ST_READY]),   1);
    mbarrier_init(smem_ptr_u32(&bars[BAR_DPT_READY]),  1);
    mbarrier_init(smem_ptr_u32(&bars[BAR_PT_STTMD]),   8);
    mbarrier_init(smem_ptr_u32(&bars[BAR_DST_READY]),  8);
    mbarrier_init(smem_ptr_u32(&bars[BAR_DV_FULL]),    1);
    mbarrier_init(smem_ptr_u32(&bars[BAR_DK_FULL]),    1);
    mbarrier_init(smem_ptr_u32(&bars[BAR_KV_FREE]),    1);
    mbarrier_init(smem_ptr_u32(&bars[BAR_EPI_DONE]),   256);
    if constexpr (USE_CLC) {
      #pragma unroll
      for (int st = 0; st < CLC_STAGES; ++st) {
        mbarrier_init(smem_ptr_u32(&clc_full[st]), 1);
        mbarrier_init(smem_ptr_u32(&clc_empty[st]), CLC_ARRIVALS);
      }
      #pragma unroll
      for (int i = 0; i < CLC_STAGES * 4; ++i) clc_response[i] = 0;
    }
  }
  fence_mbarrier_init_release_cluster();
  __syncthreads();
  // Slabs and Delta come from the preprocess, the dS buffer is still read by the previous
  // chunk's pass2: every warp waits (the load, stats and dS-store warps all touch them).
  if constexpr (KERNEL_PDL) griddepcontrol_wait();
  const uint32_t tmem_base = *tmem_slot;
  const int total = item_count;

  if (warp_id == N_WARPS - 1) {    // empty: donate registers and exit
    warp_regs_dec_24();
    return;
  }
  if (warp_id == W_SCHED) {
    warp_regs_dec_88();
    // Scheduler warp: produces try_cancel results into the clc ring; also
    // runs its own consumer fetch to know when to stop.
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
      #pragma unroll
      for (int st = 0; st < CLC_STAGES; ++st) {
        if (lane == 0)
          mbarrier_wait_parity_suspend(smem_ptr_u32(&clc_empty[prod_stage]), prod_phase);
        __syncwarp();
        advance_stage_phase<CLC_STAGES>(prod_stage, prod_phase);
      }
    }
    return;
  }
  WpCtx wpc = wp_ctx_init();
  BwdItemSource src{clc_full, clc_empty, clc_response, total};

  if (warp_id == W_LOAD) {
    warp_regs_dec_88();
    uint32_t epar[4] = {1, 1, 1, 1};   // per-stage empty parities
    int gq_l = 0;                      // global quad counter (pair roles)
    EmptyPhaseTracker<1> kvf_ph;
    for (int item = (int)blockIdx.x; item >= 0; item = src.next(item)) {
    const int aitem = item_base + item;
    const int bh = aitem / num_blocks;
    const int n_block = aitem % num_blocks;
    const int beg = pair_offset[aitem], cnt = pair_offset[aitem + 1] - beg;
    const int steps = (cnt + 3) >> 2;
    if (cnt == 0) continue;
    // Gathered-quad block ids, tail-clamped to the last valid entry.
    auto qid = [&](int j, int i) {   // entries = rank<<16 | q64 id
      return (int)(pair_union[beg + min(4 * j + i, cnt - 1)] & 0xFFFFu);
    };
    auto take_slot = [&](int st) {
      mbarrier_wait_parity_suspend(smem_ptr_u32(&bars[BAR_RING_EMPTY0 + st]),
                                   epar[st]);
      epar[st] ^= 1;
      return st;
    };
    // T-slab quad gather (Q^T or dO^T): slot p, region h = block (2h+p)
    // as (128 hd x 64 tok), 16KB each. Lanes 0-1 fire the two region TMAs;
    // lanes 2-3 the K/V piggyback (quad 0's QT slots only).
    auto load_t_quad = [&](const CUtensorMap* map, int j, int p, int st,
                           int extra_dst, const CUtensorMap* extra_map,
                           int extra_tok, int extra_bytes) {
      const int slot = take_slot(st);
      wp_begin(wpc, WP_LOAD_ISSUE_V);
      if (elect_one_sync())
        mbarrier_arrive_expect_tx(smem_ptr_u32(&bars[BAR_RING_FULL0 + slot]),
                                  RING_SLOT_BYTES + extra_bytes);
      __syncwarp();
      if (lane < 2)
        tma_load_2d(smem_ptr_u32(sRING + slot * RING_SLOT_BYTES + lane * 16384),
                    map, smem_ptr_u32(&bars[BAR_RING_FULL0 + slot]),
                    qid(j, 2 * lane + p) * KV_ROWS, bh * HEAD_DIM);
      else if (extra_bytes && lane < 4)
        tma_load_3d(smem_ptr_u32(base + extra_dst + (lane - 2) * KV_UNIT_BYTES),
                    extra_map, smem_ptr_u32(&bars[BAR_RING_FULL0 + slot]),
                    0, extra_tok, bh * 2 + (lane - 2));
      wp_end(wpc, WP_LOAD_ISSUE_V);
    };
    // Stage-pair roles ALTERNATE per global quad G: QT pair = (G&1)?{2,3}:
    // {0,1}, dOT the other. QT(G) then reuses the pair dV(G-1) freed EARLY,
    // taking the TMA off the dK->S^T critical cycle. K/V piggyback on the
    // item's first QT fills (KV_FREE-gated: the old item's S^T reads K).
    // The epilogue bounce has its OWN buffer, so item-head loads overlap
    // the old item's epilogue freely (only the MMA's dV waits EPI_DONE).
    mbarrier_wait_parity_suspend(smem_ptr_u32(&bars[BAR_KV_FREE]),
                                 kvf_ph.get_phase());
    kvf_ph.advance();
    {
      const int qp = (gq_l & 1) ? 2 : 0;
      load_t_quad(&tmap_qt, 0, 0, qp, SMEM_K, &tmap_k64, n_block * KV_ROWS,
                  2 * KV_UNIT_BYTES);
      load_t_quad(&tmap_qt, 0, 1, qp + 1, SMEM_V, &tmap_v64, n_block * KV_ROWS,
                  2 * KV_UNIT_BYTES);
      load_t_quad(&tmap_dot, 0, 0, 2 - qp, 0, nullptr, 0, 0);
      load_t_quad(&tmap_dot, 0, 1, 3 - qp, 0, nullptr, 0, 0);
    }
    for (int j = 0; j < steps; ++j) {
      wp_marker(wpc, WP_ITER, j);
      if (j + 1 < steps) {
        const int qp = ((gq_l + j + 1) & 1) ? 2 : 0;
        load_t_quad(&tmap_qt, j + 1, 0, qp, 0, nullptr, 0, 0);
        load_t_quad(&tmap_qt, j + 1, 1, qp + 1, 0, nullptr, 0, 0);
        load_t_quad(&tmap_dot, j + 1, 0, 2 - qp, 0, nullptr, 0, 0);
        load_t_quad(&tmap_dot, j + 1, 1, 3 - qp, 0, nullptr, 0, 0);
      }
    }
    gq_l += steps;
    }
    wp_flush(wpc);
    return;
  }
  else if (warp_id == W_MMA) {
    warp_regs_dec_88();
    const uint32_t lead = elect_one_sync() ? 1u : 0u;
    constexpr uint32_t DESC_SBO = 1024, DESC_LBO = 16;
    constexpr uint64_t KV_UNIT_DELTA = KV_UNIT_BYTES >> 4;
    constexpr uint64_t RING_DELTA    = RING_SLOT_BYTES >> 4;

    const uint32_t idesc_ws  = make_idesc_bf16_f32(KV_ROWS, QUAD, false, false);
    const uint32_t idesc_st  = make_idesc_bf16_f32(KV_ROWS, QUAD / 2, false, true);

    const uint64_t desc_k    = build_smem_desc_blackwell(smem_ptr_u32(sK), DESC_SBO, DESC_LBO, SmemSwizzleBlackwell::B128);
    const uint64_t desc_v    = build_smem_desc_blackwell(smem_ptr_u32(sV), DESC_SBO, DESC_LBO, SmemSwizzleBlackwell::B128);
    const uint64_t desc_ring = build_smem_desc_blackwell(smem_ptr_u32(sRING), DESC_SBO, DESC_LBO, SmemSwizzleBlackwell::B128);
    const uint64_t desc_ring_mn = build_smem_desc_blackwell(smem_ptr_u32(sRING), DESC_SBO, 16384u, SmemSwizzleBlackwell::B128);

    PhaseTracker<1> ptst_ph, dstr_ph;
    EmptyPhaseTracker<1> epi_ph;
    int gq = 0;                // running quad count: ring full parity

    // Alternating stage-pair roles per global quad G (see loader). Each
    // stage still fills exactly once per quad: per-stage parity counters.
    uint32_t fpar[4] = {0, 0, 0, 0};
    auto ring_wait_stage = [&](int st) {
      wp_begin(wpc, WP_MMA_WAIT_FULL_K);
      mbarrier_wait_parity(smem_ptr_u32(&bars[BAR_RING_FULL0 + st]), fpar[st]);
      fpar[st] ^= 1;
      wp_end(wpc, WP_MMA_WAIT_FULL_K);
    };
    // S^T/dP^T: two ws n128 tb=1 issues over the T-slab slots read
    // MN-major (SBO=1024 walks k = hd rows; LBO=16384 hops region h). Slot
    // u's two regions land in D cols [64u,+64) -- bit-identical to the
    // native dual S layout. Zero-init every quad. NO
    // ring-empty commits here: the slots stay full for dK (QT) / dV (dOT).
    auto issue_st_pair = [&](uint32_t t_dst, uint64_t desc_a_units,
                             int commit_bar, int stage0) {
      #pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int slot = stage0 + u;
        ring_wait_stage(slot);
        const uint64_t db = desc_ring_mn + (uint64_t)slot * RING_DELTA;
        #pragma unroll
        for (int su = 0; su < 2; ++su)
          #pragma unroll
          for (int ki = 0; ki < 4; ++ki) {
            const int a = su * 4 + ki;
            tcgen05_mma_ws_f16_ss_lead(lead, t_dst + (uint32_t)(64 * u),
                                       desc_a_units + (uint64_t)su * KV_UNIT_DELTA + 2 * ki,
                                       db + (uint64_t)(128 * a), idesc_st,
                                       a != 0);
          }
      }
      tcgen05_commit1_lead(lead, smem_ptr_u32(&bars[commit_bar]));
    };
    // ws TS over the resident T-slab slots (already waited by S^T/dP^T):
    // per-half partials, A = dual bf16 overlay. slot0 = 0 for dK (QT), 2
    // for dV (dOT); commits the ring empties (last consumer).
    auto issue_ws_ts = [&](uint32_t t_dst, uint32_t t_a, bool first,
                           int slot0) {
      #pragma unroll
      for (int p = 0; p < 2; ++p) {
        const int slot = slot0 + p;
        const uint64_t db = desc_ring + (uint64_t)slot * RING_DELTA;
        #pragma unroll
        for (int ki = 0; ki < 4; ++ki) {
          const int a = p * 4 + ki;
          tcgen05_mma_ws_f16_ts_lead(lead, t_dst, t_a + (uint32_t)(a * 8),
                                     db + 2 * ki, idesc_ws, !(first && a == 0));
        }
        tcgen05_commit1_lead(lead, smem_ptr_u32(&bars[BAR_RING_EMPTY0 + slot]));
      }
    };

    for (int item = (int)blockIdx.x; item >= 0; item = src.next(item)) {
    const int aitem = item_base + item;
    const int beg = pair_offset[aitem], cnt = pair_offset[aitem + 1] - beg;
    const int steps = (cnt + 3) >> 2;
    if (cnt == 0) continue;
    // Prologue: S^T(0), dP^T(0), then dV(0) gated on EPI_DONE (its first
    // atom zero-init-overwrites T_DV, which the previous item's epilogue
    // reads; dP^T's overwrite of dK's A overlay rides the in-order pipe).
    const int qp0 = (gq & 1) ? 2 : 0;
    issue_st_pair(tmem_base + T_ST, desc_k, BAR_ST_READY, qp0);
    issue_st_pair(tmem_base + T_DPT, desc_v, BAR_DPT_READY, 2 - qp0);
    wp_begin(wpc, WP_MMA_WAIT_P);
    mbarrier_wait_parity(smem_ptr_u32(&bars[BAR_PT_STTMD]), ptst_ph.get_phase());
    ptst_ph.advance();
    wp_end(wpc, WP_MMA_WAIT_P);
    wp_begin(wpc, WP_MMA_WAIT_ACC);
    mbarrier_wait_parity(smem_ptr_u32(&bars[BAR_EPI_DONE]), epi_ph.get_phase());
    epi_ph.advance();
    wp_end(wpc, WP_MMA_WAIT_ACC);
    issue_ws_ts(tmem_base + T_DV, tmem_base + T_PT_BF16, true, 2 - qp0);

    for (int j = 0; j < steps; ++j) {
      wp_marker(wpc, WP_ITER, j);
      if (j + 1 == steps)
        tcgen05_commit1_lead(lead, smem_ptr_u32(&bars[BAR_DV_FULL]));
      // dK(j): gated on the dS^T images + STTM.
      wp_begin(wpc, WP_MMA_WAIT_FULL_V);
      mbarrier_wait_parity(smem_ptr_u32(&bars[BAR_DST_READY]), dstr_ph.get_phase());
      dstr_ph.advance();
      wp_end(wpc, WP_MMA_WAIT_FULL_V);
      wp_begin(wpc, WP_MMA_ISSUE);
      issue_ws_ts(tmem_base + T_DK, tmem_base + T_DST_BF16, j == 0,
                  ((gq + j) & 1) ? 2 : 0);
      if (j + 1 == steps)
        tcgen05_commit1_lead(lead, smem_ptr_u32(&bars[BAR_DK_FULL]));
      wp_end(wpc, WP_MMA_ISSUE);
      if (j + 1 < steps) {
        const int qpn = ((gq + j + 1) & 1) ? 2 : 0;
        issue_st_pair(tmem_base + T_ST, desc_k, BAR_ST_READY, qpn);
        // dP^T(j+1) overwrites dK(j)'s A overlay: in-order pipe covers it.
        issue_st_pair(tmem_base + T_DPT, desc_v, BAR_DPT_READY, 2 - qpn);
        wp_begin(wpc, WP_MMA_WAIT_P);
        mbarrier_wait_parity(smem_ptr_u32(&bars[BAR_PT_STTMD]), ptst_ph.get_phase());
        ptst_ph.advance();
        wp_end(wpc, WP_MMA_WAIT_P);
        issue_ws_ts(tmem_base + T_DV, tmem_base + T_PT_BF16, false, 2 - qpn);
      }
    }
    tcgen05_commit1_lead(lead, smem_ptr_u32(&bars[BAR_KV_FREE]));
    gq += steps;
    }
    wp_flush(wpc);
    bar_sync<10>(416);
    tcgen05_dealloc<1>(tmem_base, 512);
    return;
  }
  else if (warp_id >= W_COMPUTE0) {
    warp_regs_inc_136();
    PhaseTracker<1> str_ph, dptr_ph, lsef_ph, dpsf_ph, dvf_ph, dkf_ph;
    for (int item = (int)blockIdx.x; item >= 0; item = src.next(item)) {
    const int aitem = item_base + item;
    const int bh = aitem / num_blocks;
    const int n_block = aitem % num_blocks;
    const int cnt = pair_offset[aitem + 1] - pair_offset[aitem];
    const int steps = (cnt + 3) >> 2;
    if (cnt == 0) continue;
    // Compute warps (4-11), split over the dual 128x128 S^T tile: warp cw
    // covers lanes [32*(cw&3),+32) and cols [64*(cw>>2),+64). Lane l = kv
    // row (l&63) of q-half (l>>6); the quad q position of (lane, col c) is
    // qpos = (l>>6)*128 + c, block = qpos/64.
    const int cw = warp_id - W_COMPUTE0;
    const int subp = cw & 3;
    const int ch   = cw >> 2;
    const uint32_t lane_base = (uint32_t)((subp * 32) << 16);
    const int row = subp * 32 + lane;                // TMEM lane
    const int q_half = row >> 6;
    const uint32_t col_off = (uint32_t)(ch * SUB);


    for (int j = 0; j < steps; ++j) {
      wp_marker(wpc, WP_ITER, j);
      const int rem = min(cnt - 4 * j, 4);
      // First PAD column of this thread's 64-col span (64 = none masked).
      const int qpos0 = q_half * 128 + ch * SUB;
      const int mask_from = max(0, min(64, 64 * rem - qpos0));
      const float* m_smem = sLSE + qpos0;
      const float* delta_smem = sDPS + qpos0;

      mbarrier_wait_parity_suspend(smem_ptr_u32(&bars[BAR_LSE_FULL]),
                                   lsef_ph.get_phase());
      lsef_ph.advance();

      wp_begin(wpc, WP_SM_WAIT_S);
      mbarrier_wait_parity_suspend(smem_ptr_u32(&bars[BAR_ST_READY]), str_ph.get_phase());
      str_ph.advance();
      wp_end(wpc, WP_SM_WAIT_S);
      wp_begin(wpc, WP_SM_SOFTMAX);

      uint32_t st_regs[64];
      uint32_t pt_pack[32];
      float* pt = reinterpret_cast<float*>(st_regs);
      tcgen05_ld_32x32b_x64(tmem_base + T_ST + col_off + lane_base, st_regs);
      tcgen05_fence_before_thread_sync();
      {
        const float2 scale2 = f32x2_splat(scale_log2);
        #pragma unroll
        for (int c0 = 0; c0 < 64; c0 += 4) {
          const float4 m4 = *reinterpret_cast<const float4*>(m_smem + c0);
          const float mv[4] = { -m4.x, -m4.y, -m4.z, -m4.w };
          #pragma unroll
          for (int c = c0; c < c0 + 4; c += 2) {
            const float2 z2 = ffma2(make_float2(pt[c], pt[c + 1]), scale2,
                                    make_float2(mv[c - c0], mv[c - c0 + 1]));
            float p0 = ex2_approx_f32(z2.x);
            float p1 = ex2_approx_f32(z2.y);
            if (c >= mask_from) p0 = 0.f;
            if (c + 1 >= mask_from) p1 = 0.f;
            pt[c] = p0; pt[c + 1] = p1;
            pt_pack[c / 2] = cvt_f32x2_to_bf16x2(p0, p1);
          }
        }
      }
      tcgen05_st_32x32b_x32(tmem_base + T_PT_BF16 + (uint32_t)(ch * 32) + lane_base, pt_pack);
      tcgen05_wait_st();
      tcgen05_fence_before_thread_sync();
      if (elect_one_sync()) {
        mbarrier_arrive(smem_ptr_u32(&bars[BAR_PT_STTMD]));
        mbarrier_arrive(smem_ptr_u32(&bars[BAR_LSE_EMPTY]));
      }
      wp_end(wpc, WP_SM_SOFTMAX);

      wp_begin(wpc, WP_CORR_WAIT);
      mbarrier_wait_parity_suspend(smem_ptr_u32(&bars[BAR_DPSUM_FULL]), dpsf_ph.get_phase());
      dpsf_ph.advance();
      mbarrier_wait_parity_suspend(smem_ptr_u32(&bars[BAR_DPT_READY]), dptr_ph.get_phase());
      dptr_ph.advance();
      wp_end(wpc, WP_CORR_WAIT);
      wp_begin(wpc, WP_SM_STORE_P);
      {
        uint32_t dpt_regs[64];
        tcgen05_ld_32x32b_x64(tmem_base + T_DPT + col_off + lane_base, dpt_regs);
        tcgen05_fence_before_thread_sync();
        const float* dpt = reinterpret_cast<const float*>(dpt_regs);
        #pragma unroll
        for (int c0 = 0; c0 < 64; c0 += 4) {
          const float4 d4 = *reinterpret_cast<const float4*>(delta_smem + c0);
          const float neg_delta4[4] = { -d4.x, -d4.y, -d4.z, -d4.w };
          #pragma unroll
          for (int c = c0; c < c0 + 4; c += 2) {
            const float2 s2 = fmul2(make_float2(pt[c], pt[c + 1]),
                                    fadd2(make_float2(dpt[c], dpt[c + 1]),
                                          make_float2(neg_delta4[c - c0], neg_delta4[c - c0 + 1])));
            st_regs[c / 2] = cvt_f32x2_to_bf16x2(s2.x, s2.y);
          }
        }
      }
      tcgen05_st_32x32b_x32(tmem_base + T_DST_BF16 + (uint32_t)(ch * 32) + lane_base,
                            reinterpret_cast<const uint32_t(&)[32]>(st_regs));
      tcgen05_wait_st();
      tcgen05_fence_before_thread_sync();
      if (elect_one_sync()) {
        mbarrier_arrive(smem_ptr_u32(&bars[BAR_DST_READY]));
        mbarrier_arrive(smem_ptr_u32(&bars[BAR_DPSUM_EMPTY]));
      }
      wp_end(wpc, WP_SM_STORE_P);
    }

    // Epilogue: merge the dual dV/dK half-partials (fwd correction pattern:
    // half 1 stages bf16 into the bounce image, one 256-thread barrier,
    // half 0 widen-adds and overwrites, second barrier, leader TMA-stores
    // the 64 kv rows). Bounce = the dedicated SMEM_BOUNCE buffer (item
    // hand-off is EPI_DONE-gated on the MMA's next dV zero-init only).
    {
      // Two DISJOINT 16KB images inside the bounce buffer: the dV TMA
      // store's reads are only waited at the end of the epilogue, so the
      // dK merge must not reuse its bytes.
      auto merge_store = [&](uint32_t t_base, float scale, const CUtensorMap* map,
                             __nv_bfloat16* bounce) {
        __nv_bfloat16* bsub = bounce + (size_t)ch * (KV_ROWS * SUB);
        const int r = row & 63;
        uint32_t acc_regs[64];
        tcgen05_ld_32x32b_x64(t_base + col_off + lane_base, acc_regs);
        tcgen05_fence_before_thread_sync();
        const float* acc = reinterpret_cast<const float*>(acc_regs);
        if (q_half == 1) {
          #pragma unroll
          for (int v = 0; v < 8; ++v) {
            uint4 packed;
            packed.x = cvt_f32x2_to_bf16x2(acc[v * 8 + 0] * scale, acc[v * 8 + 1] * scale);
            packed.y = cvt_f32x2_to_bf16x2(acc[v * 8 + 2] * scale, acc[v * 8 + 3] * scale);
            packed.z = cvt_f32x2_to_bf16x2(acc[v * 8 + 4] * scale, acc[v * 8 + 5] * scale);
            packed.w = cvt_f32x2_to_bf16x2(acc[v * 8 + 6] * scale, acc[v * 8 + 7] * scale);
            *reinterpret_cast<uint4*>(&bsub[r * SUB + ((v ^ (r & 7)) * 8)]) = packed;
          }
        }
        bar_sync<14>(256);
        if (q_half == 0) {
          #pragma unroll
          for (int v = 0; v < 8; ++v) {
            uint4* dst = reinterpret_cast<uint4*>(&bsub[r * SUB + ((v ^ (r & 7)) * 8)]);
            const uint4 other = *dst;
            const __nv_bfloat162* ob = reinterpret_cast<const __nv_bfloat162*>(&other);
            uint4 packed;
            uint32_t* pw = reinterpret_cast<uint32_t*>(&packed);
            #pragma unroll
            for (int w = 0; w < 4; ++w) {
              const float2 of = __bfloat1622float2(ob[w]);
              pw[w] = cvt_f32x2_to_bf16x2(acc[v * 8 + 2 * w] * scale + of.x,
                                          acc[v * 8 + 2 * w + 1] * scale + of.y);
            }
            *dst = packed;
          }
        }
        fence_proxy_async_shared_cta();
        bar_sync<14>(256);
        if (cw == 0 && elect_one_sync()) {
          #pragma unroll
          for (int ssub = 0; ssub < 2; ++ssub)
            tma_store_3d(map, 0, n_block * KV_ROWS, bh * 2 + ssub,
                         smem_ptr_u32(bounce + (size_t)ssub * KV_ROWS * SUB));
          cp_async_bulk_commit_group();
        }
        bar_sync<14>(256);
      };
      wp_begin(wpc, WP_EPI_WAIT_STORE);
      mbarrier_wait_parity_suspend(smem_ptr_u32(&bars[BAR_DV_FULL]), dvf_ph.get_phase());
      dvf_ph.advance();
      wp_end(wpc, WP_EPI_WAIT_STORE);
      wp_begin(wpc, WP_CORR_EPI);
      // Bounce = the dedicated buffer; the loader's next item overlaps
      // this epilogue freely (only the MMA's next dV waits EPI_DONE).
      merge_store(tmem_base + T_DV, 1.0f, &tmap_dv,
                  reinterpret_cast<__nv_bfloat16*>(base + SMEM_BOUNCE));
      mbarrier_wait_parity_suspend(smem_ptr_u32(&bars[BAR_DK_FULL]), dkf_ph.get_phase());
      dkf_ph.advance();
      merge_store(tmem_base + T_DK, sm_scale, &tmap_dk,
                  reinterpret_cast<__nv_bfloat16*>(base + SMEM_BOUNCE + 16384));
      // Item hand-off: both TMA stores' SMEM reads done, then publish
      // EPI_DONE (releases the loader's next dOT p0 and the MMA's next dV).
      if (cw == 0 && elect_one_sync()) cp_async_bulk_wait_group_read<0>();
      bar_sync<14>(256);
      mbarrier_arrive(smem_ptr_u32(&bars[BAR_EPI_DONE]));
      wp_end(wpc, WP_CORR_EPI);
    }
    }
    wp_flush(wpc);
    bar_sync<10>(416);
    return;
  }
  else {
    // Pass-1 dS warps (0-3): per quad, t2r the dS^T bf16 TMEM overlay (warp
    // w = warpgroup rank w = lane band w), build the xor-swizzled 8KB tile
    // images in SMEM, then warp 0 bulk-stores each REAL entry's tile to
    // ds_buf[(bh*nq + q64)*topk + rank]. This keeps the image production
    // off the compute lane (the pass pacer) entirely.
    warp_regs_inc_152();
    const int nq = seqlen / KV_ROWS;
    const uint32_t lane_base = (uint32_t)((warp_id * 32) << 16);
    const int row = warp_id * 32 + lane;
    const int q_half = row >> 6;
    PhaseTracker<1> dstr2_ph;
    EmptyPhaseTracker<1> lse_ph, dps_ph;
    // Warp 1 doubles as the stats loader (LSE + Delta, one quad ahead) --
    // off the T-slab loader's critical lane; it has ~50% slack here.
    auto load_stats = [&](int jq, int beg_i, int cnt_i, int bh_i) {
      mbarrier_wait_parity_suspend(smem_ptr_u32(&bars[BAR_LSE_EMPTY]),
                                   lse_ph.get_phase());
      lse_ph.advance();
      if (elect_one_sync()) {
        mbarrier_arrive_expect_tx(smem_ptr_u32(&bars[BAR_LSE_FULL]), QUAD * 4);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int qe = (int)(pair_union[beg_i + min(4 * jq + i, cnt_i - 1)] & 0xFFFFu);
          bulk_g2s(smem_ptr_u32(sLSE + i * KV_ROWS),
                   m_rows + (size_t)bh_i * seqlen + (size_t)qe * KV_ROWS,
                   KV_ROWS * 4, smem_ptr_u32(&bars[BAR_LSE_FULL]));
        }
      }
      mbarrier_wait_parity_suspend(smem_ptr_u32(&bars[BAR_DPSUM_EMPTY]),
                                   dps_ph.get_phase());
      dps_ph.advance();
      if (elect_one_sync()) {
        mbarrier_arrive_expect_tx(smem_ptr_u32(&bars[BAR_DPSUM_FULL]), QUAD * 4);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int qe = (int)(pair_union[beg_i + min(4 * jq + i, cnt_i - 1)] & 0xFFFFu);
          bulk_g2s(smem_ptr_u32(sDPS + i * KV_ROWS),
                   delta_rows + (size_t)bh_i * seqlen + (size_t)qe * KV_ROWS,
                   KV_ROWS * 4, smem_ptr_u32(&bars[BAR_DPSUM_FULL]));
        }
      }
    };
    for (int item = (int)blockIdx.x; item >= 0; item = src.next(item)) {
      const int aitem = item_base + item;
      const int bh = aitem / num_blocks;
      const int beg = pair_offset[aitem], cnt = pair_offset[aitem + 1] - beg;
      const int steps = (cnt + 3) >> 2;
      if (cnt == 0) continue;
      if (warp_id == 1) load_stats(0, beg, cnt, bh);
      for (int j = 0; j < steps; ++j) {
        // Stats for the next quad go out FIRST (gated only on the ring
        // buffers' empties, which free mid-quad) -- before the image work,
        // whose DST_READY gate lands late in the quad.
        if (warp_id == 1 && j + 1 < steps) load_stats(j + 1, beg, cnt, bh);
        wp_begin(wpc, WP_EPI_WAIT_ACC);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&bars[BAR_DST_READY]),
                                     dstr2_ph.get_phase());
        dstr2_ph.advance();
        wp_end(wpc, WP_EPI_WAIT_ACC);
        wp_begin(wpc, WP_EPI_STORE);
        // t2r the dual dS^T overlay, rebuild the xor-swizzled tile images
        // in SMEM, then warp 0 bulk-stores each REAL entry's tile (pads
        // skipped: their (id, rank) would alias the clamped last entry).
        #pragma unroll
        for (int cx = 0; cx < 2; ++cx) {
          uint32_t ds_regs[32];
          tcgen05_ld_32x32b_x32(tmem_base + T_DST_BF16 + (uint32_t)(cx * 32) + lane_base,
                                ds_regs);
          tcgen05_fence_before_thread_sync();
          __nv_bfloat16* dsub = sDST + (size_t)q_half * (DST_HALF_BYTES / 2)
                              + (size_t)cx * (KV_ROWS * SUB);
          const int r = row & 63;
          const uint4* src4 = reinterpret_cast<const uint4*>(ds_regs);
          #pragma unroll
          for (int v = 0; v < 8; ++v)
            *reinterpret_cast<uint4*>(&dsub[r * SUB + ((v ^ (r & 7)) * 8)]) = src4[v];
        }
        fence_proxy_async_shared_cta();
        bar_sync<11>(128);
        const int rem = min(cnt - 4 * j, 4);
        if (warp_id == 0 && lane == 0) {
          for (int i = 0; i < rem; ++i) {
            const unsigned e = pair_union[beg + 4 * j + i];
            const int qe = (int)(e & 0xFFFFu), rank = (int)(e >> 16);
            bulk_s2g(ds_buf + ((size_t)((bh - ds_head_base) * nq + qe) * topk + rank) * 4096,
                     smem_ptr_u32(sDST + (i >> 1) * 8192 + (i & 1) * 4096), 8192);
          }
          cp_async_bulk_commit_group();
          cp_async_bulk_wait_group_read<0>();
        }
        bar_sync<11>(128);
        wp_end(wpc, WP_EPI_STORE);
      }
    }
    wp_flush(wpc);
    // All dS tiles issued: pass2 may launch (its wait still covers their completion).
    if constexpr (KERNEL_PDL) {
      if (warp_id == 0 && lane == 0) griddepcontrol_launch_dependents();
    }
    bar_sync<10>(416);
    return;
  }
}

// Preprocess: one CTA per (128-token block, head), 256 threads. Computes
// Delta = rowsum(bf16(O) * dO) and writes the TRANSPOSED Q^T / dO^T /
// K^T slabs ([H*hd, tokens]): the first two feed pass 1's ws TS gathers,
// K^T is pass 2's B operand. (The dqaccum zeroing path is dead here --
// this two-pass file always passes nullptr.)
template <bool WRITE_KT>
__global__ void __launch_bounds__(256, 1)
vsa_bwd_preprocess_kernel(const __nv_bfloat16* __restrict__ q,
                          const __nv_bfloat16* __restrict__ o,
                          const __nv_bfloat16* __restrict__ dout,
                          const __nv_bfloat16* __restrict__ kk,
                          float* __restrict__ delta_rows,
                          float* __restrict__ dqaccum,
                          __nv_bfloat16* __restrict__ qt,
                          __nv_bfloat16* __restrict__ dot,
                          __nv_bfloat16* __restrict__ kt,
                          int num_heads, int seqlen) {
  const int m_block = (int)blockIdx.x;
  const int bh = (int)blockIdx.y;
  const int t0 = m_block * 128;
  // The previous call's pass2 still reads the K^T slab: wait before rewriting the slabs.
  if constexpr (KERNEL_PDL) griddepcontrol_wait();
  if (dqaccum != nullptr) {
    float4* z4 = reinterpret_cast<float4*>(
        dqaccum + (size_t)bh * seqlen * HEAD_DIM + (size_t)m_block * 128 * HEAD_DIM);
    const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
    #pragma unroll
    for (int i = 0; i < (128 * HEAD_DIM / 4) / 256; ++i)
      z4[i * 256 + threadIdx.x] = z;
  }
  // Delta: 16 threads per row, 8 cols each.
  const int drow = (int)threadIdx.x / 16;
  const int col16 = ((int)threadIdx.x % 16) * 8;
  #pragma unroll
  for (int rpass = 0; rpass < 128 / 16; ++rpass) {
    const int t = t0 + rpass * 16 + drow;
    const long gbase = ((long)t * num_heads + bh) * HEAD_DIM + col16;
    const uint4 ob = *reinterpret_cast<const uint4*>(o + gbase);
    const uint4 db = *reinterpret_cast<const uint4*>(dout + gbase);
    const __nv_bfloat162* o2 = reinterpret_cast<const __nv_bfloat162*>(&ob);
    const __nv_bfloat162* d2 = reinterpret_cast<const __nv_bfloat162*>(&db);
    float acc = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 of = __bfloat1622float2(o2[i]);
      const float2 df = __bfloat1622float2(d2[i]);
      acc += of.x * df.x + of.y * df.y;
    }
    #pragma unroll
    for (int off = 8; off > 0; off >>= 1)
      acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (col16 == 0) delta_rows[(size_t)bh * seqlen + t] = acc;
  }
  // Delta is out: pass1 may set up while the slabs are written (its wait covers them).
  if constexpr (KERNEL_PDL) griddepcontrol_launch_dependents();
  // Q^T / dO^T / K^T via an SMEM-staged transpose: uint4 on both global
  // sides (2B scalar stores are issue-bound). Stage [128 tok
  // x 128 hd] as sT[d][t] (the +2-token row pad de-conflicts the SMEM
  // banks on the transposed read-out), then each thread streams 8-token
  // uint4 rows of the output slab.
  __shared__ __align__(16) __nv_bfloat16 sT[128][130];
  const int tt = (int)threadIdx.x & 127;
  const int dp = (int)threadIdx.x >> 7;    // 0/1: hd halves of 64
  const int t = t0 + tt;
  const int out_row0 = (int)threadIdx.x >> 4;    // first output hd row
  const int tok0 = ((int)threadIdx.x & 15) * 8;  // 16 threads x 8 tokens
  #pragma unroll
  for (int pass = 0; pass < (WRITE_KT ? 3 : 2); ++pass) {
    const __nv_bfloat16* src = pass == 0 ? q : (pass == 1 ? dout : kk);
    __nv_bfloat16* dst = pass == 0 ? qt : (pass == 1 ? dot : kt);
    #pragma unroll
    for (int in_vec = 0; in_vec < 8; ++in_vec) {
      const int d = dp * 64 + in_vec * 8;
      const uint4 v = *reinterpret_cast<const uint4*>(
          src + ((size_t)t * num_heads + bh) * HEAD_DIM + d);
      const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&v);
      #pragma unroll
      for (int i = 0; i < 8; ++i) sT[d + i][tt] = e[i];
    }
    __syncthreads();
    #pragma unroll
    for (int row_step = 0; row_step < 8; ++row_step) {
      const int d = out_row0 + row_step * 16;
      uint4 v;
      __nv_bfloat16* e = reinterpret_cast<__nv_bfloat16*>(&v);
      #pragma unroll
      for (int i = 0; i < 8; ++i) e[i] = sT[d][tok0 + i];
      *reinterpret_cast<uint4*>(
          dst + ((size_t)bh * HEAD_DIM + d) * seqlen + t0 + tok0) = v;
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// Pass 2 constants (the design comment sits on vsa_bwd_pass2_kernel below).
// ---------------------------------------------------------------------------
#ifdef P2_STATIC_SCHED
constexpr int P2_WARPS = 6;
#else
constexpr int P2_WARPS = 7;
#endif
constexpr int P2_CLC_ARRIVALS = P2_WARPS; // workers + the sched fetch
#ifndef P2_LITE_STAGES
#define P2_LITE_STAGES 2
#endif
#ifndef P2_LITE_RANKS
#define P2_LITE_RANKS 1
#endif
constexpr int P2_STAGES  = P2_LITE_STAGES; // default: 2 stages
constexpr int P2_RANKS   = P2_LITE_RANKS;
constexpr int P2_A_STAGES = P2_RANKS == 1 ? 2 : 1;
static_assert(P2_RANKS == 1 || P2_RANKS == 2 || P2_RANKS == 4);
constexpr int P2_DS_SLOT = P2_RANKS * 16 * 1024; // ranks x 2 items x 8KB
constexpr int P2_KT_SLOT = P2_RANKS * 32 * 1024; // ranks x 2 item regions
constexpr int P2_SMEM_DS  = 0;
constexpr int P2_SMEM_KT  = P2_SMEM_DS + P2_STAGES * P2_DS_SLOT;
constexpr int P2_SMEM_OUT = P2_SMEM_KT + P2_STAGES * P2_KT_SLOT; // 2 x 8KB
constexpr int P2_SMEM_BAR = P2_SMEM_OUT + 16 * 1024;
enum {
  P2B_DS_FULL0 = 0,                        // +stage; bulk tx
  P2B_DS_EMPTY0 = P2B_DS_FULL0 + P2_STAGES,   // 4 arrivals (stage warps)
  P2B_KT_FULL0 = P2B_DS_EMPTY0 + P2_STAGES,   // +stage; TMA tx
  P2B_KT_EMPTY0 = P2B_KT_FULL0 + P2_STAGES,   // mma commit
  P2B_A_FULL0 = P2B_KT_EMPTY0 + P2_STAGES,    // +buf; 4 arrivals (stage warps)
  P2B_A_EMPTY0 = P2B_A_FULL0 + P2_A_STAGES,   // +buf; mma commit
  P2B_DQ_DONE0 = P2B_A_EMPTY0 + P2_A_STAGES,  // mma commit (per PAIR)
  P2B_DQ_TFREE0 = P2B_DQ_DONE0 + 1,           // 128 arrivals (epi warps)
  P2B_CLC_FULL0 = P2B_DQ_TFREE0 + 1,          // +stage
  P2B_CLC_EMPTY0 = P2B_CLC_FULL0 + CLC_STAGES,
};
constexpr int P2_NUM_BARS = 4 * P2_STAGES + 2 * P2_A_STAGES + 2 +
                            2 * CLC_STAGES;
constexpr int P2_SMEM_TOTAL = P2_SMEM_BAR + P2_NUM_BARS * 8 + CLC_STAGES * 16 + 48;
#ifdef P2_STATIC_SCHED
constexpr bool P2_USE_CLC = false;
#else
constexpr bool P2_USE_CLC = USE_CLC;
#endif

struct P2ItemSource {
  uint64_t* clc_full;
  uint64_t* clc_empty;
  uint32_t* clc_response;
  int total;
  int clc_stage = 0;
  uint32_t clc_phase = 0;
  __device__ __forceinline__ int next(int item) {
    if constexpr (P2_USE_CLC) {
      ClcTileInfo n = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, 1, true>(
          clc_full, clc_empty, clc_response, clc_stage, clc_phase, elect_one_sync());
      clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
      return n.valid ? (int)n.n_tile : -1;
    } else {
      // Static pairs are adjacent. An even item advances to its odd mate;
      // the mate advances to the next grid-stride pair base.
      const int nx = (item & 1) ? item + 2 * (int)gridDim.x - 1
                                : item + 1;
      return nx < total ? nx : -1;
    }
  }
};

// Pass 2, occupancy-preserving ws-TS ITEM-PAIRED form: each CTA runs TWO
// q64 items in lockstep (uniform topk). Per step = 1 rank x 2 items: 4
// ws-TS m64n256 atoms; lane-half h of D = item h's dQ[64 x 128]
// (independent per-half gemms). A' = the dS tiles
// TRANSPOSED to [q][kv] and bf16-packed into TMEM by the 4 stage warps
// (subpartition-locked lane bands); B = K^T tiles from the kt slab in a
// slot/region layout: one 32KB slot = rank j, region h
// (+16KB) = item h. dQ accumulates in TMEM across the whole list
// in one persistent 128-column allocation. A' ping-pongs at 32 columns.
// 7 warps: 0-3 stage + epilogue, 4 load, 5 mma, 6 sched. The 112KB
// SMEM / 192-column TMEM footprint permits 2 CTAs/SM.
__global__ void __launch_bounds__(P2_WARPS * 32, 2)
vsa_bwd_pass2_kernel(const __grid_constant__ CUtensorMap tmap_kt,   // K^T slab [H*hd, S], box (64,128)
                     const __grid_constant__ CUtensorMap tmap_dq,
                     const __nv_bfloat16* __restrict__ ds_buf,
                     const int* __restrict__ q2k_idx,
                     __nv_bfloat16* __restrict__ dq_dbg,
                     int num_heads, int seqlen, int topk,
                     int item_base, int item_count, int ds_head_base, float sm_scale) {
  uint8_t* base = bwd_smem;
  uint64_t* bars = reinterpret_cast<uint64_t*>(base + P2_SMEM_BAR);
  uint32_t* clc_response = reinterpret_cast<uint32_t*>(
      (reinterpret_cast<uintptr_t>(bars + P2_NUM_BARS) + 15u) & ~uintptr_t(15u));
  uint32_t* tmem_slot = clc_response + CLC_STAGES * 4;
  uint64_t* clc_full  = &bars[P2B_CLC_FULL0];
  uint64_t* clc_empty = &bars[P2B_CLC_EMPTY0];
  const int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;
  const int nq = seqlen / KV_ROWS;
  const int total = item_count;
  const int steps2 = (topk + P2_RANKS - 1) / P2_RANKS;

  if (warp_id == 0) {
    tcgen05_alloc<1>(smem_ptr_u32(tmem_slot), 256);
    tcgen05_relinquish_alloc_permit<1>();
  }
  if (tid == 0) {
    #pragma unroll
    for (int st = 0; st < P2_STAGES; ++st) {
      mbarrier_init(smem_ptr_u32(&bars[P2B_DS_FULL0 + st]),  1);
      mbarrier_init(smem_ptr_u32(&bars[P2B_DS_EMPTY0 + st]), 4);
      mbarrier_init(smem_ptr_u32(&bars[P2B_KT_FULL0 + st]),  1);
      mbarrier_init(smem_ptr_u32(&bars[P2B_KT_EMPTY0 + st]), 1);
    }
    #pragma unroll
    for (int b = 0; b < P2_A_STAGES; ++b) {
      mbarrier_init(smem_ptr_u32(&bars[P2B_A_FULL0 + b]),  4);
      mbarrier_init(smem_ptr_u32(&bars[P2B_A_EMPTY0 + b]), 1);
    }
    mbarrier_init(smem_ptr_u32(&bars[P2B_DQ_DONE0]),  1);
    mbarrier_init(smem_ptr_u32(&bars[P2B_DQ_TFREE0]), 128);
    if constexpr (P2_USE_CLC) {
      #pragma unroll
      for (int st = 0; st < CLC_STAGES; ++st) {
        mbarrier_init(smem_ptr_u32(&clc_full[st]), 1);
        mbarrier_init(smem_ptr_u32(&clc_empty[st]), P2_CLC_ARRIVALS);
      }
      #pragma unroll
      for (int i = 0; i < CLC_STAGES * 4; ++i) clc_response[i] = 0;
    }
  }
  fence_mbarrier_init_release_cluster();
  __syncthreads();
  if constexpr (KERNEL_PDL) griddepcontrol_wait();  // pass1's dS tiles and the K^T slab
  const uint32_t tmem_base = *tmem_slot;
  constexpr uint32_t T_D0 = 0, T_A0 = 128;   // D {0..127}; A' {128,160}

  if constexpr (P2_USE_CLC) if (warp_id == P2_WARPS - 1) {
    // Scheduler warp (same producer/consumer shape as pass 1).
    {
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
      #pragma unroll
      for (int st = 0; st < CLC_STAGES; ++st) {
        if (lane == 0)
          mbarrier_wait_parity_suspend(smem_ptr_u32(&clc_empty[prod_stage]), prod_phase);
        __syncwarp();
        advance_stage_phase<CLC_STAGES>(prod_stage, prod_phase);
      }
    }
    return;
  }
  P2ItemSource src{clc_full, clc_empty, clc_response, total};

  if (warp_id == 4) {
    // Load warp: per step, one 8KB dS tile and one 16KB K^T tile per
    // item. Two ring stages retain two-CTA residency.
    EmptyPhaseTracker<P2_STAGES> ds_ph, kt_ph;
    int item = P2_USE_CLC ? (int)blockIdx.x : 2 * (int)blockIdx.x;
    while (item >= 0) {
      const int item_b = src.next(item);
      const int a0 = item_base + item, a1 = item_b >= 0 ? item_base + item_b : -1;
      const int bh0 = a0 / nq, bh1 = a1 >= 0 ? a1 / nq : 0;
      for (int j = 0; j < steps2; ++j) {
        const int st = ds_ph.get_stage();
        const int rem = min(topk - P2_RANKS * j, P2_RANKS);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&bars[P2B_DS_EMPTY0 + st]),
                                     ds_ph.get_phase());
        ds_ph.advance();
        if (elect_one_sync()) {
          const int nb = (a1 >= 0) ? 2 : 1;
          mbarrier_arrive_expect_tx(smem_ptr_u32(&bars[P2B_DS_FULL0 + st]),
                                    nb * rem * 8192);
          bulk_g2s(smem_ptr_u32(base + P2_SMEM_DS + st * P2_DS_SLOT),
                   ds_buf + ((size_t)(a0 - ds_head_base * nq) * topk + P2_RANKS * j) * 4096,
                   rem * 8192, smem_ptr_u32(&bars[P2B_DS_FULL0 + st]));
          if (a1 >= 0)
            bulk_g2s(smem_ptr_u32(base + P2_SMEM_DS + st * P2_DS_SLOT +
                                  P2_RANKS * 8192),
                     ds_buf + ((size_t)(a1 - ds_head_base * nq) * topk + P2_RANKS * j) * 4096,
                     rem * 8192, smem_ptr_u32(&bars[P2B_DS_FULL0 + st]));
        }
        mbarrier_wait_parity_suspend(smem_ptr_u32(&bars[P2B_KT_EMPTY0 + st]),
                                     kt_ph.get_phase());
        kt_ph.advance();
        const int nb = (a1 >= 0) ? 2 : 1;
        if (elect_one_sync())
          mbarrier_arrive_expect_tx(smem_ptr_u32(&bars[P2B_KT_FULL0 + st]),
                                    nb * rem * 16384);
        __syncwarp();
        if (lane < 2 * rem) {
          const int p = lane >> 1, h = lane & 1;
          if (h == 0 || a1 >= 0) {
            const int ah = h ? a1 : a0;
            const int bhh = h ? bh1 : bh0;
            const int kvb = q2k_idx[(size_t)ah * topk + P2_RANKS * j + p];
            tma_load_2d(smem_ptr_u32(base + P2_SMEM_KT + st * P2_KT_SLOT
                                     + p * 32768 + h * 16384),
                        &tmap_kt, smem_ptr_u32(&bars[P2B_KT_FULL0 + st]),
                        kvb * KV_ROWS, bhh * HEAD_DIM);
          }
        }
      }
      item = (item_b >= 0) ? src.next(item_b) : -1;
    }
    return;
  }
  if (warp_id == 5) {
    // MMA warp: per step, one ws-TS group (4 k16 atoms). Lane-half h
    // reads item h's K^T region and accumulates item h's dQ. D is single
    // buffered; two resident CTAs hide the short pair epilogue.
    const uint32_t lead = elect_one_sync() ? 1u : 0u;
    const uint32_t idesc_ws2 = make_idesc_bf16_f32(64, 256, false, false);
    PhaseTracker<P2_STAGES> ktf_ph;
    PhaseTracker<P2_A_STAGES> af_ph;
    EmptyPhaseTracker<1> tfree_ph;
    int item = P2_USE_CLC ? (int)blockIdx.x : 2 * (int)blockIdx.x;
    while (item >= 0) {
      const int item_b = src.next(item);
      mbarrier_wait_parity(smem_ptr_u32(&bars[P2B_DQ_TFREE0]),
                           tfree_ph.get_phase());
      tfree_ph.advance();
      bool first = true;
      for (int j = 0; j < steps2; ++j) {
        const int st = ktf_ph.get_stage();
        const int ab = af_ph.get_stage();
        mbarrier_wait_parity(smem_ptr_u32(&bars[P2B_A_FULL0 + ab]), af_ph.get_phase());
        af_ph.advance();
        mbarrier_wait_parity(smem_ptr_u32(&bars[P2B_KT_FULL0 + st]), ktf_ph.get_phase());
        ktf_ph.advance();
        const int rem = min(topk - P2_RANKS * j, P2_RANKS);
        for (int p = 0; p < rem; ++p) {
          SmemDescPair db;
          db.u64 = build_smem_desc_blackwell(
              smem_ptr_u32(base + P2_SMEM_KT + st * P2_KT_SLOT + p * 32768),
              1024u, 16u, SmemSwizzleBlackwell::B128);
          #pragma unroll
          for (int ki = 0; ki < 4; ++ki) {
            tcgen05_mma_ws_f16_ts_lead(
                lead, tmem_base + T_D0,
                tmem_base + T_A0 +
                    (uint32_t)(ab * P2_RANKS * 32 + p * 32 + ki * 8),
                db.u64, idesc_ws2, !first || p != 0 || ki != 0);
            desc_add_lo(db, 2u);
          }
        }
        first = false;
        tcgen05_commit1_lead(lead, smem_ptr_u32(&bars[P2B_A_EMPTY0 + ab]));
        tcgen05_commit1_lead(lead, smem_ptr_u32(&bars[P2B_KT_EMPTY0 + st]));
      }
      tcgen05_commit1_lead(lead, smem_ptr_u32(&bars[P2B_DQ_DONE0]));
      item = (item_b >= 0) ? src.next(item_b) : -1;
    }
    return;
  }
  // Stage + epilogue warps (0-3): warp w = item (w>>1), q rows
  // [32*(w&1), +32) -- the subpartition-locked TMEM lane band [32w, +32).
  {
    const int h_item = warp_id >> 1;
    const int q0 = 32 * (warp_id & 1);
    const uint32_t lane_base = (uint32_t)((warp_id * 32) << 16);
    const int q = q0 + lane;                  // this thread's output row
    PhaseTracker<P2_STAGES> dsf_ph;
    EmptyPhaseTracker<P2_A_STAGES> ae_ph;
    PhaseTracker<1> done_ph;
    __nv_bfloat16* sOUT = reinterpret_cast<__nv_bfloat16*>(base + P2_SMEM_OUT);
    int item = P2_USE_CLC ? (int)blockIdx.x : 2 * (int)blockIdx.x;
    while (item >= 0) {
      const int item_b = src.next(item);
      const int my_item = h_item ? item_b : item;
      const int am = my_item >= 0 ? item_base + my_item : 0;
      const int bhm = am / nq, qbm = am % nq;
      for (int j = 0; j < steps2; ++j) {
        const int st = dsf_ph.get_stage();
        mbarrier_wait_parity_suspend(smem_ptr_u32(&bars[P2B_DS_FULL0 + st]),
                                     dsf_ph.get_phase());
        dsf_ph.advance();
        const int ab = ae_ph.get_stage();
        mbarrier_wait_parity_suspend(smem_ptr_u32(&bars[P2B_A_EMPTY0 + ab]),
                                     ae_ph.get_phase());
        ae_ph.advance();
        // Transpose-read the dS^T tile ([64 kv x 64 q], xor-swizzled):
        // element (kv, q) at kv*64 + (((q>>3) ^ (kv&7))*8 + (q&7)); pack
        // kv pairs and STTM 32 words (= rank p) at A' cols [32p, +32).
        const __nv_bfloat16* dsl = reinterpret_cast<const __nv_bfloat16*>(
            base + P2_SMEM_DS + st * P2_DS_SLOT +
            h_item * P2_RANKS * 8192);
        const int q_hi = q >> 3, q_lo = q & 7;
        const int rem = min(topk - P2_RANKS * j, P2_RANKS);
        for (int p = 0; p < rem; ++p) {
          const __nv_bfloat16* tile = dsl + p * 4096;
          uint32_t pk[32];
          #pragma unroll
          for (int kv2 = 0; kv2 < 32; ++kv2) {
            const int k0 = 2 * kv2, k1 = 2 * kv2 + 1;
            const uint16_t lo = *reinterpret_cast<const uint16_t*>(
                tile + k0 * 64 + ((q_hi ^ (k0 & 7)) * 8 + q_lo));
            const uint16_t hi = *reinterpret_cast<const uint16_t*>(
                tile + k1 * 64 + ((q_hi ^ (k1 & 7)) * 8 + q_lo));
            pk[kv2] = (uint32_t)lo | ((uint32_t)hi << 16);
          }
          tcgen05_st_32x32b_x32(
              tmem_base + T_A0 +
                  (uint32_t)(ab * P2_RANKS * 32 + p * 32) + lane_base,
              pk);
        }
        if (elect_one_sync())
          mbarrier_arrive(smem_ptr_u32(&bars[P2B_DS_EMPTY0 + st]));
        tcgen05_wait_st();
        tcgen05_fence_before_thread_sync();
#ifdef PV_DUMP_A
        // DEBUG: read A' back and dump raw packed words into dq (clobbers
        // dq; item 0 pair 0 only). Row l = warp*32+lane, 32 words.
        if (my_item == 0 && j == 0) {
          uint32_t rda[32];
          tcgen05_ld_32x32b_x32(
              tmem_base + T_A0 + (uint32_t)(ab * P2_RANKS * 32) + lane_base,
              rda);
          tcgen05_fence_before_thread_sync();
          for (int w = 0; w < 32; ++w)
            reinterpret_cast<uint32_t*>(dq_dbg)[(warp_id * 32 + lane) * 32 + w] = rda[w];
        }
#endif
        if (elect_one_sync())
          mbarrier_arrive(smem_ptr_u32(&bars[P2B_A_FULL0 + ab]));
      }
      // Epilogue: the 16KB sOUT image holds one 64-column half for each
      // item. All four warps stage the same dimension half concurrently;
      // two leaders issue the two item stores, then the image is reused for
      // the other dimension half. This preserves two-CTA residency without
      // serializing item 0 and item 1.
      mbarrier_wait_parity_suspend(smem_ptr_u32(&bars[P2B_DQ_DONE0]),
                                   done_ph.get_phase());
      done_ph.advance();
      #pragma unroll
      for (int ssub = 0; ssub < 2; ++ssub) {
        if (my_item >= 0) {
          #pragma unroll
          for (int csub = 0; csub < SUB; csub += 32) {
            const int c0 = ssub * SUB + csub;
            uint32_t regs[32];
            tcgen05_ld_32x32b_x32(tmem_base + T_D0 + (uint32_t)c0 + lane_base,
                                  regs);
            tcgen05_fence_before_thread_sync();
            const float* f = reinterpret_cast<const float*>(regs);
#ifdef PV_DIRECT_STORE
            __nv_bfloat16* dqp = dq_dbg + ((size_t)(qbm * KV_ROWS + q) * num_heads + bhm) * HEAD_DIM;
            for (int c = 0; c < 32; ++c)
              dqp[c0 + c] = __float2bfloat16(f[c] * sm_scale);
            continue;
#endif
            #pragma unroll
            for (int v4 = 0; v4 < 4; ++v4) {
              uint4 packed;
              uint32_t* pw = reinterpret_cast<uint32_t*>(&packed);
              #pragma unroll
              for (int w = 0; w < 4; ++w)
                pw[w] = cvt_f32x2_to_bf16x2(f[v4 * 8 + 2 * w] * sm_scale,
                                            f[v4 * 8 + 2 * w + 1] * sm_scale);
              const int v = csub / 8 + v4;
              *reinterpret_cast<uint4*>(
                  &sOUT[h_item * (KV_ROWS * SUB) + q * SUB +
                        ((v ^ (q & 7)) * 8)]) = packed;
            }
          }
        }
        fence_proxy_async_shared_cta();
        bar_sync<9>(128);
#ifndef PV_DUMP_A
        if ((warp_id & 1) == 0 && my_item >= 0 && elect_one_sync()) {
          tma_store_3d(&tmap_dq, 0, qbm * KV_ROWS, bhm * 2 + ssub,
                       smem_ptr_u32(sOUT + (size_t)h_item * KV_ROWS * SUB));
          cp_async_bulk_commit_group();
          cp_async_bulk_wait_group_read<0>();
        }
#endif
        bar_sync<9>(128);
      }
      mbarrier_arrive(smem_ptr_u32(&bars[P2B_DQ_TFREE0]));
      item = (item_b >= 0) ? src.next(item_b) : -1;
    }
    // All dQ stores issued: the next pass1 may launch (its wait still covers their completion).
    if constexpr (KERNEL_PDL) {
      if (warp_id == 0 && elect_one_sync()) griddepcontrol_launch_dependents();
    }
    bar_sync<9>(128);
    if (warp_id == 0) tcgen05_dealloc<1>(tmem_base, 256);
    return;
  }
}

// Short-sequence pass2.  Its single-item plain-m128 path
// avoids the fixed item-pairing/K^T cost that dominates below S=32768.
namespace short_p2 {
// ---------------------------------------------------------------------------
// Pass 2: q-stationary dQ. One item per (bh, q64 block): walk the block's
// OWN topk list in steps of P2_QB kv64 blocks; per step, one bulk load of
// the rank-contiguous dS^T tiles + a per-block K gather; dQ(64,128) +=
// dS @ K as plain m128 ta/tb atoms (A = stored
// dS^T image, ta=1, with A rows 64-127 contracting the NEXT tile -- never
// read back; B = K MN-major, tb=1). dQ accumulates in TMEM across the
// whole list (double-buffered per item), epilogue scales by sm_scale and
// stores bf16 dq directly. 5 warps: 0-1 epilogue (warpgroup ranks 0-1 --
// tcgen05.ld lane bands 0-63, where the m128 D's real rows live), 2 load,
// 3 mma, 4 sched. The 4-stage x 1-block ring keeps SMEM small enough
// for 2 CTAs/SM (two chains per SM).
// ---------------------------------------------------------------------------
constexpr int P2_WARPS = 5;
constexpr int P2_CLC_ARRIVALS = 5;      // worker warps 0-3 + the sched fetch
constexpr int P2_STAGES   = 4;          // ring depth: hide the ~1100-cyc
//                                         refill behind 3 prefetched steps
constexpr int P2_QB       = 1;          // kv blocks per step
constexpr int P2_DS_SLOT  = P2_QB * 8 * 1024;
constexpr int P2_K_SLOT   = P2_QB * 16 * 1024;
constexpr int P2_SMEM_DS  = 0;
constexpr int P2_SMEM_K   = P2_SMEM_DS + P2_STAGES * P2_DS_SLOT;
constexpr int P2_SMEM_OUT = P2_SMEM_K + P2_STAGES * P2_K_SLOT;
constexpr int P2_SMEM_BAR = P2_SMEM_OUT + 16 * 1024;
enum {
  P2B_DS_FULL0 = 0,          // +stage
  P2B_DS_EMPTY0 = P2B_DS_FULL0 + P2_STAGES,
  P2B_K_FULL0 = P2B_DS_EMPTY0 + P2_STAGES,
  P2B_K_EMPTY0 = P2B_K_FULL0 + P2_STAGES,
  P2B_DQ_DONE0 = P2B_K_EMPTY0 + P2_STAGES,  // +buf: mma committed the atoms
  P2B_DQ_TFREE0 = P2B_DQ_DONE0 + 2, // +buf: epi t2r'd the buffer (64 arrivals)
  P2B_CLC_FULL0 = P2B_DQ_TFREE0 + 2, // +stage
  P2B_CLC_EMPTY0 = P2B_CLC_FULL0 + CLC_STAGES,
};
constexpr int P2_NUM_BARS = 4 * P2_STAGES + 4 + 2 * CLC_STAGES;
constexpr int P2_SMEM_TOTAL = P2_SMEM_BAR + P2_NUM_BARS * 8 + CLC_STAGES * 16 + 48;

struct P2ItemSource {
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

__global__ void __launch_bounds__(P2_WARPS * 32, 2)
vsa_bwd_pass2_kernel(const __grid_constant__ CUtensorMap tmap_k64,
                     const __grid_constant__ CUtensorMap tmap_dq,
                     const __nv_bfloat16* __restrict__ ds_buf,
                     const int* __restrict__ q2k_idx,
                     int num_heads, int seqlen, int topk,
                     int item_base, int item_count, int ds_head_base, float sm_scale) {
  uint8_t* base = bwd_smem;
  uint64_t* bars = reinterpret_cast<uint64_t*>(base + P2_SMEM_BAR);
  uint32_t* clc_response = reinterpret_cast<uint32_t*>(
      (reinterpret_cast<uintptr_t>(bars + P2_NUM_BARS) + 15u) & ~uintptr_t(15u));
  uint32_t* tmem_slot = clc_response + CLC_STAGES * 4;
  uint64_t* clc_full  = &bars[P2B_CLC_FULL0];
  uint64_t* clc_empty = &bars[P2B_CLC_EMPTY0];
  const int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;
  const int nq = seqlen / KV_ROWS;
  const int total = item_count;
#ifdef P2_ONLY1
  const int quads = 1;
  const int topk_eff = 1;
#else
  const int quads = (topk + P2_QB - 1) / P2_QB;
  const int topk_eff = topk;
#endif

  if (warp_id == 0) {
    tcgen05_alloc<1>(smem_ptr_u32(tmem_slot), 256);
    tcgen05_relinquish_alloc_permit<1>();
  }
  if (tid == 0) {
    #pragma unroll
    for (int st = 0; st < P2_STAGES; ++st) {
      mbarrier_init(smem_ptr_u32(&bars[P2B_DS_FULL0 + st]),  1);
      mbarrier_init(smem_ptr_u32(&bars[P2B_DS_EMPTY0 + st]), 1);
      mbarrier_init(smem_ptr_u32(&bars[P2B_K_FULL0 + st]),   1);
      mbarrier_init(smem_ptr_u32(&bars[P2B_K_EMPTY0 + st]),  1);
    }
    mbarrier_init(smem_ptr_u32(&bars[P2B_DQ_DONE0]),  1);
    mbarrier_init(smem_ptr_u32(&bars[P2B_DQ_DONE0 + 1]), 1);
    mbarrier_init(smem_ptr_u32(&bars[P2B_DQ_TFREE0]), 64);
    mbarrier_init(smem_ptr_u32(&bars[P2B_DQ_TFREE0 + 1]), 64);
    if constexpr (USE_CLC) {
      #pragma unroll
      for (int st = 0; st < CLC_STAGES; ++st) {
        mbarrier_init(smem_ptr_u32(&clc_full[st]), 1);
        mbarrier_init(smem_ptr_u32(&clc_empty[st]), P2_CLC_ARRIVALS);
      }
      #pragma unroll
      for (int i = 0; i < CLC_STAGES * 4; ++i) clc_response[i] = 0;
    }
  }
  fence_mbarrier_init_release_cluster();
  __syncthreads();
  if constexpr (KERNEL_PDL) griddepcontrol_wait();  // pass1's dS tiles
  const uint32_t tmem_base = *tmem_slot;

  if (warp_id == P2_WARPS - 1) {
    // Scheduler warp (same producer/consumer shape as pass 1).
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
      #pragma unroll
      for (int st = 0; st < CLC_STAGES; ++st) {
        if (lane == 0)
          mbarrier_wait_parity_suspend(smem_ptr_u32(&clc_empty[prod_stage]), prod_phase);
        __syncwarp();
        advance_stage_phase<CLC_STAGES>(prod_stage, prod_phase);
      }
    }
    return;
  }
  P2ItemSource src{clc_full, clc_empty, clc_response, total};

  if (warp_id == 2) {
    // Load warp: per step, one 8KB dS tile bulk load + a 16KB K-block TMA.
    EmptyPhaseTracker<P2_STAGES> ds_ph, k_ph;
    for (int item = (int)blockIdx.x; item >= 0; item = src.next(item)) {
      const int aitem = item_base + item;
      const int bh = aitem / nq;
      for (int j = 0; j < quads; ++j) {
        const int st = ds_ph.get_stage();
        const int rem = min(topk_eff - P2_QB * j, P2_QB);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&bars[P2B_DS_EMPTY0 + st]),
                                     ds_ph.get_phase());
        ds_ph.advance();
        if (elect_one_sync()) {
          mbarrier_arrive_expect_tx(smem_ptr_u32(&bars[P2B_DS_FULL0 + st]),
                                    rem * 8192);
          bulk_g2s(smem_ptr_u32(base + P2_SMEM_DS + st * P2_DS_SLOT),
                   ds_buf + ((size_t)(item_base + item - ds_head_base * nq) * topk + P2_QB * j) * 4096,
                   rem * 8192, smem_ptr_u32(&bars[P2B_DS_FULL0 + st]));
        }
        mbarrier_wait_parity_suspend(smem_ptr_u32(&bars[P2B_K_EMPTY0 + st]),
                                     k_ph.get_phase());
        k_ph.advance();
        if (elect_one_sync())
          mbarrier_arrive_expect_tx(smem_ptr_u32(&bars[P2B_K_FULL0 + st]),
                                    rem * 16384);
        __syncwarp();
        if (lane < 2 * rem) {
          const int b = lane >> 1, u = lane & 1;
          const int kvb = q2k_idx[(size_t)(item_base + item) * topk + P2_QB * j + b];
          tma_load_3d(smem_ptr_u32(base + P2_SMEM_K + st * P2_K_SLOT
                                   + b * 16384 + u * 8192),
                      &tmap_k64, smem_ptr_u32(&bars[P2B_K_FULL0 + st]),
                      0, kvb * KV_ROWS, bh * 2 + u);
        }
      }
    }
    return;
  }
  if (warp_id == 3) {
    // MMA warp: dQ(item) += sum_j sum_b dS_b @ K_b, double-buffered D.
    const uint32_t lead = elect_one_sync() ? 1u : 0u;
    constexpr uint32_t K16_MN = (uint32_t)((16 * SUB * 2) >> 4);
    // dQ form (Layout D, M=128): rows 64-127 contract the
    // bytes past the 8KB slot (the next stage's tile, or the K region at
    // the last stage) -- garbage, never read back: epi t2rs lanes 0-63.
    const uint32_t idesc_dq = make_idesc_bf16_f32(128, HEAD_DIM, true, true);
    PhaseTracker<P2_STAGES> dsf_ph, kf_ph;
    EmptyPhaseTracker<2> tfree_ph;
    int pit = 0;
    for (int item = (int)blockIdx.x; item >= 0; item = src.next(item)) {
      const int buf = tfree_ph.get_stage();
      mbarrier_wait_parity(smem_ptr_u32(&bars[P2B_DQ_TFREE0 + buf]),
                           tfree_ph.get_phase());
      tfree_ph.advance();
      bool first = true;
      for (int j = 0; j < quads; ++j) {
        const int st = dsf_ph.get_stage();
        const int rem = min(topk_eff - P2_QB * j, P2_QB);
        mbarrier_wait_parity(smem_ptr_u32(&bars[P2B_DS_FULL0 + st]), dsf_ph.get_phase());
        dsf_ph.advance();
        mbarrier_wait_parity(smem_ptr_u32(&bars[P2B_K_FULL0 + st]), kf_ph.get_phase());
        kf_ph.advance();
#ifdef P2_ONLY1
        // Debug: dump the actual SMEM operands into unused ds_buf slots
        // (item 0's ranks 1, 2, 3) for host-side comparison.
        if (item == 0 && j == 0) {
          __nv_bfloat16* dbg = const_cast<__nv_bfloat16*>(ds_buf);
          const __nv_bfloat16* sds = reinterpret_cast<const __nv_bfloat16*>(base + P2_SMEM_DS + st * P2_DS_SLOT);
          const __nv_bfloat16* skk = reinterpret_cast<const __nv_bfloat16*>(base + P2_SMEM_K + st * P2_K_SLOT);
          for (int i = lane; i < 4096; i += 32) dbg[1 * 4096 + i] = sds[i];
          for (int i = lane; i < 4096; i += 32) dbg[2 * 4096 + i] = skk[i];
          for (int i = lane; i < 4096; i += 32) dbg[3 * 4096 + i] = skk[4096 + i];
          __syncwarp();
        }
#endif
        for (int b = 0; b < rem; ++b) {
          SmemDescPair da; da.u64 = build_smem_desc_blackwell(
              smem_ptr_u32(base + P2_SMEM_DS + st * P2_DS_SLOT + b * 8192),
              1024u, 8192u, SmemSwizzleBlackwell::B128);
          SmemDescPair db; db.u64 = build_smem_desc_blackwell(
              smem_ptr_u32(base + P2_SMEM_K + st * P2_K_SLOT + b * 16384),
              1024u, 8192u, SmemSwizzleBlackwell::B128);
          #pragma unroll
          for (int ki = 0; ki < 4; ++ki) {
            tcgen05_mma_f16_ss_lead1(lead, tmem_base + (uint32_t)(buf * 128),
                                     da.u64, db.u64, idesc_dq, !first);
            first = false;
            desc_add_lo(da, K16_MN);
            desc_add_lo(db, K16_MN);
          }
        }
        tcgen05_commit1_lead(lead, smem_ptr_u32(&bars[P2B_DS_EMPTY0 + st]));
        tcgen05_commit1_lead(lead, smem_ptr_u32(&bars[P2B_K_EMPTY0 + st]));
      }
      tcgen05_commit1_lead(lead, smem_ptr_u32(&bars[P2B_DQ_DONE0 + buf]));
      ++pit;
    }
    return;
  }
  if (warp_id <= 1) {
    // Epilogue warps (0,1 = warpgroup ranks 0,1 -> TMEM lane bands 0-31 /
    // 32-63): t2r the finished dQ buffer, scale, pack bf16 into the
    // swizzled out image, TMA-store, release the buffer.
    PhaseTracker<2> done_ph;
    __nv_bfloat16* sOUT = reinterpret_cast<__nv_bfloat16*>(base + P2_SMEM_OUT);
    const uint32_t lane_base = (uint32_t)((warp_id * 32) << 16);
    const int r = warp_id * 32 + lane;               // q row 0-63
    for (int item = (int)blockIdx.x; item >= 0; item = src.next(item)) {
      const int aitem = item_base + item;
      const int bh = aitem / nq;
      const int qb = aitem % nq;
      const int buf = done_ph.get_stage();
      mbarrier_wait_parity_suspend(smem_ptr_u32(&bars[P2B_DQ_DONE0 + buf]),
                                   done_ph.get_phase());
      done_ph.advance();
      #pragma unroll
      for (int c0 = 0; c0 < HEAD_DIM; c0 += 32) {
        uint32_t regs[32];
        tcgen05_ld_32x32b_x32(tmem_base + (uint32_t)(buf * 128 + c0) + lane_base, regs);
        tcgen05_fence_before_thread_sync();
        const float* f = reinterpret_cast<const float*>(regs);
        #pragma unroll
        for (int v4 = 0; v4 < 4; ++v4) {
          uint4 packed;
          uint32_t* pw = reinterpret_cast<uint32_t*>(&packed);
          #pragma unroll
          for (int w = 0; w < 4; ++w)
            pw[w] = cvt_f32x2_to_bf16x2(f[v4 * 8 + 2 * w] * sm_scale,
                                        f[v4 * 8 + 2 * w + 1] * sm_scale);
          const int v = c0 / 8 + v4;
          const int ssub = v >> 3, vv = v & 7;
          *reinterpret_cast<uint4*>(
              &sOUT[ssub * (KV_ROWS * SUB) + r * SUB + ((vv ^ (r & 7)) * 8)]) = packed;
        }
      }
      fence_proxy_async_shared_cta();
      bar_sync<9>(64);
      mbarrier_arrive(smem_ptr_u32(&bars[P2B_DQ_TFREE0 + buf]));
      if (warp_id == 0 && elect_one_sync()) {
        #pragma unroll
        for (int ssub = 0; ssub < 2; ++ssub)
          tma_store_3d(&tmap_dq, 0, qb * KV_ROWS, bh * 2 + ssub,
                       smem_ptr_u32(sOUT + (size_t)ssub * KV_ROWS * SUB));
        cp_async_bulk_commit_group();
        cp_async_bulk_wait_group_read<0>();
      }
      bar_sync<9>(64);
    }
    // All dQ stores issued: the next pass1 may launch (its wait still covers their completion).
    if constexpr (KERNEL_PDL) {
      if (warp_id == 0 && elect_one_sync()) griddepcontrol_launch_dependents();
    }
    bar_sync<9>(64);
    if (warp_id == 0) tcgen05_dealloc<1>(tmem_base, 256);
    return;
  }
  return;
}
}  // namespace short_p2

// ---------------------------------------------------------------------------
// Host launchers (stream-chained: pre -> pass 1 -> pass 2).
// ---------------------------------------------------------------------------

struct VsaBwdArgs {
  const __nv_bfloat16 *q, *k, *v, *dout;
  __nv_bfloat16* ds_buf;              // [BH*nq*topk] 8KB dS^T tiles (pass1 -> pass2)
  const int* q2k_idx;                 // [BH*nq, topk] forward selection lists
  int topk;
  __nv_bfloat16 *dk, *dv, *dq;        // [S, H, 128] bf16 (token-major, B folded)
  __nv_bfloat16 *qt, *dot, *kt;       // [H*128, S] bf16 scratch (pre writes them)
  const float *m_rows, *delta_rows;   // [BH, S]
  const int* pair_offset;             // [BH*nb64 + 1]
  const unsigned* pair_union;         // plain q64 ids
  int num_heads, seqlen, num_blocks;  // num_blocks = kv blocks (S/64)
  float sm_scale;
};

inline cudaError_t vsa_bwd_encode_bf16_3d64(CUtensorMap* map, const __nv_bfloat16* ptr,
                                            int H, long n_tokens) {
  uint64_t gd[3] = { (uint64_t)SUB, (uint64_t)n_tokens, (uint64_t)H * (HEAD_DIM / SUB) };
  uint64_t gs[2] = { (uint64_t)H * HEAD_DIM * 2, (uint64_t)SUB * 2 };
  uint32_t bd[3] = { (uint32_t)SUB, (uint32_t)KV_ROWS, 1u };
  uint32_t es[3] = { 1u, 1u, 1u };
  if (cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3,
                             const_cast<__nv_bfloat16*>(ptr), gd, gs, bd, es,
                             CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                             CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
    return cudaErrorInvalidValue;
  return cudaSuccess;
}

inline cudaError_t vsa_bwd_encode_t_2d(CUtensorMap* map, const __nv_bfloat16* ptr,
                                       int H, long n_tokens,
                                       bool p2_k_map = false) {
  uint64_t gd[2] = { (uint64_t)n_tokens, (uint64_t)H * HEAD_DIM };
  uint64_t gs[1] = { (uint64_t)n_tokens * 2 };
  uint32_t bd[2] = { (uint32_t)KV_ROWS, (uint32_t)HEAD_DIM };
  uint32_t es[2] = { 1u, 1u };
#ifdef P2_L2_PROMOTE_256
  const CUtensorMapL2promotion l2_promotion = p2_k_map
      ? CU_TENSOR_MAP_L2_PROMOTION_L2_256B
      : CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
#elif defined(P2_K_L2_NONE)
  const CUtensorMapL2promotion l2_promotion = p2_k_map
      ? CU_TENSOR_MAP_L2_PROMOTION_NONE
      : CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
#else
  constexpr CUtensorMapL2promotion l2_promotion =
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
#endif
  if (cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
                             const_cast<__nv_bfloat16*>(ptr), gd, gs, bd, es,
                             CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                             l2_promotion,
                             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
    return cudaErrorInvalidValue;
  return cudaSuccess;
}

// Every launch goes through cudaLaunchKernelEx: the cluster attribute (CLC needs it) and, with
// KERNEL_PDL, the programmatic-stream-serialization attribute (the kernels' griddepcontrol.wait
// guards the data).
inline cudaLaunchConfig_t launch_config(dim3 grid, dim3 block, size_t smem, cudaStream_t stream,
                                        cudaLaunchAttribute* at) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim            = grid;
  cfg.blockDim           = block;
  cfg.dynamicSmemBytes   = smem;
  cfg.stream             = stream;
  at[0].id               = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = 1;
  at[0].val.clusterDim.y = 1;
  at[0].val.clusterDim.z = 1;
  at[1].id               = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[1].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs              = at;
  cfg.numAttrs           = KERNEL_PDL ? 2 : 1;
  return cfg;
}

inline cudaError_t launch_vsa_bwd_sm100a(const VsaBwdArgs& a, cudaStream_t stream,
                                         int item_base = 0, int item_count = -1, int ds_head_base = 0) {
  const int H = a.num_heads, S = a.seqlen;
  const long n_tokens = (long)S;

  static CUtensorMap tk_, tv_, tqt_, tdot_, tdk_, tdv_;
  static const void* cached_q = nullptr;
  static int cached_S = 0, cached_H = 0;
  if (cached_q != (const void*)a.q || cached_S != S || cached_H != H) {
    if (vsa_bwd_encode_bf16_3d64(&tk_, a.k, H, n_tokens) != cudaSuccess) return cudaErrorInvalidValue;
    if (vsa_bwd_encode_bf16_3d64(&tv_, a.v, H, n_tokens) != cudaSuccess) return cudaErrorInvalidValue;
    if (vsa_bwd_encode_t_2d(&tqt_, a.qt, H, n_tokens) != cudaSuccess) return cudaErrorInvalidValue;
    if (vsa_bwd_encode_t_2d(&tdot_, a.dot, H, n_tokens) != cudaSuccess) return cudaErrorInvalidValue;
    if (vsa_bwd_encode_bf16_3d64(&tdk_, a.dk, H, n_tokens) != cudaSuccess) return cudaErrorInvalidValue;
    if (vsa_bwd_encode_bf16_3d64(&tdv_, a.dv, H, n_tokens) != cudaSuccess) return cudaErrorInvalidValue;
    cached_q = (const void*)a.q; cached_S = S; cached_H = H;
  }
  static bool smem_set = false;
  if (!smem_set) {
    cudaError_t e = cudaFuncSetAttribute(vsa_bwd_main_kernel,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         SMEM_TOTAL);
    if (e != cudaSuccess) return e;
    smem_set = true;
  }
  const float scale_log2 = a.sm_scale * 1.4426950408889634f;
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  if (item_count < 0) item_count = H * a.num_blocks;
  const int total = item_count;
  const int grid = USE_CLC ? total : (total < sms ? total : sms);
  cudaLaunchAttribute at[2];
  cudaLaunchConfig_t cfg = launch_config(dim3((unsigned)grid, 1, 1), dim3(N_WARPS * 32, 1, 1),
                                         SMEM_TOTAL, stream, at);
  return cudaLaunchKernelEx(&cfg, vsa_bwd_main_kernel,
      tk_, tv_, tqt_, tdot_, tdk_, tdv_,
      a.ds_buf, a.m_rows, a.delta_rows,
      a.pair_offset, a.pair_union,
      H, S, a.num_blocks, a.topk, item_base, item_count, ds_head_base,
      scale_log2, a.sm_scale);
}

inline cudaError_t launch_vsa_bwd_pass2(const VsaBwdArgs& a, cudaStream_t stream,
                                        int item_base = 0, int item_count = -1, int ds_head_base = 0) {
  const int H = a.num_heads, S = a.seqlen;
  static CUtensorMap tk2_, tdq_;
  static const void* cached_k = nullptr;
  static int cached_S = 0;
  if (cached_k != (const void*)a.kt || cached_S != S) {
    if (vsa_bwd_encode_t_2d(&tk2_, a.kt, H, S, true) != cudaSuccess) return cudaErrorInvalidValue;
    if (vsa_bwd_encode_bf16_3d64(&tdq_, a.dq, H, S) != cudaSuccess) return cudaErrorInvalidValue;
    cached_k = (const void*)a.kt; cached_S = S;
  }
  static bool smem_set2 = false;
  if (!smem_set2) {
    cudaError_t e = cudaFuncSetAttribute(vsa_bwd_pass2_kernel,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         P2_SMEM_TOTAL);
    if (e != cudaSuccess) return e;
    smem_set2 = true;
  }
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  int smem_per_sm = 0;
  cudaDeviceGetAttribute(&smem_per_sm,
                         cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0);
  if (item_count < 0) item_count = H * (S / KV_ROWS);
  const int total = item_count;
  const int pair_total = (total + 1) / 2;
  const int ctas_per_sm = std::min(2, smem_per_sm / P2_SMEM_TOTAL);
  const int resident_ctas = ctas_per_sm * sms;
  const int grid = P2_USE_CLC ? total
                              : (pair_total < resident_ctas
                                     ? pair_total : resident_ctas);
  cudaLaunchAttribute at[2];
  cudaLaunchConfig_t cfg = launch_config(dim3((unsigned)grid, 1, 1), dim3(P2_WARPS * 32, 1, 1),
                                         P2_SMEM_TOTAL, stream, at);
  return cudaLaunchKernelEx(&cfg, vsa_bwd_pass2_kernel,
      tk2_, tdq_, (const __nv_bfloat16*)a.ds_buf, a.q2k_idx, a.dq,
      H, S, a.topk, item_base, item_count, ds_head_base, a.sm_scale);
}

inline cudaError_t launch_vsa_bwd_pass2_short(
    const VsaBwdArgs& a, cudaStream_t stream,
    int item_base = 0, int item_count = -1, int ds_head_base = 0) {
  const int H = a.num_heads, S = a.seqlen;
  static CUtensorMap tk_, tdq_;
  static const void* cached_k = nullptr;
  static int cached_S = 0, cached_H = 0;
  if (cached_k != (const void*)a.k || cached_S != S || cached_H != H) {
    if (vsa_bwd_encode_bf16_3d64(&tk_, a.k, H, S) != cudaSuccess)
      return cudaErrorInvalidValue;
    if (vsa_bwd_encode_bf16_3d64(&tdq_, a.dq, H, S) != cudaSuccess)
      return cudaErrorInvalidValue;
    cached_k = (const void*)a.k;
    cached_S = S;
    cached_H = H;
  }
  static bool smem_set = false;
  if (!smem_set) {
    cudaError_t e = cudaFuncSetAttribute(
        short_p2::vsa_bwd_pass2_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        short_p2::P2_SMEM_TOTAL);
    if (e != cudaSuccess) return e;
    smem_set = true;
  }
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  if (item_count < 0) item_count = H * (S / KV_ROWS);
  const int total = item_count;
  const int grid = USE_CLC ? total : std::min(total, sms);
  cudaLaunchAttribute at[2];
  cudaLaunchConfig_t cfg = launch_config(dim3((unsigned)grid, 1, 1),
                                         dim3(short_p2::P2_WARPS * 32, 1, 1),
                                         short_p2::P2_SMEM_TOTAL, stream, at);
  return cudaLaunchKernelEx(
      &cfg, short_p2::vsa_bwd_pass2_kernel,
      tk_, tdq_, (const __nv_bfloat16*)a.ds_buf, a.q2k_idx,
      H, S, a.topk, item_base, item_count, ds_head_base, a.sm_scale);
}

inline cudaError_t launch_vsa_bwd_preprocess(const __nv_bfloat16* q, const __nv_bfloat16* o,
                                             const __nv_bfloat16* dout, const __nv_bfloat16* k,
                                             float* delta_rows, float* dqaccum,
                                             __nv_bfloat16* qt, __nv_bfloat16* dot,
                                             __nv_bfloat16* kt,
                                             int num_heads, int seqlen, cudaStream_t stream) {
  cudaLaunchAttribute at[2];
  cudaLaunchConfig_t cfg = launch_config(dim3((unsigned)(seqlen / 128), (unsigned)num_heads, 1),
                                         dim3(256, 1, 1), 0, stream, at);
  return kt != nullptr
      ? cudaLaunchKernelEx(&cfg, vsa_bwd_preprocess_kernel<true>, q, o, dout, k, delta_rows,
                           dqaccum, qt, dot, kt, num_heads, seqlen)
      : cudaLaunchKernelEx(&cfg, vsa_bwd_preprocess_kernel<false>, q, o, dout, k, delta_rows,
                           dqaccum, qt, dot, kt, num_heads, seqlen);
}

}  // namespace vsa_bwd_blk64

// ---------------------------------------------------------------------------
// Bench harness (CPU reference + verify + timing).
// ---------------------------------------------------------------------------

// Harness-side sparse block size (BLOCK env); the GPU path requires 64.
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
// sorted ascending (built by ascending q-block walk); feeds build_lists
// -> the kernel's pair_offset/pair_union.
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
      // rank (position in the q-block's own list) rides in bits 16+ so
      // pass 1 can address the q-block's ds_buf slot directly.
      inv.q_blocks[cursor[gkb]++] = mtile | (i << 16);
    }
  }
  return inv;
}

// blk128-native lists: one work item per kv block; entries = PLAIN q-block
// ids (the aligned kernel has no masking -- tile == block).
struct PairUnion {
  std::vector<int> offset;          // [B*H*nb + 1]
  std::vector<unsigned> entries;    // q-block ids
};

static PairUnion build_lists(const KvToQ& k2q, int B, int H, int nb) {
  const int items = B * H * nb;
  PairUnion pu;
  pu.offset.assign(items + 1, 0);
  for (int it = 0; it < items; ++it) {
    for (int i = k2q.offset[it]; i < k2q.offset[it + 1]; ++i)
      pu.entries.push_back((unsigned)k2q.q_blocks[i]);
    pu.offset[it + 1] = (int)pu.entries.size();
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
      const int mtile = k2q.q_blocks[qi_list] & 0xFFFF;
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

  printf("  [%-9s H%-2d num_blocks%-3d topk%-3d S%d blk%d] N_q=%ld\n",
         sh.lab, H, num_blocks, topk, S, BLOCK, tq);

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
  const PairUnion pu = build_lists(k2q, B, H, num_blocks);
  {
    long usum = pu.entries.size();
    printf("  q-lists: items=%d entries=%ld steps/item=%.1f\n",
           B * H * num_blocks, usum, (double)usum / (B * H * num_blocks));
  }
  {
    int cmin = INT_MAX, cmax = 0, zeros = 0;
    long csum = 0;
    for (int c : k2q.count) { cmin = std::min(cmin, c); cmax = std::max(cmax, c); csum += c; if (c == 0) ++zeros; }
    // kv-block count == q-block count here (square S x S block grid).
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

  // GPU backward, two-pass: preprocess (Delta + Q^T/dO^T/K^T slabs) ->
  // pass1 (dK/dV + dS tiles) -> pass2 (dQ), head-chunked across 2 streams.
  if (BLOCK != 64 || B != 1) {
    printf("  gpu: skipped (requires BLOCK=64 and B=1)\n");
    return;
  }
  {
    using namespace vsa_bwd_blk64;
    const long elems = tq * H * hd;
    __nv_bfloat16 *dQg, *dKg, *dVg, *dDOg, *dOg, *dDKout, *dDVout, *dDQout;
    float *dMg, *dDeltag;
    __nv_bfloat16* dDS;
    int *dOff, *dBlk, *dQ2K;
    CUDA_CHECK(cudaMalloc(&dQg,  elems * 2));
    CUDA_CHECK(cudaMalloc(&dKg,  elems * 2));
    CUDA_CHECK(cudaMalloc(&dVg,  elems * 2));
    CUDA_CHECK(cudaMalloc(&dDOg, elems * 2));
    CUDA_CHECK(cudaMalloc(&dOg,  elems * 2));
    CUDA_CHECK(cudaMalloc(&dDKout, elems * 2));
    CUDA_CHECK(cudaMalloc(&dDVout, elems * 2));
    CUDA_CHECK(cudaMalloc(&dDQout, elems * 2));
    CUDA_CHECK(cudaMalloc(&dMg,     (size_t)B * H * S * 4));
    CUDA_CHECK(cudaMalloc(&dDeltag, (size_t)B * H * S * 4));
    // dS tiles live only from pass1 to pass2 of the same head chunk, so the buffer is sized for
    // one chunk: the fewest chunks whose buffer the device can allocate, except one head per chunk
    // from 16384 items on (measured +2..3% at 131k / 262k, -5..-40% below 65k). P2_CHUNKS overrides.
    const size_t ds_bytes_per_head = (size_t)num_blocks * topk * 8192;
    int n_chunks = getenv("P2_CHUNKS") ? std::min(std::max(atoi(getenv("P2_CHUNKS")), 1), H)
                                       : (H * num_blocks >= 16384 ? H : 1);
    int heads_per_chunk = 0;
    size_t ds_bytes = 0;
    for (;; ++n_chunks) {
      heads_per_chunk = (H + n_chunks - 1) / n_chunks;
      ds_bytes        = (size_t)heads_per_chunk * ds_bytes_per_head;
      if (cudaMalloc(&dDS, ds_bytes) == cudaSuccess) break;
      (void)cudaGetLastError();
      if (heads_per_chunk == 1) {
        printf("  gpu: skipped (ds_buf alloc %.1f GB for one head failed)\n", ds_bytes / 1e9);
        return;
      }
    }
    n_chunks = (H + heads_per_chunk - 1) / heads_per_chunk;
    printf("  ds_buf: %d chunk(s) of %d head(s), %.1f GB\n", n_chunks, heads_per_chunk,
           ds_bytes / 1e9);
    CUDA_CHECK(cudaMalloc(&dQ2K, (size_t)B * H * num_blocks * max_kv * 4));
    CUDA_CHECK(cudaMemcpy(dQ2K, hq2k_idx.data(),
                          (size_t)B * H * num_blocks * max_kv * 4,
                          cudaMemcpyHostToDevice));
    __nv_bfloat16 *dQT, *dDOT, *dKT;
    CUDA_CHECK(cudaMalloc(&dQT,  elems * 2));
    CUDA_CHECK(cudaMalloc(&dDOT, elems * 2));
    CUDA_CHECK(cudaMalloc(&dKT,  elems * 2));
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
    args.ds_buf = dDS; args.q2k_idx = dQ2K; args.topk = topk;
    args.dk = dDKout; args.dv = dDVout; args.dq = dDQout;
    args.qt = dQT; args.dot = dDOT; args.kt = dKT;
    args.m_rows = dMg; args.delta_rows = dDeltag;
    args.pair_offset = dOff; args.pair_union = reinterpret_cast<const unsigned*>(dBlk);
    args.num_heads = H; args.seqlen = S; args.num_blocks = num_blocks;
    args.sm_scale = 1.0f / sqrtf((float)hd);

    bool any_zero_count = false;
    for (size_t i = 0; i + 1 < pu.offset.size(); ++i)
      if (pu.offset[i + 1] == pu.offset[i]) { any_zero_count = true; break; }
    // One stream: preprocess, then per head chunk pass1 (dK/dV + dS tiles into the chunk buffer)
    // followed by pass2 (dQ) reading it back; the next chunk reuses the buffer.
    const bool use_short_p2 = S < 32768;
    auto run_once = [&]() {
      if (any_zero_count) {
        CUDA_CHECK(cudaMemsetAsync(dDKout, 0, elems * 2, 0));
        CUDA_CHECK(cudaMemsetAsync(dDVout, 0, elems * 2, 0));
      }
      CUDA_CHECK(launch_vsa_bwd_preprocess(
          dQg, dOg, dDOg, use_short_p2 ? nullptr : dKg,
          dDeltag, nullptr, dQT, dDOT,
          use_short_p2 ? nullptr : dKT, H, S, 0));
      for (int c = 0; c < n_chunks; ++c) {
        const int h0 = c * heads_per_chunk, h1 = std::min(H, (c + 1) * heads_per_chunk);
        CUDA_CHECK(launch_vsa_bwd_sm100a(args, 0, h0 * num_blocks, (h1 - h0) * num_blocks, h0));
        if (!getenv("SKIP_P2")) {
          if (use_short_p2)
            CUDA_CHECK(launch_vsa_bwd_pass2_short(
                args, 0, h0 * (S / 64), (h1 - h0) * (S / 64), h0));
          else
            CUDA_CHECK(launch_vsa_bwd_pass2(
                args, 0, h0 * (S / 64), (h1 - h0) * (S / 64), h0));
        }
      }
    };
    run_once();
    CUDA_CHECK(cudaDeviceSynchronize());
    if (getenv("DUMP_DS") && n_chunks != 1) printf("  DUMP_DS skipped: chunked ds_buf\n");
    if (getenv("DUMP_DS") && n_chunks == 1) {
      const size_t n = (size_t)B * H * num_blocks * topk * 4096;
      std::vector<__nv_bfloat16> hds(n);
      CUDA_CHECK(cudaMemcpy(hds.data(), dDS, n * 2, cudaMemcpyDeviceToHost));
      std::vector<float> f(n);
      for (size_t i = 0; i < n; ++i) f[i] = __bfloat162float(hds[i]);
      npy_save_f32(std::string(getenv("DUMP_DS")), f.data(),
                   {(long)(B * H * num_blocks * topk), 4096L});
      printf("  dumped ds_buf\n");
    }

    if (run_cpu || getenv("DUMP_BWD_GPU")) {
      std::vector<float> gDelta((size_t)B * H * S);
      std::vector<__nv_bfloat16> gDK(elems), gDV(elems), gDQ(elems);
      CUDA_CHECK(cudaMemcpy(gDelta.data(), dDeltag, gDelta.size() * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(gDQ.data(), dDQout, elems * 2, cudaMemcpyDeviceToHost));
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
      // dq: pass2 already scaled + rounded to bf16 and stored token-major.
      std::vector<float> gdq(elems);
      for (long i = 0; i < elems; ++i) gdq[i] = __bfloat162float(gDQ[i]);
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
        printf("  dq argmax: t=%ld (blk %ld, row %ld) h=%d d=%d ref=%.6f got=%.6f\n",
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
      // dq gate 8e-3: the CPU-ref-vs-torch-fp32 noise floor from the
      // production bf16 quantization points is ~1.9-2.7e-3, GPU-vs-CPU can
      // legitimately reach ~2x that, and the
      // bf16-rounded dq output adds its own rounding on top.
      if (run_cpu) {
        const double rq_gate = 8e-3;
        const bool pass = dmax < 1e-4 && rq < rq_gate && rk < 8e-3 && rv < 8e-3;
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
          ok = memcmp(rDK.data(), gDK.data(), elems * 2) == 0 &&
               memcmp(rDV.data(), gDV.data(), elems * 2) == 0;
          double mq = 0;
          for (long i = 0; i < elems; ++i)
            mq = std::max(mq, fabs((double)__bfloat162float(rDQ[i]) - __bfloat162float(gDQ[i])));
          if (mq > 1e-2) ok = false;   // reduce-add order x bf16 rounding
        }
        printf("  stress x%d: %s\n", n, ok ? "OK (dk/dv bitwise, dq stable)" : "FAIL");
        if (!ok) exit(1);
      }
    }

    if (block_sparse_bwd_bf16_benchmark::enabled()) {
      const auto options = block_sparse_bwd_bf16_benchmark::options_from_env();
      const double ms = block_sparse_bwd_bf16_benchmark::measure(run_once, options);
      const double tflops = block_sparse_bwd_bf16_benchmark::tflops(hd, selected_pairs, ms);
      const int warm = options.warmup_iterations, iters = options.timed_iterations;
      printf("  gpu bwd: %.4f ms  %.1f TFLOPS (bwd 2.5x sel; pre+main+post)\n",
             ms, tflops);
      if (getenv("VSA_BENCH_CLI"))
        printf("VSA_BENCH tile=64 B=%d H=%d S=%d D=%d inputs=%s "
               "kv/q=%.3f edges=%lld warmup=%d iters=%d ms=%.6f "
               "sparse_tflops=%.3f\n",
               B, H, S, hd, load_npy ? "shared_npy" : "generated",
               (double)topk, (long long)B * H * num_blocks * topk,
               warm, iters, ms, tflops);
    }

#ifdef WARP_PROF
    {
      WpBuffer wp = wp_alloc(dim3((unsigned)(num_blocks * H), 1, 1));
      run_once();
      CUDA_CHECK(cudaDeviceSynchronize());
      wp_readback(wp);
      const char* roles[16] = {"red","red","red","red",
                               "cmp","cmp","cmp","cmp","cmp","cmp","cmp","cmp",
                               "mma","load","rly","emp"};
      printf("  WARP_PROF block %u:\n", wp.view_block);
      wp_print_busy(wp, roles, 16, wp.view_block);
      wp_dump_raw(wp, "warp_raw_vsa_bwd_blk64.bin.gz", wp.view_block, 2);
      wp_free(wp);
    }
#endif

    cudaFree(dQg); cudaFree(dKg); cudaFree(dVg); cudaFree(dDOg); cudaFree(dOg);
    cudaFree(dDKout); cudaFree(dDVout); cudaFree(dDQout); cudaFree(dMg); cudaFree(dDeltag);
    cudaFree(dDS); cudaFree(dQ2K); cudaFree(dQT); cudaFree(dDOT); cudaFree(dKT); cudaFree(dOff); cudaFree(dBlk);
  }
}

int main(int argc, char** argv) {
  if (const char* b = getenv("BLOCK")) BLOCK = atoi(b);
  CUDA_CHECK(cudaFree(0));

  if (argc > 1) {
    if (strcmp(argv[1], "--help") == 0) {
      printf("Usage: %s [--vsa-bench64 B S KV_PER_Q [ITERS]]\n", argv[0]);
      return 0;
    }
    if (strcmp(argv[1], "--vsa-bench64") != 0 ||
        (argc != 5 && argc != 6)) {
      fprintf(stderr,
              "Usage: %s [--vsa-bench64 B S KV_PER_Q [ITERS]]\n",
              argv[0]);
      return 1;
    }
    const int B = atoi(argv[2]);
    const int S = atoi(argv[3]);
    const int topk = atoi(argv[4]);
    const int iters = argc == 6 ? atoi(argv[5]) : 20;
    const bool supported_s =
        S == 1024 || S == 2048 || S == 4096 || S == 8192 ||
        S == 16384 || S == 32768 || S == 65536 || S == 131072;
    if (B != 1 || !supported_s || topk <= 0 || topk > S / 64 ||
        iters <= 0) {
      fprintf(stderr, "Invalid tile64 benchmark shape\n");
      return 1;
    }
    if (!getenv("LOAD_NPY")) setenv("LOAD_NPY", "inputs", 0);
    setenv("BENCH_WARMUP", "5", 1);
    if (argc == 6) setenv("BENCH_ITERS", argv[5], 1);
    setenv("VSA_BENCH_CLI", "1", 1);
    run(Sh{B, 8, S / 64, topk, 128, "cli"});
    return 0;
  }

  printf("VSA block-sparse BACKWARD bench bf16 (blk64-native, ws dual-pack) sm_100a\n"
         "non-persistent 16-warp kernel, mma.ws quads; pre+main+post (block=%d)\n"
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
