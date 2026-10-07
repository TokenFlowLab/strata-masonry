// block_sparse_bwd_bf16_blk256.cu -- VSA block-sparse BACKWARD,
// sm_100a, 256-token sparse blocks. Single-file kernel + bench harness.
//
// Backward of the SPARSE BASIS forward (../fmha_context_bf16_uniform_vsa.cu semantics);
// math and scope per ROADMAP.md secs 1-4:
//   - uniform per-row q2k counts (every q-block selects the same topk >= 1)
//   - every KV block is full (no variable_block_sizes)
//   - B=1 for the GPU path (the CPU reference is B-any), H heads, D=128,
//     bf16 inputs, natural layout [token, head, hd] (= [S,H,D] at B=1,
//     matching the unified bench .npy files)
//
// CPU reference math (fp32 throughout, inputs read bf16 -> fp32; models the
// production bf16 quantization points: O->bf16 before Delta, P->bf16 for dV,
// dS->bf16 for dQ/dK):
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
// Env knobs (mirrors the fv forward bench):
//   LOAD_NPY=<dir>   q_S{S}.npy k_S{S}.npy v_S{S}.npy do_S{S}.npy (uint16 raw bf16 bits,
//                    [S,H,D]) + idx_S{S}_blk{BLOCK}.npy (int32 [nb,topk], head-independent)
//   BLOCK=256        sparse block size; the GPU path runs only at BLOCK=256
//                    (other values exercise the CPU reference alone)
//   VSA_BWD_2CTA=<set>  launch the TWO_CTA=true instantiation (cluster pair;
//                    presence-tested, any value selects it)
//   SHAPE=0..2 + BATCH/HEADS/NB/TOPK   single-shape override
//   VSA_GAUSS / VSA_SORT_SEL / VSA_SEED_QBLK   built-in fill / index knobs (as fv)
//   CPU_REF=0|1      skip / force the CPU reference (default: small shapes only,
//                    forced when DUMP_BWD is set)
//   DUMP_BWD=<prefix>
//   plus the shared bench knobs: DUMP_BWD_GPU, VERIFY_ARGMAX, STRESS_N,
//   BENCH_ITERS, BENCH_WARMUP

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
#include "../../../../primitives/63_cvt_f32_to_f16_bf16.cuh"
#include "../../../../primitives/70_smem_ptr.cuh"
#include "../../../../primitives/76_packed_f32x2.cuh"
#include "../../../../primitives/77_ex2_approx.cuh"
#include "../../../../composites/118_mbarrier_phase_tracking.cuh"
#include "../../../../composites/106_clc_fetch_next_tile.cuh"
#include "../../../../primitives/_warp_prof_noop.cuh"
#include "../../../../primitives/69_griddepcontrol.cuh"

// Programmatic dependent launch (KERNEL_PDL): preprocess -> main -> postprocess each start while
// the predecessor drains; griddepcontrol.wait sits in front of every read of the predecessor's data.
#ifndef KERNEL_PDL
#define KERNEL_PDL true
#endif

// sm_100a VSA block-sparse BACKWARD, blk256-NATIVE -- FA4-256's design with
// a template switch: vsa_bwd_main_kernel<TWO_CTA>.
//
// A 256-token sparse block maps onto 128x128 tiles exactly as FA4 does it
// (kv_subtile_factor = q_subtile_factor = 2):
//   - TWO_CTA=false (FA4-256 1cta): grid = (2*nb256, H); each CTA owns one
//     128-row KV SUBTILE (blockIdx.x & 1) of block blockIdx.x >> 1 and walks
//     the block's q256 list expanded in-kernel to 128-row q tiles
//     (steps = 2*cnt, q tile j -> q256 = entries[beg + (j>>1)], half j&1).
//     Every tile is fully dense -- no masking.
//   - TWO_CTA=true (FA4-256 2cta): the adjacent CTA pair forms a cluster
//     covering the whole 256-row block: M=256 cluster MMAs, per-CTA Q/dO
//     slices via cta_group::2 TMA (no multicast; tx lands on the leader),
//     dS cluster exchange + relay warp, dQ as a K=256 cluster GEMM.
//
// Forked from the blk128-aligned kernel (block_sparse_bwd_bf16_blk128.cu); the 1cta warp/pipeline/register contract is identical to it
// (FA4_ALIGNMENT.md). Metadata: pair_offset/pair_union indexed by (bh, kv256
// block), entries = plain q256 ids.
//
namespace vsa_bwd_blk256 {

constexpr int KV_TILE_ROWS = 128;
constexpr int Q_TILE_ROWS  = 128;
constexpr int HEAD_DIM = 128;
constexpr int SUB = 64;
constexpr int Q_SUBTILE_BYTES  = Q_TILE_ROWS * SUB * 2;    // 16 KB
constexpr int Q_TILE_BYTES     = 2 * Q_SUBTILE_BYTES;      // 32 KB
constexpr int KV_SUBTILE_BYTES = KV_TILE_ROWS * SUB * 2;   // 16 KB
constexpr int KV_TILE_BYTES    = 2 * KV_SUBTILE_BYTES;     // 32 KB
constexpr int DST_SUBTILE_BYTES = KV_TILE_ROWS * SUB * 2;  // 16 KB
constexpr int DST_TILE_BYTES   = 2 * DST_SUBTILE_BYTES;    // 32 KB
constexpr int DQC_CHUNK_COLS   = 32;
constexpr int DQC_CHUNK_BYTES  = Q_TILE_ROWS * DQC_CHUNK_COLS * 4;  // 16 KB
// FA4 warp map: 0-3 reduce, 4-11 compute, 12 MMA, 13 load, 14 relay (live in
// 2cta, idle in 1cta), 15 empty (register donor).
constexpr int N_WARPS = 16;
constexpr int W_COMPUTE0 = 4, W_MMA = 12, W_LOAD = 13;

// FA4's per-warp register budgets, expressible in plain CUDA via
// __maxnreg__(128) (the PTX .maxnreg entry count) + setmaxnreg: ptxas then
// allocates each region up to its budget (SASS uses R128+ after the inc).
//   1cta: reduce 152, compute 136, mma/load 88, relay/empty 24.
//         Pool 4*152 + 8*136 + 2*88 + 2*24 = 1920 <= 2048 (512 thr x 128).
//   2cta: reduce/compute 136, mma/load/relay 104, empty 24.
//         Pool 4*136 + 8*136 + 3*104 + 24 = 1968 <= 2048.
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
__device__ __forceinline__ void warp_regs_dec_104() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 104;");
}

// --- 2-CTA cluster primitives (FA4 2cta PTX forms, fa4_bwd_extract) ---

// DSMEM peer mapping: cluster rank lives in address bit 24 on the 2-CTA pair;
// the leader (rank 0) barrier address is local_addr & ~bit24 (FA4 PTX form).
__device__ __forceinline__ uint32_t bar_leader_addr(uint32_t local_addr) {
  return local_addr & 0xFEFFFFFFu;
}

__device__ __forceinline__ uint32_t mapa_cluster_u32(uint32_t addr, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n"
               : "=r"(r) : "r"(addr), "r"(rank));
  return r;
}

__device__ __forceinline__ void barrier_cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void barrier_cluster_wait() {
  asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory");
}

__device__ __forceinline__
void mbarrier_arrive_expect_tx_cluster(uint32_t cluster_mbar, uint32_t tx_bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%0], %1;\n"
               :: "r"(cluster_mbar), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__
void bulk_s2cluster(uint32_t cluster_dst, uint32_t smem_src, int bytes,
                    uint32_t cluster_mbar) {
  asm volatile(
    "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
    " [%0], [%1], %2, [%3];\n"
    :: "r"(cluster_dst), "r"(smem_src), "r"(bytes), "r"(cluster_mbar)
    : "memory");
}

// cta_group::2 TMA tensor load: both CTAs issue their own slice; every
// completion signals the LEADER CTA's mbarrier (pass bar_leader_addr()).
__device__ __forceinline__
void tma_load_3d_cg2(uint32_t smem_dst, const void* tensormap_ptr,
                     uint32_t mbar_leader, int c0, int c1, int c2) {
  asm volatile(
    "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
    ".mbarrier::complete_tx::bytes.cta_group::2"
    " [%0], [%1, {%3, %4, %5}], [%2];\n"
    :: "r"(smem_dst), "l"(tensormap_ptr),
       "r"(mbar_leader), "r"(c0), "r"(c1), "r"(c2)
    : "memory");
}

__device__ __forceinline__ void tcgen05_mma_f16_ss2_lead(uint32_t lead,
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc,
    bool enable_input_d) {
  asm volatile(
    "{\n\t"
    ".reg .pred p, q;\n\t"
    "setp.ne.b32 q, %0, 0;\n\t"
    "setp.ne.b32 p, %5, 0;\n\t"
    "@q tcgen05.mma.cta_group::2.kind::f16 [%1], %2, %3, %4,"
    " {%6, %7, %8, %9, %10, %11, %12, %13}, p;\n\t"
    "}\n"
    :: "r"(lead), "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc),
       "r"(enable_input_d ? 1u : 0u), "r"(0u), "r"(0u), "r"(0u), "r"(0u),
       "r"(0u), "r"(0u), "r"(0u), "r"(0u));
}

__device__ __forceinline__ void tcgen05_mma_f16_ts2_lead(uint32_t lead,
    uint32_t tmem_c, uint32_t tmem_a, uint64_t desc_b, uint32_t idesc,
    bool enable_input_d) {
  asm volatile(
    "{\n\t"
    ".reg .pred p, q;\n\t"
    "setp.ne.b32 q, %0, 0;\n\t"
    "setp.ne.b32 p, %5, 0;\n\t"
    "@q tcgen05.mma.cta_group::2.kind::f16 [%1], [%2], %3, %4,"
    " {%6, %7, %8, %9, %10, %11, %12, %13}, p;\n\t"
    "}\n"
    :: "r"(lead), "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(idesc),
       "r"(enable_input_d ? 1u : 0u), "r"(0u), "r"(0u), "r"(0u), "r"(0u),
       "r"(0u), "r"(0u), "r"(0u), "r"(0u));
}

// cta_group::2 commit, multicast to BOTH CTAs' local barrier words (mask 0b11).
__device__ __forceinline__ void tcgen05_commit_mc2_lead(uint32_t lead, uint32_t mbar_smem) {
  asm volatile(
    "{\n\t"
    ".reg .pred q;\n\t"
    "setp.ne.b32 q, %0, 0;\n\t"
    "@q tcgen05.commit.cta_group::2.mbarrier::arrive::one"
    ".shared::cluster.multicast::cluster.b64 [%1], %2;\n\t"
    "}\n"
    :: "r"(lead), "r"(mbar_smem), "h"((uint16_t)3) : "memory");
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
void bulk_reduce_add_f32(const float* gmem_dst, uint32_t smem_src, int bytes) {
  asm volatile(
    "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
    " [%0], [%1], %2;\n"
    :: "l"(gmem_dst), "r"(smem_src), "r"(bytes)
    : "memory");
}

// 1cta SMEM (~226 KB): K 32K | V 32K | Q[2] 64K | dO 32K | dST 32K |
// dQc chunks [2] 32K | M/Delta rings 1.5K | barriers. The dK/dV epilogue
// bounces through the then-free sDO (dV) and sQ stage 0 (dK).
constexpr int SMEM_K    = 0;
constexpr int SMEM_V    = SMEM_K + KV_TILE_BYTES;
constexpr int SMEM_Q    = SMEM_V + KV_TILE_BYTES;
constexpr int SMEM_DO   = SMEM_Q + 2 * Q_TILE_BYTES;
constexpr int SMEM_DST  = SMEM_DO + Q_TILE_BYTES;
constexpr int SMEM_DQC  = SMEM_DST + DST_TILE_BYTES;
constexpr int SMEM_LSE  = SMEM_DQC + 2 * DQC_CHUNK_BYTES;      // (128,2) f32
constexpr int SMEM_DPS  = SMEM_LSE + 2 * Q_TILE_ROWS * 4;      // (128,1) f32
constexpr int SMEM_BARS = SMEM_DPS + Q_TILE_ROWS * 4;
constexpr int NUM_BARS  = 20;
constexpr int SMEM_TOTAL = SMEM_BARS + NUM_BARS * 8 + 16;

// 2-CTA SMEM (~225 KB; FA4 2cta SharedStorage, our field names): per-CTA
// halves of Q/dO (16K, 1 stage) + dedicated transposed operand buffers
// sQt/sdOt/sKt + the dS exchange staging buffer + a 4-stage x 4KB dQaccum
// ring. The dK/dV epilogue bounces through sV (dV) and sK (dK) -- sQ/sDO
// are only 16K here.
constexpr int Q2_SUBTILE_BYTES = 64 * SUB * 2;                  // 8 KB (64 q x 64 hd)
constexpr int DQA2_STAGE_BYTES = Q_TILE_ROWS * 8 * 4;           // 4 KB
constexpr int SMEM2_K    = 0;                                    // 32 KB
constexpr int SMEM2_V    = SMEM2_K + KV_TILE_BYTES;              // 32 KB
constexpr int SMEM2_Q    = SMEM2_V + KV_TILE_BYTES;              // 16 KB (q half x hd)
constexpr int SMEM2_DO   = SMEM2_Q + 2 * Q2_SUBTILE_BYTES;       // 16 KB (hd half x q)
constexpr int SMEM2_QT   = SMEM2_DO + Q_SUBTILE_BYTES;           // 16 KB (hd half x q)
constexpr int SMEM2_DOT  = SMEM2_QT + Q_SUBTILE_BYTES;           // 16 KB (q half x hd)
constexpr int SMEM2_KT   = SMEM2_DOT + 2 * Q2_SUBTILE_BYTES;     // 32 KB (hd half x kv256)
constexpr int SMEM2_DST  = SMEM2_KT + KV_TILE_BYTES;             // 32 KB (2 kv-half regions)
constexpr int SMEM2_XCHG = SMEM2_DST + DST_TILE_BYTES;           // 16 KB
constexpr int SMEM2_DQA  = SMEM2_XCHG + DST_SUBTILE_BYTES;       // 16 KB (4 stages)
constexpr int SMEM2_LSE  = SMEM2_DQA + 4 * DQA2_STAGE_BYTES;     // (128,1) f32
constexpr int SMEM2_DPS  = SMEM2_LSE + Q_TILE_ROWS * 4;
constexpr int SMEM2_BARS = SMEM2_DPS + Q_TILE_ROWS * 4;
constexpr int NUM_BARS_2 = 29;
constexpr int SMEM_TOTAL_2 = SMEM2_BARS + NUM_BARS_2 * 8 + 16;

enum {
  // The *1 stage entries are 1cta-only: 2cta Q and LSE are single-stage.
  BAR_Q_FULL0 = 0, BAR_Q_FULL1,   // load -> mma; j==0 stage also carries K tx
  BAR_Q_EMPTY0, BAR_Q_EMPTY1,     // commit after dK(j) in 1cta (last Q
                                  // reader); right after S(j) in 2cta
  BAR_DO_FULL,                    // load -> mma; j==0 also carries V tx
  BAR_DO_EMPTY,                   // commit after dV(j) (last dO reader)
  BAR_LSE_FULL0, BAR_LSE_FULL1,   // load -> compute (1cta rides the Q stage;
                                  // 2cta is 1-deep and rides j parity)
  BAR_LSE_EMPTY0, BAR_LSE_EMPTY1, // 8 arrivals: lane 0 of each compute warp
  BAR_DPSUM_FULL,                 // load -> compute (rides the dO stage)
  BAR_DPSUM_EMPTY,                // 8 arrivals
  BAR_ST_READY,     // commit after the ST atoms (S_P full)
  BAR_DPT_READY,    // commit after the dPT atoms (dP full)
  BAR_PT_STTMD,     // S_P empty: P STTM'd; 8 arrivals (16 in 2cta: both
                    // CTAs' compute warps arrive the leader)
  BAR_DST_READY,    // dS full: dST STTM'd + SMEM'd; 8 arrivals (2cta: 16,
                    // DEFERRED to the next S t2r -- see the compute loop)
  BAR_DQC_FULL,     // commit after the dQc atoms (dQ full)
  BAR_DQC_FREE,     // dQ empty: reduce t2r done; 4 arrivals (8 in 2cta);
                    // gates dP(j+1) in 1cta / S(j+1) + tail dQ in 2cta
  BAR_DV_FULL,      // dKV stage 0: commit when the last dV accumulation landed
  BAR_DK_FULL,      // dKV stage 1: commit after the tail dK
  // -- 2-CTA only (FA4 2cta contract) --
  BAR_QT_FULL,      // load -> mma (Qt = dK B operand; lags one q tile)
  BAR_QT_EMPTY,     // commit after dK(j)
  BAR_KT_FULL,      // load -> mma, once per tile (Kt = dQ B operand)
  BAR_KT_EMPTY,     // dangling final commit (FA4 form; never waited)
  BAR_DPT_EMPTY,    // LIVE in 2cta: dS STTM done (16 remote arrivals on leader)
  BAR_DST_EMPTY,    // dS SMEM slots free (multicast commit after dQ mma)
  BAR_DSX_FULL,     // peer 16KB dS half landed in my sDST (1 arrive + tx)
  BAR_DSX_LEADER,   // both relays certified the exchange (2 arrivals, leader)
  BAR_TMEM_DEALLOC, // peer handshake before tcgen05.dealloc (32 arrivals)
};

// 1cta TMEM (512 cols, FA4 map): ST=PT 0-127 | dV 128-255 | dPT=dST=dQc 256-383 |
// dK 384-511. PT bf16 packs at cols {0-31, 64-95} (q-half h at h*64, inside
// its OWN fp32 read range -> the overlay is self-ordered per warp); dST
// bf16 likewise at 256 + {0-31, 64-95}; dQc fp32 reuses 256-383 after dK
// read dST (in-order tcgen05 pipe).
constexpr uint32_t T_ST = 0, T_DV = 128, T_DPT = 256, T_DK = 384;
constexpr uint32_t T_PT_BF16  = T_ST;
constexpr uint32_t T_DST_BF16 = T_DPT;
constexpr uint32_t T_DQC      = T_DPT;
// 2cta: dQ acc = cols 64..127 (overlays the upper half of S; per-CTA 64 q x
// 128 hd folded as 128 lanes x 64 cols). P bf16 must pack CONTIGUOUS at
// words 0..63 (the 1cta per-half sub-overlay would put q-half 1 at words
// 64..95, inside the dQ range) -> cross-half STTM needs the pre-STTM barrier.
constexpr uint32_t T_DQC_2 = 64;

extern __shared__ __align__(1024) uint8_t bwd_smem[];

template <bool TWO_CTA>
__global__ void __maxnreg__(128)
vsa_bwd_main_kernel(const __grid_constant__ CUtensorMap tmap_q,
                    const __grid_constant__ CUtensorMap tmap_k,
                    const __grid_constant__ CUtensorMap tmap_v,
                    const __grid_constant__ CUtensorMap tmap_do,
                    const __grid_constant__ CUtensorMap tmap_dk,
                    const __grid_constant__ CUtensorMap tmap_dv,
                    const __grid_constant__ CUtensorMap tmap_q64,   // 2cta: box (64,64)
                    const __grid_constant__ CUtensorMap tmap_do64,  // 2cta: box (64,64)
                    const __grid_constant__ CUtensorMap tmap_k256,  // 2cta: box (64,256)
                    float* __restrict__ dqaccum,
                    const float* __restrict__ m_rows,
                    const float* __restrict__ delta_rows,
                    const int* __restrict__ pair_offset,
                    const unsigned* __restrict__ pair_union,
                    int num_heads, int seqlen, int num_blocks,
                    float scale_log2, float sm_scale) {
  uint8_t* base = bwd_smem;
  __nv_bfloat16* sK  = reinterpret_cast<__nv_bfloat16*>(base + (TWO_CTA ? SMEM2_K : SMEM_K));
  __nv_bfloat16* sV  = reinterpret_cast<__nv_bfloat16*>(base + (TWO_CTA ? SMEM2_V : SMEM_V));
  // 2cta Q is single-stage: sQ[1] aliases sQ[0].
  __nv_bfloat16* sQ[2]  = { reinterpret_cast<__nv_bfloat16*>(base + (TWO_CTA ? SMEM2_Q : SMEM_Q)),
                            reinterpret_cast<__nv_bfloat16*>(base + (TWO_CTA ? SMEM2_Q : SMEM_Q + Q_TILE_BYTES)) };
  __nv_bfloat16* sDO  = reinterpret_cast<__nv_bfloat16*>(base + (TWO_CTA ? SMEM2_DO : SMEM_DO));
  __nv_bfloat16* sDST = reinterpret_cast<__nv_bfloat16*>(base + (TWO_CTA ? SMEM2_DST : SMEM_DST));
  // 1cta-only ring (2cta drains dQ through sDQA instead).
  float* sDQC[2] = { reinterpret_cast<float*>(base + SMEM_DQC),
                     reinterpret_cast<float*>(base + SMEM_DQC + DQC_CHUNK_BYTES) };
  float* sLSE = reinterpret_cast<float*>(base + (TWO_CTA ? SMEM2_LSE : SMEM_LSE));
  float* sDPS = reinterpret_cast<float*>(base + (TWO_CTA ? SMEM2_DPS : SMEM_DPS));
  // 2cta-only buffers (dead pointers in the 1cta instantiation)
  __nv_bfloat16* sQT   = reinterpret_cast<__nv_bfloat16*>(base + SMEM2_QT);
  __nv_bfloat16* sDOT  = reinterpret_cast<__nv_bfloat16*>(base + SMEM2_DOT);
  __nv_bfloat16* sKT   = reinterpret_cast<__nv_bfloat16*>(base + SMEM2_KT);
  __nv_bfloat16* sXCHG = reinterpret_cast<__nv_bfloat16*>(base + SMEM2_XCHG);
  float* sDQA          = reinterpret_cast<float*>(base + SMEM2_DQA);
  uint64_t* bars = reinterpret_cast<uint64_t*>(base + (TWO_CTA ? SMEM2_BARS : SMEM_BARS));
  uint32_t* tmem_slot = reinterpret_cast<uint32_t*>(bars + (TWO_CTA ? NUM_BARS_2 : NUM_BARS));
  const int cta_rank = TWO_CTA ? ((int)blockIdx.x & 1) : 0;
  const uint32_t lead_cta = (!TWO_CTA || cta_rank == 0) ? 1u : 0u;
  (void)sQT; (void)sDOT; (void)sKT; (void)sXCHG; (void)sDQA; (void)lead_cta;

  const int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;

  if (tid == 0) {
    mbarrier_init(smem_ptr_u32(&bars[BAR_Q_FULL0]),    1);
    mbarrier_init(smem_ptr_u32(&bars[BAR_Q_FULL1]),    1);
    mbarrier_init(smem_ptr_u32(&bars[BAR_Q_EMPTY0]),   1);
    mbarrier_init(smem_ptr_u32(&bars[BAR_Q_EMPTY1]),   1);
    mbarrier_init(smem_ptr_u32(&bars[BAR_DO_FULL]),    1);
    mbarrier_init(smem_ptr_u32(&bars[BAR_DO_EMPTY]),   1);
    mbarrier_init(smem_ptr_u32(&bars[BAR_LSE_FULL0]),  1);
    mbarrier_init(smem_ptr_u32(&bars[BAR_LSE_FULL1]),  1);
    mbarrier_init(smem_ptr_u32(&bars[BAR_LSE_EMPTY0]), 8);
    mbarrier_init(smem_ptr_u32(&bars[BAR_LSE_EMPTY1]), 8);
    mbarrier_init(smem_ptr_u32(&bars[BAR_DPSUM_FULL]), 1);
    mbarrier_init(smem_ptr_u32(&bars[BAR_DPSUM_EMPTY]),8);
    mbarrier_init(smem_ptr_u32(&bars[BAR_ST_READY]),   1);
    mbarrier_init(smem_ptr_u32(&bars[BAR_DPT_READY]),  1);
    mbarrier_init(smem_ptr_u32(&bars[BAR_PT_STTMD]),   TWO_CTA ? 16 : 8);
    mbarrier_init(smem_ptr_u32(&bars[BAR_DST_READY]),  TWO_CTA ? 16 : 8);
    mbarrier_init(smem_ptr_u32(&bars[BAR_DQC_FULL]),   1);
    mbarrier_init(smem_ptr_u32(&bars[BAR_DQC_FREE]),   TWO_CTA ? 8 : 4);
    mbarrier_init(smem_ptr_u32(&bars[BAR_DV_FULL]),    1);
    mbarrier_init(smem_ptr_u32(&bars[BAR_DK_FULL]),    1);
    if (TWO_CTA) {
      mbarrier_init(smem_ptr_u32(&bars[BAR_QT_FULL]),      1);
      mbarrier_init(smem_ptr_u32(&bars[BAR_QT_EMPTY]),     1);
      mbarrier_init(smem_ptr_u32(&bars[BAR_KT_FULL]),      1);
      mbarrier_init(smem_ptr_u32(&bars[BAR_KT_EMPTY]),     1);
      mbarrier_init(smem_ptr_u32(&bars[BAR_DPT_EMPTY]),    16);
      mbarrier_init(smem_ptr_u32(&bars[BAR_DST_EMPTY]),    1);
      mbarrier_init(smem_ptr_u32(&bars[BAR_DSX_FULL]),     1);
      mbarrier_init(smem_ptr_u32(&bars[BAR_DSX_LEADER]),   2);
      mbarrier_init(smem_ptr_u32(&bars[BAR_TMEM_DEALLOC]), 32);
    }
  }
  fence_mbarrier_init_release_cluster();
  __syncthreads();
  if (TWO_CTA) {          // peer's mbarrier inits must be visible cluster-wide
    barrier_cluster_arrive_relaxed();
    barrier_cluster_wait();
  }
  // TMEM alloc protocol (FA4): the MMA warp allocates AFTER the init sync and
  // publishes via a 13-warp named barrier (warps 0-12); the load warp never
  // waits on it and starts TMA immediately.
  // Sparse lookup shared by every role. FA4-256 mapping (kv_subtile_factor =
  // q_subtile_factor = 2): the CTA owns 128 kv rows at token offset
  // blockIdx.x * 128; the lookup uses the parent 256-block (blockIdx.x >> 1);
  // the q256 list expands in-kernel to 128-row q tiles (steps = 2 * cnt).
  const int n_block = (int)blockIdx.x;               // kv tile, 128-row units
  const int bh      = (int)blockIdx.y;
  const int list_idx = bh * num_blocks + (n_block >> 1);   // num_blocks = nb256
  const int beg = pair_offset[list_idx], cnt = pair_offset[list_idx + 1] - beg;
  const int steps = 2 * cnt;                         // q_subtile_factor = 2

  if (warp_id > W_LOAD) {
    if (TWO_CTA && warp_id == W_LOAD + 1) {
      // Relay warp (FA4 warp 14, 2cta only): converts "my sDST got the peer
      // half" (local DSX_FULL, 1 arrive + 16KB tx) into a remote arrive on
      // the LEADER's DSX_LEADER (count 2 = both relays) gating every dQ mma.
      warp_regs_dec_104();
      if (cnt == 0) return;
      const uint32_t dsx = smem_ptr_u32(&bars[BAR_DSX_FULL]);
      const uint32_t dsx_leader =
          mapa_cluster_u32(smem_ptr_u32(&bars[BAR_DSX_LEADER]), 0);
      for (int j = 0; j < steps; ++j) {
        mbarrier_wait_parity_suspend(dsx, (uint32_t)(j & 1));
        if (elect_one_sync()) mbarrier_arrive_cluster_default(dsx_leader);
      }
      return;
    }
    warp_regs_dec_24();              // empty warp: register donor only
    return;
  }
  WpCtx wpc = wp_ctx_init();

  if (warp_id == W_LOAD) {
    if constexpr (TWO_CTA) {
      // FA4 2cta load contract: BOTH CTAs issue their own per-CTA slice with
      // cta_group::2 TMA; every tx completes on the LEADER's full barrier and
      // only the leader arms expect_tx. Each CTA waits its LOCAL empties
      // (multicast commits from the leader MMA). LSE/dPsum stay CTA-local.
      // Per-CTA slices: Q/dOt = own 64-q half x 128 hd; dO/Qt = own 64-hd
      // half x 128 q; K/V = own 128 kv rows; Kt = own 64-hd half x 256 kv.
      warp_regs_dec_104();
      if constexpr (KERNEL_PDL) griddepcontrol_wait();  // Delta and the zeroed dqaccum
      if (cnt == 0) return;
      EmptyPhaseTracker<1> q_empty_ph, lse_empty_ph, do_empty_ph, dps_empty_ph,
                           qt_empty_ph;
      // cta_group::2 TMA operands live in the cluster-pair frame: the DST
      // must carry the issuing CTA's rank (FA4: cvta.to.shared::cluster),
      // the mbar gets the even-CTA form (bit 24 cleared).
      auto own_rank_addr = [&](uint32_t addr) {
        return mapa_cluster_u32(addr, (uint32_t)cta_rank);
      };
      const uint32_t q_full  = smem_ptr_u32(&bars[BAR_Q_FULL0]);
      const uint32_t do_full = smem_ptr_u32(&bars[BAR_DO_FULL]);
      const uint32_t qt_full = smem_ptr_u32(&bars[BAR_QT_FULL]);
      const uint32_t kt_full = smem_ptr_u32(&bars[BAR_KT_FULL]);
      auto load_qt = [&](int jq) {
        const int q_blk = 2 * (int)pair_union[beg + (jq >> 1)] + (jq & 1);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&bars[BAR_QT_EMPTY]),
                                     qt_empty_ph.get_phase());
        qt_empty_ph.advance();
        if (elect_one_sync()) {
          if (lead_cta) mbarrier_arrive_expect_tx(qt_full, Q_TILE_BYTES);
          tma_load_3d_cg2(own_rank_addr(smem_ptr_u32(sQT)), &tmap_q, bar_leader_addr(qt_full),
                          0, q_blk * Q_TILE_ROWS, bh * 2 + cta_rank);
        }
      };
      for (int j = 0; j < steps; ++j) {
        wp_marker(wpc, WP_ITER, j);
        const int q_blk = 2 * (int)pair_union[beg + (j >> 1)] + (j & 1);
        if (j > 0) load_qt(j - 1);              // Qt lags one q tile (FA4)
        wp_begin(wpc, WP_LOAD_WAIT);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&bars[BAR_Q_EMPTY0]),
                                     q_empty_ph.get_phase());
        q_empty_ph.advance();
        wp_end(wpc, WP_LOAD_WAIT);
        wp_begin(wpc, WP_LOAD_ISSUE_Q);
        if (elect_one_sync()) {
          if (lead_cta)
            mbarrier_arrive_expect_tx(q_full,
                Q_TILE_BYTES + (j == 0 ? 2 * KV_TILE_BYTES : 0));
          #pragma unroll
          for (int ssub = 0; ssub < 2; ++ssub)
            tma_load_3d_cg2(own_rank_addr(smem_ptr_u32(reinterpret_cast<uint8_t*>(sQ[0]) + ssub * Q2_SUBTILE_BYTES)),
                            &tmap_q64, bar_leader_addr(q_full),
                            0, q_blk * Q_TILE_ROWS + 64 * cta_rank, bh * 2 + ssub);
          if (j == 0) {
            #pragma unroll
            for (int ssub = 0; ssub < 2; ++ssub)
              tma_load_3d_cg2(own_rank_addr(smem_ptr_u32(reinterpret_cast<uint8_t*>(sK) + ssub * KV_SUBTILE_BYTES)),
                              &tmap_k, bar_leader_addr(q_full),
                              0, n_block * KV_TILE_ROWS, bh * 2 + ssub);
          }
        }
        // LSE: CTA-local single-stage ring (both CTAs duplicate all 128 rows).
        mbarrier_wait_parity_suspend(smem_ptr_u32(&bars[BAR_LSE_EMPTY0]),
                                     lse_empty_ph.get_phase());
        lse_empty_ph.advance();
        if (elect_one_sync()) {
          mbarrier_arrive_expect_tx(smem_ptr_u32(&bars[BAR_LSE_FULL0]), Q_TILE_ROWS * 4);
          bulk_g2s(smem_ptr_u32(sLSE),
                   m_rows + (size_t)bh * seqlen + (size_t)q_blk * Q_TILE_ROWS,
                   Q_TILE_ROWS * 4, smem_ptr_u32(&bars[BAR_LSE_FULL0]));
        }
        wp_end(wpc, WP_LOAD_ISSUE_Q);
        wp_begin(wpc, WP_LOAD_WAIT_THROTTLE);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&bars[BAR_DO_EMPTY]),
                                     do_empty_ph.get_phase());
        do_empty_ph.advance();
        wp_end(wpc, WP_LOAD_WAIT_THROTTLE);
        wp_begin(wpc, WP_LOAD_ISSUE_V);
        if (elect_one_sync()) {
          if (lead_cta)
            mbarrier_arrive_expect_tx(do_full,
                2 * Q_TILE_BYTES + (j == 0 ? 2 * KV_TILE_BYTES : 0));
          tma_load_3d_cg2(own_rank_addr(smem_ptr_u32(sDO)), &tmap_do, bar_leader_addr(do_full),
                          0, q_blk * Q_TILE_ROWS, bh * 2 + cta_rank);
          #pragma unroll
          for (int ssub = 0; ssub < 2; ++ssub)
            tma_load_3d_cg2(own_rank_addr(smem_ptr_u32(reinterpret_cast<uint8_t*>(sDOT) + ssub * Q2_SUBTILE_BYTES)),
                            &tmap_do64, bar_leader_addr(do_full),
                            0, q_blk * Q_TILE_ROWS + 64 * cta_rank, bh * 2 + ssub);
          if (j == 0) {
            #pragma unroll
            for (int ssub = 0; ssub < 2; ++ssub)
              tma_load_3d_cg2(own_rank_addr(smem_ptr_u32(reinterpret_cast<uint8_t*>(sV) + ssub * KV_SUBTILE_BYTES)),
                              &tmap_v, bar_leader_addr(do_full),
                              0, n_block * KV_TILE_ROWS, bh * 2 + ssub);
          }
        }
        mbarrier_wait_parity_suspend(smem_ptr_u32(&bars[BAR_DPSUM_EMPTY]),
                                     dps_empty_ph.get_phase());
        dps_empty_ph.advance();
        if (elect_one_sync()) {
          mbarrier_arrive_expect_tx(smem_ptr_u32(&bars[BAR_DPSUM_FULL]), Q_TILE_ROWS * 4);
          bulk_g2s(smem_ptr_u32(sDPS),
                   delta_rows + (size_t)bh * seqlen + (size_t)q_blk * Q_TILE_ROWS,
                   Q_TILE_ROWS * 4, smem_ptr_u32(&bars[BAR_DPSUM_FULL]));
        }
        if (j == 0) {
          // Kt once per tile, after dPsum (FA4 prologue order). Its empty
          // barrier starts free and is never re-acquired.
          if (elect_one_sync()) {
            if (lead_cta) mbarrier_arrive_expect_tx(kt_full, 2 * KV_TILE_BYTES);
            tma_load_3d_cg2(own_rank_addr(smem_ptr_u32(sKT)), &tmap_k256, bar_leader_addr(kt_full),
                            0, (n_block >> 1) * 2 * KV_TILE_ROWS, bh * 2 + cta_rank);
          }
        }
        wp_end(wpc, WP_LOAD_ISSUE_V);
      }
      load_qt(steps - 1);                        // final Qt (FA4 tail)
      wp_flush(wpc);
      return;
    }
    warp_regs_dec_88();
    // Delta and the zeroed dqaccum come from the preprocess; every consumer in the CTA is
    // downstream of this warp's loads.
    if constexpr (KERNEL_PDL) griddepcontrol_wait();
    if (cnt == 0) return;
    EmptyPhaseTracker<2> q_empty_ph, lse_empty_ph;
    EmptyPhaseTracker<1> do_empty_ph, dps_empty_ph;
    for (int j = 0; j < steps; ++j) {
      wp_marker(wpc, WP_ITER, j);
      const int q_blk = 2 * (int)pair_union[beg + (j >> 1)] + (j & 1);
      const int sd = q_empty_ph.get_stage() & 1;
      wp_begin(wpc, WP_LOAD_WAIT);
      mbarrier_wait_parity_suspend(smem_ptr_u32(&bars[BAR_Q_EMPTY0 + sd]),
                                   q_empty_ph.get_phase());
      q_empty_ph.advance();
      wp_end(wpc, WP_LOAD_WAIT);
      wp_begin(wpc, WP_LOAD_ISSUE_Q);
      if (elect_one_sync()) {
        mbarrier_arrive_expect_tx(smem_ptr_u32(&bars[BAR_Q_FULL0 + sd]),
                                  Q_TILE_BYTES + (j == 0 ? KV_TILE_BYTES : 0));
        #pragma unroll
        for (int ssub = 0; ssub < 2; ++ssub)
          tma_load_3d(smem_ptr_u32(reinterpret_cast<uint8_t*>(sQ[sd]) + ssub * Q_SUBTILE_BYTES),
                      &tmap_q, smem_ptr_u32(&bars[BAR_Q_FULL0 + sd]),
                      0, q_blk * Q_TILE_ROWS, bh * 2 + ssub);
        if (j == 0) {
          #pragma unroll
          for (int ssub = 0; ssub < 2; ++ssub)
            tma_load_3d(smem_ptr_u32(reinterpret_cast<uint8_t*>(sK) + ssub * KV_SUBTILE_BYTES),
                        &tmap_k, smem_ptr_u32(&bars[BAR_Q_FULL0 + sd]),
                        0, n_block * KV_TILE_ROWS, bh * 2 + ssub);
        }
      }
      // LSE ring rides the Q stage index (FA4 pipeline_LSE, 512 B plain bulk).
      mbarrier_wait_parity_suspend(smem_ptr_u32(&bars[BAR_LSE_EMPTY0 + sd]),
                                   lse_empty_ph.get_phase());
      lse_empty_ph.advance();
      if (elect_one_sync()) {
        mbarrier_arrive_expect_tx(smem_ptr_u32(&bars[BAR_LSE_FULL0 + sd]), Q_TILE_ROWS * 4);
        bulk_g2s(smem_ptr_u32(sLSE + sd * Q_TILE_ROWS),
                 m_rows + (size_t)bh * seqlen + (size_t)q_blk * Q_TILE_ROWS,
                 Q_TILE_ROWS * 4, smem_ptr_u32(&bars[BAR_LSE_FULL0 + sd]));
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
        for (int ssub = 0; ssub < 2; ++ssub)
          tma_load_3d(smem_ptr_u32(reinterpret_cast<uint8_t*>(sDO) + ssub * Q_SUBTILE_BYTES),
                      &tmap_do, smem_ptr_u32(&bars[BAR_DO_FULL]),
                      0, q_blk * Q_TILE_ROWS, bh * 2 + ssub);
        if (j == 0) {
          #pragma unroll
          for (int ssub = 0; ssub < 2; ++ssub)
            tma_load_3d(smem_ptr_u32(reinterpret_cast<uint8_t*>(sV) + ssub * KV_SUBTILE_BYTES),
                        &tmap_v, smem_ptr_u32(&bars[BAR_DO_FULL]),
                        0, n_block * KV_TILE_ROWS, bh * 2 + ssub);
        }
      }
      // dPsum rides the dO stage (FA4 pipeline_dPsum, 1-deep).
      mbarrier_wait_parity_suspend(smem_ptr_u32(&bars[BAR_DPSUM_EMPTY]),
                                   dps_empty_ph.get_phase());
      dps_empty_ph.advance();
      if (elect_one_sync()) {
        mbarrier_arrive_expect_tx(smem_ptr_u32(&bars[BAR_DPSUM_FULL]), Q_TILE_ROWS * 4);
        bulk_g2s(smem_ptr_u32(sDPS),
                 delta_rows + (size_t)bh * seqlen + (size_t)q_blk * Q_TILE_ROWS,
                 Q_TILE_ROWS * 4, smem_ptr_u32(&bars[BAR_DPSUM_FULL]));
      }
      wp_end(wpc, WP_LOAD_ISSUE_V);
    }
    wp_flush(wpc);
    return;
  }
  else if (warp_id == W_MMA) {
    if constexpr (TWO_CTA) {
      // FA4 2cta MMA: both CTAs' warp 12 alloc/dealloc (cta_group::2 + peer
      // dealloc handshake); ONLY the leader issues MMAs and barrier traffic.
      // All full/empty commits are cta_group::2 multicast (mask 0b11); all
      // waits are on the leader's local words. Rotation: S(next) [gated
      // dQ-empty: dQ TMEM overlays S 64..127] -> dK(cur) [Qt + dP-empty =
      // "dS STTM'd"] -> dP(next) -> dQ(cur) [dS-full (deferred: also proves
      // S(next) t2r done) + DSX_LEADER] -> dV(next).
      warp_regs_dec_104();
      tcgen05_alloc<2>(smem_ptr_u32(tmem_slot), 512);
      bar_sync<10>(416);
      const uint32_t tmem_base = *tmem_slot;
      auto teardown = [&]() {
        tcgen05_relinquish_alloc_permit<2>();
        bar_sync<10>(416);
        // FA4 dealloc handshake: all 32 lanes arrive the PEER's mbar, wait
        // own (count 32), then dealloc -- neither CTA exits before the pair.
        mbarrier_arrive_cluster_default(
            mapa_cluster_u32(smem_ptr_u32(&bars[BAR_TMEM_DEALLOC]),
                             (uint32_t)(cta_rank ^ 1)));
        mbarrier_wait_parity(smem_ptr_u32(&bars[BAR_TMEM_DEALLOC]), 0);
        tcgen05_dealloc<2>(tmem_base, 512);
      };
      if (cnt == 0 || !lead_cta) { teardown(); return; }

      const uint32_t lead = elect_one_sync() ? 1u : 0u;
      constexpr uint32_t DESC_SBO = 1024, DESC_LBO = 16;
      constexpr uint32_t K16_MN = (uint32_t)((16 * SUB * 2) >> 4);
      constexpr uint64_t Q2_SUB_DELTA  = Q2_SUBTILE_BYTES >> 4;
      constexpr uint64_t KV_SUB_DELTA = KV_SUBTILE_BYTES >> 4;

      const uint32_t idesc_st2  = make_idesc_bf16_f32(2 * KV_TILE_ROWS, Q_TILE_ROWS, false, false);
      const uint32_t idesc_acc2 = make_idesc_bf16_f32(2 * KV_TILE_ROWS, HEAD_DIM, false, true);
      const uint32_t idesc_dqc  = make_idesc_bf16_f32(Q_TILE_ROWS, HEAD_DIM, true, true);

      const uint64_t desc_k   = build_smem_desc_blackwell(smem_ptr_u32(sK),  DESC_SBO, DESC_LBO, SmemSwizzleBlackwell::B128);
      const uint64_t desc_v   = build_smem_desc_blackwell(smem_ptr_u32(sV),  DESC_SBO, DESC_LBO, SmemSwizzleBlackwell::B128);
      const uint64_t desc_q2  = build_smem_desc_blackwell(smem_ptr_u32(sQ[0]), DESC_SBO, DESC_LBO, SmemSwizzleBlackwell::B128);
      const uint64_t desc_dot = build_smem_desc_blackwell(smem_ptr_u32(sDOT), DESC_SBO, DESC_LBO, SmemSwizzleBlackwell::B128);
      const uint64_t desc_do_mn  = build_smem_desc_blackwell(smem_ptr_u32(sDO),  DESC_SBO, (uint32_t)Q_SUBTILE_BYTES, SmemSwizzleBlackwell::B128);
      const uint64_t desc_qt_mn  = build_smem_desc_blackwell(smem_ptr_u32(sQT),  DESC_SBO, (uint32_t)Q_SUBTILE_BYTES, SmemSwizzleBlackwell::B128);
      const uint64_t desc_dst_mn = build_smem_desc_blackwell(smem_ptr_u32(sDST), DESC_SBO, (uint32_t)DST_SUBTILE_BYTES, SmemSwizzleBlackwell::B128);
      const uint64_t desc_kt_mn  = build_smem_desc_blackwell(smem_ptr_u32(sKT),  DESC_SBO, (uint32_t)KV_TILE_BYTES, SmemSwizzleBlackwell::B128);

      PhaseTracker<1> q_ph, qt_ph, do_ph, ptst_ph, dpte_ph, dstr_ph, dsxl_ph;
      EmptyPhaseTracker<1> dqcf_ph;

      auto commit_mc = [&](int bar) {
        tcgen05_commit_mc2_lead(lead, smem_ptr_u32(&bars[bar]));
      };
      auto issue_st2 = [&]() {
        #pragma unroll
        for (int ssub = 0; ssub < 2; ++ssub)
          #pragma unroll
          for (int ki = 0; ki < 4; ++ki)
            tcgen05_mma_f16_ss2_lead(lead, tmem_base + T_ST,
                                     desc_k + ssub * KV_SUB_DELTA + 2 * ki,
                                     desc_q2 + ssub * Q2_SUB_DELTA + 2 * ki,
                                     idesc_st2, (ssub | ki) != 0);
        commit_mc(BAR_ST_READY);
        commit_mc(BAR_Q_EMPTY0);        // Q freed right away (dK reads sQt)
      };
      auto issue_dpt2 = [&]() {
        #pragma unroll
        for (int ssub = 0; ssub < 2; ++ssub)
          #pragma unroll
          for (int ki = 0; ki < 4; ++ki)
            tcgen05_mma_f16_ss2_lead(lead, tmem_base + T_DPT,
                                     desc_v + ssub * KV_SUB_DELTA + 2 * ki,
                                     desc_dot + ssub * Q2_SUB_DELTA + 2 * ki,
                                     idesc_st2, (ssub | ki) != 0);
        commit_mc(BAR_DPT_READY);
      };
      auto issue_dv2 = [&](bool first) {
        // P bf16 is PACKED at words 0..63 in 2cta -> linear A walk.
        #pragma unroll
        for (int ki = 0; ki < 8; ++ki)
          tcgen05_mma_f16_ts2_lead(lead, tmem_base + T_DV,
                                   tmem_base + T_PT_BF16 + (uint32_t)(ki * 8),
                                   desc_do_mn + 2 * 0 + (uint64_t)ki * K16_MN,
                                   idesc_acc2, !(first && ki == 0));
        commit_mc(BAR_DO_EMPTY);
      };
      auto issue_dk2 = [&](bool first) {
        // dS bf16 keeps the per-half sub-overlay (words 256+{0-31,64-95}).
        #pragma unroll
        for (int ki = 0; ki < 8; ++ki)
          tcgen05_mma_f16_ts2_lead(lead, tmem_base + T_DK,
                                   tmem_base + T_DST_BF16 + (uint32_t)(ki * 8 + (ki >= 4 ? 32 : 0)),
                                   desc_qt_mn + (uint64_t)ki * K16_MN,
                                   idesc_acc2, !(first && ki == 0));
        commit_mc(BAR_QT_EMPTY);
      };
      auto issue_dqc2 = [&]() {
        // K = 256 (cluster-wide kv): 16 k-steps; A = own 64-q half x 256 kv
        // (two contiguous 16KB kv-half regions -> one uniform walk), B = sKt.
        SmemDescPair adst; adst.u64 = desc_dst_mn;
        SmemDescPair b_kt;  b_kt.u64 = desc_kt_mn;
        #pragma unroll
        for (int ki = 0; ki < 16; ++ki) {
          tcgen05_mma_f16_ss2_lead(lead, tmem_base + T_DQC_2,
                                   adst.u64, b_kt.u64, idesc_dqc, ki != 0);
          desc_add_lo(adst, K16_MN);
          desc_add_lo(b_kt, K16_MN);
        }
        commit_mc(BAR_DQC_FULL);
        commit_mc(BAR_DST_EMPTY);
      };

      // Prologue: S(0), dP(0), dV(0); hold Kt for the whole tile.
      wp_begin(wpc, WP_MMA_WAIT_FULL_Q);
      mbarrier_wait_parity(smem_ptr_u32(&bars[BAR_Q_FULL0]), q_ph.get_phase());
      q_ph.advance();
      wp_end(wpc, WP_MMA_WAIT_FULL_Q);
      issue_st2();
      wp_begin(wpc, WP_MMA_WAIT_FULL_V);
      mbarrier_wait_parity(smem_ptr_u32(&bars[BAR_DO_FULL]), do_ph.get_phase());
      do_ph.advance();
      wp_end(wpc, WP_MMA_WAIT_FULL_V);
      issue_dpt2();
      wp_begin(wpc, WP_MMA_WAIT_FULL_K);
      mbarrier_wait_parity(smem_ptr_u32(&bars[BAR_PT_STTMD]), ptst_ph.get_phase());
      ptst_ph.advance();
      wp_end(wpc, WP_MMA_WAIT_FULL_K);
      issue_dv2(true);
      mbarrier_wait_parity(smem_ptr_u32(&bars[BAR_KT_FULL]), 0);

      for (int j = 0; j < steps; ++j) {
        wp_marker(wpc, WP_ITER, j);
        if (j + 1 < steps) {
          wp_begin(wpc, WP_MMA_WAIT_FULL_Q);
          mbarrier_wait_parity(smem_ptr_u32(&bars[BAR_Q_FULL0]), q_ph.get_phase());
          q_ph.advance();
          wp_end(wpc, WP_MMA_WAIT_FULL_Q);
          wp_begin(wpc, WP_MMA_WAIT_ACC);
          mbarrier_wait_parity(smem_ptr_u32(&bars[BAR_DQC_FREE]), dqcf_ph.get_phase());
          dqcf_ph.advance();
          wp_end(wpc, WP_MMA_WAIT_ACC);
          issue_st2();
        }
        if (j + 1 == steps) commit_mc(BAR_DV_FULL);   // dKV stage 0
        wp_begin(wpc, WP_MMA_WAIT_P);
        mbarrier_wait_parity(smem_ptr_u32(&bars[BAR_QT_FULL]), qt_ph.get_phase());
        qt_ph.advance();
        mbarrier_wait_parity(smem_ptr_u32(&bars[BAR_DPT_EMPTY]), dpte_ph.get_phase());
        dpte_ph.advance();
        wp_end(wpc, WP_MMA_WAIT_P);
        wp_begin(wpc, WP_MMA_ISSUE);
        issue_dk2(j == 0);
        if (j + 1 == steps) commit_mc(BAR_DK_FULL);   // dKV stage 1
        wp_end(wpc, WP_MMA_ISSUE);
        if (j + 1 < steps) {
          wp_begin(wpc, WP_MMA_WAIT_FULL_V);
          mbarrier_wait_parity(smem_ptr_u32(&bars[BAR_DO_FULL]), do_ph.get_phase());
          do_ph.advance();
          wp_end(wpc, WP_MMA_WAIT_FULL_V);
          issue_dpt2();
        }
        wp_begin(wpc, WP_MMA_WAIT_P);
        mbarrier_wait_parity(smem_ptr_u32(&bars[BAR_DST_READY]), dstr_ph.get_phase());
        dstr_ph.advance();
        mbarrier_wait_parity(smem_ptr_u32(&bars[BAR_DSX_LEADER]), dsxl_ph.get_phase());
        dsxl_ph.advance();
        if (j + 1 == steps) {          // tail dQ: no S mma carries the gate
          mbarrier_wait_parity(smem_ptr_u32(&bars[BAR_DQC_FREE]), dqcf_ph.get_phase());
          dqcf_ph.advance();
        }
        wp_end(wpc, WP_MMA_WAIT_P);
        wp_begin(wpc, WP_MMA_ISSUE);
        issue_dqc2();
        wp_end(wpc, WP_MMA_ISSUE);
        if (j + 1 < steps) {
          wp_begin(wpc, WP_MMA_WAIT_FULL_K);
          mbarrier_wait_parity(smem_ptr_u32(&bars[BAR_PT_STTMD]), ptst_ph.get_phase());
          ptst_ph.advance();
          wp_end(wpc, WP_MMA_WAIT_FULL_K);
          issue_dv2(false);
        }
      }
      commit_mc(BAR_KT_EMPTY);         // dangling final Kt release (FA4 form)
      wp_flush(wpc);
      teardown();
      return;
    }
    warp_regs_dec_88();
    tcgen05_alloc<1>(smem_ptr_u32(tmem_slot), 512);
    bar_sync<10>(416);
    const uint32_t tmem_base = *tmem_slot;
    if (cnt == 0) {
      tcgen05_relinquish_alloc_permit<1>();
      bar_sync<10>(416);
      tcgen05_dealloc<1>(tmem_base, 512);
      return;
    }
    const uint32_t lead = elect_one_sync() ? 1u : 0u;
    constexpr uint32_t DESC_SBO = 1024, DESC_LBO = 16;
    constexpr uint32_t Q_LBO_MN   = (uint32_t)Q_SUBTILE_BYTES;
    constexpr uint32_t KV_LBO_MN  = (uint32_t)KV_SUBTILE_BYTES;
    constexpr uint32_t DST_LBO_MN = (uint32_t)DST_SUBTILE_BYTES;
    constexpr uint32_t K16_MN = (uint32_t)((16 * SUB * 2) >> 4);
    constexpr uint64_t Q_SUB_DELTA  = Q_SUBTILE_BYTES >> 4;
    constexpr uint64_t KV_SUB_DELTA = KV_SUBTILE_BYTES >> 4;
    constexpr uint64_t Q_SLOT_DELTA = Q_TILE_BYTES >> 4;

    const uint32_t idesc_st  = make_idesc_bf16_f32(KV_TILE_ROWS, Q_TILE_ROWS, false, false);
    const uint32_t idesc_acc = make_idesc_bf16_f32(KV_TILE_ROWS, HEAD_DIM, false, true);
    const uint32_t idesc_dqc = make_idesc_bf16_f32(Q_TILE_ROWS, HEAD_DIM, true, true);

    const uint64_t desc_k  = build_smem_desc_blackwell(smem_ptr_u32(sK),  DESC_SBO, DESC_LBO, SmemSwizzleBlackwell::B128);
    const uint64_t desc_v  = build_smem_desc_blackwell(smem_ptr_u32(sV),  DESC_SBO, DESC_LBO, SmemSwizzleBlackwell::B128);
    const uint64_t desc_q0 = build_smem_desc_blackwell(smem_ptr_u32(sQ[0]), DESC_SBO, DESC_LBO, SmemSwizzleBlackwell::B128);
    const uint64_t desc_do = build_smem_desc_blackwell(smem_ptr_u32(sDO), DESC_SBO, DESC_LBO, SmemSwizzleBlackwell::B128);
    const uint64_t desc_q0_mn = build_smem_desc_blackwell(smem_ptr_u32(sQ[0]), DESC_SBO, Q_LBO_MN, SmemSwizzleBlackwell::B128);
    const uint64_t desc_do_mn = build_smem_desc_blackwell(smem_ptr_u32(sDO), DESC_SBO, Q_LBO_MN, SmemSwizzleBlackwell::B128);
    const uint64_t desc_k_mn   = build_smem_desc_blackwell(smem_ptr_u32(sK),  DESC_SBO, KV_LBO_MN, SmemSwizzleBlackwell::B128);
    const uint64_t desc_dst_mn = build_smem_desc_blackwell(smem_ptr_u32(sDST), DESC_SBO, DST_LBO_MN, SmemSwizzleBlackwell::B128);

    PhaseTracker<2> q_ph;
    PhaseTracker<1> do_ph;
    PhaseTracker<1> ptst_ph, dstr_ph, dqcf_ph;

    auto issue_st = [&](int sd) {
      const uint64_t dq = desc_q0 + (uint64_t)sd * Q_SLOT_DELTA;
      #pragma unroll
      for (int ssub = 0; ssub < 2; ++ssub)
        #pragma unroll
        for (int ki = 0; ki < 4; ++ki)
          tcgen05_mma_f16_ss_lead(lead, tmem_base + T_ST,
                                  desc_k + ssub * KV_SUB_DELTA + 2 * ki,
                                  dq + ssub * Q_SUB_DELTA + 2 * ki,
                                  idesc_st, (ssub | ki) != 0);
      tcgen05_commit1_lead(lead, smem_ptr_u32(&bars[BAR_ST_READY]));
    };
    auto issue_dpt = [&]() {
      #pragma unroll
      for (int ssub = 0; ssub < 2; ++ssub)
        #pragma unroll
        for (int ki = 0; ki < 4; ++ki)
          tcgen05_mma_f16_ss_lead(lead, tmem_base + T_DPT,
                                  desc_v + ssub * KV_SUB_DELTA + 2 * ki,
                                  desc_do + ssub * Q_SUB_DELTA + 2 * ki,
                                  idesc_st, (ssub | ki) != 0);
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

    // Prologue: S(0), dP(0), dV(0). No DQC_FREE gate on dP(0) -- one tile
    // per CTA, the 256-383 slot starts free.
    {
      const int sd0 = q_ph.get_stage() & 1;
      wp_begin(wpc, WP_MMA_WAIT_FULL_Q);
      mbarrier_wait_parity(smem_ptr_u32(&bars[BAR_Q_FULL0 + sd0]), q_ph.get_phase());
      wp_end(wpc, WP_MMA_WAIT_FULL_Q);
      issue_st(sd0);
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
      // FA4 dKV stage 0: all dV accumulations are issued once the loop enters
      // its last iteration -- commit dV so the epilogue drains it while the
      // tail dK/dQc still run.
      if (j + 1 == steps)
        tcgen05_commit1_lead(lead, smem_ptr_u32(&bars[BAR_DV_FULL]));
      wp_begin(wpc, WP_MMA_WAIT_P);
      mbarrier_wait_parity(smem_ptr_u32(&bars[BAR_DST_READY]), dstr_ph.get_phase());
      dstr_ph.advance();
      wp_end(wpc, WP_MMA_WAIT_P);
      wp_begin(wpc, WP_MMA_ISSUE);
      issue_dk(sd, j == 0);
      if (j + 1 == steps)
        tcgen05_commit1_lead(lead, smem_ptr_u32(&bars[BAR_DK_FULL]));
      issue_dqc();
      wp_end(wpc, WP_MMA_ISSUE);
      if (j + 1 < steps) {
        wp_begin(wpc, WP_MMA_WAIT_FULL_V);
        mbarrier_wait_parity(smem_ptr_u32(&bars[BAR_DO_FULL]), do_ph.get_phase());
        do_ph.advance();
        wp_end(wpc, WP_MMA_WAIT_FULL_V);
        wp_begin(wpc, WP_MMA_WAIT_ACC);
        mbarrier_wait_parity(smem_ptr_u32(&bars[BAR_DQC_FREE]), dqcf_ph.get_phase());
        dqcf_ph.advance();
        wp_end(wpc, WP_MMA_WAIT_ACC);
        issue_dpt();
        wp_begin(wpc, WP_MMA_WAIT_FULL_K);
        mbarrier_wait_parity(smem_ptr_u32(&bars[BAR_PT_STTMD]), ptst_ph.get_phase());
        ptst_ph.advance();
        wp_end(wpc, WP_MMA_WAIT_FULL_K);
        issue_dv(false);
      }
    }
    wp_flush(wpc);
    tcgen05_relinquish_alloc_permit<1>();
    bar_sync<10>(416);
    tcgen05_dealloc<1>(tmem_base, 512);
    return;
  }
  else if (warp_id >= W_COMPUTE0) {
    warp_regs_inc_136();
    bar_sync<10>(416);
    const uint32_t tmem_base = *tmem_slot;
    if (cnt == 0) {
      // FA4 zero-fill: no q block selects this kv block -> dK/dV are zero.
      const int zcw = warp_id - W_COMPUTE0;
      const int zsubp = zcw & 3;
      const int zq_half = zcw >> 2;
      const int zrow = zsubp * 32 + lane;
      __nv_bfloat16* zsub = sDST + (size_t)zq_half * KV_TILE_ROWS * SUB;
      const uint4 z = make_uint4(0u, 0u, 0u, 0u);
      #pragma unroll
      for (int v = 0; v < 8; ++v)
        *reinterpret_cast<uint4*>(&zsub[zrow * SUB + v * 8]) = z;
      fence_proxy_async_shared_cta();
      bar_sync<14>(256);
      if (warp_id == W_COMPUTE0 && elect_one_sync()) {
        #pragma unroll
        for (int which = 0; which < 2; ++which) {
          const CUtensorMap* map = (which == 0) ? &tmap_dk : &tmap_dv;
          #pragma unroll
          for (int ssub = 0; ssub < 2; ++ssub)
            tma_store_3d(map, 0, n_block * KV_TILE_ROWS, bh * 2 + ssub,
                         smem_ptr_u32(sDST + (size_t)ssub * KV_TILE_ROWS * SUB));
        }
        cp_async_bulk_commit_group();
        cp_async_bulk_wait_group_read<0>();
      }
      bar_sync<14>(256);
      bar_sync<10>(416);
      return;
    }
    // Compute warps (4-11, FA4's 2 warpgroups), col-split over the 128x128
    // tile (2cta: this CTA's 128 of the cluster's 256 kv rows): warp cw
    // covers kv subpartition cw&3 (lanes = kv rows) and q half cw>>2 (64 of
    // 128 cols). They also run the dK/dV epilogue at tile end (FA4's role
    // split; valid under non-persistence).
    const int cw = warp_id - W_COMPUTE0;
    const int subp = cw & 3;
    const int q_half = cw >> 2;
    const uint32_t lane_base = (uint32_t)((subp * 32) << 16);
    const int row = subp * 32 + lane;
    const uint32_t col_off = (uint32_t)(q_half * SUB);
    __nv_bfloat16* sDSTsub = sDST + (size_t)q_half * KV_TILE_ROWS * SUB;   // 1cta dst

    // The trackers are 1cta-only; the 2cta waits derive their parity from j
    // directly (pj) so no tracker state stays live across the x64 t2r.
    PhaseTracker<1> str_ph, dptr_ph, dpsf_ph, dvf_ph, dkf_ph;
    PhaseTracker<TWO_CTA ? 1 : 2> lsef_ph;
    int md_slot = 0;

    for (int j = 0; j < steps; ++j) {
      wp_marker(wpc, WP_ITER, j);
      const uint32_t pj = (uint32_t)(j & 1);    // 2cta: all waits are 1-stage
      const int slot = TWO_CTA ? 0 : md_slot;
      md_slot ^= 1;
      const float* m_smem = sLSE + slot * Q_TILE_ROWS + col_off;
      const float* delta_smem = sDPS + col_off;

      // FA4 order: LSE first, then S.
      mbarrier_wait_parity_suspend(smem_ptr_u32(&bars[BAR_LSE_FULL0 + slot]),
                                   TWO_CTA ? pj : lsef_ph.get_phase());
      if (!TWO_CTA) lsef_ph.advance();

      wp_begin(wpc, WP_SM_WAIT_S);
      mbarrier_wait_parity_suspend(smem_ptr_u32(&bars[BAR_ST_READY]),
                                   TWO_CTA ? pj : str_ph.get_phase());
      if (!TWO_CTA) str_ph.advance();
      wp_end(wpc, WP_SM_WAIT_S);
      wp_begin(wpc, WP_SM_SOFTMAX);

      uint32_t st_regs[64];
      uint32_t pt_pack[32];
      float* pt = reinterpret_cast<float*>(st_regs);   // PT overwrites ST in place
      tcgen05_ld_32x32b_x64(tmem_base + T_ST + col_off + lane_base, st_regs);
      tcgen05_fence_before_thread_sync();
      if (TWO_CTA && j > 0 && elect_one_sync())
        mbarrier_arrive_cluster_default(                    // deferred dS(j-1) full
            mapa_cluster_u32(smem_ptr_u32(&bars[BAR_DST_READY]), 0));
      {
        // Packed f32x2 affine (FA4 form): z2 = st2 * scale - m2 in one
        // ffma2; exp2 stays on the HW SFU.
        const float2 scale2 = f32x2_splat(scale_log2);
        #pragma unroll
        for (int c0 = 0; c0 < 64; c0 += 4) {
          const float4 m4 = *reinterpret_cast<const float4*>(m_smem + c0);
          const float mv[4] = { -m4.x, -m4.y, -m4.z, -m4.w };
          #pragma unroll
          for (int c = c0; c < c0 + 4; c += 2) {
            const float2 z2 = ffma2(make_float2(pt[c], pt[c + 1]), scale2,
                                    make_float2(mv[c - c0], mv[c - c0 + 1]));
            const float p0 = ex2_approx_f32(z2.x);
            const float p1 = ex2_approx_f32(z2.y);
            pt[c] = p0; pt[c + 1] = p1;
            pt_pack[c / 2] = cvt_f32x2_to_bf16x2(p0, p1);
          }
        }
      }
      if (TWO_CTA) bar_sync<14>(256);   // packed P: cross-half STTM vs S t2r
      tcgen05_st_32x32b_x32(tmem_base + T_PT_BF16
                              + (uint32_t)(q_half * (TWO_CTA ? 32 : 64)) + lane_base,
                            pt_pack);
      tcgen05_wait_st();
      tcgen05_fence_before_thread_sync();
      if (elect_one_sync()) {
        if (TWO_CTA)
          mbarrier_arrive_cluster_default(
              mapa_cluster_u32(smem_ptr_u32(&bars[BAR_PT_STTMD]), 0));
        else mbarrier_arrive(smem_ptr_u32(&bars[BAR_PT_STTMD]));      // S_P empty
        mbarrier_arrive(smem_ptr_u32(&bars[BAR_LSE_EMPTY0 + slot]));  // LSE ring (8)
      }
      wp_end(wpc, WP_SM_SOFTMAX);

      wp_begin(wpc, WP_CORR_WAIT);
      mbarrier_wait_parity_suspend(smem_ptr_u32(&bars[BAR_DPSUM_FULL]),
                                   TWO_CTA ? pj : dpsf_ph.get_phase());
      if (!TWO_CTA) dpsf_ph.advance();
      mbarrier_wait_parity_suspend(smem_ptr_u32(&bars[BAR_DPT_READY]),
                                   TWO_CTA ? pj : dptr_ph.get_phase());
      if (!TWO_CTA) dptr_ph.advance();
      wp_end(wpc, WP_CORR_WAIT);
      wp_begin(wpc, WP_SM_STORE_P);
      if (TWO_CTA) {
        // dS producer_acquire: sDST regions (incl. the peer's landing zone)
        // and sXCHG must be free -- released by the leader's multicast commit
        // after the dQ mma read both CTAs' sdS.
        mbarrier_wait_parity_suspend(smem_ptr_u32(&bars[BAR_DST_EMPTY]), pj ^ 1u);
      }
      {
        // Single x64 dPT t2r at the 136-reg compute budget (FA4's).
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
      tcgen05_st_32x32b_x32(tmem_base + T_DST_BF16 + (uint32_t)(q_half * 64) + lane_base,
                            reinterpret_cast<const uint32_t(&)[32]>(st_regs));
      {
        // 2cta: this warpgroup's dS q-half either stays local (own kv-half
        // region of sDST) or ships to the peer via sXCHG (exchange = rank^1).
        __nv_bfloat16* dsub = !TWO_CTA ? sDSTsub
            : (q_half == cta_rank ? sDST + (size_t)cta_rank * KV_TILE_ROWS * SUB
                                  : sXCHG);
        const uint4* src4 = reinterpret_cast<const uint4*>(st_regs);
        #pragma unroll
        for (int v = 0; v < 8; ++v)
          *reinterpret_cast<uint4*>(&dsub[row * SUB + ((v ^ (row & 7)) * 8)]) = src4[v];
      }
      tcgen05_wait_st();
      tcgen05_fence_before_thread_sync();
      fence_proxy_async_shared_cta();
      if (elect_one_sync()) {
        if (TWO_CTA)                                                     // dS STTM'd
          mbarrier_arrive_cluster_default(
              mapa_cluster_u32(smem_ptr_u32(&bars[BAR_DPT_EMPTY]), 0));
        else mbarrier_arrive(smem_ptr_u32(&bars[BAR_DST_READY]));        // dS full (8)
        mbarrier_arrive(smem_ptr_u32(&bars[BAR_DPSUM_EMPTY]));           // dPsum (8)
      }
      if (TWO_CTA) {
        // dS exchange (FA4): after all 8 warps staged, one thread pushes the
        // 16KB sXCHG image into the PEER's sDST at region[cta_rank] and arms
        // the peer's DSX_FULL with the tx; the peer's relay certifies it.
        bar_sync<14>(256);
        if (warp_id == W_COMPUTE0 && elect_one_sync()) {
          const uint32_t peer = (uint32_t)(cta_rank ^ 1);
          const uint32_t peer_full =
              mapa_cluster_u32(smem_ptr_u32(&bars[BAR_DSX_FULL]), peer);
          mbarrier_arrive_expect_tx_cluster(peer_full, DST_SUBTILE_BYTES);
          bulk_s2cluster(
              mapa_cluster_u32(smem_ptr_u32(sDST + (size_t)cta_rank * KV_TILE_ROWS * SUB), peer),
              smem_ptr_u32(sXCHG), DST_SUBTILE_BYTES, peer_full);
        }
      }
      wp_end(wpc, WP_SM_STORE_P);
    }
    if (TWO_CTA && elect_one_sync())
      mbarrier_arrive_cluster_default(                     // final deferred dS full
          mapa_cluster_u32(smem_ptr_u32(&bars[BAR_DST_READY]), 0));

    // dK/dV epilogue (FA4: compute warps, dV FIRST as dKV stage 0 so it
    // drains while the MMA tail runs, then dK as stage 1). Per-warpgroup:
    // wg = q_half owns its 64-col d-half, bounces through the then-free
    // buffer (1cta: dV -> sDO, dK -> sQ stage 0; 2cta: dV -> sV, dK -> sK)
    // and stores its own subtile. No bulk-group wait: nothing reuses the
    // bounces before CTA exit (FA4 form).
    {
      auto drain_acc = [&](uint32_t t_base, float scale, __nv_bfloat16* bounce,
                           const CUtensorMap* map) {
        __nv_bfloat16* bsub = bounce + (size_t)q_half * KV_TILE_ROWS * SUB;
        #pragma unroll
        for (int c0 = 0; c0 < SUB; c0 += 32) {
          uint32_t acc_regs[32];
          tcgen05_ld_32x32b_x32(t_base + col_off + (uint32_t)c0 + lane_base, acc_regs);
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
            *reinterpret_cast<uint4*>(&bsub[row * SUB + ((vv ^ (row & 7)) * 8)]) = packed;
          }
        }
        fence_proxy_async_shared_cta();
        if (q_half == 0) bar_sync<12>(128); else bar_sync<13>(128);
        if (cw == q_half * 4 && elect_one_sync()) {
          tma_store_3d(map, 0, n_block * KV_TILE_ROWS, bh * 2 + q_half,
                       smem_ptr_u32(bsub));
          cp_async_bulk_commit_group();
        }
      };
      wp_begin(wpc, WP_EPI_WAIT_STORE);
      mbarrier_wait_parity_suspend(smem_ptr_u32(&bars[BAR_DV_FULL]), dvf_ph.get_phase());
      dvf_ph.advance();
      wp_end(wpc, WP_EPI_WAIT_STORE);
      wp_begin(wpc, WP_CORR_EPI);
      // 2cta: sQ/sDO are only 16KB -- bounce through sV/sK (FA4: sdV overlays
      // sV, sdK overlays sK; both dead after the last dV/S mma).
      drain_acc(tmem_base + T_DV, 1.0f, TWO_CTA ? sV : sDO, &tmap_dv);
      mbarrier_wait_parity_suspend(smem_ptr_u32(&bars[BAR_DK_FULL]), dkf_ph.get_phase());
      dkf_ph.advance();
      drain_acc(tmem_base + T_DK, sm_scale, TWO_CTA ? sK : sQ[0], &tmap_dk);
      wp_end(wpc, WP_CORR_EPI);
    }
    wp_flush(wpc);
    bar_sync<10>(416);
    return;
  }
  else {
    if constexpr (TWO_CTA) {
      // FA4 2cta reduce: per-CTA dQ = own 64 q rows x 128 hd at TMEM cols
      // 64..127 (lane = q_loc + 64*(d/64)) -> 2 x32 t2r = 64 f32/thread at
      // the 136-reg budget. 8 chunks x 4KB through a 4-stage SMEM ring;
      // CTA rank r owns gmem chunks [8r, 8r+8) of the m_block's 16K-f32
      // slice (disjoint: no cross-CTA combining, cluster_reduce_dQ=False).
      warp_regs_inc_136();
      bar_sync<10>(416);
      const uint32_t tmem_base = *tmem_slot;
      if (cnt == 0) {
        bar_sync<10>(416);
        return;
      }
      const int subp = warp_id;
      const uint32_t lane_base = (uint32_t)((subp * 32) << 16);
      const int row = subp * 32 + lane;
      const uint32_t dqc_free_leader =
          mapa_cluster_u32(smem_ptr_u32(&bars[BAR_DQC_FREE]), 0);
      PhaseTracker<1> dqcp_ph;

      for (int j = 0; j < steps; ++j) {
        wp_marker(wpc, WP_ITER, j);
        const int q_blk = 2 * (int)pair_union[beg + (j >> 1)] + (j & 1);
        const float* gq = dqaccum + (size_t)bh * seqlen * HEAD_DIM
                        + (size_t)q_blk * Q_TILE_ROWS * HEAD_DIM;

        wp_begin(wpc, WP_EPI_WAIT_ACC);
        mbarrier_wait_parity(smem_ptr_u32(&bars[BAR_DQC_FULL]), dqcp_ph.get_phase());
        dqcp_ph.advance();
        wp_end(wpc, WP_EPI_WAIT_ACC);

        wp_begin(wpc, WP_EPI_TMEM_LD);
        uint32_t qc_regs[64];
        #pragma unroll
        for (int c = 0; c < 2; ++c)
          tcgen05_ld_32x32b_x32(tmem_base + T_DQC_2 + (uint32_t)(c * 32) + lane_base,
                                reinterpret_cast<uint32_t(&)[32]>(qc_regs[c * 32]));
        tcgen05_fence_before_thread_sync();
        if (elect_one_sync())
          mbarrier_arrive_cluster_default(dqc_free_leader);
        wp_end(wpc, WP_EPI_TMEM_LD);

        wp_begin(wpc, WP_EPI_STORE);
        #pragma unroll
        for (int chunk = 0; chunk < 8; ++chunk) {
          const int buf = chunk & 3;
          {
            // chunk c holds values v = 32*(c/4) + (c%4)*8 + 0..7; float4 r
            // of row t at smem[r*512 + t*4] (FA4 staging).
            const float4* q4 = reinterpret_cast<const float4*>(
                qc_regs + 32 * (chunk >> 2) + (chunk & 3) * 8);
            *reinterpret_cast<float4*>(sDQA + buf * 1024 + 0 * 512 + row * 4) = q4[0];
            *reinterpret_cast<float4*>(sDQA + buf * 1024 + 1 * 512 + row * 4) = q4[1];
          }
          fence_proxy_async_shared_cta();
          bar_sync<11>(128);
          if (warp_id == 0 && elect_one_sync()) {
            bulk_reduce_add_f32(gq + (size_t)(8 * cta_rank + chunk) * 1024,
                                smem_ptr_u32(sDQA + buf * 1024), DQA2_STAGE_BYTES);
            cp_async_bulk_commit_group();
            cp_async_bulk_wait_group_read<3>();
          }
          bar_sync<11>(128);
        }
        wp_end(wpc, WP_EPI_STORE);
      }
      if (warp_id == 0 && elect_one_sync()) cp_async_bulk_wait_group_read<0>();
      bar_sync<11>(128);
      wp_flush(wpc);
      // All dQ pushes issued: the postprocess may launch (its wait still covers their completion).
      if constexpr (KERNEL_PDL) {
        if (warp_id == 0 && elect_one_sync()) griddepcontrol_launch_dependents();
      }
      bar_sync<10>(416);
      return;
    }
    warp_regs_inc_152();
    bar_sync<10>(416);
    const uint32_t tmem_base = *tmem_slot;
    if (cnt == 0) {
      bar_sync<10>(416);
      return;
    }
    // dQ reduce warps (0-3, FA4's role and shape): warp = TMEM subpartition
    // (lane = q row). Whole-tile t2r (4 x32 LDTMs) first -- DQC_FREE fires
    // right after -- then 4 chunks x 16KB [128 rows x 32 cols] from 2
    // rotating buffers, one 128-thread domain (FA4's dQaccReduce), one bulk
    // push per chunk with wait_group(1, read).
    const int subp = warp_id;
    const uint32_t lane_base = (uint32_t)((subp * 32) << 16);
    const int row = subp * 32 + lane;

    PhaseTracker<1> dqcp_ph;

    for (int j = 0; j < steps; ++j) {
      wp_marker(wpc, WP_ITER, j);
      const int q_blk = 2 * (int)pair_union[beg + (j >> 1)] + (j & 1);
      const float* gq = dqaccum + (size_t)bh * seqlen * HEAD_DIM
                      + (size_t)q_blk * Q_TILE_ROWS * HEAD_DIM;

      // Spin (not suspend): DQC_FULL -> t2r -> DQC_FREE gates dP(j+1).
      wp_begin(wpc, WP_EPI_WAIT_ACC);
      mbarrier_wait_parity(smem_ptr_u32(&bars[BAR_DQC_FULL]), dqcp_ph.get_phase());
      dqcp_ph.advance();
      wp_end(wpc, WP_EPI_WAIT_ACC);

      wp_begin(wpc, WP_EPI_TMEM_LD);
      // FA4's whole-tile t2r at the 152-reg budget: four x32 LDTMs
      // back-to-back (128 live regs; a single x128 op exceeds ptxas'
      // per-instruction limit), TMEM released immediately, then the 4-chunk
      // staging loop runs entirely from registers (FA4's exact shape).
      uint32_t qc_regs[128];
      #pragma unroll
      for (int c = 0; c < 4; ++c)
        tcgen05_ld_32x32b_x32(tmem_base + T_DQC + (uint32_t)(c * 32) + lane_base,
                              reinterpret_cast<uint32_t(&)[32]>(qc_regs[c * 32]));
      tcgen05_fence_before_thread_sync();
      if (elect_one_sync())
        mbarrier_arrive(smem_ptr_u32(&bars[BAR_DQC_FREE]));
      wp_end(wpc, WP_EPI_TMEM_LD);

      wp_begin(wpc, WP_EPI_STORE);
      #pragma unroll
      for (int chunk = 0; chunk < HEAD_DIM / DQC_CHUNK_COLS; ++chunk) {
        const int cb = chunk & 1;
        {
          // FA4 staging layout: reg float4 v4 of row t at smem[v4*512+t*4].
          const float4* q4 = reinterpret_cast<const float4*>(qc_regs + chunk * DQC_CHUNK_COLS);
          #pragma unroll
          for (int v4 = 0; v4 < DQC_CHUNK_COLS / 4; ++v4)
            *reinterpret_cast<float4*>(sDQC[cb] + v4 * 512 + row * 4) = q4[v4];
        }
        fence_proxy_async_shared_cta();
        bar_sync<11>(128);
        if (warp_id == 0 && elect_one_sync()) {
          bulk_reduce_add_f32(gq + (size_t)chunk * Q_TILE_ROWS * DQC_CHUNK_COLS,
                              smem_ptr_u32(sDQC[cb]), DQC_CHUNK_BYTES);
          cp_async_bulk_commit_group();
          cp_async_bulk_wait_group_read<1>();
        }
        bar_sync<11>(128);
      }
      wp_end(wpc, WP_EPI_STORE);
    }
    // Bulk reads must complete before SMEM dies with the CTA.
    if (warp_id == 0 && elect_one_sync()) cp_async_bulk_wait_group_read<0>();
    bar_sync<11>(128);
    wp_flush(wpc);
    // All dQ pushes issued: the postprocess may launch (its wait still covers their completion).
    if constexpr (KERNEL_PDL) {
      if (warp_id == 0 && elect_one_sync()) griddepcontrol_launch_dependents();
    }
    bar_sync<10>(416);
    return;
  }
}

// Preprocess (FA4 shape): one CTA per (q tile, head), 256 threads.
// Zeroes the tile's own dQaccum slice (coalesced), then Delta = rowsum
// (bf16(O) * dO) with 16 threads per row x 8 columns each + shfl reduce.
__global__ void __launch_bounds__(256, 1)
vsa_bwd_preprocess_kernel(const __nv_bfloat16* __restrict__ o,
                          const __nv_bfloat16* __restrict__ dout,
                          float* __restrict__ delta_rows,
                          float* __restrict__ dqaccum,
                          int num_heads, int seqlen) {
  const int m_block = (int)blockIdx.x;
  const int bh = (int)blockIdx.y;
  const int t0 = m_block * Q_TILE_ROWS;
  // The previous postprocess still reads dqaccum: wait before zeroing it.
  if constexpr (KERNEL_PDL) griddepcontrol_wait();
  // Zero this tile's dQaccum slice: 16384 f32 = 4096 float4 by 256 threads.
  float4* z4 = reinterpret_cast<float4*>(
      dqaccum + (size_t)bh * seqlen * HEAD_DIM + (size_t)m_block * Q_TILE_ROWS * HEAD_DIM);
  const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
  #pragma unroll
  for (int i = 0; i < (Q_TILE_ROWS * HEAD_DIM / 4) / 256; ++i)
    z4[i * 256 + threadIdx.x] = z;
  // dqaccum is zeroed: the main grid may set up while Delta forms.
  if constexpr (KERNEL_PDL) griddepcontrol_launch_dependents();
  // Delta: 16 threads per row, 8 cols each (2 x uint4 of bf16).
  const int row = (int)threadIdx.x / 16;          // 0..15 rows per pass, 8 passes
  const int col16 = ((int)threadIdx.x % 16) * 8;  // 8 bf16 per thread
  #pragma unroll
  for (int rpass = 0; rpass < Q_TILE_ROWS / 16; ++rpass) {
    const int t = t0 + rpass * 16 + row;
    const long base = ((long)t * num_heads + bh) * HEAD_DIM + col16;
    const uint4 ob = *reinterpret_cast<const uint4*>(o + base);
    const uint4 db = *reinterpret_cast<const uint4*>(dout + base);
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
}

// Postprocess -- FA4's exact scheme (fa4_bwd_postprocess.py):
// one CTA per (q tile, head), 128 threads, 64KB SMEM.
//   A. G2S: the whole 16384-f32 drain-native tile loaded CONTIGUOUSLY via
//      cp.async.cg (32 x 16B per thread), wait, barrier.
//   B. S2R unscramble: thread t = row t; for chunk j, group c: float4 at
//      smem[j*4096 + c*512 + t*4] -- warp reads 512B contiguous, no
//      conflicts (this replays the 1cta reduce warps' staging order).
//      B' (two_cta): per-rank 8-chunk x 4KB scramble -- see the inline
//      formula in the branch.
//   C. * sm_scale, cvt to bf16.
//   D. R2S: row t as 16 x 16B into a bf16 overlay of the SAME smem (after a
//      barrier; slots xor-swizzled by t&7 so the strided row stores are
//      conflict-free).
//   E. S2G: FA4's tiled store layout -- 16 threads x 8 bf16 per row, 8 rows
//      per pass: every gmem row segment is 256B contiguous.
__global__ void __launch_bounds__(128, 1)
vsa_bwd_postprocess_kernel(const float* __restrict__ dqaccum,
                           __nv_bfloat16* __restrict__ dq,
                           int num_heads, int seqlen, float sm_scale,
                           int two_cta) {
  extern __shared__ __align__(16) float post_smem[];   // 16384 f32 / bf16 overlay
  const int m_block = (int)blockIdx.x;
  const int bh = (int)blockIdx.y;
  const int t0 = m_block * Q_TILE_ROWS;
  const int tid = (int)threadIdx.x;

  // A: contiguous G2S
  const float4* src = reinterpret_cast<const float4*>(
      dqaccum + (size_t)bh * seqlen * HEAD_DIM + (size_t)m_block * Q_TILE_ROWS * HEAD_DIM);
  if constexpr (KERNEL_PDL) griddepcontrol_wait();  // the main grid's pushes are complete
  #pragma unroll
  for (int i = 0; i < (Q_TILE_ROWS * HEAD_DIM / 4) / 128; ++i) {
    const uint32_t dst_sm = smem_ptr_u32(post_smem + (i * 128 + tid) * 4);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(dst_sm), "l"(src + i * 128 + tid) : "memory");
  }
  asm volatile("cp.async.commit_group;\n" ::: "memory");
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
  __syncthreads();
  // dqaccum read out: the next preprocess may launch (it waits before re-zeroing).
  if constexpr (KERNEL_PDL) griddepcontrol_launch_dependents();

  // B+C: unscramble row `tid` into registers, scale, pack bf16
  uint32_t row_bf16[32];   // low 64 d-lanes as 32 packed bf16x2
  uint32_t row_hi[32];     // d 64..127
  if (two_cta) {
    // 2cta drain scramble (per q row `tid`): CTA rank q/64 wrote chunks
    // [8*(q/64), +8); chunk = ((d%64)/32)*4 + ((d%32)/8); within-chunk
    // offset = ((d%8)/4)*512 + (q%64 + 64*(d/64))*4 + d%4.
    #pragma unroll
    for (int d0 = 0; d0 < HEAD_DIM; d0 += 4) {
      const int chunk = ((d0 & 63) >> 5) * 4 + ((d0 & 31) >> 3);
      const int off = (tid >> 6) * 8192 + chunk * 1024 + (((d0 & 7) >> 2) << 9)
                    + ((tid & 63) + ((d0 >> 6) << 6)) * 4;
      const float4 v = *reinterpret_cast<const float4*>(post_smem + off);
      uint32_t* dst = (d0 < 64) ? row_bf16 : row_hi;
      const int o = (d0 & 63) / 2;
      dst[o + 0] = cvt_f32x2_to_bf16x2(v.x * sm_scale, v.y * sm_scale);
      dst[o + 1] = cvt_f32x2_to_bf16x2(v.z * sm_scale, v.w * sm_scale);
    }
  } else {
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      #pragma unroll
      for (int c = 0; c < 8; ++c) {
        const float4 v = *reinterpret_cast<const float4*>(post_smem + j * 4096 + c * 512 + tid * 4);
        const int d0 = j * 32 + c * 4;      // ascending d
        uint32_t* dst = (d0 < 64) ? row_bf16 : row_hi;
        const int o = (d0 & 63) / 2;
        dst[o + 0] = cvt_f32x2_to_bf16x2(v.x * sm_scale, v.y * sm_scale);
        dst[o + 1] = cvt_f32x2_to_bf16x2(v.z * sm_scale, v.w * sm_scale);
      }
    }
  }
  __syncthreads();

  // D: r2s row `tid` (xor-swizzled 16B slots)
  __nv_bfloat16* smb = reinterpret_cast<__nv_bfloat16*>(post_smem);
  {
    const uint4* lo = reinterpret_cast<const uint4*>(row_bf16);
    const uint4* hi = reinterpret_cast<const uint4*>(row_hi);
    #pragma unroll
    for (int v = 0; v < 8; ++v)
      *reinterpret_cast<uint4*>(&smb[tid * HEAD_DIM + ((v ^ (tid & 7)) * 8)]) = lo[v];
    #pragma unroll
    for (int v = 0; v < 8; ++v)
      *reinterpret_cast<uint4*>(&smb[tid * HEAD_DIM + 64 + ((v ^ (tid & 7)) * 8)]) = hi[v];
  }
  __syncthreads();

  // E: tiled S2G, 16 threads x 8 bf16 per row, 8 rows per pass
  const int srow = tid / 16;                // 0..7
  const int scol = (tid % 16) * 8;          // 8 bf16 per thread
  #pragma unroll
  for (int rpass = 0; rpass < Q_TILE_ROWS / 8; ++rpass) {
    const int r = rpass * 8 + srow;
    const int half = scol / 64;
    const int slot = ((scol % 64) / 8) ^ (r & 7);
    const uint4 v = *reinterpret_cast<const uint4*>(&smb[r * HEAD_DIM + half * 64 + slot * 8]);
    *reinterpret_cast<uint4*>(dq + ((size_t)(t0 + r) * num_heads + bh) * HEAD_DIM + scol) = v;
  }
}

// ---------------------------------------------------------------------------
// Host launchers (stream-chained: pre -> main -> post, FA4's timed path).
// ---------------------------------------------------------------------------

struct VsaBwdArgs {
  const __nv_bfloat16 *q, *k, *v, *dout;
  float* dqaccum;                 // drain-native fp32 scratch (pre zeroes it)
  __nv_bfloat16 *dk, *dv, *dq;    // [S, H, 128] bf16 (token-major, B folded)
  const float *m_rows, *delta_rows;   // [BH, S]
  const int* pair_offset;             // [BH*nb256 + 1]
  const unsigned* pair_union;         // plain q256 ids
  int num_heads, seqlen, num_blocks;  // num_blocks = kv blocks (S/256)
  bool two_cta;
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
  const long n_tokens = (long)S;

  // Tensormaps are pure functions of (pointer, shape): encode once per
  // config (FA4 builds its TMA atoms once at compile time; re-encoding per
  // launch costs ~100+ us of driver time per call).
  static CUtensorMap tq_, tk_, tv_, tdo_, tdk_, tdv_;
  static const void* cached_q = nullptr;
  static int cached_S = 0, cached_H = 0;
  static CUtensorMap tq64_, tdo64_, tk256_;   // 2cta per-CTA slice boxes
  if (cached_q != (const void*)a.q || cached_S != S || cached_H != H) {
    if (vsa_bwd_encode_bf16_3d(&tq_, a.q, H, n_tokens, Q_TILE_ROWS) != cudaSuccess) return cudaErrorInvalidValue;
    if (vsa_bwd_encode_bf16_3d(&tk_, a.k, H, n_tokens, KV_TILE_ROWS) != cudaSuccess) return cudaErrorInvalidValue;
    if (vsa_bwd_encode_bf16_3d(&tv_, a.v, H, n_tokens, KV_TILE_ROWS) != cudaSuccess) return cudaErrorInvalidValue;
    if (vsa_bwd_encode_bf16_3d(&tdo_, a.dout, H, n_tokens, Q_TILE_ROWS) != cudaSuccess) return cudaErrorInvalidValue;
    if (vsa_bwd_encode_bf16_3d(&tdk_, a.dk, H, n_tokens, KV_TILE_ROWS) != cudaSuccess) return cudaErrorInvalidValue;
    if (vsa_bwd_encode_bf16_3d(&tdv_, a.dv, H, n_tokens, KV_TILE_ROWS) != cudaSuccess) return cudaErrorInvalidValue;
    if (vsa_bwd_encode_bf16_3d(&tq64_, a.q, H, n_tokens, 64) != cudaSuccess) return cudaErrorInvalidValue;
    if (vsa_bwd_encode_bf16_3d(&tdo64_, a.dout, H, n_tokens, 64) != cudaSuccess) return cudaErrorInvalidValue;
    if (vsa_bwd_encode_bf16_3d(&tk256_, a.k, H, n_tokens, 256) != cudaSuccess) return cudaErrorInvalidValue;
    cached_q = (const void*)a.q; cached_S = S; cached_H = H;
  }
  static bool smem_set = false;
  if (!smem_set) {
    cudaError_t e = cudaFuncSetAttribute(vsa_bwd_main_kernel<false>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         SMEM_TOTAL);
    if (e != cudaSuccess) return e;
    e = cudaFuncSetAttribute(vsa_bwd_main_kernel<true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             SMEM_TOTAL_2);
    if (e != cudaSuccess) return e;
    smem_set = true;
  }

  const float scale_log2 = a.sm_scale * 1.4426950408889634f;
  const unsigned grid_x = (unsigned)(2 * a.num_blocks);   // kv tiles (128-row)
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid_x, (unsigned)H, 1);
  cfg.blockDim = dim3(N_WARPS * 32, 1, 1);
  cfg.dynamicSmemBytes = a.two_cta ? SMEM_TOTAL_2 : SMEM_TOTAL;
  cfg.stream = stream;
  cudaLaunchAttribute at[2];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = a.two_cta ? 2 : 1; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[1].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at; cfg.numAttrs = KERNEL_PDL ? 2 : 1;
  return a.two_cta
      ? cudaLaunchKernelEx(&cfg, vsa_bwd_main_kernel<true>,
            tq_, tk_, tv_, tdo_, tdk_, tdv_, tq64_, tdo64_, tk256_,
            a.dqaccum, a.m_rows, a.delta_rows, a.pair_offset, a.pair_union,
            H, S, a.num_blocks, scale_log2, a.sm_scale)
      : cudaLaunchKernelEx(&cfg, vsa_bwd_main_kernel<false>,
            tq_, tk_, tv_, tdo_, tdk_, tdv_, tq64_, tdo64_, tk256_,
            a.dqaccum, a.m_rows, a.delta_rows, a.pair_offset, a.pair_union,
            H, S, a.num_blocks, scale_log2, a.sm_scale);
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
                                             float* delta_rows, float* dqaccum,
                                             int num_heads, int seqlen, cudaStream_t stream) {
  cudaLaunchAttribute at[1];
  cudaLaunchConfig_t cfg = pdl_launch_config(
      dim3((unsigned)(seqlen / Q_TILE_ROWS), (unsigned)num_heads, 1), dim3(256, 1, 1), 0, stream,
      at);
  return cudaLaunchKernelEx(&cfg, vsa_bwd_preprocess_kernel, o, dout, delta_rows, dqaccum,
                            num_heads, seqlen);
}

inline cudaError_t launch_vsa_bwd_postprocess(const float* dqaccum, __nv_bfloat16* dq,
                                              int num_heads, int seqlen, float sm_scale,
                                              bool two_cta, cudaStream_t stream) {
  static bool post_smem_set = false;
  if (!post_smem_set) {
    cudaError_t e = cudaFuncSetAttribute(vsa_bwd_postprocess_kernel,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         Q_TILE_ROWS * HEAD_DIM * 4);
    if (e != cudaSuccess) return e;
    post_smem_set = true;
  }
  cudaLaunchAttribute at[1];
  cudaLaunchConfig_t cfg = pdl_launch_config(
      dim3((unsigned)(seqlen / Q_TILE_ROWS), (unsigned)num_heads, 1), dim3(128, 1, 1),
      Q_TILE_ROWS * HEAD_DIM * 4, stream, at);
  return cudaLaunchKernelEx(&cfg, vsa_bwd_postprocess_kernel, dqaccum, dq, num_heads, seqlen,
                            sm_scale, two_cta ? 1 : 0);
}

}  // namespace vsa_bwd_blk256

// ---------------------------------------------------------------------------
// Bench harness (CPU reference + verify + timing).
// ---------------------------------------------------------------------------

// Harness-side sparse block size (BLOCK env); the GPU path requires 256.
static int BLOCK = 256;

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
      inv.q_blocks[cursor[gkb]++] = mtile;
    }
  }
  return inv;
}

// blk256-native lists: one work item per kv256 block; entries = PLAIN q256
// ids (the kernel expands each entry into two 128-row q tiles in-kernel).
struct PairUnion {
  std::vector<int> offset;          // [B*H*nb256 + 1]
  std::vector<unsigned> entries;    // q256 ids
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

// Deterministic fill in [-1, 1) (same hash as the fv bench).
static void fillr(__nv_bfloat16* h, long n, unsigned seed) {
  for (long i = 0; i < n; ++i) {
    uint32_t x = (uint32_t)i * 2654435761u + seed * 40503u + 0x9e3779b9u;
    x ^= x >> 15; x *= 2246822519u; x ^= x >> 13; x *= 3266489917u; x ^= x >> 16;
    h[i] = __float2bfloat16((float)(x % 2039u) / 1019.5f - 1.0f);
  }
}

// N(0,1) Gaussian fill via Box-Muller (VSA_GAUSS; same hash as the fv bench).
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
  // block ids per (b,h,mtile) via partial Fisher-Yates (same knobs as the fv bench).
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
    // steps/item counts the in-kernel q256 -> 2 x q128 expansion.
    printf("  q-lists: items=%d entries=%ld steps/item=%.1f\n",
           B * H * num_blocks, usum, 2.0 * usum / (B * H * num_blocks));
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

  // GPU backward (FA4-aligned): stream-chained preprocess (Delta + dQaccum
  // zeroing) -> non-persistent main kernel -> postprocess (bf16 dQ).
  if (BLOCK != 256 || B != 1) {
    printf("  gpu: skipped (requires BLOCK=256 and B=1)\n");
    return;
  }
  {
    using namespace vsa_bwd_blk256;
    const long elems = tq * H * hd;
    __nv_bfloat16 *dQg, *dKg, *dVg, *dDOg, *dOg, *dDKout, *dDVout, *dDQout;
    float *dMg, *dDeltag, *dDQA;
    int *dOff, *dBlk;
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
    args.dqaccum = dDQA; args.dk = dDKout; args.dv = dDVout; args.dq = dDQout;
    args.m_rows = dMg; args.delta_rows = dDeltag;
    args.pair_offset = dOff; args.pair_union = reinterpret_cast<const unsigned*>(dBlk);
    args.num_heads = H; args.seqlen = S; args.num_blocks = num_blocks;
    args.two_cta = getenv("VSA_BWD_2CTA") != nullptr;
    args.sm_scale = 1.0f / sqrtf((float)hd);

    auto run_once = [&]() {
      CUDA_CHECK(launch_vsa_bwd_preprocess(dOg, dDOg, dDeltag, dDQA, H, S, 0));
      CUDA_CHECK(launch_vsa_bwd_sm100a(args, 0));
      CUDA_CHECK(launch_vsa_bwd_postprocess(dDQA, dDQout, H, S, args.sm_scale,
                                             args.two_cta, 0));
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
        printf("  dq argmax: t=%ld (q256 %ld, row256 %ld) h=%d d=%d ref=%.6f got=%.6f\n",
               t_, t_ / 256, t_ % 256, h_, d_, hdQ[am], gdq[am]);
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
      // production bf16 quantization points is ~1.9-2.7e-3 (oracle_bwd.py,
      // 2026-08-25), GPU-vs-CPU can legitimately reach ~2x that, and the
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
      printf("  gpu bwd: %.4f ms  %.1f TFLOPS (bwd 2.5x sel; pre+main+post)\n",
             ms, tflops);
    }

#ifdef WARP_PROF
    {
      WpBuffer wp = wp_alloc(dim3((unsigned)(2 * num_blocks), (unsigned)H, 1));
      run_once();
      CUDA_CHECK(cudaDeviceSynchronize());
      wp_readback(wp);
      const char* roles[16] = {"red","red","red","red",
                               "cmp","cmp","cmp","cmp","cmp","cmp","cmp","cmp",
                               "mma","load","rly","emp"};  // warp 14 = rly only in 2cta
      printf("  WARP_PROF block %u:\n", wp.view_block);
      wp_print_busy(wp, roles, 16, wp.view_block);
      wp_dump_raw(wp, "warp_raw_vsa_bwd_blk256.bin.gz", wp.view_block, 2);
      wp_free(wp);
    }
#endif

    cudaFree(dQg); cudaFree(dKg); cudaFree(dVg); cudaFree(dDOg); cudaFree(dOg);
    cudaFree(dDKout); cudaFree(dDVout); cudaFree(dDQout); cudaFree(dMg); cudaFree(dDeltag);
    cudaFree(dDQA); cudaFree(dOff); cudaFree(dBlk);
  }
}

int main() {
  if (const char* b = getenv("BLOCK")) BLOCK = atoi(b);
  CUDA_CHECK(cudaFree(0));
  printf("VSA block-sparse BACKWARD bench bf16 (blk256-native, FA4-aligned) sm_100a\n"
         "non-persistent 16-warp kernel, TWO_CTA=%d; pre+main+post (block=%d)\n"
         "=====================================\n",
         getenv("VSA_BWD_2CTA") ? 1 : 0, BLOCK);

  // Explicit SHAPE presets retain their existing IDs. The no-argument smoke
  // below is smaller so the automatic CPU reference always runs at B=1.
  // shapes: {B, H, num_blocks, topk, hd, label}.
  Sh shapes[] = {
    {1,  4,  8,  4, 128, "small"},
    {1, 16, 32,  8, 128, "fastvideo"},
    {1,  8, 64, 16, 128, "25pct"},
  };

  if (const char* s = getenv("SHAPE")) {
    char* end = nullptr;
    const long shape_index = strtol(s, &end, 10);
    constexpr int num_shapes = sizeof(shapes) / sizeof(shapes[0]);
    if (end == s || *end || shape_index < 0 || shape_index >= num_shapes) {
      fprintf(stderr, "SHAPE=%s must be an integer in [0, %d)\n", s, num_shapes);
      return 1;
    }
    Sh sh = shapes[shape_index];
    if (getenv("BATCH")) sh.B          = atoi(getenv("BATCH"));
    if (getenv("HEADS")) sh.H          = atoi(getenv("HEADS"));
    if (getenv("NB"))    sh.num_blocks = atoi(getenv("NB"));
    if (getenv("TOPK"))  sh.topk       = atoi(getenv("TOPK"));
    sh.lab = "custom";
    run(sh);
    return 0;
  }
  const int B = getenv("BATCH") ? atoi(getenv("BATCH")) : 1;
  // 8 * 4 * 1 * 256^2 = 2,097,152 selected pairs, below the 4e6 CPU-ref limit.
  // Keep the 25% density and check all dQ/dK/dV outputs before benchmarking.
  run({B, 8, 4, 1, 128, "smoke-1k"});
  return 0;
}
