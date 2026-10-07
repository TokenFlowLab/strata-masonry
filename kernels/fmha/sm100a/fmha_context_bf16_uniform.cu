// fmha_context_bf16_uniform.cu -- K2 FMHA context BF16, sm_100a.
//
// ASSUMES (baked in -- the kernel is NOT correct otherwise):
//   1. Full mask -- every q-token attends to all keys (only keys >= seqlen are masked). IS_CAUSAL
//      adds the triangular mask + K-loop cap; still uniform (non-ragged) seqlen.
//   2. ALL seqlens EQUAL (no varlen) -- one `seqlen` for every sample, so K_TILES is uniform.
//
// GEN-shape variant. The warp-specialized 16-warp kernel body + barrier contract are the SAME as
// fmha_context_bf16_gqa_nonpersistent.cu (see that file's header for Terminology, Data Layout,
// Execution flow, Barrier Contract, Memory Layout). Differences from those varlen kernels:
//   - PERSISTENT scheduling, selected by USE_CLC (phase trackers persist across tiles, primed once):
//       USE_CLC=true (default): CLC (clusterlaunchcontrol.try_cancel) HW work-stealing scheduler.
//         w15 is a standalone sched warp issuing try_cancel into a CLC_STAGES-deep tile-id ring;
//         the other 15 warps + the sched consume it (all 16 release clc_empty/tile -> arrive_count
//         = N_WARPS). Ring depth covers the load-ahead/epi-behind skew. grid = full problem.
//       USE_CLC=false: static grid-stride loop (tile_id += gridDim.x); w15 idle; grid = #SMs.
//   - Equal seqlen: uniform K_TILES + a plain (sample, q_tile_id, kv_head) decode -- no prefix sum,
//     no binary search.
//   - TMA-store epilogue: correction packs O*=1/l into a subtile-split sO, then the epi warp (w13)
//     TMA-stores it (full_bar_o_epi). Valid because equal-seqlen tiles are non-ragged;
//     the varlen kernels use a predicated STG re-tile instead.
//
// Barrier contract (additions to gqa_nonpersistent's -- unique to the TMA-sO epilogue):
//   - empty_bar_o_epi[m] (count 1): epi -> corr, "sO[m]'s TMA store drained, slot reusable".
//     epi arrives per m as its commit group drains; corr waits before packing the next tile's
//     sO[m]. Without it, tiny causal K-loops (K_TILES=1) let corr's repack race the
//     still-reading store.
//
// Budgets:
//   - Registers: per-warp budgets sum to exactly the SM file (65536 = 128*512): softmax inc<192>
//     (x8) + correction dec<80> (x4) + the four single warps dec<48> (x4). WARP_PROF's
//     wp_begin/wp_end markers fit as-is (single warps ~R26; corr ~R77; softmax ~R186).
//   - TMEM: each M-tile needs S + O = K_TILE + HEAD_DIM cols of the 512, so
//     M_TILES_PER_CTA <= 512 / (K_TILE + HEAD_DIM) = 2 for 128/128 (adjacent q-tiles of the
//     SAME sequence).
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
#include "../../../composites/118_mbarrier_phase_tracking.cuh"
#include "../../../composites/106_clc_fetch_next_tile.cuh"
#include "../../../composites/110_fmha_workitem_decode.cuh"
#include "../../../blocks/88_load_warp_blackwell.cuh"
#include "../../../blocks/93_epi_warp_blackwell.cuh"
#include "../../../blocks/97_sched_warp_clc.cuh"
#include "../../../blocks/98_softmax_warp.cuh"
#include "../../../blocks/99_correction_warp.cuh"
#include "../../../blocks/111_fmha_mma_warp_blackwell.cuh"
#include "../../../primitives/46_setmaxnreg.cuh"
#include "../../../primitives/76_packed_f32x2.cuh"
#include "../../../primitives/77_ex2_approx.cuh"
#include "../../../primitives/78_rcp_approx.cuh"
#include "../../../primitives/_warp_prof_noop.cuh"
#include "../../../composites/109_fastdivmod.cuh"
#include "../../../composites/112_fmha_softmax_utils.cuh"
#include "fmha_cpu_ref.cuh"
#include "fmha_context_bf16_benchmark.cuh"

constexpr int M_TILE = 128;
constexpr int M_TILES_PER_CTA = 2;
constexpr int K_TILE = 128;
constexpr int HEAD_DIM = 128;
// B128 swizzle atom = 128 bytes = 64 bf16: all SMEM tiles are laid out in
// 64-wide sub-tiles along the contiguous dim.
constexpr int SUB_COLS_BF16 = 64;
constexpr int SUB_COLS_BYTES = SUB_COLS_BF16 * (int)sizeof(__nv_bfloat16);   // 128 B (one swizzle atom)
constexpr int Q_SUBTILES = HEAD_DIM / SUB_COLS_BF16;  // 2
constexpr int K_SUBTILES = HEAD_DIM / SUB_COLS_BF16;  // 2 (K tile is K_TILE tokens x head_dim)
constexpr int Q_SUB_COLS_BYTES = M_TILE * SUB_COLS_BYTES;         // 16 KB
constexpr int K_SUB_COLS_BYTES = K_TILE * SUB_COLS_BYTES;         // 16 KB
constexpr int Q_TILE_BYTES = Q_SUBTILES * Q_SUB_COLS_BYTES;     // 32 KB
constexpr int K_TILE_BYTES = K_SUBTILES * K_SUB_COLS_BYTES;     // 32 KB
constexpr int NUM_KV_STAGES = 3;
constexpr int TMEM_TOTAL = 512;                     // S0,S1(128*2)+O0,O1(128*2)=512
constexpr int W_CORR0 = 8, W_MMA = 12, W_EPI = 13, W_LOAD = 14, W_SCHED = 15;
constexpr int N_WARPS = 16;
constexpr int CLC_STAGES = 2;   // matches blocks/97 ring depth; measured >= 4-deep (causal +3%)

extern __shared__ __align__(1024) uint8_t fmha_smem[];

// Compile-time kernel config (template args, set in run()'s `constexpr` block):
//   S_LD_COLS        : cols per softmax tcgen05.ld of the S row (32/64 compile; 128 aborts ptxas).
//   FULL_NAMED_BAR   : softmax->corr "scale ready": true = HW named barrier (per-band), false =
//                      mbarrier (full_bar_alpha/full_bar_l). Both use alpha_and_l_smem.
//   EX2_EMU          : route a fraction of softmax exp2 through FFMA f32x2 emulation (vs MUFU.EX2).
//   SPLIT_P          : softmax publishes P in two chunks (96+32 keys); BMM2 starts on the first,
//                      full_bar_p_last gates the tail atoms.
//   SOFTMAX_THROTTLE : FA4 pacing -- corr defers releasing the alpha/l slot until after it consumes,
//                      holding softmax ~1 stage behind correction.
//   USE_CLC          : true = CLC work-stealing sched (w15 sched warp, grid=full problem; wins long S);
//                      false = static grid-stride loop (w15 idle, grid=#SMs; wins short). Same pipe.
//   Q_RASTER         : true = q-tile-innermost raster (adjacent work-items share K/V -> hot L2);
//                      false = kv-head-innermost (the original order).
//   MHA              : true = HQ==HK (gqa_group folds to 1; M-tile = 128 tok x 1 head); false = GQA
//                      (runtime HQ/HK). Body identical; picked by run<MHA>() from main()'s env knob.
//   IS_CAUSAL        : true = triangular causal mask + K-loop cap (uniform seqlen; still non-ragged).
//   LPT              : heaviest-q-tile-first ordering to balance causal load; GATED to IS_CAUSAL.
template <int S_LD_COLS = 32, bool FULL_NAMED_BAR = false, bool EX2_EMU = false, bool SPLIT_P = true,
          bool SOFTMAX_THROTTLE = false, bool USE_CLC = true, bool Q_RASTER = true, bool MHA = false,
          bool IS_CAUSAL = false, bool LPT = false>
__global__ void __cluster_dims__(1, 1, 1) __launch_bounds__(N_WARPS * 32, 1)
fmha_context_bf16_gen_kernel(const __grid_constant__ CUtensorMap tmap_q,
    const __grid_constant__ CUtensorMap tmap_k, const __grid_constant__ CUtensorMap tmap_v_t,
    const __grid_constant__ CUtensorMap tmap_o, int seqlen_kv, int v_sample_stride,
    int num_q_heads, int num_kv_heads, float scale_log2,
    int packed_mtiles_per_seq, int num_samples,
    unsigned long long magic0, unsigned long long magic1, unsigned long long magic2) {
  uint8_t* sQ0 = fmha_smem;
  uint8_t* sQ1 = sQ0 + Q_TILE_BYTES;
  uint8_t* sKV = sQ1 + Q_TILE_BYTES;
  __nv_bfloat16* sO0 = reinterpret_cast<__nv_bfloat16*>(sKV + NUM_KV_STAGES * K_TILE_BYTES);
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
  uint64_t* throttle_full  = clc_empty + CLC_STAGES;
  uint64_t* throttle_empty = throttle_full + 2;
  uint32_t* clc_response = reinterpret_cast<uint32_t*>(
      (reinterpret_cast<uintptr_t>(throttle_empty + 2) + 15u) & ~uintptr_t(15u));
  uint32_t* tmem_slot = clc_response + CLC_STAGES * 4;
  float* alpha_and_l_smem = reinterpret_cast<float*>(tmem_slot + 2);

  const int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;
  WpCtx wpc = wp_ctx_init();

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
      for (int s = 0; s < 2; ++s) {
        mbarrier_init(smem_ptr_u32(&throttle_full[s]), 32);
        mbarrier_init(smem_ptr_u32(&throttle_empty[s]), 32);
      }
      #pragma unroll
      for (int i = 0; i < CLC_STAGES * 4; ++i)
        clc_response[i] = 0;
    }
  }
  fence_mbarrier_init_release_cluster();
  __syncthreads();

  if (warp_id == W_LOAD) {
    load_warp_blackwell_ntiles_1sm2sm_bf16_fmha<
        NUM_KV_STAGES, M_TILES_PER_CTA, M_TILE, K_TILE, HEAD_DIM,
        USE_CLC, CLC_STAGES, Q_RASTER, IS_CAUSAL, LPT, /*REG_BUDGET=*/48>(wpc, &tmap_q, &tmap_k, &tmap_v_t, sQ0, sKV,
        full_bar, empty_bar, full_bar_q, empty_bar_q,
        clc_full, clc_empty, clc_response, throttle_full, throttle_empty,
        seqlen_kv, num_q_heads, num_kv_heads, packed_mtiles_per_seq, num_samples,
        magic0, magic1, magic2, /*seqlen_q=*/-1, v_sample_stride);
  }
  else if (warp_id == W_MMA) {
    fmha_mma_warp_blackwell_ntiles_1sm_bf16<
        NUM_KV_STAGES, M_TILES_PER_CTA, M_TILE, K_TILE, HEAD_DIM,
        SPLIT_P, USE_CLC, CLC_STAGES, Q_RASTER, IS_CAUSAL, LPT,
        TMEM_TOTAL, /*REG_BUDGET=*/48>(wpc, tmem_slot, sQ0, sKV,
        full_bar, empty_bar, full_bar_q, empty_bar_q,
        full_bar_spo, empty_bar_spo, full_bar_p_last, full_bar_o_acc,
        clc_full, clc_empty, clc_response,
        seqlen_kv, num_q_heads, num_kv_heads, packed_mtiles_per_seq, num_samples, magic0, magic1, magic2);
  }
  else if (warp_id == W_EPI) {
    epi_store_warp_blackwell_ntiles_1sm2sm_bf16_fmha<
        M_TILES_PER_CTA, M_TILE, HEAD_DIM, K_TILE,
        USE_CLC, CLC_STAGES, Q_RASTER, IS_CAUSAL, LPT, /*REG_BUDGET=*/48>(wpc, &tmap_o, sO_bufs, full_bar_o_epi, empty_bar_o_epi,
        clc_full, clc_empty, clc_response,
        seqlen_kv, num_q_heads, num_kv_heads, packed_mtiles_per_seq, num_samples, magic0, magic1, magic2);
  }
  else if (warp_id == W_SCHED) {
    setmaxnreg_dec<48>();

    if constexpr (USE_CLC) {
      sched_warp_clc_blackwell_ntiles_2sm_bf16<
          /*USE_GRIDDEP_WAIT=*/false, /*CLUSTER_SHAPE_M=*/1, /*CLUSTER_SHAPE_N=*/1,
          ClcRasterOrder::AlongN, /*CTA_GROUP=*/1, /*SUSPEND=*/true>(wpc, clc_full, clc_empty, clc_response, throttle_full, throttle_empty,
          /*cluster_rank=*/0, lane);
    }
  }
  else if (warp_id >= W_CORR0 && warp_id < W_MMA) {
    correction_warp_blackwell_ntiles_1sm2sm_bf16_fmha<
        M_TILE, M_TILES_PER_CTA, HEAD_DIM, K_TILE, SUB_COLS_BF16,
        FULL_NAMED_BAR, SOFTMAX_THROTTLE, USE_CLC, CLC_STAGES, Q_RASTER, IS_CAUSAL, LPT, /*REG_BUDGET=*/80>(wpc, tmem_slot, warp_id - W_CORR0, lane, alpha_and_l_smem, sO_bufs,
        full_bar_alpha, full_bar_l, full_bar_o_acc, full_bar_o_epi,
        empty_bar_spo, empty_bar_alpha_and_l, empty_bar_o_epi,
        clc_full, clc_empty, clc_response,
        seqlen_kv, num_q_heads, num_kv_heads, packed_mtiles_per_seq, num_samples, magic0, magic1, magic2);
  }
  else {
    softmax_warp_blackwell_ntiles_1sm2sm_bf16_fmha<
        K_TILE, M_TILE, M_TILES_PER_CTA, S_LD_COLS, FULL_NAMED_BAR, SPLIT_P, EX2_EMU,
        USE_CLC, CLC_STAGES, Q_RASTER, IS_CAUSAL, LPT, /*REG_BUDGET=*/192>(wpc, tmem_slot, warp_id, lane, scale_log2, alpha_and_l_smem,
        full_bar_spo, full_bar_alpha, full_bar_l, empty_bar_spo, empty_bar_alpha_and_l, full_bar_p_last,
        clc_full, clc_empty, clc_response,
        seqlen_kv, num_q_heads, num_kv_heads, packed_mtiles_per_seq, num_samples, magic0, magic1, magic2);
  }
  wp_flush(wpc);
}

// ============================== driver ====================================

// Deterministic fill in [-1, 1) so host and device see identical inputs.
// Input fill modes -- MUST match kernels/gemm/sm100a/dense_gemm_bf16.cu and
// fa4_extract/fa4_bench_matrix.py so ours vs FA4 see BITWISE-IDENTICAL inputs (input values
// swing tensor-core power -> clock -> timing by up to ~40%). FILL env (default 2):
//   1=[-0.5,0.5)/256  2=[-1,1)/2048  3=const 1.0  4=int{-3..3}. Seeds 11/22/33 for Q/K/V.
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

// One benchmark shape. sl[s] = seqlen of sample s (all equal here); nqh/nkh =
// q/kv head counts; hd = head dim; causal toggles the mask; lab is for printing.
struct Sh {
  std::vector<int> sl;    // Q seqlen per sample (all equal); sl[0] used
  std::vector<int> slk;   // K/V seqlen (empty = self-attn, = sl); slk[0] used
  int  nqh, nkh, hd;
  bool causal;
  const char* lab;
};

template <bool MHA = false, bool IS_CAUSAL = false, bool LPT = false>
static double run(const Sh& sh, bool verify) {
  const int  ns        = (int)sh.sl.size();
  const int  seqlen_q  = sh.sl[0];
  const int  seqlen_kv = sh.slk.empty() ? seqlen_q : sh.slk[0];   // cross-attn: != seqlen_q
  const long tq        = (long)ns * seqlen_q;    // total q-tokens
  const long tk        = (long)ns * seqlen_kv;   // total k-tokens
  // PTX 5.5.3: the TMA bounding-box address must be 16-byte aligned. V_T's token
  // axis is contiguous BF16, so each sample needs an 8-element-aligned start.
  // This is storage padding only: K loads, causal masks and FLOPs use seqlen_kv.
  const int v_sample_stride = (seqlen_kv + 7) / 8 * 8;
  const long tv = (long)ns * v_sample_stride;

  // ---- device buffers (bf16; V stored transposed as V_T for the BMM2 TMA) ----
  __nv_bfloat16 *dQ, *dK, *dVT, *dO;
  CUDA_CHECK(cudaMalloc(&dQ,  tq * sh.nqh * sh.hd * 2));
  CUDA_CHECK(cudaMalloc(&dK,  tk * sh.nkh * sh.hd * 2));
  CUDA_CHECK(cudaMalloc(&dVT, (long)sh.nkh * sh.hd * tv * 2));
  CUDA_CHECK(cudaMalloc(&dO,  tq * sh.nqh * sh.hd * 2));

  // ---- host inputs + V -> V_T transpose ([tok,head,hd] -> [head,hd,tok]) ----
  std::vector<__nv_bfloat16> hQ(tq * sh.nqh * sh.hd),
                             hK(tk * sh.nkh * sh.hd),
                             hV(tk * sh.nkh * sh.hd);
  fillr(hQ.data(), hQ.size(), 11);
  fillr(hK.data(), hK.size(), 22);
  fillr(hV.data(), hV.size(), 33);

  std::vector<__nv_bfloat16> hVT((long)sh.nkh * sh.hd * tv, __float2bfloat16(0.f));
  for (long idx = 0; idx < tk; ++idx)
    for (int h = 0; h < sh.nkh; ++h)
      for (int d = 0; d < sh.hd; ++d)
        hVT[(h * sh.hd + d) * tv + (idx / seqlen_kv) * v_sample_stride
            + idx % seqlen_kv] = hV[(idx * sh.nkh + h) * sh.hd + d];

  CUDA_CHECK(cudaMemcpy(dQ,  hQ.data(),  hQ.size()  * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dK,  hK.data(),  hK.size()  * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dVT, hVT.data(), hVT.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dO, 0, hQ.size() * 2));

  // ---- TMA tensor maps ----
  // pack-GQA Q/O: 4D TMA over [hd, nqh, token-IN-SAMPLE, sample]; box [hd-subtile x
  // gqa_group_size x q_tile_per_mtile x 1] -> 128 packed rows (qh-inner). The per-sample token
  // dim makes the HW clamp the box at each sample's seqlen_q boundary: when seqlen_q is not a
  // multiple of q_tile_per_cta, the last packed M-tile's overrun rows would otherwise land in
  // the NEXT sample's tokens (a global-token 3D map stores them -- cross-sample clobber, racy).
  // With the 4D map the overrun rows read as zero-fill (Q) and are simply not written (O).
  const int gqa = sh.nqh / sh.nkh;            // q-heads per kv-head
  const int tpi = M_TILE / gqa;               // q-tokens per M-tile
  const int tpc = 2 * tpi;                    // q-tokens per CTA (2 M-tiles)
  CUtensorMap tq_, tk_, tvt_, to_;
  {
    uint64_t gd[4] = { (uint64_t)sh.hd, (uint64_t)sh.nqh, (uint64_t)seqlen_q, (uint64_t)ns };
    uint64_t gs[3] = { (uint64_t)sh.hd * 2u, (uint64_t)sh.nqh * sh.hd * 2u,
                       (uint64_t)seqlen_q * sh.nqh * sh.hd * 2u };
    uint32_t bd[4] = { (uint32_t)SUB_COLS_BF16, (uint32_t)gqa, (uint32_t)tpi, 1u };
    uint32_t es[4] = { 1u, 1u, 1u, 1u };
    CUresult r = cuTensorMapEncodeTiled(&tq_, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, dQ, gd, gs, bd, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUDA_CHECK(r == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue);
    r = cuTensorMapEncodeTiled(&to_, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, dO, gd, gs, bd, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUDA_CHECK(r == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue);
  }
  // K: ONE 3D TMA copy folds the 2 head-dim swizzle atoms (HEAD_DIM = 2 x SUB_COLS_BF16) into the box
  // (vs looping 2 x 2D copies). dims [atom-col SUB_COLS_BF16, token tk, atom (nkh*hd)/SUB_COLS_BF16]; box
  // [SUB_COLS_BF16, K_TILE, K_SUBTILES]; strides token=(nkh*hd)*2B, atom=SUB_COLS_BF16*2B. The box dim order
  // (atom outermost) reproduces the atom-outer smem layout the MMA reads (atom0 then atom1).
  {
    uint64_t gd[3] = { (uint64_t)SUB_COLS_BF16, (uint64_t)tk, (uint64_t)(sh.nkh * sh.hd / SUB_COLS_BF16) };
    uint64_t gs[2] = { (uint64_t)(sh.nkh * sh.hd) * 2u, (uint64_t)SUB_COLS_BF16 * 2u };
    uint32_t bd[3] = { (uint32_t)SUB_COLS_BF16, (uint32_t)K_TILE, (uint32_t)K_SUBTILES };
    uint32_t es[3] = { 1u, 1u, 1u };
    CUresult r = cuTensorMapEncodeTiled(&tk_, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, dK, gd, gs, bd, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUDA_CHECK(r == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue);
  }
  CUDA_CHECK(make_tma_2d_tiled(&tvt_, dVT, (long)sh.nkh * sh.hd, tv, sh.hd, SUB_COLS_BF16, 2,
                               CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B));

  // ---- shared memory budget ----
  const int packed_mtiles_per_seq = (seqlen_q + tpc - 1) / tpc;   // packed-M tiles per (sample, kv-head)
  // FastDivmod magics for decode_workitem's divides
  const unsigned long long packed_mtiles_per_sample_magic = make_magic((unsigned)(packed_mtiles_per_seq * sh.nkh));
  const unsigned long long num_kv_heads_magic = make_magic((unsigned)sh.nkh);
  const unsigned long long packed_mtiles_per_seq_magic = make_magic((unsigned)packed_mtiles_per_seq);
  const size_t smem =
        (size_t)2 * Q_TILE_BYTES + NUM_KV_STAGES * K_TILE_BYTES  // Q (x2) + shared K/V ring
      + (size_t)2 * M_TILE * HEAD_DIM * sizeof(__nv_bfloat16)     // 2 sO bufs for TMA-O
      + (2 * NUM_KV_STAGES + 22) * 8                              // mbarriers (incl full/empty_bar_o_epi)
      + (size_t)CLC_STAGES * (2 * 8 + 16) + 4 * 8 + 16    // CLC: clc_full+clc_empty + throttle[4] + response (16B aligned)
      + 8                                                         // tmem_slot
      + (size_t)2 * M_TILE * sizeof(float)                        // alpha_and_l_smem [2][M_TILE]
      + 256;                                                      // slack / alignment

  // Compile-time kernel config (see the knob docs near the top of this file).
  constexpr bool FULL_NAMED_BAR = false, EX2_EMU = true, SPLIT_P = true,
                 SOFTMAX_THROTTLE = false, USE_CLC = true, Q_RASTER = true;
  auto kfn = &fmha_context_bf16_gen_kernel<64, FULL_NAMED_BAR, EX2_EMU, SPLIT_P, SOFTMAX_THROTTLE, USE_CLC, Q_RASTER, MHA, IS_CAUSAL, LPT>;
  CUDA_CHECK(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));

  // ---- launch geometry: CLC persistent. Launch the FULL problem grid (one CTA per work tile);
  // clusterlaunchcontrol keeps only ~#SMs CTAs resident and hands the rest of the CTA-ids out via
  // try_cancel (HW work-stealing scheduler), so the grid-size is the tile count, not #SMs. ----
  const float scale_log2 = (1.0f / sqrtf((float)sh.hd)) * (float)M_LOG2E;
  int numSM = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, 0));
  const int total_work = ns * packed_mtiles_per_seq * sh.nkh;   // one CTA per (sample, packed-M tile, kv-head)
  const int nblk = USE_CLC ? total_work : std::min(total_work, numSM);
  (void)numSM;
  dim3 grid(nblk, 1, 1), block(N_WARPS * 32, 1, 1);

  // CLC must be launched via cudaLaunchKernelEx with a cluster-dimension attribute -- a plain
  // <<<grid,block>>> launch does NOT enable clusterlaunchcontrol (try_cancel silently misbehaves
  // and tiles get skipped). cluster {1,1,1} (matches __cluster_dims__(1,1,1)); no PSS attribute
  // (we don't drive griddepcontrol, so leave the dependent-launch serialization off).
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = 0;
  cudaLaunchAttribute cfgAttr[1];
  cfgAttr[0].id = cudaLaunchAttributeClusterDimension;
  cfgAttr[0].val.clusterDim.x = 1;
  cfgAttr[0].val.clusterDim.y = 1;
  cfgAttr[0].val.clusterDim.z = 1;
  cfg.attrs = cfgAttr;
  cfg.numAttrs = 1;
  auto launch = [&](cudaStream_t st = 0) {
    cfg.stream = st;
    if (USE_CLC)
      return cudaLaunchKernelEx(&cfg, kfn, tq_, tk_, tvt_, to_, seqlen_kv, v_sample_stride, sh.nqh, sh.nkh,
                                scale_log2, packed_mtiles_per_seq, ns, packed_mtiles_per_sample_magic, packed_mtiles_per_seq_magic, num_kv_heads_magic);
    kfn<<<grid, block, smem, st>>>(tq_, tk_, tvt_, to_, seqlen_kv, v_sample_stride, sh.nqh, sh.nkh,
                               scale_log2, packed_mtiles_per_seq, ns, packed_mtiles_per_sample_magic, packed_mtiles_per_seq_magic, num_kv_heads_magic);
    return cudaGetLastError();
  };

  const double ms = fmha_context_bf16_benchmark::measure(launch);
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
    wp_dump_raw(wp, "warp_raw_fmha_pgen.bin.gz", pblk, NUM_KV_STAGES);
    wp_free(wp);
  }
#endif

  const uint64_t pairs = fmha_context_bf16_benchmark::attended_pairs(
      seqlen_q, seqlen_kv, sh.causal);
  const uint64_t eff = (uint64_t)ns * pairs;
  const double tflops = fmha_context_bf16_benchmark::report(
      sh.lab, tq, sh.causal, sh.hd, sh.nqh, eff, ms);

  // DUMP_IO: write ours' exact Q/K/V (bf16, [B,S,H,D]) + O so FA4 can load the SAME bytes
  // and we can diff outputs. Q/K/V here are ours' host fill (identical to FA4's hashfill).
  if (getenv("DUMP_IO")) {
    std::vector<__nv_bfloat16> hodump(hQ.size());
    CUDA_CHECK(cudaMemcpy(hodump.data(), dO, hodump.size() * 2, cudaMemcpyDeviceToHost));
    auto wr = [](const char* fn, const void* p, size_t bytes) {
      FILE* f = fopen(fn, "wb"); fwrite(p, 1, bytes, f); fclose(f); };
    wr("/tmp/io_q.bin", hQ.data(), hQ.size() * 2);
    wr("/tmp/io_k.bin", hK.data(), hK.size() * 2);
    wr("/tmp/io_v.bin", hV.data(), hV.size() * 2);
    wr("/tmp/io_o.bin", hodump.data(), hodump.size() * 2);
    FILE* f = fopen("/tmp/io_shape.txt", "w");
    fprintf(f, "%d %d %d %d %d %d %d\n", ns, (int)seqlen_q, (int)seqlen_kv,
            sh.nqh, sh.nkh, sh.hd, (int)sh.causal);
    fclose(f);
    fprintf(stderr, "DUMP_IO: B=%d Sq=%d Sk=%d HQ=%d HK=%d D=%d causal=%d -> /tmp/io_{q,k,v,o}.bin\n",
            ns, (int)seqlen_q, (int)seqlen_kv, sh.nqh, sh.nkh, sh.hd, (int)sh.causal);
  }

  // ---- correctness check against the fp32 CPU reference ----
  if (verify) {
    std::vector<__nv_bfloat16> ho(hQ.size());
    CUDA_CHECK(cudaMemcpy(ho.data(), dO, ho.size() * 2, cudaMemcpyDeviceToHost));

    std::vector<int> cq(ns + 1, 0), ck(ns + 1, 0);
    for (int i = 0; i < ns; ++i) { cq[i + 1] = cq[i] + seqlen_q; ck[i + 1] = ck[i] + seqlen_kv; }

    std::vector<float> ref(hQ.size(), 0.f), out(hQ.size());
    char key[128];
    const int fill_mode = getenv("FILL") ? atoi(getenv("FILL")) : 2;
    snprintf(key, sizeof key, "B%d_Sq%d_Sk%d_hq%d_hk%d_hd%d_c%d_f%d",
             ns, seqlen_q, seqlen_kv, sh.nqh, sh.nkh, sh.hd, (int)sh.causal, fill_mode);
    cached_ref_f32(key, ref.data(), ref.size(), [&] {
      cpu_fmha_ref(hQ.data(), hK.data(), hV.data(), ref.data(),
                   cq, ck, sh.nqh, sh.nkh, sh.hd, sh.causal);
    });
    for (size_t i = 0; i < out.size(); ++i) out[i] = __bfloat162float(ho[i]);

    bool finite = true;
    for (float value : out) finite = finite && std::isfinite(value);
    const bool ok = finite && check_close_f32(ref.data(), out.data(), (int)out.size(), 0.05f, 0.10f);
    printf("  verify [%s]: %s\n", sh.lab, ok ? "OK" : "FAIL");
    if (!ok) std::exit(EXIT_FAILURE);
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
    printf("  STRESS [%s] N=%d: mismatches=%d cuda_errs=%d\n", sh.lab, N, mism, errs);
  }

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

  Sh s{};
  const int B = getenv("BATCH")  ? atoi(getenv("BATCH"))  : 128;
  const int S = getenv("SEQLEN") ? atoi(getenv("SEQLEN")) : 240;
  // SEQLEN_KV != SEQLEN -> cross-attention (Q length != K/V length).
  const int Sk = getenv("SEQLEN_KV") ? atoi(getenv("SEQLEN_KV")) : S;
  const int HQ = getenv("HEADS") ? atoi(getenv("HEADS")) : 32;
  // MHA=1 -> HK==HQ (gqa_group=1); default GQA -> HK=4 (gqa_group=HQ/4=8 at HQ=32).
  const bool mha = getenv("MHA") ? (atoi(getenv("MHA")) != 0) : false;
  // CAUSAL=1 -> triangular mask (top-left: keys <= q_pos). Still uniform (non-ragged) per-sample.
  const bool causal = getenv("CAUSAL") ? (atoi(getenv("CAUSAL")) != 0) : false;
  s.sl     = std::vector<int>(B, S);
  if (Sk != S) s.slk = std::vector<int>(B, Sk);   // cross-attention
  s.nqh    = HQ;
  s.nkh    = mha ? HQ : 4;
  s.hd     = 128;
  s.causal = causal;
  s.lab    = causal ? (mha ? "mha-causal" : "causal") : (mha ? "mha" : "gen");
  // CPU fp32 reference is O(B*HQ*S^2*D) -- infeasible at long S; verify only for small S
  // (override with NOVERIFY=1).
  const bool verify = getenv("NOVERIFY") ? false : (S <= 1024 && Sk <= 1024);
  constexpr bool LPT = false;   // measured ~neutral on uniform-causal (CLC already balances)
  if      (mha && causal) run</*MHA=*/true,  /*IS_CAUSAL=*/true,  /*LPT=*/LPT  >(s, verify);
  else if (mha)           run</*MHA=*/true,  /*IS_CAUSAL=*/false, /*LPT=*/false>(s, verify);
  else if (causal)        run</*MHA=*/false, /*IS_CAUSAL=*/true,  /*LPT=*/LPT  >(s, verify);
  else                    run</*MHA=*/false, /*IS_CAUSAL=*/false, /*LPT=*/false>(s, verify);
  return 0;
}
