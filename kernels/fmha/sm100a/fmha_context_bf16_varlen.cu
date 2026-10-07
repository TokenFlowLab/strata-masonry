// fmha_context_bf16_varlen.cu -- K2 FMHA context BF16, sm_100a. Variable-seqlen (varlen), causal.
//
// Persistent 16-warp context kernel, block-composed: each warp dispatches into shared blocks/
// bodies. varlen packs samples densely by cu_seqlens, so three
// blocks are varlen-specific: decode_workitem_varlen (prefix-sum seqlens, ragged K_TILES, emits
// seqlen_q), load (3D flat-global Q + k_base[] start), and corr (predicated STG re-tile O store
// on the corr warps -- no TMA epi, so W_EPI idles). softmax + MMA reuse the shared ntiles wrappers
// with VARLEN=true; sched/idle reuse blocks 97/107. The MMA warp owns TMEM (alloc<1>/dealloc<1>).
//
// Scheduling: static uniform tiling; one work tile = (sample, q_tile_id, kv_head) = 32 q-tokens.
// Short samples early-return on the `q_tile_base < seqlen_q` guard. Persistent CLC work-stealing
// with a load->sched throttle; LPT reverses the q-tile order (heaviest causal tile first).
// M_TILES_PER_CTA = 2 (S+O = K_TILE+HEAD_DIM cols each -> 512 TMEM cols).

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
#include "../../../primitives/46_setmaxnreg.cuh"
#include "../../../primitives/76_packed_f32x2.cuh"
#include "../../../primitives/77_ex2_approx.cuh"
#include "../../../primitives/78_rcp_approx.cuh"
#include "../../../composites/109_fastdivmod.cuh"
#include "../../../composites/110_fmha_workitem_decode.cuh"
#include "../../../composites/112_fmha_softmax_utils.cuh"
#include "../../../blocks/88_load_warp_blackwell.cuh"
#include "../../../blocks/98_softmax_warp.cuh"
#include "../../../blocks/97_sched_warp_clc.cuh"
#include "../../../blocks/99_correction_warp.cuh"
#include "../../../blocks/107_idle_warp_blackwell.cuh"
#include "../../../blocks/111_fmha_mma_warp_blackwell.cuh"
#include "fmha_cpu_ref.cuh"
#include "fmha_context_bf16_benchmark.cuh"

constexpr int M_TILE = 128;
constexpr int M_TILES_PER_CTA = 2;
constexpr int K_TILE = 128;
constexpr int HEAD_DIM = 128;
constexpr int SUB_COLS_BF16 = 64;   // B128 swizzle atom (64 bf16)
constexpr int SUB_COLS_BYTES = SUB_COLS_BF16 * (int)sizeof(__nv_bfloat16);
constexpr int Q_SUBTILES = HEAD_DIM / SUB_COLS_BF16;
constexpr int K_SUBTILES = HEAD_DIM / SUB_COLS_BF16;
constexpr int Q_SUB_COLS_BYTES = M_TILE * SUB_COLS_BYTES;
constexpr int K_SUB_COLS_BYTES = K_TILE * SUB_COLS_BYTES;
constexpr int Q_TILE_BYTES = Q_SUBTILES * Q_SUB_COLS_BYTES;
constexpr int K_TILE_BYTES = K_SUBTILES * K_SUB_COLS_BYTES;
constexpr int NUM_KV_STAGES = 3;
constexpr int TMEM_TOTAL = 512;   // S0,S1 + O0,O1 = 512 cols
constexpr int W_CORR0 = 8;
constexpr int W_MMA = 12, W_EPI = 13, W_LOAD = 14, W_SCHED = 15;
constexpr int N_WARPS = 16;
constexpr int CLC_STAGES = 4;

extern __shared__ __align__(1024) uint8_t dyn_smem[];

template <bool IS_CAUSAL, int S_LD_COLS = 32, bool FULL_NAMED_BAR = false, bool EX2_EMU = false, bool SPLIT_P = true, bool LPT = false,
          bool USE_CLC = true, bool Q_RASTER = true, bool MHA = false>
__global__ void __cluster_dims__(1, 1, 1) __launch_bounds__(N_WARPS * 32, 1)
fmha_context_bf16_persistent_kernel(const __grid_constant__ CUtensorMap tmap_q,
    const __grid_constant__ CUtensorMap tmap_k, const __grid_constant__ CUtensorMap tmap_v_t,
    __nv_bfloat16* __restrict__ O_out, const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ k_base, const int* __restrict__ seqlens_kv,
    int num_q_heads, int num_kv_heads, float scale_log2,
    int packed_mtiles_per_seq, int num_samples) {
  WpCtx wpc = wp_ctx_init();  // WARP_PROF TraceContext (no-op without -DWARP_PROF)

  uint8_t* sQ0 = dyn_smem;
  uint8_t* sQ1 = sQ0 + Q_TILE_BYTES;
  uint8_t* sKV = sQ1 + Q_TILE_BYTES;
  __nv_bfloat16* sO = reinterpret_cast<__nv_bfloat16*>(sKV + NUM_KV_STAGES * K_TILE_BYTES);
  uint64_t* full_bar = reinterpret_cast<uint64_t*>(
      reinterpret_cast<uint8_t*>(sO) + M_TILE * HEAD_DIM * sizeof(__nv_bfloat16));
  uint64_t* empty_bar             = full_bar + NUM_KV_STAGES;
  uint64_t* full_bar_q            = empty_bar + NUM_KV_STAGES;
  uint64_t* empty_bar_q           = full_bar_q + 2;
  uint64_t* full_bar_spo          = empty_bar_q + 2;
  uint64_t* empty_bar_spo         = full_bar_spo + 2;
  uint64_t* full_bar_o_acc        = empty_bar_spo + 2;
  uint64_t* full_bar_alpha        = full_bar_o_acc + 2;
  uint64_t* full_bar_l            = full_bar_alpha + 2;
  uint64_t* full_bar_p_last       = full_bar_l + 2;
  uint64_t* empty_bar_alpha_and_l = full_bar_p_last + 2;
  uint64_t* clc_full              = empty_bar_alpha_and_l + 2;
  uint64_t* clc_empty             = clc_full + CLC_STAGES;
  uint64_t* throttle_full         = clc_empty + CLC_STAGES;
  uint64_t* throttle_empty        = throttle_full + 2;
  uint32_t* clc_response = reinterpret_cast<uint32_t*>(
      (reinterpret_cast<uintptr_t>(throttle_empty + 2) + 15u) & ~uintptr_t(15u));
  uint32_t* tmem_slot        = clc_response + CLC_STAGES * 4;
  float*    alpha_and_l_smem = reinterpret_cast<float*>(tmem_slot + 2);

  const int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;

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
    load_warp_blackwell_ntiles_1sm_varlen_bf16_fmha<
        NUM_KV_STAGES, M_TILES_PER_CTA, M_TILE, K_TILE, HEAD_DIM,
        USE_CLC, CLC_STAGES, Q_RASTER, IS_CAUSAL, LPT, /*LOAD_REG_BUDGET=*/56>(wpc, &tmap_q, &tmap_k, &tmap_v_t, sQ0, sKV,
        full_bar, empty_bar, full_bar_q, empty_bar_q,
        clc_full, clc_empty, clc_response, throttle_full, throttle_empty,
        num_q_heads, num_kv_heads, packed_mtiles_per_seq, num_samples,
        cu_seqlens_q, k_base, seqlens_kv);
  }
  else if (warp_id == W_MMA) {
    fmha_mma_warp_blackwell_ntiles_1sm_bf16<
        NUM_KV_STAGES, M_TILES_PER_CTA, M_TILE, K_TILE, HEAD_DIM,
        SPLIT_P, USE_CLC, CLC_STAGES, Q_RASTER, IS_CAUSAL, LPT,
        /*TMEM_TOTAL_COLS=*/TMEM_TOTAL, /*MMA_REG_BUDGET=*/56, /*TAIL_DRAIN=*/false, /*VARLEN=*/true>(wpc, tmem_slot, sQ0, sKV,
        full_bar, empty_bar, full_bar_q, empty_bar_q,
        full_bar_spo, empty_bar_spo, full_bar_p_last, full_bar_o_acc,
        clc_full, clc_empty, clc_response,
        /*seqlen=*/0, num_q_heads, num_kv_heads, packed_mtiles_per_seq, num_samples,
        /*magic0/1/2=*/0, 0, 0, cu_seqlens_q, seqlens_kv);
  }
  else if (warp_id == W_EPI) {
    // Store runs on the 4 corr warps (128-thread register STG); a dedicated 32-thread epi store
    // measured 40% slower (und 237 vs 142), so W_EPI just idles as a CLC consumer here.
    if constexpr (USE_CLC) {
      idle_warp_blackwell_ntiles_2sm_bf16<
          /*IDLE_REG_BUDGET=*/56, /*CLUSTER_SHAPE_M=*/1, /*CLUSTER_SHAPE_N=*/1,
          ClcRasterOrder::AlongN, /*CTA_GROUP=*/1, /*CLC_STAGES=*/CLC_STAGES>(wpc, clc_full, clc_empty, clc_response);
    } else {
      setmaxnreg_dec<56>();   // non-CLC: no work, just keep the register-budget balance
    }
  }
  else if (warp_id == W_SCHED) {
    setmaxnreg_dec<56>();
    if constexpr (USE_CLC) {
      // 1SM CLC scheduler + throttle consumer (block 97). Throttle producer is the load warp.
      sched_warp_clc_blackwell_ntiles_2sm_bf16<
          /*USE_GRIDDEP_WAIT=*/false, /*CLUSTER_SHAPE_M=*/1, /*CLUSTER_SHAPE_N=*/1,
          ClcRasterOrder::AlongN, /*CTA_GROUP=*/1, /*CLC_STAGES=*/CLC_STAGES>(wpc, clc_full, clc_empty, clc_response, throttle_full, throttle_empty,
          /*cluster_rank=*/0, lane);
    }
  }
  else if (warp_id >= W_CORR0 && warp_id < W_MMA) {
    correction_warp_blackwell_ntiles_1sm_varlen_bf16_fmha<
        M_TILE, M_TILES_PER_CTA, HEAD_DIM, K_TILE, FULL_NAMED_BAR,
        USE_CLC, CLC_STAGES, Q_RASTER, IS_CAUSAL, LPT, MHA, /*CORR_REG_BUDGET=*/72>(
        tmem_slot, warp_id - W_CORR0, lane, alpha_and_l_smem, sO, O_out,
        full_bar_alpha, full_bar_l, full_bar_o_acc, empty_bar_spo, empty_bar_alpha_and_l,
        clc_full, clc_empty, clc_response,
        num_q_heads, num_kv_heads, packed_mtiles_per_seq, num_samples,
        cu_seqlens_q, seqlens_kv);
  }
  else {
    softmax_warp_blackwell_ntiles_1sm2sm_bf16_fmha<
        K_TILE, M_TILE, M_TILES_PER_CTA, S_LD_COLS, FULL_NAMED_BAR, SPLIT_P, EX2_EMU,
        USE_CLC, CLC_STAGES, Q_RASTER, IS_CAUSAL, LPT, /*SM_REG_BUDGET=*/192,
        /*TAIL_DRAIN=*/false, /*CLUSTER_N=*/1, /*VARLEN=*/true>(wpc, tmem_slot, warp_id, lane, scale_log2, alpha_and_l_smem,
        full_bar_spo, full_bar_alpha, full_bar_l, empty_bar_spo, empty_bar_alpha_and_l, full_bar_p_last,
        clc_full, clc_empty, clc_response,
        /*seqlen=*/0, num_q_heads, num_kv_heads, packed_mtiles_per_seq, num_samples,
        /*magic0/1/2=*/0, 0, 0, cu_seqlens_q, seqlens_kv);
  }
}

// ====================== driver ============================
// Input fill modes -- MUST match kernels/gemm/sm100a/dense_gemm_bf16.cu and
// fa4_extract/fa4_bench_matrix.py so ours vs FA4 see BITWISE-IDENTICAL inputs (input values
// swing tensor-core power -> clock -> timing by up to ~40%). FILL env (default 2):
//   1=[-0.5,0.5)/256  2=[-1,1)/2048  3=const 1.0  4=int{-3..3}. Seeds 11/22/33 for Q/K/V.
static void fillr(__nv_bfloat16 *h, long n, unsigned s) {
  const char* e = getenv("FILL");
  int mode = e ? atoi(e) : 2;
  for (long i = 0; i < n; ++i) {
    uint32_t x = (uint32_t)i * 2654435761u + s;
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
struct Sh {
  std::vector<int> sl;         // Q seqlens per sample
  std::vector<int> slk;        // K/V seqlens per sample (empty = self-attn, = sl)
  int nqh, nkh, hd;
  bool causal;
  const char *lab;
};
template <bool MHA = false>
static double run(const Sh &sh, bool verify) {
  int ns = (int)sh.sl.size();
  const std::vector<int> &skl = sh.slk.empty() ? sh.sl : sh.slk;   // cross-attn: skl != sl
  std::vector<int> cq(ns + 1, 0), ck(ns + 1, 0);
  for (int i = 0; i < ns; ++i) {
    cq[i + 1] = cq[i] + sh.sl[i];
    ck[i + 1] = ck[i] + skl[i];
  }
  long tq = cq.back();
  std::vector<int> kb(ns), sk(ns);
  long acc = 0;
  for (int i = 0; i < ns; ++i) {
    kb[i] = (int)acc;
    sk[i] = skl[i];
    acc += ((long)skl[i] + 7) / 8 * 8;
  }
  long tka = acc > 0 ? acc : 8;
  __nv_bfloat16 *dQ, *dK, *dVT, *dO;
  CUDA_CHECK(cudaMalloc(&dQ, tq * sh.nqh * sh.hd * 2));
  CUDA_CHECK(cudaMalloc(&dK, tka * sh.nkh * sh.hd * 2));
  CUDA_CHECK(cudaMalloc(&dVT, (long)sh.nkh * sh.hd * tka * 2));
  CUDA_CHECK(cudaMalloc(&dO, tq * sh.nqh * sh.hd * 2));
  long tk = ck.back();
  std::vector<__nv_bfloat16> hQ(tq * sh.nqh * sh.hd), hK(tk * sh.nkh * sh.hd),
      hV(tk * sh.nkh * sh.hd);
  fillr(hQ.data(), hQ.size(), 11);
  fillr(hK.data(), hK.size(), 22);
  fillr(hV.data(), hV.size(), 33);
  std::vector<__nv_bfloat16> hKa(tka * sh.nkh * sh.hd, __float2bfloat16(0.f)),
      hVT((long)sh.nkh * sh.hd * tka, __float2bfloat16(0.f));
  for (int s = 0; s < ns; ++s)
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
  CUDA_CHECK(cudaMalloc(&dCq, (ns + 1) * 4));
  CUDA_CHECK(cudaMalloc(&dKb, ns * 4));
  CUDA_CHECK(cudaMalloc(&dSk, ns * 4));
  CUDA_CHECK(cudaMemcpy(dCq, cq.data(), (ns + 1) * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dKb, kb.data(), ns * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dSk, sk.data(), ns * 4, cudaMemcpyHostToDevice));
  CUtensorMap tq_, tk_, tvt_;
  // pack-GQA Q: 3D TMA over [hd, nqh, total_q]; box [hd-subtile x gqa_group_size x
  // q_tile_per_mtile] -> 128 packed rows (qh-inner).
  int gqa = sh.nqh / sh.nkh, tpi = M_TILE / gqa, tpc = 2 * tpi;
  CUDA_CHECK(make_tma_3d_tiled(&tq_, dQ, sh.hd, sh.nqh, (int)tq, SUB_COLS_BF16, gqa, tpi, 2,
                               CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B));
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
  int mqt = 1;
  for (int i = 0; i < ns; ++i)
    mqt = std::max(mqt, (sh.sl[i] + tpc - 1) / tpc); // packed-M tiles (q_tile_per_cta tokens)
  constexpr bool USE_CLC = true, Q_RASTER = !MHA;
  constexpr bool SPLIT_P = true;
  constexpr bool LPT = true;   // heaviest causal q-tile first: +3.6% on UND
  size_t smem =
      (size_t)2 * Q_TILE_BYTES + NUM_KV_STAGES * K_TILE_BYTES // shared K/V ring
      + (size_t)M_TILE * HEAD_DIM * sizeof(__nv_bfloat16) // sO epilogue staging [M_TILE][HEAD_DIM]
      + (2 * NUM_KV_STAGES + 18) * 8       // mbarriers (full_bar..empty_bar_alpha_and_l)
      + (size_t)CLC_STAGES * (2 * 8 + 16) + 16  // CLC: clc_full+clc_empty + response (16B aligned)
      + 4 * 8                              // throttle_full[2] + throttle_empty[2]
      + 8                                  // tmem_slot
      + (size_t)2 * M_TILE * sizeof(float) // alpha_and_l_smem [2][M_TILE]
      + 256;                               // slack/alignment
  auto kfn = sh.causal ? &fmha_context_bf16_persistent_kernel<true, 32, false, false, SPLIT_P, LPT, USE_CLC, Q_RASTER, MHA>
                       : &fmha_context_bf16_persistent_kernel<false, 32, false, false, SPLIT_P, LPT, USE_CLC, Q_RASTER, MHA>;
  CUDA_CHECK(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  float sl2 = (1.0f / sqrtf((float)sh.hd)) * (float)M_LOG2E;
  int numSM = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, 0));
  int total_work = ns * mqt * sh.nkh;     // pack-GQA: one CTA per (kv-head, packed-M tile)
  int nblk = USE_CLC ? total_work : std::min(total_work, numSM);
  (void)numSM;
  dim3 grid(nblk, 1, 1), block(N_WARPS * 32, 1, 1);

  // CLC must be launched via cudaLaunchKernelEx with a cluster-dimension attribute -- a plain
  // <<<grid,block>>> launch does NOT enable clusterlaunchcontrol (try_cancel silently misbehaves
  // and tiles get skipped). cluster {1,1,1} (matches __cluster_dims__(1,1,1)).
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
      return cudaLaunchKernelEx(&cfg, kfn, tq_, tk_, tvt_, dO, dCq, dKb, dSk, sh.nqh, sh.nkh, sl2, mqt, ns);
    kfn<<<grid, block, smem, st>>>(tq_, tk_, tvt_, dO, dCq, dKb, dSk, sh.nqh, sh.nkh, sl2, mqt, ns);
    return cudaGetLastError();
  };
  const double ms = fmha_context_bf16_benchmark::measure(launch);
  uint64_t eff = 0;
  for (int i = 0; i < ns; ++i) {
    uint64_t Lq = sh.sl[i];
    uint64_t Lk = sh.slk.empty() ? Lq : (uint64_t)sh.slk[i];
    eff += fmha_context_bf16_benchmark::attended_pairs(Lq, Lk, sh.causal);
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
  // stress mode: STRESS_N kernel launches, cross-run output consistency (memcmp vs run 0).
  if (const char* sn = getenv("STRESS_N")) {
    const int N = atoi(sn);
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
  cudaFree(dCq);
  cudaFree(dKb);
  cudaFree(dSk);
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
  printf("K2 fmha_context_bf16 persistent (warp-spec, 2 M-tiles) "
         "sm_100a\n=====================================\n");
  // Default: GQA varlen (und = target shape). MHA=1 -> uniform MHA (HQ==HK), env BATCH/SEQLEN/
  // HEADS. CAUSAL (default 1): triangular causal vs full. NOVERIFY=1 skips the CPU ref.
  const bool causal = getenv("CAUSAL") ? (atoi(getenv("CAUSAL")) != 0) : true;
  if (getenv("CROSS") && atoi(getenv("CROSS"))) {
    // Cross-attention: per-sample Q length != K/V length (Sk >, ==, >>, < Sq).
    const bool mha = getenv("MHA") && atoi(getenv("MHA"));
    Sh s{};
    s.sl  = {100, 240, 64, 175};
    s.slk = {300, 240, 512, 90};
    s.nqh = 32; s.nkh = mha ? 32 : 4; s.hd = 128;
    s.causal = causal;
    s.lab = causal ? "cross" : "cross-full";
    if (mha) run</*MHA=*/true>(s, true); else run</*MHA=*/false>(s, true);
  } else if (getenv("MHA") && atoi(getenv("MHA"))) {
    Sh s{};
    const int B = getenv("BATCH")  ? atoi(getenv("BATCH"))  : 8;
    const int S = getenv("SEQLEN") ? atoi(getenv("SEQLEN")) : 512;
    const int H = getenv("HEADS")  ? atoi(getenv("HEADS"))  : 32;
    s.sl = std::vector<int>(B, S);
    s.nqh = H; s.nkh = H; s.hd = 128;
    s.causal = causal;
    s.lab = causal ? "mha-causal" : "mha-full";
    run</*MHA=*/true>(s, getenv("NOVERIFY") ? false : (S <= 1024));
  } else {
    Sh s{};
    s.sl = und();
    s.nqh = 32; s.nkh = 4; s.hd = 128;
    s.causal = causal;
    s.lab = causal ? "und" : "und-full";
    run</*MHA=*/false>(s, true);
  }
  return 0;
}
