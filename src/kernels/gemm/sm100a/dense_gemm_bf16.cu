// dense_gemm_bf16.cu -- K0 dense BF16 GEMM, sm_100a (GB200).
//
// 8 warps per CTA, roles:
//   warp 0 : MMA (cta_group::2)
//   warp 1 : sched (CLC try_cancel)
//   warp 2 : load (TMA producer)
//   warp 3 : idle (CLC handshake participant)
//   warps 4-7 : epilogue (each owns 32 rows of M; both peers)
//
// Tile geometry: M_TILE_CLUSTER=256, M_TILE_PER_CTA=128,
// N_TILE_CLUSTER=256/128 (template), N_TILE_PER_CTA=NTC/2, K_TILE=64,
// NUM_STAGES=6, EPI_SUB_COLS=32.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>

#include "dense_gemm_bf16_benchmark.cuh"
#include "../../../../tests/test_utils.cuh"
#include "../../../primitives/_warp_prof_noop.cuh"

#include "../../../primitives/0_tcgen05_alloc.cuh"
#include "../../../primitives/1_tcgen05_dealloc.cuh"
#include "../../../primitives/2_tcgen05_relinquish.cuh"
#include "../../../primitives/3_tcgen05_mma_f16.cuh"
#include "../../../primitives/8_tcgen05_mma_idesc.cuh"
#include "../../../primitives/9_tcgen05_ld.cuh"
#include "../../../primitives/11_tcgen05_commit.cuh"
#include "../../../primitives/12_tcgen05_wait.cuh"
#include "../../../primitives/18_tma_load.cuh"
#include "../../../primitives/19_tma_load_2sm.cuh"
#include "../../../primitives/22_tma_store.cuh"
#include "../../../primitives/23_tma_tensormap.cuh"
#include "../../../primitives/25_tma_async_group.cuh"
#include "../../../primitives/29_mbarrier_init.cuh"
#include "../../../primitives/30_mbarrier_arrive.cuh"
#include "../../../primitives/31_mbarrier_arrive_tx.cuh"
#include "../../../primitives/33_mbarrier_try_wait.cuh"
#include "../../../primitives/34_fence_proxy_async.cuh"
#include "../../../primitives/35_fence_mbarrier_init.cuh"
#include "../../../primitives/38_barrier_cluster.cuh"
#include "../../../primitives/40_stmatrix.cuh"
#include "../../../primitives/42_smem_desc_blackwell.cuh"
#include "../../../primitives/61_clc_try_cancel.cuh"
#include "../../../primitives/62_clc_query_cancel.cuh"
#include "../../../primitives/63_cvt_f32_to_f16_bf16.cuh"
#include "../../../primitives/68_l2cache_policy.cuh"
#include "../../../primitives/69_griddepcontrol.cuh"

#include "../../../composites/119_pipeline_init_blackwell.cuh"
#include "../../../composites/82_tile_rasterize.cuh"
#include "../../../composites/104_acc_pipeline_2bank_blackwell.cuh"
#include "../../../composites/106_clc_fetch_next_tile.cuh"
#include "../../../blocks/88_load_warp_blackwell.cuh"
#include "../../../blocks/90_mma_warp_blackwell.cuh"
#include "../../../blocks/93_epi_warp_blackwell.cuh"
#include "../../../blocks/97_sched_warp_clc.cuh"
#include "../../../blocks/107_idle_warp_blackwell.cuh"

// Compile-time tile geometry constants (skel_v2-style, no Cfg struct).
constexpr int K0_M_TILE_CLUSTER = 256;
constexpr int K0_M_TILE_PER_CTA = 128;
constexpr int K0_K_TILE         = 64;
constexpr int K0_NUM_STAGES     = 5;
constexpr int K0_EPI_SUB_COLS   = 64;
constexpr int K0_EPI_NUM_BUFS   = 2;
constexpr int K0_A_TILE_BYTES   = K0_M_TILE_PER_CTA * K0_K_TILE * 2;
constexpr int K0_EPI_BUF_BYTES  = K0_M_TILE_PER_CTA * K0_EPI_SUB_COLS * 2;
constexpr int K0_D_TILE_BYTES   = K0_EPI_NUM_BUFS * K0_EPI_BUF_BYTES;

extern __shared__ __align__(1024) uint8_t k0_smem[];

__device__ __forceinline__
uint32_t mapa_shared_cluster(uint32_t local_addr, int target_rank) {
  uint32_t out;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n"
               : "=r"(out) : "r"(local_addr), "r"(target_rank));
  return out;
}

template <int K_BLOCKS_T, int NTC, ClcRasterOrder ORDER = ClcRasterOrder::AlongN>
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(256, 1)
dense_gemm_bf16_k0_impl(
    const __grid_constant__ CUtensorMap tmap_a,
    const __grid_constant__ CUtensorMap tmap_b,
    const __grid_constant__ CUtensorMap tmap_d,
    int M, int N) {
  constexpr int K = K_BLOCKS_T * K0_K_TILE;
  constexpr int M_TILE_CLUSTER = K0_M_TILE_CLUSTER;
  constexpr int M_TILE_PER_CTA = K0_M_TILE_PER_CTA;
  constexpr int N_TILE_CLUSTER = NTC;
  constexpr int N_TILE_PER_CTA = NTC / 2;
  constexpr int K_TILE         = K0_K_TILE;
  constexpr int NUM_STAGES     = K0_NUM_STAGES;
  constexpr int EPI_SUB_COLS   = K0_EPI_SUB_COLS;
  constexpr int EPI_NUM_BUFS   = K0_EPI_NUM_BUFS;
  constexpr int A_TILE_BYTES   = K0_A_TILE_BYTES;
  constexpr int B_TILE_BYTES   = N_TILE_PER_CTA * K0_K_TILE * 2;
  constexpr int EPI_SUB_COUNT  = N_TILE_CLUSTER / K0_EPI_SUB_COLS;

  // CLC raster shape derived from ORDER.
  //   AlongN: cluster.x -> N (peers split N), CSM=1, CSN=2.
  //   AlongM: cluster.x -> M (peers split M), CSM=2, CSN=1.
  constexpr int CLC_CSM = (ORDER == ClcRasterOrder::AlongN) ? 1 : 2;
  constexpr int CLC_CSN = (ORDER == ClcRasterOrder::AlongN) ? 2 : 1;

  // SMEM layout: mbars at offset 0, data after.
  uint64_t* smem_bar       = reinterpret_cast<uint64_t*>(k0_smem);
  uint64_t* full_bar       = smem_bar + 0;
  uint64_t* empty_bar      = smem_bar + NUM_STAGES;
  uint64_t* acc_full       = smem_bar + NUM_STAGES * 2 + 0;
  uint64_t* acc_empty      = smem_bar + NUM_STAGES * 2 + 2;
  uint64_t* clc_full_bar   = smem_bar + NUM_STAGES * 2 + 4;
  uint64_t* clc_empty_bar  = smem_bar + NUM_STAGES * 2 + 6;
  uint64_t* throttle_full    = smem_bar + NUM_STAGES * 2 + 8;
  uint64_t* throttle_empty   = smem_bar + NUM_STAGES * 2 + 10;
  // tmem_dealloc_bar uses 1 uint64_t but reserves 2 slots so
  // clc_response stays 16-byte aligned (clusterlaunchcontrol.try_cancel
  // .b128 requires 16-byte aligned smem_dst).
  uint64_t* tmem_dealloc_bar = smem_bar + NUM_STAGES * 2 + 12;  // [1] +1 pad
  uint32_t* clc_response     = reinterpret_cast<uint32_t*>(
                                   smem_bar + NUM_STAGES * 2 + 14);
  uint32_t* tmem_slot        = clc_response + 8;
  // Data bytes start after mbars (round up for 128-byte alignment).
  constexpr int BAR_BYTES_TOTAL = (NUM_STAGES * 2 + 4 + 4 + 4 + 2) * sizeof(uint64_t)
                                  + 8 * sizeof(uint32_t) + sizeof(uint32_t);
  constexpr int DATA_OFFSET = ((BAR_BYTES_TOTAL + 127) / 128) * 128;
  uint8_t*  smem_a   = k0_smem + DATA_OFFSET;
  uint8_t*  smem_b   = smem_a + NUM_STAGES * A_TILE_BYTES;
  uint8_t*  smem_d   = smem_b + NUM_STAGES * B_TILE_BYTES;
  AccPipeline2BankBars acc_bars{ acc_full, acc_empty };
  BlackwellPipelineBars pipe_bars{
      full_bar, empty_bar,
      acc_full, acc_empty,
      clc_full_bar, clc_empty_bar,
      throttle_full, throttle_empty
  };

  const int peer = blockIdx.x & 1;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  WpCtx wpc = wp_ctx_init();  // WARP_PROF: per-warp slot + lifetime start (no-op without flag)

  if (threadIdx.x == 0) {
    for (int i = 0; i < 8; ++i) clc_response[i] = 0;
    // TMEM dealloc handshake bar: one warp from each CTA sends 32
    // arrives, so arrive_count=32. Init here (before pipeline_init's
    // fence + cluster sync) so the same fence/sync covers it.
    mbarrier_init(smem_ptr_u32(tmem_dealloc_bar), 32);
  }
  pipeline_init_blackwell<NUM_STAGES, /*CTA_GROUP=*/2>(pipe_bars);

  uint32_t tmem_base = 0;

  // A and B are both K-major: TMA loads BT (N x K row-major) so K is
  // contiguous in each B SMEM tile, matching A's K-major layout.
  constexpr uint32_t A_LBO = 16;
  constexpr uint32_t A_SBO = 1024;
  constexpr uint32_t B_LBO = 16;
  constexpr uint32_t B_SBO = 1024;
  // Single base descriptors only; per-stage offsets are computed on
  // the fly in the MMA warp via STAGE_DELTA. Putting a NUM_STAGES-sized
  // array on the kernel stack forces ptxas into local memory (LDL/STL
  // traffic stalls the MMA pipeline and trips the tcgen05.alloc HW
  // guardrail across kernel launches).
  const uint64_t desc_a_stage0 = build_smem_desc_blackwell(
      smem_ptr_u32(smem_a), A_SBO, A_LBO, SmemSwizzleBlackwell::B128);
  const uint64_t desc_b_stage0 = build_smem_desc_blackwell(
      smem_ptr_u32(smem_b), B_SBO, B_LBO, SmemSwizzleBlackwell::B128);
  // SMEM-desc start_address is in 16-byte units in low 14 bits.
  constexpr uint64_t A_STAGE_DELTA = A_TILE_BYTES / 16;
  constexpr uint64_t B_STAGE_DELTA = B_TILE_BYTES / 16;


  if (warp == 0) {
    mma_warp_blackwell_ntiles_2sm_bf16<
        NUM_STAGES, M_TILE_CLUSTER, N_TILE_CLUSTER, K_TILE,
        A_STAGE_DELTA, B_STAGE_DELTA,
        /*TMEM_NCOLS=*/2 * N_TILE_CLUSTER,
        CLC_CSM, CLC_CSN, ORDER>(wpc, desc_a_stage0, desc_b_stage0, full_bar, empty_bar,
        acc_bars, clc_full_bar, clc_empty_bar, clc_response,
        tmem_dealloc_bar,
        /*tmem_base=*/0u, K, peer, lane, /*tmem_slot=*/tmem_slot);
  } else if (warp == 1) {
    // griddepcontrol.wait at sched-warp entry pairs with the host-side
    // cudaLaunchAttributeProgrammaticStreamSerialization attribute so
    // the dependent grid's setup overlaps with the prior grid's tail
    // (K0 TODO 3, ~1-2% expected).
    sched_warp_clc_blackwell_ntiles_2sm_bf16<
        /*USE_GRIDDEP_WAIT=*/true, CLC_CSM, CLC_CSN, ORDER>(wpc, clc_full_bar, clc_empty_bar, clc_response,
        throttle_full, throttle_empty, peer, lane);
  } else if (warp == 2) {
    // L2::evict_last on every TMA load -- bias the persistent kernel's
    // working set (A and B operand slabs) toward staying in L2 across
    // CLC-stolen tiles. K0 TODO 2 (~2-5% expected on compute-bound
    // shapes per V56 static comparison).
    const uint64_t cache_policy = make_l2cache_policy_evict_last_full();
    load_warp_blackwell_ntiles_2sm_bf16<
        NUM_STAGES, M_TILE_PER_CTA, N_TILE_PER_CTA, K_TILE,
        M_TILE_CLUSTER, N_TILE_CLUSTER, /*LOAD_REG_BUDGET=*/40,
        /*USE_L2_HINT=*/true, CLC_CSM, CLC_CSN, ORDER>(wpc, &tmap_a, &tmap_b, smem_a, smem_b, full_bar, empty_bar,
        clc_full_bar, clc_empty_bar, clc_response,
        throttle_full, throttle_empty,
        K, peer, cache_policy);
  } else if (warp == 3) {
    idle_warp_blackwell_ntiles_2sm_bf16<
        /*IDLE_REG_BUDGET=*/24, CLC_CSM, CLC_CSN, ORDER>(wpc, clc_full_bar, clc_empty_bar, clc_response);
  } else {
    epi_warp_blackwell_ntiles_2sm_bf16<
        M_TILE_PER_CTA, N_TILE_PER_CTA, K_TILE,
        M_TILE_CLUSTER, N_TILE_CLUSTER,
        EPI_SUB_COLS, EPI_NUM_BUFS,
        CLC_CSM, CLC_CSN, ORDER,
        /*DRAIN_PER_TILE=*/false>(wpc, &tmap_d, smem_d, acc_bars,
        clc_full_bar, clc_empty_bar, clc_response,
        /*tmem_base=*/0u, peer, warp, lane,
        /*m_tile_remap=*/nullptr, /*tmem_slot=*/tmem_slot);
  }

  // TMEM teardown (acc_empty drain + tmem_dealloc_bar handshake +
  // tcgen05.relinquish + tcgen05.dealloc<2>) is encapsulated in the
  // mma_warp_blackwell_ntiles_2sm_bf16 helper above. EPI's per-tile
  // cp.async.bulk.wait_group<0> drains TMA
  // stores at every tile boundary; no kernel-level defensive drain
  // needed here.
  //
  // Release dependent grids early so the next kernel's
  // griddepcontrol.wait can return; pairs with PSS.
  wp_flush(wpc);  // WARP_PROF: per-warp lifetime end
  if (lane == 0) griddepcontrol_launch_dependents();
}

// CUDA-cores GEMM reference: one block per row, dot product over K.
__global__ void ref_gemm_bf16_kernel(const __nv_bfloat16* __restrict__ A,
                                     const __nv_bfloat16* __restrict__ BT,
                                     __nv_bfloat16* __restrict__ D,
                                     int M, int N, int K) {
  int m = blockIdx.x;
  if (m >= M) return;
  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      acc += __bfloat162float(A[m * K + k]) *
             __bfloat162float(BT[n * K + k]);
    }
    D[m * N + n] = __float2bfloat16(acc);
  }
}

// Fill mode selector (set from --fill=N CLI arg in main).
//   1 = [-0.5, 0.5) with 256 discrete values  (default; v56port-style)
//   2 = [-1.0, 1.0) with 2048 discrete values (wider range)
//   3 = constant 1.0                          (debug: partial-write detection)
//   4 = integers {-3,..,3} (7 values)         (cutlass_profiler FP16 default)
static int g_fill_mode = 1;
static const char* fill_mode_label(int m) {
  switch (m) {
    case 1: return "[-0.5,0.5) /256";
    case 2: return "[-1.0,1.0) /2048";
    case 3: return "const 1.0";
    case 4: return "int{-3..3} /7 (profiler)";
    default: return "unknown";
  }
}
static void fill_random_bf16(__nv_bfloat16* h, int n, unsigned seed) {
  for (int i = 0; i < n; ++i) {
    float v;
    switch (g_fill_mode) {
      case 2: v = ((i * 2654435761u + seed) % 2048) / 1024.0f - 1.0f; break;
      case 3: v = 1.0f; (void)seed; (void)i; break;
      case 4: v = (float)((int)((i * 2654435761u + seed) % 7) - 3); break;
      default: v = (((i * 2654435761u + seed) % 256) / 256.0f) - 0.5f; break;
    }
    h[i] = __float2bfloat16(v);
  }
}

template <int K_BLOCKS_T, int NTC, ClcRasterOrder ORDER = ClcRasterOrder::AlongN>
static double run_one_impl(int M, int N, bool verify, bool benchmark,
                           const char* dump_output) {
  constexpr int N_TILE_CLUSTER = NTC;
  constexpr int N_TILE_PER_CTA = NTC / 2;
  constexpr int K = K_BLOCKS_T * K0_K_TILE;
  if (M % K0_M_TILE_CLUSTER || N % N_TILE_CLUSTER) {
    printf("  shape (%d, %d, %d): NOT MULTIPLE OF TILE (NTC=%d) -- skipping\n",
           M, N, K, NTC);
    return 0.0;
  }

  __nv_bfloat16 *dA = nullptr, *dBT = nullptr, *dD = nullptr;
  CUDA_CHECK(cudaMalloc(&dA,     (size_t)M * K * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&dBT,    (size_t)N * K * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&dD,     (size_t)M * N * sizeof(__nv_bfloat16)));

  std::vector<__nv_bfloat16> hA((size_t)M * K), hB((size_t)K * N),
                              hBT((size_t)N * K);
  fill_random_bf16(hA.data(),  M * K, /*seed=*/1);
  fill_random_bf16(hB.data(),  K * N, /*seed=*/2);
  for (int k_idx = 0; k_idx < K; ++k_idx)
    for (int n_idx = 0; n_idx < N; ++n_idx)
      hBT[(size_t)n_idx * K + k_idx] = hB[(size_t)k_idx * N + n_idx];
  CUDA_CHECK(cudaMemcpy(dA,  hA.data(),  hA.size()  * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dBT, hBT.data(), hBT.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dD,     0, (size_t)M * N * sizeof(__nv_bfloat16)));

  CUtensorMap tmap_a{}, tmap_b{}, tmap_d{};
  CUDA_CHECK(make_tma_2d_tiled(&tmap_a, dA, M, K,
                               K0_M_TILE_PER_CTA, K0_K_TILE,
                               sizeof(__nv_bfloat16),
                               CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                               CU_TENSOR_MAP_SWIZZLE_128B));
  CUDA_CHECK(make_tma_2d_tiled(&tmap_b, dBT, N, K,
                               N_TILE_PER_CTA, K0_K_TILE,
                               sizeof(__nv_bfloat16),
                               CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                               CU_TENSOR_MAP_SWIZZLE_128B));
  CUDA_CHECK(make_tma_2d_tiled(&tmap_d, dD, M, N,
                               K0_M_TILE_PER_CTA, K0_EPI_SUB_COLS,
                               sizeof(__nv_bfloat16),
                               CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                               CU_TENSOR_MAP_SWIZZLE_NONE));

  const int m_clusters = M / K0_M_TILE_CLUSTER;
  const int n_clusters = N / N_TILE_CLUSTER;

  // Grid: cluster.x always spans 2 CTAs (cluster_dims=(2,1,1)); the
  // factor-of-2 axis tracks whichever logical axis the raster maps to x.
  dim3 grid;
  if constexpr (ORDER == ClcRasterOrder::AlongN) {
    grid = dim3(2 * n_clusters, m_clusters, 1);
  } else {  // AlongM
    grid = dim3(2 * m_clusters, n_clusters, 1);
  }
  dim3 block(256, 1, 1);
  size_t smem_bytes = 1024 + 2 * K0_NUM_STAGES * K0_A_TILE_BYTES
                      + K0_D_TILE_BYTES + 256;

  CUDA_CHECK(cudaFuncSetAttribute((dense_gemm_bf16_k0_impl<K_BLOCKS_T, NTC, ORDER>),
      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));
  CUDA_CHECK(cudaFuncSetAttribute((dense_gemm_bf16_k0_impl<K_BLOCKS_T, NTC, ORDER>),
      cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

  // Benchmark defaults remain 10 warmups + 100 measured launches on this stream.
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  cudaLaunchConfig_t config = {};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = smem_bytes;
  config.stream = stream;
  cudaLaunchAttribute attrs[2];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim = {2, 1, 1};
  attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[1].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attrs;
  config.numAttrs = 2;

  WpBuffer wpbuf = wp_alloc(grid);  // WARP_PROF (no-op without -DWARP_PROF)

  auto launch = [&](size_t) {
    return cudaLaunchKernelEx(&config, dense_gemm_bf16_k0_impl<K_BLOCKS_T, NTC, ORDER>,
        tmap_a, tmap_b, tmap_d, M, N);
  };
  double ms = 0.0;
  if (benchmark) {
    ms = dense_gemm_bf16_benchmark::measure(stream, launch);
  } else {
    CUDA_CHECK(launch(0));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  // WARP_PROF: one clean profiled launch.
  wp_reset(wpbuf);
  cudaLaunchKernelEx(&config, dense_gemm_bf16_k0_impl<K_BLOCKS_T, NTC, ORDER>,
      tmap_a, tmap_b, tmap_d, M, N);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  cudaStreamDestroy(stream);
  double tflops = dense_gemm_bf16_benchmark::report(M, N, K, ms, benchmark);
  {
    static const char* role[8] = {"mma","sched","load","idle","epi","epi","epi","epi"};
    wp_readback(wpbuf);
    wp_print_busy(wpbuf, role, 8, /*block=*/0);
    // A trusted harness can request one exact output path for any selected case.
    const char* requested_raw = getenv("WP_RAW_OUTPUT");
    if (requested_raw && requested_raw[0]) {
      wp_dump_raw(wpbuf, requested_raw, /*block=*/0, /*nstages=*/K0_NUM_STAGES);
    // Preserve the original standalone default for the representative shape.
    } else if (M == 14080 && N == 5120 && K == 2048) {
      char tpath[96], rpath[96];
      snprintf(tpath, sizeof(tpath), "warp_trace_dense_gemm_%dx%dx%d.json.gz", M, N, K);
      snprintf(rpath, sizeof(rpath), "warp_raw_dense_gemm_%dx%dx%d.bin.gz", M, N, K);
      wp_dump_raw(wpbuf, rpath, /*block=*/0, /*nstages=*/K0_NUM_STAGES);  // raw + pipeline depth
    }
    wp_free(wpbuf);
  }

  if (dump_output) {
    std::vector<__nv_bfloat16> output((size_t)M * N);
    CUDA_CHECK(cudaMemcpy(output.data(), dD, output.size() * 2,
                          cudaMemcpyDeviceToHost));
    FILE* file = fopen(dump_output, "wb");
    if (!file || fwrite(output.data(), 2, output.size(), file) != output.size()) {
      if (file) fclose(file);
      fprintf(stderr, "failed to write %s\n", dump_output);
      exit(1);
    }
    fclose(file);
    printf("  wrote %zu BF16 values to %s\n", output.size(), dump_output);
  }

  if (verify) {
    __nv_bfloat16* dD_ref = nullptr;
    CUDA_CHECK(cudaMalloc(&dD_ref, (size_t)M * N * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemset(dD_ref, 0, (size_t)M * N * sizeof(__nv_bfloat16)));
    ref_gemm_bf16_kernel<<<dim3(M, 1, 1), dim3(256, 1, 1)>>>(
        dA, dBT, dD_ref, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<__nv_bfloat16> hD(M * N), hRef(M * N);
    CUDA_CHECK(cudaMemcpy(hD.data(),  dD,     hD.size()  * 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hRef.data(), dD_ref, hRef.size() * 2, cudaMemcpyDeviceToHost));
    std::vector<float> fOut(M * N), fRef(M * N);
    for (size_t i = 0; i < (size_t)M * N; ++i) {
      fOut[i] = __bfloat162float(hD[i]);
      fRef[i] = __bfloat162float(hRef[i]);
    }
    // Compute max absolute diff.
    float max_diff = 0.0f;
    for (size_t i = 0; i < (size_t)M * N; ++i) {
      float d = fabsf(fOut[i] - fRef[i]);
      if (d > max_diff) max_diff = d;
    }
    bool ok = check_close_f32(fRef.data(), fOut.data(), M * N,
                              /*atol=*/0.05f, /*rtol=*/0.05f);
    printf("  verify (M=%d,N=%d,K=%d): max_diff=%.3f tight=%s\n",
           M, N, K, max_diff, ok ? "OK" : "FAIL");
    cudaFree(dD_ref);
  }

  cudaFree(dA); cudaFree(dBT); cudaFree(dD);
  return tflops;
}

template <int NTC, ClcRasterOrder ORDER>
static double run_one_kdispatch(int M, int N, int K, bool verify,
                                bool benchmark, const char* dump_output) {
  // Force-instantiate K_BLOCKS ∈ {1,2,4,8,32,64} to match skel_v2's
  // cubin instantiation set exactly.
  switch (K) {
    case   64: return run_one_impl< 1, NTC, ORDER>(M, N, verify, benchmark, dump_output);
    case  128: return run_one_impl< 2, NTC, ORDER>(M, N, verify, benchmark, dump_output);
    case  256: return run_one_impl< 4, NTC, ORDER>(M, N, verify, benchmark, dump_output);
    case  512: return run_one_impl< 8, NTC, ORDER>(M, N, verify, benchmark, dump_output);
    case 2048: return run_one_impl<32, NTC, ORDER>(M, N, verify, benchmark, dump_output);
    case 4096: return run_one_impl<64, NTC, ORDER>(M, N, verify, benchmark, dump_output);
  }
  printf("  shape (%d, %d, %d): K not in {64,128,256,512,2048,4096} -- skipping\n",
         M, N, K);
  return 0.0;
}

// Pick NTC=256 for N>=256, NTC=128 for narrow-N. N must be a multiple of NTC.
template <ClcRasterOrder ORDER = ClcRasterOrder::AlongN>
static double run_one(int M, int N, int K, bool verify, bool benchmark = true,
                      const char* dump_output = nullptr) {
  if (N >= 256 && (N % 256) == 0) {
    return run_one_kdispatch<256, ORDER>(M, N, K, verify, benchmark, dump_output);
  } else if (N >= 128 && (N % 128) == 0) {
    return run_one_kdispatch<128, ORDER>(M, N, K, verify, benchmark, dump_output);
  } else {
    printf("  shape (%d, %d, %d): N not a multiple of 128 -- skipping\n",
           M, N, K);
    return 0.0;
  }
}

template <ClcRasterOrder ORDER = ClcRasterOrder::AlongN>
static double run_verify(int M, int N, int K) {
  return run_one<ORDER>(M, N, K, /*verify=*/true);
}

#ifndef MOE_DISABLE_MAIN
int main(int argc, char** argv) {
  int custom_m = 0, custom_k = 0, custom_n = 0;
  bool benchmark = true, gpu_verify = true, along_m = false;
  const char* dump_output = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (strncmp(argv[i], "--fill=", 7) == 0) {
      g_fill_mode = atoi(argv[i] + 7);
      if (g_fill_mode < 1 || g_fill_mode > 4) {
        fprintf(stderr, "--fill must be 1, 2, 3, or 4\n"); return 1;
      }
    } else if (strncmp(argv[i], "--shape=", 8) == 0) {
      if (sscanf(argv[i] + 8, "%d,%d,%d", &custom_m, &custom_k, &custom_n) != 3) {
        fprintf(stderr, "--shape must be M,K,N\n"); return 1;
      }
    } else if (strcmp(argv[i], "--no-benchmark") == 0) {
      benchmark = false;
    } else if (strcmp(argv[i], "--no-gpu-verify") == 0) {
      gpu_verify = false;
    } else if (strcmp(argv[i], "--raster=along-m") == 0) {
      along_m = true;
    } else if (strcmp(argv[i], "--raster=along-n") == 0) {
      along_m = false;
    } else if (strncmp(argv[i], "--dump-output=", 14) == 0) {
      dump_output = argv[i] + 14;
    } else {
      fprintf(stderr, "unknown argument: %s\n", argv[i]); return 1;
    }
  }
  CUDA_CHECK(cudaFree(0));
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 64 * 1024 * 1024));
  printf("K0 dense_gemm_bf16 (Blackwell GB200, sm_100a) -- fill=%d %s\n",
         g_fill_mode, fill_mode_label(g_fill_mode));
  printf("=============================================\n");

  if (custom_m || custom_k || custom_n) {
    if (custom_m <= 0 || custom_k <= 0 || custom_n <= 0) {
      fprintf(stderr, "--shape dimensions must be positive\n"); return 1;
    }
    if (custom_m % K0_M_TILE_CLUSTER != 0 || custom_n % 128 != 0 ||
        (custom_k != 64 && custom_k != 128 && custom_k != 256 &&
         custom_k != 512 && custom_k != 2048 && custom_k != 4096)) {
      fprintf(stderr, "unsupported --shape for this kernel\n"); return 1;
    }
    if (along_m) {
      run_one<ClcRasterOrder::AlongM>(custom_m, custom_n, custom_k,
                                      gpu_verify, benchmark, dump_output);
    } else {
      run_one<ClcRasterOrder::AlongN>(custom_m, custom_n, custom_k,
                                      gpu_verify, benchmark, dump_output);
    }
    return 0;
  }
  if (dump_output) {
    fprintf(stderr, "--dump-output requires --shape=M,K,N\n"); return 1;
  }

  // V0 shape inventory (Cosmos3 256GPU iter 5000, Qwen3-VL-30B-A3B-Instruct).
  // Format: (label, M, K, N).
  //
  // UND shapes have production M=14046 which is not a multiple of
  // M_TILE_CLUSTER=256. Rounded up to 14080 (= 55*256) so the kernel
  // can bench the shape; tile-tail handler is a separate tune item.
  struct Shape { const char* label; int M; int K; int N; };
  const Shape v0_shapes[] = {
      { "und-gate", 14080, 2048,  128 },  // M padded 14046 -> 14080
      { "und-qkv",  14080, 2048, 5120 },  // M padded 14046 -> 14080
      { "und-o",    14080, 4096, 2048 },  // M padded 14046 -> 14080
      { "gen-gate", 30720, 2048,  128 },
      { "gen-qkv",  30720, 2048, 5120 },
      { "gen-o",    30720, 4096, 2048 },
  };

  printf("\n[V0 shape inventory]\n");
  for (const auto& s : v0_shapes) {
    printf("\n[%s]\n", s.label);
    printf("  AlongN:\n");
    run_verify<ClcRasterOrder::AlongN>(s.M, s.N, s.K);
    printf("  AlongM:\n");
    run_verify<ClcRasterOrder::AlongM>(s.M, s.N, s.K);
  }

  return 0;
}
#endif  // MOE_DISABLE_MAIN
