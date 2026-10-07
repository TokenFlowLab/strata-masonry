// Dense BF16 FlashAttention backward for Blackwell sm_100a.
//
// The file is intentionally self-contained: CPU reference, verification,
// preprocess, main backward (added in M2/M3), postprocess, and driver remain
// together until the monolithic kernel is correct and tuned.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda/ptx>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <vector>
#include "fmha_context_bwd_bf16_benchmark.cuh"

#include "../../../../primitives/0_tcgen05_alloc.cuh"
#include "../../../../primitives/_common.cuh"
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
#include "../../../../primitives/19_tma_load_2sm.cuh"
#include "../../../../primitives/21_tma_load_prefetch.cuh"
#include "../../../../primitives/22_tma_store.cuh"
#include "../../../../primitives/23_tma_tensormap.cuh"
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
#include "../../../../primitives/41_smem_swizzle.cuh"
#include "../../../../primitives/42_smem_desc_blackwell.cuh"
#include "../../../../primitives/45_shfl_sync.cuh"
#include "../../../../primitives/46_setmaxnreg.cuh"
#include "../../../../primitives/44_elect_sync.cuh"
#include "../../../../primitives/50_atom_global.cuh"
#include "../../../../primitives/63_cvt_f32_to_f16_bf16.cuh"
#include "../../../../primitives/67_mapa.cuh"
#include "../../../../primitives/69_griddepcontrol.cuh"
#include "../../../../primitives/70_smem_ptr.cuh"
#include "../../../../primitives/76_packed_f32x2.cuh"
#include "../../../../primitives/77_ex2_approx.cuh"
#include "../../../../composites/118_mbarrier_phase_tracking.cuh"
#include "../../../../composites/129_epi_convert.cuh"
#include "../../../../composites/112_fmha_softmax_utils.cuh"

constexpr float LOG2_E = 1.4426950408889634074f;
constexpr int M_TILE = 128;
constexpr int K_TILE = 128;
constexpr int N_WARPS = 16;
constexpr int PREPROCESS_THREADS = 256;
constexpr int POSTPROCESS_THREADS = 128;
constexpr int D128_2CTA_HEADER_BYTES = 1024;

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error_ = (call);                                                \
    if (error_ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(error_));                                 \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                          \
  } while (0)

// M6 cluster protocol gate. Each peer publishes its rank in local shared
// memory, synchronizes the 2-CTA cluster, and validates both DSMEM mappings.
// Keeping this in the end-to-end binary makes every full verifier exercise
// the exact launch attribute and peer-address protocol used by D128 2CTA.
__global__ void __cluster_dims__(2, 1, 1)
fmha_bwd_cluster_protocol_kernel(int* out) {
  __shared__ int shared_rank;
  uint32_t rank;
  asm("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rank));
  if (threadIdx.x == 0) {
    shared_rank = static_cast<int>(rank);
  }
  __syncthreads();
  barrier_cluster_arrive();
  barrier_cluster_wait();

  if (threadIdx.x == 0) {
    const uint32_t local_addr = smem_ptr_u32(&shared_rank);
    const uint32_t peer0_addr = mapa_shared_cluster_u32(local_addr, 0);
    const uint32_t peer1_addr = mapa_shared_cluster_u32(local_addr, 1);
    int peer0_value;
    int peer1_value;
    asm("ld.shared::cluster.b32 %0, [%1];\n"
        : "=r"(peer0_value) : "r"(peer0_addr));
    asm("ld.shared::cluster.b32 %0, [%1];\n"
        : "=r"(peer1_value) : "r"(peer1_addr));
    out[rank] = static_cast<int>(rank);
    out[2 + rank] = peer0_value;
    out[4 + rank] = peer1_value;
    out[6 + rank] =
        static_cast<int>(getctarank_shared_cluster_u32(peer0_addr));
    out[8 + rank] =
        static_cast<int>(getctarank_shared_cluster_u32(peer1_addr));
  }
}

bool run_cluster_protocol_gate() {
  constexpr int WORDS = 10;
  int* device_out = nullptr;
  CUDA_CHECK(cudaMalloc(&device_out, WORDS * sizeof(int)));
  CUDA_CHECK(cudaMemset(device_out, 0xff, WORDS * sizeof(int)));
  CUDA_CHECK(cudaFuncSetAttribute(
      fmha_bwd_cluster_protocol_kernel,
      cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(2, 1, 1);
  config.blockDim = dim3(32, 1, 1);
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeClusterDimension;
  attribute.val.clusterDim = {2, 1, 1};
  config.attrs = &attribute;
  config.numAttrs = 1;
  CUDA_CHECK(cudaLaunchKernelEx(&config, fmha_bwd_cluster_protocol_kernel,
                                device_out));
  CUDA_CHECK(cudaDeviceSynchronize());

  int host_out[WORDS];
  CUDA_CHECK(cudaMemcpy(host_out, device_out, sizeof(host_out),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_out));
  bool pass = true;
  for (int rank = 0; rank < 2; ++rank) {
    pass = pass && host_out[rank] == rank;
    pass = pass && host_out[2 + rank] == 0;
    pass = pass && host_out[4 + rank] == 1;
    pass = pass && host_out[6 + rank] == 0;
    pass = pass && host_out[8 + rank] == 1;
  }
  std::printf("M6 2CTA cluster launch/rank/mapa gate: %s\n",
              pass ? "PASS" : "FAIL");
  return pass;
}

template <bool USE_2CTA>
__device__ __forceinline__ void bwd_mma_f16_ss_lead(
    uint32_t lead, uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, bool enable_input_d) {
  if constexpr (!USE_2CTA) {
    tcgen05_mma_f16_ss_lead(lead, tmem_c, desc_a, desc_b, idesc,
                            enable_input_d);
  } else if (lead) {
    tcgen05_mma_f16_ss_2sm(tmem_c, desc_a, desc_b, idesc, enable_input_d,
                           0, 0, 0, 0, 0, 0, 0, 0);
  }
}

template <bool USE_2CTA>
__device__ __forceinline__ void bwd_mma_f16_ts_lead(
    uint32_t lead, uint32_t tmem_c, uint32_t tmem_a, uint64_t desc_b,
    uint32_t idesc, bool enable_input_d) {
  if constexpr (!USE_2CTA) {
    tcgen05_mma_f16_ts_1sm_lead(lead, tmem_c, tmem_a, desc_b, idesc,
                                enable_input_d);
  } else if (lead) {
    tcgen05_mma_f16_ts_2sm(tmem_c, tmem_a, desc_b, idesc, enable_input_d,
                           0, 0, 0, 0, 0, 0, 0, 0);
  }
}

template <bool USE_2CTA>
__device__ __forceinline__ void bwd_mma_commit_lead(
    uint32_t lead, uint32_t barrier_addr) {
  if constexpr (!USE_2CTA) {
    tcgen05_commit1_lead(lead, barrier_addr);
  } else if (lead) {
    tcgen05_commit_multicast<2>(barrier_addr, 0x3);
  }
}

template <bool USE_2CTA>
__device__ __forceinline__ void bwd_pipeline_consumer_wait(
    uint32_t barrier_addr, uint32_t phase) {
  if constexpr (USE_2CTA) {
    // FA4 PipelineTmaAsync / PipelineAsyncUmma consumer_wait: an initial
    // try-wait followed by the 10 ms suspend-hint retry path.
    mbarrier_wait_parity_suspend(barrier_addr, phase);
  } else {
    mbarrier_wait_parity(barrier_addr, phase);
  }
}

template <bool USE_2CTA>
__device__ __forceinline__ void bwd_pair_arrive_wait(
    uint64_t* barrier, uint32_t phase, uint32_t cta_rank, int lane) {
  if constexpr (USE_2CTA) {
    if (lane == 0) {
      const uint32_t leader_addr =
          mapa_shared_cluster_u32(smem_ptr_u32(barrier), 0);
      mbarrier_arrive_cluster_default(leader_addr);
    }
    if (cta_rank == 0) {
      bwd_pipeline_consumer_wait<USE_2CTA>(smem_ptr_u32(barrier), phase);
    }
  }
}

template <bool USE_2CTA>
__device__ __forceinline__ void bwd_pair_tma_wait_broadcast(
    uint64_t* barrier, uint32_t phase, uint32_t cta_rank, int lane) {
  if constexpr (!USE_2CTA) {
    bwd_pipeline_consumer_wait<USE_2CTA>(smem_ptr_u32(barrier), phase);
  } else {
    if (cta_rank == 0) {
      bwd_pipeline_consumer_wait<USE_2CTA>(smem_ptr_u32(barrier), phase);
      if (lane == 0) {
        const uint32_t follower_addr =
            mapa_shared_cluster_u32(smem_ptr_u32(barrier), 1);
        mbarrier_arrive_cluster_default(follower_addr);
      }
    } else {
      bwd_pipeline_consumer_wait<USE_2CTA>(smem_ptr_u32(barrier), phase);
    }
  }
}

template <int HEAD_DIM>
__global__ __launch_bounds__(PREPROCESS_THREADS)
void fmha_bwd_preprocess_kernel(const __nv_bfloat16* __restrict__ output,
                                const __nv_bfloat16* __restrict__ dout,
                                const float* __restrict__ lse,
                                float* __restrict__ d,
                                float* __restrict__ lse_log2,
                                float* __restrict__ dq_accum,
                                std::size_t rows) {
  static_assert(HEAD_DIM == 64 || HEAD_DIM == 128);
  constexpr int ROWS_PER_TILE = M_TILE;
  constexpr int BF16_PER_VECTOR = sizeof(uint4) / sizeof(__nv_bfloat16);
  constexpr int THREADS_PER_ROW = HEAD_DIM / BF16_PER_VECTOR;
  constexpr int ROWS_PER_PASS = PREPROCESS_THREADS / THREADS_PER_ROW;
  constexpr int ROW_PASSES = ROWS_PER_TILE / ROWS_PER_PASS;
  const int tid = threadIdx.x;
  const std::size_t tile_row =
      static_cast<std::size_t>(blockIdx.x) * ROWS_PER_TILE;

  // FA4 launches preprocess with programmatic stream serialization.  The
  // prologue may execute while the preceding stream kernel is still active,
  // but O, dO, and LSE cannot be consumed until its writes are visible.
  griddepcontrol_wait();

  // Exact FA4 preprocess ownership: threads 0..127 each own one LSE row.
  // Load it before the O/dO work, then keep it live until the final LSElog2
  // store after dQaccum has been cleared.
  const std::size_t lse_row = tile_row + tid;
  float lse_value = 0.0f;
  if (tid < ROWS_PER_TILE && lse_row < rows) {
    lse_value = lse[lse_row];
  }

  // A 128-bit O/dO copy is owned by one thread.  THREADS_PER_ROW is 8 for
  // D64 and 16 for D128, so every row reduction stays inside a power-of-two
  // warp subdivision, matching FA4's tiled-copy reduction profile.
  const int row_slot = tid / THREADS_PER_ROW;
  const int lane_in_row = tid % THREADS_PER_ROW;
  uint4 o_regs[ROW_PASSES];
  uint4 do_regs[ROW_PASSES];

  // FA4's tiled-copy composite loads the complete per-thread O and dO
  // fragments before starting the reduction.  Keeping all row passes live
  // exposes the same global-memory and conversion ILP as the CuTe kernel.
  #pragma unroll
  for (int pass = 0; pass < ROW_PASSES; ++pass) {
    const int row_local = pass * ROWS_PER_PASS + row_slot;
    const std::size_t row = tile_row + row_local;
    if (row < rows) {
      const std::size_t vector_offset =
          row * HEAD_DIM + lane_in_row * BF16_PER_VECTOR;
      o_regs[pass] =
          *reinterpret_cast<const uint4*>(output + vector_offset);
      do_regs[pass] =
          *reinterpret_cast<const uint4*>(dout + vector_offset);
    } else {
      o_regs[pass] = make_uint4(0, 0, 0, 0);
      do_regs[pass] = make_uint4(0, 0, 0, 0);
    }
  }

  // Match FA4's PDL boundary: all O/dO fragments are resident before the
  // preprocess grid permits its dependent launch to proceed.
  griddepcontrol_launch_dependents();

  float products[ROW_PASSES];
  #pragma unroll
  for (int pass = 0; pass < ROW_PASSES; ++pass) {
    const __nv_bfloat162* o2 =
        reinterpret_cast<const __nv_bfloat162*>(&o_regs[pass]);
    const __nv_bfloat162* do2 =
        reinterpret_cast<const __nv_bfloat162*>(&do_regs[pass]);
    float product = 0.0f;
    #pragma unroll
    for (int pair = 0; pair < BF16_PER_VECTOR / 2; ++pair) {
      const float2 of = __bfloat1622float2(o2[pair]);
      const float2 df = __bfloat1622float2(do2[pair]);
      const float2 prod = fmul2(of, df);
      product += prod.x + prod.y;
    }
    products[pass] = product;
  }

  #pragma unroll
  for (int delta = THREADS_PER_ROW / 2; delta >= 1; delta >>= 1) {
    #pragma unroll
    for (int pass = 0; pass < ROW_PASSES; ++pass) {
      const uint32_t other =
          shfl_sync_bfly(__float_as_uint(products[pass]), delta);
      products[pass] += __uint_as_float(other);
    }
  }

  #pragma unroll
  for (int pass = 0; pass < ROW_PASSES; ++pass) {
    const int row_local = pass * ROWS_PER_PASS + row_slot;
    const std::size_t row = tile_row + row_local;
    if (lane_in_row == 0 && row < rows) {
      d[row] = products[pass];
    }
  }

  // FA4 clears one complete 128-row dQaccum tile with 128-bit stores after
  // the O*dO reduction, using tiled_copy_1d ownership across all 256 threads.
  constexpr int ZERO_VECTORS = ROWS_PER_TILE * HEAD_DIM / 4;
  float* zero_dst = dq_accum + tile_row * HEAD_DIM;
  float4* zero_dst4 = reinterpret_cast<float4*>(zero_dst);
  const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  #pragma unroll
  for (int copy = 0; copy < ZERO_VECTORS / PREPROCESS_THREADS; ++copy) {
    const int vector_idx = tid + copy * PREPROCESS_THREADS;
    const std::size_t element =
        tile_row * HEAD_DIM + static_cast<std::size_t>(vector_idx) * 4;
    if (element < rows * HEAD_DIM) {
      zero_dst4[vector_idx] = zero;
    }
  }

  if (tid < ROWS_PER_TILE && lse_row < rows) {
    lse_log2[lse_row] =
        lse_value == -std::numeric_limits<float>::infinity()
            ? 0.0f
            : lse_value * LOG2_E;
  }
}

template <int HEAD_DIM, bool FA4_FLAT_DQ = false>
__global__ __launch_bounds__(POSTPROCESS_THREADS)
void fmha_bwd_postprocess_kernel(const float* __restrict__ dq_accum,
                                 __nv_bfloat16* __restrict__ dq,
                                 float scale, std::size_t rows) {
  static_assert(HEAD_DIM == 64 || HEAD_DIM == 128);
  constexpr int ROWS_PER_TILE = M_TILE;
  constexpr int FLOATS_PER_G2S = sizeof(float4) / sizeof(float);
  constexpr int BF16_PER_STORE = sizeof(uint4) / sizeof(__nv_bfloat16);
  constexpr int TILE_FLOATS = ROWS_PER_TILE * HEAD_DIM;
  constexpr int G2S_VECTORS = TILE_FLOATS / FLOATS_PER_G2S;
  constexpr int OUTPUT_VECTORS = TILE_FLOATS / BF16_PER_STORE;
  extern __shared__ __align__(1024) float s_dqaccum[];

  const int tid = threadIdx.x;
  const std::size_t tile_row =
      static_cast<std::size_t>(blockIdx.x) * ROWS_PER_TILE;
  const std::size_t tile_base = tile_row * HEAD_DIM;
  const std::size_t total = rows * HEAD_DIM;

  // FA4 postprocess step 1: 128-bit cp.async G->S for the complete dQ tile.
  #pragma unroll
  for (int copy = 0; copy < G2S_VECTORS / POSTPROCESS_THREADS; ++copy) {
    const int vector_idx = tid + copy * POSTPROCESS_THREADS;
    const std::size_t element =
        tile_base + static_cast<std::size_t>(vector_idx) * FLOATS_PER_G2S;
    cp_async_cg_16_masked(
        smem_ptr_u32(s_dqaccum + vector_idx * FLOATS_PER_G2S),
        dq_accum + element, element < total ? sizeof(float4) : 0);
  }
  cp_async_commit_group();
  cp_async_wait_group<0>();
  __syncthreads();

  if constexpr (HEAD_DIM == 128 && FA4_FLAT_DQ) {
    // Exact FA4 SM100 1CTA postprocess ownership.  The reducer publishes four
    // 128x32 column chunks, with tiled_copy_1d placing thread t's float4 at
    // chunk*4096 + vector*512 + t*4.  One postprocess thread reconstructs one
    // complete logical row in registers.
    uint32_t row_bf16[HEAD_DIM / 2];
    #pragma unroll
    for (int chunk = 0; chunk < HEAD_DIM / 32; ++chunk) {
      #pragma unroll
      for (int vector = 0; vector < 8; ++vector) {
        constexpr int CHUNK_FLOATS = M_TILE * 32;
        constexpr int VECTOR_STRIDE_FLOATS = POSTPROCESS_THREADS * 4;
        const int source = chunk * CHUNK_FLOATS +
                           vector * VECTOR_STRIDE_FLOATS + tid * 4;
        const float4 value =
            *reinterpret_cast<const float4*>(s_dqaccum + source);
        const int packed = chunk * 16 + vector * 2;
        epi_pack4<EpiOutDtype::BF16>(
            value.x * scale, value.y * scale,
            value.z * scale, value.w * scale,
            row_bf16[packed], row_bf16[packed + 1]);
      }
    }

    // FA4 reuses the FP32 staging allocation as a BF16 epilogue tile.  The
    // 128-bit R2S copy follows the standard B128 row swizzle, then the final
    // tiled S2R/G store is coalesced across 16 threads per output row.
    __syncthreads();
    __nv_bfloat16* s_dq =
        reinterpret_cast<__nv_bfloat16*>(s_dqaccum);
    const uint4* row4 = reinterpret_cast<const uint4*>(row_bf16);
    #pragma unroll
    for (int vector = 0; vector < HEAD_DIM / BF16_PER_STORE; ++vector) {
      constexpr int HALF_BF16 = 64;
      constexpr int VECTORS_PER_HALF = HALF_BF16 / BF16_PER_STORE;
      const int half = vector / VECTORS_PER_HALF;
      const int vector_in_half = vector % VECTORS_PER_HALF;
      const uint32_t col_bytes = swizzle_col_bytes<128>(
          static_cast<uint32_t>(tid),
          static_cast<uint32_t>(vector_in_half * sizeof(uint4)));
      const int physical_col = half * HALF_BF16 +
                               static_cast<int>(col_bytes / sizeof(__nv_bfloat16));
      *reinterpret_cast<uint4*>(s_dq + tid * HEAD_DIM + physical_col) =
          row4[vector];
    }

    __syncthreads();
    constexpr int THREADS_PER_OUTPUT_ROW =
        HEAD_DIM / BF16_PER_STORE;
    constexpr int OUTPUT_ROWS_PER_PASS =
        POSTPROCESS_THREADS / THREADS_PER_OUTPUT_ROW;
    const int row_slot = tid / THREADS_PER_OUTPUT_ROW;
    const int col = (tid % THREADS_PER_OUTPUT_ROW) * BF16_PER_STORE;
    #pragma unroll
    for (int pass = 0; pass < M_TILE / OUTPUT_ROWS_PER_PASS; ++pass) {
      const int row = pass * OUTPUT_ROWS_PER_PASS + row_slot;
      constexpr int HALF_BF16 = 64;
      const int half = col / HALF_BF16;
      const int col_in_half = col % HALF_BF16;
      const uint32_t col_bytes = swizzle_col_bytes<128>(
          static_cast<uint32_t>(row),
          static_cast<uint32_t>(col_in_half * sizeof(__nv_bfloat16)));
      const int physical_col = half * HALF_BF16 +
                               static_cast<int>(col_bytes / sizeof(__nv_bfloat16));
      const uint4 value = *reinterpret_cast<const uint4*>(
          s_dq + row * HEAD_DIM + physical_col);
      const std::size_t global_row = tile_row + row;
      if (global_row < rows) {
        *reinterpret_cast<uint4*>(dq + global_row * HEAD_DIM + col) = value;
      }
    }
  } else {
    // D64 and the generic-tail D128 path retain their row-major dQaccum
    // contract.  Their own FA4 reducer/postprocess layouts are independent of
    // the static D128 1CTA path above.
    #pragma unroll
    for (int copy = 0; copy < OUTPUT_VECTORS / POSTPROCESS_THREADS; ++copy) {
      const int vector_idx = tid + copy * POSTPROCESS_THREADS;
      const int float_idx = vector_idx * BF16_PER_STORE;
      const float4 lo4 =
          *reinterpret_cast<const float4*>(s_dqaccum + float_idx);
      const float4 hi4 =
          *reinterpret_cast<const float4*>(s_dqaccum + float_idx + 4);
      const float2 x0 = fmul2(make_float2(lo4.x, lo4.y), f32x2_splat(scale));
      const float2 x1 = fmul2(make_float2(lo4.z, lo4.w), f32x2_splat(scale));
      const float2 x2 = fmul2(make_float2(hi4.x, hi4.y), f32x2_splat(scale));
      const float2 x3 = fmul2(make_float2(hi4.z, hi4.w), f32x2_splat(scale));
      uint4 out;
      out.x = cvt_f32x2_to_bf16x2(x0.x, x0.y);
      out.y = cvt_f32x2_to_bf16x2(x1.x, x1.y);
      out.z = cvt_f32x2_to_bf16x2(x2.x, x2.y);
      out.w = cvt_f32x2_to_bf16x2(x3.x, x3.y);
      const std::size_t element =
          tile_base + static_cast<std::size_t>(float_idx);
      if (element < total) {
        *reinterpret_cast<uint4*>(dq + element) = out;
      }
    }
  }
}

// FA4's cta_group::2 dQ reducer writes the two CTA-owned 128x64 TMEM
// fragments as eight packed 128x8 stages per CTA.  Its postprocess therefore
// uses a different composite from the ordinary row-major path above:
//
//   packed G -> staged S -> TMEM-view R -> swizzled BF16 S -> coalesced G.
//
// Keep the same 128-thread ownership and named-barrier contract as the FA4
// use_2cta_instrs branch.  Each thread owns one output query row after the
// packed-to-TMEM-view remap.
__global__ __launch_bounds__(POSTPROCESS_THREADS)
void fmha_bwd_postprocess_2cta_kernel(
    const float* __restrict__ dq_accum,
    __nv_bfloat16* __restrict__ dq, float scale, int batch_heads,
    int seqlen) {
  constexpr int DQ_REDUCE_NCOL = 32;
  constexpr int REDUCE_STAGES = 128 / DQ_REDUCE_NCOL;
  constexpr int STAGE_FLOATS = M_TILE * DQ_REDUCE_NCOL;
  constexpr int ROW_GROUPS = 2;
  constexpr int STAGE_GROUPS = REDUCE_STAGES / ROW_GROUPS;
  static_assert(REDUCE_STAGES == 4 && STAGE_GROUPS == 2);

  extern __shared__ __align__(128) unsigned char shared_raw[];
  float* const s_dqaccum = reinterpret_cast<float*>(shared_raw);
  __nv_bfloat16* const s_dq =
      reinterpret_cast<__nv_bfloat16*>(shared_raw);

  const int tid = threadIdx.x;
  const int m_block = blockIdx.x;
  const int bh = blockIdx.y;
  const int m_start = m_block * M_TILE;
  if (m_start >= seqlen || bh >= batch_heads) {
    return;
  }

  const std::size_t tile_base =
      (static_cast<std::size_t>(bh) * seqlen + m_start) * 128;
  const float* const g_dqaccum = dq_accum + tile_base;

  // The rank half is the shared-buffer selector in FA4's S2R remap.  A
  // thread then gathers one 128-element logical dQ row into the register
  // fragment consumed by the BF16 R2S composite.
  const int rank = tid >> 6;
  const int query_local = tid & 63;
  uint32_t r_dq[128 / 2];

  #pragma unroll
  for (int stage_group = 0; stage_group < STAGE_GROUPS; ++stage_group) {
    // G -> S: stage_group and stage_group + 2 are the corresponding packed
    // 32-column stages from CTA ranks 0 and 1.  The FA4 copy atom is 128b.
    #pragma unroll
    for (int row_group = 0; row_group < ROW_GROUPS; ++row_group) {
      const int stage_idx = stage_group + row_group * STAGE_GROUPS;
      const float4* const g_vec = reinterpret_cast<const float4*>(
          g_dqaccum + stage_idx * STAGE_FLOATS);
      float4* const s_vec = reinterpret_cast<float4*>(
          s_dqaccum + row_group * STAGE_FLOATS);
      #pragma unroll
      for (int copy = 0; copy < STAGE_FLOATS / (POSTPROCESS_THREADS * 4);
           ++copy) {
        const int vector_idx = tid + copy * POSTPROCESS_THREADS;
        s_vec[vector_idx] = g_vec[vector_idx];
      }
    }
    bar_sync<6>(POSTPROCESS_THREADS);

    // S -> R: undo the eight-stage reducer packing.  The two iterations are
    // FA4's row_groups; barrier 7 protects both aliases before the next G2S.
    #pragma unroll
    for (int dim_half = 0; dim_half < ROW_GROUPS; ++dim_half) {
      #pragma unroll
      for (int substage = 0; substage < REDUCE_STAGES; ++substage) {
        const int reduce_stage = stage_group * REDUCE_STAGES + substage;
        #pragma unroll
        for (int j = 0; j < 8; j += 2) {
          const int tmem_thread = query_local + dim_half * 64;
          const int packed_lo =
              substage * 1024 + tmem_thread * 4 +
              (j < 4 ? j : 512 + j - 4);
          const int packed_hi = packed_lo + 1;
          const float lo =
              s_dqaccum[rank * STAGE_FLOATS + packed_lo] * scale;
          const float hi =
              s_dqaccum[rank * STAGE_FLOATS + packed_hi] * scale;
          const int dim = dim_half * 64 + reduce_stage * 8 + j;
          r_dq[dim / 2] = cvt_f32x2_to_bf16x2(lo, hi);
        }
      }
      bar_sync<7>(POSTPROCESS_THREADS);
    }
  }

  // R -> S: FA4's row-major BF16 epilogue layout is tiled as two B128
  // swizzled 128x64 atoms.
  #pragma unroll
  for (int pair = 0; pair < 128 / 2; ++pair) {
    const int dim = pair * 2;
    const int atom = dim >> 6;
    const int atom_dim = dim & 63;
    const int logical = tid * 64 + atom_dim;
    const int physical =
        atom * (M_TILE * 64) +
        static_cast<int>(smem_swizzle_b128(
                             logical * sizeof(__nv_bfloat16)) /
                         sizeof(__nv_bfloat16));
    reinterpret_cast<uint32_t*>(s_dq + physical)[0] = r_dq[pair];
  }
  bar_sync<8>(POSTPROCESS_THREADS);

  // S -> G: 128-bit row-major vectors, predicated only on the M tail.
  constexpr int BF16_PER_VECTOR = sizeof(uint4) / sizeof(__nv_bfloat16);
  constexpr int VECTORS_PER_ROW = 128 / BF16_PER_VECTOR;
  #pragma unroll
  for (int copy = 0;
       copy < (M_TILE * VECTORS_PER_ROW) / POSTPROCESS_THREADS; ++copy) {
    const int vector_idx = tid + copy * POSTPROCESS_THREADS;
    const int row = vector_idx / VECTORS_PER_ROW;
    const int dim = (vector_idx % VECTORS_PER_ROW) * BF16_PER_VECTOR;
    const int atom = dim >> 6;
    const int atom_dim = dim & 63;
    const int logical = row * 64 + atom_dim;
    const int physical =
        atom * (M_TILE * 64) +
        static_cast<int>(smem_swizzle_b128(
                             logical * sizeof(__nv_bfloat16)) /
                         sizeof(__nv_bfloat16));
    if (m_start + row < seqlen) {
      const uint4 value =
          reinterpret_cast<const uint4*>(s_dq + physical)[0];
      reinterpret_cast<uint4*>(dq + tile_base + row * 128 + dim)[0] = value;
    }
  }
}

// Main backward kernel using FA4's ownership and warp map. D64 runs all five
// products on tcgen05 with packed-TMEM P/dS, Q=2/dO=2/dKV=2 staging, and the
// dense prologue/main/tail lookahead order. D128 uses the same five tcgen05
// products with the dense 1-CTA overlapping TMEM contract.
//
// All role code stays directly in this one __global__ function:
//   warps 0-3   dQ reduction
//   warps 4-11  P/dS and dK/dV accumulation/epilogue
//   warp 12     tcgen05 MMA producer and TMEM owner
//   warp 13     TMA load producer
//   warp 14     relay (unused in 1CTA)
//   warp 15     scheduler/empty
template <int HEAD_DIM, bool IS_CAUSAL, bool USE_2CTA = false,
          int STATIC_SEQLEN = 0>
__global__ void __maxnreg__(128)
fmha_bwd_main_kernel(const __grid_constant__ CUtensorMap tmap_q,
                          const __grid_constant__ CUtensorMap tmap_k,
                          const __grid_constant__ CUtensorMap tmap_kt,
                          const __grid_constant__ CUtensorMap tmap_v,
                          const __grid_constant__ CUtensorMap tmap_dout,
                          const __grid_constant__ CUtensorMap tmap_dk,
                          const __grid_constant__ CUtensorMap tmap_dv,
                          const __grid_constant__ CUtensorMap tmap_q_half,
                          const __grid_constant__ CUtensorMap tmap_dout_half,
                          const CUtensorMap* __restrict__ tmap_dq,
                          const __nv_bfloat16* __restrict__ q,
                          const __nv_bfloat16* __restrict__ k,
                          const __nv_bfloat16* __restrict__ v,
                          const __nv_bfloat16* __restrict__ dout,
                          const float* __restrict__ lse_log2,
                          const float* __restrict__ d,
                          float* __restrict__ dq_accum,
                          __nv_bfloat16* __restrict__ dk,
                          __nv_bfloat16* __restrict__ dv,
                          int batch, int heads, int seqlen) {
  static_assert(HEAD_DIM == 64 || HEAD_DIM == 128);
  static_assert(!USE_2CTA || HEAD_DIM == 128,
                "D64 supports only the accepted 1CTA path");
  // Match the forward-kernel convention: USE_2CTA controls algorithmic
  // branches; CTA_GROUP is derived only for tcgen05 primitives/composites.
  constexpr int CTA_GROUP = USE_2CTA ? 2 : 1;
  constexpr float scale = HEAD_DIM == 64 ? 0.125f : 0.08838834764831845f;
  const int seq = STATIC_SEQLEN == 0 ? seqlen : STATIC_SEQLEN;
  extern __shared__ __align__(128) unsigned char shared_raw[];

  __nv_bfloat16* s_k = reinterpret_cast<__nv_bfloat16*>(shared_raw);
  __nv_bfloat16* s_v = s_k + K_TILE * HEAD_DIM;
  __nv_bfloat16* s_q = s_v + K_TILE * HEAD_DIM;
  __nv_bfloat16* s_dout = s_q + HEAD_DIM;
  float* s_dk = reinterpret_cast<float*>(s_dout + HEAD_DIM);
  float* s_dv = s_dk + K_TILE * HEAD_DIM;
  float* s_score = s_dv + K_TILE * HEAD_DIM;
  float* s_dp = s_score + K_TILE;
  float* s_p = s_dp + K_TILE;
  float* s_ds = s_p + K_TILE;
  float* s_lse_log2 = s_ds + K_TILE;
  float* s_d = s_lse_log2 + 1;

  const int tid = threadIdx.x;
  const int warp_raw = tid >> 5;
  const int lane = tid & 31;

  uint32_t cta_rank_raw = 0;
  if constexpr (USE_2CTA) {
    asm("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(cta_rank_raw));
  }
  // FA4 treats the cluster rank as warp-uniform. Keep the forward kernel's
  // lane-zero broadcast convention so rank-derived stage/address arithmetic
  // can move onto the uniform datapath.
  const uint32_t cta_rank = USE_2CTA
      ? __shfl_sync(0xffffffffu, cta_rank_raw, 0)
      : 0u;
  // FA4 prefetches every descriptor used by the load/epilogue paths from one
  // elected load-warp lane before shared-barrier initialization. For 2CTA,
  // Q/dO use half-tile atoms while Qt/dOt use the full transposed atoms.
  if (warp_raw == 13 && lane == 0) {
    if constexpr (USE_2CTA) {
      prefetch_tensormap(&tmap_q_half);     // Q
      prefetch_tensormap(&tmap_q);          // Qt
      prefetch_tensormap(&tmap_k);          // K
      prefetch_tensormap(&tmap_kt);         // Kt
      prefetch_tensormap(&tmap_v);          // V
      prefetch_tensormap(&tmap_dout_half);  // dOt
      prefetch_tensormap(&tmap_dout);       // dO
      prefetch_tensormap(&tmap_dv);         // dV
      prefetch_tensormap(&tmap_dk);         // dK
    } else {
      prefetch_tensormap(&tmap_q);
      prefetch_tensormap(&tmap_k);
      prefetch_tensormap(&tmap_v);
      prefetch_tensormap(&tmap_dout);
      prefetch_tensormap(&tmap_dv);
      prefetch_tensormap(&tmap_dk);
    }
  }
  // CUDA spelling of FA4's make_warp_uniform(warp_idx): create the role ID
  // after descriptor prefetch so it is live only across its role branch.
  const int warp = HEAD_DIM == 128
      ? __shfl_sync(0xffffffffu, warp_raw, 0)
      : warp_raw;
  const int n_tiles = (seq + K_TILE - 1) / K_TILE;
  const int linear = blockIdx.x;
  const int n_block = linear % n_tiles;
  const int head_linear = linear / n_tiles;

  const int key_start = n_block * K_TILE;
  const std::size_t bh_base =
      static_cast<std::size_t>(head_linear) * seq * HEAD_DIM;
  const std::size_t row_base =
      static_cast<std::size_t>(head_linear) * seq;
  constexpr int K_TILE_ELEMENTS = K_TILE * HEAD_DIM;

  // D64 tcgen05 bring-up. All five matrix products use tensor cores. Each
  // role remains inline in this one kernel.
  if constexpr (HEAD_DIM == 64) {
    __nv_bfloat16* t_k = reinterpret_cast<__nv_bfloat16*>(shared_raw);
    __nv_bfloat16* t_v = t_k + K_TILE * HEAD_DIM;
    __nv_bfloat16* t_q = t_v + K_TILE * HEAD_DIM;
    __nv_bfloat16* t_dout = t_q + 2 * M_TILE * HEAD_DIM;
    __nv_bfloat16* t_k_trans = t_dout + 2 * M_TILE * HEAD_DIM;
    __nv_bfloat16* t_dq_smem_storage = t_k_trans + K_TILE * HEAD_DIM;
    __nv_bfloat16* t_ds = t_dq_smem_storage + M_TILE * K_TILE;
    float* t_lse_log2 =
        reinterpret_cast<float*>(t_ds + M_TILE * K_TILE);
    float* t_d = t_lse_log2 + (USE_2CTA ? M_TILE : 2 * M_TILE);
    uint64_t* mma_full = reinterpret_cast<uint64_t*>(t_d + 2 * M_TILE);
    uint64_t* q_full = mma_full + 3;
    uint64_t* q_empty = q_full + 2;
    uint64_t* kv_full = q_empty + 2;
    uint64_t* load_full = kv_full + 1;
    uint64_t* load_empty = load_full + 2;
    uint64_t* dkv_full = load_empty + 2;
    uint64_t* dkv_empty = dkv_full + 2;
    uint64_t* role_bar = dkv_empty + 2;
    uint32_t* tmem_slot = reinterpret_cast<uint32_t*>(role_bar + 4);

    if (tid == 0) {
      mbarrier_init(smem_ptr_u32(&mma_full[0]), 1);
      mbarrier_init(smem_ptr_u32(&mma_full[1]), 1);
      mbarrier_init(smem_ptr_u32(&mma_full[2]), 1);
      mbarrier_init(smem_ptr_u32(&q_full[0]), 1);
      mbarrier_init(smem_ptr_u32(&q_full[1]), 1);
      mbarrier_init(smem_ptr_u32(&q_empty[0]), 1);
      mbarrier_init(smem_ptr_u32(&q_empty[1]), 1);
      mbarrier_init(smem_ptr_u32(kv_full), 1);
      mbarrier_init(smem_ptr_u32(&load_full[0]), 1);
      mbarrier_init(smem_ptr_u32(&load_full[1]), 1);
      mbarrier_init(smem_ptr_u32(&load_empty[0]), 1);
      mbarrier_init(smem_ptr_u32(&load_empty[1]), 1);
      mbarrier_init(smem_ptr_u32(&dkv_full[0]), 1);
      mbarrier_init(smem_ptr_u32(&dkv_full[1]), 1);
      mbarrier_init(smem_ptr_u32(&dkv_empty[0]), 8);
      mbarrier_init(smem_ptr_u32(&dkv_empty[1]), 8);
      mbarrier_init(smem_ptr_u32(&role_bar[0]), 8);
      mbarrier_init(smem_ptr_u32(&role_bar[1]), 8);
      mbarrier_init(smem_ptr_u32(&role_bar[2]), 4);
      mbarrier_init(smem_ptr_u32(&role_bar[3]), 1);
    }
    fence_mbarrier_init_release_cluster();
    __syncthreads();

    // MMA WARP (12) owns the 512-column TMEM allocation:
    // S[0:128], dP[128:256], dV[256:320], dK[320:384],
    // dQ[384:448].
    if (warp == 12) {
      tcgen05_alloc<1>(smem_ptr_u32(tmem_slot), 512);
      tcgen05_relinquish_alloc_permit<1>();
    }
    __syncthreads();
    const uint32_t tmem_base = *tmem_slot;
    uint32_t role_phase = 0;

    // LOAD WARP (13): one elected lane issues CTA-local K/V TMA loads into
    // the B128 row-major operands. The load team then forms K^T from that
    // TMA tile because the current TMA tile operation does not transpose.
    if (warp == 13 && lane == 0) {
      constexpr uint32_t K_AND_V_BYTES =
          2 * K_TILE * HEAD_DIM * sizeof(__nv_bfloat16);
      mbarrier_arrive_expect_tx(smem_ptr_u32(kv_full), K_AND_V_BYTES);
      tma_load_3d(smem_ptr_u32(t_k), &tmap_k,
                  smem_ptr_u32(kv_full), 0, key_start, head_linear);
      tma_load_3d(smem_ptr_u32(t_v), &tmap_v,
                  smem_ptr_u32(kv_full), 0, key_start, head_linear);
    }
    mbarrier_wait_parity(smem_ptr_u32(kv_full), 0);
    {
      for (int pair = tid; pair < K_TILE_ELEMENTS / 2;
           pair += N_WARPS * 32) {
        const int key_local = pair / (HEAD_DIM / 2);
        const int dim_pair = pair - key_local * (HEAD_DIM / 2);
        const int dim = 2 * dim_pair;
        const int source_logical = key_local * HEAD_DIM + dim;
        const int source_physical =
            static_cast<int>(smem_swizzle_b128(
                                 source_logical * sizeof(__nv_bfloat16)) /
                             sizeof(__nv_bfloat16));
        const int trans_subtile = key_local / 64;
        const int trans_col = key_local & 63;
        const int trans_logical0 = dim * 64 + trans_col;
        const int trans_logical1 = (dim + 1) * 64 + trans_col;
        const int trans_physical0 =
            trans_subtile * (HEAD_DIM * 64) +
            static_cast<int>(smem_swizzle_b128(
                                 trans_logical0 * sizeof(__nv_bfloat16)) /
                             sizeof(__nv_bfloat16));
        const int trans_physical1 =
            trans_subtile * (HEAD_DIM * 64) +
            static_cast<int>(smem_swizzle_b128(
                                 trans_logical1 * sizeof(__nv_bfloat16)) /
                             sizeof(__nv_bfloat16));
        const __nv_bfloat162 k_pair =
            reinterpret_cast<const __nv_bfloat162*>(
                t_k + source_physical)[0];
        t_k_trans[trans_physical0] = k_pair.x;
        t_k_trans[trans_physical1] = k_pair.y;
      }
    }
    __syncthreads();

    uint32_t mma_phase = 0;
    const int m_begin = IS_CAUSAL ? key_start : 0;

    // LOAD WARP (13): Q, dO, and D/LSE use the same two-stage index. Their
    // independent full/empty rings allow the next query tile to land early.
    if (warp == 13) {
      int load_iter = 0;
      for (int load_m = m_begin; load_m < seq;
           load_m += M_TILE, ++load_iter) {
        const int q_stage = load_iter & 1;
        const uint32_t q_empty_phase =
            1u ^ static_cast<uint32_t>((load_iter / 2) & 1);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&q_empty[q_stage]),
                                     q_empty_phase);
        if (lane == 0) {
          constexpr uint32_t Q_BYTES =
              M_TILE * HEAD_DIM * sizeof(__nv_bfloat16);
          mbarrier_arrive_expect_tx(smem_ptr_u32(&q_full[q_stage]), Q_BYTES);
          tma_load_3d(
              smem_ptr_u32(t_q + q_stage * M_TILE * HEAD_DIM), &tmap_q,
              smem_ptr_u32(&q_full[q_stage]), 0, load_m, head_linear);
        }

        const uint32_t load_empty_phase =
            1u ^ static_cast<uint32_t>((load_iter / 2) & 1);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&load_empty[q_stage]),
                                     load_empty_phase);
        float* t_lse_load = t_lse_log2 + q_stage * M_TILE;
        float* t_d_load = t_d + q_stage * M_TILE;
        for (int query_local = lane; query_local < M_TILE;
             query_local += 32) {
          const int query = load_m + query_local;
          if (query < seq) {
            t_lse_load[query_local] = lse_log2[row_base + query];
            t_d_load[query_local] = d[row_base + query];
          } else {
            t_lse_load[query_local] = 0.0f;
            t_d_load[query_local] = 0.0f;
          }
        }
        __syncwarp();
        if (lane == 0) {
          constexpr uint32_t DOUT_BYTES =
              M_TILE * HEAD_DIM * sizeof(__nv_bfloat16);
          mbarrier_arrive_expect_tx(smem_ptr_u32(&load_full[q_stage]),
                                    DOUT_BYTES);
          tma_load_3d(
              smem_ptr_u32(t_dout + q_stage * M_TILE * HEAD_DIM),
              &tmap_dout, smem_ptr_u32(&load_full[q_stage]), 0, load_m,
              head_linear);
        }
      }
    }

    if (warp != 13) {
    int consumer_iter = 0;
    for (int m_start = m_begin; m_start < seq;
         m_start += M_TILE, ++consumer_iter) {
      const int q_stage = ((m_start - m_begin) / M_TILE) & 1;
      const bool is_last_m = m_start + M_TILE >= seq;
      const bool has_next_m = !is_last_m;
      __nv_bfloat16* t_q_stage =
          t_q + q_stage * M_TILE * HEAD_DIM;
      if (warp == 12 && consumer_iter == 0) {
        const uint32_t q_full_phase =
            static_cast<uint32_t>((consumer_iter / 2) & 1);
        const uint32_t load_full_phase =
            static_cast<uint32_t>((consumer_iter / 2) & 1);
        mbarrier_wait_parity(smem_ptr_u32(&q_full[q_stage]), q_full_phase);
        mbarrier_wait_parity(smem_ptr_u32(&load_full[q_stage]),
                             load_full_phase);
      }

      // MMA WARP (12): seed S/dP for the prologue. Later iterations are
      // produced ahead at the tail of the preceding iteration.
      if (warp == 12 && consumer_iter == 0) {
        const uint32_t lead = elect_one_sync();
        if (lead) {
          tcgen05_fence_after_thread_sync();
        }
        constexpr uint32_t DESC_SBO = 1024;
        constexpr uint32_t DESC_LBO = 16;
        const uint32_t idesc =
            make_idesc_bf16_f32(M_TILE, K_TILE, false, false);
        const uint64_t desc_q = build_smem_desc_blackwell(
            smem_ptr_u32(t_q_stage), DESC_SBO, DESC_LBO,
            SmemSwizzleBlackwell::B128);
        const uint64_t desc_k = build_smem_desc_blackwell(
            smem_ptr_u32(t_k), DESC_SBO, DESC_LBO,
            SmemSwizzleBlackwell::B128);
        const uint64_t desc_do = build_smem_desc_blackwell(
            smem_ptr_u32(t_dout + q_stage * M_TILE * HEAD_DIM), DESC_SBO,
            DESC_LBO,
            SmemSwizzleBlackwell::B128);
        const uint64_t desc_v = build_smem_desc_blackwell(
            smem_ptr_u32(t_v), DESC_SBO, DESC_LBO,
            SmemSwizzleBlackwell::B128);

        #pragma unroll
        for (int ki = 0; ki < HEAD_DIM / 16; ++ki) {
          tcgen05_mma_f16_ss_lead(lead, tmem_base, desc_k + 2 * ki,
                                  desc_q + 2 * ki, idesc, ki != 0);
        }
        tcgen05_commit1_lead(lead, smem_ptr_u32(&mma_full[0]));

        #pragma unroll
        for (int ki = 0; ki < HEAD_DIM / 16; ++ki) {
          tcgen05_mma_f16_ss_lead(lead, tmem_base + K_TILE,
                                  desc_v + 2 * ki, desc_do + 2 * ki, idesc,
                                  ki != 0);
        }
        tcgen05_commit1_lead(lead, smem_ptr_u32(&mma_full[1]));
      }

      // COMPUTE WARPS (4-11): follow FA4's transposed score orientation.
      // Lanes own key rows and FA4's two compute warpgroups stripe the four
      // 32-query chunks. P^T/dS^T stay in that orientation; dS is also
      // published in query-major form for dQ.
      if (warp >= 4 && warp <= 11) {
        mbarrier_wait_parity(smem_ptr_u32(&mma_full[0]), mma_phase);
        const uint32_t metadata_phase =
            static_cast<uint32_t>((consumer_iter / 2) & 1);
        mbarrier_wait_parity(smem_ptr_u32(&load_full[q_stage]),
                             metadata_phase);
        const int metadata_stage = consumer_iter & 1;
        const float* t_lse_consumer =
            t_lse_log2 + metadata_stage * M_TILE;
        const float* t_d_consumer = t_d + metadata_stage * M_TILE;
        const int row_group = (warp - 4) & 3;
        const int key_local = row_group * 32 + lane;
        const int key = key_start + key_local;
        const uint32_t row_addr =
            static_cast<uint32_t>(row_group * 32) << 16;
        const int col_begin = warp < 8 ? 32 : 0;

        // Keep FP32 P live across the dP wait. Publishing all of packed P
        // first lets the MMA warp start dV and S(next) while dS is formed.
        float p_values[64];
        for (int chunk = 0; chunk < 2; ++chunk) {
          const int col = col_begin + chunk * 64;
          uint32_t score_regs[32];
          uint32_t p_packed[16];
          tcgen05_ld_32x32b_x32(tmem_base + row_addr + col, score_regs);
          tcgen05_wait_ld();
          float* scores = reinterpret_cast<float*>(score_regs);
          #pragma unroll
          for (int j = 0; j < 32; j += 2) {
            const int query_local0 = col + j;
            const int query_local1 = query_local0 + 1;
            const int query0 = m_start + query_local0;
            const int query1 = query0 + 1;
            const bool valid0 =
                key < seq && query0 < seq &&
                (!IS_CAUSAL || key <= query0);
            const bool valid1 =
                key < seq && query1 < seq &&
                (!IS_CAUSAL || key <= query1);
            const float p0 =
                valid0 ? ex2_approx_f32(scores[j] * scale * LOG2_E -
                                        t_lse_consumer[query_local0])
                       : 0.0f;
            const float p1 =
                valid1 ? ex2_approx_f32(scores[j + 1] * scale * LOG2_E -
                                        t_lse_consumer[query_local1])
                       : 0.0f;
            p_values[chunk * 32 + j] = p0;
            p_values[chunk * 32 + j + 1] = p1;
            p_packed[j / 2] = cvt_f32x2_to_bf16x2(p0, p1);
          }

          // Packed P aliases FP32 S. Striped ownership ensures the compressed
          // destination overlaps only score columns already loaded by all
          // eight compute warps.
          if (chunk == 0) {
            tcgen05_fence_before_thread_sync();
            bar_sync<1>(8 * 32);
            tcgen05_fence_after_thread_sync();
          }
          tcgen05_st_32x32b_x16(
              tmem_base + row_addr + col / 2, p_packed);
          tcgen05_wait_st();
        }
        tcgen05_fence_before_thread_sync();
        __syncwarp();
        if (lane == 0) {
          mbarrier_arrive_release_cta(smem_ptr_u32(&role_bar[0]));
        }

        mbarrier_wait_parity(smem_ptr_u32(&mma_full[1]), mma_phase);
        for (int chunk = 0; chunk < 2; ++chunk) {
          const int col = col_begin + chunk * 64;
          uint32_t dp_regs[32];
          uint32_t ds_packed[16];
          tcgen05_ld_32x32b_x32(
              tmem_base + K_TILE + row_addr + col, dp_regs);
          tcgen05_wait_ld();
          float* dps = reinterpret_cast<float*>(dp_regs);
          #pragma unroll
          for (int j = 0; j < 32; j += 2) {
            const int query_local0 = col + j;
            const int query_local1 = query_local0 + 1;
            const float ds0 =
                p_values[chunk * 32 + j] *
                (dps[j] - t_d_consumer[query_local0]);
            const float ds1 =
                p_values[chunk * 32 + j + 1] *
                (dps[j + 1] - t_d_consumer[query_local1]);
            ds_packed[j / 2] = cvt_f32x2_to_bf16x2(ds0, ds1);
            const int key_subtile = key_local / 64;
            const int key_col = key_local & 63;
            const int ds0_logical = query_local0 * 64 + key_col;
            const int ds1_logical = query_local1 * 64 + key_col;
            const int ds0_physical =
                key_subtile * (M_TILE * 64) +
                static_cast<int>(
                    smem_swizzle_b128(
                        ds0_logical * sizeof(__nv_bfloat16)) /
                    sizeof(__nv_bfloat16));
            const int ds1_physical =
                key_subtile * (M_TILE * 64) +
                static_cast<int>(
                    smem_swizzle_b128(
                        ds1_logical * sizeof(__nv_bfloat16)) /
                    sizeof(__nv_bfloat16));
            t_ds[ds0_physical] = __float2bfloat16_rn(ds0);
            t_ds[ds1_physical] = __float2bfloat16_rn(ds1);
          }

          // Packed dS aliases FP32 dP under the same striped-load contract.
          if (chunk == 0) {
            tcgen05_fence_before_thread_sync();
            bar_sync<1>(8 * 32);
            tcgen05_fence_after_thread_sync();
          }
          tcgen05_st_32x32b_x16(
              tmem_base + K_TILE + row_addr + col / 2, ds_packed);
          tcgen05_wait_st();
        }
        tcgen05_fence_before_thread_sync();
        __syncwarp();
        if (lane == 0) {
          mbarrier_arrive_release_cta(smem_ptr_u32(&role_bar[1]));
        }
      }
      if (warp == 12) {
        mbarrier_wait_parity(smem_ptr_u32(&role_bar[0]), role_phase);
      }

      // MMA WARP (12): dV=P^T@dO and dK=dS^T@Q use FA4's packed-TMEM TS
      // contract and accumulate across query tiles. dQ=dS@K is reset for each
      // query tile and retains the query-major dS plus explicit K^T SS path.
      if (warp == 12) {
        const uint32_t lead = elect_one_sync();
        if (lead) {
          tcgen05_fence_after_thread_sync();
        }
        constexpr uint32_t DESC_SBO = 1024;
        constexpr uint32_t DESC_LBO = 16;
        constexpr uint64_t A_TILE_DELTA =
            (K_TILE * 64 * sizeof(__nv_bfloat16)) >> 4;
        constexpr uint64_t B_TILE_DELTA =
            (HEAD_DIM * 64 * sizeof(__nv_bfloat16)) >> 4;
        const uint32_t idesc_dq =
            make_idesc_bf16_f32(M_TILE, HEAD_DIM, false, true);
        const uint32_t idesc_dkv_ts =
            make_idesc_bf16_f32(K_TILE, HEAD_DIM, false, true);
        const uint32_t idesc_score =
            make_idesc_bf16_f32(M_TILE, K_TILE, false, false);
        const uint64_t desc_do = build_smem_desc_blackwell(
            smem_ptr_u32(t_dout + q_stage * M_TILE * HEAD_DIM), DESC_SBO,
            DESC_LBO,
            SmemSwizzleBlackwell::B128);
        const uint64_t desc_q = build_smem_desc_blackwell(
            smem_ptr_u32(t_q_stage), DESC_SBO, DESC_LBO,
            SmemSwizzleBlackwell::B128);
        const uint64_t desc_ds_row = build_smem_desc_blackwell(
            smem_ptr_u32(t_ds), DESC_SBO, DESC_LBO,
            SmemSwizzleBlackwell::B128);
        const uint64_t desc_k = build_smem_desc_blackwell(
            smem_ptr_u32(t_k), DESC_SBO, DESC_LBO,
            SmemSwizzleBlackwell::B128);
        const uint64_t desc_v = build_smem_desc_blackwell(
            smem_ptr_u32(t_v), DESC_SBO, DESC_LBO,
            SmemSwizzleBlackwell::B128);

        // Dense FA4 publishes P as two BF16 values per TMEM cell. Each
        // f16 TS instruction consumes 16 query rows: eight TMEM columns and
        // 0x80 descriptor units in the row-major dO operand.
        #pragma unroll
        for (int ki = 0; ki < M_TILE / 16; ++ki) {
          tcgen05_mma_f16_ts_1sm_lead(
              lead, tmem_base + 256, tmem_base + 8 * ki,
              desc_do + 0x80 * ki, idesc_dkv_ts,
              m_start != m_begin || ki != 0);
        }
        if (is_last_m) {
          mbarrier_wait_parity(smem_ptr_u32(&dkv_empty[0]), 1);
          tcgen05_commit1_lead(lead, smem_ptr_u32(&dkv_full[0]));
        } else {
          tcgen05_commit1_lead(lead, smem_ptr_u32(&load_empty[q_stage]));
        }

        // Once dV has consumed packed P, its TMEM columns can immediately
        // become S for the next tile. Q's second stage makes this lookahead
        // independent of the current dK/dQ tail.
        if (has_next_m) {
          const int next_iter = consumer_iter + 1;
          const int next_q_stage = next_iter & 1;
          const uint32_t next_q_full_phase =
              static_cast<uint32_t>((next_iter / 2) & 1);
          mbarrier_wait_parity(smem_ptr_u32(&q_full[next_q_stage]),
                               next_q_full_phase);
          const uint64_t desc_q_next = build_smem_desc_blackwell(
              smem_ptr_u32(t_q + next_q_stage * M_TILE * HEAD_DIM),
              DESC_SBO, DESC_LBO, SmemSwizzleBlackwell::B128);
          #pragma unroll
          for (int ki = 0; ki < HEAD_DIM / 16; ++ki) {
            tcgen05_mma_f16_ss_lead(
                lead, tmem_base, desc_k + 2 * ki,
                desc_q_next + 2 * ki, idesc_score, ki != 0);
          }
          tcgen05_commit1_lead(lead, smem_ptr_u32(&mma_full[0]));
        }

        mbarrier_wait_parity(smem_ptr_u32(&role_bar[1]), role_phase);
        #pragma unroll
        for (int ki = 0; ki < M_TILE / 16; ++ki) {
          tcgen05_mma_f16_ts_1sm_lead(
              lead, tmem_base + 320, tmem_base + K_TILE + 8 * ki,
              desc_q + 0x80 * ki, idesc_dkv_ts,
              m_start != m_begin || ki != 0);
        }
        if (is_last_m) {
          mbarrier_wait_parity(smem_ptr_u32(&dkv_empty[1]), 1);
          tcgen05_commit1_lead(lead, smem_ptr_u32(&dkv_full[1]));
        } else {
          tcgen05_commit1_lead(lead, smem_ptr_u32(&q_empty[q_stage]));
        }

        // Causal tiles benefit from overlapping dP(next) with the current dQ
        // tail. Same-warp issue ordering protects packed dS after dK, while
        // dQ reads the independent query-major SMEM copy.
        if constexpr (IS_CAUSAL) {
          if (has_next_m) {
            const int next_iter = consumer_iter + 1;
            const int next_load_stage = next_iter & 1;
            const uint32_t next_load_full_phase =
                static_cast<uint32_t>((next_iter / 2) & 1);
            mbarrier_wait_parity(smem_ptr_u32(&load_full[next_load_stage]),
                                 next_load_full_phase);
            const uint64_t desc_do_next = build_smem_desc_blackwell(
                smem_ptr_u32(t_dout + next_load_stage * M_TILE * HEAD_DIM),
                DESC_SBO, DESC_LBO,
                SmemSwizzleBlackwell::B128);
            #pragma unroll
            for (int ki = 0; ki < HEAD_DIM / 16; ++ki) {
              tcgen05_mma_f16_ss_lead(
                  lead, tmem_base + K_TILE, desc_v + 2 * ki,
                  desc_do_next + 2 * ki, idesc_score, ki != 0);
            }
            tcgen05_commit1_lead(lead, smem_ptr_u32(&mma_full[1]));
          }
        }

        // The four reduce warps release this one-stage dQ TMEM slot after
        // loading it into the independent FP32 TMA buffer. Compute warps do
        // not need to wait for the subsequent global reduce-add.
        const uint32_t dq_empty_phase =
            1u ^ static_cast<uint32_t>(consumer_iter & 1);
        mbarrier_wait_parity(smem_ptr_u32(&role_bar[2]), dq_empty_phase);
        #pragma unroll
        for (int subtile = 0; subtile < 2; ++subtile) {
          #pragma unroll
          for (int ki = 0; ki < 4; ++ki) {
            tcgen05_mma_f16_ss_lead(
                lead, tmem_base + 384,
                desc_ds_row + subtile * A_TILE_DELTA + 2 * ki,
                desc_k + subtile * B_TILE_DELTA + 2 * HEAD_DIM * ki,
                idesc_dq,
                subtile != 0 || ki != 0);
          }
        }
        tcgen05_commit1_lead(lead, smem_ptr_u32(&mma_full[2]));

        // Full tiles retain dQ-before-dP; issuing dP early extends their
        // tensor-core queue and reproducibly slows both short and long cases.
        if constexpr (!IS_CAUSAL) {
          if (has_next_m) {
            const int next_iter = consumer_iter + 1;
            const int next_load_stage = next_iter & 1;
            const uint32_t next_load_full_phase =
                static_cast<uint32_t>((next_iter / 2) & 1);
            mbarrier_wait_parity(smem_ptr_u32(&load_full[next_load_stage]),
                                 next_load_full_phase);
            const uint64_t desc_do_next = build_smem_desc_blackwell(
                smem_ptr_u32(t_dout + next_load_stage * M_TILE * HEAD_DIM),
                DESC_SBO, DESC_LBO,
                SmemSwizzleBlackwell::B128);
            #pragma unroll
            for (int ki = 0; ki < HEAD_DIM / 16; ++ki) {
              tcgen05_mma_f16_ss_lead(
                  lead, tmem_base + K_TILE, desc_v + 2 * ki,
                  desc_do_next + 2 * ki, idesc_score, ki != 0);
            }
            tcgen05_commit1_lead(lead, smem_ptr_u32(&mma_full[1]));
          }
        }
      }

      if (warp < 4 || warp >= 12) {
        mbarrier_wait_parity(smem_ptr_u32(&mma_full[2]), mma_phase);
      }

      // REDUCE WARPS (0-3): drain FP32 dQ partials from TMEM into the dead
      // former P^T buffer, retiled as two 128x32 B128 FP32 subtiles.
      float* t_dq_smem = reinterpret_cast<float*>(t_dq_smem_storage);
      if (warp >= 0 && warp <= 3) {
        const int row_group = warp;
        const int query_local = row_group * 32 + lane;
        const int query = m_start + query_local;
        const uint32_t row_addr =
            static_cast<uint32_t>(row_group * 32) << 16;
        for (int col = 0; col < HEAD_DIM; col += 32) {
          uint32_t dq_regs[32];
          tcgen05_ld_32x32b_x32(
              tmem_base + 384 + row_addr + col, dq_regs);
          tcgen05_wait_ld();
          // Hold the first TMEM slice in registers while the preceding TMA
          // reduce-add finishes with this one-stage SMEM buffer. Full tiles
          // benefit; causal tiles retain the lower-pressure trailing wait.
          if constexpr (!IS_CAUSAL) {
            if (col == 0) {
              const uint32_t dq_smem_empty_phase =
                  1u ^ static_cast<uint32_t>(consumer_iter & 1);
              mbarrier_wait_parity(smem_ptr_u32(&role_bar[3]),
                                   dq_smem_empty_phase);
            }
          }
          float* dq_values = reinterpret_cast<float*>(dq_regs);
          #pragma unroll
          for (int j = 0; j < 32; ++j) {
            const int logical = query_local * 32 + j;
            const int physical =
                (col / 32) * (M_TILE * 32) +
                static_cast<int>(smem_swizzle_b128(
                                     logical * sizeof(float)) /
                                 sizeof(float));
            t_dq_smem[physical] = query < seq ? dq_values[j] : 0.0f;
          }
        }
        tcgen05_fence_before_thread_sync();
        __syncwarp();
        if (lane == 0) {
          mbarrier_arrive_release_cta(smem_ptr_u32(&role_bar[2]));
        }
      }

      // REDUCE WARP 0: two TMA reduce-add stores replace 8,192 scalar
      // atomics. A per-head 2D map prevents a tail tile from crossing into
      // the next head's sequence.
      if (warp == 0) {
        mbarrier_wait_parity(smem_ptr_u32(&role_bar[2]), role_phase);
        const uint32_t lead = elect_one_sync();
        if (lead) {
          fence_proxy_async_shared_cta();
          const CUtensorMap* dq_map = tmap_dq + head_linear;
          tma_store_2d_add(dq_map, 0, m_start, smem_ptr_u32(t_dq_smem));
          tma_store_2d_add(
              dq_map, 32, m_start,
              smem_ptr_u32(t_dq_smem + M_TILE * 32));
          cp_async_bulk_commit_group();
          cp_async_bulk_wait_group<0>();
          mbarrier_arrive_release_cta(smem_ptr_u32(&role_bar[3]));
        }
      }
      if constexpr (IS_CAUSAL) {
        if (warp <= 3) {
          mbarrier_wait_parity(smem_ptr_u32(&role_bar[3]), role_phase);
        } else if (warp >= 14 && warp <= 15) {
          mbarrier_wait_parity(smem_ptr_u32(&mma_full[2]), mma_phase);
        }
      } else {
        if (warp >= 14 && warp <= 15) {
          mbarrier_wait_parity(smem_ptr_u32(&mma_full[2]), mma_phase);
        }
      }
      role_phase ^= 1u;
      mma_phase ^= 1u;
    }
    }

    // COMPUTE WARPS (4-11): the two dKV pipeline stages publish dV before
    // dK. This lets the dV TMEM drain overlap the final dK and dQ MMAs.
    if (warp >= 4 && warp <= 11) {
      const int row_group = (warp - 4) & 3;
      const int key_local = row_group * 32 + lane;
      const uint32_t row_addr =
          static_cast<uint32_t>(row_group * 32) << 16;
      const int col = warp < 8 ? 0 : 32;

      mbarrier_wait_parity(smem_ptr_u32(&dkv_full[0]), 0);
      uint32_t dv_regs[32];
      tcgen05_ld_32x32b_x32(
          tmem_base + 256 + row_addr + col, dv_regs);
      tcgen05_wait_ld();
      float* dv_values = reinterpret_cast<float*>(dv_regs);
      #pragma unroll
      for (int j = 0; j < 32; j += 2) {
        const int logical = key_local * HEAD_DIM + col + j;
        const int physical =
            static_cast<int>(smem_swizzle_b128(
                                 logical * sizeof(__nv_bfloat16)) /
                             sizeof(__nv_bfloat16));
        reinterpret_cast<uint32_t*>(t_v + physical)[0] =
            cvt_f32x2_to_bf16x2(dv_values[j], dv_values[j + 1]);
      }
      tcgen05_fence_before_thread_sync();
      __syncwarp();
      if (lane == 0) {
        mbarrier_arrive_release_cta(smem_ptr_u32(&dkv_empty[0]));
      }
      if constexpr (USE_2CTA) {
      fence_proxy_async_shared_cta();
      bar_sync<4>(8 * 32);
      if (warp == 4 && lane == 0) {
        tma_store_3d(&tmap_dv, 0, key_start, head_linear,
                     smem_ptr_u32(t_v));
        tma_store_3d(&tmap_dv, 64, key_start, head_linear,
                     smem_ptr_u32(t_v + K_TILE * 64));
        cp_async_bulk_commit_group();
        cp_async_bulk_wait_group<0>();
      }
      bar_sync<4>(8 * 32);
      }

      mbarrier_wait_parity(smem_ptr_u32(&dkv_full[1]), 0);
      uint32_t dk_regs[32];
      tcgen05_ld_32x32b_x32(
          tmem_base + 320 + row_addr + col, dk_regs);
      tcgen05_wait_ld();
      float* dk_values = reinterpret_cast<float*>(dk_regs);
      #pragma unroll
      for (int j = 0; j < 32; j += 2) {
        const int logical = key_local * HEAD_DIM + col + j;
        const int physical =
            static_cast<int>(smem_swizzle_b128(
                                 logical * sizeof(__nv_bfloat16)) /
                             sizeof(__nv_bfloat16));
        reinterpret_cast<uint32_t*>(t_k + physical)[0] =
            cvt_f32x2_to_bf16x2(dk_values[j] * scale,
                                dk_values[j + 1] * scale);
      }
      tcgen05_fence_before_thread_sync();
      __syncwarp();
      if (lane == 0) {
        mbarrier_arrive_release_cta(smem_ptr_u32(&dkv_empty[1]));
      }
    }

    // AUX WARP (14): consume the two epilogue stages independently. dV's
    // TMA store can run while compute warps wait for and drain dK.
    if (warp == 14) {
      mbarrier_wait_parity_suspend(smem_ptr_u32(&dkv_empty[0]), 0);
      const uint32_t lead = elect_one_sync();
      if (lead) {
        fence_proxy_async_shared_cta();
        tma_store_3d(&tmap_dv, 0, key_start, head_linear,
                     smem_ptr_u32(t_v));
        cp_async_bulk_commit_group();
      }

      mbarrier_wait_parity_suspend(smem_ptr_u32(&dkv_empty[1]), 0);
      if (lead) {
        fence_proxy_async_shared_cta();
        tma_store_3d(&tmap_dk, 0, key_start, head_linear,
                     smem_ptr_u32(t_k));
        cp_async_bulk_commit_group();
        cp_async_bulk_wait_group<0>();
      }
    }

    __syncthreads();
    if (warp == 12) {
      tcgen05_dealloc<1>(tmem_base, 512);
    }
    __syncthreads();
    if (tid == 0) {
      mbarrier_inval(smem_ptr_u32(&mma_full[0]));
      mbarrier_inval(smem_ptr_u32(&mma_full[1]));
      mbarrier_inval(smem_ptr_u32(&mma_full[2]));
      mbarrier_inval(smem_ptr_u32(&q_full[0]));
      mbarrier_inval(smem_ptr_u32(&q_full[1]));
      mbarrier_inval(smem_ptr_u32(&q_empty[0]));
      mbarrier_inval(smem_ptr_u32(&q_empty[1]));
      mbarrier_inval(smem_ptr_u32(kv_full));
      mbarrier_inval(smem_ptr_u32(&load_full[0]));
      mbarrier_inval(smem_ptr_u32(&load_full[1]));
      mbarrier_inval(smem_ptr_u32(&load_empty[0]));
      mbarrier_inval(smem_ptr_u32(&load_empty[1]));
      mbarrier_inval(smem_ptr_u32(&dkv_full[0]));
      mbarrier_inval(smem_ptr_u32(&dkv_full[1]));
      mbarrier_inval(smem_ptr_u32(&dkv_empty[0]));
      mbarrier_inval(smem_ptr_u32(&dkv_empty[1]));
      mbarrier_inval(smem_ptr_u32(&role_bar[0]));
      mbarrier_inval(smem_ptr_u32(&role_bar[1]));
      mbarrier_inval(smem_ptr_u32(&role_bar[2]));
      mbarrier_inval(smem_ptr_u32(&role_bar[3]));
    }
    return;
  }

  // D128 dense 1-CTA tensor-core bring-up. The TMEM map matches the dense
  // FA4 contract: S/P[0:128], dV[128:256], dP/dS/dQ[256:384], and
  // dK[384:512]. Q/metadata are double-buffered, dO is single-buffered, and
  // dQ uses a two-stage 128x32 FP32 TMA reduce-add ring.
  if constexpr (HEAD_DIM == 128) {
    // FA4 keeps all pipeline/cluster mbarriers in a compact header before
    // the aligned matrix buffers. In particular, DSMEM arrivals must not
    // target barriers parked at the extreme end of the opt-in SMEM window.
    // Preserve the accepted 1-CTA byte layout and apply the header only to
    // the dedicated D128 2-CTA specialization.
    unsigned char* d128_payload =
        shared_raw + (USE_2CTA ? D128_2CTA_HEADER_BYTES : 0);
    // Dense FA4 USE_2CTA SharedStorage order after the barrier header:
    // sQ, sK, sV, sdO, sQt, sdOt, sdS_xchg, sKt, sdS,
    // sLSE, sdPsum, sdQaccum. Keep the accepted 1-CTA layout unchanged.
    __nv_bfloat16* t_q_score_2cta =
        reinterpret_cast<__nv_bfloat16*>(d128_payload);
    __nv_bfloat16* t_k = USE_2CTA
        ? t_q_score_2cta + M_TILE * 64
        : reinterpret_cast<__nv_bfloat16*>(d128_payload);
    __nv_bfloat16* t_v = t_k + K_TILE * HEAD_DIM;
    __nv_bfloat16* t_q = USE_2CTA
        ? t_q_score_2cta
        : t_v + K_TILE * HEAD_DIM;
    __nv_bfloat16* t_dout = USE_2CTA
        ? t_v + K_TILE * HEAD_DIM
        : t_q + 2 * M_TILE * HEAD_DIM;
    __nv_bfloat16* t_qt_2cta = t_dout + M_TILE * 64;
    __nv_bfloat16* t_dout_dp_2cta = t_qt_2cta + M_TILE * 64;
    __nv_bfloat16* t_ds_xchg_2cta = t_dout_dp_2cta + M_TILE * 64;
    __nv_bfloat16* t_kt_2cta = t_ds_xchg_2cta + M_TILE * 64;
    __nv_bfloat16* t_dq_smem_storage = USE_2CTA
        ? t_kt_2cta
        : t_dout + M_TILE * HEAD_DIM;
    __nv_bfloat16* t_ds = USE_2CTA
        ? t_kt_2cta + K_TILE * HEAD_DIM
        : t_dq_smem_storage + M_TILE * HEAD_DIM;
    float* t_lse_log2 =
        reinterpret_cast<float*>(t_ds + M_TILE * K_TILE);
    float* t_d =
        t_lse_log2 + (USE_2CTA ? M_TILE : 2 * M_TILE);
    __nv_bfloat16* t_dq_reduce_2cta =
        reinterpret_cast<__nv_bfloat16*>(t_d + M_TILE);
    uint64_t* mma_full = USE_2CTA
        ? reinterpret_cast<uint64_t*>(shared_raw)
        : reinterpret_cast<uint64_t*>(t_d + 2 * M_TILE);
    uint64_t* q_full = mma_full + 3;
    uint64_t* q_empty = q_full + 2;
    uint64_t* kv_full = q_empty + 2;
    uint64_t* load_full = kv_full + 1;
    uint64_t* load_empty = load_full + 1;
    uint64_t* dkv_full = load_empty + 1;
    uint64_t* dkv_empty = dkv_full + 2;
    uint64_t* role_bar = dkv_empty + 2;
    uint64_t* dqbuf_full = role_bar + 3;
    uint64_t* dqbuf_empty = dqbuf_full + 2;
    uint64_t* metadata_bar = dqbuf_empty + 2;
    constexpr int LSE_STAGES = USE_2CTA ? 1 : 2;
    uint64_t* lse_full = metadata_bar;
    uint64_t* lse_empty = lse_full + LSE_STAGES;
    uint64_t* dpsum_full = lse_empty + LSE_STAGES;
    uint64_t* dpsum_empty = dpsum_full + 1;
    uint64_t* pair_bar = dpsum_empty + 1;
    uint64_t* ds_full = pair_bar;
    uint64_t* ds_empty = pair_bar + 1;
    uint64_t* ds_cluster_empty = pair_bar + 5;
    uint64_t* ds_cluster_full = ds_cluster_empty + 1;
    uint64_t* ds_cluster_leader = ds_cluster_full + 1;
    uint64_t* tmem_dealloc_bar = pair_bar + 2;
    uint32_t* tmem_slot = reinterpret_cast<uint32_t*>(
        pair_bar + (USE_2CTA ? 8 : 0));

    if (tid == 0) {
      mbarrier_init(smem_ptr_u32(&mma_full[0]), 1);
      mbarrier_init(smem_ptr_u32(&mma_full[1]), 1);
      mbarrier_init(smem_ptr_u32(&mma_full[2]), 1);
      mbarrier_init(smem_ptr_u32(&q_full[0]), 1);
      mbarrier_init(smem_ptr_u32(&q_full[1]), 1);
      mbarrier_init(smem_ptr_u32(&q_empty[0]), 1);
      mbarrier_init(smem_ptr_u32(&q_empty[1]), 1);
      mbarrier_init(smem_ptr_u32(kv_full), 1);
      mbarrier_init(smem_ptr_u32(load_full), 1);
      mbarrier_init(smem_ptr_u32(load_empty), 1);
      mbarrier_init(smem_ptr_u32(&dkv_full[0]), 1);
      mbarrier_init(smem_ptr_u32(&dkv_full[1]), 1);
      mbarrier_init(smem_ptr_u32(&dkv_empty[0]), 8);
      mbarrier_init(smem_ptr_u32(&dkv_empty[1]), 8);
      mbarrier_init(smem_ptr_u32(&role_bar[0]),
                    USE_2CTA ? 16 : 8);
      mbarrier_init(smem_ptr_u32(&role_bar[1]),
                    USE_2CTA ? 16 : 8);
      mbarrier_init(smem_ptr_u32(&role_bar[2]),
                    USE_2CTA ? 8 : 4);
      mbarrier_init(smem_ptr_u32(&dqbuf_full[0]), 4);
      mbarrier_init(smem_ptr_u32(&dqbuf_full[1]), 4);
      mbarrier_init(smem_ptr_u32(&dqbuf_empty[0]), 1);
      mbarrier_init(smem_ptr_u32(&dqbuf_empty[1]), 1);
      #pragma unroll
      for (int stage = 0; stage < LSE_STAGES; ++stage) {
        mbarrier_init(smem_ptr_u32(&lse_full[stage]), 1);
        mbarrier_init(smem_ptr_u32(&lse_empty[stage]), 8);
      }
      mbarrier_init(smem_ptr_u32(dpsum_full), 1);
      mbarrier_init(smem_ptr_u32(dpsum_empty), 8);
      if constexpr (USE_2CTA) {
        // PipelineAsyncUmma dS full: one elected arrival from each of the
        // eight compute warps in both CTAs, consumed only by the leader MMA
        // warp. Its empty barrier is released by a second tcgen05 commit
        // after dQ has consumed sdS, matching FA4's one-stage dS pipeline.
        // This is distinct from the three-barrier DSMEM relay.
        mbarrier_init(smem_ptr_u32(ds_full), 16);
        mbarrier_init(smem_ptr_u32(ds_empty), 1);
      }
    }
    if constexpr (USE_2CTA) {
      // FA4 assigns the explicit DSMEM relay-barrier initialization to one
      // elected lane of compute warp 4.
      if (warp == 4 && lane == 0) {
        mbarrier_init(smem_ptr_u32(ds_cluster_empty), 1);
        mbarrier_init(smem_ptr_u32(ds_cluster_full), 1);
        mbarrier_init(smem_ptr_u32(ds_cluster_leader), 2);
      }
      if (warp == 12 && lane == 0) {
        // FA4 TmemAllocator initializes one peer-arrival per allocator-warp
        // thread.  At free, each of the 32 lanes arrives on the peer copy.
        mbarrier_init(smem_ptr_u32(tmem_dealloc_bar), 32);
      }
    }
    fence_mbarrier_init_release_cluster();
    __syncthreads();
    if constexpr (USE_2CTA) {
      barrier_cluster_arrive();
      barrier_cluster_wait();
      if (warp == 15) {
        setmaxnreg_dec<24>();
        return;
      }
    }

    // FA4's inert roles donate immediately and do not reconverge through a
    // CTA-wide teardown.  USE_2CTA retains its established relay path.
    if constexpr (!USE_2CTA) {
      if (warp >= 14) {
        setmaxnreg_dec<24>();
        return;
      }
      if (warp == 13) {
        setmaxnreg_dec<88>();
      } else if (warp == 12) {
        setmaxnreg_dec<88>();
      } else if (warp >= 4) {
        setmaxnreg_inc<136>();
      } else {
        setmaxnreg_inc<152>();
      }
    }

    uint32_t tmem_base = 0;
    if constexpr (!USE_2CTA) {
      if (warp == 12) {
        tcgen05_alloc<1>(smem_ptr_u32(tmem_slot), 512);
        tcgen05_relinquish_alloc_permit<1>();
      }
      if (warp <= 12) {
        // FA4 tmem_alloc_barrier: reduce + compute + MMA, never load.
        bar_sync<10>(13 * 32);
      }
      tmem_base = *tmem_slot;
    }

    const int cluster_key_start =
        key_start - static_cast<int>(cta_rank) * K_TILE;
    if (warp == 13 && lane == 0) {
      constexpr uint32_t K_OR_V_CLUSTER_BYTES =
          2 * K_TILE * HEAD_DIM * sizeof(__nv_bfloat16);
      if constexpr (!USE_2CTA) {
        mbarrier_arrive_expect_tx(smem_ptr_u32(kv_full),
                                  K_OR_V_CLUSTER_BYTES);
        tma_load_3d(smem_ptr_u32(t_k), &tmap_k, smem_ptr_u32(kv_full), 0,
                    key_start, head_linear);
        tma_load_3d(smem_ptr_u32(t_k + K_TILE * 64), &tmap_k,
                    smem_ptr_u32(kv_full), 64, key_start, head_linear);
        tma_load_3d(smem_ptr_u32(t_v), &tmap_v, smem_ptr_u32(kv_full), 0,
                    key_start, head_linear);
        tma_load_3d(smem_ptr_u32(t_v + K_TILE * 64), &tmap_v,
                    smem_ptr_u32(kv_full), 64, key_start, head_linear);
      }
    }
    const int m_begin = IS_CAUSAL ? cluster_key_start : 0;

    // LOAD WARP (13): issue Q and its metadata before waiting for the
    // single-stage dO slot. This lets Q(next) land while dV(current) runs.
    if (warp == 13) {
      if constexpr (USE_2CTA) {
        // FA4 donates the load role's register allocation only after entering
        // the load branch; it never traverses the TMEM allocator wait.
        setmaxnreg_dec<104>();
      }
      int load_iter = 0;
      for (int load_m = m_begin; load_m < seq;
           load_m += M_TILE, ++load_iter) {
        const int q_stage = USE_2CTA ? 0 : (load_iter & 1);
        const uint32_t q_empty_phase = USE_2CTA
            ? (1u ^ static_cast<uint32_t>(load_iter & 1))
            : (1u ^ static_cast<uint32_t>((load_iter / 2) & 1));
        if constexpr (USE_2CTA) {
          // Dense FA4 main load order is Qt(previous), Q(current). The final
          // Qt is issued by the tail below.
          if (load_iter > 0) {
            const int qt_iter = load_iter - 1;
            const uint32_t qt_empty_phase =
                1u ^ static_cast<uint32_t>(qt_iter & 1);
            mbarrier_wait_parity_suspend(smem_ptr_u32(&q_empty[1]),
                                         qt_empty_phase);
            if (lane == 0) {
              constexpr uint32_t Q_BYTES =
                  M_TILE * HEAD_DIM * sizeof(__nv_bfloat16);
              if (cta_rank == 0) {
                mbarrier_arrive_expect_tx(smem_ptr_u32(&q_full[1]),
                                          Q_BYTES);
              }
              const uint32_t qt_route =
                  tma_peer_bit_mask(smem_ptr_u32(&q_full[1]));
              tma_load_3d_2sm(smem_ptr_u32(t_qt_2cta), &tmap_q,
                              qt_route, cta_rank * 64,
                              load_m - M_TILE, head_linear);
            }
          }
          mbarrier_wait_parity_suspend(smem_ptr_u32(&q_empty[0]),
                                       q_empty_phase);
        } else {
          mbarrier_wait_parity_suspend(smem_ptr_u32(&q_empty[q_stage]),
                                       q_empty_phase);
        }
        if (lane == 0) {
          constexpr uint32_t Q_BYTES =
              M_TILE * HEAD_DIM * sizeof(__nv_bfloat16);
          __nv_bfloat16* t_q_load =
              t_q + q_stage * M_TILE * HEAD_DIM;
          if constexpr (!USE_2CTA) {
            mbarrier_arrive_expect_tx(smem_ptr_u32(&q_full[q_stage]),
                                      Q_BYTES);
            tma_load_3d(smem_ptr_u32(t_q_load), &tmap_q,
                        smem_ptr_u32(&q_full[q_stage]), 0, load_m,
                        head_linear);
            tma_load_3d(smem_ptr_u32(t_q_load + M_TILE * 64), &tmap_q,
                        smem_ptr_u32(&q_full[q_stage]), 64, load_m,
                        head_linear);
          } else {
            if (cta_rank == 0) {
              mbarrier_arrive_expect_tx(smem_ptr_u32(&q_full[0]),
                                        Q_BYTES +
                                            (load_iter == 0
                                                 ? 2 * K_TILE * HEAD_DIM *
                                                       sizeof(__nv_bfloat16)
                                                 : 0));
            }
            const uint32_t q_score_route =
                tma_peer_bit_mask(smem_ptr_u32(&q_full[0]));
            if (load_iter == 0) {
              // FA4 prologue: K and Q complete the same Q pipeline stage.
              tma_load_3d_2sm(smem_ptr_u32(t_k), &tmap_k,
                              q_score_route, 0, key_start, head_linear);
              tma_load_3d_2sm(smem_ptr_u32(t_k + K_TILE * 64),
                              &tmap_k, q_score_route, 64,
                              key_start, head_linear);
            }
            // sQ is N-split by query rows for S = K @ Q.T.
            tma_load_3d_2sm(smem_ptr_u32(t_q_score_2cta),
                            &tmap_q_half, q_score_route, 0,
                            load_m + cta_rank * 64,
                            head_linear);
            tma_load_3d_2sm(smem_ptr_u32(t_q_score_2cta + 64 * 64),
                            &tmap_q_half, q_score_route, 64,
                            load_m + cta_rank * 64, head_linear);
          }
        }
        float* t_lse_load = t_lse_log2 + q_stage * M_TILE;
        float* t_d_load = t_d + q_stage * M_TILE;
        // FA4 pairs the Q producer state with an independent LSE TMA
        // pipeline.  D128 1CTA has two metadata stages; D128 2CTA has one.
        const uint32_t lse_empty_phase = USE_2CTA
            ? (1u ^ static_cast<uint32_t>(load_iter & 1))
            : (1u ^ static_cast<uint32_t>((load_iter / 2) & 1));
        mbarrier_wait_parity_suspend(
            smem_ptr_u32(&lse_empty[q_stage]), lse_empty_phase);
        if (lane == 0) {
          constexpr uint32_t METADATA_BYTES = M_TILE * sizeof(float);
          mbarrier_arrive_expect_tx(
              smem_ptr_u32(&lse_full[q_stage]), METADATA_BYTES);
          cuda::ptx::cp_async_bulk(
              cuda::ptx::space_shared, cuda::ptx::space_global,
              t_lse_load, lse_log2 + row_base + load_m,
              METADATA_BYTES, &lse_full[q_stage]);
        }

        const uint32_t load_empty_phase =
            1u ^ static_cast<uint32_t>(load_iter & 1);
        mbarrier_wait_parity_suspend(smem_ptr_u32(load_empty),
                                     load_empty_phase);
        if (lane == 0) {
          constexpr uint32_t DOUT_BYTES =
              M_TILE * HEAD_DIM * sizeof(__nv_bfloat16);
          if constexpr (!USE_2CTA) {
            mbarrier_arrive_expect_tx(smem_ptr_u32(load_full), DOUT_BYTES);
            tma_load_3d(smem_ptr_u32(t_dout), &tmap_dout,
                        smem_ptr_u32(load_full), 0, load_m, head_linear);
            tma_load_3d(smem_ptr_u32(t_dout + M_TILE * 64), &tmap_dout,
                        smem_ptr_u32(load_full), 64, load_m, head_linear);
          } else {
            if (cta_rank == 0) {
              mbarrier_arrive_expect_tx(smem_ptr_u32(load_full),
                                        2 * DOUT_BYTES +
                                            (load_iter == 0
                                                 ? 2 * K_TILE * HEAD_DIM *
                                                       sizeof(__nv_bfloat16)
                                                 : 0));
            }
            const uint32_t dout_route =
                tma_peer_bit_mask(smem_ptr_u32(load_full));
            if (load_iter == 0) {
              // FA4 prologue: V and both dO views share the first dO stage.
              tma_load_3d_2sm(smem_ptr_u32(t_v), &tmap_v,
                              dout_route, 0, key_start, head_linear);
              tma_load_3d_2sm(smem_ptr_u32(t_v + K_TILE * 64),
                              &tmap_v, dout_route, 64,
                              key_start, head_linear);
            }
            // dO_dV: all 128 queries x this peer's 64 output dimensions.
            tma_load_3d_2sm(smem_ptr_u32(t_dout), &tmap_dout,
                            dout_route, cta_rank * 64, load_m,
                            head_linear);
            // dOt_dP: this peer's 64 query columns x all 128 dimensions.
            tma_load_3d_2sm(smem_ptr_u32(t_dout_dp_2cta),
                            &tmap_dout_half,
                            dout_route, 0,
                            load_m + cta_rank * 64, head_linear);
            tma_load_3d_2sm(smem_ptr_u32(t_dout_dp_2cta + 64 * 64),
                            &tmap_dout_half, dout_route, 64,
                            load_m + cta_rank * 64, head_linear);
            if (load_iter == 0) {
              // Dense FA4 commits the one-stage Kt pipeline after the Q and
              // dO prologue transactions.
              constexpr uint32_t KT_BYTES =
                  2 * K_TILE * HEAD_DIM * sizeof(__nv_bfloat16);
              if (cta_rank == 0) {
                mbarrier_arrive_expect_tx(smem_ptr_u32(kv_full), KT_BYTES);
              }
              const uint32_t kt_route =
                  tma_peer_bit_mask(smem_ptr_u32(kv_full));
              // FA4 partitions Kt across the dQ N mode: each CTA loads its
              // 64 output dimensions for the full 256-key cluster tile.
              tma_load_3d_2sm(smem_ptr_u32(t_kt_2cta), &tmap_kt,
                              kt_route, cta_rank * 64,
                              cluster_key_start, head_linear);
            }
          }
        }
        // dPsum is the corresponding one-stage pipeline paired with dO.
        const uint32_t dpsum_empty_phase =
            1u ^ static_cast<uint32_t>(load_iter & 1);
        mbarrier_wait_parity_suspend(smem_ptr_u32(dpsum_empty),
                                     dpsum_empty_phase);
        if (lane == 0) {
          constexpr uint32_t METADATA_BYTES = M_TILE * sizeof(float);
          mbarrier_arrive_expect_tx(smem_ptr_u32(dpsum_full),
                                    METADATA_BYTES);
          cuda::ptx::cp_async_bulk(
              cuda::ptx::space_shared, cuda::ptx::space_global,
              t_d_load, d + row_base + load_m,
              METADATA_BYTES, dpsum_full);
        }
      }
      if constexpr (USE_2CTA) {
        // Qt tail: the load main loop produces Qt(previous), so the final
        // query tile is committed after the Q/dO loop.
        if (load_iter > 0) {
          const int qt_iter = load_iter - 1;
          const uint32_t qt_empty_phase =
              1u ^ static_cast<uint32_t>(qt_iter & 1);
          mbarrier_wait_parity_suspend(smem_ptr_u32(&q_empty[1]),
                                       qt_empty_phase);
          if (lane == 0) {
            constexpr uint32_t Q_BYTES =
                M_TILE * HEAD_DIM * sizeof(__nv_bfloat16);
            if (cta_rank == 0) {
              mbarrier_arrive_expect_tx(smem_ptr_u32(&q_full[1]), Q_BYTES);
            }
            const uint32_t qt_route =
                tma_peer_bit_mask(smem_ptr_u32(&q_full[1]));
            tma_load_3d_2sm(smem_ptr_u32(t_qt_2cta), &tmap_q,
                            qt_route, cta_rank * 64,
                            m_begin + qt_iter * M_TILE, head_linear);
          }
        }
      }
    }

    // FA4 load is independent of TMEM allocation and exits after draining
    // its producer loops in both 1CTA and 2CTA.
    if (warp == 13) {
      return;
    }

    // RELAY WARP (14): exactly one arrival per CTA and query tile. The peer
    // bulk copy completes this CTA's dS-cluster-full barrier; both relay
    // warps then arrive on the leader CTA's count-2 barrier consumed by the
    // cta_group::2 dQ MMA.
    if constexpr (USE_2CTA) {
      if (warp == 14) {
        // Keep relay's donor allocation local to the relay branch, matching
        // FA4's top-level role dispatch.
        setmaxnreg_dec<104>();
        uint32_t ds_cluster_phase = 0;
        for (int relay_m = m_begin; relay_m < seq;
             relay_m += M_TILE) {
          mbarrier_wait_parity(smem_ptr_u32(ds_cluster_full),
                               ds_cluster_phase);
          if (lane == 0) {
            const uint32_t leader_bar = mapa_shared_cluster_u32(
                smem_ptr_u32(ds_cluster_leader), 0);
            mbarrier_arrive_cluster_default(leader_bar);
          }
          ds_cluster_phase ^= 1u;
        }
        return;
      }
    }

    // Instantiate a compiler-distinct consumer loop for each FA4 role in the
    // 2CTA path.  This gives ptxas the same role-local live ranges as CuTe's
    // separate mma/compute/dQacc_reduce calls while retaining the accepted
    // legacy 1CTA loop in the fallback specialization.
    auto consumer_loop = [&](auto role_tag) {
      constexpr int DENSE_ROLE = decltype(role_tag)::value;
      for (int consumer_iter = 0;
           (IS_CAUSAL && USE_2CTA) ||
               m_begin + consumer_iter * M_TILE < seq;
           ++consumer_iter) {
        // Match the forward-kernel spelling of FA4's make_warp_uniform:
        // every role advances this cursor in warp lockstep, and the lane-0
        // broadcast lets ptxas place body-wide cursor/phase arithmetic on
        // the uniform datapath instead of carrying it through dS registers.
        const int consumer_iter_u = USE_2CTA
            ? __shfl_sync(0xffffffffu, consumer_iter, 0)
            : consumer_iter;
        if constexpr (IS_CAUSAL && USE_2CTA) {
          if (m_begin + consumer_iter_u * M_TILE >= seq) {
            break;
          }
        }
        const int m_start = m_begin + consumer_iter_u * M_TILE;
        const uint32_t role_phase =
            static_cast<uint32_t>(consumer_iter_u & 1);
        const uint32_t mma_phase = role_phase;
        const bool is_last_m = m_start + M_TILE >= seq;
        const bool has_next_m = !is_last_m;
        const int q_stage = USE_2CTA ? 0 : (consumer_iter_u & 1);
        const uint32_t q_full_phase = USE_2CTA
            ? role_phase
            : static_cast<uint32_t>((consumer_iter_u / 2) & 1);
        const uint32_t load_phase =
            static_cast<uint32_t>(consumer_iter_u & 1);
        __nv_bfloat16* t_q_stage = USE_2CTA
            ? t_qt_2cta
            : t_q + q_stage * M_TILE * HEAD_DIM;

        if ((USE_2CTA ? DENSE_ROLE == 0 : warp == 12) &&
            consumer_iter_u == 0 &&
            (!USE_2CTA || cta_rank == 0)) {
          if constexpr (!USE_2CTA) {
            // K/V are issued by the independent load warp before it enters
            // the Q/dO rings.  Only the MMA producer consumes this gate.
            bwd_pipeline_consumer_wait<USE_2CTA>(smem_ptr_u32(kv_full), 0);
          }
          bwd_pair_tma_wait_broadcast<USE_2CTA>(
              &q_full[0], q_full_phase, cta_rank, lane);
          bwd_pair_tma_wait_broadcast<USE_2CTA>(
              load_full, load_phase, cta_rank, lane);
          if constexpr (USE_2CTA) {
            // The N-split score operand has its own two-stage 16 KiB TMA
            // ring in t_dq_smem_storage.  Q's second half remains free for
            // the in-place dK transpose below.
          }
          if constexpr (!USE_2CTA) {
            const uint32_t dp_empty_phase =
                1u ^ static_cast<uint32_t>(consumer_iter_u & 1);
            bwd_pipeline_consumer_wait<USE_2CTA>(
                smem_ptr_u32(&role_bar[2]), dp_empty_phase);
          }

          uint32_t lead = elect_one_sync();
          if constexpr (USE_2CTA) {
            lead &= static_cast<uint32_t>(cta_rank == 0);
          }
          if (lead) {
            tcgen05_fence_after_thread_sync();
          }
          constexpr uint32_t DESC_SBO = 1024;
          constexpr uint32_t DESC_LBO = 16;
          constexpr uint32_t DESC_LBO_MN = 0;
          constexpr uint64_t DIM_TILE_DELTA =
              (M_TILE * 64 * sizeof(__nv_bfloat16)) >> 4;
          constexpr uint64_t B_TILE_DELTA =
              ((USE_2CTA ? 64 : M_TILE) * 64 *
               sizeof(__nv_bfloat16)) >> 4;
          const uint32_t idesc_score = make_idesc_bf16_f32(
              CTA_GROUP * M_TILE, K_TILE, false, false);
          const uint64_t desc_q = build_smem_desc_blackwell(
              smem_ptr_u32(USE_2CTA
                               ? t_q_score_2cta
                                           : t_q_stage),
              DESC_SBO, DESC_LBO,
              SmemSwizzleBlackwell::B128);
          const uint64_t desc_k = build_smem_desc_blackwell(
              smem_ptr_u32(t_k), DESC_SBO, DESC_LBO,
              SmemSwizzleBlackwell::B128);
          const uint64_t desc_do = build_smem_desc_blackwell(
              smem_ptr_u32(USE_2CTA ? t_dout_dp_2cta : t_dout),
              DESC_SBO, DESC_LBO,
              SmemSwizzleBlackwell::B128);
          const uint64_t desc_do_mn = build_smem_desc_blackwell(
              smem_ptr_u32(t_dout), DESC_SBO, DESC_LBO_MN,
              SmemSwizzleBlackwell::B128);
          const uint64_t desc_v = build_smem_desc_blackwell(
              smem_ptr_u32(t_v), DESC_SBO, DESC_LBO,
              SmemSwizzleBlackwell::B128);

          #pragma unroll
          for (int dim_subtile = 0; dim_subtile < 2; ++dim_subtile) {
            #pragma unroll
            for (int ki = 0; ki < 4; ++ki) {
              bwd_mma_f16_ss_lead<USE_2CTA>(
                  lead, tmem_base,
                  desc_k + dim_subtile * DIM_TILE_DELTA + 2 * ki,
                  desc_q + dim_subtile * B_TILE_DELTA + 2 * ki,
                  idesc_score, dim_subtile != 0 || ki != 0);
            }
          }
          bwd_mma_commit_lead<USE_2CTA>(
              lead, smem_ptr_u32(&mma_full[0]));
          if constexpr (USE_2CTA) {
            if (lead) {
              // FA4 releases the TMA Q stage with a second tcgen05 commit:
              // the load warp cannot overwrite sQ until the score MMA has
              // actually retired.
              tcgen05_commit_multicast<2>(
                  smem_ptr_u32(&q_empty[0]), 0x3);
            }
          }

          #pragma unroll
          for (int dim_subtile = 0; dim_subtile < 2; ++dim_subtile) {
            #pragma unroll
            for (int ki = 0; ki < 4; ++ki) {
              bwd_mma_f16_ss_lead<USE_2CTA>(
                  lead, tmem_base + 256,
                  desc_v + dim_subtile * DIM_TILE_DELTA + 2 * ki,
                  desc_do + dim_subtile * B_TILE_DELTA + 2 * ki,
                  idesc_score, dim_subtile != 0 || ki != 0);
            }
          }
          bwd_mma_commit_lead<USE_2CTA>(
              lead, smem_ptr_u32(&mma_full[1]));
        }

        // COMPUTE WARPS (4-11): reconstruct packed P and dS in TMEM while
        // also publishing query-major dS in SMEM for the dQ MMA.
        if (USE_2CTA ? DENSE_ROLE == 1
                     : (warp >= 4 && warp <= 11)) {
          bwd_pipeline_consumer_wait<USE_2CTA>(
              smem_ptr_u32(&lse_full[q_stage]), q_full_phase);
          bwd_pipeline_consumer_wait<USE_2CTA>(
              smem_ptr_u32(&mma_full[0]), mma_phase);
          const float* t_lse_consumer =
              t_lse_log2 + q_stage * M_TILE;
          [[maybe_unused]] const uint32_t lse_smem_base = USE_2CTA
              ? __shfl_sync(0xffffffffu,
                            smem_ptr_u32(t_lse_consumer), 0)
              : 0u;
          const float* t_d_consumer = t_d;
          const int row_group = (warp - 4) & 3;
          const int key_local = row_group * 32 + lane;
          const int key = key_start + key_local;
          const uint32_t row_addr =
              static_cast<uint32_t>(row_group * 32) << 16;
          const int col_begin = warp < 8 ? 32 : 0;

          // Keep FP32 P live across the dP wait. Publishing all of packed P
          // first lets the MMA warp start dV and S(next) while dS is formed.
          uint32_t p_regs[2][32];
          if constexpr (USE_2CTA) {
            tcgen05_ld_32x32b_x32(
                tmem_base + row_addr + col_begin, p_regs[0]);
            tcgen05_ld_32x32b_x32(
                tmem_base + row_addr + col_begin + 64, p_regs[1]);
            tcgen05_wait_ld();
            // Both S fragments are now drained, matching FA4's single tiled
            // T2R copy. Publish dS(cur) before doing any P(next) arithmetic.
            tcgen05_fence_before_thread_sync();
            if (consumer_iter_u > 0 && lane == 0) {
              const uint32_t leader_ds_full = mapa_shared_cluster_u32(
                  smem_ptr_u32(ds_full), 0);
              mbarrier_arrive_cluster_default(leader_ds_full);
            }
          }
          #pragma unroll
          for (int chunk = 0; chunk < 2; ++chunk) {
            const int col = col_begin + chunk * 64;
            uint32_t p_packed[16];
            if constexpr (!USE_2CTA) {
              tcgen05_ld_32x32b_x32(
                  tmem_base + row_addr + col, p_regs[chunk]);
              tcgen05_wait_ld();
            }
            float* scores = reinterpret_cast<float*>(p_regs[chunk]);
            if constexpr (USE_2CTA) {
              if constexpr (IS_CAUSAL && STATIC_SEQLEN != 0) {
                mask_s_row_transposed_r2p<32>(
                    scores, key, m_start + col);
              }
              #pragma unroll
              for (int j = 0; j < 32; j += 4) {
                const int query_local0 = col + j;
                const float4 lse = lds_f32x4(
                    lse_smem_base + query_local0 * sizeof(float));
                {
                  const float2 exponent01 = ffma2(
                      make_float2(scores[j], scores[j + 1]),
                      f32x2_splat(scale * LOG2_E),
                      make_float2(-lse.x, -lse.y));
                  float p0 = ex2_approx_f32(exponent01.x);
                  float p1 = ex2_approx_f32(exponent01.y);
                  if constexpr (STATIC_SEQLEN == 0) {
                    const int query0 = m_start + query_local0;
                    const bool valid0 =
                        key < seq && query0 < seq &&
                        (!IS_CAUSAL || key <= query0);
                    const bool valid1 =
                        key < seq && query0 + 1 < seq &&
                        (!IS_CAUSAL || key <= query0 + 1);
                    p0 = valid0 ? p0 : 0.0f;
                    p1 = valid1 ? p1 : 0.0f;
                  }
                  scores[j] = p0;
                  scores[j + 1] = p1;
                  p_packed[j / 2] = cvt_f32x2_to_bf16x2(p0, p1);
                }
                {
                  const float2 exponent23 = ffma2(
                      make_float2(scores[j + 2], scores[j + 3]),
                      f32x2_splat(scale * LOG2_E),
                      make_float2(-lse.z, -lse.w));
                  float p2 = ex2_approx_f32(exponent23.x);
                  float p3 = ex2_approx_f32(exponent23.y);
                  if constexpr (STATIC_SEQLEN == 0) {
                    const int query2 = m_start + query_local0 + 2;
                    const bool valid2 =
                        key < seq && query2 < seq &&
                        (!IS_CAUSAL || key <= query2);
                    const bool valid3 =
                        key < seq && query2 + 1 < seq &&
                        (!IS_CAUSAL || key <= query2 + 1);
                    p2 = valid2 ? p2 : 0.0f;
                    p3 = valid3 ? p3 : 0.0f;
                  }
                  scores[j + 2] = p2;
                  scores[j + 3] = p3;
                  p_packed[j / 2 + 1] = cvt_f32x2_to_bf16x2(p2, p3);
                }
              }
            } else {
              #pragma unroll
              for (int j = 0; j < 32; j += 2) {
                const int query_local0 = col + j;
                const int query_local1 = query_local0 + 1;
                const int query0 = m_start + query_local0;
                const int query1 = query0 + 1;
                const float2 score_pair =
                    make_float2(scores[j], scores[j + 1]);
                const float2 lse_pair = make_float2(
                    -t_lse_consumer[query_local0],
                    -t_lse_consumer[query_local1]);
                const float2 exponent_pair = ffma2(
                    score_pair, f32x2_splat(scale * LOG2_E), lse_pair);
                float p0 = ex2_approx_f32(exponent_pair.x);
                float p1 = ex2_approx_f32(exponent_pair.y);
                if constexpr (IS_CAUSAL && STATIC_SEQLEN != 0) {
                  p0 = key <= query0 ? p0 : 0.0f;
                  p1 = key <= query1 ? p1 : 0.0f;
                } else if constexpr (STATIC_SEQLEN == 0) {
                  const bool valid0 =
                      key < seq && query0 < seq &&
                      (!IS_CAUSAL || key <= query0);
                  const bool valid1 =
                      key < seq && query1 < seq &&
                      (!IS_CAUSAL || key <= query1);
                  p0 = valid0 ? p0 : 0.0f;
                  p1 = valid1 ? p1 : 0.0f;
                }
                scores[j] = p0;
                scores[j + 1] = p1;
                p_packed[j / 2] = cvt_f32x2_to_bf16x2(p0, p1);
              }
            }
            if (chunk == 0) {
              tcgen05_fence_before_thread_sync();
              bar_sync<1>(8 * 32);
              tcgen05_fence_after_thread_sync();
            }
            tcgen05_st_32x32b_x16(
                tmem_base + row_addr + col / 2, p_packed);
          }
          tcgen05_wait_st();
          tcgen05_fence_before_thread_sync();
          bar_sync<1>(8 * 32);
          tcgen05_fence_after_thread_sync();
          if (lane == 0) {
            if constexpr (USE_2CTA) {
              const uint32_t leader_p_empty = mapa_shared_cluster_u32(
                  smem_ptr_u32(&role_bar[0]), 0);
              mbarrier_arrive_cluster_default(leader_p_empty);
            } else {
              mbarrier_arrive_release_cta(smem_ptr_u32(&role_bar[0]));
            }
            mbarrier_arrive_release_cta(
                smem_ptr_u32(&lse_empty[q_stage]));
          }
          bwd_pipeline_consumer_wait<USE_2CTA>(smem_ptr_u32(dpsum_full),
                                               role_phase);
          bwd_pipeline_consumer_wait<USE_2CTA>(
              smem_ptr_u32(&mma_full[1]), mma_phase);
          [[maybe_unused]] const uint32_t d_smem_base = USE_2CTA
              ? __shfl_sync(0xffffffffu,
                            smem_ptr_u32(t_d_consumer), 0)
              : 0u;
          // FA4 retains the peer-exchange dS stage in registers across the
          // two-stage loop and publishes it only after releasing dP/TMEM.
          [[maybe_unused]] uint32_t ds_xchg_packed[16];
          #pragma unroll
          for (int chunk = 0; chunk < 2; ++chunk) {
            const int col = col_begin + chunk * 64;
            uint32_t dp_regs[32];
            uint32_t ds_packed[16];
            tcgen05_ld_32x32b_x32(
                tmem_base + 256 + row_addr + col, dp_regs);
            tcgen05_wait_ld();
            tcgen05_fence_before_thread_sync();
            bar_sync<1>(8 * 32);
            tcgen05_fence_after_thread_sync();
            float* dps = reinterpret_cast<float*>(dp_regs);
            if constexpr (USE_2CTA) {
              #pragma unroll
              for (int j = 0; j < 32; j += 4) {
                const int query_local0 = col + j;
                const float4 d_values = lds_f32x4(
                    d_smem_base + query_local0 * sizeof(float));
                {
                  const float2 ds01 = fmul2(
                      make_float2(
                          reinterpret_cast<float*>(p_regs[chunk])[j],
                          reinterpret_cast<float*>(p_regs[chunk])[j + 1]),
                      fsub2(make_float2(dps[j], dps[j + 1]),
                            make_float2(d_values.x, d_values.y)));
                  ds_packed[j / 2] =
                      cvt_f32x2_to_bf16x2(ds01.x, ds01.y);
                }
                {
                  const float2 ds23 = fmul2(
                      make_float2(
                          reinterpret_cast<float*>(p_regs[chunk])[j + 2],
                          reinterpret_cast<float*>(p_regs[chunk])[j + 3]),
                      fsub2(make_float2(dps[j + 2], dps[j + 3]),
                            make_float2(d_values.z, d_values.w)));
                  ds_packed[j / 2 + 1] =
                      cvt_f32x2_to_bf16x2(ds23.x, ds23.y);
                }
              }
            } else {
              #pragma unroll
              for (int j = 0; j < 32; j += 2) {
                const int query_local0 = col + j;
                const int query_local1 = query_local0 + 1;
                const float2 dp_pair = make_float2(dps[j], dps[j + 1]);
                const float2 d_pair = make_float2(
                    t_d_consumer[query_local0],
                    t_d_consumer[query_local1]);
                const float2 p_pair = make_float2(
                    reinterpret_cast<float*>(p_regs[chunk])[j],
                    reinterpret_cast<float*>(p_regs[chunk])[j + 1]);
                const float2 ds_pair =
                    fmul2(p_pair, fsub2(dp_pair, d_pair));
                const float ds0 = ds_pair.x;
                const float ds1 = ds_pair.y;
                ds_packed[j / 2] = cvt_f32x2_to_bf16x2(ds0, ds1);
                const int key_subtile = key_local / 64;
                const int key_col = key_local & 63;
                const int ds0_logical = query_local0 * 64 + key_col;
                const int ds1_logical = query_local1 * 64 + key_col;
                const int ds0_physical =
                    key_subtile * (M_TILE * 64) +
                    static_cast<int>(smem_swizzle_b128(
                                         ds0_logical * sizeof(__nv_bfloat16)) /
                                     sizeof(__nv_bfloat16));
                const int ds1_physical =
                    key_subtile * (M_TILE * 64) +
                    static_cast<int>(smem_swizzle_b128(
                                         ds1_logical * sizeof(__nv_bfloat16)) /
                                     sizeof(__nv_bfloat16));
                t_ds[ds0_physical] = __float2bfloat16_rn(ds0);
                t_ds[ds1_physical] = __float2bfloat16_rn(ds1);
              }
            }
            if constexpr (USE_2CTA) {
              if (chunk == 0) {
                // pipeline_dS.producer_acquire: FA4 computes the first dS
                // register fragment before waiting for the one-stage SMEM
                // destination, then acquires immediately before R2T/R2S.
                bwd_pipeline_consumer_wait<USE_2CTA>(
                    smem_ptr_u32(ds_empty), 1u ^ role_phase);
              }
              // FA4 starts the asynchronous R2T publication before the R2S
              // copy, allowing the dK MMA dependency to overlap shared stores.
              tcgen05_st_32x32b_x16(
                  tmem_base + 256 + row_addr + col / 2, ds_packed);
              // FA4's R2S tiled copy writes one 128-bit vector per eight
              // contiguous BF16 query values. The B128 swizzle distributes
              // the 32 lane-owned key rows across banks; scalar scatters here
              // serialize the compute warps and are not the FA4 copy atom.
              // FA4's eight compute warps repeat four 32-row groups. In that
              // fixed role map these R2S coordinates are direct TID fields;
              // derive them here instead of retaining key_local across dP.
              const int r2s_tidx = static_cast<int>(threadIdx.x);
              const int key_subtile = (r2s_tidx >> 6) & 1;
              const int key_col = r2s_tidx & 63;
              const int query_half = col >> 6;
              const int query_in_half = col & 63;
              if (query_half == static_cast<int>(cta_rank)) {
                const int local_block =
                    (static_cast<int>(cta_rank) * 2 + key_subtile) * 64 * 64;
                #pragma unroll
                for (int vec = 0; vec < 4; ++vec) {
                  int physical;
                  if constexpr (IS_CAUSAL) {
                    const int logical_base = key_col * 64 + query_in_half;
                    const int physical_base = static_cast<int>(
                        smem_swizzle_b128(
                            logical_base * sizeof(__nv_bfloat16)) /
                        sizeof(__nv_bfloat16));
                    physical = physical_base ^ (vec * 8);
                  } else {
                    const int logical =
                        key_col * 64 + query_in_half + vec * 8;
                    physical = static_cast<int>(
                        smem_swizzle_b128(
                            logical * sizeof(__nv_bfloat16)) /
                        sizeof(__nv_bfloat16));
                  }
                  const int reg = vec * 4;
                  const uint4 values = make_uint4(
                      ds_packed[reg], ds_packed[reg + 1],
                      ds_packed[reg + 2], ds_packed[reg + 3]);
                  *reinterpret_cast<uint4*>(
                      t_ds + local_block + physical) = values;
                }
              } else {
                #pragma unroll
                for (int reg = 0; reg < 16; ++reg) {
                  ds_xchg_packed[reg] = ds_packed[reg];
                }
              }
            } else {
              tcgen05_st_32x32b_x16(
                  tmem_base + 256 + row_addr + col / 2, ds_packed);
            }
          }
          tcgen05_wait_st();
          tcgen05_fence_before_thread_sync();
          __syncwarp();
          if (lane == 0) {
            if constexpr (USE_2CTA) {
              const uint32_t leader_dp_empty = mapa_shared_cluster_u32(
                  smem_ptr_u32(&role_bar[1]), 0);
              mbarrier_arrive_cluster_default(leader_dp_empty);
            } else {
              mbarrier_arrive_release_cta(smem_ptr_u32(&role_bar[1]));
            }
          }
          if constexpr (USE_2CTA) {
            // After the loop, match FA4's tdPrdS_xchg -> sdS_xchg
            // autovec copy. The query-in-half coordinate is invariant across
            // the two 64-column stages owned by a compute warp.
            const int xchg_tidx = static_cast<int>(threadIdx.x);
            const int xchg_key_subtile = (xchg_tidx >> 6) & 1;
            const int xchg_key_col = xchg_tidx & 63;
            const int xchg_query_in_half = col_begin & 63;
            const int xchg_block = xchg_key_subtile * 64 * 64;
            #pragma unroll
            for (int vec = 0; vec < 4; ++vec) {
              int physical;
              if constexpr (IS_CAUSAL) {
                const int logical_base =
                    xchg_key_col * 64 + xchg_query_in_half;
                const int physical_base = static_cast<int>(
                    smem_swizzle_b128(
                        logical_base * sizeof(__nv_bfloat16)) /
                    sizeof(__nv_bfloat16));
                physical = physical_base ^ (vec * 8);
              } else {
                const int logical = xchg_key_col * 64 +
                    xchg_query_in_half + vec * 8;
                physical = static_cast<int>(
                    smem_swizzle_b128(
                        logical * sizeof(__nv_bfloat16)) /
                    sizeof(__nv_bfloat16));
              }
              const int reg = vec * 4;
              const uint4 values = make_uint4(
                  ds_xchg_packed[reg], ds_xchg_packed[reg + 1],
                  ds_xchg_packed[reg + 2], ds_xchg_packed[reg + 3]);
              *reinterpret_cast<uint4*>(
                  t_ds_xchg_2cta + xchg_block + physical) = values;
            }
            // All eight compute warps publish the retained half and exchange
            // scratch before one elected thread arms and launches the peer
            // DSMEM transfer. This is FA4's generic->async fence + named
            // compute barrier contract.
            fence_proxy_async_shared_cta();
            bar_sync<2>(8 * 32);
            if (lane == 0) {
              mbarrier_arrive_release_cta(smem_ptr_u32(dpsum_empty));
            }
            if (warp == 4 && lane == 0) {
              constexpr uint32_t DS_HALF_BYTES =
                  (M_TILE / 2) * K_TILE * sizeof(__nv_bfloat16);
              const uint32_t peer_rank = cta_rank ^ 1u;
              const uint32_t remote_dst = mapa_shared_cluster_u32(
                  smem_ptr_u32(t_ds + cta_rank * (M_TILE / 2) * K_TILE),
                  peer_rank);
              const uint32_t remote_full = mapa_shared_cluster_u32(
                  smem_ptr_u32(ds_cluster_full), peer_rank);
              mbarrier_arrive_expect_tx_cluster(remote_full, DS_HALF_BYTES);
              cpasync_bulk_s2cluster(
                  remote_dst,
                  smem_ptr_u32(t_ds_xchg_2cta),
                  DS_HALF_BYTES, remote_full);
            }
          }
          if constexpr (!USE_2CTA) {
          if (lane == 0) {
            mbarrier_arrive_release_cta(smem_ptr_u32(dpsum_empty));
          }
          }
        }

        if ((USE_2CTA ? DENSE_ROLE == 0 : warp == 12) &&
            (!USE_2CTA || cta_rank == 0)) {
          uint32_t lead = elect_one_sync();
          if constexpr (USE_2CTA) {
            lead &= static_cast<uint32_t>(cta_rank == 0);
          }
          if (lead) {
            tcgen05_fence_after_thread_sync();
          }
          constexpr uint32_t DESC_SBO = 1024;
          constexpr uint32_t DESC_LBO = 16;
          constexpr uint32_t DESC_LBO_MN = 0;
          constexpr uint64_t A_TILE_DELTA =
              (K_TILE * 64 * sizeof(__nv_bfloat16)) >> 4;
          constexpr uint64_t B_TILE_DELTA =
              (64 * 64 * sizeof(__nv_bfloat16)) >> 4;
          constexpr uint32_t Q_STAGE_DELTA =
              (M_TILE * HEAD_DIM * sizeof(__nv_bfloat16)) >> 4;
          const uint32_t idesc_dkv_ts = make_idesc_bf16_f32(
              CTA_GROUP * K_TILE, USE_2CTA ? HEAD_DIM : 64,
              false, true);
          const uint32_t idesc_dq =
              make_idesc_bf16_f32(M_TILE, 64, false, true);
          const uint32_t idesc_score =
              make_idesc_bf16_f32(CTA_GROUP * M_TILE, K_TILE,
                                   false, false);
          const uint64_t desc_do = build_smem_desc_blackwell(
              smem_ptr_u32(USE_2CTA ? t_dout_dp_2cta : t_dout),
              DESC_SBO, DESC_LBO,
              SmemSwizzleBlackwell::B128);
          const uint64_t desc_do_mn = build_smem_desc_blackwell(
              smem_ptr_u32(t_dout), DESC_SBO, DESC_LBO_MN,
              SmemSwizzleBlackwell::B128);
          uint64_t desc_q = build_smem_desc_blackwell(
              smem_ptr_u32(USE_2CTA ? t_qt_2cta : t_q),
              DESC_SBO, USE_2CTA ? DESC_LBO_MN : DESC_LBO,
              SmemSwizzleBlackwell::B128);
          smem_desc_add_lo(desc_q, q_stage * Q_STAGE_DELTA);
          const uint64_t desc_ds_row = build_smem_desc_blackwell(
              smem_ptr_u32(t_ds), DESC_SBO, DESC_LBO,
              SmemSwizzleBlackwell::B128);
          const uint64_t desc_k = build_smem_desc_blackwell(
              smem_ptr_u32(t_k), DESC_SBO, DESC_LBO,
              SmemSwizzleBlackwell::B128);
          const uint64_t desc_kt = build_smem_desc_blackwell(
              smem_ptr_u32(t_kt_2cta), DESC_SBO, DESC_LBO_MN,
              SmemSwizzleBlackwell::B128);
          const uint64_t desc_v = build_smem_desc_blackwell(
              smem_ptr_u32(t_v), DESC_SBO, DESC_LBO,
              SmemSwizzleBlackwell::B128);

          bwd_pipeline_consumer_wait<USE_2CTA>(
              smem_ptr_u32(&role_bar[0]), role_phase);
          if constexpr (USE_2CTA) {
            if (lead) {
              tcgen05_fence_after_thread_sync();
            }
          }
          if constexpr (!USE_2CTA) {
            #pragma unroll
            for (int dim_subtile = 0; dim_subtile < 2; ++dim_subtile) {
              #pragma unroll
              for (int ki = 0; ki < M_TILE / 16; ++ki) {
                bwd_mma_f16_ts_lead<false>(
                    lead, tmem_base + 128 + dim_subtile * 64,
                    tmem_base + 8 * ki,
                    desc_do + dim_subtile * A_TILE_DELTA + 0x80 * ki,
                    idesc_dkv_ts, m_start != m_begin || ki != 0);
              }
            }
          } else {
            #pragma unroll
            for (int atom = 0; atom < 8; ++atom) {
              bwd_mma_f16_ts_lead<true>(
                  lead, tmem_base + 128, tmem_base + 8 * atom,
                  desc_do_mn + 0x80 * atom, idesc_dkv_ts,
                  m_start != m_begin || atom != 0);
            }
          }
          if (is_last_m) {
            bwd_pipeline_consumer_wait<USE_2CTA>(
                smem_ptr_u32(&dkv_empty[0]), 1);
            bwd_mma_commit_lead<USE_2CTA>(
                lead, smem_ptr_u32(&dkv_full[0]));
          } else {
            bwd_mma_commit_lead<USE_2CTA>(lead,
                                           smem_ptr_u32(load_empty));
          }
          if constexpr (USE_2CTA) {
            if (consumer_iter_u == 0) {
              // FA4 waits for the single Kt stage after dV's prologue and
              // before entering the S(next)/dK/dP/dQ main loop.
              bwd_pair_tma_wait_broadcast<true>(
                  kv_full, 0, cta_rank, lane);
            }
          }

          // P has been consumed, so S(next) can replace it immediately.
          if (has_next_m) {
            const int next_iter = consumer_iter_u + 1;
            const int next_q_stage = USE_2CTA ? 0 : (next_iter & 1);
            const uint32_t next_q_full_phase = USE_2CTA
                ? static_cast<uint32_t>(next_iter & 1)
                : static_cast<uint32_t>((next_iter / 2) & 1);
            if constexpr (USE_2CTA) {
              // pipeline_dQ producer acquire: S(next) aliases dQ(cur), so
              // the leader waits for all four reduce warps in both CTAs.
              bwd_pipeline_consumer_wait<USE_2CTA>(
                  smem_ptr_u32(&role_bar[2]), 1u ^ role_phase);
            }
            bwd_pair_tma_wait_broadcast<USE_2CTA>(
                &q_full[0], next_q_full_phase, cta_rank, lane);
            if constexpr (USE_2CTA) {
              if (lead) {
                tcgen05_fence_after_thread_sync();
              }
            }
            uint64_t desc_q_next = build_smem_desc_blackwell(
                smem_ptr_u32(USE_2CTA
                                 ? t_q_score_2cta
                                             : t_q),
                DESC_SBO, DESC_LBO, SmemSwizzleBlackwell::B128);
            if constexpr (!USE_2CTA) {
              smem_desc_add_lo(desc_q_next,
                               next_q_stage * Q_STAGE_DELTA);
            }
            #pragma unroll
            for (int dim_subtile = 0; dim_subtile < 2; ++dim_subtile) {
              #pragma unroll
              for (int ki = 0; ki < 4; ++ki) {
                bwd_mma_f16_ss_lead<USE_2CTA>(
                    lead, tmem_base,
                    desc_k + dim_subtile * A_TILE_DELTA + 2 * ki,
                    desc_q_next +
                        dim_subtile *
                            (USE_2CTA ? B_TILE_DELTA : A_TILE_DELTA) +
                        2 * ki,
                    idesc_score, dim_subtile != 0 || ki != 0);
              }
            }
            bwd_mma_commit_lead<USE_2CTA>(
                lead, smem_ptr_u32(&mma_full[0]));
            if constexpr (USE_2CTA) {
              if (lead) {
                tcgen05_commit_multicast<2>(
                    smem_ptr_u32(&q_empty[0]), 0x3);
              }
            }
          }

          if constexpr (USE_2CTA) {
            bwd_pair_tma_wait_broadcast<true>(
                &q_full[1], role_phase, cta_rank, lane);
            if (lead) {
              tcgen05_fence_after_thread_sync();
            }
          }
          bwd_pipeline_consumer_wait<USE_2CTA>(
              smem_ptr_u32(&role_bar[1]), role_phase);
          if constexpr (!USE_2CTA) {
            #pragma unroll
            for (int dim_subtile = 0; dim_subtile < 2; ++dim_subtile) {
              #pragma unroll
              for (int ki = 0; ki < M_TILE / 16; ++ki) {
              bwd_mma_f16_ts_lead<USE_2CTA>(
                  lead, tmem_base + 384 + dim_subtile * 64,
                  tmem_base + 256 + 8 * ki,
                  desc_q + dim_subtile * A_TILE_DELTA + 0x80 * ki,
                  idesc_dkv_ts, m_start != m_begin || ki != 0);
              }
            }
          } else {
            #pragma unroll
            for (int atom = 0; atom < 8; ++atom) {
              bwd_mma_f16_ts_lead<true>(
                  lead, tmem_base + 384,
                  tmem_base + 256 + 8 * atom,
                  desc_q + 0x80 * atom, idesc_dkv_ts,
                  m_start != m_begin || atom != 0);
            }
          }
          if (is_last_m) {
            bwd_pipeline_consumer_wait<USE_2CTA>(
                smem_ptr_u32(&dkv_empty[1]), 1);
            bwd_mma_commit_lead<USE_2CTA>(
                lead, smem_ptr_u32(&dkv_full[1]));
          } else if constexpr (!USE_2CTA) {
            bwd_mma_commit_lead<USE_2CTA>(
                lead, smem_ptr_u32(&q_empty[q_stage]));
          }
          if constexpr (USE_2CTA) {
            if (lead) {
              tcgen05_commit_multicast<2>(
                  smem_ptr_u32(&q_empty[1]), 0x3);
            }

            // Dense FA4 main-loop order is dK(cur), dP(next), dQ(cur).
            // dK has consumed packed dS from TMEM, so dP(next) can replace
            // it while dQ consumes the independently assembled SMEM dS.
            if (has_next_m) {
              const uint32_t next_load_phase =
                  static_cast<uint32_t>((consumer_iter_u + 1) & 1);
              bwd_pair_tma_wait_broadcast<true>(
                  load_full, next_load_phase, cta_rank, lane);
              if (lead) {
                tcgen05_fence_after_thread_sync();
              }
              #pragma unroll
              for (int dim_subtile = 0; dim_subtile < 2; ++dim_subtile) {
                #pragma unroll
                for (int ki = 0; ki < 4; ++ki) {
                  bwd_mma_f16_ss_lead<true>(
                      lead, tmem_base + 256,
                      desc_v + dim_subtile * A_TILE_DELTA + 2 * ki,
                      desc_do + dim_subtile * B_TILE_DELTA + 2 * ki,
                      idesc_score, dim_subtile != 0 || ki != 0);
                }
              }
              bwd_mma_commit_lead<true>(
                  lead, smem_ptr_u32(&mma_full[1]));
            }
          }

          // dQ reuses the dP/dS TMEM columns. Same-warp issue ordering makes
          // the preceding dK TS consume packed dS before dQ overwrites it.
          if constexpr (!USE_2CTA) {
            #pragma unroll
            for (int dim_subtile = 0; dim_subtile < 2; ++dim_subtile) {
              #pragma unroll
              for (int key_subtile = 0; key_subtile < 2; ++key_subtile) {
                #pragma unroll
                for (int ki = 0; ki < 4; ++ki) {
                  bwd_mma_f16_ss_lead<false>(
                      lead, tmem_base + 256 + dim_subtile * 64,
                      desc_ds_row + key_subtile * A_TILE_DELTA + 2 * ki,
                      desc_k + dim_subtile * A_TILE_DELTA +
                          key_subtile * B_TILE_DELTA + 0x80 * ki,
                      idesc_dq, key_subtile != 0 || ki != 0);
                }
              }
            }
            bwd_mma_commit_lead<false>(lead,
                                    smem_ptr_u32(&mma_full[2]));
          } else {
            // FA4 dQ: dS @ K is a cta_group::2 SS MMA. The compute/relay
            // protocol assembles this CTA's 64 query rows across both
            // 128-key peers in t_ds. The instruction consumes one 128-key
            // contribution per CTA and reduces them into the 128x128 dQ
            // tile at FA4's TMEM offset 64.
            // dQ occupies TMEM columns 64..191, overlapping S.  FA4's
            // PipelineAsyncUmma dS producer commits only after both compute
            // teams have drained S(next), or at the compute tail.
            bwd_pipeline_consumer_wait<USE_2CTA>(smem_ptr_u32(ds_full),
                                                 role_phase);
            if (cta_rank == 0) {
              bwd_pipeline_consumer_wait<USE_2CTA>(
                  smem_ptr_u32(ds_cluster_leader), role_phase);
              // FA4's main-loop acquire happens before S(next), since S and
              // dQ alias TMEM.  The tail has no S(next), so it performs the
              // same dQ-empty acquire immediately before the final dQ MMA.
              if (!has_next_m) {
                bwd_pipeline_consumer_wait<USE_2CTA>(
                    smem_ptr_u32(&role_bar[2]), 1u ^ role_phase);
              }
              const uint32_t idesc_dq_2cta =
                  make_idesc_bf16_f32(M_TILE, HEAD_DIM, true, true);
              const uint64_t desc_ds_dq = build_smem_desc_blackwell(
                  smem_ptr_u32(t_ds), DESC_SBO, DESC_LBO_MN,
                  SmemSwizzleBlackwell::B128);
              // CuTe's exact dQ fragments have K-mode 16 and descriptor
              // strides {0, 0x80, ..., 0x780}: the full 256-key cluster
              // reduction, split into two eight-instruction unroll groups.
              #pragma unroll
              for (int atom = 0; atom < 16; ++atom) {
                bwd_mma_f16_ss_lead<true>(
                    lead, tmem_base + 64,
                    desc_ds_dq + 0x80 * atom,
                    desc_kt + 0x80 * atom,
                    idesc_dq_2cta, atom != 0);
              }
              bwd_mma_commit_lead<true>(lead,
                                      smem_ptr_u32(&mma_full[2]));
              if (lead) {
                // pipeline_dS.consumer_release: retire the dQ shared-memory
                // read before releasing the one-stage dS buffers in both
                // producer CTAs.
                tcgen05_commit_multicast<2>(smem_ptr_u32(ds_empty), 0x3);
              }
            }
          }

          // Preserve the accepted 1-CTA schedule. Its dP(next) remains after
          // dQ(cur); the strict FA4 2-CTA order is issued above.
          if constexpr (!USE_2CTA) {
          if (has_next_m) {
            const int next_iter = consumer_iter_u + 1;
            const uint32_t next_dp_empty_phase =
                1u ^ static_cast<uint32_t>(next_iter & 1);
            bwd_pipeline_consumer_wait<USE_2CTA>(
                smem_ptr_u32(&role_bar[2]), next_dp_empty_phase);
            const uint32_t next_load_phase =
                static_cast<uint32_t>(next_iter & 1);
            bwd_pair_tma_wait_broadcast<false>(
                load_full, next_load_phase, cta_rank, lane);
            #pragma unroll
            for (int dim_subtile = 0; dim_subtile < 2; ++dim_subtile) {
              #pragma unroll
              for (int ki = 0; ki < 4; ++ki) {
                bwd_mma_f16_ss_lead<false>(
                    lead, tmem_base + 256,
                    desc_v + dim_subtile * A_TILE_DELTA + 2 * ki,
                    desc_do + dim_subtile * A_TILE_DELTA + 2 * ki,
                    idesc_score, dim_subtile != 0 || ki != 0);
              }
            }
            bwd_mma_commit_lead<false>(
                lead, smem_ptr_u32(&mma_full[1]));
          }
          }
        }

        if constexpr (!USE_2CTA) {
        if (warp < 4) {
          bwd_pipeline_consumer_wait<USE_2CTA>(
              smem_ptr_u32(&mma_full[2]), mma_phase);
        }

        float* t_dq_smem =
            reinterpret_cast<float*>(t_dq_smem_storage);
        if (warp >= 0 && warp <= 3) {
          const int row_group = warp;
          const int query_local = row_group * 32 + lane;
          const int query = m_start + query_local;
          const uint32_t row_addr =
              static_cast<uint32_t>(row_group * 32) << 16;
          // FA4's four reduce warps own the complete TMEM->SMEM->bulk-add
          // pipeline.  Each thread publishes 128 bits per store and warp 0
          // issues the two-stage TMA reduce-add stream.
          #pragma unroll
          for (int chunk = 0; chunk < HEAD_DIM / 32; ++chunk) {
            const int col = chunk * 32;
            const int stage = chunk & 1;
            uint32_t dq_values[32];
            tcgen05_ld_32x32b_x32(
                tmem_base + 256 + row_addr + col, dq_values);
            tcgen05_wait_ld();
            float* t_dq_stage = t_dq_smem + stage * M_TILE * 32;
            #pragma unroll
            for (int j = 0; j < 32; j += 4) {
              const int logical = query_local * 32 + j;
              const int physical =
                  static_cast<int>(smem_swizzle_b128(
                                       logical * sizeof(float)) /
                                   sizeof(float));
              uint4 values;
              if constexpr (STATIC_SEQLEN != 0) {
                values = make_uint4(dq_values[j], dq_values[j + 1],
                                    dq_values[j + 2], dq_values[j + 3]);
              } else {
                values = query < seq
                    ? make_uint4(dq_values[j], dq_values[j + 1],
                                 dq_values[j + 2], dq_values[j + 3])
                    : make_uint4(0, 0, 0, 0);
              }
              *reinterpret_cast<uint4*>(t_dq_stage + physical) = values;
            }
            fence_proxy_async_shared_cta();
            bar_sync<3>(4 * 32);
            if (warp == 0 && lane == 0) {
              const CUtensorMap* dq_map = tmap_dq + head_linear;
              tma_store_2d_add(dq_map, col, m_start,
                               smem_ptr_u32(t_dq_stage));
              cp_async_bulk_commit_group();
              cp_async_bulk_wait_group_read<1>();
            }
            bar_sync<3>(4 * 32);
          }
          tcgen05_fence_before_thread_sync();
          __syncwarp();
          if (lane == 0) {
            mbarrier_arrive_release_cta(smem_ptr_u32(&role_bar[2]));
          }
          if (warp == 0 && lane == 0) {
            cp_async_bulk_wait_group_read<0>();
          }
          bar_sync<3>(4 * 32);
        }
        } else {
          // REDUCE WARPS (0-3): FA4's 2CTA dQ path drains the CTA-owned
          // 128x64 FP32 TMEM tile, reinterprets it as eight contiguous
          // 128x8 stages, and issues four-stage pipelined flat bulk-reduce
          // operations directly into dQaccum.
          if (DENSE_ROLE == 2) {
            bwd_pipeline_consumer_wait<USE_2CTA>(
                smem_ptr_u32(&mma_full[2]), mma_phase);
            const int reduce_tidx = warp * 32 + lane;
            const int row_group = warp;
            const uint32_t row_addr =
                static_cast<uint32_t>(row_group * 32) << 16;
            uint32_t dq_regs[2][32];
            #pragma unroll
            for (int chunk = 0; chunk < 2; ++chunk) {
              tcgen05_ld_32x32b_x32(
                  tmem_base + 64 + row_addr + chunk * 32,
                  dq_regs[chunk]);
            }
            // FA4's tiled TMEM->RMEM copy issues both 32-register atoms,
            // then fences the complete copy once before releasing dQ TMEM.
            tcgen05_wait_ld();
            tcgen05_fence_before_thread_sync();
            __syncwarp();
            if (lane == 0) {
              const uint32_t leader_dq_empty = mapa_shared_cluster_u32(
                  smem_ptr_u32(&role_bar[2]), 0);
              mbarrier_arrive_cluster_default(leader_dq_empty);
            }

            float* t_dq_stages = reinterpret_cast<float*>(
                t_dq_reduce_2cta);
            #pragma unroll
            for (int stage = 0; stage < 8; ++stage) {
              const int chunk = stage >> 2;
              const int value_begin = (stage & 3) * 8;
              const float* values =
                  reinterpret_cast<const float*>(dq_regs[chunk]);
              float* stage_dst =
                  t_dq_stages + (stage & 3) * (M_TILE * 8);
              #pragma unroll
              for (int j = 0; j < 4; ++j) {
                stage_dst[reduce_tidx * 4 + j] = values[value_begin + j];
                stage_dst[(M_TILE / 2) * 8 + reduce_tidx * 4 + j] =
                    values[value_begin + 4 + j];
              }
              fence_proxy_async_shared_cta();
              bar_sync<3>(4 * 32);
              if (warp == 0) {
                constexpr uint32_t DQ_STAGE_BYTES =
                    M_TILE * 8 * sizeof(float);
                float* global_stage =
                    dq_accum + bh_base +
                    static_cast<std::size_t>(m_start + cta_rank * 64) *
                        HEAD_DIM +
                    static_cast<std::size_t>(stage) * M_TILE * 8;
                const uint32_t reduce_lead = elect_one_sync();
                if (reduce_lead) {
                  cpasync_reduce_bulk_add_f32(
                      global_stage,
                      smem_ptr_u32(t_dq_stages +
                                   (stage & 3) * (M_TILE * 8)),
                      DQ_STAGE_BYTES);
                }
                cp_async_bulk_commit_group();
                cp_async_bulk_wait_group_read<3>();
              }
              bar_sync<3>(4 * 32);
            }
          }
        }

      }
      // FA4 drains the four-stage dQ bulk-reduce ring once after the complete
      // M loop.  The two named barriers inside each stage protect SMEM reuse;
      // this final barrier only makes the last outstanding group visible
      // before the reduce warps leave the role.
      if constexpr (USE_2CTA && DENSE_ROLE == 2) {
        if (warp == 0) {
          cp_async_bulk_wait_group_read<0>();
        }
        bar_sync<3>(4 * 32);
      }
      // FA4 performs the D128 2CTA compute tail commit once after the M loop.
      // The main-loop lookahead commits dS(previous) at the start of each
      // subsequent iteration; this closes the final stage with the same one
      // lane per compute warp, two-CTA (16-arrival) barrier contract.
      if constexpr (USE_2CTA && DENSE_ROLE == 1) {
        if (lane == 0) {
          const uint32_t leader_ds_full = mapa_shared_cluster_u32(
              smem_ptr_u32(ds_full), 0);
          mbarrier_arrive_cluster_default(leader_ds_full);
        }
      }
    };
    // COMPUTE WARPS (4-11): drain all 128 dV/dK columns in two 32-column
    // slices per warp, then repack the results into the dead V/K operands.
    // Keep this epilogue callable from the same top-level role branch as the
    // compute main loop. FA4 never reconverges compute with MMA/reduce before
    // the role-local epilogue and TmemAllocator arrival.
    auto compute_epilogue = [&]() {
      const int row_group = (warp - 4) & 3;
      const int key_local = row_group * 32 + lane;
      const uint32_t row_addr =
          static_cast<uint32_t>(row_group * 32) << 16;
      const int col_begin = warp < 8 ? 32 : 0;

      mbarrier_wait_parity(smem_ptr_u32(&dkv_full[0]), 0);
      for (int chunk = 0; chunk < 2; ++chunk) {
        const int col = col_begin + chunk * 64;
        uint32_t dv_regs[32];
        uint32_t dv_packed[16];
        tcgen05_ld_32x32b_x32(
            tmem_base + 128 + row_addr + col, dv_regs);
        tcgen05_wait_ld();
        float* dv_values = reinterpret_cast<float*>(dv_regs);
        #pragma unroll
        for (int j = 0; j < 32; j += 2) {
          dv_packed[j / 2] =
              cvt_f32x2_to_bf16x2(dv_values[j], dv_values[j + 1]);
        }
        #pragma unroll
        for (int vec = 0; vec < 4; ++vec) {
          const int logical =
              key_local * 64 + (col & 63) + vec * 8;
          const int physical =
              (col / 64) * (K_TILE * 64) +
              static_cast<int>(smem_swizzle_b128(
                                   logical * sizeof(__nv_bfloat16)) /
                               sizeof(__nv_bfloat16));
          const int reg = vec * 4;
          const uint4 values = make_uint4(
              dv_packed[reg], dv_packed[reg + 1],
              dv_packed[reg + 2], dv_packed[reg + 3]);
          *reinterpret_cast<uint4*>(t_v + physical) = values;
        }
      }
      tcgen05_fence_before_thread_sync();
      __syncwarp();
      if (lane == 0) {
        mbarrier_arrive_release_cta(smem_ptr_u32(&dkv_empty[0]));
      }
      if constexpr (USE_2CTA) {
        fence_proxy_async_shared_cta();
        bar_sync<4>(8 * 32);
        if (warp == 4 && lane == 0) {
          tma_store_3d(&tmap_dv, 0, key_start, head_linear,
                       smem_ptr_u32(t_v));
          tma_store_3d(&tmap_dv, 64, key_start, head_linear,
                       smem_ptr_u32(t_v + K_TILE * 64));
          cp_async_bulk_commit_group();
          cp_async_bulk_wait_group<0>();
        }
        bar_sync<4>(8 * 32);
      } else {
        fence_proxy_async_shared_cta();
        bar_sync<4>(8 * 32);
        if (warp == 4 && lane == 0) {
          tma_store_3d(&tmap_dv, 0, key_start, head_linear,
                       smem_ptr_u32(t_v));
          tma_store_3d(&tmap_dv, 64, key_start, head_linear,
                       smem_ptr_u32(t_v + K_TILE * 64));
          cp_async_bulk_commit_group();
        }
      }

      mbarrier_wait_parity(smem_ptr_u32(&dkv_full[1]), 0);
      for (int chunk = 0; chunk < 2; ++chunk) {
        const int col = col_begin + chunk * 64;
        uint32_t dk_regs[32];
        uint32_t dk_packed[16];
        tcgen05_ld_32x32b_x32(
            tmem_base + 384 + row_addr + col, dk_regs);
        tcgen05_wait_ld();
        float* dk_values = reinterpret_cast<float*>(dk_regs);
        #pragma unroll
        for (int j = 0; j < 32; j += 2) {
          const float2 dk_pair = fmul2(
              make_float2(dk_values[j], dk_values[j + 1]),
              f32x2_splat(scale));
          dk_packed[j / 2] =
              cvt_f32x2_to_bf16x2(dk_pair.x, dk_pair.y);
        }
        #pragma unroll
        for (int vec = 0; vec < 4; ++vec) {
          const int logical =
              key_local * 64 + (col & 63) + vec * 8;
          const int physical =
              (col / 64) * (K_TILE * 64) +
              static_cast<int>(smem_swizzle_b128(
                                   logical * sizeof(__nv_bfloat16)) /
                               sizeof(__nv_bfloat16));
          const int reg = vec * 4;
          const uint4 values = make_uint4(
              dk_packed[reg], dk_packed[reg + 1],
              dk_packed[reg + 2], dk_packed[reg + 3]);
          *reinterpret_cast<uint4*>(t_k + physical) = values;
        }
      }
      tcgen05_fence_before_thread_sync();
      __syncwarp();
      if (lane == 0) {
        mbarrier_arrive_release_cta(smem_ptr_u32(&dkv_empty[1]));
      }
      if constexpr (USE_2CTA) {
      fence_proxy_async_shared_cta();
      bar_sync<4>(8 * 32);
      if (warp == 4 && lane == 0) {
        tma_store_3d(&tmap_dk, 0, key_start, head_linear,
                     smem_ptr_u32(t_k));
        tma_store_3d(&tmap_dk, 64, key_start, head_linear,
                     smem_ptr_u32(t_k + K_TILE * 64));
        cp_async_bulk_commit_group();
        cp_async_bulk_wait_group<0>();
      }
      bar_sync<4>(8 * 32);
      } else {
      fence_proxy_async_shared_cta();
      bar_sync<4>(8 * 32);
      if (warp == 4 && lane == 0) {
        tma_store_3d(&tmap_dk, 0, key_start, head_linear,
                     smem_ptr_u32(t_k));
        tma_store_3d(&tmap_dk, 64, key_start, head_linear,
                     smem_ptr_u32(t_k + K_TILE * 64));
        cp_async_bulk_commit_group();
      }
      }
    };

    // Match FA4's top-level warp-role dispatch: each role owns its complete
    // loop and tail and returns without reconverging with a different
    // setmaxnreg budget. This is required for the compute warps to use the
    // donated 136-register allocation for P, dP, and delayed dS exchange.
    if constexpr (USE_2CTA) {
      if (warp == 12) {
        setmaxnreg_dec<104>();
        tcgen05_alloc<2>(smem_ptr_u32(tmem_slot), 512);
        tcgen05_relinquish_alloc_permit<2>();
        bar_sync<10>(13 * 32);
        tmem_base = *tmem_slot;
        consumer_loop(std::integral_constant<int, 0>{});
        bar_sync<10>(13 * 32);
        // Exact CuTe TmemAllocator 2CTA free protocol: every allocator-warp
        // lane arrives on the peer's count-32 barrier, then waits locally
        // before issuing the cta_group::2 deallocation.
        const uint32_t peer_dealloc_bar = mapa_shared_cluster_u32(
            smem_ptr_u32(tmem_dealloc_bar), cta_rank ^ 1u);
        mbarrier_arrive_cluster_default(peer_dealloc_bar);
        mbarrier_wait_parity(smem_ptr_u32(tmem_dealloc_bar), 0);
        tcgen05_dealloc<CTA_GROUP>(tmem_base, 512);
        return;
      } else if (warp >= 4 && warp <= 11) {
        setmaxnreg_inc<136>();
        bar_sync<10>(13 * 32);
        tmem_base = *tmem_slot;
        consumer_loop(std::integral_constant<int, 1>{});
        compute_epilogue();
        bar_sync<10>(13 * 32);
        return;
      } else if (warp <= 3) {
        setmaxnreg_inc<136>();
        bar_sync<10>(13 * 32);
        tmem_base = *tmem_slot;
        consumer_loop(std::integral_constant<int, 2>{});
        bar_sync<10>(13 * 32);
        return;
      }
    } else {
      consumer_loop(std::integral_constant<int, -1>{});
      if (warp >= 4 && warp <= 11) {
        compute_epilogue();
      }
      // Exactly the 13 TMEM users close the allocation lifetime.  Load and
      // inert roles have already returned and never join this barrier.
      bar_sync<10>(13 * 32);
      if (warp == 12) {
        tcgen05_dealloc<1>(tmem_base, 512);
      }
      return;
    }
  }

  // LOAD WARP (13): load the work item's owned K/V tile once.
  if (warp == 13) {
    for (int idx = lane; idx < K_TILE_ELEMENTS; idx += 32) {
      const int key_local = idx / HEAD_DIM;
      const int dim = idx - key_local * HEAD_DIM;
      const int key = key_start + key_local;
      if (key < seq) {
        const std::size_t global = bh_base +
                                   static_cast<std::size_t>(key) * HEAD_DIM +
                                   dim;
        s_k[idx] = k[global];
        s_v[idx] = v[global];
      } else {
        s_k[idx] = __float2bfloat16_rn(0.0f);
        s_v[idx] = __float2bfloat16_rn(0.0f);
      }
    }
  }

  // COMPUTE WARPS (4-11): initialize private dK/dV accumulators.
  if (warp >= 4 && warp <= 11) {
    const int compute_tid = (warp - 4) * 32 + lane;
    for (int idx = compute_tid; idx < K_TILE_ELEMENTS; idx += 256) {
      s_dk[idx] = 0.0f;
      s_dv[idx] = 0.0f;
    }
  }
  __syncthreads();

  const int q_begin = IS_CAUSAL ? key_start : 0;
  for (int query = q_begin; query < seq; ++query) {
    // LOAD WARP (13): publish Q, dO, LSE_log2, and D for one query row.
    if (warp == 13) {
      for (int dim = lane; dim < HEAD_DIM; dim += 32) {
        const std::size_t global =
            bh_base + static_cast<std::size_t>(query) * HEAD_DIM + dim;
        s_q[dim] = q[global];
        s_dout[dim] = dout[global];
      }
      if (lane == 0) {
        s_lse_log2[0] = lse_log2[row_base + query];
        s_d[0] = d[row_base + query];
      }
    }
    __syncthreads();

    // MMA WARP (12): correctness producer for S and dP. These scalar dot
    // products are the exact region replaced by tcgen05 in the next stage.
    if (warp == 12) {
      for (int key_local = lane; key_local < K_TILE; key_local += 32) {
        const int key = key_start + key_local;
        float score = 0.0f;
        float dp = 0.0f;
        if (key < seq) {
          const int tile_base = key_local * HEAD_DIM;
          for (int dim = 0; dim < HEAD_DIM; ++dim) {
            score += __bfloat162float(s_q[dim]) *
                     __bfloat162float(s_k[tile_base + dim]);
            dp += __bfloat162float(s_dout[dim]) *
                  __bfloat162float(s_v[tile_base + dim]);
          }
        }
        s_score[key_local] = score * scale;
        s_dp[key_local] = dp;
      }
    }
    __syncthreads();

    // COMPUTE WARPS (4-11): reconstruct P, compute dS, and update the
    // privately owned dK/dV tiles.
    if (warp >= 4 && warp <= 11) {
      const int compute_tid = (warp - 4) * 32 + lane;
      if (compute_tid < K_TILE) {
        const int key = key_start + compute_tid;
        const bool valid = key < seq && (!IS_CAUSAL || key <= query);
        const float prob =
            valid ? ex2_approx_f32(s_score[compute_tid] * LOG2_E -
                                   s_lse_log2[0])
                  : 0.0f;
        s_p[compute_tid] = prob;
        s_ds[compute_tid] = prob * (s_dp[compute_tid] - s_d[0]);
      }
    }
    __syncthreads();

    if (warp >= 4 && warp <= 11) {
      const int compute_tid = (warp - 4) * 32 + lane;
      for (int idx = compute_tid; idx < K_TILE_ELEMENTS; idx += 256) {
        const int key_local = idx / HEAD_DIM;
        const int dim = idx - key_local * HEAD_DIM;
        s_dk[idx] += s_ds[key_local] * __bfloat162float(s_q[dim]) *
                     scale;
        s_dv[idx] += s_p[key_local] * __bfloat162float(s_dout[dim]);
      }
    }
    __syncthreads();

    // REDUCE WARPS (0-3): one FP32 reduction per dQ element, then one global
    // atomic per key tile. Preprocess has already zeroed dQaccum.
    if (warp >= 0 && warp <= 3) {
      const int reduce_tid = warp * 32 + lane;
      if (reduce_tid < HEAD_DIM) {
        float dq_partial = 0.0f;
        for (int key_local = 0; key_local < K_TILE; ++key_local) {
          dq_partial +=
              s_ds[key_local] *
              __bfloat162float(s_k[key_local * HEAD_DIM + reduce_tid]);
        }
        const std::size_t global =
            bh_base + static_cast<std::size_t>(query) * HEAD_DIM + reduce_tid;
        atom_global_add_f32(dq_accum + global, dq_partial);
      }
    }
    __syncthreads();
  }

  // COMPUTE WARPS (4-11): private dK/dV epilogue. Each work item owns these
  // rows, so no global reduction is needed.
  if (warp >= 4 && warp <= 11) {
    const int compute_tid = (warp - 4) * 32 + lane;
    constexpr int N_PAIRS = K_TILE * (HEAD_DIM / 2);
    for (int pair = compute_tid; pair < N_PAIRS; pair += 256) {
      const int key_local = pair / (HEAD_DIM / 2);
      const int dim_pair = pair - key_local * (HEAD_DIM / 2);
      const int key = key_start + key_local;
      if (key < seq) {
        const int local = key_local * HEAD_DIM + 2 * dim_pair;
        const std::size_t global =
            bh_base + static_cast<std::size_t>(key) * HEAD_DIM +
            2 * dim_pair;
        reinterpret_cast<uint32_t*>(dk + global)[0] =
            cvt_f32x2_to_bf16x2(s_dk[local], s_dk[local + 1]);
        reinterpret_cast<uint32_t*>(dv + global)[0] =
            cvt_f32x2_to_bf16x2(s_dv[local], s_dv[local + 1]);
      }
    }
  }
}

// Fixed dense FA4 specialization used by the selected 1CTA optimization
// targets.  The grid is generic in B*H; sequence length, head dimension, and
// algorithmic mode are static.
//
// This is deliberately a separate specialization rather than another branch
// in fmha_bwd_main_kernel.  FA4's setmaxnreg contract requires each role to be
// a top-level branch with an independent return; otherwise ptxas merges the
// compute/MMA live ranges and spills at the 128-register entry budget.
enum DenseD128Bar : int {
  D128_Q_FULL0,
  D128_Q_FULL1,
  D128_Q_EMPTY0,
  D128_Q_EMPTY1,
  D128_DO_FULL,
  D128_DO_EMPTY,
  D128_LSE_FULL0,
  D128_LSE_FULL1,
  D128_LSE_EMPTY0,
  D128_LSE_EMPTY1,
  D128_DPS_FULL,
  D128_DPS_EMPTY,
  D128_S_READY,
  D128_DP_READY,
  D128_P_STTMD,
  D128_DS_READY,
  D128_DQ_FULL,
  D128_DQ_FREE,
  D128_DV_FULL,
  D128_DK_FULL,
  D128_NUM_BARS
};

union DenseD128SmemDesc {
  uint64_t u64;
  uint2 u32;
};

template <int DIM, int SEQ, bool IS_CAUSAL = false>
__global__ void __maxnreg__(128)
fmha_bwd_1cta_dense_kernel(
    const __grid_constant__ CUtensorMap tmap_q,
    const __grid_constant__ CUtensorMap tmap_k,
    const __grid_constant__ CUtensorMap tmap_v,
    const __grid_constant__ CUtensorMap tmap_dout,
    const __grid_constant__ CUtensorMap tmap_dk,
    const __grid_constant__ CUtensorMap tmap_dv,
    const CUtensorMap* __restrict__ tmap_dq,
    const float* __restrict__ lse_log2,
    const float* __restrict__ d,
    float* __restrict__ dq_accum) {
  static_assert(DIM == 64 || DIM == 128);
  static_assert(SEQ > 0 && SEQ % M_TILE == 0);
  constexpr int SUB = 64;
  constexpr int STEPS = SEQ / M_TILE;
  constexpr float SOFTMAX_SCALE =
      DIM == 64 ? 0.125f : 0.08838834764831845f;
  // Exact FA4 1CTA register partition. Causal shifts eight registers from
  // the dQ-reduce role to each compute warp; noncausal keeps the wider
  // reduction epilogue budget.
  constexpr int NUM_REGS_COMPUTE = IS_CAUSAL ? 144 : 136;
  constexpr int NUM_REGS_REDUCE = IS_CAUSAL ? 136 : 152;
  constexpr uint32_t TILE_BYTES = M_TILE * DIM * sizeof(__nv_bfloat16);
  constexpr uint32_t SUBTILE_BYTES = M_TILE * SUB * sizeof(__nv_bfloat16);

  constexpr uint32_t T_S = 0;
  constexpr uint32_t T_DV = 128;
  constexpr uint32_t T_DP = T_DV + DIM;
  constexpr uint32_t T_DK = T_DP + M_TILE;
  constexpr uint32_t T_P_BF16 = T_S;
  constexpr uint32_t T_DS_BF16 = T_DP;
  constexpr uint32_t T_DQ = T_DP;

  extern __shared__ __align__(1024) unsigned char d128_raw[];
  __nv_bfloat16* s_k = reinterpret_cast<__nv_bfloat16*>(d128_raw);
  __nv_bfloat16* s_v = s_k + M_TILE * DIM;
  __nv_bfloat16* s_q[2] = {
      s_v + M_TILE * DIM,
      s_v + 2 * M_TILE * DIM};
  __nv_bfloat16* s_do = s_q[1] + M_TILE * DIM;
  __nv_bfloat16* s_ds = s_do + M_TILE * DIM;
  float* s_dq[2] = {
      reinterpret_cast<float*>(s_ds + M_TILE * M_TILE),
      reinterpret_cast<float*>(s_ds + M_TILE * M_TILE) + M_TILE * 32};
  float* s_lse = s_dq[1] + M_TILE * 32;
  float* s_dpsum = s_lse + 2 * M_TILE;
  uint64_t* bars = reinterpret_cast<uint64_t*>(s_dpsum + M_TILE);
  uint32_t* tmem_slot = reinterpret_cast<uint32_t*>(bars + D128_NUM_BARS);

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_raw = tid >> 5;
  if (warp_raw == 13 && lane == 0) {
    prefetch_tensormap(&tmap_q);
    prefetch_tensormap(&tmap_k);
    prefetch_tensormap(&tmap_v);
    prefetch_tensormap(&tmap_dout);
    prefetch_tensormap(&tmap_dv);
    prefetch_tensormap(&tmap_dk);
  }
  const int warp = __shfl_sync(0xffffffffu, warp_raw, 0);
  const int n_block = static_cast<int>(blockIdx.x) & (STEPS - 1);
  const int head_linear = static_cast<int>(blockIdx.x) / STEPS;
  const int key_start = n_block * M_TILE;
  const int m_block_begin = IS_CAUSAL ? n_block : 0;
  const int q_steps = STEPS - m_block_begin;
  const std::size_t row_base =
      static_cast<std::size_t>(head_linear) * SEQ;

  if (tid == 0) {
    mbarrier_init(smem_ptr_u32(&bars[D128_Q_FULL0]), 1);
    mbarrier_init(smem_ptr_u32(&bars[D128_Q_FULL1]), 1);
    mbarrier_init(smem_ptr_u32(&bars[D128_Q_EMPTY0]), 1);
    mbarrier_init(smem_ptr_u32(&bars[D128_Q_EMPTY1]), 1);
    mbarrier_init(smem_ptr_u32(&bars[D128_DO_FULL]), 1);
    mbarrier_init(smem_ptr_u32(&bars[D128_DO_EMPTY]), 1);
    mbarrier_init(smem_ptr_u32(&bars[D128_LSE_FULL0]), 1);
    mbarrier_init(smem_ptr_u32(&bars[D128_LSE_FULL1]), 1);
    mbarrier_init(smem_ptr_u32(&bars[D128_LSE_EMPTY0]), 8);
    mbarrier_init(smem_ptr_u32(&bars[D128_LSE_EMPTY1]), 8);
    mbarrier_init(smem_ptr_u32(&bars[D128_DPS_FULL]), 1);
    mbarrier_init(smem_ptr_u32(&bars[D128_DPS_EMPTY]), 8);
    mbarrier_init(smem_ptr_u32(&bars[D128_S_READY]), 1);
    mbarrier_init(smem_ptr_u32(&bars[D128_DP_READY]), 1);
    mbarrier_init(smem_ptr_u32(&bars[D128_P_STTMD]), 8);
    mbarrier_init(smem_ptr_u32(&bars[D128_DS_READY]), 8);
    mbarrier_init(smem_ptr_u32(&bars[D128_DQ_FULL]), 1);
    mbarrier_init(smem_ptr_u32(&bars[D128_DQ_FREE]), 4);
    mbarrier_init(smem_ptr_u32(&bars[D128_DV_FULL]), 1);
    mbarrier_init(smem_ptr_u32(&bars[D128_DK_FULL]), 1);
  }
  fence_mbarrier_init_release_cluster();
  __syncthreads();

  // FA4 warp 14 relay is inert for USE_2CTA=false; warp 15 is empty.
  if (warp > 13) {
    setmaxnreg_dec<24>();
    return;
  }

  if (warp == 13) {
    // LOAD role: Q/LSE are two-stage; dO/dPsum are one-stage.  K and V
    // piggyback on the first Q and dO transactions, respectively.
    setmaxnreg_dec<88>();
    EmptyPhaseTracker<2> q_empty, lse_empty;
    EmptyPhaseTracker<1> do_empty, dps_empty;
    #pragma unroll 1
    for (int j = 0; j < q_steps; ++j) {
      const int stage = q_empty.get_stage();
      const int m_start = (m_block_begin + j) * M_TILE;

      mbarrier_wait_parity_suspend(
          smem_ptr_u32(&bars[D128_Q_EMPTY0 + stage]),
          q_empty.get_phase());
      q_empty.advance();
      if (elect_one_sync()) {
        mbarrier_arrive_expect_tx(
            smem_ptr_u32(&bars[D128_Q_FULL0 + stage]),
            TILE_BYTES + (j == 0 ? TILE_BYTES : 0));
        #pragma unroll
        for (int dim_tile = 0; dim_tile < DIM / SUB; ++dim_tile) {
          tma_load_3d(
              smem_ptr_u32(s_q[stage] + dim_tile * M_TILE * SUB),
              &tmap_q, smem_ptr_u32(&bars[D128_Q_FULL0 + stage]),
              dim_tile * SUB, m_start, head_linear);
        }
        if (j == 0) {
          #pragma unroll
          for (int dim_tile = 0; dim_tile < DIM / SUB; ++dim_tile) {
            tma_load_3d(
                smem_ptr_u32(s_k + dim_tile * M_TILE * SUB), &tmap_k,
                smem_ptr_u32(&bars[D128_Q_FULL0 + stage]),
                dim_tile * SUB, key_start, head_linear);
          }
        }
      }

      const int lse_stage = lse_empty.get_stage();
      mbarrier_wait_parity_suspend(
          smem_ptr_u32(&bars[D128_LSE_EMPTY0 + lse_stage]),
          lse_empty.get_phase());
      lse_empty.advance();
      if (elect_one_sync()) {
        constexpr uint32_t BYTES = M_TILE * sizeof(float);
        mbarrier_arrive_expect_tx(
            smem_ptr_u32(&bars[D128_LSE_FULL0 + lse_stage]), BYTES);
        cuda::ptx::cp_async_bulk(
            cuda::ptx::space_shared, cuda::ptx::space_global,
            s_lse + lse_stage * M_TILE,
            lse_log2 + row_base + m_start, BYTES,
            &bars[D128_LSE_FULL0 + lse_stage]);
      }

      mbarrier_wait_parity_suspend(
          smem_ptr_u32(&bars[D128_DO_EMPTY]), do_empty.get_phase());
      do_empty.advance();
      if (elect_one_sync()) {
        mbarrier_arrive_expect_tx(
            smem_ptr_u32(&bars[D128_DO_FULL]),
            TILE_BYTES + (j == 0 ? TILE_BYTES : 0));
        #pragma unroll
        for (int dim_tile = 0; dim_tile < DIM / SUB; ++dim_tile) {
          tma_load_3d(smem_ptr_u32(s_do + dim_tile * M_TILE * SUB),
                      &tmap_dout, smem_ptr_u32(&bars[D128_DO_FULL]),
                      dim_tile * SUB, m_start, head_linear);
        }
        if (j == 0) {
          #pragma unroll
          for (int dim_tile = 0; dim_tile < DIM / SUB; ++dim_tile) {
            tma_load_3d(smem_ptr_u32(s_v + dim_tile * M_TILE * SUB),
                        &tmap_v, smem_ptr_u32(&bars[D128_DO_FULL]),
                        dim_tile * SUB, key_start, head_linear);
          }
        }
      }

      mbarrier_wait_parity_suspend(
          smem_ptr_u32(&bars[D128_DPS_EMPTY]), dps_empty.get_phase());
      dps_empty.advance();
      if (elect_one_sync()) {
        constexpr uint32_t BYTES = M_TILE * sizeof(float);
        mbarrier_arrive_expect_tx(
            smem_ptr_u32(&bars[D128_DPS_FULL]), BYTES);
        cuda::ptx::cp_async_bulk(
            cuda::ptx::space_shared, cuda::ptx::space_global,
            s_dpsum, d + row_base + m_start, BYTES,
            &bars[D128_DPS_FULL]);
      }
    }
    return;
  } else if (warp == 12) {
    // MMA role and TMEM owner.
    setmaxnreg_dec<88>();
    tcgen05_alloc<1>(smem_ptr_u32(tmem_slot), 512);
    bar_sync<10>(13 * 32);
    const uint32_t tmem_base = *tmem_slot;
    const uint32_t lead = elect_one_sync();

    constexpr uint32_t DESC_SBO = 1024;
    constexpr uint32_t DESC_LBO = 16;
    constexpr uint32_t DESC_LBO_MN = SUBTILE_BYTES;
    constexpr uint64_t SUB_DELTA = SUBTILE_BYTES >> 4;
    constexpr uint64_t Q_SLOT_DELTA = TILE_BYTES >> 4;
    constexpr uint32_t K16_MN = (16 * SUB * sizeof(__nv_bfloat16)) >> 4;

    const uint32_t idesc_st =
        make_idesc_bf16_f32(M_TILE, M_TILE, false, false);
    const uint32_t idesc_acc =
        make_idesc_bf16_f32(M_TILE, DIM, false, true);
    const uint32_t idesc_dq =
        make_idesc_bf16_f32(M_TILE, DIM, true, true);

    const uint64_t desc_k = build_smem_desc_blackwell(
        smem_ptr_u32(s_k), DESC_SBO, DESC_LBO,
        SmemSwizzleBlackwell::B128);
    const uint64_t desc_v = build_smem_desc_blackwell(
        smem_ptr_u32(s_v), DESC_SBO, DESC_LBO,
        SmemSwizzleBlackwell::B128);
    const uint64_t desc_q0 = build_smem_desc_blackwell(
        smem_ptr_u32(s_q[0]), DESC_SBO, DESC_LBO,
        SmemSwizzleBlackwell::B128);
    const uint64_t desc_do = build_smem_desc_blackwell(
        smem_ptr_u32(s_do), DESC_SBO, DESC_LBO,
        SmemSwizzleBlackwell::B128);
    const uint64_t desc_q0_mn = build_smem_desc_blackwell(
        smem_ptr_u32(s_q[0]), DESC_SBO, DESC_LBO_MN,
        SmemSwizzleBlackwell::B128);
    const uint64_t desc_do_mn = build_smem_desc_blackwell(
        smem_ptr_u32(s_do), DESC_SBO, DESC_LBO_MN,
        SmemSwizzleBlackwell::B128);
    const uint64_t desc_k_mn = build_smem_desc_blackwell(
        smem_ptr_u32(s_k), DESC_SBO, DESC_LBO_MN,
        SmemSwizzleBlackwell::B128);
    const uint64_t desc_ds_mn = build_smem_desc_blackwell(
        smem_ptr_u32(s_ds), DESC_SBO, DESC_LBO_MN,
        SmemSwizzleBlackwell::B128);

    auto issue_s = [&](int stage) {
      const uint64_t q_desc = desc_q0 + stage * Q_SLOT_DELTA;
      #pragma unroll
      for (int subtile = 0; subtile < DIM / SUB; ++subtile) {
        #pragma unroll
        for (int ki = 0; ki < 4; ++ki) {
          tcgen05_mma_f16_ss_lead(
              lead, tmem_base + T_S,
              desc_k + subtile * SUB_DELTA + 2 * ki,
              q_desc + subtile * SUB_DELTA + 2 * ki,
              idesc_st, subtile != 0 || ki != 0);
        }
      }
      tcgen05_commit1_lead(
          lead, smem_ptr_u32(&bars[D128_S_READY]));
    };
    auto issue_dp = [&]() {
      #pragma unroll
      for (int subtile = 0; subtile < DIM / SUB; ++subtile) {
        #pragma unroll
        for (int ki = 0; ki < 4; ++ki) {
          tcgen05_mma_f16_ss_lead(
              lead, tmem_base + T_DP,
              desc_v + subtile * SUB_DELTA + 2 * ki,
              desc_do + subtile * SUB_DELTA + 2 * ki,
              idesc_st, subtile != 0 || ki != 0);
        }
      }
      tcgen05_commit1_lead(
          lead, smem_ptr_u32(&bars[D128_DP_READY]));
    };
    auto issue_dv = [&](bool first) {
      DenseD128SmemDesc b;
      b.u64 = desc_do_mn;
      #pragma unroll
      for (int ki = 0; ki < 8; ++ki) {
        tcgen05_mma_f16_ts_1sm_lead(
            lead, tmem_base + T_DV,
            tmem_base + T_P_BF16 + ki * 8 + (ki >= 4 ? 32 : 0),
            b.u64, idesc_acc, !(first && ki == 0));
        smem_desc_add_lo(b.u64, K16_MN);
      }
      tcgen05_commit1_lead(
          lead, smem_ptr_u32(&bars[D128_DO_EMPTY]));
    };
    auto issue_dk = [&](int stage, bool first) {
      DenseD128SmemDesc b;
      b.u64 = desc_q0_mn + stage * Q_SLOT_DELTA;
      #pragma unroll
      for (int ki = 0; ki < 8; ++ki) {
        tcgen05_mma_f16_ts_1sm_lead(
            lead, tmem_base + T_DK,
            tmem_base + T_DS_BF16 + ki * 8 + (ki >= 4 ? 32 : 0),
            b.u64, idesc_acc, !(first && ki == 0));
        smem_desc_add_lo(b.u64, K16_MN);
      }
      tcgen05_commit1_lead(
          lead, smem_ptr_u32(&bars[D128_Q_EMPTY0 + stage]));
    };
    auto issue_dq = [&]() {
      DenseD128SmemDesc a;
      DenseD128SmemDesc b;
      a.u64 = desc_ds_mn;
      b.u64 = desc_k_mn;
      #pragma unroll
      for (int ki = 0; ki < 8; ++ki) {
        tcgen05_mma_f16_ss_lead(
            lead, tmem_base + T_DQ, a.u64, b.u64,
            idesc_dq, ki != 0);
        smem_desc_add_lo(a.u64, K16_MN);
        smem_desc_add_lo(b.u64, K16_MN);
      }
      tcgen05_commit1_lead(
          lead, smem_ptr_u32(&bars[D128_DQ_FULL]));
    };

    PhaseTracker<2> q_full;
    PhaseTracker<1> do_full;
    PhaseTracker<1> p_ready;
    PhaseTracker<1> ds_ready;
    PhaseTracker<1> dq_free;

    mbarrier_wait_parity(
        smem_ptr_u32(&bars[D128_Q_FULL0]), q_full.get_phase());
    issue_s(0);
    mbarrier_wait_parity(
        smem_ptr_u32(&bars[D128_DO_FULL]), do_full.get_phase());
    do_full.advance();
    issue_dp();
    mbarrier_wait_parity(
        smem_ptr_u32(&bars[D128_P_STTMD]), p_ready.get_phase());
    p_ready.advance();
    issue_dv(true);

    #pragma unroll 1
    for (int j = 0; j < q_steps; ++j) {
      const int stage = q_full.get_stage();
      q_full.advance();
      const int next_stage = q_full.get_stage();
      if (j + 1 < q_steps) {
        mbarrier_wait_parity(
            smem_ptr_u32(&bars[D128_Q_FULL0 + next_stage]),
            q_full.get_phase());
        issue_s(next_stage);
      }
      if (j + 1 == q_steps) {
        tcgen05_commit1_lead(
            lead, smem_ptr_u32(&bars[D128_DV_FULL]));
      }

      mbarrier_wait_parity(
          smem_ptr_u32(&bars[D128_DS_READY]), ds_ready.get_phase());
      ds_ready.advance();
      issue_dk(stage, j == 0);
      if (j + 1 == q_steps) {
        tcgen05_commit1_lead(
            lead, smem_ptr_u32(&bars[D128_DK_FULL]));
      }
      issue_dq();

      if (j + 1 < q_steps) {
        mbarrier_wait_parity(
            smem_ptr_u32(&bars[D128_DO_FULL]), do_full.get_phase());
        do_full.advance();
        mbarrier_wait_parity(
            smem_ptr_u32(&bars[D128_DQ_FREE]), dq_free.get_phase());
        dq_free.advance();
        issue_dp();
        mbarrier_wait_parity(
            smem_ptr_u32(&bars[D128_P_STTMD]), p_ready.get_phase());
        p_ready.advance();
        issue_dv(false);
      }
    }

    tcgen05_relinquish_alloc_permit<1>();
    bar_sync<10>(13 * 32);
    tcgen05_dealloc<1>(tmem_base, 512);
    return;
  } else if (warp >= 4) {
    // Two compute warpgroups.  Each owns one contiguous 64-column half,
    // exactly matching FA4's split_wg partition.
    setmaxnreg_inc<NUM_REGS_COMPUTE>();
    bar_sync<10>(13 * 32);
    const uint32_t tmem_base = *tmem_slot;
    const int compute_warp = warp - 4;
    const int row_group = compute_warp & 3;
    const int q_half = compute_warp >> 2;
    const uint32_t row_addr =
        static_cast<uint32_t>(row_group * 32) << 16;
    const uint32_t col_off = q_half * SUB;
    __nv_bfloat16* s_ds_half = s_ds + q_half * M_TILE * SUB;

    PhaseTracker<1> s_ready;
    PhaseTracker<1> dp_ready;
    PhaseTracker<1> dps_full;
    PhaseTracker<2> lse_full;
    #pragma unroll 1
    for (int j = 0; j < q_steps; ++j) {
      const int lse_stage = lse_full.get_stage();
      mbarrier_wait_parity_suspend(
          smem_ptr_u32(&bars[D128_LSE_FULL0 + lse_stage]),
          lse_full.get_phase());
      lse_full.advance();
      mbarrier_wait_parity_suspend(
          smem_ptr_u32(&bars[D128_S_READY]), s_ready.get_phase());
      s_ready.advance();

      uint32_t s_regs[64];
      uint32_t p_packed[32];
      float* p = reinterpret_cast<float*>(s_regs);
      tcgen05_ld_32x32b_x64(
          tmem_base + T_S + col_off + row_addr, s_regs);
      tcgen05_fence_before_thread_sync();
      const float* lse_ptr =
          s_lse + lse_stage * M_TILE + col_off;
      const float2 scale_log2 =
          f32x2_splat(SOFTMAX_SCALE * LOG2_E);
      if constexpr (IS_CAUSAL) {
        // FA4 derives the logical coordinate at the mask use. Do not retain
        // it across the 64-register T2R fragment.
        const int mask_row =
            row_group * 32 + static_cast<int>(lane_id());
        mask_s_row_transposed_r2p<SUB>(
            p, mask_row, j * M_TILE + col_off);
      }
      #pragma unroll
      for (int c = 0; c < SUB; c += 2) {
        const float2 z = ffma2(
            make_float2(p[c], p[c + 1]), scale_log2,
            make_float2(-lse_ptr[c], -lse_ptr[c + 1]));
        float p0 = ex2_approx_f32(z.x);
        float p1 = ex2_approx_f32(z.y);
        p[c] = p0;
        p[c + 1] = p1;
        p_packed[c / 2] = cvt_f32x2_to_bf16x2(p0, p1);
      }
      tcgen05_st_32x32b_x32(
          tmem_base + T_P_BF16 + col_off + row_addr, p_packed);
      tcgen05_wait_st();
      tcgen05_fence_before_thread_sync();
      if (elect_one_sync()) {
        mbarrier_arrive(smem_ptr_u32(&bars[D128_P_STTMD]));
        mbarrier_arrive(
            smem_ptr_u32(&bars[D128_LSE_EMPTY0 + lse_stage]));
      }

      mbarrier_wait_parity_suspend(
          smem_ptr_u32(&bars[D128_DPS_FULL]), dps_full.get_phase());
      dps_full.advance();
      mbarrier_wait_parity_suspend(
          smem_ptr_u32(&bars[D128_DP_READY]), dp_ready.get_phase());
      dp_ready.advance();

      uint32_t dp_regs[64];
      tcgen05_ld_32x32b_x64(
          tmem_base + T_DP + col_off + row_addr, dp_regs);
      tcgen05_fence_before_thread_sync();
      const float* dp = reinterpret_cast<const float*>(dp_regs);
      const float* d_ptr = s_dpsum + col_off;
      #pragma unroll
      for (int c = 0; c < SUB; c += 2) {
        const float2 ds2 = fmul2(
            make_float2(p[c], p[c + 1]),
            fsub2(make_float2(dp[c], dp[c + 1]),
                  make_float2(d_ptr[c], d_ptr[c + 1])));
        p_packed[c / 2] = cvt_f32x2_to_bf16x2(ds2.x, ds2.y);
      }
      tcgen05_st_32x32b_x32(
          tmem_base + T_DS_BF16 + col_off + row_addr, p_packed);
      const uint4* ds4 = reinterpret_cast<const uint4*>(p_packed);
      const int r2s_row =
          row_group * 32 + static_cast<int>(lane_id());
      #pragma unroll
      for (int v4 = 0; v4 < 8; ++v4) {
        *reinterpret_cast<uint4*>(
            &s_ds_half[r2s_row * SUB +
                       ((v4 ^ (r2s_row & 7)) * 8)]) = ds4[v4];
      }
      tcgen05_wait_st();
      tcgen05_fence_before_thread_sync();
      fence_proxy_async_shared_cta();
      if (elect_one_sync()) {
        mbarrier_arrive(smem_ptr_u32(&bars[D128_DS_READY]));
        mbarrier_arrive(smem_ptr_u32(&bars[D128_DPS_EMPTY]));
      }
    }

    const int epilogue_row =
        row_group * 32 + static_cast<int>(lane_id());
    auto drain_dkv = [&](uint32_t tmem_offset, float output_scale,
                         __nv_bfloat16* bounce,
                         const CUtensorMap* map) {
      __nv_bfloat16* bounce_half =
          bounce + q_half * M_TILE * SUB;
      #pragma unroll
      for (int c0 = 0; c0 < SUB; c0 += 32) {
        uint32_t acc_regs[32];
        tcgen05_ld_32x32b_x32(
            tmem_base + tmem_offset + col_off + c0 + row_addr,
            acc_regs);
        tcgen05_fence_before_thread_sync();
        const float* acc = reinterpret_cast<const float*>(acc_regs);
        #pragma unroll
        for (int v = 0; v < 4; ++v) {
          uint4 packed;
          const float2 x0 = fmul2(
              make_float2(acc[v * 8], acc[v * 8 + 1]),
              f32x2_splat(output_scale));
          const float2 x1 = fmul2(
              make_float2(acc[v * 8 + 2], acc[v * 8 + 3]),
              f32x2_splat(output_scale));
          const float2 x2 = fmul2(
              make_float2(acc[v * 8 + 4], acc[v * 8 + 5]),
              f32x2_splat(output_scale));
          const float2 x3 = fmul2(
              make_float2(acc[v * 8 + 6], acc[v * 8 + 7]),
              f32x2_splat(output_scale));
          packed.x = cvt_f32x2_to_bf16x2(x0.x, x0.y);
          packed.y = cvt_f32x2_to_bf16x2(x1.x, x1.y);
          packed.z = cvt_f32x2_to_bf16x2(x2.x, x2.y);
          packed.w = cvt_f32x2_to_bf16x2(x3.x, x3.y);
          const int vec = c0 / 8 + v;
          *reinterpret_cast<uint4*>(
              &bounce_half[epilogue_row * SUB +
                           ((vec ^ (epilogue_row & 7)) * 8)]) = packed;
        }
      }
      fence_proxy_async_shared_cta();
      if (q_half == 0) {
        bar_sync<12>(4 * 32);
      } else {
        bar_sync<13>(4 * 32);
      }
      if (compute_warp == q_half * 4 && elect_one_sync()) {
        tma_store_3d(map, q_half * SUB, key_start, head_linear,
                     smem_ptr_u32(bounce_half));
        cp_async_bulk_commit_group();
      }
    };

    mbarrier_wait_parity_suspend(
        smem_ptr_u32(&bars[D128_DV_FULL]), 0);
    if (q_half < DIM / SUB) {
      drain_dkv(T_DV, 1.0f, s_do, &tmap_dv);
    }
    mbarrier_wait_parity_suspend(
        smem_ptr_u32(&bars[D128_DK_FULL]), 0);
    if (q_half < DIM / SUB) {
      drain_dkv(T_DK, SOFTMAX_SCALE, s_q[0], &tmap_dk);
    }

    bar_sync<10>(13 * 32);
    return;
  } else {
    // Exact FA4 1CTA dQ reduction contract: the four reduce warps each own a
    // 32-column fragment across all four 32-row groups. Load the complete
    // TMEM tile before releasing its one-stage pipeline, then retile each
    // 32-row group into one contiguous 16-KiB buffer and issue the flat
    // cp.reduce.async.bulk add through a two-buffer ring.
    setmaxnreg_inc<NUM_REGS_REDUCE>();
    bar_sync<10>(13 * 32);
    const uint32_t tmem_base = *tmem_slot;
    if constexpr (DIM == 128) {
      const int row_group = warp;
      const uint32_t row_addr =
          static_cast<uint32_t>(row_group * 32) << 16;
      PhaseTracker<1> dq_full;

      #pragma unroll 1
      for (int j = 0; j < q_steps; ++j) {
        mbarrier_wait_parity(
            smem_ptr_u32(&bars[D128_DQ_FULL]), dq_full.get_phase());
        dq_full.advance();
        uint32_t dq_regs[DIM];
        #pragma unroll
        for (int chunk = 0; chunk < DIM / 32; ++chunk) {
          tcgen05_ld_32x32b_x32(
              tmem_base + T_DQ + chunk * 32 + row_addr,
              reinterpret_cast<uint32_t(&)[32]>(dq_regs[chunk * 32]));
        }
        tcgen05_fence_before_thread_sync();
        if (elect_one_sync()) {
          mbarrier_arrive(smem_ptr_u32(&bars[D128_DQ_FREE]));
        }

        #pragma unroll
        for (int chunk = 0; chunk < DIM / 32; ++chunk) {
          const int stage_idx = chunk & 1;
          const uint4* src4 =
              reinterpret_cast<const uint4*>(dq_regs + chunk * 32);
          uint4* dst4 = reinterpret_cast<uint4*>(s_dq[stage_idx]);
          #pragma unroll
          for (int v4 = 0; v4 < 8; ++v4) {
            // FA4 tiled_copy_1d: consecutive threads own consecutive 128-bit
            // vectors; each later register vector advances by 128 threads.
            dst4[(warp * 32 + lane) + v4 * POSTPROCESS_THREADS] = src4[v4];
          }
          fence_proxy_async_shared_cta();
          bar_sync<11>(4 * 32);
          if (warp == 0 && elect_one_sync()) {
            constexpr uint32_t REDUCE_BYTES = 32 * DIM * sizeof(float);
            constexpr int CHUNK_FLOATS = M_TILE * 32;
            const int query_start = (m_block_begin + j) * M_TILE;
            cpasync_reduce_bulk_add_f32(
                dq_accum + (row_base + query_start) * DIM +
                    chunk * CHUNK_FLOATS,
                smem_ptr_u32(s_dq[stage_idx]), REDUCE_BYTES);
            cp_async_bulk_commit_group();
            cp_async_bulk_wait_group_read<1>();
          }
          bar_sync<11>(4 * 32);
        }
      }
      if (warp == 0 && elect_one_sync()) {
        cp_async_bulk_wait_group_read<0>();
      }
    } else {
      const int row_group = warp;
      const int row = row_group * 32 + lane;
      const uint32_t row_addr =
          static_cast<uint32_t>(row_group * 32) << 16;
      PhaseTracker<1> dq_full;

      #pragma unroll 1
      for (int j = 0; j < q_steps; ++j) {
        mbarrier_wait_parity(
            smem_ptr_u32(&bars[D128_DQ_FULL]), dq_full.get_phase());
        dq_full.advance();
        uint32_t dq_regs[DIM];
        #pragma unroll
        for (int chunk = 0; chunk < DIM / 32; ++chunk) {
          tcgen05_ld_32x32b_x32(
              tmem_base + T_DQ + chunk * 32 + row_addr,
              reinterpret_cast<uint32_t(&)[32]>(dq_regs[chunk * 32]));
        }
        tcgen05_fence_before_thread_sync();
        if (elect_one_sync()) {
          mbarrier_arrive(smem_ptr_u32(&bars[D128_DQ_FREE]));
        }

        #pragma unroll
        for (int chunk = 0; chunk < DIM / 32; ++chunk) {
          const int stage_idx = chunk & 1;
          const uint4* src4 =
              reinterpret_cast<const uint4*>(dq_regs + chunk * 32);
          #pragma unroll
          for (int v4 = 0; v4 < 8; ++v4) {
            const int logical = row * 32 + v4 * 4;
            const int physical = static_cast<int>(
                smem_swizzle_b128(logical * sizeof(float)) / sizeof(float));
            *reinterpret_cast<uint4*>(s_dq[stage_idx] + physical) = src4[v4];
          }
          fence_proxy_async_shared_cta();
          bar_sync<11>(4 * 32);
          if (warp == 0 && elect_one_sync()) {
            const CUtensorMap* dq_map = tmap_dq + head_linear;
            tma_store_2d_add(dq_map, chunk * 32,
                             (m_block_begin + j) * M_TILE,
                             smem_ptr_u32(s_dq[stage_idx]));
            cp_async_bulk_commit_group();
            cp_async_bulk_wait_group_read<1>();
          }
          bar_sync<11>(4 * 32);
        }
      }
      if (warp == 0 && elect_one_sync()) {
        cp_async_bulk_wait_group_read<0>();
      }
    }
    bar_sync<11>(4 * 32);
    bar_sync<10>(13 * 32);
    return;
  }
}

template <int HEAD_DIM>
void launch_preprocess(const __nv_bfloat16* output,
                       const __nv_bfloat16* dout, const float* lse, float* d,
                       float* lse_log2, float* dq_accum, std::size_t rows) {
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = 1;

  cudaLaunchConfig_t config{};
  config.gridDim =
      dim3(static_cast<unsigned>((rows + M_TILE - 1) / M_TILE), 1, 1);
  config.blockDim = dim3(PREPROCESS_THREADS, 1, 1);
  config.dynamicSmemBytes = 0;
  config.stream = nullptr;
  config.attrs = &attribute;
  config.numAttrs = 1;
  CUDA_CHECK(cudaLaunchKernelEx(
      &config, fmha_bwd_preprocess_kernel<HEAD_DIM>, output, dout, lse, d,
      lse_log2, dq_accum, rows));
}

template <int HEAD_DIM, bool FA4_FLAT_DQ = false>
void launch_postprocess(const float* dq_accum, __nv_bfloat16* dq, float scale,
                        std::size_t rows) {
  constexpr std::size_t smem_bytes =
      M_TILE * HEAD_DIM * sizeof(float);
  CUDA_CHECK(cudaFuncSetAttribute(
      fmha_bwd_postprocess_kernel<HEAD_DIM, FA4_FLAT_DQ>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_bytes)));
  fmha_bwd_postprocess_kernel<HEAD_DIM, FA4_FLAT_DQ>
      <<<static_cast<unsigned>((rows + M_TILE - 1) / M_TILE),
         POSTPROCESS_THREADS, smem_bytes>>>(dq_accum, dq, scale, rows);
  CUDA_CHECK(cudaGetLastError());
}

bool uses_fa4_1cta_flat_dq(int seqlen) {
  return seqlen == 512 || seqlen == 1024 || seqlen == 2048 ||
         seqlen == 4096 || seqlen == 8192 || seqlen == 16384;
}

void launch_postprocess_2cta(const float* dq_accum, __nv_bfloat16* dq,
                             float scale, int batch, int heads, int seqlen) {
  constexpr std::size_t smem_bytes =
      M_TILE * 128 * sizeof(float);
  CUDA_CHECK(cudaFuncSetAttribute(
      fmha_bwd_postprocess_2cta_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_bytes)));
  const dim3 grid((seqlen + M_TILE - 1) / M_TILE, batch * heads, 1);
  fmha_bwd_postprocess_2cta_kernel
      <<<grid, POSTPROCESS_THREADS, smem_bytes>>>(
          dq_accum, dq, scale, batch * heads, seqlen);
  CUDA_CHECK(cudaGetLastError());
}

template <int HEAD_DIM, bool USE_2CTA = false>
constexpr std::size_t main_smem_bytes() {
  if constexpr (HEAD_DIM == 64) {
    return (3 * K_TILE + 4 * M_TILE) * HEAD_DIM *
               sizeof(__nv_bfloat16) +
           2 * M_TILE * K_TILE * sizeof(__nv_bfloat16) +
           4 * M_TILE * sizeof(float) + 20 * sizeof(uint64_t) +
           sizeof(uint32_t);
  } else {
    if constexpr (USE_2CTA) {
      return D128_2CTA_HEADER_BYTES +
             (2 * K_TILE + 5 * M_TILE) * HEAD_DIM *
                 sizeof(__nv_bfloat16) +
             2 * M_TILE * sizeof(float);
    } else {
      return (2 * K_TILE + 5 * M_TILE) * HEAD_DIM *
                 sizeof(__nv_bfloat16) +
             4 * M_TILE * sizeof(float) + 27 * sizeof(uint64_t) +
             sizeof(uint32_t);
    }
  }
}

template <int HEAD_DIM, bool IS_CAUSAL>
void launch_main(const CUtensorMap& tmap_q, const CUtensorMap& tmap_k,
                 const CUtensorMap& tmap_v, const CUtensorMap& tmap_dout,
                 const CUtensorMap& tmap_dk, const CUtensorMap& tmap_dv,
                 const CUtensorMap* tmap_dq,
                 const __nv_bfloat16* q, const __nv_bfloat16* k,
                 const __nv_bfloat16* v, const __nv_bfloat16* dout,
                 const float* lse_log2, const float* d, float* dq_accum,
                 __nv_bfloat16* dk, __nv_bfloat16* dv, int batch, int heads,
                 int seqlen) {
  constexpr std::size_t smem_bytes = main_smem_bytes<HEAD_DIM, false>();
  auto launch_dense = [&](auto seq_tag) {
    constexpr int DENSE_SEQ = decltype(seq_tag)::value;
    CUDA_CHECK(cudaFuncSetAttribute(
        fmha_bwd_1cta_dense_kernel<HEAD_DIM, DENSE_SEQ, IS_CAUSAL>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));
    constexpr int n_tiles = DENSE_SEQ / K_TILE;
    const int grid = batch * heads * n_tiles;
    fmha_bwd_1cta_dense_kernel<HEAD_DIM, DENSE_SEQ, IS_CAUSAL>
        <<<grid, N_WARPS * 32, smem_bytes>>>(
            tmap_q, tmap_k, tmap_v, tmap_dout, tmap_dk, tmap_dv,
            tmap_dq, lse_log2, d, dq_accum);
    CUDA_CHECK(cudaGetLastError());
  };
  if (seqlen == 512) {
    launch_dense(std::integral_constant<int, 512>{});
    return;
  } else if (seqlen == 1024) {
    launch_dense(std::integral_constant<int, 1024>{});
    return;
  } else if (seqlen == 2048) {
    launch_dense(std::integral_constant<int, 2048>{});
    return;
  } else if (seqlen == 4096) {
    launch_dense(std::integral_constant<int, 4096>{});
    return;
  } else if (seqlen == 8192) {
    launch_dense(std::integral_constant<int, 8192>{});
    return;
  } else if (seqlen == 16384) {
    launch_dense(std::integral_constant<int, 16384>{});
    return;
  }
  CUDA_CHECK(cudaFuncSetAttribute(
      fmha_bwd_main_kernel<HEAD_DIM, IS_CAUSAL, false>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_bytes)));
  const int n_tiles = (seqlen + K_TILE - 1) / K_TILE;
  const int grid = batch * heads * n_tiles;
  fmha_bwd_main_kernel<HEAD_DIM, IS_CAUSAL, false>
      <<<grid, N_WARPS * 32, smem_bytes>>>(
          tmap_q, tmap_k, tmap_k, tmap_v, tmap_dout, tmap_dk, tmap_dv,
          tmap_q, tmap_dout, tmap_dq, q, k, v, dout,
          lse_log2, d, dq_accum, dk, dv, batch, heads, seqlen);
  CUDA_CHECK(cudaGetLastError());
}

// Pair adjacent D128 key tiles in one cta_group::2 cluster.  The two peers
// jointly execute FA4's score/dP/dV/dK/dQ UMMA sequence and dS relay.
template <bool IS_CAUSAL, int STATIC_SEQLEN>
void launch_main_2cta_impl(
    const CUtensorMap& tmap_q, const CUtensorMap& tmap_k,
    const CUtensorMap& tmap_v, const CUtensorMap& tmap_dout,
    const CUtensorMap& tmap_dk, const CUtensorMap& tmap_dv,
    const CUtensorMap& tmap_kt, const CUtensorMap& tmap_q_half,
    const CUtensorMap& tmap_dout_half,
    const CUtensorMap* tmap_dq, const __nv_bfloat16* q,
    const __nv_bfloat16* k, const __nv_bfloat16* v,
    const __nv_bfloat16* dout, const float* lse_log2, const float* d,
    float* dq_accum, __nv_bfloat16* dk, __nv_bfloat16* dv, int batch,
    int heads, int seqlen) {
  constexpr std::size_t smem_bytes = main_smem_bytes<128, true>();
  auto kernel =
      fmha_bwd_main_kernel<128, IS_CAUSAL, true, STATIC_SEQLEN>;
  CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_bytes)));
  CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  const int n_tiles = (seqlen + K_TILE - 1) / K_TILE;
  if ((n_tiles & 1) != 0) {
    std::fprintf(stderr,
                 "D128 USE_2CTA requires an even key-tile count\n");
    std::exit(EXIT_FAILURE);
  }

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(batch * heads * n_tiles, 1, 1);
  config.blockDim = dim3(N_WARPS * 32, 1, 1);
  config.dynamicSmemBytes = smem_bytes;
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeClusterDimension;
  attribute.val.clusterDim = {2, 1, 1};
  config.attrs = &attribute;
  config.numAttrs = 1;
  CUDA_CHECK(cudaLaunchKernelEx(
      &config, kernel, tmap_q, tmap_k, tmap_kt, tmap_v, tmap_dout, tmap_dk,
      tmap_dv, tmap_q_half, tmap_dout_half, tmap_dq, q, k, v, dout,
      lse_log2, d,
      dq_accum, dk, dv, batch, heads, seqlen));
}

template <bool IS_CAUSAL>
void launch_main_2cta(
    const CUtensorMap& tmap_q, const CUtensorMap& tmap_k,
    const CUtensorMap& tmap_v, const CUtensorMap& tmap_dout,
    const CUtensorMap& tmap_dk, const CUtensorMap& tmap_dv,
    const CUtensorMap& tmap_kt, const CUtensorMap& tmap_q_half,
    const CUtensorMap& tmap_dout_half,
    const CUtensorMap* tmap_dq, const __nv_bfloat16* q,
    const __nv_bfloat16* k, const __nv_bfloat16* v,
    const __nv_bfloat16* dout, const float* lse_log2, const float* d,
    float* dq_accum, __nv_bfloat16* dk, __nv_bfloat16* dv, int batch,
    int heads, int seqlen) {
#define LAUNCH_2CTA_STATIC(STATIC_SEQ)                                         \
  launch_main_2cta_impl<IS_CAUSAL, STATIC_SEQ>(                               \
      tmap_q, tmap_k, tmap_v, tmap_dout, tmap_dk, tmap_dv, tmap_kt,          \
      tmap_q_half, tmap_dout_half, tmap_dq, q, k, v, dout, lse_log2, d,      \
      dq_accum, dk, dv, batch, heads, seqlen)
  if (seqlen == 4096) {
    LAUNCH_2CTA_STATIC(4096);
  } else if (seqlen == 16384) {
    LAUNCH_2CTA_STATIC(16384);
  } else if (seqlen == 8192) {
    LAUNCH_2CTA_STATIC(8192);
  } else if (seqlen == 2048) {
    LAUNCH_2CTA_STATIC(2048);
  } else if (seqlen == 1024) {
    LAUNCH_2CTA_STATIC(1024);
  } else if (seqlen == 512) {
    LAUNCH_2CTA_STATIC(512);
  } else if (seqlen == 256) {
    LAUNCH_2CTA_STATIC(256);
  } else {
    LAUNCH_2CTA_STATIC(0);
  }
#undef LAUNCH_2CTA_STATIC
}

struct CpuReference {
  std::vector<__nv_bfloat16> output;
  std::vector<float> lse;
  std::vector<float> d;
  std::vector<float> dq;
  std::vector<float> dk;
  std::vector<float> dv;
};

std::size_t element_offset(int heads, int seqlen, int headdim, int b, int h,
                           int s, int d) {
  return (((static_cast<std::size_t>(b) * heads + h) * seqlen + s) *
          headdim + d);
}

std::size_t row_offset(int heads, int seqlen, int b, int h, int s) {
  return ((static_cast<std::size_t>(b) * heads + h) * seqlen + s);
}

float bf16_to_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

__nv_bfloat16 float_to_bf16(float value) {
  return __float2bfloat16_rn(value);
}

void make_inputs(int batch, int heads, int seqlen, int headdim,
                 std::vector<__nv_bfloat16>& q,
                 std::vector<__nv_bfloat16>& k,
                 std::vector<__nv_bfloat16>& v,
                 std::vector<__nv_bfloat16>& dout) {
  std::mt19937 generator(20260825u + static_cast<unsigned>(headdim * 17) +
                         static_cast<unsigned>(seqlen));
  std::uniform_real_distribution<float> qkv_dist(-0.75f, 0.75f);
  std::uniform_real_distribution<float> dout_dist(-0.50f, 0.50f);

  const std::size_t elements =
      static_cast<std::size_t>(batch) * heads * seqlen * headdim;
  q.resize(elements);
  k.resize(elements);
  v.resize(elements);
  dout.resize(elements);
  for (std::size_t i = 0; i < elements; ++i) {
    q[i] = float_to_bf16(qkv_dist(generator));
    k[i] = float_to_bf16(qkv_dist(generator));
    v[i] = float_to_bf16(qkv_dist(generator));
    dout[i] = float_to_bf16(dout_dist(generator));
  }
}

// FP32 reference for forward O/LSE and all three backward gradients. M1 uses
// O/LSE/D; M2/M3 reuse dq/dk/dv without introducing a second reference path.
CpuReference cpu_forward_backward(
    int batch, int heads, int seqlen, int headdim, bool causal,
    const std::vector<__nv_bfloat16>& q,
    const std::vector<__nv_bfloat16>& k,
    const std::vector<__nv_bfloat16>& v,
    const std::vector<__nv_bfloat16>& dout) {
  CpuReference ref;
  const std::size_t rows = static_cast<std::size_t>(batch) * heads * seqlen;
  const std::size_t elements = rows * headdim;
  ref.output.resize(elements);
  ref.lse.resize(rows);
  ref.d.resize(rows);
  ref.dq.assign(elements, 0.0f);
  ref.dk.assign(elements, 0.0f);
  ref.dv.assign(elements, 0.0f);

  const float scale = 1.0f / std::sqrt(static_cast<float>(headdim));
  std::vector<float> scores(seqlen);
  std::vector<float> probs(seqlen);

  for (int b = 0; b < batch; ++b) {
    for (int h = 0; h < heads; ++h) {
      for (int qi = 0; qi < seqlen; ++qi) {
        float row_max = -std::numeric_limits<float>::infinity();
        for (int kj = 0; kj < seqlen; ++kj) {
          if (causal && kj > qi) {
            scores[kj] = -std::numeric_limits<float>::infinity();
            continue;
          }
          float dot = 0.0f;
          for (int d = 0; d < headdim; ++d) {
            dot += bf16_to_float(
                       q[element_offset(heads, seqlen, headdim, b, h, qi, d)]) *
                   bf16_to_float(
                       k[element_offset(heads, seqlen, headdim, b, h, kj, d)]);
          }
          scores[kj] = dot * scale;
          row_max = std::max(row_max, scores[kj]);
        }

        float row_sum = 0.0f;
        for (int kj = 0; kj < seqlen; ++kj) {
          const float prob =
              (causal && kj > qi) ? 0.0f : std::exp(scores[kj] - row_max);
          probs[kj] = prob;
          row_sum += prob;
        }
        const float inv_sum = 1.0f / row_sum;
        for (int kj = 0; kj < seqlen; ++kj) {
          probs[kj] *= inv_sum;
        }
        ref.lse[row_offset(heads, seqlen, b, h, qi)] =
            std::log(row_sum) + row_max;

        for (int d = 0; d < headdim; ++d) {
          float out = 0.0f;
          for (int kj = 0; kj < seqlen; ++kj) {
            out += probs[kj] *
                   bf16_to_float(v[element_offset(heads, seqlen, headdim, b,
                                                  h, kj, d)]);
          }
          ref.output[element_offset(heads, seqlen, headdim, b, h, qi, d)] =
              float_to_bf16(out);
        }

        float d_row = 0.0f;
        for (int d = 0; d < headdim; ++d) {
          d_row += bf16_to_float(
                       dout[element_offset(heads, seqlen, headdim, b, h, qi,
                                           d)]) *
                   bf16_to_float(
                       ref.output[element_offset(heads, seqlen, headdim, b, h,
                                                 qi, d)]);
        }
        ref.d[row_offset(heads, seqlen, b, h, qi)] = d_row;

        for (int kj = 0; kj < seqlen; ++kj) {
          if (causal && kj > qi) {
            continue;
          }
          float dp = 0.0f;
          for (int d = 0; d < headdim; ++d) {
            dp += bf16_to_float(
                      dout[element_offset(heads, seqlen, headdim, b, h, qi,
                                          d)]) *
                  bf16_to_float(v[element_offset(heads, seqlen, headdim, b, h,
                                                 kj, d)]);
          }
          const float ds = probs[kj] * (dp - d_row);
          for (int d = 0; d < headdim; ++d) {
            ref.dq[element_offset(heads, seqlen, headdim, b, h, qi, d)] +=
                scale * ds *
                bf16_to_float(k[element_offset(heads, seqlen, headdim, b, h,
                                              kj, d)]);
            ref.dk[element_offset(heads, seqlen, headdim, b, h, kj, d)] +=
                scale * ds *
                bf16_to_float(q[element_offset(heads, seqlen, headdim, b, h,
                                              qi, d)]);
            ref.dv[element_offset(heads, seqlen, headdim, b, h, kj, d)] +=
                probs[kj] *
                bf16_to_float(dout[element_offset(heads, seqlen, headdim, b, h,
                                                 qi, d)]);
          }
        }
      }
    }
  }
  return ref;
}

struct ErrorStats {
  float max_abs = 0.0f;
  float max_rel = 0.0f;
  std::size_t max_abs_index = 0;
  int mismatches = 0;
};

ErrorStats compare_float(const std::vector<float>& got,
                         const std::vector<float>& expected, float atol,
                         float rtol) {
  ErrorStats stats;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const float abs_error = std::abs(got[i] - expected[i]);
    const float rel_error =
        abs_error / std::max(std::abs(expected[i]), 1.0e-6f);
    if (abs_error > stats.max_abs) {
      stats.max_abs = abs_error;
      stats.max_abs_index = i;
    }
    stats.max_rel = std::max(stats.max_rel, rel_error);
    if (!std::isfinite(got[i]) ||
        (abs_error > atol && rel_error > rtol)) {
      ++stats.mismatches;
    }
  }
  return stats;
}

ErrorStats compare_bf16(const std::vector<__nv_bfloat16>& got,
                        const std::vector<__nv_bfloat16>& expected) {
  ErrorStats stats;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const float got_f = bf16_to_float(got[i]);
    const float expected_f = bf16_to_float(expected[i]);
    const float abs_error = std::abs(got_f - expected_f);
    if (abs_error > stats.max_abs) {
      stats.max_abs = abs_error;
      stats.max_abs_index = i;
    }
    if (!std::isfinite(got_f) || got_f != expected_f) {
      ++stats.mismatches;
    }
  }
  return stats;
}

ErrorStats compare_bf16_to_float(const std::vector<__nv_bfloat16>& got,
                                 const std::vector<float>& expected,
                                 float atol, float rtol) {
  ErrorStats stats;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const float got_f = bf16_to_float(got[i]);
    const float expected_f = expected[i];
    const float abs_error = std::abs(got_f - expected_f);
    const float rel_error =
        abs_error / std::max(std::abs(expected_f), 1.0e-6f);
    if (abs_error > stats.max_abs) {
      stats.max_abs = abs_error;
      stats.max_abs_index = i;
    }
    stats.max_rel = std::max(stats.max_rel, rel_error);
    if (!std::isfinite(got_f) ||
        (abs_error > atol && rel_error > rtol)) {
      ++stats.mismatches;
    }
  }
  return stats;
}

bool run_m1_case(int batch, int heads, int seqlen, int head_dim,
                 bool is_causal, const char* label) {
  const std::size_t rows =
      static_cast<std::size_t>(batch) * heads * seqlen;
  const std::size_t elements = rows * head_dim;
  std::vector<__nv_bfloat16> q;
  std::vector<__nv_bfloat16> k;
  std::vector<__nv_bfloat16> v;
  std::vector<__nv_bfloat16> dout;
  make_inputs(batch, heads, seqlen, head_dim, q, k, v, dout);
  const CpuReference ref = cpu_forward_backward(
      batch, heads, seqlen, head_dim, is_causal, q, k, v, dout);

  std::vector<float> initial_dq_accum(elements);
  for (std::size_t i = 0; i < elements; ++i) {
    initial_dq_accum[i] =
        1.0f + 0.125f * std::sin(static_cast<float>(i) * 0.03125f);
  }

  __nv_bfloat16* d_output = nullptr;
  __nv_bfloat16* d_dout = nullptr;
  float* d_lse = nullptr;
  float* d_d = nullptr;
  float* d_lse_log2 = nullptr;
  float* d_dq_accum = nullptr;
  __nv_bfloat16* d_dq = nullptr;
  CUDA_CHECK(cudaMalloc(&d_output, elements * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_dout, elements * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_lse, rows * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_d, rows * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_lse_log2, rows * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dq_accum, elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dq, elements * sizeof(__nv_bfloat16)));

  CUDA_CHECK(cudaMemcpy(d_output, ref.output.data(),
                        elements * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_dout, dout.data(),
                        elements * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_lse, ref.lse.data(),
                        rows * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_dq_accum, initial_dq_accum.data(),
                        elements * sizeof(float),
                        cudaMemcpyHostToDevice));

  if (head_dim == 64) {
    launch_preprocess<64>(d_output, d_dout, d_lse, d_d, d_lse_log2,
                          d_dq_accum, rows);
  } else {
    launch_preprocess<128>(d_output, d_dout, d_lse, d_d, d_lse_log2,
                           d_dq_accum, rows);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> got_d(rows);
  std::vector<float> got_lse_log2(rows);
  std::vector<float> got_zero(elements);
  CUDA_CHECK(cudaMemcpy(got_d.data(), d_d, rows * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(got_lse_log2.data(), d_lse_log2,
                        rows * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(got_zero.data(), d_dq_accum,
                        elements * sizeof(float), cudaMemcpyDeviceToHost));

  std::vector<float> expected_lse_log2(rows);
  std::vector<float> expected_zero(elements, 0.0f);
  for (std::size_t i = 0; i < rows; ++i) {
    expected_lse_log2[i] = ref.lse[i] * LOG2_E;
  }

  const ErrorStats d_stats = compare_float(got_d, ref.d, 2.0e-4f, 2.0e-4f);
  const ErrorStats lse_stats =
      compare_float(got_lse_log2, expected_lse_log2, 1.0e-6f, 1.0e-6f);
  const ErrorStats zero_stats =
      compare_float(got_zero, expected_zero, 0.0f, 0.0f);

  // Refill dQaccum with known FP32 values, then verify the scale/cast stage.
  for (std::size_t i = 0; i < elements; ++i) {
    initial_dq_accum[i] =
        0.75f * std::sin(static_cast<float>(i) * 0.017f) -
        0.25f * std::cos(static_cast<float>(i) * 0.011f);
  }
  CUDA_CHECK(cudaMemcpy(d_dq_accum, initial_dq_accum.data(),
                        elements * sizeof(float),
                        cudaMemcpyHostToDevice));

  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  if (head_dim == 64) {
    launch_postprocess<64>(d_dq_accum, d_dq, scale, rows);
  } else {
    launch_postprocess<128>(d_dq_accum, d_dq, scale, rows);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> got_dq(elements);
  std::vector<__nv_bfloat16> expected_dq(elements);
  CUDA_CHECK(cudaMemcpy(got_dq.data(), d_dq,
                        elements * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));
  for (std::size_t i = 0; i < elements; ++i) {
    expected_dq[i] = float_to_bf16(initial_dq_accum[i] * scale);
  }
  const ErrorStats post_stats = compare_bf16(got_dq, expected_dq);

  const bool pass = d_stats.mismatches == 0 &&
                    lse_stats.mismatches == 0 &&
                    zero_stats.mismatches == 0 &&
                    post_stats.mismatches == 0;
  std::printf(
      "M1 %-18s B=%d H=%d S=%d D=%d causal=%d: %s "
      "[D abs=%.3e, LSE2 abs=%.3e, clear bad=%d, post bad=%d]\n",
      label, batch, heads, seqlen, head_dim,
      static_cast<int>(is_causal), pass ? "PASS" : "FAIL", d_stats.max_abs,
      lse_stats.max_abs, zero_stats.mismatches, post_stats.mismatches);
  if (!pass) {
    std::printf(
        "  mismatches: D=%d LSE2=%d clear=%d post=%d; "
        "post max_abs=%.3e at %zu\n",
        d_stats.mismatches, lse_stats.mismatches, zero_stats.mismatches,
        post_stats.mismatches, post_stats.max_abs, post_stats.max_abs_index);
  }
  CUDA_CHECK(cudaFree(d_dq));
  CUDA_CHECK(cudaFree(d_dq_accum));
  CUDA_CHECK(cudaFree(d_lse_log2));
  CUDA_CHECK(cudaFree(d_d));
  CUDA_CHECK(cudaFree(d_lse));
  CUDA_CHECK(cudaFree(d_dout));
  CUDA_CHECK(cudaFree(d_output));
  return pass;
}

bool run_m1_suite() {
  bool pass = true;
  pass = run_m1_case(1, 1, 31, 64, false, "d64-nc-tail") && pass;
  pass = run_m1_case(2, 2, 129, 64, true, "d64-c-tail") && pass;
  pass = run_m1_case(1, 2, 33, 128, false, "d128-nc-tail") && pass;
  pass = run_m1_case(2, 1, 130, 128, true, "d128-c-tail") && pass;
  std::printf("M1 preprocess/postprocess suite: %s\n",
              pass ? "PASS" : "FAIL");
  return pass;
}

bool run_main_case(int batch, int heads, int seqlen, int head_dim,
                   bool is_causal, const char* label,
                   bool use_2cta = false) {
  const std::size_t rows =
      static_cast<std::size_t>(batch) * heads * seqlen;
  const std::size_t elements = rows * head_dim;
  std::vector<__nv_bfloat16> q;
  std::vector<__nv_bfloat16> k;
  std::vector<__nv_bfloat16> v;
  std::vector<__nv_bfloat16> dout;
  make_inputs(batch, heads, seqlen, head_dim, q, k, v, dout);
  const CpuReference ref = cpu_forward_backward(
      batch, heads, seqlen, head_dim, is_causal, q, k, v, dout);

  __nv_bfloat16* d_q_input = nullptr;
  __nv_bfloat16* d_k_input = nullptr;
  __nv_bfloat16* d_v_input = nullptr;
  __nv_bfloat16* d_output = nullptr;
  __nv_bfloat16* d_dout = nullptr;
  float* d_lse = nullptr;
  float* d_d = nullptr;
  float* d_lse_log2 = nullptr;
  float* d_dq_accum = nullptr;
  __nv_bfloat16* d_dq = nullptr;
  __nv_bfloat16* d_dk = nullptr;
  __nv_bfloat16* d_dv = nullptr;
  CUDA_CHECK(cudaMalloc(&d_q_input, elements * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_k_input, elements * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_v_input, elements * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_output, elements * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_dout, elements * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_lse, rows * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_d, rows * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_lse_log2, rows * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dq_accum, elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dq, elements * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_dk, elements * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_dv, elements * sizeof(__nv_bfloat16)));

  CUDA_CHECK(cudaMemcpy(d_q_input, q.data(),
                        elements * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k_input, k.data(),
                        elements * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v_input, v.data(),
                        elements * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_output, ref.output.data(),
                        elements * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_dout, dout.data(),
                        elements * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_lse, ref.lse.data(),
                        rows * sizeof(float), cudaMemcpyHostToDevice));

  CUtensorMap tmap_q;
  CUtensorMap tmap_k;
  CUtensorMap tmap_kt;
  CUtensorMap tmap_v;
  CUtensorMap tmap_dout;
  CUtensorMap tmap_dk;
  CUtensorMap tmap_dv;
  CUtensorMap tmap_q_half;
  CUtensorMap tmap_dout_half;
  CUDA_CHECK(make_tma_3d_tiled(
      &tmap_q, d_q_input, head_dim, seqlen, batch * heads, 64, M_TILE, 1,
      sizeof(__nv_bfloat16), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16));
  CUDA_CHECK(make_tma_3d_tiled(
      &tmap_k, d_k_input, head_dim, seqlen, batch * heads, 64, K_TILE, 1,
      sizeof(__nv_bfloat16), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16));
  CUDA_CHECK(make_tma_3d_tiled(
      &tmap_kt, d_k_input, head_dim, seqlen, batch * heads,
      64, 2 * K_TILE, 1, sizeof(__nv_bfloat16),
      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16));
  CUDA_CHECK(make_tma_3d_tiled(
      &tmap_v, d_v_input, head_dim, seqlen, batch * heads, 64, K_TILE, 1,
      sizeof(__nv_bfloat16), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16));
  CUDA_CHECK(make_tma_3d_tiled(
      &tmap_dout, d_dout, head_dim, seqlen, batch * heads, 64, M_TILE, 1,
      sizeof(__nv_bfloat16), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16));
  CUDA_CHECK(make_tma_3d_tiled(
      &tmap_dk, d_dk, head_dim, seqlen, batch * heads, 64, K_TILE, 1,
      sizeof(__nv_bfloat16), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16));
  CUDA_CHECK(make_tma_3d_tiled(
      &tmap_dv, d_dv, head_dim, seqlen, batch * heads, 64, K_TILE, 1,
      sizeof(__nv_bfloat16), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16));
  CUDA_CHECK(make_tma_3d_tiled(
      &tmap_q_half, d_q_input, head_dim, seqlen, batch * heads, 64, 64, 1,
      sizeof(__nv_bfloat16), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16));
  CUDA_CHECK(make_tma_3d_tiled(
      &tmap_dout_half, d_dout, head_dim, seqlen, batch * heads, 64, 64, 1,
      sizeof(__nv_bfloat16), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16));
  std::vector<CUtensorMap> host_tmap_dq(batch * heads);
  for (int bh = 0; bh < batch * heads; ++bh) {
    CUDA_CHECK(make_tma_2d_tiled(
        &host_tmap_dq[bh],
        d_dq_accum + static_cast<std::size_t>(bh) * seqlen * head_dim,
        seqlen, head_dim, M_TILE, 32, sizeof(float),
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32));
  }
  CUtensorMap* tmap_dq = nullptr;
  CUDA_CHECK(cudaMalloc(&tmap_dq, host_tmap_dq.size() * sizeof(CUtensorMap)));
  CUDA_CHECK(cudaMemcpy(tmap_dq, host_tmap_dq.data(),
                        host_tmap_dq.size() * sizeof(CUtensorMap),
                        cudaMemcpyHostToDevice));

  if (head_dim == 64) {
    launch_preprocess<64>(d_output, d_dout, d_lse, d_d, d_lse_log2,
                          d_dq_accum, rows);
    if (is_causal) {
      launch_main<64, true>(
          tmap_q, tmap_k, tmap_v, tmap_dout, tmap_dk, tmap_dv, tmap_dq,
          d_q_input, d_k_input, d_v_input, d_dout, d_lse_log2, d_d,
          d_dq_accum, d_dk, d_dv, batch, heads, seqlen);
    } else {
      launch_main<64, false>(
          tmap_q, tmap_k, tmap_v, tmap_dout, tmap_dk, tmap_dv, tmap_dq,
          d_q_input, d_k_input, d_v_input, d_dout, d_lse_log2, d_d,
          d_dq_accum, d_dk, d_dv, batch, heads, seqlen);
    }
    launch_postprocess<64>(
        d_dq_accum, d_dq,
        1.0f / std::sqrt(static_cast<float>(head_dim)), rows);
  } else {
    launch_preprocess<128>(d_output, d_dout, d_lse, d_d, d_lse_log2,
                           d_dq_accum, rows);
    if (use_2cta && is_causal) {
      launch_main_2cta<true>(
          tmap_q, tmap_k, tmap_v, tmap_dout, tmap_dk, tmap_dv, tmap_kt,
          tmap_q_half, tmap_dout_half, tmap_dq, d_q_input, d_k_input,
          d_v_input, d_dout, d_lse_log2, d_d, d_dq_accum, d_dk, d_dv,
          batch, heads, seqlen);
    } else if (use_2cta) {
      launch_main_2cta<false>(
          tmap_q, tmap_k, tmap_v, tmap_dout, tmap_dk, tmap_dv, tmap_kt,
          tmap_q_half, tmap_dout_half, tmap_dq, d_q_input, d_k_input,
          d_v_input, d_dout, d_lse_log2, d_d, d_dq_accum, d_dk, d_dv,
          batch, heads, seqlen);
    } else if (is_causal) {
      launch_main<128, true>(
          tmap_q, tmap_k, tmap_v, tmap_dout, tmap_dk, tmap_dv, tmap_dq,
          d_q_input, d_k_input, d_v_input, d_dout, d_lse_log2, d_d,
          d_dq_accum, d_dk, d_dv, batch, heads, seqlen);
    } else {
      launch_main<128, false>(
          tmap_q, tmap_k, tmap_v, tmap_dout, tmap_dk, tmap_dv, tmap_dq,
          d_q_input, d_k_input, d_v_input, d_dout, d_lse_log2, d_d,
          d_dq_accum, d_dk, d_dv, batch, heads, seqlen);
    }
    if (use_2cta) {
      launch_postprocess_2cta(
          d_dq_accum, d_dq,
          1.0f / std::sqrt(static_cast<float>(head_dim)), batch, heads,
          seqlen);
    } else {
      if (uses_fa4_1cta_flat_dq(seqlen)) {
        launch_postprocess<128, true>(
            d_dq_accum, d_dq,
            1.0f / std::sqrt(static_cast<float>(head_dim)), rows);
      } else {
        launch_postprocess<128>(
            d_dq_accum, d_dq,
            1.0f / std::sqrt(static_cast<float>(head_dim)), rows);
      }
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> got_dq(elements);
  std::vector<__nv_bfloat16> got_dk(elements);
  std::vector<__nv_bfloat16> got_dv(elements);
  CUDA_CHECK(cudaMemcpy(got_dq.data(), d_dq,
                        elements * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(got_dk.data(), d_dk,
                        elements * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(got_dv.data(), d_dv,
                        elements * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));

  const ErrorStats dq_stats =
      compare_bf16_to_float(got_dq, ref.dq, 2.0e-2f, 5.0e-2f);
  const ErrorStats dk_stats =
      compare_bf16_to_float(got_dk, ref.dk, 2.0e-2f, 5.0e-2f);
  const ErrorStats dv_stats =
      compare_bf16_to_float(got_dv, ref.dv, 2.0e-2f, 5.0e-2f);
  const bool pass = dq_stats.mismatches == 0 && dk_stats.mismatches == 0 &&
                    dv_stats.mismatches == 0;
  std::printf(
      "M2 %-18s B=%d H=%d S=%d D=%d causal=%d: %s "
      "[dQ abs=%.3e bad=%d, dK abs=%.3e bad=%d, "
      "dV abs=%.3e bad=%d]\n",
      label, batch, heads, seqlen, head_dim,
      static_cast<int>(is_causal), pass ? "PASS" : "FAIL", dq_stats.max_abs,
      dq_stats.mismatches, dk_stats.max_abs, dk_stats.mismatches,
      dv_stats.max_abs, dv_stats.mismatches);
  if (!pass) {
    std::printf("  max indices: dQ=%zu dK=%zu dV=%zu\n",
                dq_stats.max_abs_index, dk_stats.max_abs_index,
                dv_stats.max_abs_index);
    const auto print_max_value = [&](const char* name,
                                     const std::vector<__nv_bfloat16>& got,
                                     const std::vector<float>& expected,
                                     std::size_t index) {
      std::printf("  %s max value: got=%.6f want=%.6f\n", name,
                  bf16_to_float(got[index]), expected[index]);
    };
    print_max_value("dQ", got_dq, ref.dq, dq_stats.max_abs_index);
    print_max_value("dK", got_dk, ref.dk, dk_stats.max_abs_index);
    print_max_value("dV", got_dv, ref.dv, dv_stats.max_abs_index);
  }
  CUDA_CHECK(cudaFree(d_dv));
  CUDA_CHECK(cudaFree(d_dk));
  CUDA_CHECK(cudaFree(d_dq));
  CUDA_CHECK(cudaFree(tmap_dq));
  CUDA_CHECK(cudaFree(d_dq_accum));
  CUDA_CHECK(cudaFree(d_lse_log2));
  CUDA_CHECK(cudaFree(d_d));
  CUDA_CHECK(cudaFree(d_lse));
  CUDA_CHECK(cudaFree(d_dout));
  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaFree(d_v_input));
  CUDA_CHECK(cudaFree(d_k_input));
  CUDA_CHECK(cudaFree(d_q_input));
  return pass;
}

bool run_main_suite() {
  bool pass = true;
  pass = run_main_case(1, 1, 31, 64, false, "d64-nc-tail") && pass;
  pass = run_main_case(1, 1, 129, 64, true, "d64-c-2tiles") && pass;
  pass = run_main_case(2, 2, 129, 64, false, "d64-nc-bh-tail") && pass;
  pass = run_main_case(1, 1, 512, 64, false, "d64-nc-s512") && pass;
  pass = run_main_case(1, 1, 512, 64, true, "d64-c-s512") && pass;
  pass = run_main_case(1, 1, 1024, 64, false, "d64-nc-s1024") && pass;
  pass = run_main_case(1, 1, 1024, 64, true, "d64-c-s1024") && pass;
  pass = run_main_case(1, 1, 33, 128, false, "d128-nc-tail") && pass;
  pass = run_main_case(1, 1, 130, 128, true, "d128-c-2tiles") && pass;
  pass = run_main_case(1, 1, 512, 128, false, "d128-nc-1cta-s512") && pass;
  pass = run_main_case(1, 1, 256, 128, false, "d128-nc-2cta-launch",
                       true) && pass;
  pass = run_main_case(1, 1, 256, 128, true, "d128-c-2cta-launch",
                       true) && pass;
  pass = run_main_case(1, 1, 512, 128, false, "d128-nc-2cta-s512",
                       true) && pass;
  pass = run_main_case(1, 1, 512, 128, true, "d128-c-2cta-s512",
                       true) && pass;
  pass = run_main_case(2, 2, 256, 128, false, "d128-nc-2cta-bh",
                       true) && pass;
  pass = run_main_case(2, 2, 256, 128, true, "d128-c-2cta-bh",
                       true) && pass;
  std::printf("M2/M3 ownership/tcgen05 suite: %s\n",
              pass ? "PASS" : "FAIL");
  return pass;
}

template <int HEAD_DIM, bool IS_CAUSAL>
void run_benchmark(int batch, int seqlen, int warmup, int iterations,
                   bool use_2cta = false) {
  constexpr int MODEL_DIM = 2048;
  constexpr float SCALE =
      HEAD_DIM == 64 ? 0.125f : 0.08838834764831845f;
  const int heads = MODEL_DIM / HEAD_DIM;
  const std::size_t rows =
      static_cast<std::size_t>(batch) * heads * seqlen;
  const std::size_t elements = rows * HEAD_DIM;

  const auto host = fmha_context_bwd_bf16_benchmark::inputs(
      batch, heads, seqlen, HEAD_DIM, IS_CAUSAL);
  warmup = fmha_context_bwd_bf16_benchmark::env_count("BENCH_WARMUP", warmup, 0);
  iterations = fmha_context_bwd_bf16_benchmark::env_count("BENCH_ITERS", iterations, 1);

  __nv_bfloat16* q = nullptr;
  __nv_bfloat16* k = nullptr;
  __nv_bfloat16* v = nullptr;
  __nv_bfloat16* output = nullptr;
  __nv_bfloat16* dout = nullptr;
  __nv_bfloat16* dq = nullptr;
  __nv_bfloat16* dk = nullptr;
  __nv_bfloat16* dv = nullptr;
  float* lse = nullptr;
  float* d = nullptr;
  float* lse_log2 = nullptr;
  float* dq_accum = nullptr;
  const std::size_t bf16_bytes = elements * sizeof(__nv_bfloat16);
  CUDA_CHECK(cudaMalloc(&q, bf16_bytes));
  CUDA_CHECK(cudaMalloc(&k, bf16_bytes));
  CUDA_CHECK(cudaMalloc(&v, bf16_bytes));
  CUDA_CHECK(cudaMalloc(&output, bf16_bytes));
  CUDA_CHECK(cudaMalloc(&dout, bf16_bytes));
  CUDA_CHECK(cudaMalloc(&dq, bf16_bytes));
  CUDA_CHECK(cudaMalloc(&dk, bf16_bytes));
  CUDA_CHECK(cudaMalloc(&dv, bf16_bytes));
  CUDA_CHECK(cudaMalloc(&lse, rows * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d, rows * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&lse_log2, rows * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dq_accum, elements * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(q, host.q.data(), bf16_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(k, host.k.data(), bf16_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(v, host.v.data(), bf16_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(output, host.output.data(), bf16_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dout, host.dout.data(), bf16_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(lse, host.lse.data(), rows * sizeof(float),
                        cudaMemcpyHostToDevice));

  CUtensorMap tmap_q;
  CUtensorMap tmap_k;
  CUtensorMap tmap_kt;
  CUtensorMap tmap_v;
  CUtensorMap tmap_dout;
  CUtensorMap tmap_dk;
  CUtensorMap tmap_dv;
  CUtensorMap tmap_q_half;
  CUtensorMap tmap_dout_half;
  CUDA_CHECK(make_tma_3d_tiled(
      &tmap_q, q, HEAD_DIM, seqlen, batch * heads, 64, M_TILE, 1,
      sizeof(__nv_bfloat16), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16));
  CUDA_CHECK(make_tma_3d_tiled(
      &tmap_k, k, HEAD_DIM, seqlen, batch * heads, 64, K_TILE, 1,
      sizeof(__nv_bfloat16), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16));
  CUDA_CHECK(make_tma_3d_tiled(
      &tmap_kt, k, HEAD_DIM, seqlen, batch * heads,
      64, 2 * K_TILE, 1, sizeof(__nv_bfloat16),
      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16));
  CUDA_CHECK(make_tma_3d_tiled(
      &tmap_v, v, HEAD_DIM, seqlen, batch * heads, 64, K_TILE, 1,
      sizeof(__nv_bfloat16), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16));
  CUDA_CHECK(make_tma_3d_tiled(
      &tmap_dout, dout, HEAD_DIM, seqlen, batch * heads, 64, M_TILE, 1,
      sizeof(__nv_bfloat16), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16));
  CUDA_CHECK(make_tma_3d_tiled(
      &tmap_dk, dk, HEAD_DIM, seqlen, batch * heads, 64, K_TILE, 1,
      sizeof(__nv_bfloat16), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16));
  CUDA_CHECK(make_tma_3d_tiled(
      &tmap_dv, dv, HEAD_DIM, seqlen, batch * heads, 64, K_TILE, 1,
      sizeof(__nv_bfloat16), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16));
  CUDA_CHECK(make_tma_3d_tiled(
      &tmap_q_half, q, HEAD_DIM, seqlen, batch * heads, 64, 64, 1,
      sizeof(__nv_bfloat16), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16));
  CUDA_CHECK(make_tma_3d_tiled(
      &tmap_dout_half, dout, HEAD_DIM, seqlen, batch * heads, 64, 64, 1,
      sizeof(__nv_bfloat16), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16));
  std::vector<CUtensorMap> host_tmap_dq(batch * heads);
  for (int bh = 0; bh < batch * heads; ++bh) {
    CUDA_CHECK(make_tma_2d_tiled(
        &host_tmap_dq[bh],
        dq_accum + static_cast<std::size_t>(bh) * seqlen * HEAD_DIM,
        seqlen, HEAD_DIM, M_TILE, 32, sizeof(float),
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32));
  }
  CUtensorMap* tmap_dq = nullptr;
  CUDA_CHECK(cudaMalloc(&tmap_dq, host_tmap_dq.size() * sizeof(CUtensorMap)));
  CUDA_CHECK(cudaMemcpy(tmap_dq, host_tmap_dq.data(),
                        host_tmap_dq.size() * sizeof(CUtensorMap),
                        cudaMemcpyHostToDevice));

  const auto launch_iteration = [&]() {
    launch_preprocess<HEAD_DIM>(output, dout, lse, d, lse_log2, dq_accum,
                                rows);
    if constexpr (HEAD_DIM == 128) {
      if (use_2cta) {
        launch_main_2cta<IS_CAUSAL>(
            tmap_q, tmap_k, tmap_v, tmap_dout, tmap_dk, tmap_dv,
            tmap_kt, tmap_q_half, tmap_dout_half, tmap_dq, q, k, v, dout,
            lse_log2, d, dq_accum, dk, dv, batch, heads, seqlen);
      } else {
        launch_main<HEAD_DIM, IS_CAUSAL>(
            tmap_q, tmap_k, tmap_v, tmap_dout, tmap_dk, tmap_dv, tmap_dq,
            q, k, v, dout, lse_log2, d, dq_accum, dk, dv, batch, heads,
            seqlen);
      }
    } else {
      launch_main<HEAD_DIM, IS_CAUSAL>(
          tmap_q, tmap_k, tmap_v, tmap_dout, tmap_dk, tmap_dv, tmap_dq,
          q, k, v, dout, lse_log2, d, dq_accum, dk, dv, batch, heads,
          seqlen);
    }
    if constexpr (HEAD_DIM == 128) {
      if (use_2cta) {
        launch_postprocess_2cta(dq_accum, dq, SCALE, batch, heads, seqlen);
      } else {
        if (uses_fa4_1cta_flat_dq(seqlen)) {
          launch_postprocess<HEAD_DIM, true>(dq_accum, dq, SCALE, rows);
        } else {
          launch_postprocess<HEAD_DIM>(dq_accum, dq, SCALE, rows);
        }
      }
    } else {
      launch_postprocess<HEAD_DIM>(dq_accum, dq, SCALE, rows);
    }
  };
  const double latency_ms = fmha_context_bwd_bf16_benchmark::measure(
      launch_iteration, warmup, iterations);
  if (latency_ms > 0.0) {
    // Preserve the historical backward convention: causal work is S*S/2.
    double forward_flops = 4.0 * batch * seqlen * seqlen * heads * HEAD_DIM;
    if constexpr (IS_CAUSAL) forward_flops *= 0.5;
    const double tflops = 2.5 * forward_flops / latency_ms / 1.0e9;
    std::printf(
        "BENCH B=%d H=%d S=%d D=%d causal=%d cta_group=%d "
        "warmup=%d iters=%d ms=%.6f tflops=%.3f peak_pct=%.2f\n",
        batch, heads, seqlen, HEAD_DIM, static_cast<int>(IS_CAUSAL),
        use_2cta ? 2 : 1, warmup, iterations, latency_ms, tflops, tflops / 25.0);
  }
  fmha_context_bwd_bf16_benchmark::dump("dq", dq, elements);
  fmha_context_bwd_bf16_benchmark::dump("dk", dk, elements);
  fmha_context_bwd_bf16_benchmark::dump("dv", dv, elements);

  CUDA_CHECK(cudaFree(dq_accum));
  CUDA_CHECK(cudaFree(tmap_dq));
  CUDA_CHECK(cudaFree(lse_log2));
  CUDA_CHECK(cudaFree(d));
  CUDA_CHECK(cudaFree(lse));
  CUDA_CHECK(cudaFree(dv));
  CUDA_CHECK(cudaFree(dk));
  CUDA_CHECK(cudaFree(dq));
  CUDA_CHECK(cudaFree(dout));
  CUDA_CHECK(cudaFree(output));
  CUDA_CHECK(cudaFree(v));
  CUDA_CHECK(cudaFree(k));
  CUDA_CHECK(cudaFree(q));
}

void print_usage(const char* program) {
  std::printf(
      "Usage: %s [--verify|--m1-verify|--bench B S D CAUSAL [ITERS]|"
      "--bench-2cta B S CAUSAL [ITERS]]\n",
      program);
}

int main(int argc, char** argv) {
  const bool benchmark_1cta =
      argc >= 2 && std::string(argv[1]) == "--bench";
  const bool benchmark_2cta =
      argc >= 2 && std::string(argv[1]) == "--bench-2cta";
  const bool benchmark = benchmark_1cta || benchmark_2cta;
  const bool benchmark_arity_ok =
      (benchmark_1cta && (argc == 6 || argc == 7)) ||
      (benchmark_2cta && (argc == 5 || argc == 6));
  if ((!benchmark && argc > 2) || (benchmark && !benchmark_arity_ok) ||
      (argc == 2 && std::string(argv[1]) != "--verify" &&
       std::string(argv[1]) != "--m1-verify" &&
       std::string(argv[1]) != "--help")) {
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }
  if (argc == 2 && std::string(argv[1]) == "--help") {
    print_usage(argv[0]);
    return EXIT_SUCCESS;
  }

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  std::printf("device=%d %s cc=%d.%d\n", device, properties.name,
              properties.major, properties.minor);
  if (properties.major != 10) {
    std::fprintf(stderr, "fmha_context_bwd_bf16 requires sm_100a\n");
    return EXIT_FAILURE;
  }

  if (benchmark) {
    const int batch = fmha_context_bwd_bf16_benchmark::argument(argv[2]);
    const int seqlen = fmha_context_bwd_bf16_benchmark::argument(argv[3]);
    const int head_dim = benchmark_2cta ? 128 : fmha_context_bwd_bf16_benchmark::argument(argv[4]);
    const int is_causal =
        fmha_context_bwd_bf16_benchmark::argument(argv[benchmark_2cta ? 4 : 5]);
    const int iterations =
        benchmark_2cta ? (argc == 6 ? fmha_context_bwd_bf16_benchmark::argument(argv[5]) : 20)
                       : (argc == 7 ? fmha_context_bwd_bf16_benchmark::argument(argv[6]) : 20);
    if (batch <= 0 || seqlen <= 0 ||
        (head_dim != 64 && head_dim != 128) ||
        (is_causal != 0 && is_causal != 1) || iterations <= 0 ||
        (benchmark_2cta && (((seqlen + K_TILE - 1) / K_TILE) & 1))) {
      print_usage(argv[0]);
      return EXIT_FAILURE;
    }
    if (head_dim == 64 && is_causal == 0) {
      run_benchmark<64, false>(batch, seqlen, 5, iterations);
    } else if (head_dim == 64) {
      run_benchmark<64, true>(batch, seqlen, 5, iterations);
    } else if (is_causal == 0) {
      run_benchmark<128, false>(batch, seqlen, 5, iterations,
                                benchmark_2cta);
    } else {
      run_benchmark<128, true>(batch, seqlen, 5, iterations,
                               benchmark_2cta);
    }
    return EXIT_SUCCESS;
  }

  const bool m1_pass = run_m1_suite();
  if (argc == 2 && std::string(argv[1]) == "--m1-verify") {
    return m1_pass ? EXIT_SUCCESS : EXIT_FAILURE;
  }
  const bool cluster_pass = m1_pass && run_cluster_protocol_gate();
  const bool main_pass = cluster_pass && run_main_suite();
  return main_pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
