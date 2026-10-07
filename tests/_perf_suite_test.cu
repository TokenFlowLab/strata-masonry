#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// ARCH: sm_100a
// _perf_suite_test.cu -- microbenchmark for representative primitives/composites/blocks.
//
// Uses cudaEvent-based kernel timing. Each kernel runs N_WARMUP warm-ups
// then N_ITERS timed iterations; reports min / avg microseconds per launch.

#include "test_utils.cuh"
#include "../src/primitives/0_tcgen05_alloc.cuh"
#include "../src/primitives/1_tcgen05_dealloc.cuh"
#include "../src/primitives/2_tcgen05_relinquish.cuh"
#include "../src/primitives/9_tcgen05_ld.cuh"
#include "../src/primitives/10_tcgen05_st.cuh"
#include "../src/primitives/11_tcgen05_commit.cuh"
#include "../src/primitives/12_tcgen05_wait.cuh"
#include "../src/primitives/18_tma_load.cuh"
#include "../src/primitives/22_tma_store.cuh"
#include "../src/primitives/23_tma_tensormap.cuh"
#include "../src/primitives/25_tma_async_group.cuh"
#include "../src/primitives/29_mbarrier_init.cuh"
#include "../src/primitives/30_mbarrier_arrive.cuh"
#include "../src/primitives/31_mbarrier_arrive_tx.cuh"
#include "../src/primitives/33_mbarrier_try_wait.cuh"
#include "../src/primitives/34_fence_proxy_async.cuh"
#include "../src/primitives/40_stmatrix.cuh"
#include "../src/primitives/44_elect_sync.cuh"
#include "../src/primitives/48_redux_sync.cuh"
#include "../src/primitives/50_atom_global.cuh"
#include "../src/primitives/63_cvt_f32_to_f16_bf16.cuh"
#include "../src/composites/119_pipeline_init_blackwell.cuh"
#include "../src/composites/85_online_softmax.cuh"

static constexpr int N_WARMUP = 10;
static constexpr int N_ITERS  = 100;

// ---- per-kernel minimal launches for timing ----

__global__ void k_tmem_alloc_dealloc() {
  __shared__ __align__(16) uint32_t slot;
  if (threadIdx.x == 0) slot = 0;
  __syncthreads();
  if (threadIdx.x < 32) {
    tcgen05_alloc<1>(smem_ptr_u32(&slot), 128);
    tcgen05_relinquish_alloc_permit<1>();
  }
  __syncthreads();
  uint32_t tbase = slot;
  __syncthreads();
  if (threadIdx.x == 0) tcgen05_dealloc<1>(tbase, 128);
}

__global__ void k_tmem_ld_st_roundtrip() {
  __shared__ __align__(16) uint32_t slot;
  if (threadIdx.x == 0) slot = 0;
  __syncthreads();
  if (threadIdx.x < 32) {
    tcgen05_alloc<1>(smem_ptr_u32(&slot), 32);
    tcgen05_relinquish_alloc_permit<1>();
  }
  __syncthreads();
  uint32_t tbase = slot;
  if (threadIdx.x < 32) {
    uint32_t w[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) w[i] = threadIdx.x * 8 + i;
    tcgen05_st_32x32b_x8(tbase, w);
    tcgen05_wait_st();
    uint32_t r[8];
    tcgen05_ld_32x32b_x8(tbase, r);
    tcgen05_wait_ld();
    (void)r;
  }
  __syncthreads();
  if (threadIdx.x == 0) tcgen05_dealloc<1>(tbase, 32);
}

__global__ void k_tma_load(const __grid_constant__ CUtensorMap t, float* out) {
  __shared__ __align__(128) float smem[16 * 16];
  __shared__ __align__(16) uint64_t mbar;
  if (threadIdx.x == 0) {
    mbarrier_init(smem_ptr_u32(&mbar), 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    mbarrier_arrive_expect_tx(smem_ptr_u32(&mbar), 16 * 16 * sizeof(float));
    tma_load_2d(smem_ptr_u32(smem), &t, smem_ptr_u32(&mbar), 0, 0);
  }
  mbarrier_wait_parity(smem_ptr_u32(&mbar), 0);
  if (threadIdx.x < 16 * 16) out[threadIdx.x] = smem[threadIdx.x];
  __syncthreads();
  if (threadIdx.x == 0)
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n"
                 :: "r"(smem_ptr_u32(&mbar)));
}

__global__ void k_tma_store(const __grid_constant__ CUtensorMap t) {
  __shared__ __align__(128) float smem[16 * 16];
  if (threadIdx.x < 16 * 16) smem[threadIdx.x] = (float)threadIdx.x;
  __syncthreads();
  if (threadIdx.x == 0) {
    fence_proxy_async_shared_cta();
    tma_store_2d(&t, 0, 0, smem_ptr_u32(smem));
    cp_async_bulk_commit_group();
    cp_async_bulk_wait_group<0>();
  }
}

__global__ void k_atom_add(uint32_t* cnt) {
  atom_global_add_u32(cnt, 1u);
}

__global__ void k_redux_add(uint32_t* out) {
  uint32_t s = redux_sync_add_u32(threadIdx.x);
  if (threadIdx.x == 0) *out = s;
}

__global__ void k_elect(uint32_t* out) {
  if (elect_one_sync()) atomicAdd(out, 1u);
}

__global__ void k_cvt(uint32_t* out) {
  uint32_t r0 = cvt_pack_f32_to_f16x2(1.5f, -2.25f);
  uint32_t r1 = cvt_pack_f32_to_bf16x2(1.5f, -2.25f);
  if (threadIdx.x == 0) { out[0] = r0; out[1] = r1; }
}

__global__ void k_stmatrix(uint32_t* out) {
  __shared__ __align__(128) uint32_t smem[8 * 4];
  if (threadIdx.x < 8 * 4) smem[threadIdx.x] = 0;
  __syncthreads();
  int row = threadIdx.x / 4;
  uint32_t addr = smem_ptr_u32(&smem[row * 4]);
  uint32_t r0 = 0x11110000u + threadIdx.x;
  stmatrix_x1(addr, r0);
  __syncthreads();
  if (threadIdx.x < 8 * 4) out[threadIdx.x] = smem[threadIdx.x];
}

__global__ void __cluster_dims__(1,1,1) k_pipeline_bw_cycle(int* out) {
  __shared__ __align__(16) uint64_t full[2], empty[2];
  __shared__ __align__(16) uint64_t acc_full[2];
  BlackwellPipelineBars b{
      full, empty,
      acc_full, /*acc_empty=*/nullptr,
      /*clc_full=*/nullptr, /*clc_empty=*/nullptr,
      /*throttle_full=*/nullptr, /*throttle_empty=*/nullptr
  };
  BlackwellPipelineArriveCounts<2> a{};
  a.full = 1; a.empty = 1; a.acc_full = 1;
  pipeline_init_blackwell<2, /*CTA_GROUP=*/2>(b, a);
  if (threadIdx.x == 0) {
    for (auto* m : { &full[0], &full[1], &empty[0], &empty[1], &acc_full[0] }) {
      mbarrier_arrive(smem_ptr_u32(m));
      mbarrier_wait_parity(smem_ptr_u32(m), 0);
    }
    *out = 1;
  }
}

__global__ void k_softmax(const float* s, int K, float* om, float* ol) {
  SoftmaxState state; softmax_state_init(state);
  for (int k = 0; k < K; k += 32) {
    float x = (k + threadIdx.x < K) ? s[k + threadIdx.x] : -INFINITY;
    float m = x;
    for (int off = 16; off > 0; off >>= 1)
      m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, off));
    float se = exp2f(x - m);
    for (int off = 16; off > 0; off >>= 1)
      se += __shfl_xor_sync(0xFFFFFFFFu, se, off);
    softmax_state_update(state, m, se);
  }
  if (threadIdx.x == 0) { *om = state.m; *ol = state.l; }
}

// ---- generic timing harness ----

struct Bench { const char* name; double min_us; double avg_us; };

template <typename Launch>
Bench measure(const char* name, Launch launch) {
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  for (int i = 0; i < N_WARMUP; ++i) launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  double min_us = 1e18;
  double total_us = 0;
  for (int i = 0; i < N_ITERS; ++i) {
    CUDA_CHECK(cudaEventRecord(a));
    launch();
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    double us = ms * 1e3;
    if (us < min_us) min_us = us;
    total_us += us;
  }
  cudaEventDestroy(a); cudaEventDestroy(b);
  return { name, min_us, total_us / N_ITERS };
}

int main() {
  // Allocate scratch buffers once.
  uint32_t* d_cnt = nullptr;
  uint32_t* d_u32 = nullptr;
  float*    d_f32 = nullptr;
  int*      d_int = nullptr;
  CUDA_CHECK(cudaMalloc(&d_cnt, 4));
  CUDA_CHECK(cudaMalloc(&d_u32, 1024));
  CUDA_CHECK(cudaMalloc(&d_f32, 4096));
  CUDA_CHECK(cudaMalloc(&d_int, 4));

  // TMA tensormap
  float* d_tensor = nullptr;
  CUDA_CHECK(cudaMalloc(&d_tensor, 16 * 16 * 4));
  CUtensorMap tm_load, tm_store;
  CUDA_CHECK(make_tma_2d_tiled(&tm_load,  d_tensor, 16, 16, 16, 16,
                                 sizeof(float),
                                 CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                                 CU_TENSOR_MAP_SWIZZLE_NONE));
  CUDA_CHECK(make_tma_2d_tiled(&tm_store, d_tensor, 16, 16, 16, 16,
                                 sizeof(float),
                                 CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                                 CU_TENSOR_MAP_SWIZZLE_NONE));

  // Softmax input
  const int K = 128;
  float* d_sm_in = nullptr;
  float* d_sm_m  = nullptr;
  float* d_sm_l  = nullptr;
  CUDA_CHECK(cudaMalloc(&d_sm_in, K * 4));
  CUDA_CHECK(cudaMalloc(&d_sm_m, 4));
  CUDA_CHECK(cudaMalloc(&d_sm_l, 4));

  std::vector<Bench> results;
  results.push_back(measure("tmem.alloc+relinquish+dealloc",
      [&]{ k_tmem_alloc_dealloc<<<1, 128>>>(); }));
  results.push_back(measure("tmem.st+wait+ld+wait (32x8 regs)",
      [&]{ k_tmem_ld_st_roundtrip<<<1, 128>>>(); }));
  results.push_back(measure("tma_load_2d 16x16 FP32",
      [&]{ k_tma_load<<<1, 256>>>(tm_load, d_f32); }));
  results.push_back(measure("tma_store_2d 16x16 FP32",
      [&]{ k_tma_store<<<1, 256>>>(tm_store); }));
  results.push_back(measure("atom.global.add.u32 (32 threads)",
      [&]{ CUDA_CHECK(cudaMemsetAsync(d_cnt, 0, 4));
           k_atom_add<<<1, 32>>>(d_cnt); }));
  results.push_back(measure("redux.sync.add.u32",
      [&]{ k_redux_add<<<1, 32>>>(d_u32); }));
  results.push_back(measure("elect.sync",
      [&]{ CUDA_CHECK(cudaMemsetAsync(d_cnt, 0, 4));
           k_elect<<<1, 128>>>(d_cnt); }));
  results.push_back(measure("cvt f32->f16x2+bf16x2",
      [&]{ k_cvt<<<1, 32>>>(d_u32); }));
  results.push_back(measure("stmatrix.x1 + read-back",
      [&]{ k_stmatrix<<<1, 32>>>(d_u32); }));
  results.push_back(measure("pipeline_init_blackwell 5-barrier cycle",
      [&]{ k_pipeline_bw_cycle<<<1, 128>>>(d_int); }));
  // Init softmax input for the timed runs.
  std::vector<float> hsm(K);
  for (int i = 0; i < K; ++i) hsm[i] = (float)(i % 17) - 5.f;
  CUDA_CHECK(cudaMemcpy(d_sm_in, hsm.data(), K * 4, cudaMemcpyHostToDevice));
  results.push_back(measure("online_softmax K=128",
      [&]{ k_softmax<<<1, 32>>>(d_sm_in, K, d_sm_m, d_sm_l); }));

  cudaFree(d_cnt); cudaFree(d_u32); cudaFree(d_f32); cudaFree(d_int);
  cudaFree(d_tensor); cudaFree(d_sm_in); cudaFree(d_sm_m); cudaFree(d_sm_l);

  printf("\n=== perf on this GPU ===\n");
  printf("%-45s %10s %10s\n", "kernel", "min_us", "avg_us");
  printf("%-45s %10s %10s\n", "---", "---", "---");
  for (auto& r : results)
    printf("%-45s %10.2f %10.2f\n", r.name, r.min_us, r.avg_us);
  return 0;
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
