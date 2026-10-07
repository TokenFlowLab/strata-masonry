#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// ARCH: sm_100a
// 88_load_warp_blackwell_test.cu -- exercises three exports of
// blocks/88_load_warp_blackwell.cuh:
//
//   1. __device__ load_warp_blackwell_block<NUM_STAGES, TILE_FLOATS>
//      (single-tile A-only smoke body; FP32 16x16 tile). Test owns the
//      __global__ wrapper, SMEM/mbar setup, and post-call inval.
//   2. __device__ load_warp_blackwell_1tile_2sm_bf16        (wpc, production
//                                                            single-tile body
//                                                            for BF16 A+B).
//   3. __device__ load_warp_blackwell_1tile_2sm_bf16<USE_L2_HINT=true>
//      (#2 with L2::evict_last hint; pairs with primitive 68 createpolicy).
//
// The full-warp-role wrapper `load_warp_blackwell_ntiles_2sm_bf16` is
// validated at the kernel level by K0 dense_gemm_bf16 (it composes CLC
// dispatch, sched/idle warps, and the acc-pipeline; smoke-mocking that
// at the block level adds infrastructure complexity without coverage
// beyond what K0 already provides).
//
// Each test launches a 2x1x1 cluster (cta_group::2) and verifies the loaded
// SMEM stage 0 contains the expected operand byte pattern after the body
// completes -- end-to-end smoke that the producer side of the load-warp
// role works (full_bar is signaled, TMA delivers bytes to peer SMEM).

#include "test_utils.cuh"
#include "../src/primitives/23_tma_tensormap.cuh"
#include "../src/primitives/68_l2cache_policy.cuh"
#include "../src/blocks/88_load_warp_blackwell.cuh"

// ============================================================================
// Test 1: load_warp_blackwell_block<NUM_STAGES, TILE_FLOATS>.
// ============================================================================
//
// The .cuh exposes the body as a __device__ template. This test owns the
// __global__ wrapper that allocates SMEM + mbar arrays, inits the mbars,
// fences, calls the block, then invalidates the mbars on the way out.
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(128, 1)
load_warp_blackwell_test_kernel(const __grid_constant__ CUtensorMap tma_A,
                                float* __restrict__ out,
                                int k_tiles) {
  WpCtx wpc = wp_ctx_init();  // WARP_PROF TraceContext (no-op without -DWARP_PROF)

  constexpr int NUM_STAGES = 2;
  constexpr int TILE_FLOATS = 16 * 16;
  __shared__ __align__(128) float smem_a[NUM_STAGES * TILE_FLOATS];
  __shared__ __align__(16)  uint64_t full_bar[NUM_STAGES];
  __shared__ __align__(16)  uint64_t empty_bar[NUM_STAGES];

  if (threadIdx.x == 0) {
    for (int s = 0; s < NUM_STAGES; ++s) {
      mbarrier_init(smem_ptr_u32(&full_bar[s]),  /*arrive_count=*/1);
      mbarrier_init(smem_ptr_u32(&empty_bar[s]), /*arrive_count=*/1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();

  load_warp_blackwell_block<NUM_STAGES, TILE_FLOATS>(wpc, tma_A, smem_a, full_bar, empty_bar, k_tiles, out);

  __syncthreads();
  if (threadIdx.x == 0) {
    for (int s = 0; s < NUM_STAGES; ++s) {
      asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n"
                   :: "r"(smem_ptr_u32(&full_bar[s])));
      asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n"
                   :: "r"(smem_ptr_u32(&empty_bar[s])));
    }
  }
}

static int test_legacy_load_warp_blackwell() {
  const int K = 32;
  std::vector<float> hA(16 * K);
  for (int i = 0; i < 16 * K; ++i) hA[i] = (float)i;

  float* dA = nullptr;
  CUDA_CHECK(cudaMalloc(&dA, 16 * K * 4));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), 16 * K * 4, cudaMemcpyHostToDevice));

  CUtensorMap desc;
  CUDA_CHECK(make_tma_2d_tiled(&desc, dA, K, 16, 16, 16, sizeof(float),
                                CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                                CU_TENSOR_MAP_SWIZZLE_NONE));

  float* dOut = nullptr;
  CUDA_CHECK(cudaMalloc(&dOut, 16 * 16 * 4));
  CUDA_CHECK(cudaMemset(dOut, 0, 16 * 16 * 4));

  load_warp_blackwell_test_kernel<<<2, 128>>>(desc, dOut, 2);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> hOut(16 * 16);
  CUDA_CHECK(cudaMemcpy(hOut.data(), dOut, 16 * 16 * 4, cudaMemcpyDeviceToHost));
  cudaFree(dA);
  cudaFree(dOut);

  int nz = 0;
  for (auto v : hOut) if (v > 0) ++nz;
  printf("test 1 (load_warp_blackwell_block)             : non-zero = %d / 256\n",
         nz);
  if (nz < 100) FAIL("expected most of the tile to be non-zero (got %d)", nz);
  return 0;
}

// ============================================================================
// Test 2: load_warp_blackwell_1tile_2sm_bf16 (wpc, single-tile BF16 A+B body).
// ============================================================================
//
// Setup: K=64, K_TILE=32 so K_BLOCKS = 2 = NUM_STAGES. Phase-1 init idiom
// makes both empty_bar.try_wait passes through (no consumer needed). Body
// issues 2 TMA cta_group::2 loads of A and 2 of B, signaling full_bar via
// expect_tx + complete_tx::bytes from each.
//
// After the body returns, the test waits full_bar[0] to confirm TMA delivered
// stage 0's bytes, then dumps the first row of A and first row of B from
// SMEM stage 0 to GMEM. Host checks the pattern matches the seeded GMEM.

static constexpr int kT2_NUM_STAGES = 2;
static constexpr int kT2_M           = 64;
static constexpr int kT2_N           = 64;
static constexpr int kT2_K           = 64;
static constexpr int kT2_K_TILE      = 32;
static constexpr int kT2_A_BYTES     = kT2_M * kT2_K_TILE * sizeof(__nv_bfloat16);
static constexpr int kT2_B_BYTES     = kT2_N * kT2_K_TILE * sizeof(__nv_bfloat16);
static constexpr int kT2_PIPE_BYTES  = kT2_NUM_STAGES * (kT2_A_BYTES + kT2_B_BYTES);
static constexpr int kT2_BAR_BYTES   = kT2_NUM_STAGES * 2 * sizeof(uint64_t);
static constexpr int kT2_SMEM_BYTES  = kT2_PIPE_BYTES + kT2_BAR_BYTES;

__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(256, 1)
test_tile_2sm_bf16_kernel(
    const __grid_constant__ CUtensorMap tmap_a,
    const __grid_constant__ CUtensorMap tmap_b,
    __nv_bfloat16* d_out_a,   // [kT2_K_TILE]
    __nv_bfloat16* d_out_b) {
  WpCtx wpc = wp_ctx_init();  // WARP_PROF TraceContext (no-op without -DWARP_PROF)
 // [kT2_K_TILE]
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t*  smem_a    = smem;
  uint8_t*  smem_b    = smem + kT2_NUM_STAGES * kT2_A_BYTES;
  uint64_t* full_bar  = reinterpret_cast<uint64_t*>(
      smem + kT2_NUM_STAGES * (kT2_A_BYTES + kT2_B_BYTES));
  uint64_t* empty_bar = full_bar + kT2_NUM_STAGES;

  const int peer = blockIdx.x & 1;
  const int warp = threadIdx.x >> 5;

  if (threadIdx.x == 0) {
    for (int s = 0; s < kT2_NUM_STAGES; ++s) {
      mbarrier_init(smem_ptr_u32(&full_bar[s]),  /*arrive_count=*/1);
      mbarrier_init(smem_ptr_u32(&empty_bar[s]), /*arrive_count=*/1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  asm volatile("barrier.cluster.arrive;\n" ::: "memory");
  asm volatile("barrier.cluster.wait;\n" ::: "memory");

  if (warp == 2) {
    EmptyPhaseTracker<kT2_NUM_STAGES> empty_ph;
    load_warp_blackwell_1tile_2sm_bf16<
        kT2_NUM_STAGES, kT2_M, kT2_N, kT2_K_TILE>(wpc, &tmap_a, &tmap_b, smem_a, smem_b, full_bar, empty_bar,
        kT2_K, /*m_offset=*/0, /*n_offset_b=*/0, peer, empty_ph);
  }

  // Wait stage 0's full_bar (consumer side) to confirm TMA delivered. Then
  // peer 0 dumps row 0 of stage-0 A and stage-0 B to GMEM for host check.
  __syncthreads();
  if (peer == 0 && threadIdx.x == 0) {
    mbarrier_wait_parity(smem_ptr_u32(&full_bar[0]), /*phase=*/0);
  }
  __syncthreads();
  if (peer == 0 && threadIdx.x < kT2_K_TILE) {
    const __nv_bfloat16* a_stage0 = reinterpret_cast<const __nv_bfloat16*>(smem_a);
    const __nv_bfloat16* b_stage0 = reinterpret_cast<const __nv_bfloat16*>(smem_b);
    d_out_a[threadIdx.x] = a_stage0[threadIdx.x];  // row 0, col 0..K_TILE-1
    d_out_b[threadIdx.x] = b_stage0[threadIdx.x];
  }

  __syncthreads();
  asm volatile("barrier.cluster.arrive;\n" ::: "memory");
  asm volatile("barrier.cluster.wait;\n" ::: "memory");
}

static int test_tile_2sm_bf16() {
  std::vector<__nv_bfloat16> hA(kT2_M * kT2_K), hBT(kT2_N * kT2_K);
  for (int m = 0; m < kT2_M; ++m)
    for (int k = 0; k < kT2_K; ++k)
      hA[m * kT2_K + k] = __float2bfloat16((float)(m * 100 + k));
  for (int n = 0; n < kT2_N; ++n)
    for (int k = 0; k < kT2_K; ++k)
      hBT[n * kT2_K + k] = __float2bfloat16((float)(n * 100 + k + 1));

  __nv_bfloat16* dA  = nullptr;
  __nv_bfloat16* dBT = nullptr;
  CUDA_CHECK(cudaMalloc(&dA,  hA.size()  * 2));
  CUDA_CHECK(cudaMalloc(&dBT, hBT.size() * 2));
  CUDA_CHECK(cudaMemcpy(dA,  hA.data(),  hA.size()  * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dBT, hBT.data(), hBT.size() * 2, cudaMemcpyHostToDevice));

  CUtensorMap tmap_a{}, tmap_b{};
  CUDA_CHECK(make_tma_2d_tiled(&tmap_a, dA, kT2_M, kT2_K, kT2_M, kT2_K_TILE,
                                sizeof(__nv_bfloat16),
                                CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                CU_TENSOR_MAP_SWIZZLE_NONE));
  CUDA_CHECK(make_tma_2d_tiled(&tmap_b, dBT, kT2_N, kT2_K, kT2_N, kT2_K_TILE,
                                sizeof(__nv_bfloat16),
                                CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                CU_TENSOR_MAP_SWIZZLE_NONE));

  __nv_bfloat16* dOutA = nullptr;
  __nv_bfloat16* dOutB = nullptr;
  CUDA_CHECK(cudaMalloc(&dOutA, kT2_K_TILE * 2));
  CUDA_CHECK(cudaMalloc(&dOutB, kT2_K_TILE * 2));
  CUDA_CHECK(cudaMemset(dOutA, 0, kT2_K_TILE * 2));
  CUDA_CHECK(cudaMemset(dOutB, 0, kT2_K_TILE * 2));

  CUDA_CHECK(cudaFuncSetAttribute(test_tile_2sm_bf16_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kT2_SMEM_BYTES));
  test_tile_2sm_bf16_kernel<<<dim3(2, 1, 1), dim3(256, 1, 1), kT2_SMEM_BYTES>>>(
      tmap_a, tmap_b, dOutA, dOutB);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> hOutA(kT2_K_TILE), hOutB(kT2_K_TILE);
  CUDA_CHECK(cudaMemcpy(hOutA.data(), dOutA, kT2_K_TILE * 2, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hOutB.data(), dOutB, kT2_K_TILE * 2, cudaMemcpyDeviceToHost));
  cudaFree(dA); cudaFree(dBT); cudaFree(dOutA); cudaFree(dOutB);

  // Expected: row 0 of A and B (first K_TILE cols).
  int mismatch = 0;
  for (int k = 0; k < kT2_K_TILE; ++k) {
    float got_a = __bfloat162float(hOutA[k]);
    float exp_a = (float)(0 * 100 + k);   // row 0
    float got_b = __bfloat162float(hOutB[k]);
    float exp_b = (float)(0 * 100 + k + 1);
    if (got_a != exp_a || got_b != exp_b) ++mismatch;
  }
  printf("test 2 (load_warp_blackwell_1tile_2sm_bf16)     : mismatches = %d / %d\n",
         mismatch, kT2_K_TILE);
  if (mismatch != 0) FAIL("tile_2sm_bf16: row 0 mismatch");
  return 0;
}

#if 0  // Test 3 removed -- ntiles wrapper validated end-to-end by K0.
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(256, 1)
test_persistent_2sm_bf16_kernel(
    const __grid_constant__ CUtensorMap tmap_a,
    const __grid_constant__ CUtensorMap tmap_b,
    __nv_bfloat16* d_out_a,
    __nv_bfloat16* d_out_b) {
  WpCtx wpc = wp_ctx_init();  // WARP_PROF TraceContext (no-op without -DWARP_PROF)

  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t*  smem_a    = smem;
  uint8_t*  smem_b    = smem + kT2_NUM_STAGES * kT2_A_BYTES;
  uint64_t* full_bar  = reinterpret_cast<uint64_t*>(
      smem + kT2_NUM_STAGES * (kT2_A_BYTES + kT2_B_BYTES));
  uint64_t* empty_bar = full_bar + kT2_NUM_STAGES;

  const int peer = blockIdx.x & 1;
  const int warp = threadIdx.x >> 5;

  if (threadIdx.x == 0) {
    for (int s = 0; s < kT2_NUM_STAGES; ++s) {
      mbarrier_init(smem_ptr_u32(&full_bar[s]),  /*arrive_count=*/1);
      mbarrier_init(smem_ptr_u32(&empty_bar[s]), /*arrive_count=*/1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  asm volatile("barrier.cluster.arrive;\n" ::: "memory");
  asm volatile("barrier.cluster.wait;\n" ::: "memory");

  // Mock consumer (LEADER CTA only). Waits leader's full_bar[s] (TMA-
  // delivered by the body's multicast complete_tx + leader's arrive_
  // expect_tx). Then arrives BOTH peers' empty_bar[s] -- mimicking the
  // production MMA's tcgen05.commit.multicast::cluster pattern. Without
  // arriving on follower's empty_bar via mapa, the follower's wrapper
  // tail-drain wait would hang. arrive_count=1 per bar, so one arrive
  // per peer per stage suffices.
  if (warp == 4 && (threadIdx.x & 31) == 0 && peer == 0) {
    for (int s = 0; s < kT2_NUM_STAGES; ++s) {
      mbarrier_wait_parity(smem_ptr_u32(&full_bar[s]), /*phase=*/0);
      // Arrive leader's local empty_bar[s].
      mbarrier_arrive_nostate(smem_ptr_u32(&empty_bar[s]));
      // Arrive follower's empty_bar[s] via mapa (cluster-shared addr).
      uint32_t local_addr = smem_ptr_u32(&empty_bar[s]);
      uint32_t remote_addr;
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n"
                   : "=r"(remote_addr) : "r"(local_addr), "r"(1));
      asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];\n"
                   :: "r"(remote_addr) : "memory");
    }
  }

  if (warp == 2) {
    auto fetch_next = [] () {
      struct R { int m_tile, n_tile; bool valid; };
      return R{0, 0, false};   // one-shot: terminate after first body call
    };
    load_warp_blackwell_ntiles_2sm_bf16<
        kT2_NUM_STAGES, kT2_M, kT2_N, kT2_K_TILE,
        /*M_TILE_CLUSTER=*/2 * kT2_M, /*N_TILE_CLUSTER=*/2 * kT2_N>(wpc, &tmap_a, &tmap_b, smem_a, smem_b, full_bar, empty_bar,
        kT2_K, peer,
        /*init_m_cluster_tile=*/0, /*init_n_cluster_tile=*/0,
        fetch_next);
  }

  __syncthreads();
  if (peer == 0 && threadIdx.x < kT2_K_TILE) {
    const __nv_bfloat16* a_stage0 = reinterpret_cast<const __nv_bfloat16*>(smem_a);
    const __nv_bfloat16* b_stage0 = reinterpret_cast<const __nv_bfloat16*>(smem_b);
    d_out_a[threadIdx.x] = a_stage0[threadIdx.x];
    d_out_b[threadIdx.x] = b_stage0[threadIdx.x];
  }

  __syncthreads();
  asm volatile("barrier.cluster.arrive;\n" ::: "memory");
  asm volatile("barrier.cluster.wait;\n" ::: "memory");
}

static int test_persistent_2sm_bf16() {
  std::vector<__nv_bfloat16> hA(kT2_M * kT2_K), hBT(kT2_N * kT2_K);
  for (int m = 0; m < kT2_M; ++m)
    for (int k = 0; k < kT2_K; ++k)
      hA[m * kT2_K + k] = __float2bfloat16((float)(m * 100 + k));
  for (int n = 0; n < kT2_N; ++n)
    for (int k = 0; k < kT2_K; ++k)
      hBT[n * kT2_K + k] = __float2bfloat16((float)(n * 100 + k + 1));

  __nv_bfloat16* dA  = nullptr;
  __nv_bfloat16* dBT = nullptr;
  CUDA_CHECK(cudaMalloc(&dA,  hA.size()  * 2));
  CUDA_CHECK(cudaMalloc(&dBT, hBT.size() * 2));
  CUDA_CHECK(cudaMemcpy(dA,  hA.data(),  hA.size()  * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dBT, hBT.data(), hBT.size() * 2, cudaMemcpyHostToDevice));

  CUtensorMap tmap_a{}, tmap_b{};
  CUDA_CHECK(make_tma_2d_tiled(&tmap_a, dA, kT2_M, kT2_K, kT2_M, kT2_K_TILE,
                                sizeof(__nv_bfloat16),
                                CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                CU_TENSOR_MAP_SWIZZLE_NONE));
  CUDA_CHECK(make_tma_2d_tiled(&tmap_b, dBT, kT2_N, kT2_K, kT2_N, kT2_K_TILE,
                                sizeof(__nv_bfloat16),
                                CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                CU_TENSOR_MAP_SWIZZLE_NONE));

  __nv_bfloat16* dOutA = nullptr;
  __nv_bfloat16* dOutB = nullptr;
  CUDA_CHECK(cudaMalloc(&dOutA, kT2_K_TILE * 2));
  CUDA_CHECK(cudaMalloc(&dOutB, kT2_K_TILE * 2));
  CUDA_CHECK(cudaMemset(dOutA, 0, kT2_K_TILE * 2));
  CUDA_CHECK(cudaMemset(dOutB, 0, kT2_K_TILE * 2));

  CUDA_CHECK(cudaFuncSetAttribute(test_persistent_2sm_bf16_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kT2_SMEM_BYTES));
  test_persistent_2sm_bf16_kernel<<<
      dim3(2, 1, 1), dim3(256, 1, 1), kT2_SMEM_BYTES>>>(
      tmap_a, tmap_b, dOutA, dOutB);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> hOutA(kT2_K_TILE), hOutB(kT2_K_TILE);
  CUDA_CHECK(cudaMemcpy(hOutA.data(), dOutA, kT2_K_TILE * 2, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hOutB.data(), dOutB, kT2_K_TILE * 2, cudaMemcpyDeviceToHost));
  cudaFree(dA); cudaFree(dBT); cudaFree(dOutA); cudaFree(dOutB);

  int mismatch = 0;
  for (int k = 0; k < kT2_K_TILE; ++k) {
    float got_a = __bfloat162float(hOutA[k]);
    float exp_a = (float)(0 * 100 + k);
    float got_b = __bfloat162float(hOutB[k]);
    float exp_b = (float)(0 * 100 + k + 1);
    if (got_a != exp_a || got_b != exp_b) ++mismatch;
  }
  printf("test 3 (load_warp_blackwell_ntiles_2sm_bf16): mismatches = %d / %d\n",
         mismatch, kT2_K_TILE);
  if (mismatch != 0) FAIL("persistent_2sm_bf16: row 0 mismatch");
  return 0;
}
#endif  // disabled test 3

// ============================================================================
// Test 4: load_warp_blackwell_1tile_2sm_bf16 with USE_L2_HINT=true.
// ============================================================================
//
// Same shape as test 2 (K=64, K_TILE=32, 2 stages). Builds a real
// L2::evict_last cache-policy via primitive 68 in the kernel prologue,
// then calls the _1tile_ body with `USE_L2_HINT=true` so the inner TMA
// loads emit `cp.async.bulk.tensor.*.L2::cache_hint` and consume the
// policy. The bytes delivered must still match the seeded GMEM
// pattern -- the hint affects L2 eviction priority, not transport
// correctness. This is the producer-consumer end-to-end smoke for
// step 2's createpolicy primitive paired with primitive 19's
// tma_load_2d_2sm_l2hint.

__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(256, 1)
test_tile_2sm_bf16_l2hint_kernel(
    const __grid_constant__ CUtensorMap tmap_a,
    const __grid_constant__ CUtensorMap tmap_b,
    __nv_bfloat16* d_out_a,
    __nv_bfloat16* d_out_b) {
  WpCtx wpc = wp_ctx_init();  // WARP_PROF TraceContext (no-op without -DWARP_PROF)

  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t*  smem_a    = smem;
  uint8_t*  smem_b    = smem + kT2_NUM_STAGES * kT2_A_BYTES;
  uint64_t* full_bar  = reinterpret_cast<uint64_t*>(
      smem + kT2_NUM_STAGES * (kT2_A_BYTES + kT2_B_BYTES));
  uint64_t* empty_bar = full_bar + kT2_NUM_STAGES;

  const int peer = blockIdx.x & 1;
  const int warp = threadIdx.x >> 5;

  if (threadIdx.x == 0) {
    for (int s = 0; s < kT2_NUM_STAGES; ++s) {
      mbarrier_init(smem_ptr_u32(&full_bar[s]),  /*arrive_count=*/1);
      mbarrier_init(smem_ptr_u32(&empty_bar[s]), /*arrive_count=*/1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  asm volatile("barrier.cluster.arrive;\n" ::: "memory");
  asm volatile("barrier.cluster.wait;\n" ::: "memory");

  if (warp == 2) {
    // (stale-call fix, pre-existing rot: cache_policy moved into the template
    // L2CACHE_POLICY_* args long ago; the trailing runtime arg never matched.)
    EmptyPhaseTracker<kT2_NUM_STAGES> empty_ph;
    load_warp_blackwell_1tile_2sm_bf16<
        kT2_NUM_STAGES, kT2_M, kT2_N, kT2_K_TILE,
        /*USE_L2_HINT=*/true>(wpc, &tmap_a, &tmap_b, smem_a, smem_b, full_bar, empty_bar,
        kT2_K, /*m_offset=*/0, /*n_offset_b=*/0, peer, empty_ph);
  }

  __syncthreads();
  if (peer == 0 && threadIdx.x == 0) {
    mbarrier_wait_parity(smem_ptr_u32(&full_bar[0]), /*phase=*/0);
  }
  __syncthreads();
  if (peer == 0 && threadIdx.x < kT2_K_TILE) {
    const __nv_bfloat16* a_stage0 = reinterpret_cast<const __nv_bfloat16*>(smem_a);
    const __nv_bfloat16* b_stage0 = reinterpret_cast<const __nv_bfloat16*>(smem_b);
    d_out_a[threadIdx.x] = a_stage0[threadIdx.x];
    d_out_b[threadIdx.x] = b_stage0[threadIdx.x];
  }

  __syncthreads();
  asm volatile("barrier.cluster.arrive;\n" ::: "memory");
  asm volatile("barrier.cluster.wait;\n" ::: "memory");
}

static int test_tile_2sm_bf16_l2hint() {
  std::vector<__nv_bfloat16> hA(kT2_M * kT2_K), hBT(kT2_N * kT2_K);
  for (int m = 0; m < kT2_M; ++m)
    for (int k = 0; k < kT2_K; ++k)
      hA[m * kT2_K + k] = __float2bfloat16((float)(m * 100 + k));
  for (int n = 0; n < kT2_N; ++n)
    for (int k = 0; k < kT2_K; ++k)
      hBT[n * kT2_K + k] = __float2bfloat16((float)(n * 100 + k + 1));

  __nv_bfloat16* dA  = nullptr;
  __nv_bfloat16* dBT = nullptr;
  CUDA_CHECK(cudaMalloc(&dA,  hA.size()  * 2));
  CUDA_CHECK(cudaMalloc(&dBT, hBT.size() * 2));
  CUDA_CHECK(cudaMemcpy(dA,  hA.data(),  hA.size()  * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dBT, hBT.data(), hBT.size() * 2, cudaMemcpyHostToDevice));

  CUtensorMap tmap_a{}, tmap_b{};
  CUDA_CHECK(make_tma_2d_tiled(&tmap_a, dA, kT2_M, kT2_K, kT2_M, kT2_K_TILE,
                                sizeof(__nv_bfloat16),
                                CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                CU_TENSOR_MAP_SWIZZLE_NONE));
  CUDA_CHECK(make_tma_2d_tiled(&tmap_b, dBT, kT2_N, kT2_K, kT2_N, kT2_K_TILE,
                                sizeof(__nv_bfloat16),
                                CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                CU_TENSOR_MAP_SWIZZLE_NONE));

  __nv_bfloat16* dOutA = nullptr;
  __nv_bfloat16* dOutB = nullptr;
  CUDA_CHECK(cudaMalloc(&dOutA, kT2_K_TILE * 2));
  CUDA_CHECK(cudaMalloc(&dOutB, kT2_K_TILE * 2));
  CUDA_CHECK(cudaMemset(dOutA, 0, kT2_K_TILE * 2));
  CUDA_CHECK(cudaMemset(dOutB, 0, kT2_K_TILE * 2));

  CUDA_CHECK(cudaFuncSetAttribute(test_tile_2sm_bf16_l2hint_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kT2_SMEM_BYTES));
  test_tile_2sm_bf16_l2hint_kernel<<<dim3(2, 1, 1), dim3(256, 1, 1), kT2_SMEM_BYTES>>>(
      tmap_a, tmap_b, dOutA, dOutB);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> hOutA(kT2_K_TILE), hOutB(kT2_K_TILE);
  CUDA_CHECK(cudaMemcpy(hOutA.data(), dOutA, kT2_K_TILE * 2, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hOutB.data(), dOutB, kT2_K_TILE * 2, cudaMemcpyDeviceToHost));
  cudaFree(dA); cudaFree(dBT); cudaFree(dOutA); cudaFree(dOutB);

  int mismatch = 0;
  for (int k = 0; k < kT2_K_TILE; ++k) {
    float got_a = __bfloat162float(hOutA[k]);
    float exp_a = (float)(0 * 100 + k);
    float got_b = __bfloat162float(hOutB[k]);
    float exp_b = (float)(0 * 100 + k + 1);
    if (got_a != exp_a || got_b != exp_b) ++mismatch;
  }
  printf("test 4 (load_warp_blackwell_1tile_2sm_bf16 l2hint): mismatches = %d / %d\n",
         mismatch, kT2_K_TILE);
  if (mismatch != 0) FAIL("tile_2sm_bf16_l2hint: row 0 mismatch");
  return 0;
}

// ============================================================================
// Test 5: load_warp_blackwell_ntiles_2sm_bf16_groupedgemm.
// ============================================================================
//
// Validates the grouped-GEMM wrapper's unique behavior: per-tile expert
// decode via find_group_id (composite 83) + folding expert_id into B's
// leading axis (n_offset_b). The ntiles wrapper drives an internal CLC
// loop, so we mock the scaffolding to terminate after one tile:
//
//   - clc_full_bar[0] pre-arrived (flips to phase 1) so the first CLC
//     wait passes. clc_response[2] low bit cleared -> valid=false ->
//     wrapper breaks out of its loop after one tile.
//   - throttle_full/empty pre-init only (phase-1 init idiom makes the
//     first throttle wait pass through).
//   - Mock MMA on peer 0 warp 4 arrives empty_bar[s] on BOTH peers' SMEM
//     after full_bar[s] fires (mimicking tcgen05.commit.multicast::
//     cluster); without this the wrapper's tail drain would hang.
//
// Problem: E=3, cumul={0, 2, 3} so find_group_id(0, cumul, 3) == 1.
// Grid: (2,1,1) cluster, gridDim=(2,1,1). m_tile_p1 = blockIdx.y = 0.
// Peer 0 -> n_offset_b = expert_id*N + 0 + 0 = N (= start of expert 1's
// B region). Peer 1 -> n_offset_b = N + N_TILE_PER_CTA.
//
// B GMEM layout (E*N rows x K): expert e fills rows [e*N, (e+1)*N).
// We seed each expert's rows with a distinct float scale (10*e + col)
// so a mis-decoded load (e.g., reading expert 0 or expert 2 instead of
// 1) shows up as a magnitude mismatch on the dumped row.

static constexpr int kTG_NUM_STAGES = 2;
static constexpr int kTG_E          = 3;
static constexpr int kTG_M          = 64;
static constexpr int kTG_N          = 64;
static constexpr int kTG_K          = 64;
static constexpr int kTG_K_TILE     = 32;
static constexpr int kTG_M_CLUSTER  = 128;
static constexpr int kTG_N_CLUSTER  = 128;
static constexpr int kTG_A_BYTES    = kTG_M * kTG_K_TILE * sizeof(__nv_bfloat16);
static constexpr int kTG_B_BYTES    = kTG_N * kTG_K_TILE * sizeof(__nv_bfloat16);

__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(256, 1)
test_tile_2sm_bf16_groupedgemm_kernel(
    const __grid_constant__ CUtensorMap tmap_a,
    const __grid_constant__ CUtensorMap tmap_b,
    __nv_bfloat16* d_out_b) {
  WpCtx wpc = wp_ctx_init();  // WARP_PROF TraceContext (no-op without -DWARP_PROF)
  // peer 0 dumps stage-0 B row 0 here
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t*  smem_a    = smem;
  uint8_t*  smem_b    = smem + kTG_NUM_STAGES * kTG_A_BYTES;
  uint64_t* bars      = reinterpret_cast<uint64_t*>(
      smem + kTG_NUM_STAGES * (kTG_A_BYTES + kTG_B_BYTES));
  uint64_t* full_bar       = bars + 0;
  uint64_t* empty_bar      = bars + kTG_NUM_STAGES;
  uint64_t* throttle_full  = bars + 2 * kTG_NUM_STAGES + 0;
  uint64_t* throttle_empty = bars + 2 * kTG_NUM_STAGES + 2;
  uint64_t* clc_full_bar   = bars + 2 * kTG_NUM_STAGES + 4;
  uint64_t* clc_empty_bar  = bars + 2 * kTG_NUM_STAGES + 6;
  uint32_t* clc_response   = reinterpret_cast<uint32_t*>(
      bars + 2 * kTG_NUM_STAGES + 8);
  int*      expert_cumul   = reinterpret_cast<int*>(clc_response + 8);

  const int peer = blockIdx.x & 1;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

  if (threadIdx.x == 0) {
    for (int s = 0; s < kTG_NUM_STAGES; ++s) {
      mbarrier_init(smem_ptr_u32(&full_bar[s]),       /*count=*/1);
      mbarrier_init(smem_ptr_u32(&empty_bar[s]),      /*count=*/1);
      // throttle_full: all 32 lanes of the load warp arrive (matches
      // production canonical count in composite 70).
      mbarrier_init(smem_ptr_u32(&throttle_full[s]),  /*count=*/32);
      mbarrier_init(smem_ptr_u32(&throttle_empty[s]), /*count=*/1);
    }
    for (int s = 0; s < 2; ++s) {
      mbarrier_init(smem_ptr_u32(&clc_full_bar[s]),   /*count=*/1);
      // clc_empty: each peer's elect_one_sync arrives once on peer 0's
      // bar via clc_consumer_release; count=2 for the 2-peer cluster.
      mbarrier_init(smem_ptr_u32(&clc_empty_bar[s]),  /*count=*/2);
    }
    for (int i = 0; i < 8; ++i) clc_response[i] = 0;  // valid=0
    expert_cumul[0] = 0;  // -> find_group_id(0, .., 3) == 1
    expert_cumul[1] = 2;
    expert_cumul[2] = 3;
    // Pre-arrive clc_full_bar[0] so wrapper's first CLC wait passes.
    mbarrier_arrive_nostate(smem_ptr_u32(&clc_full_bar[0]));
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  asm volatile("barrier.cluster.arrive;\n" ::: "memory");
  asm volatile("barrier.cluster.wait;\n" ::: "memory");

  // Mock MMA (peer 0 only). For each of the NUM_STAGES K-loop iters,
  // wait full_bar[s] phase 0 (TMA-delivered), then arrive empty_bar[s]
  // on BOTH peers via mapa so the wrapper's tail drain on peer 1 also
  // unblocks.
  if (warp == 4 && lane == 0 && peer == 0) {
    for (int s = 0; s < kTG_NUM_STAGES; ++s) {
      mbarrier_wait_parity(smem_ptr_u32(&full_bar[s]), /*phase=*/0);
      mbarrier_arrive_nostate(smem_ptr_u32(&empty_bar[s]));
      uint32_t local_addr = smem_ptr_u32(&empty_bar[s]);
      uint32_t remote_addr;
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n"
                   : "=r"(remote_addr) : "r"(local_addr), "r"(1));
      asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];\n"
                   :: "r"(remote_addr) : "memory");
    }
  }

  if (warp == 2) {
    load_warp_blackwell_ntiles_2sm_bf16<
        kTG_NUM_STAGES, kTG_M, kTG_N, kTG_K_TILE,
        kTG_M_CLUSTER, kTG_N_CLUSTER,
        /*LOAD_REG_BUDGET=*/40, /*USE_L2_HINT=*/true,
        /*CLUSTER_SHAPE_M=*/1, /*CLUSTER_SHAPE_N=*/2,
        ClcRasterOrder::AlongN,
        /*GROUPED_GEMM=*/true,
        /*L2CACHE_POLICY_A=*/0, /*L2CACHE_POLICY_B=*/0>(wpc, &tmap_a, &tmap_b, smem_a, smem_b,
        full_bar, empty_bar,
        clc_full_bar, clc_empty_bar, clc_response,
        throttle_full, throttle_empty,
        kTG_K, peer,
        kTG_N, kTG_E, expert_cumul, /*m_tile_remap=*/nullptr);
  }

  __syncthreads();
  if (peer == 0 && threadIdx.x < kTG_K_TILE) {
    const __nv_bfloat16* b_stage0 = reinterpret_cast<const __nv_bfloat16*>(smem_b);
    d_out_b[threadIdx.x] = b_stage0[threadIdx.x];
  }
  __syncthreads();
  asm volatile("barrier.cluster.arrive;\n" ::: "memory");
  asm volatile("barrier.cluster.wait;\n" ::: "memory");
}

static int test_tile_2sm_bf16_groupedgemm() {
  // A: shape (M_CLUSTER, K) = (128, 64). Seed irrelevant (only B is checked).
  std::vector<__nv_bfloat16> hA(kTG_M_CLUSTER * kTG_K);
  for (int i = 0; i < (int)hA.size(); ++i)
    hA[i] = __float2bfloat16((float)(i & 1023));

  // B: shape (E*N, K) = (192, 64). Expert e fills rows [e*N, (e+1)*N) with
  // a distinct pattern (10*e + col, so col 0 of expert 1 = 10, col 1 = 11,
  // col 0 of expert 2 = 20, etc.).
  std::vector<__nv_bfloat16> hBT(kTG_E * kTG_N * kTG_K);
  for (int e = 0; e < kTG_E; ++e)
    for (int n = 0; n < kTG_N; ++n)
      for (int k = 0; k < kTG_K; ++k)
        hBT[(e * kTG_N + n) * kTG_K + k] =
            __float2bfloat16((float)(10 * e + k));

  __nv_bfloat16 *dA = nullptr, *dBT = nullptr, *dOutB = nullptr;
  CUDA_CHECK(cudaMalloc(&dA,  hA.size()  * 2));
  CUDA_CHECK(cudaMalloc(&dBT, hBT.size() * 2));
  CUDA_CHECK(cudaMalloc(&dOutB, kTG_K_TILE * 2));
  CUDA_CHECK(cudaMemcpy(dA,  hA.data(),  hA.size()  * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dBT, hBT.data(), hBT.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dOutB, 0, kTG_K_TILE * 2));

  CUtensorMap tmap_a{}, tmap_b{};
  CUDA_CHECK(make_tma_2d_tiled(&tmap_a, dA, kTG_M_CLUSTER, kTG_K,
                                kTG_M, kTG_K_TILE,
                                sizeof(__nv_bfloat16),
                                CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                CU_TENSOR_MAP_SWIZZLE_NONE));
  CUDA_CHECK(make_tma_2d_tiled(&tmap_b, dBT, kTG_E * kTG_N, kTG_K,
                                kTG_N, kTG_K_TILE,
                                sizeof(__nv_bfloat16),
                                CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                CU_TENSOR_MAP_SWIZZLE_NONE));

  // SMEM: stages + 8 bars (full[2]+empty[2]+thr_full[2]+thr_empty[2]+clc_full[2]
  // +clc_empty[2]) + clc_response (8*u32) + expert_cumul (3*i32) + alignment.
  const int smem_bytes = kTG_NUM_STAGES * (kTG_A_BYTES + kTG_B_BYTES)
                         + 12 * (int)sizeof(uint64_t)
                         + 8  * (int)sizeof(uint32_t)
                         + kTG_E * (int)sizeof(int)
                         + 128;

  CUDA_CHECK(cudaFuncSetAttribute(test_tile_2sm_bf16_groupedgemm_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
  test_tile_2sm_bf16_groupedgemm_kernel<<<
      dim3(2, 1, 1), dim3(256, 1, 1), smem_bytes>>>(tmap_a, tmap_b, dOutB);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> hOutB(kTG_K_TILE);
  CUDA_CHECK(cudaMemcpy(hOutB.data(), dOutB, kTG_K_TILE * 2,
                        cudaMemcpyDeviceToHost));
  cudaFree(dA); cudaFree(dBT); cudaFree(dOutB);

  // Peer 0 with cumul={0,2,3} -> expert_id=1, n_offset_b=N. So SMEM stage 0
  // row 0 should match expert 1's row 0 (cols 0..K_TILE-1): value = 10 + k.
  int mismatch = 0;
  for (int k = 0; k < kTG_K_TILE; ++k) {
    float got = __bfloat162float(hOutB[k]);
    float exp = (float)(10 * 1 + k);
    if (got != exp) ++mismatch;
  }
  printf("test 5 (load_warp_blackwell_ntiles_2sm_bf16 GROUPED_GEMM=true): mismatches = %d / %d\n",
         mismatch, kTG_K_TILE);
  if (mismatch != 0) FAIL("groupedgemm: expert-1 row 0 mismatch");
  return 0;
}

int main() {
  CUDA_CHECK(cudaFree(0));
  if (int rc = test_legacy_load_warp_blackwell(); rc != 0) return rc;
  if (int rc = test_tile_2sm_bf16();              rc != 0) return rc;
  if (int rc = test_tile_2sm_bf16_l2hint();       rc != 0) return rc;
  if (int rc = test_tile_2sm_bf16_groupedgemm();  rc != 0) return rc;
  PASS();
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
