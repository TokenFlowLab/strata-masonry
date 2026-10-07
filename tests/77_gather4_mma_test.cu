// ARCH: sm_100a
// 77_gather4_mma_test.cu -- standalone gather4 -> SMEM -> tcgen05.mma
// integration test, 1SM (cta_group::1).
//
// Pipeline:
//   A (128 x 64) bf16 in GMEM, identity perm. gather4 loads 32 calls
//   (4 rows each) into SMEM A with SWIZZLE_128B.
//   B (8 x 64) bf16 in GMEM. Standard tile-mode TMA load into SMEM B
//   with SWIZZLE_128B.
//   tcgen05.mma.kind::f16 SS 1SM, idesc M=128 N=8 K=64, enable_d=false.
//   Computes acc[128, 8] in TMEM.
//   Warp 0 reads TMEM rows 0..31 via tcgen05.ld.32x32b.x8 and writes 8 fp32
//   per lane to D[lane, 0..7] in GMEM.
//   Host compares D[0..31, :] vs A[0..31, :] @ B[:, :]^T computed in fp32.
//
// Goal: confirm gather4 + MMA work together at the data-path level,
// independent of pipeline orchestration (mbarrier protocol, 2SM signaling).

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "test_utils.cuh"
#include "../src/primitives/0_tcgen05_alloc.cuh"
#include "../src/primitives/1_tcgen05_dealloc.cuh"
#include "../src/primitives/2_tcgen05_relinquish.cuh"
#include "../src/primitives/3_tcgen05_mma_f16.cuh"
#include "../src/primitives/8_tcgen05_mma_idesc.cuh"
#include "../src/primitives/9_tcgen05_ld.cuh"
#include "../src/primitives/11_tcgen05_commit.cuh"
#include "../src/primitives/12_tcgen05_wait.cuh"
#include "../src/primitives/23_tma_tensormap.cuh"
#include "../src/primitives/29_mbarrier_init.cuh"
#include "../src/primitives/31_mbarrier_arrive_tx.cuh"
#include "../src/primitives/33_mbarrier_try_wait.cuh"
#include "../src/primitives/42_smem_desc_blackwell.cuh"
#include "../src/primitives/70_smem_ptr.cuh"
#include "../src/primitives/73_tma_load_2d_gather4.cuh"
#include "../src/primitives/18_tma_load.cuh"

constexpr int M = 128;
constexpr int N = 8;
constexpr int K = 64;
constexpr int N_GATHER = M / 4;   // = 32 gather4 calls
constexpr int A_BYTES = M * K * (int)sizeof(__nv_bfloat16);  // 16 KB
constexpr int B_BYTES = N * K * (int)sizeof(__nv_bfloat16);  //  1 KB

__device__ __constant__ int d_perm[M];  // identity in main()

__global__ void k_gather4_mma(const __grid_constant__ CUtensorMap tmap_a,
                              const __grid_constant__ CUtensorMap tmap_b,
                              float* d_out) {
  __shared__ __align__(1024) __nv_bfloat16 smA[M * K];
  __shared__ __align__(1024) __nv_bfloat16 smB[N * K];
  __shared__ __align__(16)   uint32_t      slot;
  __shared__ __align__(16)   uint64_t      load_bar;
  __shared__ __align__(16)   uint64_t      mma_bar;

  if (threadIdx.x == 0) {
    slot = 0;
    mbarrier_init(smem_ptr_u32(&load_bar), 1);
    mbarrier_init(smem_ptr_u32(&mma_bar),  1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();

  // TMEM allocation.
  if (threadIdx.x < 32) {
    tcgen05_alloc<1>(smem_ptr_u32(&slot), 128);
    tcgen05_relinquish_alloc_permit<1>();
  }
  __syncthreads();
  uint32_t tmem_base = slot;

  // === LOAD: 32 gather4 + 1 B TMA ===
  if (threadIdx.x == 0) {
    mbarrier_arrive_expect_tx(smem_ptr_u32(&load_bar),
                              (uint32_t)(A_BYTES + B_BYTES));
    #pragma unroll
    for (int g = 0; g < N_GATHER; ++g) {
      const int r0 = d_perm[g * 4 + 0];
      const int r1 = d_perm[g * 4 + 1];
      const int r2 = d_perm[g * 4 + 2];
      const int r3 = d_perm[g * 4 + 3];
      const uint32_t dst_a = smem_ptr_u32(smA)
                           + (uint32_t)(g * 4 * K * (int)sizeof(__nv_bfloat16));
      tma_load_2d_gather4(dst_a, &tmap_a, smem_ptr_u32(&load_bar),
                          /*col=*/0, r0, r1, r2, r3);
    }
    tma_load_2d(smem_ptr_u32(smB), &tmap_b, smem_ptr_u32(&load_bar),
                /*x=*/0, /*y=*/0);
  }
  mbarrier_wait_parity(smem_ptr_u32(&load_bar), 0);

  // === MMA ===
  if (threadIdx.x == 0) {
    // SMEM descriptors: SWIZZLE_128B matches the TMA descriptor's swizzle.
    // SBO/LBO conventions per CUTLASS / primitive 42.
    constexpr uint32_t A_LBO = 16;
    constexpr uint32_t A_SBO = 1024;
    constexpr uint32_t B_LBO = 16;
    constexpr uint32_t B_SBO = 1024;
    const uint64_t desc_a = build_smem_desc_blackwell(
        smem_ptr_u32(smA), A_SBO, A_LBO, SmemSwizzleBlackwell::B128);
    const uint64_t desc_b = build_smem_desc_blackwell(
        smem_ptr_u32(smB), B_SBO, B_LBO, SmemSwizzleBlackwell::B128);
    uint32_t idesc = make_idesc_bf16_f32(M, N);
    // K=64, K_ATOM_K=16 -> 4 atoms. Each atom advances the SMEM
    // descriptor by 2 units (= 32 bytes) along K.
    constexpr int K_ATOMS = K / 16;
    #pragma unroll
    for (int ki = 0; ki < K_ATOMS; ++ki) {
      const bool enable_d = (ki != 0);
      tcgen05_mma_f16_ss<1>(tmem_base,
                            desc_a + 2 * ki,
                            desc_b + 2 * ki,
                            idesc, enable_d);
    }
    tcgen05_commit<1>(smem_ptr_u32(&mma_bar));
  }
  mbarrier_wait_parity(smem_ptr_u32(&mma_bar), 0);
  __syncthreads();

  // === READ TMEM rows 0..31 via warp 0 ===
  if (threadIdx.x < 32) {
    const int lane = (int)threadIdx.x;
    uint32_t regs[N];  // N=8 fp32
    tcgen05_ld_32x32b_x8(tmem_base, regs);
    tcgen05_wait_ld();
    // Write D[lane, 0..7] = regs[0..7].
    #pragma unroll
    for (int n = 0; n < N; ++n) {
      d_out[lane * N + n] = __int_as_float(regs[n]);
    }
  }
  __syncthreads();

  if (threadIdx.x < 32) {
    tcgen05_dealloc<1>(tmem_base, 128);
  }
}

int main() {
  CUDA_CHECK(cudaFree(0));

  // Init A, B with random integer values to keep bf16 exact.
  std::vector<__nv_bfloat16> hA(M * K), hB(N * K);
  for (int i = 0; i < M * K; ++i)
    hA[i] = __float2bfloat16((float)((i * 31 + 1) % 7 - 3));  // {-3..3}
  for (int i = 0; i < N * K; ++i)
    hB[i] = __float2bfloat16((float)((i * 37 + 2) % 7 - 3));

  __nv_bfloat16 *dA = nullptr, *dB = nullptr;
  float* dOut = nullptr;
  CUDA_CHECK(cudaMalloc(&dA, A_BYTES));
  CUDA_CHECK(cudaMalloc(&dB, B_BYTES));
  CUDA_CHECK(cudaMalloc(&dOut, 32 * N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), A_BYTES, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), B_BYTES, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dOut, 0, 32 * N * sizeof(float)));

  // Identity perm.
  std::vector<int> h_perm(M);
  for (int i = 0; i < M; ++i) h_perm[i] = i;
  CUDA_CHECK(cudaMemcpyToSymbol(d_perm, h_perm.data(), M * sizeof(int)));

  // gather4 descriptor for A: box=(1, K).
  CUtensorMap tmap_a;
  CUDA_CHECK(make_tma_2d_tiled(&tmap_a, dA, M, K,
                               /*box_rows=*/1, /*box_cols=*/K,
                               sizeof(__nv_bfloat16),
                               CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                               CU_TENSOR_MAP_SWIZZLE_128B));
  // Standard tile descriptor for B: box=(N, K).
  CUtensorMap tmap_b;
  CUDA_CHECK(make_tma_2d_tiled(&tmap_b, dB, N, K,
                               /*box_rows=*/N, /*box_cols=*/K,
                               sizeof(__nv_bfloat16),
                               CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                               CU_TENSOR_MAP_SWIZZLE_128B));

  k_gather4_mma<<<1, 128>>>(tmap_a, tmap_b, dOut);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> hOut(32 * N);
  CUDA_CHECK(cudaMemcpy(hOut.data(), dOut, 32 * N * sizeof(float),
                        cudaMemcpyDeviceToHost));
  cudaFree(dA); cudaFree(dB); cudaFree(dOut);

  // Host ref: D[m, n] = sum_k A[m, k] * B[n, k] for m in 0..31, n in 0..7.
  int fails = 0;
  float max_d = 0.0f;
  for (int m = 0; m < 32; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        acc += __bfloat162float(hA[m * K + k]) *
               __bfloat162float(hB[n * K + k]);
      }
      float got = hOut[m * N + n];
      float d = fabsf(got - acc);
      if (d > 0.5f) {
        if (fails < 8)
          fprintf(stderr, "  D[%d, %d] = %.1f, want %.1f\n", m, n, got, acc);
        ++fails;
      }
      if (d > max_d) max_d = d;
    }
  }
  printf("gather4 + MMA (M=%d, N=%d, K=%d, verify first 32 rows): "
         "fails = %d / %d, max_diff = %.3f\n",
         M, N, K, fails, 32 * N, max_d);
  if (fails) FAIL("gather4 + MMA mismatch");
  PASS();
}
