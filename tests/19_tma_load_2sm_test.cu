#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// ARCH: sm_100a
// 19_tma_load_2sm_test.cu -- 2SM TMA load across a 2-CTA cluster.

#include <vector>
#include "test_utils.cuh"
#include "../src/primitives/19_tma_load_2sm.cuh"
#include "../src/primitives/23_tma_tensormap.cuh"
#include "../src/primitives/29_mbarrier_init.cuh"
#include "../src/primitives/31_mbarrier_arrive_tx.cuh"
#include "../src/primitives/33_mbarrier_try_wait.cuh"

__global__ void __cluster_dims__(2, 1, 1)
k_tma2sm(const __grid_constant__ CUtensorMap desc, float* out) {
  __shared__ __align__(128) float smem[16 * 16];
  __shared__ __align__(16)  uint64_t mbar;
  if (threadIdx.x == 0) {
    mbarrier_init(smem_ptr_u32(&mbar), 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    mbarrier_arrive_expect_tx(smem_ptr_u32(&mbar),
                                16 * 16 * sizeof(float));
    uint32_t mbar_masked = tma_peer_bit_mask(smem_ptr_u32(&mbar));
    tma_load_2d_2sm(smem_ptr_u32(smem), &desc, mbar_masked, 0, 0);
  }
  if (blockIdx.x == 0)
    mbarrier_wait_parity(smem_ptr_u32(&mbar), 0);
  if (blockIdx.x == 0 && threadIdx.x < 16 * 16)
    out[threadIdx.x] = smem[threadIdx.x];
  __syncthreads();
  if (threadIdx.x == 0)
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n"
                 :: "r"(smem_ptr_u32(&mbar)));
}

// 2SM TMA load with .L2::cache_hint -- same shape as k_tma2sm but
// exercises the cache-policy operand.
__global__ void __cluster_dims__(2, 1, 1)
k_tma2sm_l2hint(const __grid_constant__ CUtensorMap desc, float* out,
                uint64_t cache_policy) {
  __shared__ __align__(128) float smem[16 * 16];
  __shared__ __align__(16)  uint64_t mbar;
  if (threadIdx.x == 0) {
    mbarrier_init(smem_ptr_u32(&mbar), 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    mbarrier_arrive_expect_tx(smem_ptr_u32(&mbar),
                                16 * 16 * sizeof(float));
    uint32_t mbar_masked = tma_peer_bit_mask(smem_ptr_u32(&mbar));
    tma_load_2d_2sm_l2hint(smem_ptr_u32(smem), &desc, mbar_masked,
                           0, 0, cache_policy);
  }
  if (blockIdx.x == 0)
    mbarrier_wait_parity(smem_ptr_u32(&mbar), 0);
  if (blockIdx.x == 0 && threadIdx.x < 16 * 16)
    out[threadIdx.x] = smem[threadIdx.x];
  __syncthreads();
  if (threadIdx.x == 0)
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n"
                 :: "r"(smem_ptr_u32(&mbar)));
}

int main() {
  std::vector<float> hIn(16 * 16);
  for (int i = 0; i < 16 * 16; ++i) hIn[i] = (float)i;
  float* dIn = nullptr; CUDA_CHECK(cudaMalloc(&dIn, 16 * 16 * 4));
  CUDA_CHECK(cudaMemcpy(dIn, hIn.data(), 16 * 16 * 4,
                         cudaMemcpyHostToDevice));

  CUtensorMap desc;
  CUDA_CHECK(make_tma_2d_tiled(&desc, dIn, 16, 16, 16, 16, sizeof(float),
                                 CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                                 CU_TENSOR_MAP_SWIZZLE_NONE));

  float* dOut = nullptr; CUDA_CHECK(cudaMalloc(&dOut, 16 * 16 * 4));
  CUDA_CHECK(cudaMemset(dOut, 0, 16 * 16 * 4));

  k_tma2sm<<<2, 256>>>(desc, dOut);
  CUDA_CHECK(cudaDeviceSynchronize());

  // l2hint variant with a fresh cache policy. The policy
  // descriptor is opaque (built via inline-PTX `createpolicy.fractional`
  // in production code); for a smoke test we use 0 = default eviction.
  CUDA_CHECK(cudaMemset(dOut, 0, 16 * 16 * 4));
  k_tma2sm_l2hint<<<2, 256>>>(desc, dOut, /*cache_policy=*/0ull);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> hOut(16 * 16);
  CUDA_CHECK(cudaMemcpy(hOut.data(), dOut, 16 * 16 * 4,
                        cudaMemcpyDeviceToHost));
  for (int i = 0; i < 16 * 16; ++i) {
    if (hOut[i] != (float)i) {
      cudaFree(dIn); cudaFree(dOut);
      fprintf(stderr, "tma_load_2d_2sm_l2hint: out[%d]=%.1f expected %d\n",
              i, hOut[i], i);
      FAIL("2sm_l2hint did not deliver the tile");
    }
  }
  printf("tma_load_2d_2sm{,_l2hint} : compile + run OK (2SM cluster)\n");
  cudaFree(dIn); cudaFree(dOut);
  PASS();
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
