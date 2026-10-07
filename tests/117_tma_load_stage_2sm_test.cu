#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// ARCH: sm_100a
// 117_tma_load_stage_2sm_test.cu -- compile smoke for the 2SM stage wrapper.

#include "test_utils.cuh"
#include "../src/primitives/23_tma_tensormap.cuh"
#include "../src/primitives/29_mbarrier_init.cuh"
#include "../src/primitives/33_mbarrier_try_wait.cuh"
#include "../src/composites/117_tma_load_stage_2sm.cuh"

__global__ void __cluster_dims__(2, 1, 1)
k(const __grid_constant__ CUtensorMap ta) {
  __shared__ __align__(128) float smem[16 * 16];
  __shared__ __align__(16)  uint64_t mbar;
  if (threadIdx.x == 0) {
    mbarrier_init(smem_ptr_u32(&mbar), 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    tma_load_stage_2sm_1tensor(smem_ptr_u32(&mbar), 16 * 16 * sizeof(float),
                                smem_ptr_u32(smem), &ta, 0, 0);
  }
  if (blockIdx.x == 0)
    mbarrier_wait_parity(smem_ptr_u32(&mbar), 0);
  __syncthreads();
  if (threadIdx.x == 0)
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n"
                 :: "r"(smem_ptr_u32(&mbar)));
}

int main() {
  std::vector<float> hin(16 * 16, 1.f);
  float* din = nullptr; CUDA_CHECK(cudaMalloc(&din, 16 * 16 * 4));
  CUDA_CHECK(cudaMemcpy(din, hin.data(), 16 * 16 * 4, cudaMemcpyHostToDevice));
  CUtensorMap ta;
  CUDA_CHECK(make_tma_2d_tiled(&ta, din, 16, 16, 16, 16, sizeof(float),
                                 CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                                 CU_TENSOR_MAP_SWIZZLE_NONE));
  k<<<2, 128>>>(ta);
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaFree(din);
  printf("tma_load_stage_2sm : compile + run OK\n");
  PASS();
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
