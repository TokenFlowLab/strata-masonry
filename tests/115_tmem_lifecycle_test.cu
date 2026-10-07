#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// ARCH: sm_100a
// 115_tmem_lifecycle_test.cu -- alloc / use / dealloc round-trip.

#include "test_utils.cuh"
#include "../composites/115_tmem_lifecycle.cuh"

__global__ void k_life(uint32_t* out) {
  __shared__ __align__(16) uint32_t slot;
  if (threadIdx.x == 0) slot = 0;
  __syncthreads();
  tmem_lifecycle<1>(smem_ptr_u32(&slot), 128, [&] __device__ (uint32_t base) {
    if (threadIdx.x == 0) *out = base;
  });
}

int main() {
  uint32_t* d = nullptr; CUDA_CHECK(cudaMalloc(&d, 4));
  CUDA_CHECK(cudaMemset(d, 0xFF, 4));
  k_life<<<1, 128>>>(d);
  CUDA_CHECK(cudaDeviceSynchronize());
  uint32_t h; CUDA_CHECK(cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost));
  cudaFree(d);
  if ((h >> 16) != 0) FAIL("tmem_lifecycle: unexpected lane index");
  printf("tmem_lifecycle : base = 0x%08x\n", h);
  PASS();
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
