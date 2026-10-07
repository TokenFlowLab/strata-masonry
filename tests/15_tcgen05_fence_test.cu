#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// ARCH: sm_100a
// 15_tcgen05_fence_test.cu -- compile + emit smoke. tcgen05.fence instructions
// have no standalone observable effect; this test verifies they compile and
// execute without error.

#include "test_utils.cuh"
#include "../primitives/15_tcgen05_fence.cuh"

__global__ void k_fence() {
  tcgen05_fence_before_thread_sync();
  __syncthreads();
  tcgen05_fence_after_thread_sync();
}

int main() {
  k_fence<<<1, 32>>>();
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("tcgen05.fence::{before,after}_thread_sync : compile OK\n");
  PASS();
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
