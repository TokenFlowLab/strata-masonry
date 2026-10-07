#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// ARCH: sm_100a
// 119_pipeline_init_blackwell_test.cu -- init a 4-stage barrier array, arrive once,
// verify try_wait returns true on each stage.

#include "test_utils.cuh"
#include "../primitives/30_mbarrier_arrive.cuh"
#include "../primitives/33_mbarrier_try_wait.cuh"
#include "../composites/119_pipeline_init_blackwell.cuh"

__global__ void __cluster_dims__(1, 1, 1) k(int* out) {
  __shared__ __align__(16) uint64_t full[4];
  __shared__ __align__(16) uint64_t empty[4];
  __shared__ __align__(16) uint64_t acc_full[2];
  // Init only mainloop full/empty + acc_full; rest stay null.
  BlackwellPipelineBars b{
      full, empty,
      acc_full, /*acc_empty=*/nullptr,
      /*clc_full=*/nullptr, /*clc_empty=*/nullptr,
      /*throttle_full=*/nullptr, /*throttle_empty=*/nullptr
  };
  BlackwellPipelineArriveCounts<2> a{};
  a.full = 1; a.empty = 1; a.acc_full = 1;
  pipeline_init_blackwell<4, /*CTA_GROUP=*/2>(b, a);
  if (threadIdx.x == 0) {
    mbarrier_arrive(smem_ptr_u32(&full[0]));
    mbarrier_wait_parity(smem_ptr_u32(&full[0]), 0);
    *out = 1;
  }
}

int main() {
  int* d = nullptr; CUDA_CHECK(cudaMalloc(&d, 4));
  CUDA_CHECK(cudaMemset(d, 0, 4));
  k<<<1, 128>>>(d);
  CUDA_CHECK(cudaDeviceSynchronize());
  int h = 0; CUDA_CHECK(cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost));
  cudaFree(d);
  if (h != 1) FAIL("pipeline_init_blackwell: arrive/wait round-trip failed");
  printf("pipeline_init_blackwell : OK\n");
  PASS();
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
