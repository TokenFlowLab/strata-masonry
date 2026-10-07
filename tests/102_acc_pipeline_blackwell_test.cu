// 102_acc_pipeline_blackwell_test.cu -- smoke test for acc_pipeline_blackwell.
//
// Runs a single 2-CTA cluster with an MMA warp role + an "epilogue"
// role that just produces 256 arrives per stage. Cycles 4 tiles to
// cover both pipeline stages twice (T=0,1: warm-up; T=2,3: bars
// fully cycled).
//
// Bar: compiles, launches, and a writable flag in GMEM gets set to 1
// by the producer warp after the 4-tile cycle completes (no hang on
// any consumer/producer wait).
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)

#include <cstdio>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include "test_utils.cuh"
#include "../composites/102_acc_pipeline_blackwell.cuh"

// 2-CTA cluster, 256 threads/CTA. Warp 0 = producer (MMA proxy);
// warps 4-7 = consumer (epi proxy, 128 threads/CTA -> 256/cluster).
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(256, 1)
acc_pipeline_smoke_kernel(int* done_flag) {
  __shared__ __align__(16) uint64_t acc_full[2];
  __shared__ __align__(16) uint64_t acc_empty[2];
  AccPipelineBars bars{ acc_full, acc_empty };

  const int peer = blockIdx.x & 1;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

  if (threadIdx.x == 0) {
    // arrive_count = 2 (CTAs) x 128 (epi threads) = 256.
    acc_pipeline_init(bars, /*consumer_arv_count=*/256);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  asm volatile("barrier.cluster.arrive;\n" ::: "memory");
  asm volatile("barrier.cluster.wait;\n" ::: "memory");

  AccPipelineState prod_s = acc_pipeline_state_init();
  AccPipelineState cons_s = acc_pipeline_state_init();

  for (int T = 0; T < 4; ++T) {
    if (warp == 0 && peer == 0 && lane == 0) {
      acc_pipeline_producer_acquire(bars, prod_s);
      // (No actual MMA in smoke test; just commit immediately.)
      acc_pipeline_producer_commit<2>(bars, prod_s, /*ctamask=*/0x3);
    }
    if (warp == 0) acc_pipeline_state_advance(prod_s);

    if (warp >= 4) {
      acc_pipeline_consumer_wait(bars, cons_s);
      acc_pipeline_consumer_release(bars, cons_s);
      acc_pipeline_state_advance(cons_s);
    }
  }

  asm volatile("barrier.cluster.arrive;\n" ::: "memory");
  asm volatile("barrier.cluster.wait;\n" ::: "memory");
  if (peer == 0 && threadIdx.x == 0 && done_flag != nullptr) {
    *done_flag = 1;
  }
}

int main() {
  CUDA_CHECK(cudaFree(0));
  printf("acc_pipeline_blackwell smoke test\n");

  int* d_done;
  CUDA_CHECK(cudaMalloc(&d_done, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_done, 0, sizeof(int)));

  acc_pipeline_smoke_kernel<<<dim3(2, 1, 1), dim3(256, 1, 1)>>>(d_done);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("FAIL: kernel launch error: %s\n", cudaGetErrorString(err));
    return 1;
  }

  int h_done = 0;
  CUDA_CHECK(cudaMemcpy(&h_done, d_done, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_done));

  if (h_done != 1) {
    printf("FAIL: done_flag = %d (expected 1)\n", h_done);
    return 1;
  }
  printf("PASS: 4-tile cycle completed\n");
  return 0;
}

#else
int main() { return 0; }
#endif
