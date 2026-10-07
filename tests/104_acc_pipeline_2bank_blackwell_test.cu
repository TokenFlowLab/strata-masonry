// 104_acc_pipeline_2bank_blackwell_test.cu -- smoke test for the 2-bank
// acc-pipeline composite.
//
// Exercises the producer-consumer mbarrier dance over 4 tiles (covers
// both stages twice) on a 2-CTA cluster. Producer = warp 0 of leader
// CTA; consumer = epi_tid==0 of each CTA.
//
// Bar: compiles, launches, and a writable flag in GMEM gets set to 1
// by the producer after the 4-tile cycle completes (no hang on any
// consumer wait or producer wait, no trap).
//
// This is a composite-level smoke test (no tcgen05.alloc / tcgen05.mma
// to exercise the sec-8.1 alloc state machine -- that is verified at
// kernel level via K0 dense_gemm_bf16.cu).
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)

#include <cstdio>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include "test_utils.cuh"
#include "../composites/104_acc_pipeline_2bank_blackwell.cuh"

// 2-CTA cluster, 256 threads/CTA. Warp 0 = producer (MMA proxy);
// warp 4 = consumer leader (epi_tid==0 proxy). Other warps idle.
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(256, 1)
acc_pipeline_2bank_smoke_kernel(int* done_flag) {
  __shared__ __align__(16) uint64_t acc_full[2];
  __shared__ __align__(16) uint64_t acc_empty[2];
  AccPipeline2BankBars bars{ acc_full, acc_empty };

  const int peer = blockIdx.x & 1;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

  if (threadIdx.x == 0) {
    // Smoke setup: 1 consumer thread per CTA -> empty_arrive_count = 2.
    acc_pipeline_2bank_init(bars, /*empty_arrive_count=*/2);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  asm volatile("barrier.cluster.arrive;\n" ::: "memory");
  asm volatile("barrier.cluster.wait;\n" ::: "memory");

  AccPipeline2BankState prod_s = acc_pipeline_2bank_state_init();
  AccPipeline2BankState cons_s = acc_pipeline_2bank_state_init();

  for (int T = 0; T < 4; ++T) {
    // Producer: leader CTA's warp 0, single thread.
    if (warp == 0 && peer == 0 && lane == 0) {
      acc_pipeline_2bank_producer_acquire(bars, prod_s);
      // (No actual MMA in smoke test; just commit immediately.)
      acc_pipeline_2bank_producer_commit_cluster<2>(bars, prod_s, /*ctamask=*/0x3);
    }
    if (warp == 0 && peer == 0 && lane == 0) {
      acc_pipeline_2bank_state_advance(prod_s);
    }

    // Consumer: warp 4 lane 0 of each CTA arrives (i.e., 1 thread
    // per CTA -> 2 arrives total per stage, matching arrive_count=2).
    if (warp == 4 && lane == 0) {
      acc_pipeline_2bank_consumer_wait(bars, cons_s);
      acc_pipeline_2bank_consumer_release(bars, cons_s);
      acc_pipeline_2bank_state_advance(cons_s);
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
  printf("acc_pipeline_2bank_blackwell smoke test\n");

  int* d_done;
  CUDA_CHECK(cudaMalloc(&d_done, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_done, 0, sizeof(int)));

  acc_pipeline_2bank_smoke_kernel<<<dim3(2, 1, 1), dim3(256, 1, 1)>>>(d_done);
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
