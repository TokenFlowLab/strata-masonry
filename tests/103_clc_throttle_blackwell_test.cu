// 103_clc_throttle_blackwell_test.cu -- smoke test for clc_throttle.
//
// Single CTA. Warp 0 plays the LOAD role; warp 1 plays the SCHED role.
// Both cycle 6 tiles through a 2-stage throttle pair. Set done_flag if
// no hang.
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)

#include <cstdio>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include "test_utils.cuh"
#include "../composites/103_clc_throttle_blackwell.cuh"

constexpr int N_STAGES = 2;
constexpr int N_TILES  = 6;

__global__ void __launch_bounds__(64, 1)
clc_throttle_smoke_kernel(int* done_flag) {
  __shared__ __align__(16) uint64_t throttle_full[N_STAGES];
  __shared__ __align__(16) uint64_t throttle_empty[N_STAGES];
  ClcThrottleBars<N_STAGES> bars{ throttle_full, throttle_empty };

  if (threadIdx.x == 0) {
    clc_throttle_init(bars, /*pre_arrive=*/1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();

  ClcThrottleState<N_STAGES> ld_state = clc_throttle_state_init<N_STAGES>();
  ClcThrottleState<N_STAGES> sc_state = clc_throttle_state_init<N_STAGES>();

  const int warp = threadIdx.x >> 5;
  for (int T = 0; T < N_TILES; ++T) {
    if (warp == 0 && (threadIdx.x & 31) == 0) {
      // LOAD role: wait empty, arrive full.
      clc_throttle_load_acquire(bars, ld_state);
      clc_throttle_load_commit(bars, ld_state);
    }
    if (warp == 1 && (threadIdx.x & 31) == 0) {
      // SCHED role: wait full, arrive empty.
      clc_throttle_sched_wait(bars, sc_state);
      clc_throttle_sched_release(bars, sc_state);
    }
    if (warp == 0) clc_throttle_state_advance(ld_state);
    if (warp == 1) clc_throttle_state_advance(sc_state);
  }

  __syncthreads();
  if (threadIdx.x == 0 && done_flag != nullptr) *done_flag = 1;
}

int main() {
  CUDA_CHECK(cudaFree(0));
  printf("clc_throttle_blackwell smoke test\n");

  int* d_done;
  CUDA_CHECK(cudaMalloc(&d_done, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_done, 0, sizeof(int)));

  clc_throttle_smoke_kernel<<<dim3(1), dim3(64)>>>(d_done);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("FAIL: kernel error: %s\n", cudaGetErrorString(err));
    return 1;
  }
  int h_done = 0;
  CUDA_CHECK(cudaMemcpy(&h_done, d_done, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_done));
  if (h_done != 1) { printf("FAIL: done_flag = %d\n", h_done); return 1; }
  printf("PASS: %d-tile throttle cycle completed\n", N_TILES);
  return 0;
}

#else
int main() { return 0; }
#endif
