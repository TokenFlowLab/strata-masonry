// ARCH: sm_90a
// 35_fence_mbarrier_init_test.cu -- compile smoke.
//
// Two test sets in one binary: run_ours() and run_theirs().

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "test_utils.cuh"
#include "../src/primitives/29_mbarrier_init.cuh"
#include "../src/primitives/35_fence_mbarrier_init.cuh"
#include "35_fence_mbarrier_init.cuh"

// =============================================================================
// ours
// =============================================================================

// ARCH: sm_90a


__global__ void k_f() {
  __shared__ __align__(16) uint64_t mbar;
  if (threadIdx.x == 0) {
    mbarrier_init(smem_ptr_u32(&mbar), 1);
    fence_mbarrier_init_release_cluster();
    mbarrier_inval(smem_ptr_u32(&mbar));
  }
}

static int run_ours() {
  k_f<<<1, 32>>>();
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("fence.mbarrier_init.release : compile OK\n");
  PASS();
}

// =============================================================================
// theirs
// =============================================================================

// Test: fence.mbarrier_init.release -- compile + PTX verification

__global__ void fence_mbarrier_kernel() {
    fence_mbarrier_init_release_cluster();
}

static int run_theirs() {
  printf("35_fence_mbarrier_init: compiled.\n"); PASS(); return 0; }


int main() {
  int rc_ours   = run_ours();
  int rc_theirs = run_theirs();
  return (rc_ours == 0 && rc_theirs == 0) ? 0 : 1;
}
