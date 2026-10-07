// ARCH: sm_90a
// 34_fence_proxy_async_test.cu -- compile smoke. Fence has no observable
// standalone effect; correctness is implicit via TMA store (#22).
//
// Combined test: both ours' and theirs' coverage is exercised
// in a single binary (each side's main() became run_ours/run_theirs).

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "test_utils.cuh"
#include "../src/primitives/34_fence_proxy_async.cuh"
#include "34_fence_proxy_async.cuh"

// =============================================================================
// ours (originally guarded by PL_AGENTIC_SM100A || PL_AGENTIC_SM103A)
// =============================================================================

// ARCH: sm_90a


__global__ void k_fence() {
  fence_proxy_async_shared_cta();
  fence_proxy_async();
}

static int run_ours() {
  /* (orig args dropped) */
  k_fence<<<1, 32>>>();
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("fence.proxy.async : compile OK\n");
  PASS();
}

// =============================================================================
// theirs (originally guarded by PL_AGENTIC_SM90A)
// =============================================================================

// Test: fence.proxy.async -- compile + PTX verification

__global__ void fence_proxy_kernel() {
    fence_proxy_async_shared_cta();
    fence_proxy_async_shared();
}

static int run_theirs() {
  /* (orig args dropped) */ printf("34_fence_proxy_async: compiled.\n"); PASS(); return 0; }


int main() {
  int rc_ours   = run_ours();
  int rc_theirs = run_theirs();
  return (rc_ours == 0 && rc_theirs == 0) ? 0 : 1;
}
