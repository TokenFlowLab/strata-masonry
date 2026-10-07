// ARCH: sm_90a
// 47_nanosleep_test.cu -- compile smoke; nanosleep is a timing hint.
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
#include "../src/primitives/47_nanosleep.cuh"
#include "47_nanosleep.cuh"

// =============================================================================
// ours (originally guarded by PL_AGENTIC_SM100A || PL_AGENTIC_SM103A)
// =============================================================================

// ARCH: sm_90a


__global__ void k_n() {
  nanosleep(100);
}

static int run_ours() {
  /* (orig args dropped) */
  k_n<<<1, 32>>>();
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("nanosleep.u32 : compile + run OK\n");
  PASS();
}

// =============================================================================
// theirs (originally guarded by PL_AGENTIC_SM90A)
// =============================================================================

// Test: nanosleep -- compile + PTX verification + perf timing

// Kernel that invokes nanosleep(100) repeatedly so the observed time-per-launch
// reflects the sleep duration rather than launch overhead.
__global__ void nanosleep_kernel() {
    // Single invocation of each variant; launched N times in the host loop.
    nanosleep(100);
    nanosleep_short();
}

static int run_theirs() {
  /* (orig args dropped) */
    CUDA_CHECK(cudaFree(0));

    // Warm-up launch to surface any JIT/init errors before timing.
    nanosleep_kernel<<<1, 32>>>();
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("  nanosleep: LAUNCH FAIL: %s\n", cudaGetErrorString(err));
        FAIL("kernel launch failed");
        return 1;
    }

    // --- perf ---
    GpuTimer t;
    const int ITERS = 1000;  // very short kernel -> 1000 iters
    t.begin();
    for (int i = 0; i < ITERS; i++) nanosleep_kernel<<<1, 32>>>();
    t.end();
    printf("  perf: %.2f us/launch (nanosleep(100) + nanosleep_short() per launch, 1 warp)\n",
           t.elapsed_ms() * 1000.0f / ITERS);

    printf("47_nanosleep: compiled.\n");
    PASS();
    return 0;
}


int main() {
  int rc_ours   = run_ours();
  int rc_theirs = run_theirs();
  return (rc_ours == 0 && rc_theirs == 0) ? 0 : 1;
}
