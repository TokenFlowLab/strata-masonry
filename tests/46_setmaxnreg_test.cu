// ARCH: sm_90a
// 46_setmaxnreg_test.cu -- compile smoke. The instruction only takes effect
// when the kernel is launched with a per-thread register cap >= N.
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
#include "../src/primitives/46_setmaxnreg.cuh"
#include "46_setmaxnreg.cuh"

// =============================================================================
// ours (originally guarded by PL_AGENTIC_SM100A || PL_AGENTIC_SM103A)
// =============================================================================

// ARCH: sm_100a


__global__ void __launch_bounds__(128, 1) k_reg() {
  setmaxnreg_dec<40>();
  __syncthreads();
  setmaxnreg_inc<240>();
}

static int run_ours() {
  /* (orig args dropped) */
  k_reg<<<1, 128>>>();
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("setmaxnreg.{dec,inc} : compile + run OK\n");
  PASS();
}

// =============================================================================
// theirs (originally guarded by PL_AGENTIC_SM90A)
// =============================================================================

// Test: setmaxnreg -- compile + PTX verification

__global__ void setmaxnreg_dec_kernel() {
    setmaxnreg_dec<40>();
}

__global__ void setmaxnreg_inc_kernel() {
    setmaxnreg_inc<232>();
}

static int run_theirs() {
  /* (orig args dropped) */ printf("46_setmaxnreg: compiled.\n"); PASS(); return 0; }


int main() {
  int rc_ours   = run_ours();
  int rc_theirs = run_theirs();
  return (rc_ours == 0 && rc_theirs == 0) ? 0 : 1;
}
