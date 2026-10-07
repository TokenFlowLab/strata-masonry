// ARCH: sm_90a
// 29_mbarrier_init_test.cu -- init an mbarrier, arrive on it, verify the
// phase flips (indirectly via try_wait returning true).
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
#include "../src/primitives/29_mbarrier_init.cuh"
#include "../src/primitives/30_mbarrier_arrive.cuh"
#include "../src/primitives/33_mbarrier_try_wait.cuh"
#include "29_mbarrier_init.cuh"

// =============================================================================
// ours (originally guarded by PL_AGENTIC_SM100A || PL_AGENTIC_SM103A)
// =============================================================================

// ARCH: sm_90a


__global__ void k_init(int* ok) {
  __shared__ __align__(16) uint64_t mbar;
  if (threadIdx.x == 0) {
    mbarrier_init(smem_ptr_u32(&mbar), 1);
    mbarrier_arrive(smem_ptr_u32(&mbar));
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    mbarrier_wait_parity(smem_ptr_u32(&mbar), 0);
    *ok = 1;
    mbarrier_inval(smem_ptr_u32(&mbar));
  }
}

static int run_ours() {
  /* (orig args dropped) */
  int* d = nullptr; CUDA_CHECK(cudaMalloc(&d, 4));
  CUDA_CHECK(cudaMemset(d, 0, 4));
  k_init<<<1, 32>>>(d);
  CUDA_CHECK(cudaDeviceSynchronize());
  int h = 0; CUDA_CHECK(cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost));
  cudaFree(d);
  if (h != 1) FAIL("mbarrier init/arrive/wait round-trip failed");
  printf("mbarrier.init + arrive + try_wait : OK\n");
  PASS();
}

// =============================================================================
// theirs (originally guarded by PL_AGENTIC_SM90A)
// =============================================================================

// Test: mbarrier.init / mbarrier.inval -- compile + PTX verification

__shared__ uint64_t mbar_buf[1];

__global__ void mbarrier_init_kernel() {
    if (threadIdx.x == 0) {
        uint32_t addr = smem_ptr_u32(mbar_buf);
        mbarrier_init(addr, 1);
        mbarrier_inval(addr);
    }
}

static int run_theirs() {
  /* (orig args dropped) */
    printf("29_mbarrier_init: compiled successfully.\n");
    PASS();
    return 0;
}


int main() {
  int rc_ours   = run_ours();
  int rc_theirs = run_theirs();
  return (rc_ours == 0 && rc_theirs == 0) ? 0 : 1;
}
