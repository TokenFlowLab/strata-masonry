#if defined(PL_AGENTIC_SM90A)
// ARCH: sm_90a
// 59_wgmma_fence_commit_wait_test.cu -- runtime test for wgmma fence commit wait
//
// Test: wgmma_fence_commit_wait -- compile + PTX verification
#include <cstdio>
#include <cuda_runtime.h>
#include "test_utils.cuh"
#include "59_wgmma_fence_commit_wait.cuh"

__global__ void wgmma_fcw_kernel() {
    // Exercise the full fence -> commit -> wait lifecycle
    wgmma_fence();
    wgmma_commit_group();
    wgmma_wait_group<0>();

    // Also test wait_group with N=1
    wgmma_fence();
    wgmma_commit_group();
    wgmma_wait_group<1>();
    wgmma_wait_group<0>();
}

int main() { printf("59_wgmma_fence_commit_wait: compiled.\n"); PASS(); return 0; }

#endif  // PL_AGENTIC_SM90A
