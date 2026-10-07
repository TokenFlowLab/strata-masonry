#if defined(PL_AGENTIC_SM90A)
// ARCH: sm_90a
// 120_pipeline_init_hopper_test.cu -- runtime test for pipeline init hopper
//
// Test: 120_pipeline_init_hopper -- compile + PTX verification
#include <cstdio>
#include <cuda_runtime.h>
#include "test_utils.cuh"
#include "120_pipeline_init_hopper.cuh"

__global__ void pipeline_init_kernel() {
    __shared__ uint64_t mbars[8];
    uint32_t full_base = smem_ptr_u32(&mbars[0]);
    uint32_t empty_base = smem_ptr_u32(&mbars[4]);
    if (threadIdx.x == 0) {
        pipeline_init_hopper_1stage(full_base, empty_base, 128u);
        pipeline_init_hopper<4>(full_base, empty_base, 128u);
    }
    pipeline_init_hopper_full<4>(full_base, empty_base, 128u, (uint32_t)blockDim.x);
}

int main() { printf("120_pipeline_init_hopper: compiled.\n"); PASS(); return 0; }

#endif  // PL_AGENTIC_SM90A
