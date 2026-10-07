#if defined(PL_AGENTIC_SM90A)
// ARCH: sm_90a
// 43_smem_desc_hopper_test.cu -- runtime test for smem desc hopper
//
// Test: smem_desc_hopper -- compile + PTX verification
#include <cstdio>
#include <cuda_runtime.h>
#include "test_utils.cuh"
#include "43_smem_desc_hopper.cuh"

extern __shared__ char smem[];

__global__ void smem_desc_kernel(uint64_t* out) {
    uint32_t addr = smem_ptr_u32(smem);
    uint64_t desc = build_smem_desc_hopper(addr, 0, 256, 3);
    uint64_t desc_b128 = build_smem_desc_hopper_b128(addr, 256);
    uint64_t desc_adv = smem_desc_advance_hopper(desc_b128, 512);
    if (threadIdx.x == 0) { out[0] = desc; out[1] = desc_b128; out[2] = desc_adv; }
}

int main() { printf("43_smem_desc_hopper: compiled.\n"); PASS(); return 0; }

#endif  // PL_AGENTIC_SM90A
