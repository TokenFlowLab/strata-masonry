#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// ARCH: sm_100a
// 62_clc_query_cancel_test.cu -- compile smoke for query_cancel.

#include "test_utils.cuh"
#include "../primitives/62_clc_query_cancel.cuh"

__global__ void k_q(uint32_t* out) {
  __shared__ __align__(16) uint32_t slot[4];
  for (int i = threadIdx.x; i < 4; i += blockDim.x) slot[i] = 0;
  __syncthreads();
  if (threadIdx.x == 0) {
    uint32_t r0, r1, r2, r3;
    clc_load_response(smem_ptr_u32(slot), r0, r1, r2, r3);
    out[0] = clc_query_is_canceled(r0, r1, r2, r3);
    out[1] = clc_query_first_ctaid_x(r0, r1, r2, r3);
    out[2] = clc_query_first_ctaid_y(r0, r1, r2, r3);
    out[3] = clc_query_first_ctaid_z(r0, r1, r2, r3);
  }
}

int main() {
  uint32_t* d = nullptr; CUDA_CHECK(cudaMalloc(&d, 16));
  k_q<<<1, 32>>>(d);
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaFree(d);
  printf("clusterlaunchcontrol.query_cancel : compile + run OK (x/y/z + is_canceled)\n");
  PASS();
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
