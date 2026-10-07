#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// ARCH: sm_100a
// 81_clc_scheduler_loop_test.cu -- compile smoke for clc_fetch_next_tile.

#include "test_utils.cuh"
#include "../primitives/29_mbarrier_init.cuh"
#include "../composites/81_clc_scheduler_loop.cuh"

__global__ void k(uint32_t* out) {
  __shared__ __align__(16) uint32_t slot[4];
  __shared__ __align__(16) uint64_t mbar;
  for (int i = threadIdx.x; i < 4; i += blockDim.x) slot[i] = 0;
  if (threadIdx.x == 0) mbarrier_init(smem_ptr_u32(&mbar), 1);
  __syncthreads();
  ClcTile t = clc_fetch_next_tile(smem_ptr_u32(slot),
                                     smem_ptr_u32(&mbar), 0);
  if (threadIdx.x == 0) {
    out[0] = t.canceled;
    out[1] = t.ctaid_x;
    out[2] = t.ctaid_y;
  }
  __syncthreads();
  if (threadIdx.x == 0)
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n"
                 :: "r"(smem_ptr_u32(&mbar)));
}

int main() {
  uint32_t* d = nullptr; CUDA_CHECK(cudaMalloc(&d, 12));
  k<<<1, 32>>>(d);
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaFree(d);
  printf("clc_fetch_next_tile : compile + run OK\n");
  PASS();
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
