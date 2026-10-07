#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// ARCH: sm_100a
// 125_epi_subtile_blackwell_test.cu -- compile smoke (real epilogue ties back
// to MMA; covered by block #93).

#include "test_utils.cuh"
#include "../src/primitives/0_tcgen05_alloc.cuh"
#include "../src/primitives/1_tcgen05_dealloc.cuh"
#include "../src/primitives/2_tcgen05_relinquish.cuh"
#include "../src/composites/125_epi_subtile_blackwell.cuh"

__global__ void k() {
  __shared__ __align__(16)   uint32_t slot;
  __shared__ __align__(1024) uint16_t smem_out[64];  // 4 x 2 x 4 bytes
  if (threadIdx.x == 0) slot = 0;
  __syncthreads();
  if (threadIdx.x < 32) {
    tcgen05_alloc<1>(smem_ptr_u32(&slot), 32);
    tcgen05_relinquish_alloc_permit<1>();
  }
  __syncthreads();
  uint32_t tbase = slot;
  if (threadIdx.x < 32) {
    epi_subtile_blackwell_fp16(tbase, smem_ptr_u32(smem_out));
  }
  __syncthreads();
  if (threadIdx.x == 0) tcgen05_dealloc<1>(tbase, 32);
}

int main() {
  k<<<1, 32>>>();
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("epi_subtile_blackwell_fp16 : compile + run OK\n");
  PASS();
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
