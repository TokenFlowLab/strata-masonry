#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// ARCH: sm_100a
// 14_tcgen05_shift_test.cu -- smoke. Allocate 128 columns and issue shift + commit + wait.

#include "test_utils.cuh"
#include "../src/primitives/33_mbarrier_try_wait.cuh"
#include "../src/primitives/0_tcgen05_alloc.cuh"
#include "../src/primitives/1_tcgen05_dealloc.cuh"
#include "../src/primitives/2_tcgen05_relinquish.cuh"
#include "../src/primitives/11_tcgen05_commit.cuh"
#include "../src/primitives/14_tcgen05_shift.cuh"

__global__ void k_shift() {
  __shared__ __align__(16) uint32_t slot;
  __shared__ __align__(16) uint64_t mbar;
  if (threadIdx.x == 0) {
    slot = 0;
    mbarrier_init_helper(smem_ptr_u32(&mbar), 1);
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    tcgen05_alloc<1>(smem_ptr_u32(&slot), 128);
    tcgen05_relinquish_alloc_permit<1>();
  }
  __syncthreads();
  uint32_t tbase = slot;
  if (threadIdx.x == 0) {
    tcgen05_shift<1>(tbase);
    tcgen05_commit<1>(smem_ptr_u32(&mbar));
  }
  mbarrier_wait_parity(smem_ptr_u32(&mbar), 0);
  __syncthreads();
  if (threadIdx.x == 0) tcgen05_dealloc<1>(tbase, 128);
}

int main() {
  k_shift<<<1, 128>>>();
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("tcgen05.shift : compile + run OK\n");
  PASS();
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
