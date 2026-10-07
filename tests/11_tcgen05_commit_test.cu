#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// ARCH: sm_100a
// 11_tcgen05_commit_test.cu -- tcgen05.commit arrives on an mbarrier. Issue
// commit after zero tcgen05.mma (legal; arrives immediately), try_wait the
// mbarrier, verify the phase flipped.

#include "test_utils.cuh"
#include "../primitives/33_mbarrier_try_wait.cuh"
#include "../primitives/0_tcgen05_alloc.cuh"
#include "../primitives/1_tcgen05_dealloc.cuh"
#include "../primitives/2_tcgen05_relinquish.cuh"
#include "../primitives/11_tcgen05_commit.cuh"

__global__ void k_commit(int* out) {
  __shared__ __align__(16) uint32_t slot;
  __shared__ __align__(16) uint64_t mbar;
  if (threadIdx.x == 0) {
    slot = 0;
    mbarrier_init_helper(smem_ptr_u32(&mbar), 1);
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    tcgen05_alloc<1>(smem_ptr_u32(&slot), 32);
    tcgen05_relinquish_alloc_permit<1>();
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    tcgen05_commit<1>(smem_ptr_u32(&mbar));
  }
  mbarrier_wait_parity(smem_ptr_u32(&mbar), 0);
  __syncthreads();
  if (threadIdx.x == 0) {
    *out = 1;
    tcgen05_dealloc<1>(slot, 32);
  }
}

int main() {
  int* d = nullptr; CUDA_CHECK(cudaMalloc(&d, 4));
  CUDA_CHECK(cudaMemset(d, 0, 4));
  k_commit<<<1, 32>>>(d);
  CUDA_CHECK(cudaDeviceSynchronize());
  int h = 0; CUDA_CHECK(cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost));
  cudaFree(d);
  if (h != 1) FAIL("commit did not arrive on mbarrier");
  printf("tcgen05.commit : mbarrier arrival observed\n");
  PASS();
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
