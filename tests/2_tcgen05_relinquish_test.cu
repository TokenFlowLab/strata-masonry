#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// ARCH: sm_100a
// 2_tcgen05_relinquish_test.cu -- smoke test: alloc, relinquish alloc permit,
// then dealloc. After relinquish, the CTA must NOT call alloc again (that
// would be undefined) -- we only verify that relinquish+dealloc succeed.

#include "test_utils.cuh"
#include "../src/primitives/0_tcgen05_alloc.cuh"
#include "../src/primitives/1_tcgen05_dealloc.cuh"
#include "../src/primitives/2_tcgen05_relinquish.cuh"

__global__ void k_alloc_relinquish_dealloc(uint32_t n_cols, uint32_t* out_base) {
  __shared__ uint32_t smem_base;
  if (threadIdx.x == 0) smem_base = 0;
  __syncthreads();

  const uint32_t slot = smem_ptr_u32(&smem_base);
  tcgen05_alloc<1>(slot, n_cols);
  __syncthreads();

  tcgen05_relinquish_alloc_permit<1>();

  if (threadIdx.x == 0) *out_base = smem_base;

  tcgen05_dealloc<1>(smem_base, n_cols);
}

int main() {
  uint32_t* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(uint32_t)));

  const uint32_t n_cols = 128;
  k_alloc_relinquish_dealloc<<<1, 32>>>(n_cols, d_out);
  CUDA_CHECK(cudaDeviceSynchronize());

  uint32_t h_out = 0;
  CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  cudaFree(d_out);

  printf("alloc+relinquish+dealloc OK, base=0x%08x\n", h_out);
  if ((h_out >> 16) != 0) FAIL("unexpected lane index");
  PASS();
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
