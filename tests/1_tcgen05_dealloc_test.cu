#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// ARCH: sm_100a
// 1_tcgen05_dealloc_test.cu -- smoke test: repeat alloc+dealloc many times
// in a single kernel launch. If dealloc is broken, subsequent allocs will
// eventually hang or fault.

#include "test_utils.cuh"
#include "../src/primitives/0_tcgen05_alloc.cuh"
#include "../src/primitives/1_tcgen05_dealloc.cuh"

__global__ void k_alloc_dealloc_loop(uint32_t iters, uint32_t n_cols,
                                     uint32_t* out_last_base) {
  __shared__ uint32_t smem_base;
  const uint32_t slot = smem_ptr_u32(&smem_base);

  for (uint32_t i = 0; i < iters; ++i) {
    if (threadIdx.x == 0) smem_base = 0;
    __syncthreads();

    tcgen05_alloc<1>(slot, n_cols);
    __syncthreads();

    tcgen05_dealloc<1>(smem_base, n_cols);
  }

  if (threadIdx.x == 0) *out_last_base = smem_base;
}

int main() {
  uint32_t* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(uint32_t)));

  const uint32_t iters  = 64;
  const uint32_t n_cols = 64;

  GpuTimer t;
  t.begin();
  k_alloc_dealloc_loop<<<1, 32>>>(iters, n_cols, d_out);
  CUDA_CHECK(cudaDeviceSynchronize());
  t.end(); float ms = t.elapsed_ms();

  uint32_t h_out = 0;
  CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  cudaFree(d_out);

  printf("%u iters x (alloc+dealloc %u cols) in %.3f ms -> %.1f us/iter, last=0x%08x\n",
         iters, n_cols, ms, ms * 1000.f / iters, h_out);

  if ((h_out >> 16) != 0) FAIL("unexpected lane index from final alloc");
  PASS();
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
