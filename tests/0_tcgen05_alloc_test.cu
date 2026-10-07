#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// ARCH: sm_100a
// 0_tcgen05_alloc_test.cu -- smoke test: alloc 128 TMEM columns, read back
// the base address, dealloc. Verifies that the instruction executes and the
// returned address encodes lane=0 (top 16 bits).

#include "test_utils.cuh"
#include "../src/primitives/0_tcgen05_alloc.cuh"
#include "../src/primitives/1_tcgen05_dealloc.cuh"

__global__ void k_alloc(uint32_t* out_base, uint32_t n_cols) {
  __shared__ uint32_t smem_base;
  if (threadIdx.x == 0) smem_base = 0xDEADBEEFu;
  __syncthreads();

  const uint32_t slot = smem_ptr_u32(&smem_base);
  tcgen05_alloc<1>(slot, n_cols);
  // tcgen05.alloc is synchronous; smem_base now holds the TMEM base addr.
  __syncthreads();

  if (threadIdx.x == 0) *out_base = smem_base;

  tcgen05_dealloc<1>(smem_base, n_cols);
}

int main() {
  uint32_t* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(uint32_t)));

  const uint32_t n_cols = 128;
  k_alloc<<<1, 32>>>(d_out, n_cols);
  CUDA_CHECK(cudaDeviceSynchronize());

  uint32_t h_out = 0;
  CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  cudaFree(d_out);

  // Base address: lane index in bits [31:16] (0), column index in bits [15:0].
  // For the first alloc in a CTA, column index is also 0 on most configs.
  uint32_t lane = h_out >> 16;
  uint32_t col  = h_out & 0xFFFFu;
  printf("tcgen05.alloc(%u) -> 0x%08x (lane=%u, col=%u)\n",
         n_cols, h_out, lane, col);

  if (lane != 0) FAIL("tcgen05.alloc returned non-zero lane index");
  if (h_out == 0xDEADBEEFu) FAIL("tcgen05.alloc did not write the SMEM slot");
  PASS();
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
