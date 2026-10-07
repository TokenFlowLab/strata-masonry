#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// ARCH: sm_100a
// 90_mma_warp_blackwell_test.cu -- exercises blocks/90_mma_warp_blackwell.cuh.
//
// Verifies (1) the alloc / relinquish / commit / wait / dealloc envelope
// runs to completion (via *ok = 1 after the commit-wait), and (2) the TMEM
// base returned by tcgen05.alloc has lane=0 / col=0 (top 16 bits zero).
// Real MMA correctness is covered in #100 pipeline_blackwell.
//
// PTX sniff: `cuobjdump --dump-ptx build/90_mma_warp_blackwell_test |
// grep -E 'tcgen05.(alloc|commit|dealloc).cta_group::1'` should show 3 hits.

#include "test_utils.cuh"
#include "../primitives/29_mbarrier_init.cuh"
#include "../blocks/90_mma_warp_blackwell.cuh"

// __global__ wrapper -- owns SMEM slot + mbar. The .cuh exports the
// commit/wait body as __device__ __forceinline__.
__global__ void mma_warp_blackwell_test_kernel(int* ok, uint32_t* out_base) {
  WpCtx wpc = wp_ctx_init();  // WARP_PROF TraceContext (no-op without -DWARP_PROF)

  __shared__ __align__(16) uint32_t slot;
  __shared__ __align__(16) uint64_t mbar;
  if (threadIdx.x == 0) {
    slot = 0;
    mbarrier_init(smem_ptr_u32(&mbar), 1);
  }
  __syncthreads();
  mma_warp_blackwell_block(wpc, &slot, &mbar, ok, out_base);
}

int main() {
  int* d_ok = nullptr;
  uint32_t* d_base = nullptr;
  CUDA_CHECK(cudaMalloc(&d_ok, 4));
  CUDA_CHECK(cudaMalloc(&d_base, 4));
  CUDA_CHECK(cudaMemset(d_ok, 0, 4));
  CUDA_CHECK(cudaMemset(d_base, 0xFF, 4));

  mma_warp_blackwell_test_kernel<<<1, 128>>>(d_ok, d_base);
  CUDA_CHECK(cudaDeviceSynchronize());

  int h_ok = 0;
  uint32_t h_base = 0;
  CUDA_CHECK(cudaMemcpy(&h_ok, d_ok, 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&h_base, d_base, 4, cudaMemcpyDeviceToHost));
  cudaFree(d_ok);
  cudaFree(d_base);

  if (h_ok != 1) FAIL("mma warp commit/wait chain failed");
  // The tcgen05.alloc result encodes (lane << 16) | col with lane = 0 and
  // col = some allocated column. Top 16 bits should be zero.
  if ((h_base >> 16) != 0)
    FAIL("tcgen05.alloc base has non-zero lane bits: 0x%08x", h_base);
  printf("mma_warp_blackwell : OK (tmem_base=0x%08x)\n", h_base);
  PASS();
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
