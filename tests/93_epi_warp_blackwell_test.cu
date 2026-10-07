#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// ARCH: sm_100a
// 93_epi_warp_blackwell_test.cu -- exercises blocks/93_epi_warp_blackwell.cuh.
//
// Verifies (1) the alloc / tcgen05.st seed / wait / cvt+stmatrix /
// dealloc chain runs to completion in a single warp, and (2) the resulting
// 64-FP16 SMEM staging buffer contains non-zero data (the cvt path
// produced something from a non-zero FP32 input pattern).
//
// Real epilogue correctness (cvt rounding + TMA store landing pattern) is
// covered in #100 pipeline_blackwell.
//
// PTX sniff: `cuobjdump --dump-ptx build/93_epi_warp_blackwell_test |
// grep -E 'tcgen05.alloc|stmatrix.sync.aligned|cvt.rn.f16x2.f32'` should
// show all three.

#include "test_utils.cuh"
#include "../src/blocks/93_epi_warp_blackwell.cuh"

// __global__ wrapper -- owns SMEM (slot + smem_stage). The .cuh exports
// the alloc / seed / cvt+stmatrix / dealloc body as __device__ __forceinline__.
__global__ void epi_warp_blackwell_test_kernel(uint16_t* out) {
  WpCtx wpc = wp_ctx_init();  // WARP_PROF TraceContext (no-op without -DWARP_PROF)

  __shared__ __align__(16)   uint32_t slot;
  __shared__ __align__(1024) uint16_t smem_stage[64];
  if (threadIdx.x == 0) slot = 0;
  __syncthreads();
  epi_warp_blackwell_block(wpc, &slot, smem_stage, out);
}

int main() {
  uint16_t* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, 64 * 2));
  CUDA_CHECK(cudaMemset(d, 0, 64 * 2));

  epi_warp_blackwell_test_kernel<<<1, 32>>>(d);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<uint16_t> h(64);
  CUDA_CHECK(cudaMemcpy(h.data(), d, 64 * 2, cudaMemcpyDeviceToHost));
  cudaFree(d);

  int nz = 0;
  for (auto v : h) if (v) ++nz;
  printf("epi_warp_blackwell : non-zero fp16 = %d / 64\n", nz);
  if (!nz) FAIL("epilogue produced no non-zero FP16 output");
  PASS();
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
