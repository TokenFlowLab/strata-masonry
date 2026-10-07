#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// ARCH: sm_100a
// 99_correction_warp_test.cu -- exercises blocks/99_correction_warp.cuh.
//
// Owns the __global__ smoke-test wrapper for block 99 (blocks are
// header-only, device-callable building blocks). The
// wrapper does the test-only envelope: tcgen05.alloc, seed the slice
// with 1.0, tcgen05.wait::st, call the block to multiply by `factor`,
// tcgen05.ld read-back, tcgen05.wait::ld, dealloc. The block itself
// only does step (5) -- the actual rescale arithmetic.
//
// Verifies (1) the alloc / seed / wait_st / correction_warp_block /
// read-back / dealloc envelope runs in a single warp, (2) every lane's
// 8 output registers equal the rescale factor exactly (within 1e-5) --
// i.e. the underlying composite acc_correction_fp32_x8 multiplies every
// TMEM cell by the supplied scalar without missing or doubling any.
//
// Real FMHA correction (running max -> rescale factor coming from #85,
// #98 softmax) is exercised at the kernel level; this is the warp-
// isolated rescale-arithmetic path.
//
// PTX sniff: `cuobjdump --dump-ptx build/99_correction_warp_test |
// grep -E 'tcgen05.st.sync.aligned.32x32b.x8|tcgen05.ld.sync.aligned.32x32b.x8'`
// should show both the seed-store and the read-back.

#include "test_utils.cuh"
#include "../blocks/99_correction_warp.cuh"
#include "../primitives/0_tcgen05_alloc.cuh"
#include "../primitives/1_tcgen05_dealloc.cuh"
#include "../primitives/2_tcgen05_relinquish.cuh"
#include "../primitives/9_tcgen05_ld.cuh"
#include "../primitives/10_tcgen05_st.cuh"
#include "../primitives/12_tcgen05_wait.cuh"

__device__ __forceinline__
static uint32_t bb_smem_ptr_u32_99_test(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__global__ void correction_warp_test_kernel(float* __restrict__ out, float factor) {
  __shared__ __align__(16) uint32_t slot;
  if (threadIdx.x == 0) slot = 0;
  __syncthreads();
  if (threadIdx.x < 32) {
    tcgen05_alloc<1>(bb_smem_ptr_u32_99_test(&slot), 32);
    tcgen05_relinquish_alloc_permit<1>();
  }
  __syncthreads();
  uint32_t tbase = slot;

  if (threadIdx.x < 32) {
    uint32_t in[8];
    float* f = reinterpret_cast<float*>(in);
    #pragma unroll
    for (int i = 0; i < 8; ++i) f[i] = 1.0f;
    tcgen05_st_32x32b_x8(tbase, in);
    tcgen05_wait_st();

    correction_warp_block(tbase, factor);

    uint32_t got[8];
    tcgen05_ld_32x32b_x8(tbase, got);
    tcgen05_wait_ld();
    float* g = reinterpret_cast<float*>(got);
    if (out != nullptr) {
      #pragma unroll
      for (int i = 0; i < 8; ++i)
        out[threadIdx.x * 8 + i] = g[i];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) tcgen05_dealloc<1>(tbase, 32);
}

int main() {
  const float F = 0.125f;  // simulates exp2(-3) when running max grew by 3.
  float* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, 32 * 8 * 4));
  CUDA_CHECK(cudaMemset(d, 0, 32 * 8 * 4));

  correction_warp_test_kernel<<<1, 32>>>(d, F);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h(32 * 8);
  CUDA_CHECK(cudaMemcpy(h.data(), d, 32 * 8 * 4, cudaMemcpyDeviceToHost));
  cudaFree(d);

  int fails = 0;
  for (auto v : h) if (std::fabs(v - F) > 1e-5f) ++fails;
  printf("correction_warp x%.3f : fails = %d / %zu\n", F, fails, h.size());
  if (fails) FAIL("rescale mismatch (%d cells off)", fails);
  PASS();
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
