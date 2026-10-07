#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// ARCH: sm_100a
// 86_acc_correction_test.cu -- scale a TMEM accumulator by 0.5 and verify.

#include "test_utils.cuh"
#include "../primitives/0_tcgen05_alloc.cuh"
#include "../primitives/1_tcgen05_dealloc.cuh"
#include "../primitives/2_tcgen05_relinquish.cuh"
#include "../primitives/9_tcgen05_ld.cuh"
#include "../primitives/10_tcgen05_st.cuh"
#include "../primitives/12_tcgen05_wait.cuh"
#include "../composites/86_acc_correction.cuh"

__global__ void k(float* out) {
  __shared__ __align__(16) uint32_t slot;
  if (threadIdx.x == 0) slot = 0;
  __syncthreads();
  if (threadIdx.x < 32) {
    tcgen05_alloc<1>(smem_ptr_u32(&slot), 32);
    tcgen05_relinquish_alloc_permit<1>();
  }
  __syncthreads();
  uint32_t tbase = slot;
  if (threadIdx.x < 32) {
    uint32_t in[8];
    float* f = reinterpret_cast<float*>(in);
    #pragma unroll
    for (int i = 0; i < 8; ++i) f[i] = (float)(threadIdx.x + i) * 2.f;
    tcgen05_st_32x32b_x8(tbase, in);
    tcgen05_wait_st();
    acc_correction_fp32_x8(tbase, 0.5f);
    uint32_t got[8];
    tcgen05_ld_32x32b_x8(tbase, got);
    tcgen05_wait_ld();
    float* g = reinterpret_cast<float*>(got);
    #pragma unroll
    for (int i = 0; i < 8; ++i)
      out[threadIdx.x * 8 + i] = g[i];
  }
  __syncthreads();
  if (threadIdx.x == 0) tcgen05_dealloc<1>(tbase, 32);
}

int main() {
  float* d = nullptr; CUDA_CHECK(cudaMalloc(&d, 32 * 8 * 4));
  CUDA_CHECK(cudaMemset(d, 0, 32 * 8 * 4));
  k<<<1, 32>>>(d);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> h(32 * 8);
  CUDA_CHECK(cudaMemcpy(h.data(), d, 32 * 8 * 4, cudaMemcpyDeviceToHost));
  cudaFree(d);
  int fails = 0;
  for (int lane = 0; lane < 32; ++lane)
    for (int i = 0; i < 8; ++i) {
      float want = (float)(lane + i) * 2.f * 0.5f;
      if (std::fabs(h[lane * 8 + i] - want) > 1e-3f) ++fails;
    }
  printf("acc_correction(x0.5) : fails = %d / %d\n", fails, 32 * 8);
  if (fails) FAIL("acc rescale wrong");
  PASS();
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
