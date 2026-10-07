#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// ARCH: sm_100a
// 98_softmax_warp_test.cu -- exercises blocks/98_softmax_warp.cuh.
//
// Owns the __global__ smoke-test wrapper. Test envelope: init the
// SoftmaxState, call the block to walk K=96 scores, write final
// (m, l) to GMEM. Block does not allocate state itself.

#include "test_utils.cuh"
#include "../blocks/98_softmax_warp.cuh"

__global__ void softmax_warp_test_kernel(const float* __restrict__ scores,
                                         int K,
                                         float* __restrict__ out_m,
                                         float* __restrict__ out_l) {
  SoftmaxState s;
  softmax_state_init(s);
  softmax_warp_block(scores, K, s);
  if (threadIdx.x == 0) {
    if (out_m != nullptr) *out_m = s.m;
    if (out_l != nullptr) *out_l = s.l;
  }
}

int main() {
  const int K = 96;
  std::vector<float> h(K);
  for (int i = 0; i < K; ++i) h[i] = (float)((i * 13) % 17) - 8.f;

  float *dIn = nullptr, *dm = nullptr, *dl = nullptr;
  CUDA_CHECK(cudaMalloc(&dIn, K * 4));
  CUDA_CHECK(cudaMalloc(&dm, 4));
  CUDA_CHECK(cudaMalloc(&dl, 4));
  CUDA_CHECK(cudaMemcpy(dIn, h.data(), K * 4, cudaMemcpyHostToDevice));

  softmax_warp_test_kernel<<<1, 32>>>(dIn, K, dm, dl);
  CUDA_CHECK(cudaDeviceSynchronize());

  float m = 0.f, l = 0.f;
  CUDA_CHECK(cudaMemcpy(&m, dm, 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&l, dl, 4, cudaMemcpyDeviceToHost));
  cudaFree(dIn);
  cudaFree(dm);
  cudaFree(dl);

  float ref_m = -INFINITY;
  for (float x : h) ref_m = std::fmax(ref_m, x);
  float ref_l = 0.f;
  for (float x : h) ref_l += exp2f(x - ref_m);

  printf("softmax_warp : m=%g (want %g), l=%g (want %g)\n",
         m, ref_m, l, ref_l);
  if (std::fabs(m - ref_m) > 1e-4f) FAIL("softmax max wrong");
  if (std::fabs(l - ref_l) > 1e-3f * ref_l) FAIL("softmax sum wrong");
  PASS();
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
