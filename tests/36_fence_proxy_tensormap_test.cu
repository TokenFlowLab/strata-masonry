// ARCH: sm_90a
// 36_fence_proxy_tensormap_test.cu -- compile smoke.
//
// Two test sets in one binary: run_ours() and run_theirs().

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "test_utils.cuh"
#include "../src/primitives/23_tma_tensormap.cuh"
#include "../src/primitives/36_fence_proxy_tensormap.cuh"
#include "36_fence_proxy_tensormap.cuh"

// =============================================================================
// ours
// =============================================================================

// ARCH: sm_90a


__global__ void k_f(CUtensorMap* desc) {
  if (threadIdx.x == 0) fence_proxy_tensormap_release_gpu();
}

static int run_ours() {
  float* d = nullptr; CUDA_CHECK(cudaMalloc(&d, 64 * 64 * 4));
  CUtensorMap desc;
  CUDA_CHECK(make_tma_2d_tiled(&desc, d, 64, 64, 32, 32, sizeof(float),
                                 CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                                 CU_TENSOR_MAP_SWIZZLE_NONE));
  CUtensorMap* dd = nullptr;
  CUDA_CHECK(cudaMalloc(&dd, sizeof(CUtensorMap)));
  CUDA_CHECK(cudaMemcpy(dd, &desc, sizeof(desc), cudaMemcpyHostToDevice));
  k_f<<<1, 32>>>(dd);
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaFree(d); cudaFree(dd);
  printf("fence.proxy.tensormap : compile OK\n");
  PASS();
}

// =============================================================================
// theirs
// =============================================================================

// Test: fence.proxy.tensormap -- compile + PTX verification

__global__ void fence_tensormap_kernel() {
    fence_proxy_tensormap_release_gpu();
    fence_proxy_tensormap_release_sys();
    fence_proxy_tensormap_release_cta();
    fence_proxy_tensormap_release_cluster();
}

static int run_theirs() {
  printf("36_fence_proxy_tensormap: compiled.\n"); PASS(); return 0; }


int main() {
  int rc_ours   = run_ours();
  int rc_theirs = run_theirs();
  return (rc_ours == 0 && rc_theirs == 0) ? 0 : 1;
}
