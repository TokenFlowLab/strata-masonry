// ARCH: sm_90a
// 67_mapa_test.cu -- mapa + getctarank cross-CTA address remap.
//
// Launches a 2-CTA cluster. Each CTA writes its own rank into a shared
// variable at a fixed compile-time offset. Then both CTAs use mapa to
// translate the local SMEM address to peer 0's SMEM address and read
// the value at that location -- which must equal 0 (CTA 0's rank).
// Symmetric check via mapa to peer 1 must read 1.
//
// Also verifies getctarank: feeding the remapped peer addresses back
// through getctarank must return the rank of the owner CTA.

#include <cstdio>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include "test_utils.cuh"
#include "../primitives/67_mapa.cuh"
#include "../primitives/38_barrier_cluster.cuh"

#if defined(PL_AGENTIC_SM90A) || defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)

__global__ void __cluster_dims__(2, 1, 1)
mapa_kernel(int* out_self_rank, int* out_peer0_value, int* out_peer1_value,
            int* out_owner_of_peer0_addr, int* out_owner_of_peer1_addr) {
  __shared__ int s_rank;

  // Each CTA stores its own intra-cluster rank into local SMEM.
  uint32_t cluster_rank;
  asm("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(cluster_rank));
  if (threadIdx.x == 0) {
    s_rank = static_cast<int>(cluster_rank);
  }
  __syncthreads();
  // Cluster-wide sync so peer SMEM is initialized before we read it.
  barrier_cluster_arrive();
  barrier_cluster_wait();

  if (threadIdx.x == 0) {
    const uint32_t local_addr =
        static_cast<uint32_t>(__cvta_generic_to_shared(&s_rank));

    // Remap to peer 0 and peer 1.
    const uint32_t peer0_addr = mapa_shared_cluster_u32(local_addr, 0);
    const uint32_t peer1_addr = mapa_shared_cluster_u32(local_addr, 1);

    int v0, v1;
    asm("ld.shared::cluster.b32 %0, [%1];\n" : "=r"(v0) : "r"(peer0_addr));
    asm("ld.shared::cluster.b32 %0, [%1];\n" : "=r"(v1) : "r"(peer1_addr));

    // getctarank back-check.
    const uint32_t owner0 = getctarank_shared_cluster_u32(peer0_addr);
    const uint32_t owner1 = getctarank_shared_cluster_u32(peer1_addr);

    out_self_rank[cluster_rank]            = static_cast<int>(cluster_rank);
    out_peer0_value[cluster_rank]          = v0;
    out_peer1_value[cluster_rank]          = v1;
    out_owner_of_peer0_addr[cluster_rank]  = static_cast<int>(owner0);
    out_owner_of_peer1_addr[cluster_rank]  = static_cast<int>(owner1);
  }
}

int main() {
  CUDA_CHECK(cudaFree(0));
  printf("67_mapa_test: 2-CTA cluster, mapa + getctarank cross-CTA verify\n");

  int *d_self = nullptr, *d_v0 = nullptr, *d_v1 = nullptr,
      *d_o0   = nullptr, *d_o1 = nullptr;
  CUDA_CHECK(cudaMalloc(&d_self, 2 * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_v0,   2 * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_v1,   2 * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_o0,   2 * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_o1,   2 * sizeof(int)));

  CUDA_CHECK(cudaFuncSetAttribute(mapa_kernel,
      cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

  cudaLaunchConfig_t config = {};
  config.gridDim          = dim3(2, 1, 1);
  config.blockDim         = dim3(32, 1, 1);
  config.dynamicSmemBytes = 0;
  cudaLaunchAttribute attr;
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim = {2, 1, 1};
  config.attrs    = &attr;
  config.numAttrs = 1;
  CUDA_CHECK(cudaLaunchKernelEx(&config, mapa_kernel,
                                d_self, d_v0, d_v1, d_o0, d_o1));
  CUDA_CHECK(cudaDeviceSynchronize());

  int self[2], v0[2], v1[2], o0[2], o1[2];
  CUDA_CHECK(cudaMemcpy(self, d_self, 2 * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(v0,   d_v0,   2 * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(v1,   d_v1,   2 * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(o0,   d_o0,   2 * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(o1,   d_o1,   2 * sizeof(int), cudaMemcpyDeviceToHost));

  bool ok = true;
  for (int r = 0; r < 2; ++r) {
    if (self[r] != r)  { printf("FAIL: self_rank[%d]=%d (expected %d)\n", r, self[r], r); ok = false; }
    if (v0[r]   != 0)  { printf("FAIL: peer0_value[%d]=%d (expected 0)\n", r, v0[r]); ok = false; }
    if (v1[r]   != 1)  { printf("FAIL: peer1_value[%d]=%d (expected 1)\n", r, v1[r]); ok = false; }
    if (o0[r]   != 0)  { printf("FAIL: owner_of_peer0_addr[%d]=%d (expected 0)\n", r, o0[r]); ok = false; }
    if (o1[r]   != 1)  { printf("FAIL: owner_of_peer1_addr[%d]=%d (expected 1)\n", r, o1[r]); ok = false; }
  }

  cudaFree(d_self); cudaFree(d_v0); cudaFree(d_v1);
  cudaFree(d_o0);   cudaFree(d_o1);

  if (ok) printf("PASS tests/67_mapa_test.cu\n");
  return ok ? 0 : 1;
}

#else
int main() { return 0; }
#endif
