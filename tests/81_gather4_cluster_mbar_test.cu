// 81_gather4_cluster_mbar_test.cu -- safe gather4 cross-CTA handoff.
//
// Each peer issues gather4 against its local completion mbarrier and waits
// locally. Peer 1 then explicitly arrives on peer 0's handoff mbarrier.
// Peer 0 waits for that handoff before the cluster drains SMEM.
//
// gather4 has no cta_group::2 form. It must not target a remote mbarrier via
// the peer-bit trick: that unsupported probe can deadlock. This test covers
// the deterministic local-wait plus cross-CTA-arrive protocol used by K1.
//
// Build: nvcc -arch=sm_100a tests/81_gather4_cluster_mbar_test.cu -o test_81

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "test_utils.cuh"

#include "../primitives/23_tma_tensormap.cuh"
#include "../primitives/29_mbarrier_init.cuh"
#include "../primitives/30_mbarrier_arrive.cuh"
#include "../primitives/31_mbarrier_arrive_tx.cuh"
#include "../primitives/33_mbarrier_try_wait.cuh"
#include "../primitives/35_fence_mbarrier_init.cuh"
#include "../primitives/38_barrier_cluster.cuh"
#include "../primitives/70_smem_ptr.cuh"

#include "../primitives/73_tma_load_2d_gather4.cuh"

constexpr int R = 32;             // rows in A
constexpr int C = 32;             // cols in A (BF16)
constexpr int N_PER_PEER = 4;     // 1 gather4 call per peer

constexpr int PEER_BYTES = N_PER_PEER * C * (int)sizeof(__nv_bfloat16);

__global__ void __cluster_dims__(2, 1, 1)
k_gather4_cluster_mbar(const __grid_constant__ CUtensorMap tmap_in,
                       __nv_bfloat16* out_p0, __nv_bfloat16* out_p1,
                       int* d_status_p0) {
  __shared__ __align__(1024) __nv_bfloat16 smA[N_PER_PEER * C];
  __shared__ __align__(16)   uint64_t      local_bar;
  __shared__ __align__(16)   uint64_t      peer_done;

  const int peer = blockIdx.x & 1;

  if (threadIdx.x == 0) {
    mbarrier_init(smem_ptr_u32(&local_bar), 1);
    mbarrier_init(smem_ptr_u32(&peer_done), 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  barrier_cluster_arrive();
  barrier_cluster_wait();

  const int base = peer * N_PER_PEER;

  if (threadIdx.x == 0) {
    const uint32_t bar_addr = smem_ptr_u32(&local_bar);
    mbarrier_arrive_expect_tx(bar_addr, (uint32_t)PEER_BYTES);

    const uint32_t dst = smem_ptr_u32(smA);
    tma_load_2d_gather4(dst, &tmap_in, bar_addr,
                        /*col=*/0,
                        /*r0=*/base + 0,
                        /*r1=*/base + 1,
                        /*r2=*/base + 2,
                        /*r3=*/base + 3);
  }

  mbarrier_wait_parity(smem_ptr_u32(&local_bar), /*parity=*/0);

  if (peer == 1 && threadIdx.x == 0) {
    constexpr uint32_t SM100_PEER_MASK = 0xFEFFFFFFu;
    const uint32_t peer0_done = smem_ptr_u32(&peer_done) & SM100_PEER_MASK;
    mbarrier_arrive_cluster_default(peer0_done);
  }
  if (peer == 0) {
    mbarrier_wait_parity(smem_ptr_u32(&peer_done), /*parity=*/0);
    if (threadIdx.x == 0) {
      *d_status_p0 = 1;
    }
  }
  barrier_cluster_arrive();
  barrier_cluster_wait();

  __nv_bfloat16* dst = (peer == 0) ? out_p0 : out_p1;
  for (int i = threadIdx.x; i < N_PER_PEER * C; i += blockDim.x) {
    dst[i] = smA[i];
  }
}

int main() {
  CUDA_CHECK(cudaFree(0));

  std::vector<__nv_bfloat16> hA(R * C);
  for (int r = 0; r < R; ++r) {
    for (int c = 0; c < C; ++c) {
      hA[r * C + c] = __float2bfloat16((float)(r * 1000 + c));
    }
  }

  __nv_bfloat16 *dA, *dOut0, *dOut1; int *d_status;
  CUDA_CHECK(cudaMalloc(&dA, sizeof(__nv_bfloat16) * R * C));
  CUDA_CHECK(cudaMalloc(&dOut0, sizeof(__nv_bfloat16) * N_PER_PEER * C));
  CUDA_CHECK(cudaMalloc(&dOut1, sizeof(__nv_bfloat16) * N_PER_PEER * C));
  CUDA_CHECK(cudaMalloc(&d_status, sizeof(int)));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(),
                        sizeof(__nv_bfloat16) * R * C, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_status, 0, sizeof(int)));

  CUtensorMap tmap_in;
  CUDA_CHECK(make_tma_2d_tiled(&tmap_in, dA, R, C,
                               /*box_rows=*/1, /*box_cols=*/C,
                               sizeof(__nv_bfloat16),
                               CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                               CU_TENSOR_MAP_SWIZZLE_NONE));

  cudaLaunchConfig_t config = {};
  config.gridDim  = {2, 1, 1};
  config.blockDim = {32, 1, 1};
  cudaLaunchAttribute attrs[1] = {};
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim = {2, 1, 1};
  config.numAttrs = 1;
  config.attrs = attrs;

  cudaLaunchKernelEx(&config, k_gather4_cluster_mbar,
                     tmap_in, dOut0, dOut1, d_status);
  cudaError_t err = cudaDeviceSynchronize();

  int status = 0;
  CUDA_CHECK(cudaMemcpy(&status, d_status, sizeof(int), cudaMemcpyDeviceToHost));

  printf("cudaDeviceSynchronize result: %s\n",
         err == cudaSuccess ? "OK" : cudaGetErrorString(err));
  printf("Peer 0 observed peer 1's completed gather4: %s\n",
         status == 1 ? "YES" : "NO");

  if (status != 1) {
    FAIL("cross-CTA gather4 handoff did not complete");
  }

  std::vector<__nv_bfloat16> h0(N_PER_PEER * C), h1(N_PER_PEER * C);
  CUDA_CHECK(cudaMemcpy(h0.data(), dOut0,
                        sizeof(__nv_bfloat16) * N_PER_PEER * C,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h1.data(), dOut1,
                        sizeof(__nv_bfloat16) * N_PER_PEER * C,
                        cudaMemcpyDeviceToHost));
  int fails = 0;
  for (int peer = 0; peer < 2; ++peer) {
    const auto& out = peer == 0 ? h0 : h1;
    for (int r = 0; r < N_PER_PEER; ++r) {
      for (int c = 0; c < C; ++c) {
        float want = __bfloat162float(
            hA[(peer * N_PER_PEER + r) * C + c]);
        float got  = __bfloat162float(out[r * C + c]);
        if (got != want) ++fails;
      }
    }
  }
  printf("Both peers' data check: %s (%d fails)\n",
         fails == 0 ? "OK" : "FAIL", fails);
  if (fails != 0) {
    FAIL("gather4 data mismatch");
  }

  cudaFree(dA);
  cudaFree(dOut0);
  cudaFree(dOut1);
  cudaFree(d_status);
  PASS();
}
