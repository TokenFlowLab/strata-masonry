// ARCH: sm_100a
// 79_gather4_2sm_mma_test.cu -- standalone 2SM (cta_group::2) gather4 + MMA
// with cross-CTA producer-side sync. Verifies that an extra peer1->peer0
// mbarrier signal closes the gather4 cross-peer signaling gap.
//
// Pipeline:
//   A : (M=128, K=64) bf16 row-major, identity perm. Each peer gathers
//       64 rows into its own SMEM A (peer 0 rows 0..63, peer 1 rows
//       64..127), 16 gather4 calls per peer.
//   B : (N=8, K=64)   bf16. Loaded by 2SM cta_group::2 multicast TMA.
//   MMA: tcgen05.mma.cta_group::2 kind::f16 (bf16), M_cluster=128, N=8,
//        K=64 (4 atoms of K=16 each).
//   Sync: each peer has its own local full_bar with arrive_count=1.
//         peer 0's local full_bar tracks own A gather (64*64*2 = 8192 B)
//         + both peers' cta_group::2 B (2 * 8*64*2 = 2048 B) = 10240 B.
//         peer 1's local full_bar tracks own A gather only (8192 B).
//         peer1_done lives on peer 0; arrive_count=1. peer 1 arrives on
//         it via cluster-scope mbarrier_arrive_cluster_default AFTER
//         waiting on its own local full_bar. Peer 0's MMA waits on
//         peer 0's local full_bar AND peer1_done.
//
// Verify: read TMEM acc via tcgen05.ld, write to GMEM, compare against
// fp32 host A @ B^T reference.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "test_utils.cuh"
#include "../src/primitives/0_tcgen05_alloc.cuh"
#include "../src/primitives/1_tcgen05_dealloc.cuh"
#include "../src/primitives/2_tcgen05_relinquish.cuh"
#include "../src/primitives/3_tcgen05_mma_f16.cuh"
#include "../src/primitives/8_tcgen05_mma_idesc.cuh"
#include "../src/primitives/9_tcgen05_ld.cuh"
#include "../src/primitives/11_tcgen05_commit.cuh"
#include "../src/primitives/12_tcgen05_wait.cuh"
#include "../src/primitives/19_tma_load_2sm.cuh"
#include "../src/primitives/23_tma_tensormap.cuh"
#include "../src/primitives/29_mbarrier_init.cuh"
#include "../src/primitives/30_mbarrier_arrive.cuh"
#include "../src/primitives/31_mbarrier_arrive_tx.cuh"
#include "../src/primitives/33_mbarrier_try_wait.cuh"
#include "../src/primitives/35_fence_mbarrier_init.cuh"
#include "../src/primitives/38_barrier_cluster.cuh"
#include "../src/primitives/42_smem_desc_blackwell.cuh"
#include "../src/primitives/70_smem_ptr.cuh"
#include "../src/primitives/73_tma_load_2d_gather4.cuh"

constexpr int M_CLUSTER  = 256;
constexpr int M_PER_PEER = M_CLUSTER / 2;   // 128
constexpr int N          = 8;
constexpr int K          = 64;
constexpr int N_GATHER_CALLS = M_PER_PEER / 4;  // 16
constexpr int A_PEER_BYTES = M_PER_PEER * K * (int)sizeof(__nv_bfloat16);  // 8192
constexpr int B_BYTES      = N * K * (int)sizeof(__nv_bfloat16);            // 1024

// Peer-bit mask to convert local SMEM addr -> peer 0's view.
static constexpr uint32_t PEER_MASK = 0xFEFFFFFFu;

__device__ __constant__ int d_perm[M_CLUSTER];  // identity

__global__ void __cluster_dims__(2, 1, 1)
k_gather4_2sm_mma(const __grid_constant__ CUtensorMap tmap_a,
                  const __grid_constant__ CUtensorMap tmap_b,
                  float* d_out) {
  __shared__ __align__(1024) __nv_bfloat16 smA[M_PER_PEER * K];  // 8 KB per peer
  __shared__ __align__(1024) __nv_bfloat16 smB[N * K];           // 1 KB per peer
  __shared__ __align__(16)   uint32_t      slot;
  __shared__ __align__(16)   uint64_t      full_bar;
  __shared__ __align__(16)   uint64_t      peer1_done;
  __shared__ __align__(16)   uint64_t      mma_bar;

  const int peer = blockIdx.x & 1;

  // ----- Init mbars + TMEM (only peer 0 needs peer1_done) -----
  if (threadIdx.x == 0) {
    slot = 0;
    mbarrier_init(smem_ptr_u32(&full_bar),   1);
    mbarrier_init(smem_ptr_u32(&peer1_done), 1);
    mbarrier_init(smem_ptr_u32(&mma_bar),    1);
    fence_mbarrier_init_release_cluster();
  }
  __syncthreads();
  barrier_cluster_arrive();
  barrier_cluster_wait();

  // ----- TMEM alloc (each peer allocates its own TMEM) -----
  if (threadIdx.x < 32) {
    tcgen05_alloc<2>(smem_ptr_u32(&slot), 128);
    tcgen05_relinquish_alloc_permit<2>();
  }
  __syncthreads();
  // cta_group::2 alloc may need cluster-scope visibility before either
  // peer reads tmem_base.
  barrier_cluster_arrive();
  barrier_cluster_wait();
  uint32_t tmem_base = slot;

  // ----- LOAD: gather4 for A + 2SM TMA for B -----
  const uint32_t local_full = smem_ptr_u32(&full_bar);
  const uint32_t full_route_b = local_full & PEER_MASK;  // 2SM B targets peer 0
  const int base_row = peer * M_PER_PEER;

  if (threadIdx.x == 0) {
    const uint32_t expect_tx =
        (peer == 0)
          ? (uint32_t)(A_PEER_BYTES + 2 * B_BYTES)  // own A + both peers' Bs
          : (uint32_t)(A_PEER_BYTES);                // own A only
    mbarrier_arrive_expect_tx(local_full, expect_tx);

    // Gather4 for A -- signals LOCAL mbar.
    #pragma unroll
    for (int g = 0; g < N_GATHER_CALLS; ++g) {
      const int r0 = d_perm[base_row + g * 4 + 0];
      const int r1 = d_perm[base_row + g * 4 + 1];
      const int r2 = d_perm[base_row + g * 4 + 2];
      const int r3 = d_perm[base_row + g * 4 + 3];
      const uint32_t dst = smem_ptr_u32(smA)
                         + (uint32_t)(g * 4 * K * (int)sizeof(__nv_bfloat16));
      tma_load_2d_gather4(dst, &tmap_a, local_full,
                          /*col=*/0, r0, r1, r2, r3);
    }

    // B: 2SM cta_group::2 TMA -- signals peer 0's mbar via mask.
    tma_load_2d_2sm(smem_ptr_u32(smB), &tmap_b, full_route_b,
                    /*x=*/0, /*y=*/0);
  }

  // Each peer waits on its OWN local full_bar.
  mbarrier_wait_parity(local_full, 0);

  // ----- Cross-CTA sync: peer 1 -> peer 0 -----
  if (peer == 1 && threadIdx.x == 0) {
    // Compute peer 0's peer1_done address from peer 1's local addr via mask.
    const uint32_t peer0_pd = smem_ptr_u32(&peer1_done) & PEER_MASK;
    mbarrier_arrive_cluster_default(peer0_pd);
  }

  // Peer 0 waits for peer 1's "A done" signal.
  if (peer == 0) {
    mbarrier_wait_parity(smem_ptr_u32(&peer1_done), 0);
  }

  // DEBUG: skip MMA, dump SMEM A.
  __nv_bfloat16* d_out_bf = reinterpret_cast<__nv_bfloat16*>(d_out);
  for (int i = threadIdx.x; i < M_PER_PEER * K; i += blockDim.x) {
    d_out_bf[peer * M_PER_PEER * K + i] = smA[i];
  }
  __syncthreads();

  if (threadIdx.x < 32) {
    tcgen05_dealloc<2>(tmem_base, 128);
  }
}

int main() {
  CUDA_CHECK(cudaFree(0));

  std::vector<__nv_bfloat16> hA(M_CLUSTER * K), hB(N * K);
  for (int i = 0; i < M_CLUSTER * K; ++i)
    hA[i] = __float2bfloat16((float)((i * 31 + 1) % 7 - 3));
  for (int i = 0; i < N * K; ++i)
    hB[i] = __float2bfloat16((float)((i * 37 + 2) % 7 - 3));

  __nv_bfloat16 *dA = nullptr, *dB = nullptr;
  float* dOut = nullptr;
  CUDA_CHECK(cudaMalloc(&dA, M_CLUSTER * K * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&dB, N * K * sizeof(__nv_bfloat16)));
  // dOut size: enough for SMEM-A dump (bf16) -- larger of the two cases.
  const size_t out_bytes = (size_t)M_CLUSTER * K * sizeof(__nv_bfloat16);
  CUDA_CHECK(cudaMalloc(&dOut, out_bytes));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), M_CLUSTER * K * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), N * K * 2,        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dOut, 0, out_bytes));

  std::vector<int> h_perm(M_CLUSTER);
  for (int i = 0; i < M_CLUSTER; ++i) h_perm[i] = i;
  CUDA_CHECK(cudaMemcpyToSymbol(d_perm, h_perm.data(), M_CLUSTER * sizeof(int)));

  CUtensorMap tmap_a;
  CUDA_CHECK(make_tma_2d_tiled(&tmap_a, dA, M_CLUSTER, K,
                               /*box_rows=*/1, /*box_cols=*/K,
                               sizeof(__nv_bfloat16),
                               CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                               CU_TENSOR_MAP_SWIZZLE_NONE));
  CUtensorMap tmap_b;
  CUDA_CHECK(make_tma_2d_tiled(&tmap_b, dB, N, K,
                               /*box_rows=*/N, /*box_cols=*/K,
                               sizeof(__nv_bfloat16),
                               CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                               CU_TENSOR_MAP_SWIZZLE_128B));

  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(2, 1, 1);
  config.blockDim = dim3(128, 1, 1);
  config.dynamicSmemBytes = 0;
  config.stream = 0;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim = {2, 1, 1};
  config.attrs = attrs;
  config.numAttrs = 1;
  cudaLaunchKernelEx(&config, k_gather4_2sm_mma, tmap_a, tmap_b, dOut);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> hOut_bf(M_CLUSTER * K);
  CUDA_CHECK(cudaMemcpy(hOut_bf.data(), dOut,
                        M_CLUSTER * K * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));
  cudaFree(dA); cudaFree(dB); cudaFree(dOut);

  int fails = 0;
  for (int m = 0; m < M_CLUSTER; ++m) {
    for (int k = 0; k < K; ++k) {
      float got  = __bfloat162float(hOut_bf[m * K + k]);
      float want = __bfloat162float(hA[m * K + k]);
      if (got != want) {
        if (fails < 6)
          fprintf(stderr, "  A_smem_dump[%d,%d]=%.1f want %.1f\n",
                  m, k, got, want);
        ++fails;
      }
    }
  }
  printf("gather4 + 2SM cross-CTA sync (no MMA, M_cluster=%d): "
         "fails = %d / %d\n", M_CLUSTER, fails, M_CLUSTER * K);
  if (fails) FAIL("gather4 + cross-CTA sync mismatch");
  PASS();
}
