// ARCH: sm_100a
// 80_gather4_2sm_mma_test.cu -- standalone 2SM (cta_group::2) gather4 + MMA
// with cross-CTA producer-side sync. Verifies that an extra peer1->peer0
// mbarrier signal closes the gather4 cross-peer signaling gap.
//
// Pipeline:
//   A : (M=256, K=64) bf16 row-major, identity perm. Each peer gathers
//       128 rows into its own SMEM A, 32 gather4 calls per peer.
//   B : (N=256, K=64) bf16. Each peer loads its 128-column half with a
//       cta_group::2 TMA operation.
//   MMA: tcgen05.mma.cta_group::2 kind::f16 (bf16), M_cluster=256, N=256,
//        K=64 (4 atoms of K=16 each).
//   Sync: each peer has its own local full_bar with arrive_count=1.
//         peer 0's local full_bar tracks its A gather plus both peers' B
//         loads. Peer 1's local full_bar tracks its own A gather only.
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
#include "../primitives/0_tcgen05_alloc.cuh"
#include "../primitives/1_tcgen05_dealloc.cuh"
#include "../primitives/2_tcgen05_relinquish.cuh"
#include "../primitives/3_tcgen05_mma_f16.cuh"
#include "../primitives/8_tcgen05_mma_idesc.cuh"
#include "../primitives/9_tcgen05_ld.cuh"
#include "../primitives/11_tcgen05_commit.cuh"
#include "../primitives/12_tcgen05_wait.cuh"
#include "../primitives/19_tma_load_2sm.cuh"
#include "../primitives/23_tma_tensormap.cuh"
#include "../primitives/29_mbarrier_init.cuh"
#include "../primitives/44_elect_sync.cuh"
#include "../primitives/30_mbarrier_arrive.cuh"
#include "../primitives/31_mbarrier_arrive_tx.cuh"
#include "../primitives/33_mbarrier_try_wait.cuh"
#include "../primitives/35_fence_mbarrier_init.cuh"
#include "../primitives/38_barrier_cluster.cuh"
#include "../primitives/42_smem_desc_blackwell.cuh"
#include "../primitives/70_smem_ptr.cuh"
#include "../primitives/73_tma_load_2d_gather4.cuh"

constexpr int M_CLUSTER  = 256;
constexpr int M_PER_PEER = M_CLUSTER / 2;   // 128
constexpr int N          = 256;             // MMA atom N = K1's N_TILE_CLUSTER
constexpr int N_PER_PEER = N / 2;
constexpr int K          = 64;
constexpr int N_GATHER_CALLS = M_PER_PEER / 4;
constexpr int A_PEER_BYTES = M_PER_PEER * K * (int)sizeof(__nv_bfloat16);
constexpr int B_BYTES = N_PER_PEER * K * (int)sizeof(__nv_bfloat16);

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
    tcgen05_alloc<2>(smem_ptr_u32(&slot), N);
    tcgen05_relinquish_alloc_permit<2>();
  }
  __syncthreads();
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
                    /*x=*/0, /*y=*/peer * N_PER_PEER);
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

  // MMA must be issued from a warp-coordinated context. All 32 lanes of
  // peer 0's warp 0 enter; elect_one_sync() picks the actual issuer.
  if (peer == 0 && threadIdx.x < 32) {
    if (elect_one_sync()) {
      constexpr uint32_t A_LBO = 16;
      constexpr uint32_t A_SBO = 1024;
      constexpr uint32_t B_LBO = 16;
      constexpr uint32_t B_SBO = 1024;
      const uint64_t desc_a = build_smem_desc_blackwell(
          smem_ptr_u32(smA), A_SBO, A_LBO, SmemSwizzleBlackwell::B128);
      const uint64_t desc_b = build_smem_desc_blackwell(
          smem_ptr_u32(smB), B_SBO, B_LBO, SmemSwizzleBlackwell::B128);
      const uint32_t idesc = make_idesc_bf16_f32(M_CLUSTER, N);
      constexpr int K_ATOMS = K / 16;
      #pragma unroll
      for (int ki = 0; ki < K_ATOMS; ++ki) {
        const bool enable_d = (ki != 0);
        tcgen05_mma_f16_ss<2>(tmem_base,
                              desc_a + 2 * ki,
                              desc_b + 2 * ki,
                              idesc, enable_d);
      }
      tcgen05_commit<2>(smem_ptr_u32(&mma_bar));
    }
  }
  if (peer == 0) mbarrier_wait_parity(smem_ptr_u32(&mma_bar), 0);
  __syncthreads();
  barrier_cluster_arrive();
  barrier_cluster_wait();

  // Read first 32 TMEM rows per peer (peer 0: rows 0..31, peer 1: rows
  // 128..159). Each peer reads the corresponding local TMEM lanes.
  if (threadIdx.x < 32) {
    const int lane = (int)threadIdx.x;
    uint32_t regs[128];
    tcgen05_ld_32x32b_x128(tmem_base, regs);
    tcgen05_wait_ld();
    const int global_row = peer * M_PER_PEER + lane;
    #pragma unroll
    for (int n = 0; n < 128; ++n) {
      d_out[global_row * N + n] = __int_as_float(regs[n]);
    }
    tcgen05_ld_32x32b_x128(tmem_base + 128, regs);
    tcgen05_wait_ld();
    #pragma unroll
    for (int n = 0; n < 128; ++n) {
      d_out[global_row * N + 128 + n] = __int_as_float(regs[n]);
    }
  }
  __syncthreads();
  barrier_cluster_arrive();
  barrier_cluster_wait();

  if (threadIdx.x < 32) {
    tcgen05_dealloc<2>(tmem_base, N);
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
  CUDA_CHECK(cudaMalloc(&dOut, M_CLUSTER * N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), M_CLUSTER * K * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), N * K * 2,        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dOut, 0, M_CLUSTER * N * sizeof(float)));

  std::vector<int> h_perm(M_CLUSTER);
  for (int i = 0; i < M_CLUSTER; ++i) h_perm[i] = i;
  CUDA_CHECK(cudaMemcpyToSymbol(d_perm, h_perm.data(), M_CLUSTER * sizeof(int)));

  CUtensorMap tmap_a;
  CUDA_CHECK(make_tma_2d_tiled(&tmap_a, dA, M_CLUSTER, K,
                               /*box_rows=*/1, /*box_cols=*/K,
                               sizeof(__nv_bfloat16),
                               CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                               CU_TENSOR_MAP_SWIZZLE_128B));
  CUtensorMap tmap_b;
  CUDA_CHECK(make_tma_2d_tiled(&tmap_b, dB, N, K,
                               /*box_rows=*/N_PER_PEER, /*box_cols=*/K,
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

  std::vector<float> hOut(M_CLUSTER * N);
  CUDA_CHECK(cudaMemcpy(hOut.data(), dOut, M_CLUSTER * N * sizeof(float),
                        cudaMemcpyDeviceToHost));
  cudaFree(dA); cudaFree(dB); cudaFree(dOut);

  // Verify first 32 rows of each peer's half.
  int fails = 0;
  float max_d = 0.0f;
  for (int p = 0; p < 2; ++p) {
    for (int lane = 0; lane < 32; ++lane) {
      int m = p * M_PER_PEER + lane;
      for (int n = 0; n < N; ++n) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
          acc += __bfloat162float(hA[m * K + k]) *
                 __bfloat162float(hB[n * K + k]);
        }
        float got = hOut[m * N + n];
        float d = fabsf(got - acc);
        if (d > 0.5f) {
          if (fails < 6)
            fprintf(stderr, "  D[%d, %d] = %.1f, want %.1f (peer=%d)\n",
                    m, n, got, acc, p);
          ++fails;
        }
        if (d > max_d) max_d = d;
      }
    }
  }
  printf("gather4 + 2SM MMA + peer1_done cross-CTA sync (M=%d, N=%d, K=%d): "
         "fails = %d / %d, max_diff = %.3f\n",
         M_CLUSTER, N, K, fails, 64 * N, max_d);
  if (fails) FAIL("gather4 + 2SM MMA + cross-CTA sync mismatch");
  PASS();
}
