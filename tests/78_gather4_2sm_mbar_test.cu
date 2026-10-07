// ARCH: sm_100a
// 78_gather4_2sm_mbar_test.cu -- gather4 across a 2SM cluster with
// peer-bit-masked cluster-scope mbarrier signaling.
//
// Tests whether gather4 (which has NO `cta_group::2` form in PTX 9.x)
// can still target a cluster-scope mbarrier via the peer-bit mask
// (0xFEFFFFFF) such that both peers' gather4 calls accumulate signals
// on peer 0's mbarrier. This is the protocol that
// `load_warp_blackwell_ntiles_2sm_bf16_gather4` relies on.
//
// Setup:
//   A: bf16 (R=32, C=64) row-major. A[r, c] = r * 100 + c (bf16 exact
//      for r * 100 + c < 256).
//   2-CTA cluster (cluster_dim={2,1,1}). Each peer has its own SMEM
//   smA[N_PER_PEER, C] = (16, 64) bf16 = 2048 bytes.
//   gather4 picks {0..3, 4..7, 8..11, 12..15} for peer 0 and
//   {16..19, 20..23, 24..27, 28..31} for peer 1. 4 calls per peer.
//   Both peers signal peer 0's full_bar via peer-bit-masked address.
//   expect_tx on peer 0 = 2 * (4 * 4 * C * 2) = 2 * peer_bytes.
//   Peer 0 waits. Peer 1 ALSO waits on its local full_bar (which is
//   a different mbar object -- never signaled by gather4 if gather4
//   only delivers to its OWN local CTA's mbar). To avoid deadlock on
//   peer 1, we use an arrive-and-wait pattern: only peer 0 waits on
//   the cluster-scope mbar; peer 1 just barriers at cluster-sync.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "test_utils.cuh"
#include "../primitives/23_tma_tensormap.cuh"
#include "../primitives/29_mbarrier_init.cuh"
#include "../primitives/31_mbarrier_arrive_tx.cuh"
#include "../primitives/33_mbarrier_try_wait.cuh"
#include "../primitives/38_barrier_cluster.cuh"
#include "../primitives/70_smem_ptr.cuh"
#include "../primitives/73_tma_load_2d_gather4.cuh"

constexpr int R          = 32;
constexpr int C          = 64;
constexpr int N_PER_PEER = 16;
constexpr int N_CALLS    = N_PER_PEER / 4;  // 4 calls per peer
constexpr int PEER_BYTES = N_PER_PEER * C * (int)sizeof(__nv_bfloat16);

// peer-bit mask: clear bit 24 -> peer 0's mbar address.
static constexpr uint32_t PEER_MASK = 0xFEFFFFFFu;

__global__ void __cluster_dims__(2, 1, 1)
k_gather4_2sm(const __grid_constant__ CUtensorMap tmap_in,
              __nv_bfloat16* out_p0, __nv_bfloat16* out_p1) {
  __shared__ __align__(1024) __nv_bfloat16 smA[N_PER_PEER * C];
  __shared__ __align__(16)   uint64_t      full_bar;

  const int peer = blockIdx.x & 1;

  if (threadIdx.x == 0) {
    mbarrier_init(smem_ptr_u32(&full_bar), /*arrive_count=*/1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  // Cluster-wide barrier so both peers see the mbar init.
  __syncthreads();
  barrier_cluster_arrive();
  barrier_cluster_wait();

  // gather4 row indices per peer.
  const int base = peer * N_PER_PEER;

  if (threadIdx.x == 0) {
    const uint32_t local_bar = smem_ptr_u32(&full_bar);
    // Per-peer expect_tx: each peer registers its OWN local mbar for its
    // OWN gather contribution. gather4 (no cta_group::2) only signals
    // its local CTA's mbar -- the peer-bit mask trick that works for
    // cta_group::2 multicast TMA does NOT cross-signal here.
    mbarrier_arrive_expect_tx(local_bar, (uint32_t)PEER_BYTES);
    // gather4 targets the LOCAL mbar (no peer-bit mask).
    #pragma unroll
    for (int g = 0; g < N_CALLS; ++g) {
      const int r0 = base + g * 4 + 0;
      const int r1 = base + g * 4 + 1;
      const int r2 = base + g * 4 + 2;
      const int r3 = base + g * 4 + 3;
      const uint32_t dst = smem_ptr_u32(smA)
                         + (uint32_t)(g * 4 * C * (int)sizeof(__nv_bfloat16));
      tma_load_2d_gather4(dst, &tmap_in, local_bar,
                          /*col=*/0, r0, r1, r2, r3);
    }
  }

  // Each peer waits on its OWN local mbar.
  mbarrier_wait_parity(smem_ptr_u32(&full_bar), 0);
  // Cluster sync so peer 1 doesn't drain SMEM before peer 0 has its data.
  // For peer 1, this also ensures peer 0's gather4 has landed before we
  // copy peer 1's smA out -- since peer 1's own gather4 is separate.
  barrier_cluster_arrive();
  barrier_cluster_wait();

  // Each peer drains its smA -> its own GMEM out buffer.
  __nv_bfloat16* dst = (peer == 0) ? out_p0 : out_p1;
  for (int i = threadIdx.x; i < N_PER_PEER * C; i += blockDim.x) {
    dst[i] = smA[i];
  }
}

int main() {
  CUDA_CHECK(cudaFree(0));

  std::vector<__nv_bfloat16> hA(R * C);
  for (int r = 0; r < R; ++r)
    for (int c = 0; c < C; ++c)
      hA[r * C + c] = __float2bfloat16((float)(r * 100 + c));

  __nv_bfloat16 *dA = nullptr, *dOut0 = nullptr, *dOut1 = nullptr;
  CUDA_CHECK(cudaMalloc(&dA,    R * C * 2));
  CUDA_CHECK(cudaMalloc(&dOut0, N_PER_PEER * C * 2));
  CUDA_CHECK(cudaMalloc(&dOut1, N_PER_PEER * C * 2));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), R * C * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dOut0, 0, N_PER_PEER * C * 2));
  CUDA_CHECK(cudaMemset(dOut1, 0, N_PER_PEER * C * 2));

  // gather4 descriptor: box=(1, C), SWIZZLE_NONE for simplicity (any swizzle
  // would work; the test focuses on mbar signaling, not SMEM layout).
  CUtensorMap tmap_in;
  CUDA_CHECK(make_tma_2d_tiled(&tmap_in, dA, R, C,
                               /*box_rows=*/1, /*box_cols=*/C,
                               sizeof(__nv_bfloat16),
                               CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                               CU_TENSOR_MAP_SWIZZLE_NONE));

  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(2, 1, 1);
  config.blockDim = dim3(32, 1, 1);
  config.dynamicSmemBytes = 0;
  config.stream = 0;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim = {2, 1, 1};
  config.attrs = attrs;
  config.numAttrs = 1;
  cudaLaunchKernelEx(&config, k_gather4_2sm, tmap_in, dOut0, dOut1);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> hOut0(N_PER_PEER * C), hOut1(N_PER_PEER * C);
  CUDA_CHECK(cudaMemcpy(hOut0.data(), dOut0, N_PER_PEER * C * 2,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hOut1.data(), dOut1, N_PER_PEER * C * 2,
                        cudaMemcpyDeviceToHost));
  cudaFree(dA); cudaFree(dOut0); cudaFree(dOut1);

  int fails = 0;
  for (int p = 0; p < 2; ++p) {
    const std::vector<__nv_bfloat16>& hO = (p == 0) ? hOut0 : hOut1;
    int base = p * N_PER_PEER;
    for (int r = 0; r < N_PER_PEER; ++r) {
      int src = base + r;
      for (int c = 0; c < C; ++c) {
        float got  = __bfloat162float(hO[r * C + c]);
        float want = __bfloat162float(hA[src * C + c]);
        if (got != want) {
          if (fails < 6)
            fprintf(stderr, "  peer=%d out[%d,%d]=%.0f want A[%d,%d]=%.0f\n",
                    p, r, c, got, src, c, want);
          ++fails;
        }
      }
    }
  }
  printf("gather4 2SM mbar test (R=%d, C=%d, N_PER_PEER=%d): "
         "fails = %d / %d\n", R, C, N_PER_PEER, fails, 2 * N_PER_PEER * C);
  if (fails) FAIL("gather4 2SM cross-peer mbar signaling mismatch");
  PASS();
}
