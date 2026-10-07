#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// ARCH: sm_100a
// 61_clc_try_cancel_test.cu -- compile smoke + multicast::all runtime check.

#include "test_utils.cuh"
#include "../src/primitives/29_mbarrier_init.cuh"
#include "../src/primitives/31_mbarrier_arrive_tx.cuh"
#include "../src/primitives/33_mbarrier_try_wait.cuh"
#include "../src/primitives/61_clc_try_cancel.cuh"
#include "../src/primitives/67_mapa.cuh"

__global__ void k_clc() {
  __shared__ __align__(16) uint8_t slot[16];
  __shared__ __align__(16) uint64_t mbar;
  if (threadIdx.x == 0) {
    mbarrier_init(smem_ptr_u32(&mbar), 1);
    clc_try_cancel_async(smem_ptr_u32(slot), smem_ptr_u32(&mbar));
  }
}

// 2-CTA cluster, leader issues multicast::cluster::all; both CTAs init
// their own mbar (arrive_count=1) and wait. The multicast credits each
// CTA's mbar via complete_tx. After the wait returns, both CTAs decode
// the response payload and report ctaid_x / canceled to gmem.
//
// Grid is sized to 2 clusters (4 CTAs) so the scheduler has at least one
// cancelable tile remaining when CLUSTER 0 issues. The first cluster's
// own try_cancel succeeds (canceled=true) and reports the next available
// cluster's coords; both CTAs in cluster 0 must see the same payload.
__global__ void __cluster_dims__(2, 1, 1)
k_clc_multicast_all(uint32_t* out) {
  __shared__ __align__(16) uint32_t slot[4];
  __shared__ __align__(16) uint64_t mbar;

  // We only inspect cluster 0's view; cluster 1 just fills its slot to
  // keep the scheduler honest.
  uint32_t cluster_rank;
  asm("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(cluster_rank));
  uint32_t cluster_id;
  asm("mov.u32 %0, %%clusterid.x;\n" : "=r"(cluster_id));

  for (int i = threadIdx.x; i < 4; i += blockDim.x) slot[i] = 0;
  if (threadIdx.x == 0) {
    mbarrier_init(smem_ptr_u32(&mbar), 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  asm volatile("barrier.cluster.arrive.relaxed;\n" ::: "memory");
  asm volatile("barrier.cluster.wait.acquire;\n" ::: "memory");

  // Cluster 0 leader sets up the cluster handshake. Lane k of cluster 0
  // CTA 0 mapas its local mbar address to peer k and issues
  // arrive.expect_tx(16) -- this primes both CTAs' mbars to expect 16
  // bytes (size of a try_cancel response payload) and consumes the
  // arrive_count slot. Then lane 0 issues the multicast::all, which
  // credits 16 bytes to each CTA's mbar via complete_tx -- satisfying
  // the wait.
  if (cluster_id == 0 && cluster_rank == 0) {
    const int lane = threadIdx.x;
    if (lane < 2) {
      uint32_t local_mbar = smem_ptr_u32(&mbar);
      uint32_t peer_mbar = mapa_shared_cluster_u32(local_mbar, lane);
      mbarrier_arrive_expect_tx_cluster(peer_mbar, 16u);
    }
    if (threadIdx.x == 0) {
      clc_try_cancel_async_multicast_all(smem_ptr_u32(slot),
                                         smem_ptr_u32(&mbar));
    }
  }

  // Cluster 0: both CTAs wait on their local mbar and read their slot.
  if (cluster_id == 0) {
    mbarrier_wait_parity(smem_ptr_u32(&mbar), 0);

    if (threadIdx.x == 0) {
      uint32_t d0, d2;
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      asm volatile("ld.shared.b32 %0, [%1];\n"
                   : "=r"(d0) : "r"(smem_ptr_u32(&slot[0])));
      asm volatile("ld.shared.b32 %0, [%1];\n"
                   : "=r"(d2) : "r"(smem_ptr_u32(&slot[2])));
      // Lay out cluster-0 results at indices [rank * 2 + 0/1].
      out[cluster_rank * 2 + 0] = d0;            // ctaid_x of canceled tile
      out[cluster_rank * 2 + 1] = (d2 & 1u);     // canceled flag
    }
  }

  __syncthreads();
  if (threadIdx.x == 0)
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n"
                 :: "r"(smem_ptr_u32(&mbar)));
}

int main() {
  // -- A: compile smoke (single-CTA, non-multicast variant) ----------
  k_clc<<<1, 32>>>();
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("clusterlaunchcontrol.try_cancel : compile + run OK\n");

  // -- B: multicast::cluster::all runtime check (2-CTA cluster, 2 clusters)
  uint32_t* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, 4 * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d, 0xff, 4 * sizeof(uint32_t)));

  cudaLaunchConfig_t config = {};
  config.gridDim          = dim3(4, 1, 1);  // 2 clusters of 2 CTAs each
  config.blockDim         = dim3(32, 1, 1);
  config.dynamicSmemBytes = 0;
  cudaLaunchAttribute attr;
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim = {2, 1, 1};
  config.attrs    = &attr;
  config.numAttrs = 1;
  CUDA_CHECK(cudaLaunchKernelEx(&config, k_clc_multicast_all, d));
  CUDA_CHECK(cudaDeviceSynchronize());

  uint32_t h[4];
  CUDA_CHECK(cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost));
  cudaFree(d);

  // Both CTAs in cluster 0 must see the same payload (multicast::all
  // writes the same 128-bit result into every CTA's slot).
  bool ok = (h[0] == h[2]) && (h[1] == h[3]);
  printf("  multicast::cluster::all rank0=(ctaid_x=%u,canceled=%u) "
         "rank1=(ctaid_x=%u,canceled=%u)\n", h[0], h[1], h[2], h[3]);
  if (!ok) FAIL("multicast::all peer mismatch");

  printf("clusterlaunchcontrol.try_cancel.multicast::cluster::all : OK\n");
  PASS();
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
