// ARCH: sm_90a
// 31_mbarrier_arrive_tx_test.cu -- arrive.expect_tx + complete_tx round-trip.
// complete_tx is issued by a separate thread to mimic the TMA completion
// path; the mbarrier should flip phase when arrival+tx are satisfied.
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
#include "../src/primitives/29_mbarrier_init.cuh"
#include "../src/primitives/31_mbarrier_arrive_tx.cuh"
#include "../src/primitives/33_mbarrier_try_wait.cuh"
#include "../src/primitives/67_mapa.cuh"
#include "29_mbarrier_init.cuh"
#include "31_mbarrier_arrive_tx.cuh"
#include "33_mbarrier_try_wait.cuh"

// =============================================================================
// ours
// =============================================================================

// ARCH: sm_90a


__global__ void k_tx(int* ok) {
  __shared__ __align__(16) uint64_t mbar;
  if (threadIdx.x == 0) mbarrier_init(smem_ptr_u32(&mbar), 1);
  __syncthreads();
  if (threadIdx.x == 0)
    mbarrier_arrive_expect_tx(smem_ptr_u32(&mbar), 128);
  __syncthreads();
  // Another thread credits the 128 bytes.
  if (threadIdx.x == 1) mbarrier_complete_tx(smem_ptr_u32(&mbar), 128);
  __syncthreads();
  if (threadIdx.x == 0) {
    mbarrier_wait_parity(smem_ptr_u32(&mbar), 0);
    *ok = 1;
    mbarrier_inval(smem_ptr_u32(&mbar));
  }
}

static int run_ours() {
  int* d = nullptr; CUDA_CHECK(cudaMalloc(&d, 4));
  CUDA_CHECK(cudaMemset(d, 0, 4));
  k_tx<<<1, 32>>>(d);
  CUDA_CHECK(cudaDeviceSynchronize());
  int h = 0; CUDA_CHECK(cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost));
  cudaFree(d);
  if (h != 1) FAIL("arrive.expect_tx + complete_tx did not flip phase");
  printf("mbarrier.arrive.expect_tx + complete_tx : OK\n");
  PASS();
}

// =============================================================================
// theirs
// =============================================================================

// Runtime test: mbarrier.arrive.expect_tx + mbarrier.complete_tx
//
// This test manually credits tx-bytes using complete_tx so the barrier is
// satisfied WITHOUT a real TMA load. Normally TMA's cp.async.bulk.tensor.*
// writes credit the tx-count automatically; here we stand in for that path
// to exercise the barrier half of the handshake in a tiny unit test.
// (Real TMA-integrated expect_tx is covered by TMA-load tests.)
//
// Flow:
//   init(arrive_count=1)
//   thread 0: arrive_expect_tx(1024)   -> arrivals=1/1, tx_expected=1024
//   thread 0: complete_tx(1024)        -> tx_credited=1024, phase completes
//   all threads: wait_parity(0)        -> must return immediately
//   thread 0: write sentinel to out

// -- Test A: arrive_expect_tx + manual complete_tx ---------------------
__global__ void expect_complete_kernel(uint32_t* out, uint32_t expected_bytes) {
    __shared__ __align__(8) uint64_t mbar[1];
    uint32_t addr = smem_ptr_u32(mbar);

    if (threadIdx.x == 0) {
        mbarrier_init(addr, 1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        // Register expected bytes and consume the only arrival slot.
        mbarrier_arrive_expect_tx(addr, expected_bytes);
        // Credit the bytes manually. This is what TMA would do on completion.
        mbarrier_complete_tx(addr, expected_bytes);
    }

    // All threads wait for phase 0. Phase is complete iff arrivals==0 AND
    // tx_credited >= tx_expected. If wait returns, both conditions were met.
    mbarrier_wait_parity(addr, 0);

    // Write sentinel AFTER wait so passing means wait returned.
    if (threadIdx.x == 0) out[0] = 0xC0FFEE01u;
    if (threadIdx.x == 1) out[1] = 0xC0FFEE02u;
    if (threadIdx.x == 2) out[2] = expected_bytes;

    __syncthreads();
    if (threadIdx.x == 0) mbarrier_inval(addr);
}

// -- Test B: standalone expect_tx + complete_tx, then separate arrive -
// Same overall result but decoupled. arrive_count=1, and thread 0 does a
// plain mbarrier_arrive (via arrive_expect_tx_state with 0 bytes would also
// work, but we use the dedicated standalone functions for variety).
__global__ void standalone_expect_kernel(uint32_t* out, uint32_t expected_bytes) {
    __shared__ __align__(8) uint64_t mbar[1];
    uint32_t addr = smem_ptr_u32(mbar);

    if (threadIdx.x == 0) mbarrier_init(addr, 1);
    __syncthreads();

    if (threadIdx.x == 0) {
        // Register the expectation BEFORE any arrival.
        mbarrier_expect_tx(addr, expected_bytes);
        // Credit bytes.
        mbarrier_complete_tx(addr, expected_bytes);
        // Consume the arrival slot with a combined op (0 extra bytes).
        // arrive_expect_tx_state(addr, 0) -> increments tx_expected by 0 and arrives.
        uint64_t st = mbarrier_arrive_expect_tx_state(addr, 0);
        (void)st;
    }

    mbarrier_wait_parity(addr, 0);

    if (threadIdx.x == 0) out[0] = 0xDEAD0001u;
    if (threadIdx.x == 3) out[3] = 0xDEAD0003u;

    __syncthreads();
    if (threadIdx.x == 0) mbarrier_inval(addr);
}

static int run_theirs() {
    CUDA_CHECK(cudaFree(0));
    bool all_pass = true;

    const int THREADS = 32;
    uint32_t h_out[8];

    // --- Test A ---
    uint32_t* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(h_out)));
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(h_out)));
    expect_complete_kernel<<<1, THREADS>>>(d_out, 1024u);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));
    bool okA = (h_out[0] == 0xC0FFEE01u && h_out[1] == 0xC0FFEE02u && h_out[2] == 1024u);
    if (okA) printf("  expect_tx + complete_tx:      OK (wait returned, sentinels %08x %08x bytes=%u)\n",
                    h_out[0], h_out[1], h_out[2]);
    else { printf("  expect_tx + complete_tx: FAIL (got %08x %08x %u)\n",
                  h_out[0], h_out[1], h_out[2]); all_pass = false; }

    // --- Test B ---
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(h_out)));
    standalone_expect_kernel<<<1, THREADS>>>(d_out, 2048u);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));
    bool okB = (h_out[0] == 0xDEAD0001u && h_out[3] == 0xDEAD0003u);
    if (okB) printf("  standalone expect/complete:   OK (decoupled path returned)\n");
    else { printf("  standalone expect/complete: FAIL (got %08x %08x)\n",
                  h_out[0], h_out[3]); all_pass = false; }

    // --- perf ---
    GpuTimer t;
    const int ITERS = 200;
    t.begin();
    for (int i = 0; i < ITERS; i++)
        expect_complete_kernel<<<1, THREADS>>>(d_out, 1024u);
    t.end();
    printf("  perf:                         %.2f us/launch (expect/complete cycle)\n",
           t.elapsed_ms() * 1000.0f / ITERS);

    cudaFree(d_out);
    if (all_pass) { PASS(); return 0; }
    else { FAIL("some subtests failed"); return 1; }
}


// arrive.expect_tx.release.cluster -- 2-CTA cluster, peer CTA
// (CTA 0) registers expected bytes against an mbar visible to peers via
// .cluster scope; CTA 1 credits the bytes via mbarrier.complete_tx using
// a cluster-translated mbar address (mapa). CTA 0 waits and confirms.
__global__ void __cluster_dims__(2, 1, 1)
k_arrive_expect_tx_release_cluster(uint32_t* g_done) {
  __shared__ __align__(16) uint64_t mbar;
  uint32_t mbar_addr = smem_ptr_u32(&mbar);
  if (threadIdx.x == 0) {
    mbarrier_init(mbar_addr, /*arrive_count=*/1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  // CTA 0 (issuer) registers expected bytes + arrives via .release.cluster.
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    mbarrier_arrive_expect_tx_release_cluster(mbar_addr, 64u);
  }
  // CTA 1 (peer) credits the bytes against CTA 0's mbar via cluster-mapped
  // address (mapa translates a local SMEM addr to a peer CTA's SMEM).
  if (threadIdx.x == 0 && blockIdx.x == 1) {
    uint32_t peer_mbar_addr;
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;\n"
                 : "=r"(peer_mbar_addr) : "r"(mbar_addr));
    mbarrier_complete_tx(peer_mbar_addr, 64u);
  }
  if (blockIdx.x == 0) {
    mbarrier_wait_parity(mbar_addr, 0);
    if (threadIdx.x == 0) g_done[0] = 1;
  }
  __syncthreads();
  if (threadIdx.x == 0)
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n"
                 :: "r"(mbar_addr));
}

static int run_release_cluster() {
  uint32_t* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d, 0, sizeof(uint32_t)));
  k_arrive_expect_tx_release_cluster<<<2, 32>>>(d);
  CUDA_CHECK(cudaDeviceSynchronize());
  uint32_t h = 0;
  CUDA_CHECK(cudaMemcpy(&h, d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  cudaFree(d);
  if (h != 1u) {
    fprintf(stderr, "arrive.expect_tx.release.cluster: done=%u (expected 1)\n", h);
    FAIL("CTA 0 did not observe completion");
  }
  printf("mbarrier_arrive_expect_tx_release_cluster: cluster handshake OK\n");
  PASS();
}

// arrive.expect_tx.shared::cluster -- 2-CTA cluster. CTA 0 produces a
// cluster-shared address pointing at CTA 1's mbar via mapa, then arrives +
// registers expected bytes on the remote mbar. CTA 1 credits the bytes
// locally via complete_tx and waits. Exercises the .shared::cluster.b64
// form of arrive.expect_tx (used in composite 106 / sched_warp CLC paths).
__global__ void __cluster_dims__(2, 1, 1)
k_arrive_expect_tx_cluster(uint32_t* g_done) {
  __shared__ __align__(16) uint64_t mbar;
  uint32_t mbar_addr = smem_ptr_u32(&mbar);
  if (threadIdx.x == 0) {
    mbarrier_init(mbar_addr, /*arrive_count=*/1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  asm volatile("barrier.cluster.arrive.relaxed;\n" ::: "memory");
  asm volatile("barrier.cluster.wait.acquire;\n" ::: "memory");

  // CTA 0 (issuer): remap our local mbar address to CTA 1's view, then
  // arrive + expect_tx on that remote mbar via .shared::cluster.b64.
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    uint32_t peer1_mbar = mapa_shared_cluster_u32(mbar_addr, /*rank=*/1);
    mbarrier_arrive_expect_tx_cluster(peer1_mbar, 64u);
  }

  // CTA 1 (owner): credit the bytes locally; this is the path TMA HW would
  // normally take on completion. Then wait for the barrier and confirm.
  if (blockIdx.x == 1) {
    if (threadIdx.x == 0) {
      mbarrier_complete_tx(mbar_addr, 64u);
    }
    mbarrier_wait_parity(mbar_addr, 0);
    if (threadIdx.x == 0) g_done[0] = 1;
  }

  __syncthreads();
  if (threadIdx.x == 0)
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n"
                 :: "r"(mbar_addr));
}

static int run_arrive_expect_tx_cluster() {
  uint32_t* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d, 0, sizeof(uint32_t)));

  cudaLaunchConfig_t config = {};
  config.gridDim          = dim3(2, 1, 1);
  config.blockDim         = dim3(32, 1, 1);
  config.dynamicSmemBytes = 0;
  cudaLaunchAttribute attr;
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim = {2, 1, 1};
  config.attrs    = &attr;
  config.numAttrs = 1;
  CUDA_CHECK(cudaLaunchKernelEx(&config, k_arrive_expect_tx_cluster, d));
  CUDA_CHECK(cudaDeviceSynchronize());

  uint32_t h = 0;
  CUDA_CHECK(cudaMemcpy(&h, d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  cudaFree(d);
  if (h != 1u) {
    fprintf(stderr, "arrive.expect_tx.shared::cluster: done=%u (expected 1)\n", h);
    FAIL("CTA 1 did not observe completion");
  }
  printf("mbarrier_arrive_expect_tx_cluster: cluster-shared arrive OK\n");
  PASS();
}

int main() {
  int rc_ours   = run_ours();
  int rc_theirs = run_theirs();
  int rc_rc     = run_release_cluster();
  int rc_aec    = run_arrive_expect_tx_cluster();
  return (rc_ours == 0 && rc_theirs == 0 && rc_rc == 0 && rc_aec == 0) ? 0 : 1;
}
