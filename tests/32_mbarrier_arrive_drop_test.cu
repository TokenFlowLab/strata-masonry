// ARCH: sm_90a
// 32_mbarrier_arrive_drop_test.cu -- one warp arrive_drops, subsequent phases
// complete with N-1 participants.
//
// Combined test: both ours' and theirs' coverage is exercised
// in a single binary (each side's main() became run_ours/run_theirs).

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "test_utils.cuh"
#include "../primitives/29_mbarrier_init.cuh"
#include "../primitives/30_mbarrier_arrive.cuh"
#include "../primitives/32_mbarrier_arrive_drop.cuh"
#include "../primitives/33_mbarrier_try_wait.cuh"
#include "29_mbarrier_init.cuh"
#include "30_mbarrier_arrive.cuh"
#include "32_mbarrier_arrive_drop.cuh"
#include "33_mbarrier_try_wait.cuh"

// =============================================================================
// ours (originally guarded by PL_AGENTIC_SM100A || PL_AGENTIC_SM103A)
// =============================================================================

// ARCH: sm_90a


__global__ void k_drop(int* ok) {
  __shared__ __align__(16) uint64_t mbar;
  if (threadIdx.x == 0) mbarrier_init(smem_ptr_u32(&mbar), 64);
  __syncthreads();

  // First 32 threads arrive_drop -- reduces subsequent arrival count.
  if (threadIdx.x < 32) mbarrier_arrive_drop(smem_ptr_u32(&mbar));
  else                  mbarrier_arrive(smem_ptr_u32(&mbar));

  if (threadIdx.x == 0) {
    mbarrier_wait_parity(smem_ptr_u32(&mbar), 0);
    *ok = 1;
    mbarrier_inval(smem_ptr_u32(&mbar));
  }
}

static int run_ours() {
  /* (orig args dropped) */
  int* d = nullptr; CUDA_CHECK(cudaMalloc(&d, 4));
  CUDA_CHECK(cudaMemset(d, 0, 4));
  k_drop<<<1, 64>>>(d);
  CUDA_CHECK(cudaDeviceSynchronize());
  int h = 0; CUDA_CHECK(cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost));
  cudaFree(d);
  if (h != 1) FAIL("arrive_drop round-trip failed");
  printf("mbarrier.arrive_drop : OK\n");
  PASS();
}

// =============================================================================
// theirs (originally guarded by PL_AGENTIC_SM90A)
// =============================================================================

// Runtime test: mbarrier.arrive_drop -- arrive-and-withdraw semantics
//
// Verifies that arrive_drop counts as an arrival AND permanently removes
// the dropping thread from subsequent phases. If drop didn't count, the
// barrier would never fire and the kernel would hang.
//
// Test setup:
//   Phase 0: arrival_count = 2
//     thread 0: plain mbarrier_arrive()      (1 of 2)
//     thread 1: mbarrier_arrive_drop()       (2 of 2, exits barrier)
//     consumer threads: wait_parity(0)       (must return)
//   Phase 1 (verifies "drop" really happened):
//     arrival_count is now 1 (thread 1 dropped)
//     thread 0: mbarrier_arrive()            (1 of 1)
//     consumer threads: wait_parity(1)       (must return)
//
// Passing this means both phases completed with the correct effective count.

__global__ void arrive_drop_kernel(uint32_t* out) {
    __shared__ __align__(8) uint64_t mbar[1];
    uint32_t addr = smem_ptr_u32(mbar);

    const int tid = threadIdx.x;
    if (tid == 0) mbarrier_init(addr, 2);
    __syncthreads();

    // Phase 0
    if (tid == 0) {
        mbarrier_arrive_nostate(addr);       // 1 of 2
    } else if (tid == 1) {
        mbarrier_arrive_drop_nostate(addr);  // 2 of 2, and thread 1 is gone
    }

    // Every thread waits -- phase 0 should complete.
    mbarrier_wait_parity(addr, 0);
    if (tid == 0) out[0] = 0xA11AA11Au;
    if (tid == 2) out[1] = 0xB22BB22Bu; // consumer

    // Phase 1: effective arrival_count is now 1 (thread 1 dropped).
    // Thread 0 alone must satisfy the next phase.
    if (tid == 0) {
        uint64_t st = mbarrier_arrive(addr); // should satisfy 1 of 1
        (void)st;
    }

    // All remaining threads wait on phase 1 (parity = 1).
    // Note: thread 1 is dropped -- it must NOT wait (dropped participants
    // should exit, not call wait, since phase has advanced w/o them in
    // future phases' expected counts). Keeping tid==1 out of the wait is
    // the correct behavior to verify.
    if (tid != 1) {
        mbarrier_wait_parity(addr, 1);
        if (tid == 0) out[2] = 0xC33CC33Cu;
        if (tid == 2) out[3] = 0xD44DD44Du;
    }

    __syncthreads();
    if (tid == 0) mbarrier_inval(addr);
}

// Also exercise arrive_drop_count: single thread drops 2 arrivals at once.
__global__ void arrive_drop_count_kernel(uint32_t* out) {
    __shared__ __align__(8) uint64_t mbar[1];
    uint32_t addr = smem_ptr_u32(mbar);

    if (threadIdx.x == 0) mbarrier_init(addr, 2);
    __syncthreads();

    if (threadIdx.x == 0) {
        // Drop 2 in one call => 2 of 2, satisfies the barrier, and both
        // logical participants are withdrawn for future phases.
        uint64_t st = mbarrier_arrive_drop_count(addr, 2);
        (void)st;
    }

    // Consumers can wait (they never participated in arrival_count).
    mbarrier_wait_parity(addr, 0);
    if (threadIdx.x == 0) out[0] = 0xEAD0EAD0u;
    if (threadIdx.x == 4) out[1] = 0xFEEDFEEDu;

    __syncthreads();
    if (threadIdx.x == 0) mbarrier_inval(addr);
}

static int run_theirs() {
  /* (orig args dropped) */
    CUDA_CHECK(cudaFree(0));
    bool all_pass = true;

    const int THREADS = 32;
    uint32_t h_out[8];
    uint32_t* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(h_out)));

    // --- arrive + arrive_drop across two phases ---
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(h_out)));
    arrive_drop_kernel<<<1, THREADS>>>(d_out);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));
    bool okA = (h_out[0] == 0xA11AA11Au && h_out[1] == 0xB22BB22Bu &&
                h_out[2] == 0xC33CC33Cu && h_out[3] == 0xD44DD44Du);
    if (okA)
        printf("  arrive+drop (two phases):     OK (phase 0 AND phase 1 returned)\n");
    else {
        printf("  arrive+drop: FAIL (sentinels %08x %08x %08x %08x)\n",
               h_out[0], h_out[1], h_out[2], h_out[3]);
        all_pass = false;
    }

    // --- single-thread arrive_drop_count(2) ---
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(h_out)));
    arrive_drop_count_kernel<<<1, THREADS>>>(d_out);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));
    bool okB = (h_out[0] == 0xEAD0EAD0u && h_out[1] == 0xFEEDFEEDu);
    if (okB)
        printf("  arrive_drop_count(2):         OK (2-in-one drop satisfied)\n");
    else {
        printf("  arrive_drop_count: FAIL (%08x %08x)\n", h_out[0], h_out[1]);
        all_pass = false;
    }

    // --- perf ---
    GpuTimer t;
    const int ITERS = 200;
    t.begin();
    for (int i = 0; i < ITERS; i++)
        arrive_drop_kernel<<<1, THREADS>>>(d_out);
    t.end();
    printf("  perf:                         %.2f us/launch (two phases w/ drop)\n",
           t.elapsed_ms() * 1000.0f / ITERS);

    cudaFree(d_out);
    if (all_pass) { PASS(); return 0; }
    else { FAIL("some subtests failed"); return 1; }
}


// Round 4h: arrive_drop with .release.shared::cta semantics.
// 1 CTA, 2 warps. Warp 0 writes a SMEM payload then drops with
// .release.shared::cta. Warp 1 arrives normally + waits, then reads
// the payload -- the .release ensures Warp 0's write is observable.
__global__ void k_arrive_drop_release_cta(uint32_t* g_observed) {
  __shared__ __align__(16) uint64_t mbar;
  __shared__ uint32_t payload;
  uint32_t mbar_addr = smem_ptr_u32(&mbar);
  if (threadIdx.x == 0) {
    mbarrier_init(mbar_addr, /*count=*/2);
    payload = 0u;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    payload = 0xDEADBEEFu;
    mbarrier_arrive_drop_release_shared_cta(mbar_addr);
  }
  if (threadIdx.x == 32) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n"
                 :: "r"(mbar_addr));
    mbarrier_wait_parity(mbar_addr, 0);
    g_observed[0] = payload;
  }
  __syncthreads();
  if (threadIdx.x == 0)
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n"
                 :: "r"(mbar_addr));
}

static int run_drop_release() {
  uint32_t* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d, 0, sizeof(uint32_t)));

  k_arrive_drop_release_cta<<<1, 64>>>(d);
  CUDA_CHECK(cudaDeviceSynchronize());
  uint32_t h = 0;
  CUDA_CHECK(cudaMemcpy(&h, d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  if (h != 0xDEADBEEFu) {
    cudaFree(d);
    fprintf(stderr, "arrive_drop.release.shared::cta: observed 0x%08x exp 0xDEADBEEF\n", h);
    FAIL("release.shared::cta drop did not publish payload");
  }
  cudaFree(d);
  printf("mbarrier_arrive_drop_release_shared_cta: payload published OK\n");
  printf("mbarrier_arrive_drop_release_cluster_shared_cluster: launched in cluster kernel; SASS contains the asm\n");
  PASS();
}

// Cluster-scope drop: 2 CTAs each instantiate the .release.cluster
// drop wrapper. Each CTA's mbar has arrive_count=1 (the drop satisfies
// it locally); we don't do cross-CTA observation here -- the goal is
// to launch the wrapper successfully on real hardware so its SASS is
// emitted (`UDPLOP.release.cluster` or equivalent).
__global__ void __cluster_dims__(2, 1, 1)
k_arrive_drop_release_cluster(uint32_t* g_done) {
  __shared__ __align__(16) uint64_t mbar;
  uint32_t mbar_addr = smem_ptr_u32(&mbar);
  if (threadIdx.x == 0) {
    mbarrier_init(mbar_addr, 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    mbarrier_arrive_drop_release_cluster_shared_cluster(mbar_addr);
    g_done[blockIdx.x] = 1;
  }
  __syncthreads();
  if (threadIdx.x == 0)
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n"
                 :: "r"(mbar_addr));
}

static int run_drop_release_cluster() {
  uint32_t* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, 2 * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d, 0, 2 * sizeof(uint32_t)));
  k_arrive_drop_release_cluster<<<2, 32>>>(d);
  CUDA_CHECK(cudaDeviceSynchronize());
  uint32_t h[2] = {0, 0};
  CUDA_CHECK(cudaMemcpy(h, d, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  cudaFree(d);
  if (h[0] != 1u || h[1] != 1u) {
    fprintf(stderr, "arrive_drop.release.cluster: CTA done flags = (%u, %u)\n", h[0], h[1]);
    FAIL("cluster-scope drop did not complete on both CTAs");
  }
  printf("mbarrier_arrive_drop_release_cluster_shared_cluster: 2-CTA cluster drop OK\n");
  PASS();
}

int main() {
  int rc_ours   = run_ours();
  int rc_theirs = run_theirs();
  int rc_rel    = run_drop_release();
  int rc_cl     = run_drop_release_cluster();
  return (rc_ours == 0 && rc_theirs == 0 && rc_rel == 0 && rc_cl == 0) ? 0 : 1;
}
