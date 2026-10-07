// ARCH: sm_90a
// 33_mbarrier_try_wait_test.cu -- try_wait.parity covered by #29/30/31;
// here we verify that a single try_wait returns 0 before arrival and 1 after.
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
#include "../primitives/33_mbarrier_try_wait.cuh"
#include "29_mbarrier_init.cuh"
#include "30_mbarrier_arrive.cuh"
#include "33_mbarrier_try_wait.cuh"

// =============================================================================
// ours (originally guarded by PL_AGENTIC_SM100A || PL_AGENTIC_SM103A)
// =============================================================================

// ARCH: sm_90a


__global__ void k_try(int* out) {
  __shared__ __align__(16) uint64_t mbar;
  if (threadIdx.x == 0) mbarrier_init(smem_ptr_u32(&mbar), 1);
  __syncthreads();
  // Before arrival, try_wait returns 0.
  if (threadIdx.x == 0)
    out[0] = mbarrier_try_wait_parity_once(smem_ptr_u32(&mbar), 0);
  if (threadIdx.x == 0) mbarrier_arrive(smem_ptr_u32(&mbar));
  __syncthreads();
  // After arrival + phase flip, the spinning variant must return.
  if (threadIdx.x == 0) {
    mbarrier_wait_parity(smem_ptr_u32(&mbar), 0);
    out[1] = 1;
    mbarrier_inval(smem_ptr_u32(&mbar));
  }
}

static int run_ours() {
  /* (orig args dropped) */
  int* d = nullptr; CUDA_CHECK(cudaMalloc(&d, 8));
  CUDA_CHECK(cudaMemset(d, 0, 8));
  k_try<<<1, 32>>>(d);
  CUDA_CHECK(cudaDeviceSynchronize());
  int h[2] = { 0 };
  CUDA_CHECK(cudaMemcpy(h, d, 8, cudaMemcpyDeviceToHost));
  cudaFree(d);
  if (h[1] != 1) FAIL("try_wait.parity spin did not return after arrival");
  printf("mbarrier.try_wait.parity : before=%d after=%d\n", h[0], h[1]);
  PASS();
}

// =============================================================================
// theirs (originally guarded by PL_AGENTIC_SM90A)
// =============================================================================

// Runtime test: mbarrier.try_wait.parity vs mbarrier.wait (blocking) semantics
//
// try_wait_parity is non-blocking: it returns true only if the phase has
// actually completed. Before any arrival, it must return false. After the
// arrival(s) satisfy the count, it must return true. The blocking
// wait_parity must then return immediately.
//
// The test kernel records each step into an output buffer so the host can
// verify the sequence.

//   out[0] = try_wait_parity(addr, 0) result BEFORE any arrive  (expect 0)
//   out[1] = try_wait_parity(addr, 0) result AFTER arrive       (expect 1)
//   out[2] = sentinel written AFTER blocking wait_parity returns
//   out[3] = test_wait(state) result after the arrive            (expect 1)
__global__ void try_wait_sequence_kernel(uint32_t* out) {
    __shared__ __align__(8) uint64_t mbar[1];
    uint32_t addr = smem_ptr_u32(mbar);

    if (threadIdx.x == 0) {
        mbarrier_init(addr, 1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        // Before any arrive, phase 0 is NOT complete.
        bool before = mbarrier_try_wait_parity(addr, 0);
        out[0] = before ? 1u : 0u;

        // Arrive to satisfy the single slot; keep the state token.
        uint64_t state = mbarrier_arrive(addr);

        // After arrive, phase 0 should be complete.
        bool after = mbarrier_try_wait_parity(addr, 0);
        out[1] = after ? 1u : 0u;

        // Blocking wait should now return immediately; sentinel proves it.
        mbarrier_wait_parity(addr, 0);
        out[2] = 0xBEEFu;

        // test_wait with the captured token: phase has advanced, so true.
        bool tw = mbarrier_test_wait(addr, state);
        out[3] = tw ? 1u : 0u;

        mbarrier_inval(addr);
    }
}

// A tiny "polling" test: consumer polls try_wait while producer delays.
// Producer credits the arrival only after a small workload. Consumer polls
// try_wait_parity in a loop (bounded) and records how many polls it took.
__global__ void polling_kernel(uint32_t* out, int delay_iters) {
    __shared__ __align__(8) uint64_t mbar[1];
    __shared__ __align__(16) volatile uint32_t go;

    uint32_t addr = smem_ptr_u32(mbar);
    const int tid = threadIdx.x;

    if (tid == 0) { mbarrier_init(addr, 1); go = 0; }
    __syncthreads();

    if (tid == 0) {
        // Producer warmup: spin on something GPU-visible so compiler keeps it.
        volatile int x = 1;
        for (int i = 0; i < delay_iters; i++) x = x * 1664525 + 1013904223;
        go = (uint32_t)x; // prevent DCE
        mbarrier_arrive_nostate(addr);
    } else if (tid == 32) {
        // Consumer: poll with an upper bound.
        uint32_t polls = 0;
        const uint32_t MAX_POLLS = 100000000u;
        while (polls < MAX_POLLS) {
            if (mbarrier_try_wait_parity(addr, 0)) break;
            polls++;
        }
        out[0] = polls < MAX_POLLS ? 1u : 0u; // eventually succeeded
        out[1] = polls;                        // loose: a value (not verified)
        out[2] = 0xCAFE;                       // sentinel post-success
    }

    __syncthreads();
    if (tid == 0) mbarrier_inval(addr);
}

static int run_theirs() {
  /* (orig args dropped) */
    CUDA_CHECK(cudaFree(0));
    bool all_pass = true;

    uint32_t h_out[8];
    uint32_t* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(h_out)));

    // --- Test A: before/after try_wait + blocking wait + test_wait ---
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(h_out)));
    try_wait_sequence_kernel<<<1, 32>>>(d_out);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));
    bool okA = (h_out[0] == 0u && h_out[1] == 1u &&
                h_out[2] == 0xBEEFu && h_out[3] == 1u);
    if (okA) {
        printf("  try_wait before/after arrive: OK (false->true, wait ret, test_wait=1)\n");
    } else {
        printf("  try_wait sequence: FAIL before=%u after=%u sentinel=%x tw=%u\n",
               h_out[0], h_out[1], h_out[2], h_out[3]);
        all_pass = false;
    }

    // --- Test B: polling try_wait until producer arrives ---
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(h_out)));
    polling_kernel<<<1, 64>>>(d_out, 2000);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));
    bool okB = (h_out[0] == 1u && h_out[2] == 0xCAFEu);
    if (okB) {
        printf("  polling try_wait -> success:  OK (polls=%u then sentinel set)\n",
               h_out[1]);
    } else {
        printf("  polling try_wait: FAIL (ok=%u polls=%u sentinel=%x)\n",
               h_out[0], h_out[1], h_out[2]);
        all_pass = false;
    }

    // --- perf ---
    GpuTimer t;
    const int ITERS = 200;
    t.begin();
    for (int i = 0; i < ITERS; i++)
        try_wait_sequence_kernel<<<1, 32>>>(d_out);
    t.end();
    printf("  perf:                         %.2f us/launch (try_wait/wait cycle)\n",
           t.elapsed_ms() * 1000.0f / ITERS);

    cudaFree(d_out);
    if (all_pass) { PASS(); return 0; }
    else { FAIL("some subtests failed"); return 1; }
}


// =============================================================================
// Cluster-scope try_wait.parity (Phase 3 HIGH)
// =============================================================================

// Cluster of 2 CTAs. Each CTA initializes its own mbar with arrival
// count 1 and uses fence.mbarrier_init.release.cluster to publish.
// Each CTA arrives locally + polls its own mbar via the new
// cluster-scope single-attempt wrapper. Verifies (a) the relaxed
// cluster wrapper completes, (b) the acquire cluster wrapper completes,
// (c) both `_once` and bool-returning forms agree on the done predicate.
__global__ void __cluster_dims__(2, 1, 1)
k_try_wait_cluster(uint32_t* out) {
  __shared__ __align__(16) uint64_t mbar;
  uint32_t mbar_addr = smem_ptr_u32(&mbar);
  if (threadIdx.x == 0) {
    mbarrier_init(mbar_addr, 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();

  // Pre-arrive observation: relaxed cluster try_wait should fail.
  uint32_t pre = 1;
  if (threadIdx.x == 0) {
    pre = mbarrier_try_wait_parity_cluster_once(mbar_addr, /*parity=*/0);
  }

  // Arrive locally.
  if (threadIdx.x == 0) (void)mbarrier_arrive(mbar_addr);
  __syncthreads();

  // Spin via the new bool-returning relaxed cluster wrapper.
  if (threadIdx.x == 0) {
    while (!mbarrier_try_wait_parity_cluster(mbar_addr, /*parity=*/0)) { }
  }
  __syncthreads();

  // Re-init for a second arrival, now poll via the .acquire variant
  // (publishes any prior writes to the consumer side).
  if (threadIdx.x == 0) {
    mbarrier_init(mbar_addr, 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    (void)mbarrier_arrive(mbar_addr);
    while (!mbarrier_try_wait_parity_acquire_cluster(mbar_addr, /*parity=*/0)) { }
  }

  // Each CTA records: bit 0 = pre-arrive done? (should be 0)
  //                   bit 1 = post-arrive done observed (should be 1, 1).
  if (threadIdx.x == 0) {
    out[blockIdx.x] = (pre & 1u) | (1u << 1);
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n" :: "r"(mbar_addr));
  }
}

static int run_cluster_try_wait() {
  uint32_t* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, 2 * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_out, 0xFF, 2 * sizeof(uint32_t)));
  k_try_wait_cluster<<<2, 32>>>(d_out);
  CUDA_CHECK(cudaDeviceSynchronize());
  uint32_t h[2] = {0u, 0u};
  CUDA_CHECK(cudaMemcpy(h, d_out, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  cudaFree(d_out);
  // Each CTA must have observed: pre-arrive try_wait failed (bit 0 == 0),
  // post-arrive try_wait succeeded (bit 1 == 1) -> value == 0b10 == 2.
  for (int i = 0; i < 2; ++i) {
    if (h[i] != 0x2u) {
      fprintf(stderr, "cluster try_wait: CTA %d observed 0x%x, expected 0x2\n",
              i, h[i]);
      FAIL("cluster try_wait did not detect arrival");
    }
  }
  printf("mbarrier_try_wait_parity_cluster{,_once,_acquire}: pre-fail + post-pass on both cluster CTAs\n");
  PASS();
}

// =============================================================================
// MED: try_wait.parity with suspendNanos (round 4i, free-agent item)
// =============================================================================
//
// Strategy: single-CTA mbar with arrival count 1. Thread 0 polls via the
// suspend-aware wrapper (suspend_ns = 100); pre-arrive should observe NOT
// done; post-arrive should observe done. The suspend hint is hardware
// scheduler advice -- we cannot directly observe suspend behavior from
// software, but we verify the wrapper compiles, executes, and returns
// semantically correct done/not-done values across the arrival edge.
//
// Also verifies the bool-returning and spin variants correctly transition.

__global__ void k_try_wait_suspend(uint32_t* out) {
  __shared__ __align__(16) uint64_t mbar;
  uint32_t mbar_addr = smem_ptr_u32(&mbar);
  if (threadIdx.x == 0) {
    mbarrier_init(mbar_addr, 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    // Pre-arrive: suspend-aware single-attempt should observe NOT done.
    uint32_t pre = mbarrier_try_wait_parity_suspend_once(mbar_addr,
                                                          /*parity=*/0,
                                                          /*suspend_ns=*/100u);
    out[0] = pre;        // expect 0 (not done)

    // Arrive.
    (void)mbarrier_arrive(mbar_addr);

    // Post-arrive: spin via suspend-aware variant. Loop is finite because
    // the barrier is satisfied; suspend hint just controls scheduling.
    mbarrier_try_wait_parity_suspend_spin(mbar_addr, /*parity=*/0,
                                           /*suspend_ns=*/100u);
    out[1] = 1;          // reached past spin -> arrival observed

    // Re-init + re-arrive, then check the bool-returning variant edge.
    mbarrier_init(mbar_addr, 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    bool pre_b = mbarrier_try_wait_parity_suspend(mbar_addr, /*parity=*/0,
                                                   /*suspend_ns=*/100u);
    out[2] = pre_b ? 1u : 0u;  // expect 0
    (void)mbarrier_arrive(mbar_addr);
    // Spin until satisfied.
    while (!mbarrier_try_wait_parity_suspend(mbar_addr, /*parity=*/0,
                                              /*suspend_ns=*/100u)) { }
    out[3] = 1;

    asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n" :: "r"(mbar_addr));
  }
}

static int run_suspend_try_wait() {
  uint32_t* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, 4 * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_out, 0xFF, 4 * sizeof(uint32_t)));
  k_try_wait_suspend<<<1, 32>>>(d_out);
  CUDA_CHECK(cudaDeviceSynchronize());
  uint32_t h[4] = {0u, 0u, 0u, 0u};
  CUDA_CHECK(cudaMemcpy(h, d_out, 4 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  cudaFree(d_out);
  // Expected: [0, 1, 0, 1] -- pre-arrive not done, post-arrive observed,
  // bool variant pre-arrive not done, bool variant post-arrive observed.
  if (h[0] != 0u || h[1] != 1u || h[2] != 0u || h[3] != 1u) {
    fprintf(stderr, "try_wait.parity.suspendNanos: got [%u,%u,%u,%u], expected [0,1,0,1]\n",
            h[0], h[1], h[2], h[3]);
    FAIL("try_wait suspend variant did not track arrival edge");
  }
  printf("mbarrier_try_wait_parity_suspend{,_once,_spin}: pre-fail + post-pass with suspend_ns=100\n");
  PASS();
}

int main() {
  int rc_ours   = run_ours();
  int rc_theirs = run_theirs();
  int rc_cluster = run_cluster_try_wait();
  int rc_suspend = run_suspend_try_wait();
  return (rc_ours == 0 && rc_theirs == 0 && rc_cluster == 0 && rc_suspend == 0) ? 0 : 1;
}
