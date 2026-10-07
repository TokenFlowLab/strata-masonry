// ARCH: sm_90a
// 37_bar_sync_test.cu -- two warp groups sync at separate named barriers,
// then exchange via a counter they both increment.
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
#include "../primitives/37_bar_sync.cuh"
#include "37_bar_sync.cuh"

// =============================================================================
// ours (originally guarded by PL_AGENTIC_SM100A || PL_AGENTIC_SM103A)
// =============================================================================

// ARCH: sm_90a


__global__ void k_bar(int* out) {
  __shared__ int cnt;
  if (threadIdx.x == 0) cnt = 0;
  __syncthreads();

  // Warp 0 + warp 1 (64 threads) sync at barrier 1, each contributes 1.
  if (threadIdx.x < 64) {
    atomicAdd(&cnt, 1);
    bar_sync<1>(64u);
    if (threadIdx.x == 0) out[0] = cnt;
  }
  // Warp 2 + warp 3 (64 threads) sync at barrier 2, each contributes 2.
  if (threadIdx.x >= 64) {
    atomicAdd(&cnt, 2);
    bar_sync<2>(64u);
    if (threadIdx.x == 64) out[1] = cnt;
  }
}

static int run_ours() {
  /* (orig args dropped) */
  int* d = nullptr; CUDA_CHECK(cudaMalloc(&d, 8));
  CUDA_CHECK(cudaMemset(d, 0, 8));
  k_bar<<<1, 128>>>(d);
  CUDA_CHECK(cudaDeviceSynchronize());
  int h[2] = {0, 0};
  CUDA_CHECK(cudaMemcpy(h, d, 8, cudaMemcpyDeviceToHost));
  cudaFree(d);
  printf("bar.sync 64 threads -- first-group cnt=%d, second-group cnt=%d\n",
         h[0], h[1]);
  if (h[0] < 64) FAIL("first bar.sync group did not aggregate");
  PASS();
}

// =============================================================================
// theirs (originally guarded by PL_AGENTIC_SM90A)
// =============================================================================

// Runtime test: bar.sync (named barriers) + barrier.cta.red.popc
//
// Scenario 1 (half-CTA named barrier):
//   128 threads per block. The first 64 threads ("producers", tid < 64)
//   write SMEM[tid] = pattern, then sync on named barrier 1 with 64 threads.
//   After the barrier, the same 64 threads read SMEM[64 - 1 - tid] (i.e.
//   swapped within the group) and write to GMEM output[tid]. The second
//   half of the CTA must NOT touch barrier 1. Full-CTA sync on barrier 2
//   at the end serves as a safety net.
//
// Scenario 2 (reduction barrier):
//   barrier.cta.red.popc.aligned counts how many threads of a named
//   barrier's cohort have a predicate true. Test: 128 threads, pred =
//   (tid % 3 == 0) -> expected count = 43 (tid in {0,3,6,...,126}).

__global__ void half_cta_named_barrier_kernel(uint32_t* out) {
    __shared__ uint32_t s[64];

    const int tid = threadIdx.x;
    if (tid < 64) {
        s[tid] = (uint32_t)(tid * 10 + 1);
        // 64-thread named barrier -- only the first 64 threads participate.
        bar_sync<1>(64);
        // After sync, read the swapped slot.
        uint32_t val = s[63 - tid];
        out[tid] = val;
    } else {
        // Second half just waits at the full-CTA barrier below.
        out[tid] = 0u;
    }

    // Full-CTA sync: all 128 threads. Uses named barrier 2.
    barrier_cta_sync_aligned<2>(128);

    // After full sync: every thread writes a confirmation sentinel above 128.
    if (tid >= 64) out[tid] = 0x8000u | (uint32_t)tid;
}

__global__ void red_popc_kernel(uint32_t* out) {
    const int tid = threadIdx.x;
    bool pred = (tid % 3 == 0);
    // All 128 threads participate on named barrier 3.
    uint32_t count = barrier_cta_red_popc<3>(128, pred);
    // Every thread sees the same reduced count -- write it once from tid 0.
    if (tid == 0) out[0] = count;

    // Also verify a pred-all-true case to cross-check.
    uint32_t count_all = barrier_cta_red_popc<4>(128, true);
    if (tid == 0) out[1] = count_all;

    // And pred-all-false.
    uint32_t count_none = barrier_cta_red_popc<5>(128, false);
    if (tid == 0) out[2] = count_none;

    bar_sync<6>(128);
}

static int run_theirs() {
  /* (orig args dropped) */
    CUDA_CHECK(cudaFree(0));
    bool all_pass = true;

    // --- Scenario 1: half-CTA named barrier ---
    const int THREADS = 128;
    uint32_t h_out[THREADS];
    uint32_t* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(h_out)));
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(h_out)));
    half_cta_named_barrier_kernel<<<1, THREADS>>>(d_out);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));

    bool okA = true;
    int mism = 0;
    for (int tid = 0; tid < 64; tid++) {
        uint32_t expect = (uint32_t)((63 - tid) * 10 + 1);
        if (h_out[tid] != expect) {
            if (mism < 4) printf("    half-CTA mismatch [%d]: got %u expected %u\n",
                                 tid, h_out[tid], expect);
            mism++; okA = false;
        }
    }
    for (int tid = 64; tid < 128; tid++) {
        uint32_t expect = 0x8000u | (uint32_t)tid;
        if (h_out[tid] != expect) {
            if (mism < 4) printf("    post-full-sync mismatch [%d]: got %u expected %u\n",
                                 tid, h_out[tid], expect);
            mism++; okA = false;
        }
    }
    if (okA) printf("  half-CTA named barrier:       OK (swap across bar<1>(64) + full sync bar<2>)\n");
    else { printf("  half-CTA named barrier: FAIL (%d mismatches)\n", mism); all_pass = false; }

    // --- Scenario 2: barrier.cta.red.popc ---
    uint32_t h_red[8];
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(h_out)));
    red_popc_kernel<<<1, THREADS>>>(d_out);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_red, d_out, sizeof(h_red), cudaMemcpyDeviceToHost));

    // Expect: tid in {0,3,6,...,126} => ceil(128/3) = 43
    int expect_mod3 = 0;
    for (int i = 0; i < 128; i++) if (i % 3 == 0) expect_mod3++;
    bool okB = (h_red[0] == (uint32_t)expect_mod3 &&
                h_red[1] == 128u &&
                h_red[2] == 0u);
    if (okB) {
        printf("  red.popc (mod3, all, none):   OK (%u, %u, %u)\n",
               h_red[0], h_red[1], h_red[2]);
    } else {
        printf("  red.popc: FAIL (got %u %u %u, expected %d 128 0)\n",
               h_red[0], h_red[1], h_red[2], expect_mod3);
        all_pass = false;
    }

    // --- perf ---
    GpuTimer t;
    const int ITERS = 200;
    t.begin();
    for (int i = 0; i < ITERS; i++)
        half_cta_named_barrier_kernel<<<1, THREADS>>>(d_out);
    t.end();
    printf("  perf:                         %.2f us/launch (named barrier CTA)\n",
           t.elapsed_ms() * 1000.0f / ITERS);

    cudaFree(d_out);
    if (all_pass) { PASS(); return 0; }
    else { FAIL("some subtests failed"); return 1; }
}


// Round 4h: barrier.cta.red.{and,or}.aligned + barrier.cta.arrive.aligned.
// 64 threads in 1 CTA (= 2 warps). Verifies:
//   - barrier_cta_red_and  : returns true iff every thread's pred is true
//   - barrier_cta_red_or   : returns true iff at least one thread's pred is true
//   - barrier_cta_arrive_aligned + barrier_cta_sync_aligned pair (non-blocking
//     producer arrive followed by consumer sync, on a separate barrier id).
__global__ void k_red_and_or_arrive(uint32_t* out_and_mixed,
                                    uint32_t* out_and_all1,
                                    uint32_t* out_or_mixed,
                                    uint32_t* out_or_all0,
                                    uint32_t* out_arrive_ok) {
  uint32_t tid = threadIdx.x;

  // Mixed predicate: low warp -> 1, high warp -> 0.
  bool mixed_pred = (tid < 32);
  bool a_mixed = barrier_cta_red_and<1>(/*threads=*/64u, mixed_pred);
  if (tid == 0) *out_and_mixed = (uint32_t)a_mixed;
  bool o_mixed = barrier_cta_red_or<2>(/*threads=*/64u, mixed_pred);
  if (tid == 0) *out_or_mixed = (uint32_t)o_mixed;

  // Saturated cases.
  bool a_all1 = barrier_cta_red_and<3>(/*threads=*/64u, /*pred=*/true);
  if (tid == 0) *out_and_all1 = (uint32_t)a_all1;
  bool o_all0 = barrier_cta_red_or<4>(/*threads=*/64u, /*pred=*/false);
  if (tid == 0) *out_or_all0 = (uint32_t)o_all0;

  // arrive.aligned (warp 0, non-blocking) + sync.aligned (warp 1, blocks).
  // After sync returns, the pair has met -> warp 1 flags ok.
  if (tid < 32) {
    barrier_cta_arrive_aligned<5>(/*threads=*/64u);
  } else {
    barrier_cta_sync_aligned<5>(/*threads=*/64u);
    if (tid == 32) *out_arrive_ok = 1u;
  }
}

static int run_red_and_or() {
  uint32_t *d_a_mixed=nullptr, *d_a_all1=nullptr, *d_o_mixed=nullptr,
           *d_o_all0=nullptr, *d_arrive=nullptr;
  CUDA_CHECK(cudaMalloc(&d_a_mixed, 4));
  CUDA_CHECK(cudaMalloc(&d_a_all1,  4));
  CUDA_CHECK(cudaMalloc(&d_o_mixed, 4));
  CUDA_CHECK(cudaMalloc(&d_o_all0,  4));
  CUDA_CHECK(cudaMalloc(&d_arrive,  4));
  CUDA_CHECK(cudaMemset(d_a_mixed, 0xFF, 4));
  CUDA_CHECK(cudaMemset(d_a_all1,  0xFF, 4));
  CUDA_CHECK(cudaMemset(d_o_mixed, 0xFF, 4));
  CUDA_CHECK(cudaMemset(d_o_all0,  0xFF, 4));
  CUDA_CHECK(cudaMemset(d_arrive,  0,    4));

  k_red_and_or_arrive<<<1, 64>>>(d_a_mixed, d_a_all1, d_o_mixed, d_o_all0, d_arrive);
  CUDA_CHECK(cudaDeviceSynchronize());

  uint32_t am, a1, om, o0, ar;
  CUDA_CHECK(cudaMemcpy(&am, d_a_mixed, 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&a1, d_a_all1,  4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&om, d_o_mixed, 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&o0, d_o_all0,  4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&ar, d_arrive,  4, cudaMemcpyDeviceToHost));
  cudaFree(d_a_mixed); cudaFree(d_a_all1); cudaFree(d_o_mixed);
  cudaFree(d_o_all0);  cudaFree(d_arrive);

  if (am != 0u) { fprintf(stderr, "red.and(mixed)=%u exp 0\n", am); FAIL("red.and mixed"); }
  if (a1 != 1u) { fprintf(stderr, "red.and(all1)=%u exp 1\n", a1); FAIL("red.and all1"); }
  if (om != 1u) { fprintf(stderr, "red.or(mixed)=%u exp 1\n", om);  FAIL("red.or mixed"); }
  if (o0 != 0u) { fprintf(stderr, "red.or(all0)=%u exp 0\n", o0);   FAIL("red.or all0"); }
  if (ar != 1u) { fprintf(stderr, "arrive_aligned/sync_aligned pair stuck\n"); FAIL("arrive.aligned"); }
  printf("barrier.cta.red.{and,or}.aligned + barrier.cta.arrive.aligned: 5/5 OK\n");
  PASS();
}

int main() {
  int rc_ours   = run_ours();
  int rc_theirs = run_theirs();
  int rc_red    = run_red_and_or();
  return (rc_ours == 0 && rc_theirs == 0 && rc_red == 0) ? 0 : 1;
}
