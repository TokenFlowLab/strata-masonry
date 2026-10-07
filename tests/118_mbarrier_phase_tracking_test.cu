// ARCH: sm_90a
// 118_mbarrier_phase_tracking_test.cu -- host: verify parity flip + stage wraparound.
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
#include "../src/composites/118_mbarrier_phase_tracking.cuh"
#include "118_mbarrier_phase_tracking.cuh"

// =============================================================================
// ours (originally guarded by PL_AGENTIC_SM100A || PL_AGENTIC_SM103A)
// =============================================================================

// ARCH: sm_90a


__global__ void k(uint32_t* out) {
  MbarrierPhaseTracker<4> t;
  t.init();
  // Simulate 10 advances; record (stage, phase) at each step.
  for (int i = 0; i < 10; ++i) {
    out[i * 2    ] = t.stage();
    out[i * 2 + 1] = t.current_phase();
    t.advance();
  }
}

static int run_ours() {
  /* (orig args dropped) */
  uint32_t* d = nullptr; CUDA_CHECK(cudaMalloc(&d, 20 * 4));
  k<<<1, 1>>>(d);
  CUDA_CHECK(cudaDeviceSynchronize());
  uint32_t h[20];
  CUDA_CHECK(cudaMemcpy(h, d, 20 * 4, cudaMemcpyDeviceToHost));
  cudaFree(d);
  // Stages: 0,1,2,3,0,1,2,3,0,1
  // Phases: 0,0,0,0,1,1,1,1,0,0 (flip after wrap back to stage 0)
  const uint32_t want[20] = {
    0,0, 1,0, 2,0, 3,0, 0,1, 1,1, 2,1, 3,1, 0,0, 1,0 };
  int fails = 0;
  for (int i = 0; i < 20; ++i) if (h[i] != want[i]) ++fails;
  printf("phase tracker : fails = %d / 20\n", fails);
  if (fails) FAIL("phase/stage sequence wrong");
  PASS();
}

// =============================================================================
// theirs (originally guarded by PL_AGENTIC_SM90A)
// =============================================================================

// Runtime test: PhaseTracker<4> state machine + idx2phase / idx2stage

// Instantiate a PhaseTracker<4>, call advance() STEPS-1 times, record
// (stage, phase) before AND after each advance. STEPS=10 so we see:
//   step 0: (0, 0)    initial
//   step 1: (1, 0)
//   step 2: (2, 0)
//   step 3: (3, 0)
//   step 4: (0, 1)    wrap + phase flip
//   step 5: (1, 1)
//   step 6: (2, 1)
//   step 7: (3, 1)
//   step 8: (0, 0)    wrap again
//   step 9: (1, 0)
__global__ void phase_tracking_kernel(int* out_full, int* out_empty) {
    if (threadIdx.x != 0) return;
    PhaseTracker<4> pt;
    EmptyPhaseTracker<4> ept;
    out_full[0]  = pt.get_stage();
    out_full[1]  = (int)pt.get_phase();
    out_empty[0] = ept.get_stage();
    out_empty[1] = (int)ept.get_phase();
    for (int i = 1; i < 10; i++) {
        pt.advance();
        ept.advance();
        out_full[i * 2 + 0]  = pt.get_stage();
        out_full[i * 2 + 1]  = (int)pt.get_phase();
        out_empty[i * 2 + 0] = ept.get_stage();
        out_empty[i * 2 + 1] = (int)ept.get_phase();
    }
}

// idx2phase / idx2stage are __device__ functions that take integers.
// Drive 16 iterations through them and copy back to verify on host.
__global__ void idx2phase_kernel(int num_stages, int* out) {
    if (threadIdx.x != 0) return;
    for (int i = 0; i < 16; i++) {
        out[i * 2 + 0] = idx2stage(i, num_stages);
        out[i * 2 + 1] = (int)idx2phase(i, num_stages);
    }
}

__global__ void phase_tracking_perf_kernel(int iters, int* sink) {
    if (threadIdx.x != 0) return;
    PhaseTracker<4> pt;
    for (int i = 0; i < iters; i++) pt.advance();
    sink[0] = pt.get_stage() + (int)pt.get_phase();
}

static int run_theirs() {
  /* (orig args dropped) */
    CUDA_CHECK(cudaFree(0));
    bool all_pass = true;

    // PhaseTracker<4> expected sequence
    int exp_full[10][2] = {
        {0, 0},  // step 0
        {1, 0},  // step 1
        {2, 0},  // step 2
        {3, 0},  // step 3
        {0, 1},  // step 4  wrap
        {1, 1},
        {2, 1},
        {3, 1},
        {0, 0},  // step 8  wrap again
        {1, 0},
    };

    int *d_full, *d_empty;
    CUDA_CHECK(cudaMalloc(&d_full,  10 * 2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_empty, 10 * 2 * sizeof(int)));

    phase_tracking_kernel<<<1, 32>>>(d_full, d_empty);
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_full[10 * 2], h_empty[10 * 2];
    CUDA_CHECK(cudaMemcpy(h_full,  d_full,  10 * 2 * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_empty, d_empty, 10 * 2 * sizeof(int), cudaMemcpyDeviceToHost));

    int bad_full = 0;
    for (int i = 0; i < 10; i++) {
        int s = h_full[i * 2 + 0], p = h_full[i * 2 + 1];
        if (s != exp_full[i][0] || p != exp_full[i][1]) {
            if (bad_full < 3)
                printf("    PhaseTracker step %d: got (%d,%d) exp (%d,%d)\n",
                       i, s, p, exp_full[i][0], exp_full[i][1]);
            bad_full++;
        }
    }
    if (bad_full == 0) printf("  PhaseTracker<4> advance (10 steps): OK\n");
    else               { printf("  PhaseTracker<4>: FAIL (%d)\n", bad_full); all_pass = false; }

    // EmptyPhaseTracker<4>: same stage sequence, but phase starts at 1.
    int exp_empty[10][2] = {
        {0, 1}, {1, 1}, {2, 1}, {3, 1},
        {0, 0}, {1, 0}, {2, 0}, {3, 0},
        {0, 1}, {1, 1},
    };
    int bad_empty = 0;
    for (int i = 0; i < 10; i++) {
        int s = h_empty[i * 2 + 0], p = h_empty[i * 2 + 1];
        if (s != exp_empty[i][0] || p != exp_empty[i][1]) {
            if (bad_empty < 3)
                printf("    EmptyPhaseTracker step %d: got (%d,%d) exp (%d,%d)\n",
                       i, s, p, exp_empty[i][0], exp_empty[i][1]);
            bad_empty++;
        }
    }
    if (bad_empty == 0) printf("  EmptyPhaseTracker<4> advance (phase inverted): OK\n");
    else                 { printf("  EmptyPhaseTracker<4>: FAIL (%d)\n", bad_empty); all_pass = false; }

    // idx2phase / idx2stage direct calls (num_stages=4)
    int *d_iter; CUDA_CHECK(cudaMalloc(&d_iter, 16 * 2 * sizeof(int)));
    idx2phase_kernel<<<1, 32>>>(4, d_iter);
    CUDA_CHECK(cudaDeviceSynchronize());
    int h_iter[16 * 2];
    CUDA_CHECK(cudaMemcpy(h_iter, d_iter, 16 * 2 * sizeof(int), cudaMemcpyDeviceToHost));

    int bad_iter = 0;
    for (int i = 0; i < 16; i++) {
        int s_exp = i % 4;
        int p_exp = (i / 4) & 1;
        if (h_iter[i * 2 + 0] != s_exp || h_iter[i * 2 + 1] != p_exp) {
            if (bad_iter < 3)
                printf("    iter=%d: got s=%d p=%d exp s=%d p=%d\n",
                       i, h_iter[i * 2 + 0], h_iter[i * 2 + 1], s_exp, p_exp);
            bad_iter++;
        }
    }
    if (bad_iter == 0) printf("  idx2phase / idx2stage (16 iters): OK\n");
    else               { printf("  idx2phase: FAIL (%d)\n", bad_iter); all_pass = false; }

    // Correspondence: idx2phase(i, S) should equal PhaseTracker<S> state
    // at step i reached via i advances from ctor (which itself is step 0).
    // i.e. after i advances, stage==i%S and phase==(i/S)&1.
    // Our captured h_full uses `before-advance at step 0, after-advance at step i>=1`,
    // so at step i we've called advance() i times. Expected matches.
    int corr_bad = 0;
    for (int i = 0; i < 10; i++) {
        int s_full = h_full[i * 2 + 0], p_full = h_full[i * 2 + 1];
        int s_exp = i % 4, p_exp = (i / 4) & 1;
        if (s_full != s_exp || p_full != p_exp) corr_bad++;
    }
    if (corr_bad == 0) printf("  PhaseTracker matches idx2phase formula: OK\n");
    else                { printf("  correspondence: FAIL (%d)\n", corr_bad); all_pass = false; }

    // Perf
    int *d_sink; CUDA_CHECK(cudaMalloc(&d_sink, 4));
    GpuTimer t;
    t.begin();
    for (int i = 0; i < 1000; i++) phase_tracking_perf_kernel<<<1, 32>>>(100, d_sink);
    t.end();
    printf("  perf: %.2f us/launch (1000 launches, 100 advance/launch)\n",
           t.elapsed_ms() * 1000.0f / 1000);

    cudaFree(d_full); cudaFree(d_empty); cudaFree(d_iter); cudaFree(d_sink);
    if (all_pass) { PASS(); return 0; }
    else          { FAIL("some subtests failed"); return 1; }
}


int main() {
  int rc_ours   = run_ours();
  int rc_theirs = run_theirs();
  return (rc_ours == 0 && rc_theirs == 0) ? 0 : 1;
}
