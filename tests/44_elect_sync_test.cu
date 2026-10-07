// ARCH: sm_90a
// 44_elect_sync_test.cu -- exactly one lane per warp returns true.
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
#include "../primitives/44_elect_sync.cuh"
#include "44_elect_sync.cuh"

// =============================================================================
// ours (originally guarded by PL_AGENTIC_SM100A || PL_AGENTIC_SM103A)
// =============================================================================

// ARCH: sm_90a


__global__ void k_elect(uint32_t* out) {
  bool picked = elect_one_sync();
  if (picked) atomicAdd(out, 1u);
}

static int run_ours() {
  /* (orig args dropped) */
  uint32_t* d = nullptr; CUDA_CHECK(cudaMalloc(&d, 4));
  CUDA_CHECK(cudaMemset(d, 0, 4));
  k_elect<<<1, 128>>>(d); // 4 warps
  CUDA_CHECK(cudaDeviceSynchronize());
  uint32_t h = 0; CUDA_CHECK(cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost));
  cudaFree(d);
  printf("elect.sync : %u warps elected one\n", h);
  if (h != 4) FAIL("elect.sync did not pick one lane per warp");
  PASS();
}

// =============================================================================
// theirs (originally guarded by PL_AGENTIC_SM90A)
// =============================================================================

// Runtime test: elect.sync -- verify exactly one lane per warp elected + timing

__global__ void elect_sync_kernel(uint32_t* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t elected = elect_one_sync();
    if (tid < n) out[tid] = elected;
}

static int run_theirs() {
  /* (orig args dropped) */
    CUDA_CHECK(cudaFree(0));
    const int THREADS = 128;         // 4 warps
    const int WARPS = THREADS / 32;  // 4
    bool all_pass = true;

    uint32_t* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, THREADS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_out, 0xFF, THREADS * sizeof(uint32_t)));

    elect_sync_kernel<<<1, THREADS>>>(d_out, THREADS);
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t h_out[THREADS];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, THREADS * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Count total elected lanes and per-warp electeds.
    int total_elected = 0;
    int per_warp[WARPS] = {0};
    int bad_values = 0;
    for (int i = 0; i < THREADS; i++) {
        if (h_out[i] != 0 && h_out[i] != 1) bad_values++;
        if (h_out[i] == 1) {
            total_elected++;
            per_warp[i / 32]++;
        }
    }

    if (bad_values != 0) {
        printf("  elect.sync: FAIL (found %d out-of-range values)\n", bad_values);
        all_pass = false;
    } else if (total_elected != WARPS) {
        printf("  elect.sync: FAIL (total elected = %d, expected %d)\n", total_elected, WARPS);
        all_pass = false;
    } else {
        bool per_warp_ok = true;
        for (int w = 0; w < WARPS; w++) if (per_warp[w] != 1) per_warp_ok = false;
        if (!per_warp_ok) {
            printf("  elect.sync: FAIL (per-warp counts ");
            for (int w = 0; w < WARPS; w++) printf("%d ", per_warp[w]);
            printf(", expected 1 per warp)\n");
            all_pass = false;
        } else {
            printf("  elect.sync: OK (%d warps, exactly 1 elected each)\n", WARPS);
        }
    }

    // Dump which lane got elected per warp -- informative, not a hard check
    for (int w = 0; w < WARPS; w++) {
        int which = -1;
        for (int l = 0; l < 32; l++) if (h_out[w * 32 + l] == 1) { which = l; break; }
        printf("    warp %d: elected lane %d\n", w, which);
    }

    // --- Timing ---
    GpuTimer t;
    t.begin();
    for (int i = 0; i < 100; i++) elect_sync_kernel<<<1, THREADS>>>(d_out, THREADS);
    t.end();
    printf("  perf: %.2f us/launch\n", t.elapsed_ms() * 1000.0f / 100);

    cudaFree(d_out);
    if (all_pass) { PASS(); return 0; }
    else { FAIL("some subtests failed"); return 1; }
}


int main() {
  int rc_ours   = run_ours();
  int rc_theirs = run_theirs();
  return (rc_ours == 0 && rc_theirs == 0) ? 0 : 1;
}
