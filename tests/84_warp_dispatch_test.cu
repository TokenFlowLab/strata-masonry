// ARCH: sm_90a
// 84_warp_dispatch_test.cu -- verify role assignments.
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
#include "../composites/84_warp_dispatch.cuh"
#include "84_warp_dispatch.cuh"

// =============================================================================
// ours (originally guarded by PL_AGENTIC_SM100A || PL_AGENTIC_SM103A)
// =============================================================================

// ARCH: sm_100a


__global__ void k(uint32_t* out) {
  int warp_id = threadIdx.x / 32;
  WarpRole r = warp_role_blackwell16(warp_id);
  if ((threadIdx.x & 31) == 0) out[warp_id] = (uint32_t)r;
}

static int run_ours() {
  /* (orig args dropped) */
  uint32_t* d = nullptr; CUDA_CHECK(cudaMalloc(&d, 16 * 4));
  CUDA_CHECK(cudaMemset(d, 0xFF, 16 * 4));
  k<<<1, 16 * 32>>>(d);
  CUDA_CHECK(cudaDeviceSynchronize());
  uint32_t h[16];
  CUDA_CHECK(cudaMemcpy(h, d, 16 * 4, cudaMemcpyDeviceToHost));
  cudaFree(d);
  for (int i = 0; i < 16; ++i)
    printf("  warp %2d -> role %u\n", i, h[i]);
  if (h[0] != (uint32_t)WarpRole::Loader)    FAIL("warp 0 not Loader");
  if (h[1] != (uint32_t)WarpRole::Scheduler) FAIL("warp 1 not Scheduler");
  if (h[2] != (uint32_t)WarpRole::MmaDriver) FAIL("warp 2 not MmaDriver");
  PASS();
}

// =============================================================================
// theirs (originally guarded by PL_AGENTIC_SM90A)
// =============================================================================

// Runtime test: warp dispatch -- warp_id, warpgroup_id, lane_id, Hopper role

// Launch 12 warps (384 threads). Each thread records:
//   out[tid*4 + 0] = get_warp_id()
//   out[tid*4 + 1] = get_warpgroup_id()
//   out[tid*4 + 2] = get_lane_id()
//   out[tid*4 + 3] = (int)hopper_role_dispatch_3wg()
// Do NOT call hopper_apply_regbudget_3wg -- it would mutate reg budget.
__global__ void warp_dispatch_kernel(int* out) {
    int tid = threadIdx.x;
    int w    = get_warp_id();
    int wg   = get_warpgroup_id();
    int lane = get_lane_id();
    WarpRole role = hopper_role_dispatch_3wg();
    out[tid * 4 + 0] = w;
    out[tid * 4 + 1] = wg;
    out[tid * 4 + 2] = lane;
    out[tid * 4 + 3] = (int)role;
}

__global__ void warp_dispatch_perf_kernel(int* sink) {
    int w = get_warp_id();
    int wg = get_warpgroup_id();
    int lane = get_lane_id();
    WarpRole role = hopper_role_dispatch_3wg();
    if (threadIdx.x == 0) sink[0] = w + wg + lane + (int)role;
}

static int run_theirs() {
  /* (orig args dropped) */
    CUDA_CHECK(cudaFree(0));
    bool all_pass = true;

    const int THREADS = 12 * 32;  // 384
    int *d_out; CUDA_CHECK(cudaMalloc(&d_out, THREADS * 4 * sizeof(int)));

    warp_dispatch_kernel<<<1, THREADS>>>(d_out);
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_out[THREADS * 4];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, THREADS * 4 * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify: for each thread t, warp_id=t/32, wg_id=warp_id/4, lane=t%32
    //         warps 0-3 -> Load, 4-7 and 8-11 -> Mma
    int bad_w = 0, bad_wg = 0, bad_lane = 0, bad_role = 0;
    for (int t = 0; t < THREADS; t++) {
        int w    = h_out[t * 4 + 0];
        int wg   = h_out[t * 4 + 1];
        int lane = h_out[t * 4 + 2];
        int role = h_out[t * 4 + 3];
        int exp_w    = t / 32;
        int exp_wg   = exp_w / 4;
        int exp_lane = t % 32;
        int exp_role = (exp_wg == 0) ? (int)WarpRole::Load
                     : (exp_wg == 1 || exp_wg == 2) ? (int)WarpRole::Mma
                     : (int)WarpRole::Idle;
        if (w    != exp_w)    bad_w++;
        if (wg   != exp_wg)   bad_wg++;
        if (lane != exp_lane) bad_lane++;
        if (role != exp_role) {
            if (bad_role < 3) printf("    role mismatch t=%d w=%d wg=%d role=%d exp=%d\n",
                                      t, w, wg, role, exp_role);
            bad_role++;
        }
    }

    if (bad_w == 0)    printf("  get_warp_id: OK (%d threads)\n", THREADS);
    else               { printf("  get_warp_id: FAIL (%d)\n", bad_w); all_pass = false; }

    if (bad_wg == 0)   printf("  get_warpgroup_id: OK\n");
    else               { printf("  get_warpgroup_id: FAIL (%d)\n", bad_wg); all_pass = false; }

    if (bad_lane == 0) printf("  get_lane_id: OK\n");
    else               { printf("  get_lane_id: FAIL (%d)\n", bad_lane); all_pass = false; }

    if (bad_role == 0) printf("  hopper_role_dispatch_3wg: OK (4 Load + 4+4 Mma)\n");
    else               { printf("  hopper_role_dispatch_3wg: FAIL (%d)\n", bad_role); all_pass = false; }

    // Spot checks per per-task spec:
    //   warp 0 lane 0 -> tid 0
    //   warp 4 lane 0 -> tid 128
    //   warp 8 lane 0 -> tid 256
    auto role_of = [&](int tid) { return (WarpRole)h_out[tid * 4 + 3]; };
    bool spot = (role_of(0)   == WarpRole::Load)
             && (role_of(127) == WarpRole::Load)
             && (role_of(128) == WarpRole::Mma)
             && (role_of(255) == WarpRole::Mma)
             && (role_of(256) == WarpRole::Mma)
             && (role_of(383) == WarpRole::Mma);
    if (spot) printf("  WG boundaries (0-127 Load, 128-255 Mma, 256-383 Mma): OK\n");
    else      { printf("  WG boundaries: FAIL\n"); all_pass = false; }

    // Perf
    int *d_sink; CUDA_CHECK(cudaMalloc(&d_sink, 4));
    GpuTimer t;
    t.begin();
    for (int i = 0; i < 1000; i++) warp_dispatch_perf_kernel<<<1, THREADS>>>(d_sink);
    t.end();
    printf("  perf: %.2f us/launch (1000 launches, %d threads each)\n",
           t.elapsed_ms() * 1000.0f / 1000, THREADS);

    cudaFree(d_out); cudaFree(d_sink);
    if (all_pass) { PASS(); return 0; }
    else          { FAIL("some subtests failed"); return 1; }
}


int main() {
  int rc_ours   = run_ours();
  int rc_theirs = run_theirs();
  return (rc_ours == 0 && rc_theirs == 0) ? 0 : 1;
}
