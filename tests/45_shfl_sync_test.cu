// ARCH: sm_90a
// 45_shfl_sync_test.cu -- verify bfly XOR shuffle between pairs.
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
#include "../src/primitives/45_shfl_sync.cuh"
#include "45_shfl_sync.cuh"

// =============================================================================
// ours (originally guarded by PL_AGENTIC_SM100A || PL_AGENTIC_SM103A)
// =============================================================================

// ARCH: sm_90a


__global__ void k_sh(uint32_t* out) {
  uint32_t v = threadIdx.x;
  uint32_t s = shfl_sync_bfly(v, 1);
  out[threadIdx.x] = s;
}

static int run_ours() {
  /* (orig args dropped) */
  uint32_t* d = nullptr; CUDA_CHECK(cudaMalloc(&d, 32 * 4));
  k_sh<<<1, 32>>>(d);
  CUDA_CHECK(cudaDeviceSynchronize());
  uint32_t h[32];
  CUDA_CHECK(cudaMemcpy(h, d, 32 * 4, cudaMemcpyDeviceToHost));
  cudaFree(d);
  int fails = 0;
  for (int i = 0; i < 32; ++i)
    if (h[i] != (uint32_t)(i ^ 1)) ++fails;
  printf("shfl.sync.bfly(1) : fails = %d / 32\n", fails);
  if (fails) FAIL("bfly shuffle wrong");
  PASS();
}

// =============================================================================
// theirs (originally guarded by PL_AGENTIC_SM90A)
// =============================================================================

// Runtime test: shfl.sync -- verify broadcast, butterfly, scan, and reductions

// Per-lane output layout (9 u32/f32 slots per lane):
//   [0] shfl_idx(val, 0)              -- expect 0 on every lane
//   [1] shfl_idx(val, 7)              -- expect 7 on every lane
//   [2] shfl_bfly(val, 1)             -- expect lane ^ 1
//   [3] shfl_bfly(val, 16)            -- expect lane ^ 16
//   [4] shfl_up(val, 1)               -- expect (lane==0)? 0 : lane-1
//   [5] shfl_down(val, 1)             -- expect (lane==31)? 31 : lane+1
//   [6] warp_reduce_sum_f32(lane)     -- expect 496 on every lane
//   [7] warp_reduce_max_f32(lane)     -- expect 31 on every lane
//   [8] warp_reduce_sum_f32(1.0f)     -- expect 32 on every lane
__global__ void shfl_kernel(uint32_t* u_out, float* f_out) {
    int lane = threadIdx.x;
    uint32_t val = (uint32_t)lane;

    u_out[lane * 6 + 0] = shfl_sync_idx(val, 0);
    u_out[lane * 6 + 1] = shfl_sync_idx(val, 7);
    u_out[lane * 6 + 2] = shfl_sync_bfly(val, 1);
    u_out[lane * 6 + 3] = shfl_sync_bfly(val, 16);
    u_out[lane * 6 + 4] = shfl_sync_up(val, 1);
    u_out[lane * 6 + 5] = shfl_sync_down(val, 1);

    float fv = (float)lane;
    f_out[lane * 3 + 0] = warp_reduce_sum_f32(fv);
    f_out[lane * 3 + 1] = warp_reduce_max_f32(fv);
    f_out[lane * 3 + 2] = warp_reduce_sum_f32(1.0f);
}

// Perf kernel: do many sum reductions back-to-back
__global__ void shfl_perf_kernel(float* out, int iters) {
    float v = (float)threadIdx.x;
    for (int i = 0; i < iters; i++) {
        v = warp_reduce_sum_f32(v) + 1.0f;
    }
    if (threadIdx.x == 0) out[blockIdx.x] = v;
}

static int run_theirs() {
  /* (orig args dropped) */
    CUDA_CHECK(cudaFree(0));
    bool all_pass = true;

    uint32_t *d_u; CUDA_CHECK(cudaMalloc(&d_u, 32 * 6 * sizeof(uint32_t)));
    float    *d_f; CUDA_CHECK(cudaMalloc(&d_f, 32 * 3 * sizeof(float)));
    shfl_kernel<<<1, 32>>>(d_u, d_f);
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t h_u[32 * 6]; float h_f[32 * 3];
    CUDA_CHECK(cudaMemcpy(h_u, d_u, sizeof(h_u), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_f, d_f, sizeof(h_f), cudaMemcpyDeviceToHost));

    // shfl_idx broadcasts
    bool ok = true;
    for (int l = 0; l < 32; l++) {
        if (h_u[l * 6 + 0] != 0)  { ok = false; break; }
        if (h_u[l * 6 + 1] != 7)  { ok = false; break; }
    }
    if (ok) printf("  shfl.idx (broadcast): OK\n");
    else    { printf("  shfl.idx: FAIL\n"); all_pass = false; }

    // shfl_bfly
    ok = true;
    for (int l = 0; l < 32; l++) {
        if (h_u[l * 6 + 2] != (uint32_t)(l ^ 1))  { ok = false; break; }
        if (h_u[l * 6 + 3] != (uint32_t)(l ^ 16)) { ok = false; break; }
    }
    if (ok) printf("  shfl.bfly (xor-delta): OK\n");
    else    { printf("  shfl.bfly: FAIL\n"); all_pass = false; }

    // shfl_up: clamp at 0 means lane 0 reads its own value
    ok = true;
    for (int l = 0; l < 32; l++) {
        uint32_t expected = (l == 0) ? 0u : (uint32_t)(l - 1);
        if (h_u[l * 6 + 4] != expected) { ok = false; break; }
    }
    if (ok) printf("  shfl.up: OK\n");
    else    { printf("  shfl.up: FAIL\n"); all_pass = false; }

    // shfl_down: clamp at 31 -- lane 31 reads its own value
    ok = true;
    for (int l = 0; l < 32; l++) {
        uint32_t expected = (l == 31) ? 31u : (uint32_t)(l + 1);
        if (h_u[l * 6 + 5] != expected) { ok = false; break; }
    }
    if (ok) printf("  shfl.down: OK\n");
    else    { printf("  shfl.down: FAIL\n"); all_pass = false; }

    // warp_reduce_sum_f32(lane) = 0+1+...+31 = 496 on every lane
    ok = true;
    for (int l = 0; l < 32; l++) {
        if (fabsf(h_f[l * 3 + 0] - 496.0f) > 1e-3f) { ok = false; break; }
    }
    if (ok) printf("  warp_reduce_sum_f32(lane): OK (496)\n");
    else    { printf("  warp_reduce_sum_f32: FAIL (lane0=%.1f)\n", h_f[0]); all_pass = false; }

    // warp_reduce_max_f32(lane) = 31 on every lane
    ok = true;
    for (int l = 0; l < 32; l++) {
        if (h_f[l * 3 + 1] != 31.0f) { ok = false; break; }
    }
    if (ok) printf("  warp_reduce_max_f32(lane): OK (31)\n");
    else    { printf("  warp_reduce_max_f32: FAIL\n"); all_pass = false; }

    // warp_reduce_sum_f32(1.0f) = 32 on every lane
    ok = true;
    for (int l = 0; l < 32; l++) {
        if (h_f[l * 3 + 2] != 32.0f) { ok = false; break; }
    }
    if (ok) printf("  warp_reduce_sum_f32(1.0): OK (32)\n");
    else    { printf("  warp_reduce_sum_f32(1.0): FAIL\n"); all_pass = false; }

    // Perf: 1000 launches of a kernel doing 100 warp reductions each
    float *d_perf; CUDA_CHECK(cudaMalloc(&d_perf, 4 * sizeof(float)));
    GpuTimer t;
    t.begin();
    for (int i = 0; i < 1000; i++) shfl_perf_kernel<<<1, 32>>>(d_perf, 100);
    t.end();
    printf("  perf: %.2f us/launch (1000 launches, 100 reductions each)\n",
           t.elapsed_ms() * 1000.0f / 1000);

    cudaFree(d_u); cudaFree(d_f); cudaFree(d_perf);
    if (all_pass) { PASS(); return 0; }
    else          { FAIL("some subtests failed"); return 1; }
}


int main() {
  int rc_ours   = run_ours();
  int rc_theirs = run_theirs();
  return (rc_ours == 0 && rc_theirs == 0) ? 0 : 1;
}
