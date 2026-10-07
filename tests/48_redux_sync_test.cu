// ARCH: sm_90a
// 48_redux_sync_test.cu -- sum[0..31] = 496 via redux.sync.add.u32.
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
#include "../src/primitives/48_redux_sync.cuh"
#include "48_redux_sync.cuh"

// =============================================================================
// ours (originally guarded by PL_AGENTIC_SM100A || PL_AGENTIC_SM103A)
// =============================================================================

// ARCH: sm_90a


__global__ void k_redux(uint32_t* out) {
  uint32_t s = redux_sync_add_u32(threadIdx.x);
  if (threadIdx.x == 0) *out = s;
}

static int run_ours() {
  /* (orig args dropped) */
  uint32_t* d = nullptr; CUDA_CHECK(cudaMalloc(&d, 4));
  CUDA_CHECK(cudaMemset(d, 0, 4));
  k_redux<<<1, 32>>>(d);
  CUDA_CHECK(cudaDeviceSynchronize());
  uint32_t h = 0; CUDA_CHECK(cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost));
  cudaFree(d);
  if (h != 496u) FAIL("redux.sync.add.u32 wrong");
  printf("redux.sync.add.u32(0..31) = %u\n", h);
  PASS();
}

// =============================================================================
// theirs (originally guarded by PL_AGENTIC_SM90A)
// =============================================================================

// Runtime test: redux.sync -- hardware warp-wide reduction

// Every lane writes its redux result to a separate slot so we can verify that
// the result is identical on every lane (a property of redux.sync).
__global__ void redux_kernel(uint32_t* out_u, int32_t* out_s) {
    int lane = threadIdx.x;
    uint32_t uv = (uint32_t)lane;
    int32_t  sv = lane - 16;  // -16..15

    out_u[lane * 6 + 0] = redux_sync_add_u32(uv);
    out_u[lane * 6 + 1] = redux_sync_min_u32(uv);
    out_u[lane * 6 + 2] = redux_sync_max_u32(uv);
    out_u[lane * 6 + 3] = redux_sync_and_b32(uv);
    out_u[lane * 6 + 4] = redux_sync_or_b32(uv);
    out_u[lane * 6 + 5] = redux_sync_xor_b32(uv);

    out_s[lane * 3 + 0] = redux_sync_add_s32(sv);
    out_s[lane * 3 + 1] = redux_sync_min_s32(sv);
    out_s[lane * 3 + 2] = redux_sync_max_s32(sv);
}

__global__ void redux_perf_kernel(uint32_t* out, int iters) {
    uint32_t v = (uint32_t)threadIdx.x;
    for (int i = 0; i < iters; i++) {
        v = redux_sync_add_u32(v) + 1;
    }
    if (threadIdx.x == 0) out[blockIdx.x] = v;
}

static int run_theirs() {
  /* (orig args dropped) */
    CUDA_CHECK(cudaFree(0));
    bool all_pass = true;

    uint32_t *d_u; CUDA_CHECK(cudaMalloc(&d_u, 32 * 6 * sizeof(uint32_t)));
    int32_t  *d_s; CUDA_CHECK(cudaMalloc(&d_s, 32 * 3 * sizeof(int32_t)));
    redux_kernel<<<1, 32>>>(d_u, d_s);
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t h_u[32 * 6]; int32_t h_s[32 * 3];
    CUDA_CHECK(cudaMemcpy(h_u, d_u, sizeof(h_u), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_s, d_s, sizeof(h_s), cudaMemcpyDeviceToHost));

    // 0+1+...+31 = 496
    auto check_all_eq_u = [&](int col, uint32_t expected, const char* name) {
        bool ok = true;
        for (int l = 0; l < 32; l++) {
            if (h_u[l * 6 + col] != expected) { ok = false; break; }
        }
        if (ok) printf("  %s: OK (%u)\n", name, expected);
        else    { printf("  %s: FAIL (lane0=%u, expected %u)\n", name, h_u[col], expected); all_pass = false; }
    };
    auto check_all_eq_s = [&](int col, int32_t expected, const char* name) {
        bool ok = true;
        for (int l = 0; l < 32; l++) {
            if (h_s[l * 3 + col] != expected) { ok = false; break; }
        }
        if (ok) printf("  %s: OK (%d)\n", name, expected);
        else    { printf("  %s: FAIL (lane0=%d, expected %d)\n", name, h_s[col], expected); all_pass = false; }
    };

    check_all_eq_u(0, 496u, "redux.add.u32");
    check_all_eq_u(1, 0u,   "redux.min.u32");
    check_all_eq_u(2, 31u,  "redux.max.u32");
    check_all_eq_u(3, 0u,   "redux.and.b32");   // bit 0 zero on lane 0
    check_all_eq_u(4, 31u,  "redux.or.b32");    // bits 0-4 covered across lanes
    // xor of 0..31: pairs (0^1)^(2^3)^... -- 0..31 has balanced pairs, result 0
    check_all_eq_u(5, 0u,   "redux.xor.b32");

    // signed: sv = lane - 16, so lane0 = -16, lane31 = 15
    // sum(-16..15) = -16
    check_all_eq_s(0, -16, "redux.add.s32");
    check_all_eq_s(1, -16, "redux.min.s32");
    check_all_eq_s(2,  15, "redux.max.s32");

    // Perf
    uint32_t *d_perf; CUDA_CHECK(cudaMalloc(&d_perf, 4 * sizeof(uint32_t)));
    GpuTimer t;
    t.begin();
    for (int i = 0; i < 1000; i++) redux_perf_kernel<<<1, 32>>>(d_perf, 100);
    t.end();
    printf("  perf: %.2f us/launch (1000 launches, 100 reductions each)\n",
           t.elapsed_ms() * 1000.0f / 1000);

    cudaFree(d_u); cudaFree(d_s); cudaFree(d_perf);
    if (all_pass) { PASS(); return 0; }
    else          { FAIL("some subtests failed"); return 1; }
}


int main() {
  int rc_ours   = run_ours();
  int rc_theirs = run_theirs();
  return (rc_ours == 0 && rc_theirs == 0) ? 0 : 1;
}
