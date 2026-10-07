// ARCH: sm_90a
// 83_grouped_gemm_tile_map_test.cu -- verify group boundaries + within-group coords.
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
#include "../src/composites/83_grouped_gemm_tile_map.cuh"
#include "83_grouped_gemm_tile_map.cuh"

// =============================================================================
// ours (originally guarded by PL_AGENTIC_SM100A || PL_AGENTIC_SM103A)
// =============================================================================

// ARCH: sm_90a


__global__ void k(int* out,
                  const int* offsets,
                  const int* tm, const int* tn, int ng) {
  int total = offsets[ng];
  for (int i = threadIdx.x; i < total; i += blockDim.x) {
    GroupedTileCoord c = grouped_tile_map(i, offsets, tm, tn, ng);
    out[i * 3 + 0] = c.group;
    out[i * 3 + 1] = c.tile_m;
    out[i * 3 + 2] = c.tile_n;
  }
}

static int run_ours() {
  /* (orig args dropped) */
  int offsets[] = { 0, 6, 10, 16 };  // 3 groups: 6, 4, 6 tiles
  int tm[] = { 2, 2, 3 };  // rows per group
  int tn[] = { 3, 2, 2 };  // cols per group
  int ng = 3, total = offsets[3];

  int *d_off = nullptr, *d_tm = nullptr, *d_tn = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_off, 4 * 4));
  CUDA_CHECK(cudaMalloc(&d_tm,  3 * 4));
  CUDA_CHECK(cudaMalloc(&d_tn,  3 * 4));
  CUDA_CHECK(cudaMalloc(&d_out, total * 3 * 4));
  CUDA_CHECK(cudaMemcpy(d_off, offsets, 16, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_tm,  tm,      12, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_tn,  tn,      12, cudaMemcpyHostToDevice));

  k<<<1, 32>>>(d_out, d_off, d_tm, d_tn, ng);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<int> h(total * 3);
  CUDA_CHECK(cudaMemcpy(h.data(), d_out, total * 3 * 4,
                         cudaMemcpyDeviceToHost));
  cudaFree(d_off); cudaFree(d_tm); cudaFree(d_tn); cudaFree(d_out);

  // Expect: tile 0..5 in group 0, 6..9 in group 1, 10..15 in group 2.
  int fails = 0;
  for (int i = 0; i < total; ++i) {
    int g = h[i * 3];
    int expect_g = (i < 6) ? 0 : (i < 10 ? 1 : 2);
    if (g != expect_g) ++fails;
  }
  printf("grouped_tile_map group assignment : fails = %d / %d\n", fails, total);
  if (fails) FAIL("group assignment wrong");
  PASS();
}

// =============================================================================
// theirs (originally guarded by PL_AGENTIC_SM90A)
// =============================================================================

// Runtime test: grouped GEMM tile mapping (tile_id -> group, tile_m, tile_n)

// Groups: tiles_m = [8, 4, 6], tiles_n = 3.
// Per group total tiles: 8*3=24, 4*3=12, 6*3=18 -> cumul = [24, 36, 54].
// Grand total = 54 tiles.
//
// For each tile_id in [0..53] the kernel writes:
//   out[tid*4 + 0] = group_id (via grouped_gemm_tile_map)
//   out[tid*4 + 1] = tile_m
//   out[tid*4 + 2] = tile_n
//   out[tid*4 + 3] = find_group_id (sanity -- should match column 0)
__global__ void grouped_map_kernel(const int* group_cumul,
                                     const int* group_tiles_m,
                                     int num_groups, int tiles_n,
                                     int total_tiles,
                                     int* out) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= total_tiles) return;
    int gid, tm, tn;
    grouped_gemm_tile_map(tid, group_cumul, group_tiles_m, tiles_n, num_groups, gid, tm, tn);
    int gid2 = find_group_id(tid, group_cumul, num_groups);
    out[tid * 4 + 0] = gid;
    out[tid * 4 + 1] = tm;
    out[tid * 4 + 2] = tn;
    out[tid * 4 + 3] = gid2;
}

__global__ void grouped_map_perf_kernel(const int* group_cumul,
                                          const int* group_tiles_m,
                                          int num_groups, int tiles_n,
                                          int total_tiles,
                                          int* sink) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= total_tiles) return;
    int gid, tm, tn;
    grouped_gemm_tile_map(tid, group_cumul, group_tiles_m, tiles_n, num_groups, gid, tm, tn);
    if (tid == 0) sink[0] = gid + tm + tn;
}

static int run_theirs() {
  /* (orig args dropped) */
    CUDA_CHECK(cudaFree(0));
    bool all_pass = true;

    const int num_groups = 3;
    const int tiles_n = 3;
    int h_group_tiles_m[num_groups] = {8, 4, 6};
    int h_group_tiles[num_groups];
    int h_group_cumul[num_groups];
    for (int g = 0; g < num_groups; g++) h_group_tiles[g] = h_group_tiles_m[g] * tiles_n;
    int sum = 0;
    for (int g = 0; g < num_groups; g++) { sum += h_group_tiles[g]; h_group_cumul[g] = sum; }
    const int total_tiles = h_group_cumul[num_groups - 1];  // 24 + 12 + 18 = 54

    if (total_tiles != 54) { printf("  setup: total_tiles=%d != 54\n", total_tiles); return 1; }

    int *d_cumul, *d_tiles_m, *d_out;
    CUDA_CHECK(cudaMalloc(&d_cumul,   num_groups * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tiles_m, num_groups * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out,     total_tiles * 4 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_cumul,   h_group_cumul,   num_groups * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tiles_m, h_group_tiles_m, num_groups * sizeof(int), cudaMemcpyHostToDevice));

    grouped_map_kernel<<<1, 64>>>(d_cumul, d_tiles_m, num_groups, tiles_n, total_tiles, d_out);
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_out[54 * 4];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, total_tiles * 4 * sizeof(int), cudaMemcpyDeviceToHost));

    auto check = [&](int tid, int exp_g, int exp_m, int exp_n, const char* tag) {
        int g = h_out[tid * 4 + 0], m = h_out[tid * 4 + 1];
        int n = h_out[tid * 4 + 2], g2 = h_out[tid * 4 + 3];
        bool ok = (g == exp_g && m == exp_m && n == exp_n && g2 == exp_g);
        if (!ok) {
            printf("    %s tile_id=%d: got (g=%d,m=%d,n=%d,g2=%d), expected (g=%d,m=%d,n=%d)\n",
                   tag, tid, g, m, n, g2, exp_g, exp_m, exp_n);
        }
        return ok;
    };

    // Per-task spec:
    //   tile_id=0  -> group=0, tile_m=0, tile_n=0
    //   tile_id=23 -> group=0, tile_n=2, tile_m=7 (last of group 0; 8*3=24 tiles)
    //   tile_id=24 -> group=1, tile_m=0, tile_n=0
    //   tile_id=35 -> group=1, tile_n=2, tile_m=3 (last of group 1; 12 tiles)
    //   tile_id=36 -> group=2, tile_m=0, tile_n=0
    bool ok = true;
    ok &= check(0,  0, 0, 0, "boundary");
    ok &= check(23, 0, 7, 2, "boundary");
    ok &= check(24, 1, 0, 0, "boundary");
    ok &= check(35, 1, 3, 2, "boundary");
    ok &= check(36, 2, 0, 0, "boundary");
    ok &= check(53, 2, 5, 2, "boundary");  // last tile of group 2 (6*3 = 18)

    if (ok) printf("  boundary tiles: OK\n");
    else    { printf("  boundary tiles: FAIL\n"); all_pass = false; }

    // Full sweep: recompute expected on host
    int bad = 0;
    for (int tid = 0; tid < total_tiles; tid++) {
        int exp_g = 0;
        while (exp_g < num_groups && h_group_cumul[exp_g] <= tid) exp_g++;
        int gstart = (exp_g > 0) ? h_group_cumul[exp_g - 1] : 0;
        int tig = tid - gstart;
        int gtm = h_group_tiles_m[exp_g];
        int exp_n = tig / gtm;
        int exp_m = tig % gtm;
        int g  = h_out[tid * 4 + 0];
        int m  = h_out[tid * 4 + 1];
        int n  = h_out[tid * 4 + 2];
        int g2 = h_out[tid * 4 + 3];
        if (g != exp_g || m != exp_m || n != exp_n || g2 != exp_g) {
            if (bad < 3) {
                printf("    sweep tile_id=%d: got (g=%d,m=%d,n=%d,g2=%d), expected (g=%d,m=%d,n=%d)\n",
                       tid, g, m, n, g2, exp_g, exp_m, exp_n);
            }
            bad++;
        }
    }
    if (bad == 0) printf("  full sweep (%d tiles): OK\n", total_tiles);
    else          { printf("  full sweep: FAIL (%d bad)\n", bad); all_pass = false; }

    // find_group_id standalone spot checks at boundaries
    // boundary within host replica of kernel path: trivial sanity
    // tile_id just inside group g has group_id == g
    // cumul = [24, 36, 54]; so group 0: [0..23], group 1: [24..35], group 2: [36..53]
    int bad_fg = 0;
    for (int tid : {0, 1, 23, 24, 25, 35, 36, 53}) {
        int got = h_out[tid * 4 + 3];
        int exp = (tid < 24) ? 0 : (tid < 36) ? 1 : 2;
        if (got != exp) { bad_fg++; printf("    fg tid=%d got=%d exp=%d\n", tid, got, exp); }
    }
    if (bad_fg == 0) printf("  find_group_id boundaries: OK\n");
    else             { printf("  find_group_id: FAIL (%d)\n", bad_fg); all_pass = false; }

    // Perf: bundle every tile once per launch
    int *d_sink; CUDA_CHECK(cudaMalloc(&d_sink, 4));
    GpuTimer t;
    t.begin();
    for (int i = 0; i < 1000; i++) {
        grouped_map_perf_kernel<<<1, 64>>>(d_cumul, d_tiles_m, num_groups, tiles_n, total_tiles, d_sink);
    }
    t.end();
    printf("  perf: %.2f us/launch (1000 launches, %d tiles each)\n",
           t.elapsed_ms() * 1000.0f / 1000, total_tiles);

    cudaFree(d_cumul); cudaFree(d_tiles_m); cudaFree(d_out); cudaFree(d_sink);
    if (all_pass) { PASS(); return 0; }
    else          { FAIL("some subtests failed"); return 1; }
}


int main() {
  int rc_ours   = run_ours();
  int rc_theirs = run_theirs();
  return (rc_ours == 0 && rc_theirs == 0) ? 0 : 1;
}
