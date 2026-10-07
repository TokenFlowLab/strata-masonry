// ARCH: sm_90a
// 82_tile_rasterize_test.cu -- verify all three rasterizers cover a grid.
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
#include "../src/composites/82_tile_rasterize.cuh"
#include <set>
#include <utility>
#include "82_tile_rasterize.cuh"

// =============================================================================
// ours (originally guarded by PL_AGENTIC_SM100A || PL_AGENTIC_SM103A)
// =============================================================================

// ARCH: sm_90a


static int run_ours() {
  /* (orig args dropped) */
  const int TM = 4, TN = 5;
  std::set<std::pair<int,int>> seen_col, seen_row, seen_sw;
  for (int t = 0; t < TM * TN; ++t) {
    int m, n;
    tile_rasterize_colmajor(t, TM, TN, m, n); seen_col.insert({m, n});
    tile_rasterize_rowmajor(t, TM, TN, m, n); seen_row.insert({m, n});
    tile_rasterize_swizzled<2>(t, TM, TN, m, n); seen_sw.insert({m, n});
  }
  printf("rasterize coverage: col=%zu row=%zu swizzled=%zu (want %d)\n",
         seen_col.size(), seen_row.size(), seen_sw.size(), TM * TN);
  if ((int)seen_col.size() != TM * TN) FAIL("colmajor missed tiles");
  if ((int)seen_row.size() != TM * TN) FAIL("rowmajor missed tiles");
  PASS();
}

// =============================================================================
// theirs (originally guarded by PL_AGENTIC_SM90A)
// =============================================================================

// Runtime test: tile rasterization (col/row-major, swizzled, snake)

// For an 8x4 grid (tiles_m=8, tiles_n=4, 32 tiles total), the 1-block kernel
// walks tile_id = 0..31 and writes (tile_m, tile_n) for each raster function
// into separate output arrays.
__global__ void rasterize_8x4_kernel(int* out_col, int* out_row, int* out_snake) {
    int tid = threadIdx.x;
    int tiles_m = 8, tiles_n = 4;
    if (tid >= tiles_m * tiles_n) return;
    int m, n;
    tile_rasterize_col_major(tid, tiles_m, tiles_n, m, n);
    out_col[tid * 2 + 0] = m; out_col[tid * 2 + 1] = n;
    tile_rasterize_row_major(tid, tiles_m, tiles_n, m, n);
    out_row[tid * 2 + 0] = m; out_row[tid * 2 + 1] = n;
    tile_rasterize_snake(tid, tiles_m, tiles_n, m, n);
    out_snake[tid * 2 + 0] = m; out_snake[tid * 2 + 1] = n;
}

// Swizzled<4> test uses an 8x8 grid (64 tiles) so the full group structure
// of tiles_per_group=32 is visible.
__global__ void rasterize_swizzle_kernel(int* out) {
    int tid = threadIdx.x;
    int tiles_m = 8, tiles_n = 8;
    if (tid >= tiles_m * tiles_n) return;
    int m, n;
    tile_rasterize_swizzled<4>(tid, tiles_m, tiles_n, m, n);
    out[tid * 2 + 0] = m; out[tid * 2 + 1] = n;
}

// Perf: run several raster functions on every lane
__global__ void rasterize_perf_kernel(int* out, int iters) {
    int tid = threadIdx.x;
    int m = 0, n = 0;
    for (int i = 0; i < iters; i++) {
        tile_rasterize_col_major(tid + i, 16, 16, m, n);
    }
    if (tid == 0) out[blockIdx.x] = m + n;
}

static bool check_pair(const int* buf, int idx, int em, int en, const char* tag) {
    int m = buf[idx * 2 + 0], n = buf[idx * 2 + 1];
    if (m != em || n != en) {
        printf("    %s tile_id=%d: got (%d,%d), expected (%d,%d)\n",
               tag, idx, m, n, em, en);
        return false;
    }
    return true;
}

static int run_theirs() {
  /* (orig args dropped) */
    CUDA_CHECK(cudaFree(0));
    bool all_pass = true;

    const int TILES84 = 8 * 4;
    int *d_col, *d_row, *d_snake;
    CUDA_CHECK(cudaMalloc(&d_col,   TILES84 * 2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_row,   TILES84 * 2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_snake, TILES84 * 2 * sizeof(int)));

    rasterize_8x4_kernel<<<1, 32>>>(d_col, d_row, d_snake);
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_col[TILES84 * 2], h_row[TILES84 * 2], h_snake[TILES84 * 2];
    CUDA_CHECK(cudaMemcpy(h_col,   d_col,   sizeof(h_col),   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_row,   d_row,   sizeof(h_row),   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_snake, d_snake, sizeof(h_snake), cudaMemcpyDeviceToHost));

    // col_major(8x4): tile_id=0->(0,0), 1->(1,0), 7->(7,0), 8->(0,1), 31->(7,3)
    bool ok = true;
    ok &= check_pair(h_col, 0,  0, 0, "col");
    ok &= check_pair(h_col, 1,  1, 0, "col");
    ok &= check_pair(h_col, 7,  7, 0, "col");
    ok &= check_pair(h_col, 8,  0, 1, "col");
    ok &= check_pair(h_col, 9,  1, 1, "col");
    ok &= check_pair(h_col, 31, 7, 3, "col");
    // Full sweep vs formula
    for (int id = 0; id < TILES84; id++) {
        int em = id % 8, en = id / 8;
        if (h_col[id*2+0] != em || h_col[id*2+1] != en) { ok = false; }
    }
    if (ok) printf("  col_major: OK (%d tiles)\n", TILES84);
    else    { printf("  col_major: FAIL\n"); all_pass = false; }

    // row_major(8x4): tile_m = id/tiles_n=id/4, tile_n = id%tiles_n=id%4
    // 0->(0,0), 4->(1,0), 31->(7,3)
    ok = true;
    ok &= check_pair(h_row, 0,  0, 0, "row");
    ok &= check_pair(h_row, 4,  1, 0, "row");
    ok &= check_pair(h_row, 8,  2, 0, "row");
    ok &= check_pair(h_row, 31, 7, 3, "row");
    for (int id = 0; id < TILES84; id++) {
        int em = id / 4, en = id % 4;
        if (h_row[id*2+0] != em || h_row[id*2+1] != en) { ok = false; }
    }
    if (ok) printf("  row_major: OK (%d tiles)\n", TILES84);
    else    { printf("  row_major: FAIL\n"); all_pass = false; }

    // snake(8x4): tile_n=0 forward (0,0..7), tile_n=1 reversed (7..0,1)
    // id=0 -> (0,0), id=7 -> (7,0), id=8 -> (7,1), id=15 -> (0,1), id=16 -> (0,2)
    ok = true;
    ok &= check_pair(h_snake, 0,  0, 0, "snake");
    ok &= check_pair(h_snake, 7,  7, 0, "snake");
    ok &= check_pair(h_snake, 8,  7, 1, "snake");
    ok &= check_pair(h_snake, 15, 0, 1, "snake");
    ok &= check_pair(h_snake, 16, 0, 2, "snake");
    ok &= check_pair(h_snake, 23, 7, 2, "snake");
    ok &= check_pair(h_snake, 24, 7, 3, "snake");
    for (int id = 0; id < TILES84; id++) {
        int raw_m = id % 8, en = id / 8;
        int em = (en & 1) ? (8 - 1 - raw_m) : raw_m;
        if (h_snake[id*2+0] != em || h_snake[id*2+1] != en) { ok = false; }
    }
    if (ok) printf("  snake: OK (%d tiles)\n", TILES84);
    else    { printf("  snake: FAIL\n"); all_pass = false; }

    // Swizzled<4> on 8x8 grid: tiles_per_group = 8*4 = 32
    //   id=0  -> group=0, tile_in_group=0, col=0, (m,n)=(0,0)
    //   id=7  -> tile_in_group=7, col=0,     (m,n)=(7,0)
    //   id=8  -> tile_in_group=8, col=1,     (m,n)=(0,1)
    //   id=31 -> tile_in_group=31, col=3,    (m,n)=(7,3)
    //   id=32 -> group=1, tile_in_group=0,   (m,n)=(0,4)
    //   id=63 -> group=1, tile_in_group=31,  (m,n)=(7,7)
    const int TILES88 = 8 * 8;
    int *d_sw;
    CUDA_CHECK(cudaMalloc(&d_sw, TILES88 * 2 * sizeof(int)));
    rasterize_swizzle_kernel<<<1, 64>>>(d_sw);
    CUDA_CHECK(cudaDeviceSynchronize());
    int h_sw[TILES88 * 2];
    CUDA_CHECK(cudaMemcpy(h_sw, d_sw, sizeof(h_sw), cudaMemcpyDeviceToHost));

    ok = true;
    ok &= check_pair(h_sw, 0,  0, 0, "sw<4>");
    ok &= check_pair(h_sw, 7,  7, 0, "sw<4>");
    ok &= check_pair(h_sw, 8,  0, 1, "sw<4>");
    ok &= check_pair(h_sw, 15, 7, 1, "sw<4>");
    ok &= check_pair(h_sw, 31, 7, 3, "sw<4>");
    ok &= check_pair(h_sw, 32, 0, 4, "sw<4>");
    ok &= check_pair(h_sw, 63, 7, 7, "sw<4>");
    for (int id = 0; id < TILES88; id++) {
        int tiles_per_group = 8 * 4;
        int group_id = id / tiles_per_group;
        int tile_in_group = id % tiles_per_group;
        int col_in_group = tile_in_group / 8;
        int em = tile_in_group % 8;
        int en = group_id * 4 + col_in_group;
        if (h_sw[id*2+0] != em || h_sw[id*2+1] != en) { ok = false; }
    }
    if (ok) printf("  swizzled<4>: OK (%d tiles, group size 4)\n", TILES88);
    else    { printf("  swizzled<4>: FAIL\n"); all_pass = false; }

    // Perf
    int *d_perf; CUDA_CHECK(cudaMalloc(&d_perf, 4 * sizeof(int)));
    GpuTimer t;
    t.begin();
    for (int i = 0; i < 1000; i++) rasterize_perf_kernel<<<1, 32>>>(d_perf, 100);
    t.end();
    printf("  perf: %.2f us/launch (1000 launches, 100 ops each)\n",
           t.elapsed_ms() * 1000.0f / 1000);

    cudaFree(d_col); cudaFree(d_row); cudaFree(d_snake); cudaFree(d_sw); cudaFree(d_perf);
    if (all_pass) { PASS(); return 0; }
    else          { FAIL("some subtests failed"); return 1; }
}


int main() {
  int rc_ours   = run_ours();
  int rc_theirs = run_theirs();
  return (rc_ours == 0 && rc_theirs == 0) ? 0 : 1;
}
