// ARCH: sm_90a
// 95_sched_warp_blockidx_test.cu -- runtime correctness for blockIdx scheduler.
//
// Two test sets in one binary: run_ours() and run_theirs(). Kernels live
// in the header-only block (blocks/95_sched_warp_blockidx.cuh).

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <set>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "test_utils.cuh"
#include "../src/blocks/95_sched_warp_blockidx.cuh"

__global__ void sched_warp_blockidx_test_kernel(int tiles_m, int tiles_n,
                                                int2* out) {
  int tile = blockIdx.x;
  int m, n;
  sched_warp_blockidx_block(tile, tiles_m, tiles_n, m, n);
  if (threadIdx.x == 0) out[tile] = make_int2(m, n);
}

__global__ void sched_warp_blockidx_all_patterns_test_kernel(
    int tiles_m, int tiles_n,
    int* out_col, int* out_row, int* out_sw, int* out_snake,
    uint32_t* hit_col, uint32_t* hit_row,
    uint32_t* hit_sw,  uint32_t* hit_snake) {
  if (threadIdx.x != 0) return;
  sched_warp_blockidx_all_patterns_block(
      blockIdx.x, tiles_m, tiles_n,
      out_col, out_row, out_sw, out_snake,
      hit_col, hit_row, hit_sw, hit_snake);
}

// =============================================================================
// ours
// =============================================================================

static int run_ours() {
    const int TM = 5, TN = 6, TOTAL = TM * TN;
    int2* d = nullptr; CUDA_CHECK(cudaMalloc(&d, TOTAL * sizeof(int2)));
    sched_warp_blockidx_test_kernel<<<TOTAL, 32>>>(TM, TN, d);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<int2> h(TOTAL);
    CUDA_CHECK(cudaMemcpy(h.data(), d, TOTAL * sizeof(int2),
                          cudaMemcpyDeviceToHost));
    cudaFree(d);
    std::set<std::pair<int,int>> seen;
    for (auto& p : h) seen.insert({p.x, p.y});
    printf("sched blockidx coverage : %zu / %d\n", seen.size(), TOTAL);
    if ((int)seen.size() != TOTAL) FAIL("tiles missed");
    PASS();
}

// =============================================================================
// theirs
// =============================================================================

// CPU reference implementations (mirror device composite code).
static void cpu_col_major(int tid, int tm, int /*tn*/, int& m, int& n) {
    n = tid / tm; m = tid % tm;
}
static void cpu_row_major(int tid, int /*tm*/, int tn, int& m, int& n) {
    m = tid / tn; n = tid % tn;
}
static void cpu_swizzled(int tid, int tm, int /*tn*/, int& m, int& n, int RASTER_GROUP) {
    int tiles_per_group = tm * RASTER_GROUP;
    int group_id = tid / tiles_per_group;
    int tile_in_group = tid % tiles_per_group;
    int col_in_group = tile_in_group / tm;
    m = tile_in_group % tm;
    n = group_id * RASTER_GROUP + col_in_group;
}
static void cpu_snake(int tid, int tm, int /*tn*/, int& m, int& n) {
    n = tid / tm;
    int raw = tid % tm;
    m = (n & 1) ? (tm - 1 - raw) : raw;
}

static bool check_coverage(const char* name, const uint32_t* hit, int total) {
    int missing = 0, doubled = 0;
    for (int i = 0; i < total; i++) {
        if (hit[i] == 0) missing++;
        else if (hit[i] > 1) doubled++;
    }
    if (missing == 0 && doubled == 0) return true;
    printf("  [%s] coverage FAIL: missing=%d doubled=%d\n", name, missing, doubled);
    return false;
}

static bool check_order(const char* name,
                        const int* dev, int total, int tiles_m, int tiles_n,
                        void (*cpu)(int, int, int, int&, int&)) {
    for (int tid = 0; tid < total; tid++) {
        int m_ref, n_ref;
        cpu(tid, tiles_m, tiles_n, m_ref, n_ref);
        if (dev[tid*2+0] != m_ref || dev[tid*2+1] != n_ref) {
            printf("  [%s] order FAIL tid=%d: got (%d,%d) expected (%d,%d)\n",
                   name, tid, dev[tid*2+0], dev[tid*2+1], m_ref, n_ref);
            return false;
        }
    }
    return true;
}

static int run_theirs() {
    CUDA_CHECK(cudaFree(0));
    srand(2026);
    bool all_pass = true;

    constexpr int tiles_m = 8;
    constexpr int tiles_n = 6;
    constexpr int total = tiles_m * tiles_n;

    size_t pair_bytes = (size_t)total * 2 * sizeof(int);
    size_t hit_bytes  = (size_t)total * sizeof(uint32_t);

    int *d_col, *d_row, *d_sw, *d_snake;
    uint32_t *d_hit_col, *d_hit_row, *d_hit_sw, *d_hit_snake;
    CUDA_CHECK(cudaMalloc(&d_col,       pair_bytes));
    CUDA_CHECK(cudaMalloc(&d_row,       pair_bytes));
    CUDA_CHECK(cudaMalloc(&d_sw,        pair_bytes));
    CUDA_CHECK(cudaMalloc(&d_snake,     pair_bytes));
    CUDA_CHECK(cudaMalloc(&d_hit_col,   hit_bytes));
    CUDA_CHECK(cudaMalloc(&d_hit_row,   hit_bytes));
    CUDA_CHECK(cudaMalloc(&d_hit_sw,    hit_bytes));
    CUDA_CHECK(cudaMalloc(&d_hit_snake, hit_bytes));

    CUDA_CHECK(cudaMemset(d_hit_col,   0, hit_bytes));
    CUDA_CHECK(cudaMemset(d_hit_row,   0, hit_bytes));
    CUDA_CHECK(cudaMemset(d_hit_sw,    0, hit_bytes));
    CUDA_CHECK(cudaMemset(d_hit_snake, 0, hit_bytes));

    GpuTimer t;
    t.begin();
    sched_warp_blockidx_all_patterns_test_kernel<<<total, 1>>>(
        tiles_m, tiles_n,
        d_col, d_row, d_sw, d_snake,
        d_hit_col, d_hit_row, d_hit_sw, d_hit_snake);
    t.end();
    CUDA_CHECK(cudaGetLastError());

    std::vector<int>      h_col(total*2), h_row(total*2), h_sw(total*2), h_snake(total*2);
    std::vector<uint32_t> h_hit_col(total), h_hit_row(total), h_hit_sw(total), h_hit_snake(total);
    CUDA_CHECK(cudaMemcpy(h_col.data(),       d_col,       pair_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_row.data(),       d_row,       pair_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sw.data(),        d_sw,        pair_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_snake.data(),     d_snake,     pair_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_hit_col.data(),   d_hit_col,   hit_bytes,  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_hit_row.data(),   d_hit_row,   hit_bytes,  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_hit_sw.data(),    d_hit_sw,    hit_bytes,  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_hit_snake.data(), d_hit_snake, hit_bytes,  cudaMemcpyDeviceToHost));

    // Coverage checks (all four patterns must cover every tile exactly once)
    bool cov_col   = check_coverage("col",   h_hit_col.data(),   total);
    bool cov_row   = check_coverage("row",   h_hit_row.data(),   total);
    bool cov_sw    = check_coverage("swz",   h_hit_sw.data(),    total);
    bool cov_snake = check_coverage("snake", h_hit_snake.data(), total);

    // Order checks (each tile_id maps to the expected CPU reference)
    bool ord_col   = check_order("col",   h_col.data(),   total, tiles_m, tiles_n, cpu_col_major);
    bool ord_row   = check_order("row",   h_row.data(),   total, tiles_m, tiles_n, cpu_row_major);
    bool ord_snake = check_order("snake", h_snake.data(), total, tiles_m, tiles_n, cpu_snake);

    bool ord_sw = true;
    for (int tid = 0; tid < total; tid++) {
        int m_ref, n_ref; cpu_swizzled(tid, tiles_m, tiles_n, m_ref, n_ref, 4);
        if (h_sw[tid*2+0] != m_ref || h_sw[tid*2+1] != n_ref) {
            printf("  [swz] order FAIL tid=%d got (%d,%d) exp (%d,%d)\n",
                   tid, h_sw[tid*2+0], h_sw[tid*2+1], m_ref, n_ref);
            ord_sw = false; break;
        }
    }

    bool ok = cov_col && cov_row && cov_sw && cov_snake
           && ord_col && ord_row && ord_sw && ord_snake;
    if (ok) {
        printf("  col/row/swz<4>/snake: OK (tiles %dx%d, %d CTAs, %.3f ms)\n",
               tiles_m, tiles_n, total, t.elapsed_ms());
    } else {
        all_pass = false;
    }

    // Perf: just average the launch over 100 iters.
    GpuTimer tp;
    tp.begin();
    for (int i = 0; i < 100; i++) {
        sched_warp_blockidx_all_patterns_test_kernel<<<total, 1>>>(
            tiles_m, tiles_n,
            d_col, d_row, d_sw, d_snake,
            d_hit_col, d_hit_row, d_hit_sw, d_hit_snake);
    }
    tp.end();
    printf("  perf: %.2f us/launch (%d blocks)\n",
           tp.elapsed_ms() * 1000.0f / 100, total);

    cudaFree(d_col); cudaFree(d_row); cudaFree(d_sw); cudaFree(d_snake);
    cudaFree(d_hit_col); cudaFree(d_hit_row); cudaFree(d_hit_sw); cudaFree(d_hit_snake);

    if (all_pass) { PASS(); return 0; }
    else { FAIL("subtests failed"); return 1; }
}

int main() {
    int rc_ours   = run_ours();
    int rc_theirs = run_theirs();
    return (rc_ours == 0 && rc_theirs == 0) ? 0 : 1;
}
