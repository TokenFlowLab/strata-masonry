// ARCH: sm_90a
// 96_sched_warp_persistent_test.cu -- runtime correctness for persistent scheduler.
//
// Combined test: both ours' and theirs' coverage is exercised in a single
// binary (each side's main() became run_ours / run_theirs). Kernels live
// in the new header-only block (blocks/96_sched_warp_persistent.cuh).

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "test_utils.cuh"
#include "../src/blocks/96_sched_warp_persistent.cuh"

__global__ void sched_warp_persistent_test_kernel(
    int total_tiles, int tiles_m, int tiles_n,
    uint32_t* counter, int* tile_log) {
  sched_warp_persistent_block(total_tiles, tiles_m, tiles_n, counter, tile_log);
}

__global__ void sched_warp_persistent_all_test_kernel(
    uint32_t* tile_counter,
    int total_tiles, int tiles_m, int tiles_n,
    int* out_tile_m, int* out_tile_n, int* out_owner_cta,
    uint32_t* hit_grid, uint32_t* claim_count) {
  sched_warp_persistent_all_block(
      tile_counter, total_tiles, tiles_m, tiles_n,
      out_tile_m, out_tile_n, out_owner_cta, hit_grid, claim_count);
}

// =============================================================================
// ours (originally guarded by PL_AGENTIC_SM100A || PL_AGENTIC_SM103A)
// =============================================================================

static int run_ours() {
    const int TM = 5, TN = 8, N = TM * TN;
    uint32_t* d_cnt = nullptr; int* d_log = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cnt, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_log, N * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_cnt, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_log, 0xFF, N * sizeof(int)));
    sched_warp_persistent_test_kernel<<<8, 32>>>(N, TM, TN, d_cnt, d_log);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<int> h(N);
    CUDA_CHECK(cudaMemcpy(h.data(), d_log, N * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_cnt); cudaFree(d_log);
    int assigned = 0; for (int v : h) if (v != -1) ++assigned;
    printf("persistent scheduler : tiles assigned = %d / %d\n", assigned, N);
    if (assigned != N) FAIL("persistent scheduler missed tiles");
    PASS();
}

// =============================================================================
// theirs (originally guarded by PL_AGENTIC_SM90A)
// =============================================================================

static void cpu_swizzled(int tid, int tm, int /*tn*/, int& m, int& n, int RASTER_GROUP) {
    int tiles_per_group = tm * RASTER_GROUP;
    int group_id = tid / tiles_per_group;
    int tile_in_group = tid % tiles_per_group;
    int col_in_group = tile_in_group / tm;
    m = tile_in_group % tm;
    n = group_id * RASTER_GROUP + col_in_group;
}

static int run_theirs() {
    CUDA_CHECK(cudaFree(0));
    bool all_pass = true;

    constexpr int tiles_m     = 16;
    constexpr int tiles_n     = 8;
    constexpr int total_tiles = tiles_m * tiles_n; // 128
    constexpr int num_ctas    = 16;

    uint32_t* d_counter;
    int* d_tm; int* d_tn; int* d_owner;
    uint32_t* d_hit_grid; uint32_t* d_claim_count;
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_tm,    total_tiles * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tn,    total_tiles * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_owner, total_tiles * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_hit_grid,    total_tiles * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_claim_count, num_ctas    * sizeof(uint32_t)));

    CUDA_CHECK(cudaMemset(d_counter,     0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_hit_grid,    0, total_tiles * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_claim_count, 0, num_ctas    * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_owner, 0xFF, total_tiles * sizeof(int)));

    GpuTimer t;
    t.begin();
    sched_warp_persistent_all_test_kernel<<<num_ctas, 32>>>(
        d_counter, total_tiles, tiles_m, tiles_n,
        d_tm, d_tn, d_owner, d_hit_grid, d_claim_count);
    t.end();
    CUDA_CHECK(cudaGetLastError());

    std::vector<int>      h_tm(total_tiles), h_tn(total_tiles), h_owner(total_tiles);
    std::vector<uint32_t> h_hit(total_tiles);
    std::vector<uint32_t> h_claim(num_ctas);
    uint32_t h_counter = 0;
    CUDA_CHECK(cudaMemcpy(h_tm.data(),    d_tm,    total_tiles * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_tn.data(),    d_tn,    total_tiles * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_owner.data(), d_owner, total_tiles * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_hit.data(),   d_hit_grid, total_tiles * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_claim.data(), d_claim_count, num_ctas * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_counter, d_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // (a) Coverage
    int missing = 0, doubled = 0;
    for (int i = 0; i < total_tiles; i++) {
        if (h_hit[i] == 0) missing++;
        else if (h_hit[i] > 1) doubled++;
    }
    if (missing == 0 && doubled == 0) {
        printf("  coverage: OK (all %d tiles hit exactly once)\n", total_tiles);
    } else {
        printf("  coverage: FAIL missing=%d doubled=%d\n", missing, doubled);
        all_pass = false;
    }

    // (b) Ownership
    int unassigned = 0;
    for (int i = 0; i < total_tiles; i++) {
        if (h_owner[i] < 0 || h_owner[i] >= num_ctas) unassigned++;
    }
    if (unassigned == 0) {
        printf("  ownership: OK (every tile has a valid CTA owner)\n");
    } else {
        printf("  ownership: FAIL %d tiles unassigned\n", unassigned);
        all_pass = false;
    }

    // (c) Claim balance
    uint32_t total_claimed = 0;
    for (int i = 0; i < num_ctas; i++) total_claimed += h_claim[i];
    if (total_claimed == (uint32_t)total_tiles) {
        printf("  claim balance: OK (sum=%u across %d CTAs)\n", total_claimed, num_ctas);
    } else {
        printf("  claim balance: FAIL (sum=%u expected %d)\n", total_claimed, total_tiles);
        all_pass = false;
    }

    // (d) Raster order vs CPU swizzled<4>
    int wrong = 0;
    for (int tid = 0; tid < total_tiles; tid++) {
        int m_ref, n_ref; cpu_swizzled(tid, tiles_m, tiles_n, m_ref, n_ref, 4);
        if (h_tm[tid] != m_ref || h_tn[tid] != n_ref) {
            if (wrong < 3)
                printf("    tid=%d got (%d,%d) expected (%d,%d)\n",
                       tid, h_tm[tid], h_tn[tid], m_ref, n_ref);
            wrong++;
        }
    }
    if (wrong == 0) {
        printf("  raster order: OK (matches CPU swizzled<4>)\n");
    } else {
        printf("  raster order: FAIL %d mismatches\n", wrong);
        all_pass = false;
    }

    // counter exit value
    uint32_t expected_counter = (uint32_t)(total_tiles + num_ctas);
    if (h_counter == expected_counter) {
        printf("  counter final: OK (=%u)\n", h_counter);
    } else {
        printf("  counter final: FAIL got=%u expected=%u\n", h_counter, expected_counter);
        all_pass = false;
    }

    // Perf
    GpuTimer tp;
    tp.begin();
    for (int i = 0; i < 100; i++) {
        CUDA_CHECK(cudaMemsetAsync(d_counter, 0, sizeof(uint32_t)));
        sched_warp_persistent_all_test_kernel<<<num_ctas, 32>>>(
            d_counter, total_tiles, tiles_m, tiles_n,
            d_tm, d_tn, d_owner, d_hit_grid, d_claim_count);
    }
    tp.end();
    printf("  perf: %.2f us/launch (%d CTAs, %d tiles)\n",
           tp.elapsed_ms() * 1000.0f / 100, num_ctas, total_tiles);

    cudaFree(d_counter); cudaFree(d_tm); cudaFree(d_tn); cudaFree(d_owner);
    cudaFree(d_hit_grid); cudaFree(d_claim_count);

    if (all_pass) { PASS(); return 0; }
    else { FAIL("subtests failed"); return 1; }
}

int main() {
    int rc_ours   = run_ours();
    int rc_theirs = run_theirs();
    return (rc_ours == 0 && rc_theirs == 0) ? 0 : 1;
}
