// ARCH: sm_90a
// 25_tma_async_group_test.cu -- bulk commit/wait covered by #22. Compile smoke.
//
// Two test sets in one binary: run_ours() and run_theirs().

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "test_utils.cuh"
#include "../src/primitives/25_tma_async_group.cuh"
#include <cuda_fp16.h>
#include "22_tma_store.cuh"
#include "23_tma_tensormap.cuh"
#include "25_tma_async_group.cuh"
#include "34_fence_proxy_async.cuh"

// =============================================================================
// ours
// =============================================================================

// ARCH: sm_90a


__global__ void k_group() {
  cp_async_bulk_commit_group();
  cp_async_bulk_wait_group<0>();
}

static int run_ours() {
  k_group<<<1, 32>>>();
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("cp.async.bulk.commit/wait_group : compile OK\n");
  PASS();
}

// =============================================================================
// theirs
// =============================================================================

// Runtime test: TMA bulk async-group commit/wait -- correctness + timing
// Build: nvcc -gencode arch=compute_90a,code=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr
//        -DNDEBUG -lineinfo -I primitives -I tests -lcuda -o build/25_test tests/25_tma_async_group_test.cu
//
// We issue 3 TMA stores from SMEM -> 3 distinct GMEM regions. After each
// store, we call tma_store_commit_group to close a group. Then we
// progressively drain: wait_group<2> (at most 2 pending), wait_group<1>,
// wait_group<0> (all drained). After each wait, the corresponding store's
// GMEM must be visible on the next read.
//
// The main correctness check is: after the final wait_group<0>, all three
// target buffers contain the expected pattern.

static constexpr int TILE_ROWS = 32;
static constexpr int TILE_COLS = 32;
static constexpr int TILE_ELEMS = TILE_ROWS * TILE_COLS;
static constexpr int TILE_BYTES = TILE_ELEMS * (int)sizeof(half);

extern __shared__ __align__(128) char smem_buf[];

// One CTA fills 3 SMEM tiles with distinct patterns, launches 3 separate
// TMA store groups, and drains them in order. We use SWIZZLE_NONE so the
// SMEM image equals the GMEM image after the round-trip.
__global__ void multi_store_group_kernel(
    const __grid_constant__ CUtensorMap tmap_a,
    const __grid_constant__ CUtensorMap tmap_b,
    const __grid_constant__ CUtensorMap tmap_c)
{
    // Three back-to-back tiles in SMEM
    half* smem_a = reinterpret_cast<half*>(smem_buf);
    half* smem_b = reinterpret_cast<half*>(smem_buf + TILE_BYTES);
    half* smem_c = reinterpret_cast<half*>(smem_buf + 2 * TILE_BYTES);

    int tid = threadIdx.x;
    // Cooperatively populate each tile with a simple pattern.
    for (int i = tid; i < TILE_ELEMS; i += blockDim.x) {
        smem_a[i] = __float2half((float)(i + 1));        // 1..TILE_ELEMS
        smem_b[i] = __float2half((float)(i + 1000));     // 1000..
        smem_c[i] = __float2half((float)(i * 2));        // 0,2,4,...
    }
    __syncthreads();

    if (tid == 0) {
        // SMEM generic-proxy writes must be ordered before TMA async-proxy reads.
        fence_proxy_async_shared_cta();

        // Store 1: tile A -> group G1
        tma_store_2d(&tmap_a, 0, 0, smem_ptr_u32(smem_a));
        tma_store_commit_group();
        // Store 2: tile B -> group G2
        tma_store_2d(&tmap_b, 0, 0, smem_ptr_u32(smem_b));
        tma_store_commit_group();
        // Store 3: tile C -> group G3
        tma_store_2d(&tmap_c, 0, 0, smem_ptr_u32(smem_c));
        tma_store_commit_group();

        // Drain progressively: at most 2 pending, then 1, then 0.
        tma_store_wait_group<2>();  // waits for G1
        tma_store_wait_group<1>();  // waits for G2
        tma_store_wait_group<0>();  // waits for G3
    }
}

static int run_theirs() {
    CUDA_CHECK(cudaFree(0));  // init driver API

    // Allocate 3 GMEM tiles
    half *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, TILE_BYTES));
    CUDA_CHECK(cudaMalloc(&d_b, TILE_BYTES));
    CUDA_CHECK(cudaMalloc(&d_c, TILE_BYTES));
    CUDA_CHECK(cudaMemset(d_a, 0, TILE_BYTES));
    CUDA_CHECK(cudaMemset(d_b, 0, TILE_BYTES));
    CUDA_CHECK(cudaMemset(d_c, 0, TILE_BYTES));

    // Build 3 tensormaps -- SWIZZLE_NONE so SMEM == GMEM layout
    CUtensorMap tmap_a{}, tmap_b{}, tmap_c{};
    create_tma_2d_desc(&tmap_a, d_a, TILE_ROWS, TILE_COLS, TILE_ROWS, TILE_COLS,
                       2, CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
                       CU_TENSOR_MAP_SWIZZLE_NONE);
    create_tma_2d_desc(&tmap_b, d_b, TILE_ROWS, TILE_COLS, TILE_ROWS, TILE_COLS,
                       2, CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
                       CU_TENSOR_MAP_SWIZZLE_NONE);
    create_tma_2d_desc(&tmap_c, d_c, TILE_ROWS, TILE_COLS, TILE_ROWS, TILE_COLS,
                       2, CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
                       CU_TENSOR_MAP_SWIZZLE_NONE);

    size_t smem_bytes = 3 * TILE_BYTES;  // need enough for 3 tiles
    CUDA_CHECK(cudaFuncSetAttribute(multi_store_group_kernel,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)smem_bytes));

    GpuTimer t;
    t.begin();
    multi_store_group_kernel<<<1, 128, smem_bytes>>>(tmap_a, tmap_b, tmap_c);
    t.end();
    CUDA_CHECK(cudaGetLastError());

    std::vector<half> h_a(TILE_ELEMS), h_b(TILE_ELEMS), h_c(TILE_ELEMS);
    CUDA_CHECK(cudaMemcpy(h_a.data(), d_a, TILE_BYTES, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b.data(), d_b, TILE_BYTES, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, TILE_BYTES, cudaMemcpyDeviceToHost));

    int mism = 0;
    for (int i = 0; i < TILE_ELEMS; i++) {
        float a = __half2float(h_a[i]);
        float b = __half2float(h_b[i]);
        float c = __half2float(h_c[i]);
        if (a != (float)(i + 1))    { if (mism < 5) printf("  a[%d]=%.1f exp=%d\n", i, a, i+1);    mism++; }
        if (b != (float)(i + 1000)) { if (mism < 5) printf("  b[%d]=%.1f exp=%d\n", i, b, i+1000); mism++; }
        if (c != (float)(i * 2))    { if (mism < 5) printf("  c[%d]=%.1f exp=%d\n", i, c, i*2);    mism++; }
    }

    printf("  kernel time: %.3f ms (smem %zu B)\n", t.elapsed_ms(), smem_bytes);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    if (mism == 0) { PASS(); return 0; }
    else           { FAIL("%d mismatches", mism); return 1; }
}


int main() {
  int rc_ours   = run_ours();
  int rc_theirs = run_theirs();
  return (rc_ours == 0 && rc_theirs == 0) ? 0 : 1;
}
