// ARCH: sm_90a
// 22_tma_store_test.cu -- write pattern to SMEM, TMA-store to GMEM, verify.
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
#include "../src/primitives/22_tma_store.cuh"
#include "../src/primitives/23_tma_tensormap.cuh"
#include "../src/primitives/25_tma_async_group.cuh"
#include "../src/primitives/34_fence_proxy_async.cuh"
#include <cuda_fp16.h>
#include "22_tma_store.cuh"
#include "23_tma_tensormap.cuh"
#include "25_tma_async_group.cuh"
#include "34_fence_proxy_async.cuh"

// =============================================================================
// ours (originally guarded by PL_AGENTIC_SM100A || PL_AGENTIC_SM103A)
// =============================================================================

// ARCH: sm_90a


__global__ void k_tma_store(const __grid_constant__ CUtensorMap desc) {
  __shared__ __align__(128) float smem[16 * 16];
  if (threadIdx.x < 16 * 16) smem[threadIdx.x] = (float)threadIdx.x * 0.5f;
  __syncthreads();
  if (threadIdx.x == 0) {
    fence_proxy_async_shared_cta();
    tma_store_2d(&desc, 0, 0, smem_ptr_u32(smem));
    cp_async_bulk_commit_group();
    cp_async_bulk_wait_group<0>();
  }
}

static int run_ours() {
  /* (orig args dropped) */
  float* dOut = nullptr; CUDA_CHECK(cudaMalloc(&dOut, 16 * 16 * 4));
  CUDA_CHECK(cudaMemset(dOut, 0, 16 * 16 * 4));
  CUtensorMap desc;
  CUDA_CHECK(make_tma_2d_tiled(&desc, dOut, 16, 16, 16, 16, sizeof(float),
                                 CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                                 CU_TENSOR_MAP_SWIZZLE_NONE));
  k_tma_store<<<1, 256>>>(desc);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> hOut(16 * 16);
  CUDA_CHECK(cudaMemcpy(hOut.data(), dOut, 16 * 16 * 4,
                         cudaMemcpyDeviceToHost));
  cudaFree(dOut);
  int fails = 0;
  for (int i = 0; i < 16 * 16; ++i)
    if (hOut[i] != (float)i * 0.5f) ++fails;
  printf("tma_store_2d 16x16 FP32 : fails = %d / %d\n", fails, 16 * 16);
  if (fails) FAIL("tma_store content mismatch");
  PASS();
}

// =============================================================================
// theirs (originally guarded by PL_AGENTIC_SM90A)
// =============================================================================

// Runtime test: TMA 2D store (SMEM -> GMEM) -- correctness + timing
// Build: nvcc -gencode arch=compute_90a,code=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr
//        -DNDEBUG -lineinfo -I primitives -I tests -lcuda -o build/22_test tests/22_tma_store_test.cu
//
// Use SWIZZLE_NONE tensormap so SMEM and GMEM share the same row-major
// layout -- easy to verify byte-for-byte.
//
// Flow:
//  1. CPU generates a known pattern, writes it to a "reference" GMEM buffer
//     and copies it to a staging buffer that the kernel uses as SMEM source.
//  2. Kernel: cooperatively loads the staging buffer into SMEM via plain
//     st.shared (generic proxy), then issues fence.proxy.async.shared::cta,
//     then one thread issues tma_store_2d -> target GMEM.
//  3. Commit + wait_group<0>.
//  4. Host compares target GMEM vs reference pattern.

static constexpr int TILE_ROWS = 64;
static constexpr int TILE_COLS = 64;
static constexpr int TILE_ELEMS = TILE_ROWS * TILE_COLS;
static constexpr int TILE_BYTES = TILE_ELEMS * (int)sizeof(half);

extern __shared__ __align__(128) char smem_buf[];

// Plain TMA store: SMEM -> GMEM, using SWIZZLE_NONE tensormap.
// SMEM is initialized from an input GMEM buffer via generic-proxy loads.
__global__ void tma_store_kernel(const __grid_constant__ CUtensorMap tmap_out,
                                 const half* __restrict__ src_gmem) {
    half* smem_tile = reinterpret_cast<half*>(smem_buf);
    int tid = threadIdx.x;

    // Cooperative load: src_gmem -> smem_tile via generic proxy (plain ld/st).
    for (int i = tid; i < TILE_ELEMS; i += blockDim.x) {
        smem_tile[i] = src_gmem[i];
    }
    __syncthreads();

    if (tid == 0) {
        // SMEM was written in the generic proxy; fence before TMA read.
        fence_proxy_async_shared_cta();
        tma_store_2d(&tmap_out, /*coord_x=*/0, /*coord_y=*/0,
                     smem_ptr_u32(smem_tile));
        tma_store_commit_group();
        tma_store_wait_group<0>();
    }
}

// Variant using the L2-hint form -- identical semantics with a cache policy.
__global__ void tma_store_l2hint_kernel(const __grid_constant__ CUtensorMap tmap_out,
                                         const half* __restrict__ src_gmem) {
    half* smem_tile = reinterpret_cast<half*>(smem_buf);
    int tid = threadIdx.x;
    for (int i = tid; i < TILE_ELEMS; i += blockDim.x) {
        smem_tile[i] = src_gmem[i];
    }
    __syncthreads();
    if (tid == 0) {
        fence_proxy_async_shared_cta();
        tma_store_2d_l2hint(&tmap_out, 0, 0, smem_ptr_u32(smem_tile),
                            /*cache_policy=*/0ull);
        tma_store_commit_group();
        tma_store_wait_group<0>();
    }
}

static int run_theirs() {
  /* (orig args dropped) */
    CUDA_CHECK(cudaFree(0));
    bool all_pass = true;

    // Allocate input and output GMEM
    half *d_src, *d_dst, *d_dst2;
    CUDA_CHECK(cudaMalloc(&d_src,  TILE_BYTES));
    CUDA_CHECK(cudaMalloc(&d_dst,  TILE_BYTES));
    CUDA_CHECK(cudaMalloc(&d_dst2, TILE_BYTES));

    // Host pattern
    std::vector<half> h_src(TILE_ELEMS);
    for (int r = 0; r < TILE_ROWS; r++) {
        for (int c = 0; c < TILE_COLS; c++) {
            h_src[r * TILE_COLS + c] = __float2half((float)(r * 100 + c));
        }
    }
    CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), TILE_BYTES, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_dst,  0xAA, TILE_BYTES));
    CUDA_CHECK(cudaMemset(d_dst2, 0x55, TILE_BYTES));

    // Build SWIZZLE_NONE tensormaps pointing at the output buffers
    CUtensorMap tmap_out{}, tmap_out2{};
    create_tma_2d_desc(&tmap_out,  d_dst,  TILE_ROWS, TILE_COLS, TILE_ROWS, TILE_COLS,
                       2, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, CU_TENSOR_MAP_SWIZZLE_NONE);
    create_tma_2d_desc(&tmap_out2, d_dst2, TILE_ROWS, TILE_COLS, TILE_ROWS, TILE_COLS,
                       2, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, CU_TENSOR_MAP_SWIZZLE_NONE);

    size_t smem_bytes = TILE_BYTES;
    CUDA_CHECK(cudaFuncSetAttribute(tma_store_kernel,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)smem_bytes));
    CUDA_CHECK(cudaFuncSetAttribute(tma_store_l2hint_kernel,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)smem_bytes));

    GpuTimer t1;
    t1.begin();
    tma_store_kernel<<<1, 128, smem_bytes>>>(tmap_out, d_src);
    t1.end();
    CUDA_CHECK(cudaGetLastError());

    std::vector<half> h_dst(TILE_ELEMS);
    CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst, TILE_BYTES, cudaMemcpyDeviceToHost));
    int mism1 = 0;
    for (int i = 0; i < TILE_ELEMS; i++) {
        if (__half2float(h_dst[i]) != __half2float(h_src[i])) {
            if (mism1 < 5) {
                printf("  [plain] mismatch[%d]: got=%.1f exp=%.1f\n",
                       i, __half2float(h_dst[i]), __half2float(h_src[i]));
            }
            mism1++;
        }
    }
    if (mism1 == 0) printf("  plain store:  OK  (%.3f ms)\n", t1.elapsed_ms());
    else { printf("  plain store:  FAIL (%d mismatches)\n", mism1); all_pass = false; }

    GpuTimer t2;
    t2.begin();
    tma_store_l2hint_kernel<<<1, 128, smem_bytes>>>(tmap_out2, d_src);
    t2.end();
    CUDA_CHECK(cudaGetLastError());

    std::vector<half> h_dst2(TILE_ELEMS);
    CUDA_CHECK(cudaMemcpy(h_dst2.data(), d_dst2, TILE_BYTES, cudaMemcpyDeviceToHost));
    int mism2 = 0;
    for (int i = 0; i < TILE_ELEMS; i++) {
        if (__half2float(h_dst2[i]) != __half2float(h_src[i])) {
            if (mism2 < 5) {
                printf("  [l2hint] mismatch[%d]: got=%.1f exp=%.1f\n",
                       i, __half2float(h_dst2[i]), __half2float(h_src[i]));
            }
            mism2++;
        }
    }
    if (mism2 == 0) printf("  l2hint store: OK  (%.3f ms)\n", t2.elapsed_ms());
    else { printf("  l2hint store: FAIL (%d mismatches)\n", mism2); all_pass = false; }

    // Repeated-launch perf
    GpuTimer tp;
    tp.begin();
    const int ITERS = 200;
    for (int i = 0; i < ITERS; i++) {
        tma_store_kernel<<<1, 128, smem_bytes>>>(tmap_out, d_src);
    }
    tp.end();
    printf("  perf: %.2f us/launch (tile %dx%d f16, %d B)\n",
           tp.elapsed_ms() * 1000.0f / ITERS, TILE_ROWS, TILE_COLS, TILE_BYTES);

    cudaFree(d_src); cudaFree(d_dst); cudaFree(d_dst2);
    if (all_pass) { PASS(); return 0; }
    else          { FAIL("subtests failed"); return 1; }
}


// =============================================================================
// TMA reduction stores (Round 4h): cp.reduce.async.bulk.tensor.{add,min,max}
// =============================================================================

// One CTA writes a known SMEM tile through a reduction-store tensormap.
// Per PTX 9.7.10.28 the .add/.min/.max ops require integer or FP16/BF16
// dtype on the tensormap (FP32 is rejected at runtime as "illegal
// instruction"). We use INT32 here -- the natural type for split-K
// accumulating epilogues that go through INT8 GEMM with INT32 accum.
__global__ void k_tma_store_red(const __grid_constant__ CUtensorMap desc,
                                int op /*0=add 1=min 2=max*/,
                                int32_t fill_value) {
  __shared__ __align__(128) int32_t smem[16 * 16];
  if (threadIdx.x < 16 * 16) smem[threadIdx.x] = fill_value;
  __syncthreads();
  if (threadIdx.x == 0) {
    fence_proxy_async_shared_cta();
    if (op == 0)      tma_store_2d_add(&desc, 0, 0, smem_ptr_u32(smem));
    else if (op == 1) tma_store_2d_min(&desc, 0, 0, smem_ptr_u32(smem));
    else              tma_store_2d_max(&desc, 0, 0, smem_ptr_u32(smem));
    cp_async_bulk_commit_group();
    cp_async_bulk_wait_group<0>();
  }
}

static int run_red_stores() {
  const int N = 16 * 16;
  int32_t* dOut = nullptr; CUDA_CHECK(cudaMalloc(&dOut, N * 4));
  CUtensorMap desc;
  CUDA_CHECK(make_tma_2d_tiled(&desc, dOut, 16, 16, 16, 16, sizeof(int32_t),
                               CU_TENSOR_MAP_DATA_TYPE_INT32,
                               CU_TENSOR_MAP_SWIZZLE_NONE));
  std::vector<int32_t> hOut(N);

  // ADD: GMEM init = 1; store 5 twice -> 1 + 5 + 5 = 11
  std::vector<int32_t> hInit(N, 1);
  CUDA_CHECK(cudaMemcpy(dOut, hInit.data(), N * 4, cudaMemcpyHostToDevice));
  k_tma_store_red<<<1, 256>>>(desc, /*op=*/0, /*fill=*/5);
  k_tma_store_red<<<1, 256>>>(desc, /*op=*/0, /*fill=*/5);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(hOut.data(), dOut, N * 4, cudaMemcpyDeviceToHost));
  for (int i = 0; i < N; ++i) {
    if (hOut[i] != 11) {
      cudaFree(dOut);
      fprintf(stderr, "tma_store_2d_add: out[%d]=%d expected 11\n", i, hOut[i]);
      FAIL("TMA reduction store .add did not accumulate");
    }
  }

  // MIN: GMEM init = 100; store 3 (smaller) -> 3
  std::fill(hInit.begin(), hInit.end(), 100);
  CUDA_CHECK(cudaMemcpy(dOut, hInit.data(), N * 4, cudaMemcpyHostToDevice));
  k_tma_store_red<<<1, 256>>>(desc, /*op=*/1, /*fill=*/3);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(hOut.data(), dOut, N * 4, cudaMemcpyDeviceToHost));
  for (int i = 0; i < N; ++i) {
    if (hOut[i] != 3) {
      cudaFree(dOut);
      fprintf(stderr, "tma_store_2d_min: out[%d]=%d expected 3\n", i, hOut[i]);
      FAIL("TMA reduction store .min did not pick smaller");
    }
  }

  // MAX: GMEM init = 1; store 7 (larger) -> 7
  std::fill(hInit.begin(), hInit.end(), 1);
  CUDA_CHECK(cudaMemcpy(dOut, hInit.data(), N * 4, cudaMemcpyHostToDevice));
  k_tma_store_red<<<1, 256>>>(desc, /*op=*/2, /*fill=*/7);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(hOut.data(), dOut, N * 4, cudaMemcpyDeviceToHost));
  for (int i = 0; i < N; ++i) {
    if (hOut[i] != 7) {
      cudaFree(dOut);
      fprintf(stderr, "tma_store_2d_max: out[%d]=%d expected 7\n", i, hOut[i]);
      FAIL("TMA reduction store .max did not pick larger");
    }
  }

  cudaFree(dOut);
  printf("tma_store_2d_{add,min,max}: 3/3 OK (INT32: add 1+5+5=11, min(100,3)=3, max(1,7)=7)\n");
  PASS();
}

int main() {
  int rc_ours   = run_ours();
  int rc_theirs = run_theirs();
  int rc_red    = run_red_stores();
  return (rc_ours == 0 && rc_theirs == 0 && rc_red == 0) ? 0 : 1;
}
