// ARCH: sm_90a
// 18_tma_load_test.cu -- TMA-load a 16x16 FP32 tile from GMEM to SMEM,
// copy back to GMEM from a consumer thread, verify bytes match.
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
#include "../src/primitives/18_tma_load.cuh"
#include "../src/primitives/23_tma_tensormap.cuh"
#include "../src/primitives/29_mbarrier_init.cuh"
#include "../src/primitives/31_mbarrier_arrive_tx.cuh"
#include "../src/primitives/33_mbarrier_try_wait.cuh"
#include <cuda_fp16.h>
#include "18_tma_load.cuh"
#include "22_tma_store.cuh"
#include "23_tma_tensormap.cuh"
#include "25_tma_async_group.cuh"
#include "34_fence_proxy_async.cuh"

// =============================================================================
// ours (originally guarded by PL_AGENTIC_SM100A || PL_AGENTIC_SM103A)
// =============================================================================

// ARCH: sm_100a


__global__ void k_tma(const __grid_constant__ CUtensorMap desc,
                      float* out, int rows, int cols) {
  __shared__ __align__(128) float smem[16 * 16];
  __shared__ __align__(16)  uint64_t mbar;

  if (threadIdx.x == 0) {
    mbarrier_init(smem_ptr_u32(&mbar), 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    mbarrier_arrive_expect_tx(smem_ptr_u32(&mbar), 16 * 16 * sizeof(float));
    tma_load_2d(smem_ptr_u32(smem), &desc, smem_ptr_u32(&mbar), 0, 0);
  }
  mbarrier_wait_parity(smem_ptr_u32(&mbar), 0);

  if (threadIdx.x < 16 * 16)
    out[threadIdx.x] = smem[threadIdx.x];
  __syncthreads();
  if (threadIdx.x == 0)
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n"
                 :: "r"(smem_ptr_u32(&mbar)));
}

static int run_ours() {
  /* (orig args dropped) */
  const int R = 16, C = 16;
  std::vector<float> hIn(R * C);
  for (int i = 0; i < R * C; ++i) hIn[i] = (float)i * 0.125f;
  float* dIn = nullptr; CUDA_CHECK(cudaMalloc(&dIn, R * C * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dIn, hIn.data(), R * C * sizeof(float),
                         cudaMemcpyHostToDevice));

  CUtensorMap desc;
  CUDA_CHECK(make_tma_2d_tiled(&desc, dIn, R, C, R, C,
                                 sizeof(float),
                                 CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                                 CU_TENSOR_MAP_SWIZZLE_NONE));

  float* dOut = nullptr;
  CUDA_CHECK(cudaMalloc(&dOut, R * C * sizeof(float)));
  CUDA_CHECK(cudaMemset(dOut, 0, R * C * sizeof(float)));

  k_tma<<<1, 256>>>(desc, dOut, R, C);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> hOut(R * C);
  CUDA_CHECK(cudaMemcpy(hOut.data(), dOut, R * C * sizeof(float),
                         cudaMemcpyDeviceToHost));
  cudaFree(dIn); cudaFree(dOut);

  int fails = 0;
  for (int i = 0; i < R * C; ++i) {
    if (hOut[i] != hIn[i]) {
      if (fails < 4)
        fprintf(stderr, "  [%d] got %g, want %g\n", i, hOut[i], hIn[i]);
      ++fails;
    }
  }
  printf("tma_load_2d 16x16 FP32 : fails = %d / %d\n", fails, R * C);
  if (fails) FAIL("tma_load round-trip mismatch");
  PASS();
}

// =============================================================================
// theirs (originally guarded by PL_AGENTIC_SM90A)
// =============================================================================

// Runtime test: TMA 2D load (GMEM -> SMEM) -- correctness + timing
// Build: nvcc -gencode arch=compute_90a,code=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr
//        -DNDEBUG -lineinfo -I primitives -I tests -lcuda -o build/18_test tests/18_tma_load_test.cu
//
// Approach: TMA load with SWIZZLE_128B stores the tile into SMEM with a
// hardware-specific address swizzle. To verify without manually decoding
// the swizzle, we round-trip through TMA: load with SWIZZLE_128B into SMEM,
// then TMA store with the *same* SWIZZLE_128B from SMEM into a separate
// GMEM region. The swizzle cancels in the round-trip so the destination
// GMEM should equal the source GMEM.
//
// We also include a SWIZZLE_NONE variant that lets us directly compare
// SMEM (via a cooperative st.global) to the source to sanity-check the
// basic load path.
//
// mbarrier setup:
//   1 thread (leader) inits the mbarrier, issues arrive.expect_tx with
//   the tile byte count, then issues cp.async.bulk.tensor.2d (TMA load).
//   All threads then spin on mbarrier.try_wait.parity until the async-proxy
//   completion writes back.

static constexpr int TILE_ROWS = 64;
static constexpr int TILE_COLS = 64;
static constexpr int TILE_ELEMS = TILE_ROWS * TILE_COLS;
static constexpr int TILE_BYTES = TILE_ELEMS * (int)sizeof(half);

extern __shared__ __align__(128) char smem_buf[];

// Round-trip kernel: TMA load (src, SWIZZLE_128B) -> SMEM -> TMA store
// (dst, SWIZZLE_128B). The swizzle round-trips transparently.
__global__ void tma_load_then_store_kernel(
    const __grid_constant__ CUtensorMap tmap_in,
    const __grid_constant__ CUtensorMap tmap_out)
{
    half* smem_tile    = reinterpret_cast<half*>(smem_buf);
    uint64_t* mbar     = reinterpret_cast<uint64_t*>(smem_buf + TILE_BYTES);
    int tid = threadIdx.x;

    if (tid == 0) {
        uint32_t mbar_addr = smem_ptr_u32(mbar);
        // Single arrival expected (the TMA completion).
        asm volatile("mbarrier.init.shared.b64 [%0], %1;\n"
                     :: "r"(mbar_addr), "r"(1) : "memory");
        // Register expected byte-count for async-proxy transfer.
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n"
                     :: "r"(mbar_addr), "r"((uint32_t)TILE_BYTES) : "memory");
        // Issue TMA load -- completion writes TILE_BYTES into the mbarrier tx counter.
        tma_load_2d_cta(&tmap_in,
                        smem_ptr_u32(smem_tile),
                        mbar_addr,
                        /*coord_x=*/0, /*coord_y=*/0);
    }
    __syncthreads();

    // All threads wait for the TMA to complete (phase 0 parity).
    uint32_t mbar_addr = smem_ptr_u32(mbar);
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "WAIT_LOOP_18:\n"
        "  mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
        "  @!p bra WAIT_LOOP_18;\n"
        "}\n"
        :: "r"(mbar_addr), "r"(0u));

    // TMA-completion gives implicit async->generic fence for the loaded data
    // (we don't read via generic proxy here; we re-issue a TMA store, which
    // is also async-proxy, so no extra fence required).
    if (tid == 0) {
        tma_store_2d(&tmap_out, 0, 0, smem_ptr_u32(smem_tile));
        tma_store_commit_group();
        tma_store_wait_group<0>();
    }
}

// L2-hint variant (exercises tma_load_2d_cta_l2hint)
__global__ void tma_load_l2hint_then_store_kernel(
    const __grid_constant__ CUtensorMap tmap_in,
    const __grid_constant__ CUtensorMap tmap_out)
{
    half* smem_tile    = reinterpret_cast<half*>(smem_buf);
    uint64_t* mbar     = reinterpret_cast<uint64_t*>(smem_buf + TILE_BYTES);
    int tid = threadIdx.x;

    if (tid == 0) {
        uint32_t mbar_addr = smem_ptr_u32(mbar);
        asm volatile("mbarrier.init.shared.b64 [%0], %1;\n"
                     :: "r"(mbar_addr), "r"(1) : "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n"
                     :: "r"(mbar_addr), "r"((uint32_t)TILE_BYTES) : "memory");
        tma_load_2d_cta_l2hint(&tmap_in,
                               smem_ptr_u32(smem_tile),
                               mbar_addr,
                               0, 0,
                               /*cache_policy=*/0ull);
    }
    __syncthreads();
    uint32_t mbar_addr = smem_ptr_u32(mbar);
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "WAIT_LOOP_18B:\n"
        "  mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
        "  @!p bra WAIT_LOOP_18B;\n"
        "}\n"
        :: "r"(mbar_addr), "r"(0u));
    if (tid == 0) {
        tma_store_2d(&tmap_out, 0, 0, smem_ptr_u32(smem_tile));
        tma_store_commit_group();
        tma_store_wait_group<0>();
    }
}

// SWIZZLE_NONE sanity check: TMA load with no swizzle, then each thread
// copies SMEM -> output GMEM directly via generic proxy. Verifies that
// the load path works correctly for simple (non-swizzled) layouts.
__global__ void tma_load_noswizzle_kernel(
    const __grid_constant__ CUtensorMap tmap_in,
    half* __restrict__ out)
{
    half* smem_tile    = reinterpret_cast<half*>(smem_buf);
    uint64_t* mbar     = reinterpret_cast<uint64_t*>(smem_buf + TILE_BYTES);
    int tid = threadIdx.x;

    if (tid == 0) {
        uint32_t mbar_addr = smem_ptr_u32(mbar);
        asm volatile("mbarrier.init.shared.b64 [%0], %1;\n"
                     :: "r"(mbar_addr), "r"(1) : "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n"
                     :: "r"(mbar_addr), "r"((uint32_t)TILE_BYTES) : "memory");
        tma_load_2d_cta(&tmap_in,
                        smem_ptr_u32(smem_tile),
                        mbar_addr,
                        0, 0);
    }
    __syncthreads();

    uint32_t mbar_addr = smem_ptr_u32(mbar);
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "WAIT_LOOP_18C:\n"
        "  mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
        "  @!p bra WAIT_LOOP_18C;\n"
        "}\n"
        :: "r"(mbar_addr), "r"(0u));

    // Cooperative SMEM -> GMEM via generic-proxy ld.shared + st.global.
    for (int i = tid; i < TILE_ELEMS; i += blockDim.x) {
        out[i] = smem_tile[i];
    }
}

static int run_theirs() {
  /* (orig args dropped) */
    CUDA_CHECK(cudaFree(0));
    bool all_pass = true;

    half *d_src, *d_dst_swz, *d_dst_swz2, *d_dst_plain;
    CUDA_CHECK(cudaMalloc(&d_src,       TILE_BYTES));
    CUDA_CHECK(cudaMalloc(&d_dst_swz,   TILE_BYTES));
    CUDA_CHECK(cudaMalloc(&d_dst_swz2,  TILE_BYTES));
    CUDA_CHECK(cudaMalloc(&d_dst_plain, TILE_BYTES));

    // Host pattern
    std::vector<half> h_src(TILE_ELEMS);
    for (int r = 0; r < TILE_ROWS; r++)
        for (int c = 0; c < TILE_COLS; c++)
            h_src[r * TILE_COLS + c] = __float2half((float)(r * 64 + c));
    CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), TILE_BYTES, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_dst_swz,   0, TILE_BYTES));
    CUDA_CHECK(cudaMemset(d_dst_swz2,  0, TILE_BYTES));
    CUDA_CHECK(cudaMemset(d_dst_plain, 0, TILE_BYTES));

    // Tensormaps:
    //  - tmap_in_swz  : SWIZZLE_128B on d_src
    //  - tmap_out_swz : SWIZZLE_128B on d_dst_swz (matches load swizzle)
    //  - tmap_out_swz2: SWIZZLE_128B on d_dst_swz2 (for L2-hint variant)
    //  - tmap_in_none : SWIZZLE_NONE on d_src (for direct verification)
    CUtensorMap tmap_in_swz{}, tmap_out_swz{}, tmap_out_swz2{}, tmap_in_none{};
    create_tma_2d_f16(&tmap_in_swz,   d_src,      TILE_ROWS, TILE_COLS, TILE_ROWS, TILE_COLS);
    create_tma_2d_f16(&tmap_out_swz,  d_dst_swz,  TILE_ROWS, TILE_COLS, TILE_ROWS, TILE_COLS);
    create_tma_2d_f16(&tmap_out_swz2, d_dst_swz2, TILE_ROWS, TILE_COLS, TILE_ROWS, TILE_COLS);
    create_tma_2d_desc(&tmap_in_none, d_src, TILE_ROWS, TILE_COLS, TILE_ROWS, TILE_COLS,
                       2, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, CU_TENSOR_MAP_SWIZZLE_NONE);

    size_t smem_bytes = TILE_BYTES + 16;  // tile + mbarrier
    CUDA_CHECK(cudaFuncSetAttribute(tma_load_then_store_kernel,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)smem_bytes));
    CUDA_CHECK(cudaFuncSetAttribute(tma_load_l2hint_then_store_kernel,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)smem_bytes));
    CUDA_CHECK(cudaFuncSetAttribute(tma_load_noswizzle_kernel,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)smem_bytes));

    // --- Test 1: SWIZZLE_128B round-trip (load + store) ------------------
    GpuTimer t1;
    t1.begin();
    tma_load_then_store_kernel<<<1, 128, smem_bytes>>>(tmap_in_swz, tmap_out_swz);
    t1.end();
    CUDA_CHECK(cudaGetLastError());
    std::vector<half> h_dst_swz(TILE_ELEMS);
    CUDA_CHECK(cudaMemcpy(h_dst_swz.data(), d_dst_swz, TILE_BYTES, cudaMemcpyDeviceToHost));
    int mism1 = 0;
    for (int i = 0; i < TILE_ELEMS; i++) {
        if (__half2float(h_dst_swz[i]) != __half2float(h_src[i])) {
            if (mism1 < 5)
                printf("  [swz rt] [%d] got=%.1f exp=%.1f\n",
                       i, __half2float(h_dst_swz[i]), __half2float(h_src[i]));
            mism1++;
        }
    }
    if (mism1 == 0) printf("  swizzle128 round-trip: OK  (%.3f ms)\n", t1.elapsed_ms());
    else { printf("  swizzle128 round-trip: FAIL (%d mismatches)\n", mism1); all_pass = false; }

    // --- Test 2: L2-hint variant, same round-trip ------------------------
    GpuTimer t2;
    t2.begin();
    tma_load_l2hint_then_store_kernel<<<1, 128, smem_bytes>>>(tmap_in_swz, tmap_out_swz2);
    t2.end();
    CUDA_CHECK(cudaGetLastError());
    std::vector<half> h_dst_swz2(TILE_ELEMS);
    CUDA_CHECK(cudaMemcpy(h_dst_swz2.data(), d_dst_swz2, TILE_BYTES, cudaMemcpyDeviceToHost));
    int mism2 = 0;
    for (int i = 0; i < TILE_ELEMS; i++) {
        if (__half2float(h_dst_swz2[i]) != __half2float(h_src[i])) {
            if (mism2 < 5)
                printf("  [l2 rt] [%d] got=%.1f exp=%.1f\n",
                       i, __half2float(h_dst_swz2[i]), __half2float(h_src[i]));
            mism2++;
        }
    }
    if (mism2 == 0) printf("  l2hint   round-trip: OK  (%.3f ms)\n", t2.elapsed_ms());
    else { printf("  l2hint   round-trip: FAIL (%d mismatches)\n", mism2); all_pass = false; }

    // --- Test 3: SWIZZLE_NONE load, direct SMEM->GMEM verify -------------
    GpuTimer t3;
    t3.begin();
    tma_load_noswizzle_kernel<<<1, 128, smem_bytes>>>(tmap_in_none, d_dst_plain);
    t3.end();
    CUDA_CHECK(cudaGetLastError());
    std::vector<half> h_dst_plain(TILE_ELEMS);
    CUDA_CHECK(cudaMemcpy(h_dst_plain.data(), d_dst_plain, TILE_BYTES, cudaMemcpyDeviceToHost));
    int mism3 = 0;
    for (int i = 0; i < TILE_ELEMS; i++) {
        if (__half2float(h_dst_plain[i]) != __half2float(h_src[i])) {
            if (mism3 < 5)
                printf("  [no swz] [%d] got=%.1f exp=%.1f\n",
                       i, __half2float(h_dst_plain[i]), __half2float(h_src[i]));
            mism3++;
        }
    }
    if (mism3 == 0) printf("  no-swizzle load path:  OK  (%.3f ms)\n", t3.elapsed_ms());
    else { printf("  no-swizzle load path:  FAIL (%d mismatches)\n", mism3); all_pass = false; }

    // --- Perf ------------------------------------------------------------
    GpuTimer tp;
    const int ITERS = 200;
    tp.begin();
    for (int i = 0; i < ITERS; i++) {
        tma_load_then_store_kernel<<<1, 128, smem_bytes>>>(tmap_in_swz, tmap_out_swz);
    }
    tp.end();
    printf("  perf (load+store round-trip): %.2f us/launch\n",
           tp.elapsed_ms() * 1000.0f / ITERS);

    cudaFree(d_src); cudaFree(d_dst_swz); cudaFree(d_dst_swz2); cudaFree(d_dst_plain);
    if (all_pass) { PASS(); return 0; }
    else          { FAIL("subtests failed"); return 1; }
}


int main() {
  int rc_ours   = run_ours();
  int rc_theirs = run_theirs();
  return (rc_ours == 0 && rc_theirs == 0) ? 0 : 1;
}
