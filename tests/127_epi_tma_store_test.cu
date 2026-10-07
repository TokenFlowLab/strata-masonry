// ARCH: sm_90a
// 127_epi_tma_store_test.cu -- write pattern + TMA store + verify.
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
#include "../src/primitives/23_tma_tensormap.cuh"
#include "../src/composites/127_epi_tma_store.cuh"
#include <cuda_fp16.h>
#include "23_tma_tensormap.cuh"
#include "25_tma_async_group.cuh"
#include "127_epi_tma_store.cuh"

// =============================================================================
// ours (originally guarded by PL_AGENTIC_SM100A || PL_AGENTIC_SM103A)
// =============================================================================

// ARCH: sm_90a


__global__ void k(const __grid_constant__ CUtensorMap t) {
  __shared__ __align__(128) float smem[16 * 16];
  if (threadIdx.x < 16 * 16) smem[threadIdx.x] = (float)threadIdx.x * 0.25f;
  __syncthreads();
  epi_tma_store_2d<1>(256, &t, 0, 0, smem_ptr_u32(smem));
}

static int run_ours() {
  /* (orig args dropped) */
  float* d = nullptr; CUDA_CHECK(cudaMalloc(&d, 16 * 16 * 4));
  CUDA_CHECK(cudaMemset(d, 0, 16 * 16 * 4));
  CUtensorMap t;
  CUDA_CHECK(make_tma_2d_tiled(&t, d, 16, 16, 16, 16, sizeof(float),
                                 CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                                 CU_TENSOR_MAP_SWIZZLE_NONE));
  k<<<1, 256>>>(t);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> h(16 * 16);
  CUDA_CHECK(cudaMemcpy(h.data(), d, 16 * 16 * 4, cudaMemcpyDeviceToHost));
  cudaFree(d);
  int fails = 0;
  for (int i = 0; i < 16 * 16; ++i)
    if (h[i] != (float)i * 0.25f) ++fails;
  printf("epi_tma_store_2d : fails = %d / %d\n", fails, 16 * 16);
  if (fails) FAIL("content mismatch");
  PASS();
}

// =============================================================================
// theirs (originally guarded by PL_AGENTIC_SM90A)
// =============================================================================

// Runtime test: 127_epi_tma_store -- bar.sync + fence + TMA store + commit + wait.
//
// Strategy: 128 threads write a known pattern into SMEM via plain st.shared,
// then the composite handles the fence + TMA store + commit + wait. Host
// verifies GMEM matches source pattern.
//
// We use SWIZZLE_NONE tensormap so SMEM bytes map 1:1 to GMEM bytes, making
// verification straightforward.
//
// Two sub-tests:
//   1) epi_tma_store_full: bar.sync + fence + store + commit + wait (the
//      composite)
//   2) epi_tma_store_2d_elected: just fence + store (caller commits/waits)


static constexpr int TILE_ROWS = 64;
static constexpr int TILE_COLS = 64;
static constexpr int TILE_ELEMS = TILE_ROWS * TILE_COLS;
static constexpr int TILE_BYTES = TILE_ELEMS * (int)sizeof(half);

extern __shared__ __align__(128) char smem_buf_78[];

// Kernel: 128 threads fill SMEM tile with known values, call epi_tma_store_full.
__global__ void kernel_epi_full(const __grid_constant__ CUtensorMap tmap_out) {
    half* smem_tile = reinterpret_cast<half*>(smem_buf_78);
    int tid = threadIdx.x;
    // Write known pattern: element [r*COLS+c] = (r * 64 + c) * 0.25f.
    for (int i = tid; i < TILE_ELEMS; i += blockDim.x) {
        int r = i / TILE_COLS;
        int c = i % TILE_COLS;
        smem_tile[i] = __float2half((float)((r * 64 + c) & 0x7F));
    }
    // Composite: bar.sync + fence + TMA store + commit + wait.
    bool is_elected = (tid == 0);
    uint32_t src = smem_ptr_u32(smem_tile);
    epi_tma_store_full<1>(&tmap_out, 0, 0, src, blockDim.x, is_elected);
}

// Kernel: same pattern but only uses epi_tma_store_2d_elected (elected thread)
// + caller-managed __syncthreads + commit + wait.
__global__ void kernel_epi_elected(const __grid_constant__ CUtensorMap tmap_out) {
    half* smem_tile = reinterpret_cast<half*>(smem_buf_78);
    int tid = threadIdx.x;
    for (int i = tid; i < TILE_ELEMS; i += blockDim.x) {
        int r = i / TILE_COLS;
        int c = i % TILE_COLS;
        smem_tile[i] = __float2half((float)((r * 64 + c) & 0x7F));
    }
    __syncthreads();  // all threads' stores must land before fence+store
    if (tid == 0) {
        uint32_t src = smem_ptr_u32(smem_tile);
        epi_tma_store_2d_elected(&tmap_out, 0, 0, src);
        tma_store_commit_group();
        tma_store_wait_group<0>();
    }
}

static int run_theirs() {
  /* (orig args dropped) */
    CUDA_CHECK(cudaFree(0));
    bool all_pass = true;

    half *d_out_full, *d_out_elected;
    CUDA_CHECK(cudaMalloc(&d_out_full,    TILE_BYTES));
    CUDA_CHECK(cudaMalloc(&d_out_elected, TILE_BYTES));
    CUDA_CHECK(cudaMemset(d_out_full,    0xaa, TILE_BYTES));  // sentinel
    CUDA_CHECK(cudaMemset(d_out_elected, 0xaa, TILE_BYTES));

    // SWIZZLE_NONE tensormaps for direct GMEM <-> SMEM mapping.
    CUtensorMap tmap_full{}, tmap_elected{};
    create_tma_2d_desc(&tmap_full,    d_out_full,    TILE_ROWS, TILE_COLS, TILE_ROWS, TILE_COLS,
                       2, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, CU_TENSOR_MAP_SWIZZLE_NONE);
    create_tma_2d_desc(&tmap_elected, d_out_elected, TILE_ROWS, TILE_COLS, TILE_ROWS, TILE_COLS,
                       2, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, CU_TENSOR_MAP_SWIZZLE_NONE);

    size_t smem_bytes = TILE_BYTES;
    CUDA_CHECK(cudaFuncSetAttribute(kernel_epi_full,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)smem_bytes));
    CUDA_CHECK(cudaFuncSetAttribute(kernel_epi_elected,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)smem_bytes));

    // -- Test 1: epi_tma_store_full --
    {
        GpuTimer t;
        t.begin();
        kernel_epi_full<<<1, 128, smem_bytes>>>(tmap_full);
        t.end();
        CUDA_CHECK(cudaGetLastError());
        std::vector<half> h_out(TILE_ELEMS);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out_full, TILE_BYTES, cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int i = 0; i < TILE_ELEMS; i++) {
            int r = i / TILE_COLS;
            int c = i % TILE_COLS;
            float exp = (float)((r * 64 + c) & 0x7F);
            if (__half2float(h_out[i]) != exp) {
                if (bad < 5) printf("    full [%d r=%d c=%d] got=%.3f exp=%.3f\n",
                                     i, r, c, __half2float(h_out[i]), exp);
                bad++;
            }
        }
        if (bad == 0) printf("  epi_tma_store_full: OK (%.3f ms, %d elts)\n",
                              t.elapsed_ms(), TILE_ELEMS);
        else { printf("  epi_tma_store_full: FAIL (%d mismatches)\n", bad); all_pass = false; }
    }

    // -- Test 2: epi_tma_store_2d_elected (+ caller-managed commit/wait) --
    {
        GpuTimer t;
        t.begin();
        kernel_epi_elected<<<1, 128, smem_bytes>>>(tmap_elected);
        t.end();
        CUDA_CHECK(cudaGetLastError());
        std::vector<half> h_out(TILE_ELEMS);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out_elected, TILE_BYTES, cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int i = 0; i < TILE_ELEMS; i++) {
            int r = i / TILE_COLS;
            int c = i % TILE_COLS;
            float exp = (float)((r * 64 + c) & 0x7F);
            if (__half2float(h_out[i]) != exp) {
                if (bad < 5) printf("    elected [%d] got=%.3f exp=%.3f\n",
                                     i, __half2float(h_out[i]), exp);
                bad++;
            }
        }
        if (bad == 0) printf("  epi_tma_store_2d_elected: OK (%.3f ms)\n", t.elapsed_ms());
        else { printf("  epi_tma_store_2d_elected: FAIL (%d mismatches)\n", bad); all_pass = false; }
    }

    // -- Perf --
    GpuTimer tp;
    const int ITERS = 200;
    tp.begin();
    for (int i = 0; i < ITERS; i++)
        kernel_epi_full<<<1, 128, smem_bytes>>>(tmap_full);
    tp.end();
    printf("  perf (full store, 64x64 FP16): %.2f us/launch\n",
           tp.elapsed_ms() * 1000.0f / ITERS);

    cudaFree(d_out_full);
    cudaFree(d_out_elected);

    if (all_pass) { PASS(); return 0; }
    else          { FAIL("some subtests failed"); return 1; }
}


int main() {
  int rc_ours   = run_ours();
  int rc_theirs = run_theirs();
  return (rc_ours == 0 && rc_theirs == 0) ? 0 : 1;
}
