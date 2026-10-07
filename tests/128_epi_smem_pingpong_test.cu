// ARCH: sm_90a
// 128_epi_smem_pingpong_test.cu -- swap alternates between two SMEM bases.
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
#include "../src/composites/128_epi_smem_pingpong.cuh"
#include <cuda_fp16.h>
#include "22_tma_store.cuh"
#include "23_tma_tensormap.cuh"
#include "25_tma_async_group.cuh"
#include "34_fence_proxy_async.cuh"
#include "128_epi_smem_pingpong.cuh"

// =============================================================================
// ours
// =============================================================================

// ARCH: sm_90a


__global__ void k(uint32_t* out) {
  __shared__ __align__(128) uint8_t buf0[64];
  __shared__ __align__(128) uint8_t buf1[64];
  EpiSmemPingpong p;
  p.init(smem_ptr_u32(buf0), smem_ptr_u32(buf1));
  out[0] = p.current();
  p.swap();
  out[1] = p.current();
  p.swap();
  out[2] = p.current();
}

static int run_ours() {
  uint32_t* d = nullptr; CUDA_CHECK(cudaMalloc(&d, 12));
  k<<<1, 1>>>(d);
  CUDA_CHECK(cudaDeviceSynchronize());
  uint32_t h[3];
  CUDA_CHECK(cudaMemcpy(h, d, 12, cudaMemcpyDeviceToHost));
  cudaFree(d);
  printf("epi pingpong swap : %u -> %u -> %u\n", h[0], h[1], h[2]);
  if (h[0] == h[1]) FAIL("swap did not change the active buffer");
  if (h[0] != h[2]) FAIL("double-swap did not restore the first buffer");
  PASS();
}

// =============================================================================
// theirs
// =============================================================================

// Runtime test: 128_epi_smem_pingpong -- double-buffered epilogue SMEM.
//
// Strategy: use N sub-tiles. Each sub-tile:
//   - waits for the current-stage SMEM to be free
//   - fills SMEM with a sub-tile-specific pattern
//   - issues a TMA store of that SMEM into GMEM sub-region i
//   - commits + swaps
// Then drain. Verify every GMEM sub-region got the right pattern.
//
// This exercises the EpiPingPong's tma_store_wait_group<1> and <0> calls
// (via commit_and_swap / wait_write_ready / drain).
//
// Use SWIZZLE_NONE tensormap for easy verification.


// SMEM sub-tile is 32x32 FP16 = 2048 bytes; ping-pong buffer = 4096 bytes.
static constexpr int SUB_ROWS  = 32;
static constexpr int SUB_COLS  = 32;
static constexpr int SUB_ELEMS = SUB_ROWS * SUB_COLS;
// static constexpr int SUB_BYTES = SUB_ELEMS * (int)sizeof(half);  // unused: 2048
static constexpr int N_SUBTILES = 6;                                // number of stages
static constexpr int EPI_SMEM_BYTES_T = 2048;                       // template param for ping-pong

extern __shared__ __align__(128) char smem_79[];

// Kernel: N_SUBTILES worth of ping-pong store. Output GMEM is laid out as
// N_SUBTILES x (32x32) so each sub-tile has its own tensormap origin.
__global__ void kernel_pingpong(const __grid_constant__ CUtensorMap tmap_out,
                                 int num_subtiles) {
    char* smem_base_char = smem_79;
    uint32_t smem_base = smem_ptr_u32(smem_base_char);
    EpiPingPong<EPI_SMEM_BYTES_T> pp(smem_base);

    int tid = threadIdx.x;
    for (int i = 0; i < num_subtiles; i++) {
        pp.wait_write_ready();
        // Fill current write-stage SMEM with pattern (r*32 + c + i*1000).
        uint32_t write_addr = pp.write_addr();
        half* smem_tile = reinterpret_cast<half*>(
            smem_base_char + (write_addr - smem_base));
        for (int k = tid; k < SUB_ELEMS; k += blockDim.x) {
            int r = k / SUB_COLS;
            int c = k % SUB_COLS;
            // Values in FP16 representable range.
            smem_tile[k] = __float2half((float)((r * 32 + c + i * 128) & 0x7F));
        }
        // bar.sync to ensure all writes land, then fence before TMA store.
        __syncthreads();
        if (tid == 0) {
            fence_proxy_async_shared_cta();
            tma_store_2d(&tmap_out, /*coord_x=*/0, /*coord_y=*/i * SUB_ROWS, write_addr);
            // commit + swap via the composite (all threads must call).
        }
        // commit_and_swap updates state; all threads should agree on
        // current_stage. But EpiPingPong is a local per-thread struct, so
        // state tracking is per-thread. Only the elected thread issues the
        // actual tma_store_commit_group (from inside commit_and_swap).
        // We gate commit_and_swap on the elected thread only to avoid
        // double-counting the commit group; other threads update local state
        // separately.
        if (tid == 0) {
            pp.commit_and_swap();
        } else {
            pp.current_stage ^= 1;
            pp.outstanding_stores++;
        }
    }
    __syncthreads();
    if (tid == 0) {
        pp.drain();
    }
}

static int run_theirs() {
    CUDA_CHECK(cudaFree(0));
    bool all_pass = true;

    // Output: 1 column of tiles stacked vertically. Full tensor = (N_SUBTILES * 32) x 32.
    const int OUT_ROWS = N_SUBTILES * SUB_ROWS;
    const int OUT_COLS = SUB_COLS;
    const int OUT_ELEMS = OUT_ROWS * OUT_COLS;
    const int OUT_BYTES = OUT_ELEMS * (int)sizeof(half);

    half* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, OUT_BYTES));
    CUDA_CHECK(cudaMemset(d_out, 0xaa, OUT_BYTES));

    // SWIZZLE_NONE tensormap: GMEM is (OUT_ROWS, OUT_COLS), box is (SUB_ROWS, SUB_COLS).
    CUtensorMap tmap_out{};
    create_tma_2d_desc(&tmap_out, d_out,
                       OUT_ROWS, OUT_COLS,
                       SUB_ROWS, SUB_COLS,
                       2, CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
                       CU_TENSOR_MAP_SWIZZLE_NONE);

    // SMEM budget: 2 x EPI_SMEM_BYTES (double buffer)
    size_t smem_bytes = 2 * EPI_SMEM_BYTES_T;
    CUDA_CHECK(cudaFuncSetAttribute(kernel_pingpong,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)smem_bytes));

    GpuTimer t;
    t.begin();
    kernel_pingpong<<<1, 128, smem_bytes>>>(tmap_out, N_SUBTILES);
    t.end();
    CUDA_CHECK(cudaGetLastError());

    std::vector<half> h_out(OUT_ELEMS);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, OUT_BYTES, cudaMemcpyDeviceToHost));

    int bad = 0;
    for (int i = 0; i < N_SUBTILES; i++) {
        for (int r = 0; r < SUB_ROWS; r++) {
            for (int c = 0; c < SUB_COLS; c++) {
                float exp = (float)((r * 32 + c + i * 128) & 0x7F);
                int idx = (i * SUB_ROWS + r) * OUT_COLS + c;
                float got = __half2float(h_out[idx]);
                if (got != exp) {
                    if (bad < 5) printf("    [sub=%d r=%d c=%d] got=%.1f exp=%.1f\n",
                                         i, r, c, got, exp);
                    bad++;
                }
            }
        }
    }
    if (bad == 0)
        printf("  pingpong (%d subtiles, 32x32 each): OK (%.3f ms)\n", N_SUBTILES, t.elapsed_ms());
    else {
        printf("  pingpong: FAIL (%d mismatches)\n", bad);
        all_pass = false;
    }

    // Perf
    GpuTimer tp;
    const int ITERS = 100;
    tp.begin();
    for (int i = 0; i < ITERS; i++)
        kernel_pingpong<<<1, 128, smem_bytes>>>(tmap_out, N_SUBTILES);
    tp.end();
    printf("  perf (%d subtiles via ping-pong): %.2f us/launch\n",
           N_SUBTILES, tp.elapsed_ms() * 1000.0f / ITERS);

    cudaFree(d_out);

    if (all_pass) { PASS(); return 0; }
    else          { FAIL("some subtests failed"); return 1; }
}


int main() {
  int rc_ours   = run_ours();
  int rc_theirs = run_theirs();
  return (rc_ours == 0 && rc_theirs == 0) ? 0 : 1;
}
