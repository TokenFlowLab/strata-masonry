#if defined(PL_AGENTIC_SM90A)
// ARCH: sm_90a
// 89_load_warp_hopper_test.cu -- runtime test for load warp hopper
//
// Builds an 8-tile source GMEM buffer, runs a 2-warp test kernel (warp 0
// = load_warp_hopper_block, warp 1 = consumer), then verifies the output
// equals the corresponding source slabs.

#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "test_utils.cuh"
#include "../src/blocks/89_load_warp_hopper.cuh"

namespace block89 {
static constexpr int TILE_M     = 64;
static constexpr int TILE_K     = 64;
static constexpr int NUM_STAGES = 4;
static constexpr int TILE_ELEMS = TILE_M * TILE_K;
static constexpr int TILE_BYTES = TILE_ELEMS * (int)sizeof(half);
}

extern __shared__ __align__(128) char block89_smem_buf[];

__global__ void load_warp_hopper_test_kernel(
    const __grid_constant__ CUtensorMap tma_a,
    half* __restrict__ out,
    int num_k_tiles) {
  using namespace block89;
  int tid  = threadIdx.x;
  int warp = tid / 32;
  int lane = tid % 32;

  char*      smem_a   = block89_smem_buf;
  uint64_t*  full_mb  = reinterpret_cast<uint64_t*>(smem_a + NUM_STAGES * TILE_BYTES);
  uint64_t*  empty_mb = full_mb + NUM_STAGES;

  if (tid == 0) {
    uint32_t fb = static_cast<uint32_t>(__cvta_generic_to_shared(full_mb));
    uint32_t eb = static_cast<uint32_t>(__cvta_generic_to_shared(empty_mb));
    #pragma unroll
    for (int s = 0; s < NUM_STAGES; ++s) {
      mbarrier_init(fb + s * 8, 1);
      mbarrier_init(eb + s * 8, 1);
    }
  }
  __syncthreads();

  if (warp == 0) {
    load_warp_hopper_block<NUM_STAGES, TILE_BYTES, TILE_K>(
        tma_a, smem_a, full_mb, empty_mb, num_k_tiles, lane);
  } else if (warp == 1) {
    load_warp_hopper_consumer_block<NUM_STAGES, TILE_M, TILE_K, TILE_BYTES>(
        out, smem_a, full_mb, empty_mb, num_k_tiles, lane);
  }
}

int main() {
    using namespace block89;
    CUDA_CHECK(cudaFree(0));
    bool all_pass = true;

    constexpr int NUM_K_TILES = 8;
    constexpr int SRC_ROWS    = TILE_M;
    constexpr int SRC_COLS    = TILE_K * NUM_K_TILES;
    constexpr int DST_ELEMS   = TILE_ELEMS * NUM_K_TILES;

    std::vector<half> h_src(SRC_ROWS * SRC_COLS);
    for (int r = 0; r < SRC_ROWS; r++)
        for (int c = 0; c < SRC_COLS; c++)
            h_src[r * SRC_COLS + c] = __float2half((float)(r * 1024 + c));

    half *d_src, *d_out;
    CUDA_CHECK(cudaMalloc(&d_src, (size_t)SRC_ROWS * SRC_COLS * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)DST_ELEMS * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(d_src, h_src.data(),
                          (size_t)SRC_ROWS * SRC_COLS * sizeof(half),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, (size_t)DST_ELEMS * sizeof(half)));

    CUtensorMap tma_a{};
    create_tma_2d_desc(&tma_a, d_src, SRC_ROWS, SRC_COLS, TILE_M, TILE_K,
                       /*elem_bytes=*/2,
                       CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
                       CU_TENSOR_MAP_SWIZZLE_NONE);

    size_t smem_bytes = (size_t)NUM_STAGES * TILE_BYTES + 2 * NUM_STAGES * sizeof(uint64_t);
    CUDA_CHECK(cudaFuncSetAttribute(load_warp_hopper_test_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)smem_bytes));

    GpuTimer t;
    t.begin();
    load_warp_hopper_test_kernel<<<1, 64, smem_bytes>>>(tma_a, d_out, NUM_K_TILES);
    t.end();
    CUDA_CHECK(cudaGetLastError());

    std::vector<half> h_out(DST_ELEMS);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, (size_t)DST_ELEMS * sizeof(half), cudaMemcpyDeviceToHost));

    int total_mismatches = 0;
    for (int k = 0; k < NUM_K_TILES; k++) {
        int mism_this = 0;
        for (int r = 0; r < TILE_M; r++) {
            for (int c = 0; c < TILE_K; c++) {
                float got = __half2float(h_out[k * TILE_ELEMS + r * TILE_K + c]);
                float ref = __half2float(h_src[r * SRC_COLS + k * TILE_K + c]);
                if (got != ref) {
                    if (total_mismatches < 5)
                        printf("  tile=%d (r=%d,c=%d) got=%.1f ref=%.1f\n",
                               k, r, c, got, ref);
                    total_mismatches++; mism_this++;
                }
            }
        }
        if (mism_this == 0)
            printf("  tile[%d] OK\n", k);
    }
    if (total_mismatches == 0) {
        printf("  4-stage TMA round-trip: OK (%d tiles, %.3f ms)\n", NUM_K_TILES, t.elapsed_ms());
    } else {
        printf("  4-stage TMA round-trip: FAIL (%d total mismatches)\n", total_mismatches);
        all_pass = false;
    }

    GpuTimer tp;
    const int ITERS = 100;
    tp.begin();
    for (int i = 0; i < ITERS; i++) {
        load_warp_hopper_test_kernel<<<1, 64, smem_bytes>>>(tma_a, d_out, NUM_K_TILES);
    }
    tp.end();
    printf("  perf: %.2f us/launch (%d stages, %d k-tiles, %dx%d fp16)\n",
           tp.elapsed_ms() * 1000.0f / ITERS, NUM_STAGES, NUM_K_TILES, TILE_M, TILE_K);

    cudaFree(d_src); cudaFree(d_out);
    if (all_pass) { PASS(); return 0; }
    else { FAIL("TMA round-trip mismatches"); return 1; }
}

#endif  // PL_AGENTIC_SM90A
