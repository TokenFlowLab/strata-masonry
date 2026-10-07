// ARCH: sm_100a
// 74_tma_store_2d_scatter4_test.cu -- TMA scatter4 writes 4 SMEM rows to 4
// non-contiguous GMEM rows; host verifies the targeted GMEM rows match the
// SMEM source and untouched rows remain zero.
//
// Setup:
//   SMEM src: (4, C) fp32, src[i, c] = (i + 100) * 10000 + c so each row
//     is distinct and easy to spot in GMEM.
//   D: (R, C) fp32 GMEM, initialized to 0.
//   Tensormap on D: box = (1, C), SWIZZLE_NONE, FLOAT32.
//   Scatter indices: {3, 0, 9, 14} -- SMEM row i lands at D row scatter[i].
//
// Verify:
//   D[scatter[i], c] == src[i, c] for i in 0..3, c in 0..C-1.
//   D[other rows]    == 0.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "test_utils.cuh"
#include "../primitives/74_tma_store_2d_scatter4.cuh"
#include "../primitives/23_tma_tensormap.cuh"
#include "../primitives/25_tma_async_group.cuh"
#include "../primitives/34_fence_proxy_async.cuh"
#include "../primitives/70_smem_ptr.cuh"

constexpr int R = 16;
constexpr int C = 16;
constexpr int N_SCATTER = 4;
__device__ __constant__ int d_scatter_idx[N_SCATTER] = { 3, 0, 9, 14 };

__global__ void k_scatter4(const __grid_constant__ CUtensorMap desc) {
  __shared__ __align__(128) float smem[N_SCATTER * C];

  // One thread fills SMEM with the row-distinct pattern.
  if (threadIdx.x == 0) {
    for (int i = 0; i < N_SCATTER; ++i)
      for (int c = 0; c < C; ++c)
        smem[i * C + c] = (float)((i + 100) * 10000 + c);
  }
  __syncthreads();

  // SMEM was written via generic proxy; fence before async-proxy read.
  if (threadIdx.x == 0) {
    fence_proxy_async_shared_cta();
    tma_store_2d_scatter4(&desc,
                          /*col=*/0,
                          d_scatter_idx[0], d_scatter_idx[1],
                          d_scatter_idx[2], d_scatter_idx[3],
                          smem_ptr_u32(smem));
    cp_async_bulk_commit_group();
    cp_async_bulk_wait_group<0>();
  }
}

int main() {
  CUDA_CHECK(cudaFree(0));

  float* dD = nullptr;
  CUDA_CHECK(cudaMalloc(&dD, R * C * sizeof(float)));
  CUDA_CHECK(cudaMemset(dD, 0, R * C * sizeof(float)));

  CUtensorMap desc;
  CUDA_CHECK(make_tma_2d_tiled(&desc, dD, R, C,
                               /*box_rows=*/1, /*box_cols=*/C,
                               sizeof(float),
                               CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                               CU_TENSOR_MAP_SWIZZLE_NONE));

  k_scatter4<<<1, 32>>>(desc);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> hD(R * C);
  CUDA_CHECK(cudaMemcpy(hD.data(), dD, R * C * sizeof(float),
                        cudaMemcpyDeviceToHost));
  cudaFree(dD);

  const int scatter_idx[N_SCATTER] = { 3, 0, 9, 14 };
  int fails = 0;
  // Scattered rows must match SMEM source.
  for (int i = 0; i < N_SCATTER; ++i) {
    int dst_row = scatter_idx[i];
    for (int c = 0; c < C; ++c) {
      float got  = hD[dst_row * C + c];
      float want = (float)((i + 100) * 10000 + c);
      if (got != want) {
        if (fails < 8)
          fprintf(stderr, "  D[%d, %d] = %g, want %g (src i=%d)\n",
                  dst_row, c, got, want, i);
        ++fails;
      }
    }
  }
  // Untouched rows must be zero.
  bool is_target[R] = { false };
  for (int i = 0; i < N_SCATTER; ++i) is_target[scatter_idx[i]] = true;
  for (int r = 0; r < R; ++r) {
    if (is_target[r]) continue;
    for (int c = 0; c < C; ++c) {
      if (hD[r * C + c] != 0.0f) {
        if (fails < 16)
          fprintf(stderr, "  D[%d, %d] = %g, want 0 (untouched row)\n",
                  r, c, hD[r * C + c]);
        ++fails;
      }
    }
  }
  printf("tma_store_2d_scatter4 (R=%d, C=%d, picks={%d,%d,%d,%d}): fails = %d / %d\n",
         R, C, scatter_idx[0], scatter_idx[1], scatter_idx[2], scatter_idx[3],
         fails, R * C);
  if (fails) FAIL("scatter4 mismatch");
  PASS();
}
