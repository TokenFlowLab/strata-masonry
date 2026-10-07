// ARCH: sm_100a
// 76_tma_load_gather4_swizzle_test.cu -- gather4 + SWIZZLE_128B
// round-trip through SMEM and back.
//
// Verifies that the SMEM layout gather4 produces under SWIZZLE_128B
// is the SAME as what a standard tile-mode TMA load would produce
// for the same rows -- i.e., that gather4 honors the descriptor's
// swizzle pattern correctly. The round-trip uses a tile-mode TMA
// store with the SAME swizzle: the descriptor's swizzle cancels
// in the round-trip, so the destination GMEM should equal the
// source GMEM rows (after the identity gather permutation).
//
// Setup:
//   A: bf16, (R=16, C=64) row-major, A[r, c] = r * 1000 + c.
//   gather4 descriptor on A: box=(1, C), SWIZZLE_128B (C * 2 = 128
//     bytes per row -- minimum for B128 swizzle).
//   tile-store descriptor on out: box=(4, C), SWIZZLE_128B.
//   gather4 picks rows {0, 1, 2, 3} (identity). One call fetches 4
//   rows of C cols (= 512 bytes) into SMEM with B128 swizzle.
//   tile-store writes the same 4 rows back to out at coord (0, 0).
//
// If gather4 + B128 swizzle is broken, the round-trip would produce
// garbage (the swizzle cancellation only works if the SMEM contents
// match the expected SWIZZLE_128B layout). If the test passes,
// gather4 + B128 delivers SMEM data identical to a tile-mode load
// of the same rows.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "test_utils.cuh"
#include "../src/primitives/22_tma_store.cuh"
#include "../src/primitives/23_tma_tensormap.cuh"
#include "../src/primitives/25_tma_async_group.cuh"
#include "../src/primitives/29_mbarrier_init.cuh"
#include "../src/primitives/31_mbarrier_arrive_tx.cuh"
#include "../src/primitives/33_mbarrier_try_wait.cuh"
#include "../src/primitives/34_fence_proxy_async.cuh"
#include "../src/primitives/70_smem_ptr.cuh"
#include "../src/primitives/73_tma_load_2d_gather4.cuh"

constexpr int R           = 16;
constexpr int C           = 64;   // C * 2 = 128 bytes per row -> SWIZZLE_128B legal
constexpr int N_GATHER    = 4;
constexpr int TILE_BYTES  = N_GATHER * C * (int)sizeof(__nv_bfloat16);

__device__ __constant__ int d_gather_idx[N_GATHER] = { 0, 1, 2, 3 };

__global__ void k_round_trip(const __grid_constant__ CUtensorMap tmap_in,
                             const __grid_constant__ CUtensorMap tmap_out) {
  __shared__ __align__(1024) __nv_bfloat16 smem[N_GATHER * C];
  __shared__ __align__(16)   uint64_t      mbar;

  if (threadIdx.x == 0) {
    mbarrier_init(smem_ptr_u32(&mbar), 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    mbarrier_arrive_expect_tx(smem_ptr_u32(&mbar), TILE_BYTES);
    tma_load_2d_gather4(smem_ptr_u32(smem), &tmap_in, smem_ptr_u32(&mbar),
                        /*col=*/0,
                        d_gather_idx[0], d_gather_idx[1],
                        d_gather_idx[2], d_gather_idx[3]);
  }
  mbarrier_wait_parity(smem_ptr_u32(&mbar), 0);

  if (threadIdx.x == 0) {
    fence_proxy_async_shared_cta();
    tma_store_2d(&tmap_out, /*x=*/0, /*y=*/0, smem_ptr_u32(smem));
    cp_async_bulk_commit_group();
    cp_async_bulk_wait_group<0>();
  }
}

int main() {
  CUDA_CHECK(cudaFree(0));

  std::vector<__nv_bfloat16> hA(R * C);
  for (int r = 0; r < R; ++r)
    for (int c = 0; c < C; ++c)
      hA[r * C + c] = __float2bfloat16((float)(r * 1000 + c));

  __nv_bfloat16 *dA = nullptr, *dOut = nullptr;
  CUDA_CHECK(cudaMalloc(&dA,  R * C * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&dOut, R * C * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), R * C * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dOut, 0, R * C * 2));

  // gather4 descriptor: box=(1, C), SWIZZLE_128B. C*2 = 128 bytes per
  // row is the minimum width for B128 swizzle.
  CUtensorMap tmap_in;
  CUDA_CHECK(make_tma_2d_tiled(&tmap_in, dA, R, C,
                               /*box_rows=*/1, /*box_cols=*/C,
                               sizeof(__nv_bfloat16),
                               CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                               CU_TENSOR_MAP_SWIZZLE_128B));

  // Tile-store descriptor: box=(N_GATHER, C), SWIZZLE_128B. Same swizzle
  // cancels with the gather4 load's swizzle.
  CUtensorMap tmap_out;
  CUDA_CHECK(make_tma_2d_tiled(&tmap_out, dOut, R, C,
                               /*box_rows=*/N_GATHER, /*box_cols=*/C,
                               sizeof(__nv_bfloat16),
                               CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                               CU_TENSOR_MAP_SWIZZLE_128B));

  k_round_trip<<<1, 32>>>(tmap_in, tmap_out);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> hOut(R * C);
  CUDA_CHECK(cudaMemcpy(hOut.data(), dOut, R * C * 2, cudaMemcpyDeviceToHost));
  cudaFree(dA); cudaFree(dOut);

  int fails = 0;
  for (int r = 0; r < N_GATHER; ++r) {
    int src_row = d_gather_idx[r];  // identity
    (void)src_row;
    for (int c = 0; c < C; ++c) {
      float got  = __bfloat162float(hOut[r * C + c]);
      float want = __bfloat162float(hA[r * C + c]);
      if (got != want) {
        if (fails < 8)
          fprintf(stderr, "  out[%d, %d] = %.0f, want A[%d, %d] = %.0f\n",
                  r, c, got, r, c, want);
        ++fails;
      }
    }
  }
  printf("tma_load_2d_gather4 + SWIZZLE_128B round-trip (R=%d, C=%d): "
         "fails = %d / %d\n", R, C, fails, N_GATHER * C);
  if (fails) FAIL("gather4 SWIZZLE_128B mismatch");
  PASS();
}
