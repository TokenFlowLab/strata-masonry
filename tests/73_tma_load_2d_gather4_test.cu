// ARCH: sm_100a
// 73_tma_load_2d_gather4_test.cu -- TMA gather4 fetches 4 non-contiguous
// rows of a GMEM tensor into one SMEM tile, host verifies each SMEM row
// equals the corresponding source row.
//
// Setup:
//   A: (R, C) fp32 GMEM, A[r, c] = r * 10000 + c so each row is distinct.
//   Tensormap: box = (1, C), SWIZZLE_NONE, FLOAT32. With box_rows = 1
//     gather4 fetches 4 independent 1-row strips of length C into a
//     contiguous (4, C) SMEM destination.
//   Gather indices: {2, 5, 11, 7} (arbitrary, in-range row picks).
//
// Verify: out[i, c] == A[gather_idx[i], c] for i in 0..3, c in 0..C-1.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "test_utils.cuh"
#include "../primitives/73_tma_load_2d_gather4.cuh"
#include "../primitives/23_tma_tensormap.cuh"
#include "../primitives/29_mbarrier_init.cuh"
#include "../primitives/31_mbarrier_arrive_tx.cuh"
#include "../primitives/33_mbarrier_try_wait.cuh"
#include "../primitives/70_smem_ptr.cuh"

constexpr int R = 16;
constexpr int C = 16;
constexpr int N_GATHER = 4;  // gather4 is fixed-width-4
__device__ __constant__ int d_gather_idx[N_GATHER] = { 2, 5, 11, 7 };

__global__ void k_gather4(const __grid_constant__ CUtensorMap desc,
                          float* out) {
  __shared__ __align__(128) float smem[N_GATHER * C];
  __shared__ __align__(16)  uint64_t mbar;

  if (threadIdx.x == 0) {
    mbarrier_init(smem_ptr_u32(&mbar), 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    mbarrier_arrive_expect_tx(smem_ptr_u32(&mbar),
                              N_GATHER * C * sizeof(float));
    // Single gather4 call: 1 col-index + 4 row-indices.
    tma_load_2d_gather4(smem_ptr_u32(smem), &desc, smem_ptr_u32(&mbar),
                        /*col=*/0,
                        d_gather_idx[0], d_gather_idx[1],
                        d_gather_idx[2], d_gather_idx[3]);
  }
  mbarrier_wait_parity(smem_ptr_u32(&mbar), 0);

  // Drain SMEM -> out for the host to verify. Row layout in SMEM is
  // (N_GATHER, C): row i of SMEM corresponds to gather_idx[i] in GMEM.
  if (threadIdx.x < N_GATHER * C)
    out[threadIdx.x] = smem[threadIdx.x];
  __syncthreads();
  if (threadIdx.x == 0)
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n"
                 :: "r"(smem_ptr_u32(&mbar)));
}

int main() {
  CUDA_CHECK(cudaFree(0));

  // Init A with a row-distinct pattern.
  std::vector<float> hA(R * C);
  for (int r = 0; r < R; ++r)
    for (int c = 0; c < C; ++c)
      hA[r * C + c] = (float)(r * 10000 + c);

  float* dA = nullptr;
  CUDA_CHECK(cudaMalloc(&dA, R * C * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), R * C * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Tensormap with box = (1, C). gather4 fetches 4 such 1-row strips.
  CUtensorMap desc;
  CUDA_CHECK(make_tma_2d_tiled(&desc, dA, R, C,
                               /*box_rows=*/1, /*box_cols=*/C,
                               sizeof(float),
                               CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                               CU_TENSOR_MAP_SWIZZLE_NONE));

  float* dOut = nullptr;
  CUDA_CHECK(cudaMalloc(&dOut, N_GATHER * C * sizeof(float)));
  CUDA_CHECK(cudaMemset(dOut, 0, N_GATHER * C * sizeof(float)));

  k_gather4<<<1, 256>>>(desc, dOut);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> hOut(N_GATHER * C);
  CUDA_CHECK(cudaMemcpy(hOut.data(), dOut, N_GATHER * C * sizeof(float),
                        cudaMemcpyDeviceToHost));
  cudaFree(dA); cudaFree(dOut);

  // Host verify: out[i, c] == A[gather_idx[i], c].
  const int gather_idx[N_GATHER] = { 2, 5, 11, 7 };
  int fails = 0;
  for (int i = 0; i < N_GATHER; ++i) {
    int src_row = gather_idx[i];
    for (int c = 0; c < C; ++c) {
      float got  = hOut[i * C + c];
      float want = hA[src_row * C + c];
      if (got != want) {
        if (fails < 8)
          fprintf(stderr, "  out[%d, %d] = %g, want A[%d, %d] = %g\n",
                  i, c, got, src_row, c, want);
        ++fails;
      }
    }
  }
  printf("tma_load_2d_gather4 (R=%d, C=%d, picks={%d,%d,%d,%d}): fails = %d / %d\n",
         R, C, gather_idx[0], gather_idx[1], gather_idx[2], gather_idx[3],
         fails, N_GATHER * C);
  if (fails) FAIL("gather4 mismatch");
  PASS();
}
