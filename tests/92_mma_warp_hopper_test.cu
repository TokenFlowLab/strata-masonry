#if defined(PL_AGENTIC_SM90A)
// ARCH: sm_90a
// 92_mma_warp_hopper_test.cu -- runtime test for mma warp hopper
//
// Pre-fills SMEM A and B with all-ones, drives mma_warp_hopper_block
// across multiple K-iteration counts, and verifies every accumulator
// element equals K*num_k_tiles.

#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "test_utils.cuh"
#include "../blocks/92_mma_warp_hopper.cuh"

namespace block92 {
static constexpr int TILE_M = 64;
static constexpr int TILE_N = 64;
static constexpr int TILE_K = 16;
static constexpr int NUM_STAGES   = 3;
static constexpr int TILE_A_BYTES = TILE_M * TILE_K * 2;
static constexpr int TILE_B_BYTES = TILE_N * TILE_K * 2;
}

extern __shared__ __align__(128) char block92_smem_buf[];

__global__ void mma_warp_hopper_test_kernel(int num_k_tiles, float* __restrict__ out) {
  using namespace block92;
  int tid = threadIdx.x;

  char*     smem_a  = block92_smem_buf;
  char*     smem_b  = smem_a + NUM_STAGES * TILE_A_BYTES;
  uint64_t* full_mb = reinterpret_cast<uint64_t*>(smem_b + NUM_STAGES * TILE_B_BYTES);
  uint64_t* emp_mb  = full_mb + NUM_STAGES;

  half one = __float2half(1.0f);
  int a_elems_total = NUM_STAGES * TILE_A_BYTES / 2;
  int b_elems_total = NUM_STAGES * TILE_B_BYTES / 2;
  half* A = reinterpret_cast<half*>(smem_a);
  half* B = reinterpret_cast<half*>(smem_b);
  for (int i = tid; i < a_elems_total; i += 128) A[i] = one;
  for (int i = tid; i < b_elems_total; i += 128) B[i] = one;

  uint32_t fb = static_cast<uint32_t>(__cvta_generic_to_shared(full_mb));
  uint32_t eb = static_cast<uint32_t>(__cvta_generic_to_shared(emp_mb));
  if (tid == 0) {
    #pragma unroll
    for (int s = 0; s < NUM_STAGES; ++s) {
      mbarrier_init(fb + s * 8, 1);
      mbarrier_init(eb + s * 8, 1);
      mbarrier_arrive_nostate(fb + s * 8);
    }
  }
  __syncthreads();

  float d[32] = {};
  mma_warp_hopper_block<NUM_STAGES, TILE_A_BYTES, TILE_B_BYTES>(
      smem_a, smem_b, full_mb, emp_mb, num_k_tiles, d, tid);

  #pragma unroll
  for (int i = 0; i < 32; ++i) out[tid * 32 + i] = d[i];
}

int main() {
    using namespace block92;
    CUDA_CHECK(cudaFree(0));
    bool all_pass = true;

    const int test_k_tiles[] = {1, 2, 4, 7};
    const int num_trials = (int)(sizeof(test_k_tiles) / sizeof(int));

    size_t smem_bytes = NUM_STAGES * (TILE_A_BYTES + TILE_B_BYTES)
                      + 2 * NUM_STAGES * sizeof(uint64_t);
    CUDA_CHECK(cudaFuncSetAttribute(mma_warp_hopper_test_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));

    const int NELT   = 128 * 32;
    const int NBYTES = NELT * 4;
    float* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, NBYTES));

    for (int ti = 0; ti < num_trials; ti++) {
        int num_k_tiles = test_k_tiles[ti];
        float expected  = (float)(TILE_K * num_k_tiles);
        CUDA_CHECK(cudaMemset(d_out, 0, NBYTES));

        GpuTimer t; t.begin();
        mma_warp_hopper_test_kernel<<<1, 128, smem_bytes>>>(num_k_tiles, d_out);
        t.end();
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  k_tiles=%d: LAUNCH FAIL: %s\n", num_k_tiles, cudaGetErrorString(err));
            all_pass = false; continue;
        }
        std::vector<float> h(NELT);
        CUDA_CHECK(cudaMemcpy(h.data(), d_out, NBYTES, cudaMemcpyDeviceToHost));
        int bad = 0; int first_idx = -1; float first_val = 0.f;
        for (int i = 0; i < NELT; i++) {
            if (fabsf(h[i] - expected) > 1e-3f) {
                if (bad == 0) { first_idx = i; first_val = h[i]; }
                bad++;
            }
        }
        if (bad == 0) {
            printf("  k_tiles=%d: OK (all %d == %.1f, %.3f ms)\n",
                   num_k_tiles, NELT, expected, t.elapsed_ms());
        } else {
            printf("  k_tiles=%d: FAIL (%d/%d mismatches; first [%d]=%.3f exp %.1f)\n",
                   num_k_tiles, bad, NELT, first_idx, first_val, expected);
            printf("    samples [0]=%.3f [32]=%.3f [1024]=%.3f [%d]=%.3f\n",
                   h[0], h[32], h[1024], NELT-1, h[NELT-1]);
            all_pass = false;
        }
    }

    GpuTimer tp;
    const int ITERS = 50;
    tp.begin();
    for (int i = 0; i < ITERS; i++) {
        mma_warp_hopper_test_kernel<<<1, 128, smem_bytes>>>(4, d_out);
    }
    tp.end();
    printf("  perf: %.2f us/launch (m%dn%dk%d x 4 k-tiles, %d stages)\n",
           tp.elapsed_ms() * 1000.0f / ITERS, TILE_M, TILE_N, TILE_K, NUM_STAGES);

    cudaFree(d_out);
    if (all_pass) { PASS(); return 0; }
    else { FAIL("mma k-loop correctness failed"); return 1; }
}

#endif  // PL_AGENTIC_SM90A
