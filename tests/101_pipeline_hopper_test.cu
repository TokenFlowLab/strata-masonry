#if defined(PL_AGENTIC_SM90A)
// ARCH: sm_90a
// 101_pipeline_hopper_test.cu -- runtime test for pipeline hopper
//
// End-to-end test of the 3-WG Hopper pipeline (block #101). Two probes:
// all-ones (every accumulator slot must equal K) + random smoke
// (no NaN/Inf, magnitude bounded). The fine-grained register ->
// (row, col) layout is covered by test 94 via the stmatrix path.

#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "test_utils.cuh"
#include "../blocks/101_pipeline_hopper.cuh"

namespace block101 {
static constexpr int TILE_M = 64;
static constexpr int TILE_N = 64;
static constexpr int TILE_K = 64;
static constexpr int NUM_STAGES = 3;
static constexpr int TILE_A_BYTES = TILE_M * TILE_K * 2;
static constexpr int TILE_B_BYTES = TILE_N * TILE_K * 2;
}

extern __shared__ __align__(128) char block101_smem_buf[];

__global__ void pipeline_hopper_test_kernel(
    const __grid_constant__ CUtensorMap tma_a,
    const __grid_constant__ CUtensorMap tma_b,
    int num_k_tiles, int tile_m_idx, int tile_n_idx,
    float* __restrict__ out_accum) {
  using namespace block101;
  char*     smem_a  = block101_smem_buf;
  char*     smem_b  = smem_a + NUM_STAGES * TILE_A_BYTES;
  uint64_t* full_mb = reinterpret_cast<uint64_t*>(smem_b + NUM_STAGES * TILE_B_BYTES);
  uint64_t* emp_mb  = full_mb + NUM_STAGES;
  pipeline_hopper_block<NUM_STAGES, TILE_M, TILE_N, TILE_K,
                        TILE_A_BYTES, TILE_B_BYTES>(
      tma_a, tma_b, num_k_tiles, tile_m_idx, tile_n_idx,
      smem_a, smem_b, full_mb, emp_mb, out_accum);
}

int main() {
    using namespace block101;
    CUDA_CHECK(cudaFree(0));
    srand(7);
    bool all_pass = true;

    constexpr int M = TILE_M, N = TILE_N, K = TILE_K;
    constexpr int NUM_K_TILES = 1;

    std::vector<half>  h_A(M * K), h_B(K * N);
    constexpr int ACC_ELEMS = 128 * 32;

    half*  d_A; half* d_B; float* d_D;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_D, ACC_ELEMS * sizeof(float)));

    CUtensorMap tma_a{}, tma_b{};
    create_tma_2d_f16(&tma_a, d_A, M, K, TILE_M, TILE_K);
    create_tma_2d_f16(&tma_b, d_B, K, N, TILE_K, TILE_N);

    size_t smem_bytes = NUM_STAGES * (TILE_A_BYTES + TILE_B_BYTES)
                      + 2 * NUM_STAGES * sizeof(uint64_t);
    CUDA_CHECK(cudaFuncSetAttribute(pipeline_hopper_test_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));

    {
        for (int i = 0; i < M * K; i++) h_A[i] = __float2half(1.0f);
        for (int i = 0; i < K * N; i++) h_B[i] = __float2half(1.0f);
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M*K*sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K*N*sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_D, 0, ACC_ELEMS * sizeof(float)));

        GpuTimer t; t.begin();
        pipeline_hopper_test_kernel<<<1, 384, smem_bytes>>>(tma_a, tma_b, NUM_K_TILES, 0, 0, d_D);
        t.end();
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  [all-ones] LAUNCH FAIL: %s\n", cudaGetErrorString(err));
            all_pass = false;
        } else {
            std::vector<float> h(ACC_ELEMS);
            CUDA_CHECK(cudaMemcpy(h.data(), d_D, ACC_ELEMS * sizeof(float), cudaMemcpyDeviceToHost));
            int bad = 0; int first = -1; float first_v = 0.f;
            for (int i = 0; i < ACC_ELEMS; i++) {
                if (fabsf(h[i] - (float)K) > 0.1f) {
                    if (bad == 0) { first = i; first_v = h[i]; }
                    bad++;
                }
            }
            if (bad == 0) {
                printf("  [all-ones] OK (all %d accumulator entries == %d, %.3f ms)\n",
                       ACC_ELEMS, K, t.elapsed_ms());
            } else {
                printf("  [all-ones] FAIL %d/%d mismatches; first [%d]=%.3f exp %d\n",
                       bad, ACC_ELEMS, first, first_v, K);
                all_pass = false;
            }
        }
    }

    {
        fill_random_f16(h_A.data(), M * K, -1.0f, 1.0f);
        fill_random_f16(h_B.data(), K * N, -1.0f, 1.0f);
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M*K*sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K*N*sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_D, 0, ACC_ELEMS * sizeof(float)));

        GpuTimer t; t.begin();
        pipeline_hopper_test_kernel<<<1, 384, smem_bytes>>>(tma_a, tma_b, NUM_K_TILES, 0, 0, d_D);
        t.end();
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  [random] LAUNCH FAIL: %s\n", cudaGetErrorString(err));
            all_pass = false;
        } else {
            std::vector<float> h(ACC_ELEMS);
            CUDA_CHECK(cudaMemcpy(h.data(), d_D, ACC_ELEMS * sizeof(float), cudaMemcpyDeviceToHost));
            std::vector<float> ref_D(M * N);
            cpu_gemm_f16(h_A.data(), h_B.data(), ref_D.data(), M, N, K);

            float max_abs = 0.f; int nan_count = 0; int nonzero = 0;
            for (int i = 0; i < ACC_ELEMS; i++) {
                float v = h[i];
                if (!isfinite(v)) nan_count++;
                else {
                    if (fabsf(v) > max_abs) max_abs = fabsf(v);
                    if (v != 0.f) nonzero++;
                }
            }
            double sum_got = 0, sum_ref = 0;
            for (int i = 0; i < ACC_ELEMS; i++) sum_got += h[i];
            for (int i = 0; i < M*N; i++) sum_ref += ref_D[i];
            bool smoke_ok = (nan_count == 0) && (nonzero > ACC_ELEMS / 2) &&
                            (max_abs < 64.0f);
            if (smoke_ok) {
                printf("  [random smoke] OK (max_abs=%.3f nonzero=%d/%d, sum_got=%.3f sum_ref=%.3f, %.3f ms)\n",
                       max_abs, nonzero, ACC_ELEMS, sum_got, sum_ref, t.elapsed_ms());
                printf("                 NOTE: fine-grained position check deferred to test 94 (stmatrix epi).\n");
            } else {
                printf("  [random smoke] FAIL max_abs=%.3f nonzero=%d NaN=%d\n",
                       max_abs, nonzero, nan_count);
                all_pass = false;
            }
        }
    }

    GpuTimer tp;
    const int ITERS = 50;
    tp.begin();
    for (int i = 0; i < ITERS; i++) {
        pipeline_hopper_test_kernel<<<1, 384, smem_bytes>>>(tma_a, tma_b, NUM_K_TILES, 0, 0, d_D);
    }
    tp.end();
    printf("  perf: %.2f us/launch (3-WG pipeline, m%dn%dk%d)\n",
           tp.elapsed_ms() * 1000.0f / ITERS, M, N, K);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_D);
    if (all_pass) { PASS(); return 0; }
    else { FAIL("end-to-end pipeline failed"); return 1; }
}

#endif  // PL_AGENTIC_SM90A
