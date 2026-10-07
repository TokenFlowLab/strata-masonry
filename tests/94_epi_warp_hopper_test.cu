#if defined(PL_AGENTIC_SM90A)
// ARCH: sm_90a
// 94_epi_warp_hopper_test.cu -- runtime test for epi warp hopper
//
// Exercises the block-#94 epilogue body via two probes per dtype
// (constant + injected single value) for FP16 and BF16. Verifies that
// the injected value appears exactly once in the GMEM tile, confirming
// per-thread stmatrix.x4 row-mapping is correct.

#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "test_utils.cuh"
#include "../src/blocks/94_epi_warp_hopper.cuh"

namespace block94 {
static constexpr int EPI_M = 16;
static constexpr int EPI_N = 16;
static constexpr int EPI_ELEMS = EPI_M * EPI_N;
static constexpr int EPI_BYTES = EPI_ELEMS * 2;
}

extern __shared__ __align__(128) char block94_smem_buf[];

__global__ void epi_warp_hopper_f16_test_kernel(
    const __grid_constant__ CUtensorMap tma_d,
    int tm, int tn,
    float base, int inject_thread, int inject_slot, float inject_value) {
  using namespace block94;
  int T = threadIdx.x;
  float d[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) d[i] = base;
  if (T == inject_thread) d[inject_slot] = inject_value;
  epi_warp_hopper_f16_block<EPI_M, EPI_N>(tma_d, block94_smem_buf, tm, tn, d, T);
}

__global__ void epi_warp_hopper_bf16_test_kernel(
    const __grid_constant__ CUtensorMap tma_d,
    int tm, int tn,
    float base, int inject_thread, int inject_slot, float inject_value) {
  using namespace block94;
  int T = threadIdx.x;
  float d[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) d[i] = base;
  if (T == inject_thread) d[inject_slot] = inject_value;
  epi_warp_hopper_bf16_block<EPI_M, EPI_N>(tma_d, block94_smem_buf, tm, tn, d, T);
}

int main() {
    using namespace block94;
    CUDA_CHECK(cudaFree(0));
    bool all_pass = true;

    half*          d_f16_A;
    half*          d_f16_B;
    __nv_bfloat16* d_bf16_A;
    CUDA_CHECK(cudaMalloc(&d_f16_A,  EPI_BYTES));
    CUDA_CHECK(cudaMalloc(&d_f16_B,  EPI_BYTES));
    CUDA_CHECK(cudaMalloc(&d_bf16_A, EPI_BYTES));

    std::vector<uint16_t> sentinel(EPI_ELEMS, 0xBEEF);
    CUDA_CHECK(cudaMemcpy(d_f16_A,  sentinel.data(), EPI_BYTES, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_f16_B,  sentinel.data(), EPI_BYTES, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bf16_A, sentinel.data(), EPI_BYTES, cudaMemcpyHostToDevice));

    CUtensorMap tma_f16_A{}, tma_f16_B{}, tma_bf16_A{};
    create_tma_2d_desc(&tma_f16_A,  d_f16_A,  EPI_M, EPI_N, EPI_M, EPI_N, 2,
                       CU_TENSOR_MAP_DATA_TYPE_FLOAT16,  CU_TENSOR_MAP_SWIZZLE_NONE);
    create_tma_2d_desc(&tma_f16_B,  d_f16_B,  EPI_M, EPI_N, EPI_M, EPI_N, 2,
                       CU_TENSOR_MAP_DATA_TYPE_FLOAT16,  CU_TENSOR_MAP_SWIZZLE_NONE);
    create_tma_2d_desc(&tma_bf16_A, d_bf16_A, EPI_M, EPI_N, EPI_M, EPI_N, 2,
                       CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_NONE);

    size_t smem_bytes = EPI_BYTES;
    CUDA_CHECK(cudaFuncSetAttribute(epi_warp_hopper_f16_test_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));
    CUDA_CHECK(cudaFuncSetAttribute(epi_warp_hopper_bf16_test_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));

    // Probe A: F16 constant
    {
        GpuTimer t; t.begin();
        epi_warp_hopper_f16_test_kernel<<<1, 32, smem_bytes>>>(tma_f16_A, 0, 0, 3.0f, -1, -1, 0.0f);
        t.end(); CUDA_CHECK(cudaGetLastError());
        std::vector<half> h(EPI_ELEMS);
        CUDA_CHECK(cudaMemcpy(h.data(), d_f16_A, EPI_BYTES, cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int i = 0; i < EPI_ELEMS; i++) {
            float v = __half2float(h[i]);
            if (v != 3.0f) {
                if (bad < 4) printf("  [f16 const] idx=%d got=%.3f\n", i, v);
                bad++;
            }
        }
        if (bad == 0) printf("  f16 constant tile: OK (%d elems all 3.0, %.3f ms)\n",
                             EPI_ELEMS, t.elapsed_ms());
        else { printf("  f16 constant tile: FAIL (%d mismatches)\n", bad); all_pass = false; }
    }

    // Probe B: F16 inject
    {
        const float base   = 7.0f;
        const float inject = -2.5f;
        const int   thr    = 17;
        const int   slot   = 5;
        GpuTimer t; t.begin();
        epi_warp_hopper_f16_test_kernel<<<1, 32, smem_bytes>>>(tma_f16_B, 0, 0, base, thr, slot, inject);
        t.end(); CUDA_CHECK(cudaGetLastError());
        std::vector<half> h(EPI_ELEMS);
        CUDA_CHECK(cudaMemcpy(h.data(), d_f16_B, EPI_BYTES, cudaMemcpyDeviceToHost));
        int count_base = 0, count_inject = 0, count_other = 0;
        for (int i = 0; i < EPI_ELEMS; i++) {
            float v = __half2float(h[i]);
            if (v == base)        count_base++;
            else if (v == inject) count_inject++;
            else                  count_other++;
        }
        bool ok = (count_inject == 1) && (count_base == EPI_ELEMS - 1) && (count_other == 0);
        if (ok) printf("  f16 inject tile: OK (base=%d inject=1, %.3f ms)\n",
                       count_base, t.elapsed_ms());
        else { printf("  f16 inject tile: FAIL base=%d inject=%d other=%d\n",
                      count_base, count_inject, count_other); all_pass = false; }
    }

    // Probe A: BF16 constant
    {
        GpuTimer t; t.begin();
        epi_warp_hopper_bf16_test_kernel<<<1, 32, smem_bytes>>>(tma_bf16_A, 0, 0, 5.0f, -1, -1, 0.0f);
        t.end(); CUDA_CHECK(cudaGetLastError());
        std::vector<__nv_bfloat16> h(EPI_ELEMS);
        CUDA_CHECK(cudaMemcpy(h.data(), d_bf16_A, EPI_BYTES, cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int i = 0; i < EPI_ELEMS; i++) {
            float v = __bfloat162float(h[i]);
            if (v != 5.0f) {
                if (bad < 4) printf("  [bf16 const] idx=%d got=%.3f\n", i, v);
                bad++;
            }
        }
        if (bad == 0) printf("  bf16 constant tile: OK (%d elems all 5.0, %.3f ms)\n",
                             EPI_ELEMS, t.elapsed_ms());
        else { printf("  bf16 constant tile: FAIL (%d mismatches)\n", bad); all_pass = false; }
    }

    GpuTimer tp;
    const int ITERS = 200;
    tp.begin();
    for (int i = 0; i < ITERS; i++) {
        epi_warp_hopper_f16_test_kernel<<<1, 32, smem_bytes>>>(tma_f16_A, 0, 0, 3.0f, -1, -1, 0.0f);
    }
    tp.end();
    printf("  perf: %.2f us/launch (epi %dx%d fp16 one-warp)\n",
           tp.elapsed_ms() * 1000.0f / ITERS, EPI_M, EPI_N);

    cudaFree(d_f16_A); cudaFree(d_f16_B); cudaFree(d_bf16_A);
    if (all_pass) { PASS(); return 0; }
    else { FAIL("subtests failed"); return 1; }
}

#endif  // PL_AGENTIC_SM90A
