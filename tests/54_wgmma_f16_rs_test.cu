#if defined(PL_AGENTIC_SM90A)
// ARCH: sm_90a
// 54_wgmma_f16_rs_test.cu -- runtime test for wgmma f16 rs
//
// Runtime test: wgmma.mma_async FP16 RS -- numerical correctness.
// A in registers (4 b32 regs per thread = 8 FP16 per thread, K=16).
// B in SMEM via descriptor. All-ones -> D == K == 16.

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "test_utils.cuh"
#include "43_smem_desc_hopper.cuh"
#include "54_wgmma_f16_rs.cuh"
#include "59_wgmma_fence_commit_wait.cuh"

// Pack two FP16 1.0 values (0x3C00) into one uint32_t.
__device__ __forceinline__ uint32_t fp16_ones_pair() {
    return 0x3C003C00u;
}

__global__ void wgmma_f16_rs_m64n8k16_allones_kernel(float* __restrict__ out) {
    constexpr int N = 8, K = 16;
    __shared__ __align__(128) half smem_B[(N + 8) * K];
    int tid = threadIdx.x;

    for (int i = tid; i < (N + 8) * K; i += 128) smem_B[i] = __float2half(1.0f);
    __syncthreads();

    // A in registers: each of 128 threads holds 4 u32 regs = 8 FP16 = "1.0" each.
    uint32_t a0 = fp16_ones_pair();
    uint32_t a1 = fp16_ones_pair();
    uint32_t a2 = fp16_ones_pair();
    uint32_t a3 = fp16_ones_pair();

    uint32_t b_addr = smem_ptr_u32(&smem_B[0]);
    uint64_t b_desc = build_smem_desc_hopper(b_addr, 32, 256, 0);

    float d[4] = {0, 0, 0, 0};
    wgmma_fence();
    wgmma_f16_rs_m64n8k16(d, a0, a1, a2, a3, b_desc, /*scale_d*/ false);
    wgmma_commit_group();
    wgmma_wait_group<0>();

    for (int i = 0; i < 4; i++) out[tid * 4 + i] = d[i];
}

__global__ void wgmma_f16_rs_m64n16k16_allones_kernel(float* __restrict__ out) {
    constexpr int N = 16, K = 16;
    __shared__ __align__(128) half smem_B[(N + 8) * K];
    int tid = threadIdx.x;

    for (int i = tid; i < (N + 8) * K; i += 128) smem_B[i] = __float2half(1.0f);
    __syncthreads();

    uint32_t a0 = fp16_ones_pair();
    uint32_t a1 = fp16_ones_pair();
    uint32_t a2 = fp16_ones_pair();
    uint32_t a3 = fp16_ones_pair();

    uint32_t b_addr = smem_ptr_u32(&smem_B[0]);
    uint64_t b_desc = build_smem_desc_hopper(b_addr, 32, 256, 0);

    float d[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    wgmma_fence();
    wgmma_f16_rs_m64n16k16(d, a0, a1, a2, a3, b_desc, /*scale_d*/ false);
    wgmma_commit_group();
    wgmma_wait_group<0>();

    for (int i = 0; i < 8; i++) out[tid * 8 + i] = d[i];
}

// FP16 RS sparse 2:4 m64n8k32: A_compact in 4 regs (= 8 FP16), B in SMEM.
// A is logically 64x32 with 50% zeros; compactly stored in regs as 8 FP16
// per thread, all 1.0. B is dense 32x8. K_active = 16, expected D = 16.
__global__ void wgmma_f16_rs_m64n8k32_sp_allones_kernel(float* __restrict__ out) {
    constexpr int N = 8, K_LOG = 32;
    __shared__ __align__(128) half smem_B[(N + 8) * K_LOG];
    int tid = threadIdx.x;

    for (int i = tid; i < (N + 8) * K_LOG; i += 128) smem_B[i] = __float2half(1.0f);
    __syncthreads();

    uint32_t a0 = fp16_ones_pair();
    uint32_t a1 = fp16_ones_pair();
    uint32_t a2 = fp16_ones_pair();
    uint32_t a3 = fp16_ones_pair();

    uint32_t b_addr = smem_ptr_u32(&smem_B[0]);
    uint64_t b_desc = build_smem_desc_hopper(b_addr, 32, 256, 0);

    float d[4] = {0, 0, 0, 0};
    uint32_t e_meta = 0x44444444u;
    wgmma_fence();
    wgmma_f16_rs_sp_m64n8k32<0>(d, a0, a1, a2, a3, b_desc, e_meta, /*scale_d*/ false);
    wgmma_commit_group();
    wgmma_wait_group<0>();

    for (int i = 0; i < 4; i++) out[tid * 4 + i] = d[i];
}

int main() {
    CUDA_CHECK(cudaFree(0));
    bool all_pass = true;
    const float expected_K = 16.0f;

    {
        const int NELT = 128 * 4;
        float* d_out; CUDA_CHECK(cudaMalloc(&d_out, NELT * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_out, 0, NELT * sizeof(float)));
        wgmma_f16_rs_m64n8k16_allones_kernel<<<1, 128>>>(d_out);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  m64n8k16 RS: LAUNCH FAIL: %s\n", cudaGetErrorString(err));
            all_pass = false;
        } else {
            float h_out[NELT];
            CUDA_CHECK(cudaMemcpy(h_out, d_out, NELT * sizeof(float), cudaMemcpyDeviceToHost));
            int bad = 0;
            for (int i = 0; i < NELT; i++)
                if (fabsf(h_out[i] - expected_K) > 1e-3f) bad++;
            if (bad == 0) printf("  m64n8k16 RS: OK (all 512 values == %.1f)\n", expected_K);
            else { printf("  m64n8k16 RS: FAIL (%d bad)\n", bad);
                   printf("    samples: [0]=%.3f [4]=%.3f [100]=%.3f [511]=%.3f\n",
                          h_out[0], h_out[4], h_out[100], h_out[511]);
                   all_pass = false; }
        }
        cudaFree(d_out);
    }

    {
        const int NELT = 128 * 8;
        float* d_out; CUDA_CHECK(cudaMalloc(&d_out, NELT * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_out, 0, NELT * sizeof(float)));
        wgmma_f16_rs_m64n16k16_allones_kernel<<<1, 128>>>(d_out);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  m64n16k16 RS: LAUNCH FAIL: %s\n", cudaGetErrorString(err));
            all_pass = false;
        } else {
            float h_out[NELT];
            CUDA_CHECK(cudaMemcpy(h_out, d_out, NELT * sizeof(float), cudaMemcpyDeviceToHost));
            int bad = 0;
            for (int i = 0; i < NELT; i++)
                if (fabsf(h_out[i] - expected_K) > 1e-3f) bad++;
            if (bad == 0) printf("  m64n16k16 RS: OK (all 1024 values == %.1f)\n", expected_K);
            else { printf("  m64n16k16 RS: FAIL (%d bad)\n", bad);
                   all_pass = false; }
        }
        cudaFree(d_out);
    }

    // --- perf: m64n8k16 RS ---
    {
        const int NELT = 128 * 4;
        float* d_out; CUDA_CHECK(cudaMalloc(&d_out, NELT * sizeof(float)));
        GpuTimer t;
        const int ITERS = 200;
        t.begin();
        for (int i = 0; i < ITERS; i++) wgmma_f16_rs_m64n8k16_allones_kernel<<<1, 128>>>(d_out);
        t.end();
        printf("  perf: %.2f us/launch (wgmma m64n8k16 f16 RS, 1 CTA x 128 thr)\n",
               t.elapsed_ms() * 1000.0f / ITERS);
        cudaFree(d_out);
    }

    // --- m64n8k32 SPARSE FP16 RS ---
    {
        const int NELT = 128 * 4;
        float* d_out; CUDA_CHECK(cudaMalloc(&d_out, NELT * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_out, 0, NELT * sizeof(float)));
        wgmma_f16_rs_m64n8k32_sp_allones_kernel<<<1, 128>>>(d_out);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  m64n8k32 sparse f16 RS: LAUNCH FAIL: %s\n", cudaGetErrorString(err));
            all_pass = false;
        } else {
            float* h_out = new float[NELT];
            CUDA_CHECK(cudaMemcpy(h_out, d_out, NELT * sizeof(float), cudaMemcpyDeviceToHost));
            const float expected_sp = 16.0f;
            int bad = 0;
            for (int i = 0; i < NELT; i++)
                if (fabsf(h_out[i] - expected_sp) > 1e-3f) bad++;
            if (bad == 0) printf("  m64n8k32 sparse f16 RS: OK (all %d == %.0f)\n", NELT, expected_sp);
            else { printf("  m64n8k32 sparse f16 RS: FAIL (%d bad of %d)\n", bad, NELT); all_pass = false; }
            delete[] h_out;
        }
        cudaFree(d_out);
    }

    if (all_pass) { PASS(); return 0; }
    else { FAIL("some subtests failed"); return 1; }
}

#endif  // PL_AGENTIC_SM90A
