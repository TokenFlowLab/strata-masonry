#if defined(PL_AGENTIC_SM90A)
// ARCH: sm_90a
// 55_wgmma_bf16_ss_test.cu -- runtime test for wgmma bf16 ss
//
// Runtime test: wgmma.mma_async BF16 SS -- numerical correctness.
// Same all-ones strategy as #53: D = A*B where A,B are all 1.0 -> D == K.

#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "test_utils.cuh"
#include "43_smem_desc_hopper.cuh"
#include "55_wgmma_bf16_ss.cuh"
#include "59_wgmma_fence_commit_wait.cuh"

__global__ void wgmma_bf16_m64n8k16_allones_kernel(float* __restrict__ out) {
    constexpr int M = 64, N = 8, K = 16;
    __shared__ __align__(128) __nv_bfloat16 smem_A[M * K];
    __shared__ __align__(128) __nv_bfloat16 smem_B[N * K];
    int tid = threadIdx.x;

    for (int i = tid; i < M * K; i += 128) smem_A[i] = __float2bfloat16(1.0f);
    if (tid < N * K) smem_B[tid] = __float2bfloat16(1.0f);
    __syncthreads();

    uint32_t a_addr = smem_ptr_u32(&smem_A[0]);
    uint32_t b_addr = smem_ptr_u32(&smem_B[0]);
    // K=16 BF16 = 32 bytes per row; SBO=8*32=256 bytes; INTERLEAVE.
    uint64_t a_desc = build_smem_desc_hopper(a_addr, 32, 256, 0);
    uint64_t b_desc = build_smem_desc_hopper(b_addr, 32, 256, 0);

    float d[4] = {0.f, 0.f, 0.f, 0.f};
    wgmma_fence();
    wgmma_bf16_ss_m64n8k16(d, a_desc, b_desc, /*scale_d*/ false);
    wgmma_commit_group();
    wgmma_wait_group<0>();

    out[tid * 4 + 0] = d[0];
    out[tid * 4 + 1] = d[1];
    out[tid * 4 + 2] = d[2];
    out[tid * 4 + 3] = d[3];
}

__global__ void wgmma_bf16_m64n64k16_allones_kernel(float* __restrict__ out) {
    constexpr int M = 64, N = 64, K = 16;
    __shared__ __align__(128) __nv_bfloat16 smem_A[M * K];
    __shared__ __align__(128) __nv_bfloat16 smem_B[N * K];
    int tid = threadIdx.x;

    for (int i = tid; i < M * K; i += 128) smem_A[i] = __float2bfloat16(1.0f);
    for (int i = tid; i < N * K; i += 128) smem_B[i] = __float2bfloat16(1.0f);
    __syncthreads();

    uint32_t a_addr = smem_ptr_u32(&smem_A[0]);
    uint32_t b_addr = smem_ptr_u32(&smem_B[0]);
    uint64_t a_desc = build_smem_desc_hopper(a_addr, 32, 256, 0);
    uint64_t b_desc = build_smem_desc_hopper(b_addr, 32, 256, 0);

    float d[32] = {};
    wgmma_fence();
    wgmma_bf16_ss_m64n64k16(d, a_desc, b_desc, /*scale_d*/ false);
    wgmma_commit_group();
    wgmma_wait_group<0>();

    for (int i = 0; i < 32; i++) out[tid * 32 + i] = d[i];
}

__global__ void wgmma_bf16_m64n16k16_allones_kernel(float* __restrict__ out) {
    constexpr int M = 64, N = 16, K = 16;
    __shared__ __align__(128) __nv_bfloat16 smem_A[M * K];
    __shared__ __align__(128) __nv_bfloat16 smem_B[N * K];
    int tid = threadIdx.x;

    for (int i = tid; i < M * K; i += 128) smem_A[i] = __float2bfloat16(1.0f);
    for (int i = tid; i < N * K; i += 128) smem_B[i] = __float2bfloat16(1.0f);
    __syncthreads();

    uint32_t a_addr = smem_ptr_u32(&smem_A[0]);
    uint32_t b_addr = smem_ptr_u32(&smem_B[0]);
    uint64_t a_desc = build_smem_desc_hopper(a_addr, 32, 256, 0);
    uint64_t b_desc = build_smem_desc_hopper(b_addr, 32, 256, 0);

    float d[8] = {0,0,0,0,0,0,0,0};
    wgmma_fence();
    wgmma_bf16_ss_m64n16k16(d, a_desc, b_desc, /*scale_d*/ false);
    wgmma_commit_group();
    wgmma_wait_group<0>();

    for (int i = 0; i < 8; i++) out[tid * 8 + i] = d[i];
}

__global__ void wgmma_bf16_m64n8k32_sp_allones_kernel(float* __restrict__ out) {
    constexpr int M = 64, N = 8, K_LOG = 32, K_COMPACT = K_LOG / 2;
    __shared__ __align__(128) __nv_bfloat16 smem_A[M * K_COMPACT];
    __shared__ __align__(128) __nv_bfloat16 smem_B[N * K_LOG];
    int tid = threadIdx.x;

    for (int i = tid; i < M * K_COMPACT; i += 128) smem_A[i] = __float2bfloat16(1.0f);
    if (tid < N * K_LOG) smem_B[tid] = __float2bfloat16(1.0f);
    __syncthreads();

    uint32_t a_addr = smem_ptr_u32(&smem_A[0]);
    uint32_t b_addr = smem_ptr_u32(&smem_B[0]);
    uint64_t a_desc = build_smem_desc_hopper(a_addr, 32, 256, 0);
    uint64_t b_desc = build_smem_desc_hopper(b_addr, 32, 256, 0);

    float d[4] = {0,0,0,0};
    uint32_t e_meta = 0x44444444u;
    wgmma_fence();
    wgmma_bf16_ss_sp_m64n8k32<0>(d, a_desc, b_desc, e_meta, /*scale_d*/ false);
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
        wgmma_bf16_m64n8k16_allones_kernel<<<1, 128>>>(d_out);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  m64n8k16 bf16: LAUNCH FAIL: %s\n", cudaGetErrorString(err));
            all_pass = false;
        } else {
            float h_out[NELT];
            CUDA_CHECK(cudaMemcpy(h_out, d_out, NELT * sizeof(float), cudaMemcpyDeviceToHost));
            int bad = 0;
            float first_bad = -1.f; int first_idx = -1;
            for (int i = 0; i < NELT; i++) {
                if (fabsf(h_out[i] - expected_K) > 1e-2f) {
                    if (bad == 0) { first_bad = h_out[i]; first_idx = i; }
                    bad++;
                }
            }
            if (bad == 0) printf("  m64n8k16 bf16: OK (all 512 values == %.1f)\n", expected_K);
            else { printf("  m64n8k16 bf16: FAIL (%d bad; first [%d]=%.3f)\n", bad, first_idx, first_bad);
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
        wgmma_bf16_m64n16k16_allones_kernel<<<1, 128>>>(d_out);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  m64n16k16 bf16: LAUNCH FAIL: %s\n", cudaGetErrorString(err));
            all_pass = false;
        } else {
            float h_out[NELT];
            CUDA_CHECK(cudaMemcpy(h_out, d_out, NELT * sizeof(float), cudaMemcpyDeviceToHost));
            int bad = 0;
            float first_bad = -1.f; int first_idx = -1;
            for (int i = 0; i < NELT; i++) {
                if (fabsf(h_out[i] - expected_K) > 1e-2f) {
                    if (bad == 0) { first_bad = h_out[i]; first_idx = i; }
                    bad++;
                }
            }
            if (bad == 0) printf("  m64n16k16 bf16: OK (all 1024 values == %.1f)\n", expected_K);
            else { printf("  m64n16k16 bf16: FAIL (%d bad; first [%d]=%.3f)\n", bad, first_idx, first_bad);
                   all_pass = false; }
        }
        cudaFree(d_out);
    }

    // --- m64n64k16 bf16 ---
    {
        const int NELT = 128 * 32;
        float* d_out; CUDA_CHECK(cudaMalloc(&d_out, NELT * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_out, 0, NELT * sizeof(float)));
        wgmma_bf16_m64n64k16_allones_kernel<<<1, 128>>>(d_out);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  m64n64k16 bf16: LAUNCH FAIL: %s\n", cudaGetErrorString(err));
            all_pass = false;
        } else {
            std::vector<float> h_out(NELT);
            CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, NELT * sizeof(float), cudaMemcpyDeviceToHost));
            int bad = 0;
            for (int i = 0; i < NELT; i++) {
                if (fabsf(h_out[i] - expected_K) > 1e-2f) bad++;
            }
            if (bad == 0) printf("  m64n64k16 bf16: OK (all %d values == %.1f)\n", NELT, expected_K);
            else { printf("  m64n64k16 bf16: FAIL (%d bad of %d)\n", bad, NELT); all_pass = false; }
        }
        cudaFree(d_out);
    }

    // --- perf: m64n8k16 bf16 ---
    {
        const int NELT = 128 * 4;
        float* d_out; CUDA_CHECK(cudaMalloc(&d_out, NELT * sizeof(float)));
        GpuTimer t;
        const int ITERS = 200;
        t.begin();
        for (int i = 0; i < ITERS; i++) wgmma_bf16_m64n8k16_allones_kernel<<<1, 128>>>(d_out);
        t.end();
        printf("  perf: %.2f us/launch (wgmma m64n8k16 bf16 SS, 1 CTA x 128 thr)\n",
               t.elapsed_ms() * 1000.0f / ITERS);
        cudaFree(d_out);
    }

    // --- m64n8k32 SPARSE bf16 ---
    {
        const int NELT = 128 * 4;
        float* d_out; CUDA_CHECK(cudaMalloc(&d_out, NELT * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_out, 0, NELT * sizeof(float)));
        wgmma_bf16_m64n8k32_sp_allones_kernel<<<1, 128>>>(d_out);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  m64n8k32 sparse bf16: LAUNCH FAIL: %s\n", cudaGetErrorString(err));
            all_pass = false;
        } else {
            std::vector<float> h_out(NELT);
            CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, NELT * sizeof(float), cudaMemcpyDeviceToHost));
            const float expected_sp = 16.0f;
            int bad = 0;
            for (int i = 0; i < NELT; i++)
                if (fabsf(h_out[i] - expected_sp) > 1e-2f) bad++;
            if (bad == 0) printf("  m64n8k32 sparse bf16: OK (all %d == %.0f)\n", NELT, expected_sp);
            else { printf("  m64n8k32 sparse bf16: FAIL (%d bad of %d)\n", bad, NELT); all_pass = false; }
        }
        cudaFree(d_out);
    }

    if (all_pass) { PASS(); return 0; }
    else { FAIL("some subtests failed"); return 1; }
}

#endif  // PL_AGENTIC_SM90A
