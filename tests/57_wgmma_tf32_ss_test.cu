#if defined(PL_AGENTIC_SM90A)
// ARCH: sm_90a
// 57_wgmma_tf32_ss_test.cu -- runtime test for wgmma tf32 ss
//
// Runtime test: wgmma.mma_async TF32 SS -- numerical correctness.
// TF32 has K=8. Each TF32 element is stored in 32 bits (upper 19 bits matter).
// All-ones strategy: D = A*B with A,B all 1.0 -> D == K == 8.

#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include "test_utils.cuh"
#include "43_smem_desc_hopper.cuh"
#include "57_wgmma_tf32_ss.cuh"
#include "59_wgmma_fence_commit_wait.cuh"

// TF32 1.0: same bit pattern as float 1.0 with low 13 bits zero = 0x3F800000.
// float 1.0 already has low 13 bits zero, so reinterpret_cast works.
__device__ __forceinline__ uint32_t tf32_one() {
    return 0x3F800000u;
}

__global__ void wgmma_tf32_m64n8k8_allones_kernel(float* __restrict__ out) {
    constexpr int M = 64, N = 8, K = 8;
    // TF32 elements are 32 bits; store as uint32_t.
    __shared__ __align__(128) uint32_t smem_A[M * K];                // 2048 bytes
    __shared__ __align__(128) uint32_t smem_B[(N + 8) * K];          // headroom
    int tid = threadIdx.x;

    uint32_t one = tf32_one();
    for (int i = tid; i < M * K; i += 128) smem_A[i] = one;
    for (int i = tid; i < (N + 8) * K; i += 128) smem_B[i] = one;
    __syncthreads();

    uint32_t a_addr = smem_ptr_u32(&smem_A[0]);
    uint32_t b_addr = smem_ptr_u32(&smem_B[0]);
    // K=8 TF32 = 32 bytes per row. INTERLEAVE K-major canonical: LBO=128, SBO=256.
    uint64_t a_desc = build_smem_desc_hopper(a_addr, 128, 256, 0);
    uint64_t b_desc = build_smem_desc_hopper(b_addr, 128, 256, 0);

    float d[4] = {0, 0, 0, 0};
    wgmma_fence();
    wgmma_tf32_ss_m64n8k8(d, a_desc, b_desc, /*scale_d*/ false);
    wgmma_commit_group();
    wgmma_wait_group<0>();

    for (int i = 0; i < 4; i++) out[tid * 4 + i] = d[i];
}

__global__ void wgmma_tf32_m64n16k8_allones_kernel(float* __restrict__ out) {
    constexpr int M = 64, N = 16, K = 8;
    __shared__ __align__(128) uint32_t smem_A[M * K];
    __shared__ __align__(128) uint32_t smem_B[(N + 8) * K];
    int tid = threadIdx.x;

    uint32_t one = tf32_one();
    for (int i = tid; i < M * K; i += 128) smem_A[i] = one;
    for (int i = tid; i < (N + 8) * K; i += 128) smem_B[i] = one;
    __syncthreads();

    uint32_t a_addr = smem_ptr_u32(&smem_A[0]);
    uint32_t b_addr = smem_ptr_u32(&smem_B[0]);
    uint64_t a_desc = build_smem_desc_hopper(a_addr, 128, 256, 0);
    uint64_t b_desc = build_smem_desc_hopper(b_addr, 128, 256, 0);

    float d[8] = {0,0,0,0,0,0,0,0};
    wgmma_fence();
    wgmma_tf32_ss_m64n16k8(d, a_desc, b_desc, /*scale_d*/ false);
    wgmma_commit_group();
    wgmma_wait_group<0>();

    for (int i = 0; i < 8; i++) out[tid * 8 + i] = d[i];
}

__global__ void wgmma_tf32_m64n64k8_allones_kernel(float* __restrict__ out) {
    constexpr int M = 64, N = 64, K = 8;
    __shared__ __align__(128) uint32_t smem_A[M * K];
    __shared__ __align__(128) uint32_t smem_B[(N + 8) * K];
    int tid = threadIdx.x;

    uint32_t one = tf32_one();
    for (int i = tid; i < M * K; i += 128) smem_A[i] = one;
    for (int i = tid; i < (N + 8) * K; i += 128) smem_B[i] = one;
    __syncthreads();

    uint32_t a_addr = smem_ptr_u32(&smem_A[0]);
    uint32_t b_addr = smem_ptr_u32(&smem_B[0]);
    uint64_t a_desc = build_smem_desc_hopper(a_addr, 128, 256, 0);
    uint64_t b_desc = build_smem_desc_hopper(b_addr, 128, 256, 0);

    float d[32] = {};
    wgmma_fence();
    wgmma_tf32_ss_m64n64k8(d, a_desc, b_desc, /*scale_d*/ false);
    wgmma_commit_group();
    wgmma_wait_group<0>();

    for (int i = 0; i < 32; i++) out[tid * 32 + i] = d[i];
}

// TF32 sparse 1:2: K_logical=16, K_compact=8. Expected D = K_active = 8.
__global__ void wgmma_tf32_m64n8k16_sp_allones_kernel(float* __restrict__ out) {
    constexpr int M = 64, N = 8, K_LOG = 16, K_COMPACT = K_LOG / 2;
    __shared__ __align__(128) uint32_t smem_A[M * K_COMPACT];
    __shared__ __align__(128) uint32_t smem_B[(N + 8) * K_LOG];
    int tid = threadIdx.x;

    uint32_t one = tf32_one();
    for (int i = tid; i < M * K_COMPACT; i += 128) smem_A[i] = one;
    for (int i = tid; i < (N + 8) * K_LOG; i += 128) smem_B[i] = one;
    __syncthreads();

    uint32_t a_addr = smem_ptr_u32(&smem_A[0]);
    uint32_t b_addr = smem_ptr_u32(&smem_B[0]);
    uint64_t a_desc = build_smem_desc_hopper(a_addr, 128, 256, 0);
    uint64_t b_desc = build_smem_desc_hopper(b_addr, 128, 256, 0);

    float d[4] = {0,0,0,0};
    // 1:2 sparsity legal indices: 0b1110 (T0,T1) per PTX 9.7.17.6.2.1
    uint32_t e_meta = 0xeeeeeeeeu;
    wgmma_fence();
    wgmma_tf32_ss_sp_m64n8k16<0>(d, a_desc, b_desc, e_meta, /*scale_d*/ false);
    wgmma_commit_group();
    wgmma_wait_group<0>();

    for (int i = 0; i < 4; i++) out[tid * 4 + i] = d[i];
}

int main() {
    CUDA_CHECK(cudaFree(0));
    bool all_pass = true;
    const float expected_K = 8.0f;

    {
        const int NELT = 128 * 4;
        float* d_out; CUDA_CHECK(cudaMalloc(&d_out, NELT * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_out, 0, NELT * sizeof(float)));
        wgmma_tf32_m64n8k8_allones_kernel<<<1, 128>>>(d_out);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  m64n8k8 tf32: LAUNCH FAIL: %s\n", cudaGetErrorString(err));
            all_pass = false;
        } else {
            float h_out[NELT];
            CUDA_CHECK(cudaMemcpy(h_out, d_out, NELT * sizeof(float), cudaMemcpyDeviceToHost));
            int bad = 0;
            for (int i = 0; i < NELT; i++)
                if (fabsf(h_out[i] - expected_K) > 1e-3f) bad++;
            if (bad == 0) printf("  m64n8k8 tf32: OK (all 512 values == %.1f)\n", expected_K);
            else { printf("  m64n8k8 tf32: FAIL (%d bad)\n", bad);
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
        wgmma_tf32_m64n16k8_allones_kernel<<<1, 128>>>(d_out);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  m64n16k8 tf32: LAUNCH FAIL: %s\n", cudaGetErrorString(err));
            all_pass = false;
        } else {
            float h_out[NELT];
            CUDA_CHECK(cudaMemcpy(h_out, d_out, NELT * sizeof(float), cudaMemcpyDeviceToHost));
            int bad = 0;
            for (int i = 0; i < NELT; i++)
                if (fabsf(h_out[i] - expected_K) > 1e-3f) bad++;
            if (bad == 0) printf("  m64n16k8 tf32: OK (all 1024 values == %.1f)\n", expected_K);
            else { printf("  m64n16k8 tf32: FAIL (%d bad)\n", bad);
                   all_pass = false; }
        }
        cudaFree(d_out);
    }

    {
        const int NELT = 128 * 32;
        float* d_out; CUDA_CHECK(cudaMalloc(&d_out, NELT * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_out, 0, NELT * sizeof(float)));
        wgmma_tf32_m64n64k8_allones_kernel<<<1, 128>>>(d_out);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  m64n64k8 tf32: LAUNCH FAIL: %s\n", cudaGetErrorString(err));
            all_pass = false;
        } else {
            std::vector<float> h_out(NELT);
            CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, NELT * sizeof(float), cudaMemcpyDeviceToHost));
            int bad = 0;
            for (int i = 0; i < NELT; i++)
                if (fabsf(h_out[i] - expected_K) > 1e-3f) bad++;
            if (bad == 0) printf("  m64n64k8 tf32: OK (all %d values == %.1f)\n", NELT, expected_K);
            else { printf("  m64n64k8 tf32: FAIL (%d bad of %d)\n", bad, NELT); all_pass = false; }
        }
        cudaFree(d_out);
    }

    // --- perf: m64n8k8 tf32 ---
    {
        const int NELT = 128 * 4;
        float* d_out; CUDA_CHECK(cudaMalloc(&d_out, NELT * sizeof(float)));
        GpuTimer t;
        const int ITERS = 200;
        t.begin();
        for (int i = 0; i < ITERS; i++) wgmma_tf32_m64n8k8_allones_kernel<<<1, 128>>>(d_out);
        t.end();
        printf("  perf: %.2f us/launch (wgmma m64n8k8 tf32 SS, 1 CTA x 128 thr)\n",
               t.elapsed_ms() * 1000.0f / ITERS);
        cudaFree(d_out);
    }

    // --- m64n8k16 SPARSE tf32 (1:2) ---
    {
        const int NELT = 128 * 4;
        float* d_out; CUDA_CHECK(cudaMalloc(&d_out, NELT * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_out, 0, NELT * sizeof(float)));
        wgmma_tf32_m64n8k16_sp_allones_kernel<<<1, 128>>>(d_out);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  m64n8k16 sparse tf32: LAUNCH FAIL: %s\n", cudaGetErrorString(err));
            all_pass = false;
        } else {
            std::vector<float> h_out(NELT);
            CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, NELT * sizeof(float), cudaMemcpyDeviceToHost));
            const float expected_sp = 8.0f;  // K_active = K_logical/2 = 8
            int bad = 0;
            for (int i = 0; i < NELT; i++)
                if (fabsf(h_out[i] - expected_sp) > 1e-3f) bad++;
            if (bad == 0) printf("  m64n8k16 sparse tf32 (1:2): OK (all %d == %.0f)\n", NELT, expected_sp);
            else { printf("  m64n8k16 sparse tf32: FAIL (%d bad of %d; sample [0]=%.3f exp %.0f)\n",
                          bad, NELT, h_out[0], expected_sp);
                   all_pass = false; }
        }
        cudaFree(d_out);
    }

    if (all_pass) { PASS(); return 0; }
    else { FAIL("some subtests failed"); return 1; }
}

#endif  // PL_AGENTIC_SM90A
