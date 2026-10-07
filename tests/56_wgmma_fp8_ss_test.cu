#if defined(PL_AGENTIC_SM90A)
// ARCH: sm_90a
// 56_wgmma_fp8_ss_test.cu -- runtime test for wgmma fp8 ss
//
// Runtime test: wgmma.mma_async FP8 (E4M3) SS -- numerical correctness.
// FP8 has K=32. All-ones strategy: D = A*B with A,B all 1.0 -> D == 32.

#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include "test_utils.cuh"
#include "43_smem_desc_hopper.cuh"
#include "56_wgmma_fp8_ss.cuh"
#include "59_wgmma_fence_commit_wait.cuh"

// FP8 E4M3 encoding for 1.0: exponent bias=7, so 1.0 = 0 01111 000 = 0x38.
static constexpr uint8_t E4M3_ONE = 0x38;
// FP8 E5M2 encoding for 1.0: exponent bias=15, so 1.0 = 0 01111 00 = 0x3C.
static constexpr uint8_t E5M2_ONE = 0x3C;

__global__ void wgmma_fp8_m64n8k32_allones_kernel(float* __restrict__ out) {
    constexpr int M = 64, N = 8, K = 32;
    __shared__ __align__(128) uint8_t smem_A[M * K];          // 64x32 FP8 = 2048 bytes
    // Over-allocate B so descriptor strides cannot read past the end:
    // give it full 512 bytes (2 atoms worth) even though only 8 rows needed.
    __shared__ __align__(128) uint8_t smem_B[2 * 8 * K];      // 512 bytes
    int tid = threadIdx.x;

    for (int i = tid; i < M * K; i += 128) smem_A[i] = E4M3_ONE;
    for (int i = tid; i < 2 * 8 * K; i += 128) smem_B[i] = E4M3_ONE;
    __syncthreads();

    uint32_t a_addr = smem_ptr_u32(&smem_A[0]);
    uint32_t b_addr = smem_ptr_u32(&smem_B[0]);
    // INTERLEAVE K-major canonical: LBO = 8 u128 = 128 bytes (stride between
    // the two u128 K-halves within an 8-row atom); SBO = 16 u128 = 256 bytes
    // (stride between 8-row atoms).
    uint64_t a_desc = build_smem_desc_hopper(a_addr, 128, 256, 0);
    uint64_t b_desc = build_smem_desc_hopper(b_addr, 128, 256, 0);

    float d[4] = {0, 0, 0, 0};
    wgmma_fence();
    wgmma_e4m3_ss_m64n8k32(d, a_desc, b_desc, /*scale_d*/ false);
    wgmma_commit_group();
    wgmma_wait_group<0>();

    for (int i = 0; i < 4; i++) out[tid * 4 + i] = d[i];
}

__global__ void wgmma_fp8_m64n16k32_allones_kernel(float* __restrict__ out) {
    constexpr int M = 64, N = 16, K = 32;
    __shared__ __align__(128) uint8_t smem_A[M * K];
    // Over-allocate B with headroom (one extra 8-row tile) so stride math
    // cannot read past the end.
    __shared__ __align__(128) uint8_t smem_B[(N + 8) * K];
    int tid = threadIdx.x;

    for (int i = tid; i < M * K; i += 128) smem_A[i] = E4M3_ONE;
    for (int i = tid; i < (N + 8) * K; i += 128) smem_B[i] = E4M3_ONE;
    __syncthreads();

    uint32_t a_addr = smem_ptr_u32(&smem_A[0]);
    uint32_t b_addr = smem_ptr_u32(&smem_B[0]);
    uint64_t a_desc = build_smem_desc_hopper(a_addr, 128, 256, 0);
    uint64_t b_desc = build_smem_desc_hopper(b_addr, 128, 256, 0);

    float d[8] = {0,0,0,0,0,0,0,0};
    wgmma_fence();
    wgmma_e4m3_ss_m64n16k32(d, a_desc, b_desc, /*scale_d*/ false);
    wgmma_commit_group();
    wgmma_wait_group<0>();

    for (int i = 0; i < 8; i++) out[tid * 8 + i] = d[i];
}

__global__ void wgmma_fp8_m64n64k32_e4m3_allones_kernel(float* __restrict__ out) {
    constexpr int M = 64, N = 64, K = 32;
    __shared__ __align__(128) uint8_t smem_A[M * K];
    __shared__ __align__(128) uint8_t smem_B[(N + 8) * K];
    int tid = threadIdx.x;

    for (int i = tid; i < M * K; i += 128) smem_A[i] = E4M3_ONE;
    for (int i = tid; i < (N + 8) * K; i += 128) smem_B[i] = E4M3_ONE;
    __syncthreads();

    uint32_t a_addr = smem_ptr_u32(&smem_A[0]);
    uint32_t b_addr = smem_ptr_u32(&smem_B[0]);
    uint64_t a_desc = build_smem_desc_hopper(a_addr, 128, 256, 0);
    uint64_t b_desc = build_smem_desc_hopper(b_addr, 128, 256, 0);

    float d[32] = {};
    wgmma_fence();
    wgmma_e4m3_ss_m64n64k32(d, a_desc, b_desc, /*scale_d*/ false);
    wgmma_commit_group();
    wgmma_wait_group<0>();

    for (int i = 0; i < 32; i++) out[tid * 32 + i] = d[i];
}

__global__ void wgmma_fp8_m64n8k32_e5m2_allones_kernel(float* __restrict__ out) {
    constexpr int M = 64, N = 8, K = 32;
    __shared__ __align__(128) uint8_t smem_A[M * K];
    __shared__ __align__(128) uint8_t smem_B[2 * 8 * K];
    int tid = threadIdx.x;

    for (int i = tid; i < M * K; i += 128) smem_A[i] = E5M2_ONE;
    for (int i = tid; i < 2 * 8 * K; i += 128) smem_B[i] = E5M2_ONE;
    __syncthreads();

    uint32_t a_addr = smem_ptr_u32(&smem_A[0]);
    uint32_t b_addr = smem_ptr_u32(&smem_B[0]);
    uint64_t a_desc = build_smem_desc_hopper(a_addr, 128, 256, 0);
    uint64_t b_desc = build_smem_desc_hopper(b_addr, 128, 256, 0);

    float d[4] = {0, 0, 0, 0};
    wgmma_fence();
    wgmma_e5m2_ss_m64n8k32(d, a_desc, b_desc, /*scale_d*/ false);
    wgmma_commit_group();
    wgmma_wait_group<0>();

    for (int i = 0; i < 4; i++) out[tid * 4 + i] = d[i];
}

__global__ void wgmma_fp8_m64n64k32_e5m2_allones_kernel(float* __restrict__ out) {
    constexpr int M = 64, N = 64, K = 32;
    __shared__ __align__(128) uint8_t smem_A[M * K];
    __shared__ __align__(128) uint8_t smem_B[(N + 8) * K];
    int tid = threadIdx.x;

    for (int i = tid; i < M * K; i += 128) smem_A[i] = E5M2_ONE;
    for (int i = tid; i < (N + 8) * K; i += 128) smem_B[i] = E5M2_ONE;
    __syncthreads();

    uint32_t a_addr = smem_ptr_u32(&smem_A[0]);
    uint32_t b_addr = smem_ptr_u32(&smem_B[0]);
    uint64_t a_desc = build_smem_desc_hopper(a_addr, 128, 256, 0);
    uint64_t b_desc = build_smem_desc_hopper(b_addr, 128, 256, 0);

    float d[32] = {};
    wgmma_fence();
    wgmma_e5m2_ss_m64n64k32(d, a_desc, b_desc, /*scale_d*/ false);
    wgmma_commit_group();
    wgmma_wait_group<0>();

    for (int i = 0; i < 32; i++) out[tid * 32 + i] = d[i];
}

// FP8 sparse 2:4: K_logical=64, K_compact=32. Expected D = K_active = 32.
__global__ void wgmma_fp8_m64n8k64_sp_e4m3_allones_kernel(float* __restrict__ out) {
    constexpr int M = 64, N = 8, K_LOG = 64, K_COMPACT = K_LOG / 2;
    __shared__ __align__(128) uint8_t smem_A[M * K_COMPACT];
    __shared__ __align__(128) uint8_t smem_B[2 * 8 * K_LOG];
    int tid = threadIdx.x;

    for (int i = tid; i < M * K_COMPACT; i += 128) smem_A[i] = E4M3_ONE;
    for (int i = tid; i < 2 * 8 * K_LOG; i += 128) smem_B[i] = E4M3_ONE;
    __syncthreads();

    uint32_t a_addr = smem_ptr_u32(&smem_A[0]);
    uint32_t b_addr = smem_ptr_u32(&smem_B[0]);
    uint64_t a_desc = build_smem_desc_hopper(a_addr, 128, 256, 0);
    uint64_t b_desc = build_smem_desc_hopper(b_addr, 128, 256, 0);

    float d[4] = {0,0,0,0};
    uint32_t e_meta = 0x44444444u;
    wgmma_fence();
    wgmma_e4m3_ss_sp_m64n8k64<0>(d, a_desc, b_desc, e_meta, /*scale_d*/ false);
    wgmma_commit_group();
    wgmma_wait_group<0>();

    for (int i = 0; i < 4; i++) out[tid * 4 + i] = d[i];
}

int main() {
    CUDA_CHECK(cudaFree(0));
    bool all_pass = true;
    const float expected_K = 32.0f;

    {
        const int NELT = 128 * 4;
        float* d_out; CUDA_CHECK(cudaMalloc(&d_out, NELT * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_out, 0, NELT * sizeof(float)));
        wgmma_fp8_m64n8k32_allones_kernel<<<1, 128>>>(d_out);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  m64n8k32 e4m3: LAUNCH FAIL: %s\n", cudaGetErrorString(err));
            all_pass = false;
        } else {
            float h_out[NELT];
            CUDA_CHECK(cudaMemcpy(h_out, d_out, NELT * sizeof(float), cudaMemcpyDeviceToHost));
            int bad = 0;
            float first_bad = -1.f; int first_idx = -1;
            for (int i = 0; i < NELT; i++) {
                if (fabsf(h_out[i] - expected_K) > 0.5f) {
                    if (bad == 0) { first_bad = h_out[i]; first_idx = i; }
                    bad++;
                }
            }
            if (bad == 0) printf("  m64n8k32 e4m3: OK (all 512 values == %.1f)\n", expected_K);
            else { printf("  m64n8k32 e4m3: FAIL (%d bad; first [%d]=%.3f)\n", bad, first_idx, first_bad);
                   // Dump per-thread-per-reg view
                   printf("    per-thread 4 regs (first 16 threads):\n");
                   for (int t = 0; t < 16; t++) {
                       printf("      t%d: %.1f %.1f %.1f %.1f\n", t,
                              h_out[t*4+0], h_out[t*4+1], h_out[t*4+2], h_out[t*4+3]);
                   }
                   all_pass = false; }
        }
        cudaFree(d_out);
    }

    {
        const int NELT = 128 * 8;
        float* d_out; CUDA_CHECK(cudaMalloc(&d_out, NELT * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_out, 0, NELT * sizeof(float)));
        wgmma_fp8_m64n16k32_allones_kernel<<<1, 128>>>(d_out);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  m64n16k32 e4m3: LAUNCH FAIL: %s\n", cudaGetErrorString(err));
            all_pass = false;
        } else {
            float h_out[NELT];
            CUDA_CHECK(cudaMemcpy(h_out, d_out, NELT * sizeof(float), cudaMemcpyDeviceToHost));
            int bad = 0;
            float first_bad = -1.f; int first_idx = -1;
            for (int i = 0; i < NELT; i++) {
                if (fabsf(h_out[i] - expected_K) > 0.5f) {
                    if (bad == 0) { first_bad = h_out[i]; first_idx = i; }
                    bad++;
                }
            }
            if (bad == 0) printf("  m64n16k32 e4m3: OK (all 1024 values == %.1f)\n", expected_K);
            else { printf("  m64n16k32 e4m3: FAIL (%d bad; first [%d]=%.3f)\n", bad, first_idx, first_bad);
                   all_pass = false; }
        }
        cudaFree(d_out);
    }

    // --- m64n64k32 e4m3 ---
    {
        const int NELT = 128 * 32;
        float* d_out; CUDA_CHECK(cudaMalloc(&d_out, NELT * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_out, 0, NELT * sizeof(float)));
        wgmma_fp8_m64n64k32_e4m3_allones_kernel<<<1, 128>>>(d_out);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  m64n64k32 e4m3: LAUNCH FAIL: %s\n", cudaGetErrorString(err));
            all_pass = false;
        } else {
            std::vector<float> h(NELT);
            CUDA_CHECK(cudaMemcpy(h.data(), d_out, NELT * sizeof(float), cudaMemcpyDeviceToHost));
            int bad = 0;
            for (int i = 0; i < NELT; i++)
                if (fabsf(h[i] - expected_K) > 0.5f) bad++;
            if (bad == 0) printf("  m64n64k32 e4m3: OK (all %d values == %.1f)\n", NELT, expected_K);
            else { printf("  m64n64k32 e4m3: FAIL (%d bad of %d)\n", bad, NELT); all_pass = false; }
        }
        cudaFree(d_out);
    }

    // --- m64n8k32 e5m2 ---
    {
        const int NELT = 128 * 4;
        float* d_out; CUDA_CHECK(cudaMalloc(&d_out, NELT * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_out, 0, NELT * sizeof(float)));
        wgmma_fp8_m64n8k32_e5m2_allones_kernel<<<1, 128>>>(d_out);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  m64n8k32 e5m2: LAUNCH FAIL: %s\n", cudaGetErrorString(err));
            all_pass = false;
        } else {
            std::vector<float> h(NELT);
            CUDA_CHECK(cudaMemcpy(h.data(), d_out, NELT * sizeof(float), cudaMemcpyDeviceToHost));
            int bad = 0;
            for (int i = 0; i < NELT; i++)
                if (fabsf(h[i] - expected_K) > 0.5f) bad++;
            if (bad == 0) printf("  m64n8k32 e5m2: OK (all %d values == %.1f)\n", NELT, expected_K);
            else { printf("  m64n8k32 e5m2: FAIL (%d bad of %d)\n", bad, NELT); all_pass = false; }
        }
        cudaFree(d_out);
    }

    // --- m64n64k32 e5m2 ---
    {
        const int NELT = 128 * 32;
        float* d_out; CUDA_CHECK(cudaMalloc(&d_out, NELT * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_out, 0, NELT * sizeof(float)));
        wgmma_fp8_m64n64k32_e5m2_allones_kernel<<<1, 128>>>(d_out);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  m64n64k32 e5m2: LAUNCH FAIL: %s\n", cudaGetErrorString(err));
            all_pass = false;
        } else {
            std::vector<float> h(NELT);
            CUDA_CHECK(cudaMemcpy(h.data(), d_out, NELT * sizeof(float), cudaMemcpyDeviceToHost));
            int bad = 0;
            for (int i = 0; i < NELT; i++)
                if (fabsf(h[i] - expected_K) > 0.5f) bad++;
            if (bad == 0) printf("  m64n64k32 e5m2: OK (all %d values == %.1f)\n", NELT, expected_K);
            else { printf("  m64n64k32 e5m2: FAIL (%d bad of %d)\n", bad, NELT); all_pass = false; }
        }
        cudaFree(d_out);
    }

    // --- perf: m64n8k32 e4m3 ---
    {
        const int NELT = 128 * 4;
        float* d_out; CUDA_CHECK(cudaMalloc(&d_out, NELT * sizeof(float)));
        GpuTimer t;
        const int ITERS = 200;
        t.begin();
        for (int i = 0; i < ITERS; i++) wgmma_fp8_m64n8k32_allones_kernel<<<1, 128>>>(d_out);
        t.end();
        printf("  perf: %.2f us/launch (wgmma m64n8k32 e4m3 SS, 1 CTA x 128 thr)\n",
               t.elapsed_ms() * 1000.0f / ITERS);
        cudaFree(d_out);
    }

    // --- m64n8k64 SPARSE e4m3 ---
    {
        const int NELT = 128 * 4;
        float* d_out; CUDA_CHECK(cudaMalloc(&d_out, NELT * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_out, 0, NELT * sizeof(float)));
        wgmma_fp8_m64n8k64_sp_e4m3_allones_kernel<<<1, 128>>>(d_out);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  m64n8k64 sparse e4m3: LAUNCH FAIL: %s\n", cudaGetErrorString(err));
            all_pass = false;
        } else {
            std::vector<float> h(NELT);
            CUDA_CHECK(cudaMemcpy(h.data(), d_out, NELT * sizeof(float), cudaMemcpyDeviceToHost));
            const float expected_sp = 32.0f;
            int bad = 0;
            for (int i = 0; i < NELT; i++)
                if (fabsf(h[i] - expected_sp) > 0.5f) bad++;
            if (bad == 0) printf("  m64n8k64 sparse e4m3: OK (all %d == %.0f)\n", NELT, expected_sp);
            else { printf("  m64n8k64 sparse e4m3: FAIL (%d bad of %d)\n", bad, NELT); all_pass = false; }
        }
        cudaFree(d_out);
    }

    if (all_pass) { PASS(); return 0; }
    else { FAIL("some subtests failed"); return 1; }
}

#endif  // PL_AGENTIC_SM90A
