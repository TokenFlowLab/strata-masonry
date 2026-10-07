#if defined(PL_AGENTIC_SM90A)
// ARCH: sm_90a
// 53_wgmma_f16_ss_test.cu -- runtime test for wgmma f16 ss
//
// Runtime test: wgmma.mma_async FP16 SS -- numerical correctness.
//
// Strategy: use all-ones A (64xK) and all-ones B (NxK) matrices. Since every
// FP16 byte value is the same (0x3C3C -> 1.0,1.0 packed pair), the exact SMEM
// layout/swizzle used by WGMMA does not affect the computed values: the MMA
// reads all-ones no matter which byte it loads. Expected: every accumulator
// element = K (accumulate K products of 1.0 * 1.0).
//
// This exercises:
//   * SMEM descriptor construction (PTX 9.7.17.5.1.2.2 bit layout)
//   * wgmma.fence / wgmma.commit_group / wgmma.wait_group lifecycle
//   * Two shapes: m64n8k16 (4 f32 regs/thread) and m64n16k16 (8 f32 regs)

#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "test_utils.cuh"
#include "43_smem_desc_hopper.cuh"
#include "53_wgmma_f16_ss.cuh"
#include "59_wgmma_fence_commit_wait.cuh"

// m64n8k16: 4 f32 regs per thread, 128 threads -> 512 floats total.
__global__ void wgmma_m64n8k16_allones_kernel(float* __restrict__ out) {
    constexpr int M = 64, N = 8, K = 16;
    __shared__ __align__(128) half smem_A[M * K];  // 64x16 FP16 = 2048 bytes
    __shared__ __align__(128) half smem_B[N * K];  // 8x16  FP16 =  256 bytes
    int tid = threadIdx.x;

    // Fill SMEM with FP16 1.0 -- every element is 1.0.
    // 128 threads, M*K = 1024 halves for A, N*K=128 halves for B.
    #pragma unroll
    for (int i = tid; i < M * K; i += 128) smem_A[i] = __float2half(1.0f);
    if (tid < N * K) smem_B[tid] = __float2half(1.0f);
    __syncthreads();

    // Build descriptors. INTERLEAVE (no swizzle) K-major layout.
    // For K-major INTERLEAVE: LBO = row stride in bytes, SBO = 8-row stride.
    // A is 64 rows x 16 FP16 -> row_stride = 16*2 = 32 bytes; 8-row = 256 bytes.
    // B is  8 rows x 16 FP16 -> same per-row stride; 8 rows == full matrix.
    uint32_t a_addr = smem_ptr_u32(&smem_A[0]);
    uint32_t b_addr = smem_ptr_u32(&smem_B[0]);

    // layout_type_ = 0 (INTERLEAVE / no swizzle).
    uint64_t a_desc = build_smem_desc_hopper(a_addr,
                                             /*LBO*/ 32,
                                             /*SBO*/ 256,
                                             /*swizzle*/ 0);
    uint64_t b_desc = build_smem_desc_hopper(b_addr,
                                             /*LBO*/ 32,
                                             /*SBO*/ 256,
                                             /*swizzle*/ 0);

    // scale_d=false -> overwrite (D = A*B), avoids needing to zero accumulator.
    float d[4] = {0.f, 0.f, 0.f, 0.f};
    wgmma_fence();
    wgmma_f16_ss_m64n8k16(d, a_desc, b_desc, /*scale_d*/ false);
    wgmma_commit_group();
    wgmma_wait_group<0>();

    // Write accumulator: each thread writes 4 f32 regs contiguously.
    out[tid * 4 + 0] = d[0];
    out[tid * 4 + 1] = d[1];
    out[tid * 4 + 2] = d[2];
    out[tid * 4 + 3] = d[3];
}

// m64n16k16: 8 f32 regs per thread, 128 threads -> 1024 floats total.
__global__ void wgmma_m64n16k16_allones_kernel(float* __restrict__ out) {
    constexpr int M = 64, N = 16, K = 16;
    __shared__ __align__(128) half smem_A[M * K];  // 2048 bytes
    __shared__ __align__(128) half smem_B[N * K];  //  512 bytes
    int tid = threadIdx.x;

    #pragma unroll
    for (int i = tid; i < M * K; i += 128) smem_A[i] = __float2half(1.0f);
    #pragma unroll
    for (int i = tid; i < N * K; i += 128) smem_B[i] = __float2half(1.0f);
    __syncthreads();

    uint32_t a_addr = smem_ptr_u32(&smem_A[0]);
    uint32_t b_addr = smem_ptr_u32(&smem_B[0]);
    uint64_t a_desc = build_smem_desc_hopper(a_addr, 32, 256, 0);
    uint64_t b_desc = build_smem_desc_hopper(b_addr, 32, 256, 0);

    float d[8] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    wgmma_fence();
    wgmma_f16_ss_m64n16k16(d, a_desc, b_desc, /*scale_d*/ false);
    wgmma_commit_group();
    wgmma_wait_group<0>();

    #pragma unroll
    for (int i = 0; i < 8; i++) out[tid * 8 + i] = d[i];
}

// m64n8k32 SPARSE 2:4 (logical K=32, A stored compact 16x16 -> 256 halves):
// A_compact all-ones, B (32 K-rows x 8 N-cols) all-ones, e_meta=0x44444444
// selects positions {0,1} of every 4-group -> K_active = 16. Expected D = 16.
__global__ void wgmma_m64n8k32_sp_allones_kernel(float* __restrict__ out) {
    constexpr int M = 64, N = 8, K_LOG = 32, K_COMPACT = K_LOG / 2;
    __shared__ __align__(128) half smem_A[M * K_COMPACT];   // 2048 bytes
    __shared__ __align__(128) half smem_B[N * K_LOG];       //  512 bytes
    int tid = threadIdx.x;

    for (int i = tid; i < M * K_COMPACT; i += 128) smem_A[i] = __float2half(1.0f);
    if (tid < N * K_LOG) smem_B[tid] = __float2half(1.0f);
    __syncthreads();

    uint32_t a_addr = smem_ptr_u32(&smem_A[0]);
    uint32_t b_addr = smem_ptr_u32(&smem_B[0]);
    uint64_t a_desc = build_smem_desc_hopper(a_addr, 32, 256, 0);
    uint64_t b_desc = build_smem_desc_hopper(b_addr, 32, 256, 0);

    float d[4] = {0.f, 0.f, 0.f, 0.f};
    uint32_t e_meta = 0x44444444u;
    wgmma_fence();
    wgmma_f16_ss_sp_m64n8k32<0>(d, a_desc, b_desc, e_meta, /*scale_d*/ false);
    wgmma_commit_group();
    wgmma_wait_group<0>();

    out[tid * 4 + 0] = d[0];
    out[tid * 4 + 1] = d[1];
    out[tid * 4 + 2] = d[2];
    out[tid * 4 + 3] = d[3];
}

// m64n128k16: 64 f32 regs/thread (production tile size).
__global__ void wgmma_m64n128k16_allones_kernel(float* __restrict__ out) {
    constexpr int M = 64, N = 128, K = 16;
    __shared__ __align__(128) half smem_A[M * K];   // 2048 bytes
    __shared__ __align__(128) half smem_B[N * K];   // 4096 bytes
    int tid = threadIdx.x;

    for (int i = tid; i < M * K; i += 128) smem_A[i] = __float2half(1.0f);
    for (int i = tid; i < N * K; i += 128) smem_B[i] = __float2half(1.0f);
    __syncthreads();

    uint32_t a_addr = smem_ptr_u32(&smem_A[0]);
    uint32_t b_addr = smem_ptr_u32(&smem_B[0]);
    uint64_t a_desc = build_smem_desc_hopper(a_addr, 32, 256, 0);
    uint64_t b_desc = build_smem_desc_hopper(b_addr, 32, 256, 0);

    float d[64];
    #pragma unroll
    for (int i = 0; i < 64; i++) d[i] = 0.f;
    wgmma_fence();
    wgmma_f16_ss_m64n128k16(d, a_desc, b_desc, /*scale_d*/ false);
    wgmma_commit_group();
    wgmma_wait_group<0>();

    #pragma unroll
    for (int i = 0; i < 64; i++) out[tid * 64 + i] = d[i];
}

// m64n256k16: 128 f32 regs/thread (corner-case tile).
__global__ void wgmma_m64n256k16_allones_kernel(float* __restrict__ out) {
    constexpr int M = 64, N = 256, K = 16;
    __shared__ __align__(128) half smem_A[M * K];   // 2048 bytes
    __shared__ __align__(128) half smem_B[N * K];   // 8192 bytes
    int tid = threadIdx.x;

    for (int i = tid; i < M * K; i += 128) smem_A[i] = __float2half(1.0f);
    for (int i = tid; i < N * K; i += 128) smem_B[i] = __float2half(1.0f);
    __syncthreads();

    uint32_t a_addr = smem_ptr_u32(&smem_A[0]);
    uint32_t b_addr = smem_ptr_u32(&smem_B[0]);
    uint64_t a_desc = build_smem_desc_hopper(a_addr, 32, 256, 0);
    uint64_t b_desc = build_smem_desc_hopper(b_addr, 32, 256, 0);

    float d[128];
    #pragma unroll
    for (int i = 0; i < 128; i++) d[i] = 0.f;
    wgmma_fence();
    wgmma_f16_ss_m64n256k16(d, a_desc, b_desc, /*scale_d*/ false);
    wgmma_commit_group();
    wgmma_wait_group<0>();

    #pragma unroll
    for (int i = 0; i < 128; i++) out[tid * 128 + i] = d[i];
}

int main() {
    CUDA_CHECK(cudaFree(0));
    bool all_pass = true;
    const float expected_K = 16.0f;

    // --- test 1: m64n8k16 ---
    {
        const int NELT = 128 * 4;
        float* d_out; CUDA_CHECK(cudaMalloc(&d_out, NELT * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_out, 0, NELT * sizeof(float)));
        wgmma_m64n8k16_allones_kernel<<<1, 128>>>(d_out);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  m64n8k16: LAUNCH FAIL: %s\n", cudaGetErrorString(err));
            all_pass = false;
        } else {
            float h_out[NELT];
            CUDA_CHECK(cudaMemcpy(h_out, d_out, NELT * sizeof(float), cudaMemcpyDeviceToHost));
            int bad = 0;
            float first_bad_val = -1.f; int first_bad_idx = -1;
            for (int i = 0; i < NELT; i++) {
                if (fabsf(h_out[i] - expected_K) > 1e-3f) {
                    if (bad == 0) { first_bad_val = h_out[i]; first_bad_idx = i; }
                    bad++;
                }
            }
            if (bad == 0) {
                printf("  m64n8k16: OK (all 512 values == %.1f)\n", expected_K);
            } else {
                printf("  m64n8k16: FAIL (%d/%d mismatches; first [%d]=%.3f)\n",
                       bad, NELT, first_bad_idx, first_bad_val);
                // Show a few samples
                printf("    samples: [0]=%.3f [4]=%.3f [100]=%.3f [511]=%.3f\n",
                       h_out[0], h_out[4], h_out[100], h_out[511]);
                all_pass = false;
            }
        }
        cudaFree(d_out);
    }

    // --- perf: m64n8k16 ---
    {
        const int NELT = 128 * 4;
        float* d_out; CUDA_CHECK(cudaMalloc(&d_out, NELT * sizeof(float)));
        GpuTimer t;
        const int ITERS = 200;
        t.begin();
        for (int i = 0; i < ITERS; i++) wgmma_m64n8k16_allones_kernel<<<1, 128>>>(d_out);
        t.end();
        printf("  perf: %.2f us/launch (wgmma m64n8k16 f16 SS, 1 CTA x 128 thr)\n",
               t.elapsed_ms() * 1000.0f / ITERS);
        cudaFree(d_out);
    }

    // --- test 2: m64n16k16 ---
    {
        const int NELT = 128 * 8;
        float* d_out; CUDA_CHECK(cudaMalloc(&d_out, NELT * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_out, 0, NELT * sizeof(float)));
        wgmma_m64n16k16_allones_kernel<<<1, 128>>>(d_out);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  m64n16k16: LAUNCH FAIL: %s\n", cudaGetErrorString(err));
            all_pass = false;
        } else {
            float h_out[NELT];
            CUDA_CHECK(cudaMemcpy(h_out, d_out, NELT * sizeof(float), cudaMemcpyDeviceToHost));
            int bad = 0;
            float first_bad_val = -1.f; int first_bad_idx = -1;
            for (int i = 0; i < NELT; i++) {
                if (fabsf(h_out[i] - expected_K) > 1e-3f) {
                    if (bad == 0) { first_bad_val = h_out[i]; first_bad_idx = i; }
                    bad++;
                }
            }
            if (bad == 0) {
                printf("  m64n16k16: OK (all 1024 values == %.1f)\n", expected_K);
            } else {
                printf("  m64n16k16: FAIL (%d/%d mismatches; first [%d]=%.3f)\n",
                       bad, NELT, first_bad_idx, first_bad_val);
                printf("    samples: [0]=%.3f [8]=%.3f [100]=%.3f [1023]=%.3f\n",
                       h_out[0], h_out[8], h_out[100], h_out[1023]);
                all_pass = false;
            }
        }
        cudaFree(d_out);
    }

    // --- perf: m64n16k16 ---
    {
        const int NELT = 128 * 8;
        float* d_out; CUDA_CHECK(cudaMalloc(&d_out, NELT * sizeof(float)));
        GpuTimer t;
        const int ITERS = 200;
        t.begin();
        for (int i = 0; i < ITERS; i++) wgmma_m64n16k16_allones_kernel<<<1, 128>>>(d_out);
        t.end();
        printf("  perf: %.2f us/launch (wgmma m64n16k16 f16 SS, 1 CTA x 128 thr)\n",
               t.elapsed_ms() * 1000.0f / ITERS);
        cudaFree(d_out);
    }

    // --- MED: m64n128k16 (production tile size) ---
    {
        const int NELT = 128 * 64;
        float* d_out; CUDA_CHECK(cudaMalloc(&d_out, NELT * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_out, 0, NELT * sizeof(float)));
        wgmma_m64n128k16_allones_kernel<<<1, 128>>>(d_out);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  m64n128k16: LAUNCH FAIL: %s\n", cudaGetErrorString(err));
            all_pass = false;
        } else {
            std::vector<float> h_out(NELT);
            CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, NELT * sizeof(float), cudaMemcpyDeviceToHost));
            int bad = 0; int first_bad_idx = -1; float first_bad_val = -1.f;
            for (int i = 0; i < NELT; i++) {
                if (fabsf(h_out[i] - expected_K) > 1e-3f) {
                    if (bad == 0) { first_bad_idx = i; first_bad_val = h_out[i]; }
                    bad++;
                }
            }
            if (bad == 0) printf("  m64n128k16: OK (all %d values == %.1f)\n", NELT, expected_K);
            else { printf("  m64n128k16: FAIL (%d/%d mismatches; first [%d]=%.3f)\n",
                          bad, NELT, first_bad_idx, first_bad_val);
                   all_pass = false; }
        }
        cudaFree(d_out);
    }

    // --- MED: m64n256k16 (corner-case tile size) ---
    {
        const int NELT = 128 * 128;
        float* d_out; CUDA_CHECK(cudaMalloc(&d_out, NELT * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_out, 0, NELT * sizeof(float)));
        wgmma_m64n256k16_allones_kernel<<<1, 128>>>(d_out);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  m64n256k16: LAUNCH FAIL: %s\n", cudaGetErrorString(err));
            all_pass = false;
        } else {
            std::vector<float> h_out(NELT);
            CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, NELT * sizeof(float), cudaMemcpyDeviceToHost));
            int bad = 0; int first_bad_idx = -1; float first_bad_val = -1.f;
            for (int i = 0; i < NELT; i++) {
                if (fabsf(h_out[i] - expected_K) > 1e-3f) {
                    if (bad == 0) { first_bad_idx = i; first_bad_val = h_out[i]; }
                    bad++;
                }
            }
            if (bad == 0) printf("  m64n256k16: OK (all %d values == %.1f)\n", NELT, expected_K);
            else { printf("  m64n256k16: FAIL (%d/%d mismatches; first [%d]=%.3f)\n",
                          bad, NELT, first_bad_idx, first_bad_val);
                   all_pass = false; }
        }
        cudaFree(d_out);
    }

    // --- test 3: m64n8k32 SPARSE ---
    {
        const int NELT = 128 * 4;
        float* d_out; CUDA_CHECK(cudaMalloc(&d_out, NELT * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_out, 0, NELT * sizeof(float)));
        wgmma_m64n8k32_sp_allones_kernel<<<1, 128>>>(d_out);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  m64n8k32 sparse: LAUNCH FAIL: %s\n", cudaGetErrorString(err));
            all_pass = false;
        } else {
            std::vector<float> h_out(NELT);
            CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, NELT * sizeof(float), cudaMemcpyDeviceToHost));
            // K_logical=32, K_active=16 -> D[m,n] = 16 (16 ones contributing)
            const float expected_sp = 16.0f;
            int bad = 0; int first_idx = -1; float first_val = 0.f;
            for (int i = 0; i < NELT; i++) {
                if (fabsf(h_out[i] - expected_sp) > 1e-3f) {
                    if (bad == 0) { first_idx = i; first_val = h_out[i]; }
                    bad++;
                }
            }
            if (bad == 0) printf("  m64n8k32 sparse: OK (all %d values == %.0f)\n", NELT, expected_sp);
            else { printf("  m64n8k32 sparse: FAIL (%d/%d bad; first [%d]=%.3f exp %.0f)\n",
                          bad, NELT, first_idx, first_val, expected_sp);
                   all_pass = false; }
        }
        cudaFree(d_out);
    }

    if (all_pass) { PASS(); return 0; }
    else { FAIL("some subtests failed"); return 1; }
}

#endif  // PL_AGENTIC_SM90A
