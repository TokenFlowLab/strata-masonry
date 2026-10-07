#if defined(PL_AGENTIC_SM90A)
// ARCH: sm_90a
// 58_wgmma_i8_ss_test.cu -- runtime test for wgmma i8 ss
//
// Runtime test: wgmma.mma_async INT8 SS -- numerical correctness.
// INT8 has K=32. All-ones (int8 = 1) test: D = A*B with A,B all 1 -> D == K == 32.

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include "test_utils.cuh"
#include "43_smem_desc_hopper.cuh"
#include "58_wgmma_i8_ss.cuh"
#include "59_wgmma_fence_commit_wait.cuh"

__global__ void wgmma_i8_m64n8k32_allones_kernel(int32_t* __restrict__ out) {
    constexpr int M = 64, N = 8, K = 32;
    __shared__ __align__(128) int8_t smem_A[M * K];
    __shared__ __align__(128) int8_t smem_B[(N + 8) * K];  // headroom
    int tid = threadIdx.x;

    for (int i = tid; i < M * K; i += 128) smem_A[i] = (int8_t)1;
    for (int i = tid; i < (N + 8) * K; i += 128) smem_B[i] = (int8_t)1;
    __syncthreads();

    uint32_t a_addr = smem_ptr_u32(&smem_A[0]);
    uint32_t b_addr = smem_ptr_u32(&smem_B[0]);
    // K=32 s8 = 32 bytes per row. INTERLEAVE K-major canonical: LBO=128, SBO=256.
    uint64_t a_desc = build_smem_desc_hopper(a_addr, 128, 256, 0);
    uint64_t b_desc = build_smem_desc_hopper(b_addr, 128, 256, 0);

    int32_t d[4] = {0, 0, 0, 0};
    wgmma_fence();
    wgmma_i8_ss_m64n8k32(d, a_desc, b_desc, /*scale_d*/ false);
    wgmma_commit_group();
    wgmma_wait_group<0>();

    for (int i = 0; i < 4; i++) out[tid * 4 + i] = d[i];
}

__global__ void wgmma_i8_m64n16k32_allones_kernel(int32_t* __restrict__ out) {
    constexpr int M = 64, N = 16, K = 32;
    __shared__ __align__(128) int8_t smem_A[M * K];
    __shared__ __align__(128) int8_t smem_B[(N + 8) * K];
    int tid = threadIdx.x;

    for (int i = tid; i < M * K; i += 128) smem_A[i] = (int8_t)1;
    for (int i = tid; i < (N + 8) * K; i += 128) smem_B[i] = (int8_t)1;
    __syncthreads();

    uint32_t a_addr = smem_ptr_u32(&smem_A[0]);
    uint32_t b_addr = smem_ptr_u32(&smem_B[0]);
    uint64_t a_desc = build_smem_desc_hopper(a_addr, 128, 256, 0);
    uint64_t b_desc = build_smem_desc_hopper(b_addr, 128, 256, 0);

    int32_t d[8] = {0,0,0,0,0,0,0,0};
    wgmma_fence();
    wgmma_i8_ss_m64n16k32(d, a_desc, b_desc, /*scale_d*/ false);
    wgmma_commit_group();
    wgmma_wait_group<0>();

    for (int i = 0; i < 8; i++) out[tid * 8 + i] = d[i];
}

// INT8 sparse 2:4: K_logical=64, K_compact=32. Expected D = K_active = 32.
__global__ void wgmma_i8_m64n8k64_sp_allones_kernel(int32_t* __restrict__ out) {
    constexpr int M = 64, N = 8, K_LOG = 64, K_COMPACT = K_LOG / 2;
    __shared__ __align__(128) int8_t smem_A[M * K_COMPACT];
    __shared__ __align__(128) int8_t smem_B[(N + 8) * K_LOG];
    int tid = threadIdx.x;

    for (int i = tid; i < M * K_COMPACT; i += 128) smem_A[i] = (int8_t)1;
    for (int i = tid; i < (N + 8) * K_LOG; i += 128) smem_B[i] = (int8_t)1;
    __syncthreads();

    uint32_t a_addr = smem_ptr_u32(&smem_A[0]);
    uint32_t b_addr = smem_ptr_u32(&smem_B[0]);
    uint64_t a_desc = build_smem_desc_hopper(a_addr, 128, 256, 0);
    uint64_t b_desc = build_smem_desc_hopper(b_addr, 128, 256, 0);

    int32_t d[4] = {0, 0, 0, 0};
    uint32_t e_meta = 0x44444444u;
    wgmma_fence();
    wgmma_i8_ss_sp_m64n8k64<0>(d, a_desc, b_desc, e_meta, /*scale_d*/ false);
    wgmma_commit_group();
    wgmma_wait_group<0>();

    for (int i = 0; i < 4; i++) out[tid * 4 + i] = d[i];
}

// MED: m64n64k32 s8.s8 (production tile size).
__global__ void wgmma_i8_m64n64k32_allones_kernel(int32_t* __restrict__ out) {
    constexpr int M = 64, N = 64, K = 32;
    __shared__ __align__(128) int8_t smem_A[M * K];
    __shared__ __align__(128) int8_t smem_B[N * K];
    int tid = threadIdx.x;

    for (int i = tid; i < M * K; i += 128) smem_A[i] = (int8_t)1;
    for (int i = tid; i < N * K; i += 128) smem_B[i] = (int8_t)1;
    __syncthreads();

    uint32_t a_addr = smem_ptr_u32(&smem_A[0]);
    uint32_t b_addr = smem_ptr_u32(&smem_B[0]);
    uint64_t a_desc = build_smem_desc_hopper(a_addr, 128, 256, 0);
    uint64_t b_desc = build_smem_desc_hopper(b_addr, 128, 256, 0);

    int32_t d[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) d[i] = 0;
    wgmma_fence();
    wgmma_i8_ss_m64n64k32(d, a_desc, b_desc, /*scale_d*/ false);
    wgmma_commit_group();
    wgmma_wait_group<0>();

    #pragma unroll
    for (int i = 0; i < 32; i++) out[tid * 32 + i] = d[i];
}

// MED: u8.u8, s8.u8, u8.s8 mixed-sign m64n8k32. With all-ones (treated as 1
// in both signed and unsigned), expected accumulator == K == 32.
__global__ void wgmma_u8_m64n8k32_allones_kernel(int32_t* __restrict__ out) {
    constexpr int M = 64, N = 8, K = 32;
    __shared__ __align__(128) uint8_t smem_A[M * K];
    __shared__ __align__(128) uint8_t smem_B[(N + 8) * K];
    int tid = threadIdx.x;

    for (int i = tid; i < M * K; i += 128) smem_A[i] = (uint8_t)1;
    for (int i = tid; i < (N + 8) * K; i += 128) smem_B[i] = (uint8_t)1;
    __syncthreads();

    uint32_t a_addr = smem_ptr_u32(&smem_A[0]);
    uint32_t b_addr = smem_ptr_u32(&smem_B[0]);
    uint64_t a_desc = build_smem_desc_hopper(a_addr, 128, 256, 0);
    uint64_t b_desc = build_smem_desc_hopper(b_addr, 128, 256, 0);

    int32_t d[4] = {0,0,0,0};
    wgmma_fence();
    wgmma_u8_ss_m64n8k32(d, a_desc, b_desc, /*scale_d*/ false);
    wgmma_commit_group();
    wgmma_wait_group<0>();
    for (int i = 0; i < 4; i++) out[tid * 4 + i] = d[i];
}

__global__ void wgmma_s8u8_m64n8k32_allones_kernel(int32_t* __restrict__ out) {
    constexpr int M = 64, N = 8, K = 32;
    __shared__ __align__(128) int8_t  smem_A[M * K];
    __shared__ __align__(128) uint8_t smem_B[(N + 8) * K];
    int tid = threadIdx.x;

    for (int i = tid; i < M * K; i += 128) smem_A[i] = (int8_t)1;
    for (int i = tid; i < (N + 8) * K; i += 128) smem_B[i] = (uint8_t)1;
    __syncthreads();

    uint32_t a_addr = smem_ptr_u32(&smem_A[0]);
    uint32_t b_addr = smem_ptr_u32(&smem_B[0]);
    uint64_t a_desc = build_smem_desc_hopper(a_addr, 128, 256, 0);
    uint64_t b_desc = build_smem_desc_hopper(b_addr, 128, 256, 0);

    int32_t d[4] = {0,0,0,0};
    wgmma_fence();
    wgmma_s8u8_ss_m64n8k32(d, a_desc, b_desc, /*scale_d*/ false);
    wgmma_commit_group();
    wgmma_wait_group<0>();
    for (int i = 0; i < 4; i++) out[tid * 4 + i] = d[i];
}

__global__ void wgmma_u8s8_m64n8k32_allones_kernel(int32_t* __restrict__ out) {
    constexpr int M = 64, N = 8, K = 32;
    __shared__ __align__(128) uint8_t smem_A[M * K];
    __shared__ __align__(128) int8_t  smem_B[(N + 8) * K];
    int tid = threadIdx.x;

    for (int i = tid; i < M * K; i += 128) smem_A[i] = (uint8_t)1;
    for (int i = tid; i < (N + 8) * K; i += 128) smem_B[i] = (int8_t)1;
    __syncthreads();

    uint32_t a_addr = smem_ptr_u32(&smem_A[0]);
    uint32_t b_addr = smem_ptr_u32(&smem_B[0]);
    uint64_t a_desc = build_smem_desc_hopper(a_addr, 128, 256, 0);
    uint64_t b_desc = build_smem_desc_hopper(b_addr, 128, 256, 0);

    int32_t d[4] = {0,0,0,0};
    wgmma_fence();
    wgmma_u8s8_ss_m64n8k32(d, a_desc, b_desc, /*scale_d*/ false);
    wgmma_commit_group();
    wgmma_wait_group<0>();
    for (int i = 0; i < 4; i++) out[tid * 4 + i] = d[i];
}

// MED: .satfinite -- intentionally drive to overflow to verify saturation.
// All A elements = 127 (s8 max), all B elements = 127. K=32 contributions of
// 127*127 = 16129 each, sum = 32 * 16129 = 516128 -- still within int32 range,
// so saturation does not kick in for K=32. Verify the modifier compiles and
// produces the same value as the non-saturating wrapper (functional check).
// The PTX-level effect (clamp to INT_MAX/MIN on overflow) requires K large
// enough to overflow int32, which is not reachable in a single wgmma op.
__global__ void wgmma_i8_m64n8k32_satfinite_kernel(int32_t* __restrict__ out,
                                                    int8_t fillA, int8_t fillB) {
    constexpr int M = 64, N = 8, K = 32;
    __shared__ __align__(128) int8_t smem_A[M * K];
    __shared__ __align__(128) int8_t smem_B[(N + 8) * K];
    int tid = threadIdx.x;

    for (int i = tid; i < M * K; i += 128) smem_A[i] = fillA;
    for (int i = tid; i < (N + 8) * K; i += 128) smem_B[i] = fillB;
    __syncthreads();

    uint32_t a_addr = smem_ptr_u32(&smem_A[0]);
    uint32_t b_addr = smem_ptr_u32(&smem_B[0]);
    uint64_t a_desc = build_smem_desc_hopper(a_addr, 128, 256, 0);
    uint64_t b_desc = build_smem_desc_hopper(b_addr, 128, 256, 0);

    int32_t d[4] = {0,0,0,0};
    wgmma_fence();
    wgmma_i8_ss_m64n8k32_satfinite(d, a_desc, b_desc, /*scale_d*/ false);
    wgmma_commit_group();
    wgmma_wait_group<0>();
    for (int i = 0; i < 4; i++) out[tid * 4 + i] = d[i];
}

int main() {
    CUDA_CHECK(cudaFree(0));
    bool all_pass = true;
    const int32_t expected_K = 32;

    {
        const int NELT = 128 * 4;
        int32_t* d_out; CUDA_CHECK(cudaMalloc(&d_out, NELT * sizeof(int32_t)));
        CUDA_CHECK(cudaMemset(d_out, 0, NELT * sizeof(int32_t)));
        wgmma_i8_m64n8k32_allones_kernel<<<1, 128>>>(d_out);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  m64n8k32 s8: LAUNCH FAIL: %s\n", cudaGetErrorString(err));
            all_pass = false;
        } else {
            int32_t h_out[NELT];
            CUDA_CHECK(cudaMemcpy(h_out, d_out, NELT * sizeof(int32_t), cudaMemcpyDeviceToHost));
            int bad = 0;
            for (int i = 0; i < NELT; i++) if (h_out[i] != expected_K) bad++;
            if (bad == 0) printf("  m64n8k32 s8: OK (all 512 values == %d)\n", expected_K);
            else { printf("  m64n8k32 s8: FAIL (%d bad)\n", bad);
                   printf("    samples: [0]=%d [4]=%d [100]=%d [511]=%d\n",
                          h_out[0], h_out[4], h_out[100], h_out[511]);
                   all_pass = false; }
        }
        cudaFree(d_out);
    }

    {
        const int NELT = 128 * 8;
        int32_t* d_out; CUDA_CHECK(cudaMalloc(&d_out, NELT * sizeof(int32_t)));
        CUDA_CHECK(cudaMemset(d_out, 0, NELT * sizeof(int32_t)));
        wgmma_i8_m64n16k32_allones_kernel<<<1, 128>>>(d_out);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  m64n16k32 s8: LAUNCH FAIL: %s\n", cudaGetErrorString(err));
            all_pass = false;
        } else {
            int32_t h_out[NELT];
            CUDA_CHECK(cudaMemcpy(h_out, d_out, NELT * sizeof(int32_t), cudaMemcpyDeviceToHost));
            int bad = 0;
            for (int i = 0; i < NELT; i++) if (h_out[i] != expected_K) bad++;
            if (bad == 0) printf("  m64n16k32 s8: OK (all 1024 values == %d)\n", expected_K);
            else { printf("  m64n16k32 s8: FAIL (%d bad)\n", bad);
                   all_pass = false; }
        }
        cudaFree(d_out);
    }

    // --- perf: m64n8k32 s8 ---
    {
        const int NELT = 128 * 4;
        int32_t* d_out; CUDA_CHECK(cudaMalloc(&d_out, NELT * sizeof(int32_t)));
        GpuTimer t;
        const int ITERS = 200;
        t.begin();
        for (int i = 0; i < ITERS; i++) wgmma_i8_m64n8k32_allones_kernel<<<1, 128>>>(d_out);
        t.end();
        printf("  perf: %.2f us/launch (wgmma m64n8k32 s8 SS, 1 CTA x 128 thr)\n",
               t.elapsed_ms() * 1000.0f / ITERS);
        cudaFree(d_out);
    }

    // --- MED: m64n64k32 s8 (production tile size) ---
    {
        const int NELT = 128 * 32;
        int32_t* d_out; CUDA_CHECK(cudaMalloc(&d_out, NELT * sizeof(int32_t)));
        CUDA_CHECK(cudaMemset(d_out, 0, NELT * sizeof(int32_t)));
        wgmma_i8_m64n64k32_allones_kernel<<<1, 128>>>(d_out);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  m64n64k32 s8: LAUNCH FAIL: %s\n", cudaGetErrorString(err));
            all_pass = false;
        } else {
            int32_t* h_out = new int32_t[NELT];
            CUDA_CHECK(cudaMemcpy(h_out, d_out, NELT * sizeof(int32_t), cudaMemcpyDeviceToHost));
            int bad = 0; int first_bad_idx = -1; int first_bad_val = -1;
            for (int i = 0; i < NELT; i++) if (h_out[i] != expected_K) { if(bad==0){first_bad_idx=i; first_bad_val=h_out[i];} bad++; }
            if (bad == 0) printf("  m64n64k32 s8: OK (all %d == %d)\n", NELT, expected_K);
            else { printf("  m64n64k32 s8: FAIL (%d/%d bad; first [%d]=%d)\n", bad, NELT, first_bad_idx, first_bad_val); all_pass = false; }
            delete[] h_out;
        }
        cudaFree(d_out);
    }

    // --- MED: u8.u8 mixed-sign m64n8k32 ---
    {
        const int NELT = 128 * 4;
        int32_t* d_out; CUDA_CHECK(cudaMalloc(&d_out, NELT * sizeof(int32_t)));
        CUDA_CHECK(cudaMemset(d_out, 0, NELT * sizeof(int32_t)));
        wgmma_u8_m64n8k32_allones_kernel<<<1, 128>>>(d_out);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  m64n8k32 u8.u8: LAUNCH FAIL: %s\n", cudaGetErrorString(err));
            all_pass = false;
        } else {
            int32_t h_out[NELT];
            CUDA_CHECK(cudaMemcpy(h_out, d_out, NELT * sizeof(int32_t), cudaMemcpyDeviceToHost));
            int bad = 0;
            for (int i = 0; i < NELT; i++) if (h_out[i] != expected_K) bad++;
            if (bad == 0) printf("  m64n8k32 u8.u8: OK (all %d == %d)\n", NELT, expected_K);
            else { printf("  m64n8k32 u8.u8: FAIL (%d bad)\n", bad); all_pass = false; }
        }
        cudaFree(d_out);
    }

    // --- MED: s8.u8 mixed-sign m64n8k32 ---
    {
        const int NELT = 128 * 4;
        int32_t* d_out; CUDA_CHECK(cudaMalloc(&d_out, NELT * sizeof(int32_t)));
        CUDA_CHECK(cudaMemset(d_out, 0, NELT * sizeof(int32_t)));
        wgmma_s8u8_m64n8k32_allones_kernel<<<1, 128>>>(d_out);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  m64n8k32 s8.u8: LAUNCH FAIL: %s\n", cudaGetErrorString(err));
            all_pass = false;
        } else {
            int32_t h_out[NELT];
            CUDA_CHECK(cudaMemcpy(h_out, d_out, NELT * sizeof(int32_t), cudaMemcpyDeviceToHost));
            int bad = 0;
            for (int i = 0; i < NELT; i++) if (h_out[i] != expected_K) bad++;
            if (bad == 0) printf("  m64n8k32 s8.u8: OK (all %d == %d)\n", NELT, expected_K);
            else { printf("  m64n8k32 s8.u8: FAIL (%d bad)\n", bad); all_pass = false; }
        }
        cudaFree(d_out);
    }

    // --- MED: u8.s8 mixed-sign m64n8k32 ---
    {
        const int NELT = 128 * 4;
        int32_t* d_out; CUDA_CHECK(cudaMalloc(&d_out, NELT * sizeof(int32_t)));
        CUDA_CHECK(cudaMemset(d_out, 0, NELT * sizeof(int32_t)));
        wgmma_u8s8_m64n8k32_allones_kernel<<<1, 128>>>(d_out);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  m64n8k32 u8.s8: LAUNCH FAIL: %s\n", cudaGetErrorString(err));
            all_pass = false;
        } else {
            int32_t h_out[NELT];
            CUDA_CHECK(cudaMemcpy(h_out, d_out, NELT * sizeof(int32_t), cudaMemcpyDeviceToHost));
            int bad = 0;
            for (int i = 0; i < NELT; i++) if (h_out[i] != expected_K) bad++;
            if (bad == 0) printf("  m64n8k32 u8.s8: OK (all %d == %d)\n", NELT, expected_K);
            else { printf("  m64n8k32 u8.s8: FAIL (%d bad)\n", bad); all_pass = false; }
        }
        cudaFree(d_out);
    }

    // --- MED: .satfinite functional check ---
    // With small inputs that don't overflow, .satfinite must produce identical
    // values to the non-saturating instruction. With fillA=fillB=1, K=32, the
    // expected value is K = 32 (same as the basic test).
    {
        const int NELT = 128 * 4;
        int32_t* d_out; CUDA_CHECK(cudaMalloc(&d_out, NELT * sizeof(int32_t)));
        CUDA_CHECK(cudaMemset(d_out, 0, NELT * sizeof(int32_t)));
        wgmma_i8_m64n8k32_satfinite_kernel<<<1, 128>>>(d_out, 1, 1);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  m64n8k32 s8 .satfinite: LAUNCH FAIL: %s\n", cudaGetErrorString(err));
            all_pass = false;
        } else {
            int32_t h_out[NELT];
            CUDA_CHECK(cudaMemcpy(h_out, d_out, NELT * sizeof(int32_t), cudaMemcpyDeviceToHost));
            int bad = 0;
            for (int i = 0; i < NELT; i++) if (h_out[i] != expected_K) bad++;
            if (bad == 0) printf("  m64n8k32 s8 .satfinite: OK (all %d == %d, no overflow)\n", NELT, expected_K);
            else { printf("  m64n8k32 s8 .satfinite: FAIL (%d bad)\n", bad); all_pass = false; }
        }
        cudaFree(d_out);
    }

    // --- m64n8k64 SPARSE s8 ---
    {
        const int NELT = 128 * 4;
        int32_t* d_out; CUDA_CHECK(cudaMalloc(&d_out, NELT * sizeof(int32_t)));
        CUDA_CHECK(cudaMemset(d_out, 0, NELT * sizeof(int32_t)));
        wgmma_i8_m64n8k64_sp_allones_kernel<<<1, 128>>>(d_out);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  m64n8k64 sparse s8: LAUNCH FAIL: %s\n", cudaGetErrorString(err));
            all_pass = false;
        } else {
            int32_t* h_out = new int32_t[NELT];
            CUDA_CHECK(cudaMemcpy(h_out, d_out, NELT * sizeof(int32_t), cudaMemcpyDeviceToHost));
            const int32_t expected_sp = 32;
            int bad = 0;
            for (int i = 0; i < NELT; i++) if (h_out[i] != expected_sp) bad++;
            if (bad == 0) printf("  m64n8k64 sparse s8: OK (all %d == %d)\n", NELT, expected_sp);
            else { printf("  m64n8k64 sparse s8: FAIL (%d bad of %d)\n", bad, NELT); all_pass = false; }
            delete[] h_out;
        }
        cudaFree(d_out);
    }

    if (all_pass) { PASS(); return 0; }
    else { FAIL("some subtests failed"); return 1; }
}

#endif  // PL_AGENTIC_SM90A
