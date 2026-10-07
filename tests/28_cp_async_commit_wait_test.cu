// ARCH: sm_90a
// 28_cp_async_commit_wait_test.cu -- compile smoke (covered end-to-end by #26, #27).
//
// Two test sets in one binary: run_ours() and run_theirs().

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "test_utils.cuh"
#include "../src/primitives/28_cp_async_commit_wait.cuh"
#include "26_cp_async_ca.cuh"
#include "28_cp_async_commit_wait.cuh"

// =============================================================================
// ours
// =============================================================================

// ARCH: sm_90a


__global__ void k_cw() {
  cp_async_commit_group();
  cp_async_wait_group<0>();
  cp_async_wait_all();
}

static int run_ours() {
  k_cw<<<1, 32>>>();
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("cp.async.commit/wait_group/wait_all : compile OK\n");
  PASS();
}

// =============================================================================
// theirs
// =============================================================================

// Runtime test: cp.async.commit_group / wait_group -- round-trip + timing
// Kernel issues cp.async.ca.16 GMEM->SMEM, commit, wait, SMEM->GMEM.
// Verifies that data correctly round-trips through SMEM via async copies.

// Each thread copies one 16-byte chunk (4 floats) through SMEM.
// Total copied per block = THREADS * 16 bytes.
// Stage 1: GMEM[in]  -> SMEM via cp.async.ca.16
// Stage 2: commit_group + wait_group<0>
// Stage 3: SMEM      -> GMEM[out] via ordinary stores
__global__ void round_trip_wait0_kernel(const float* gmem_in, float* gmem_out) {
    extern __shared__ __align__(16) char smem_raw[];
    float* smem = reinterpret_cast<float*>(smem_raw);

    const int tid = threadIdx.x;
    const int off = tid * 4; // 4 floats per thread
    uint32_t smem_addr = smem_ptr_u32(&smem[off]);

    cp_async_ca_16(smem_addr, &gmem_in[off]);
    cp_async_commit_group();
    cp_async_wait_group<0>();
    __syncthreads();

    // Read through SMEM to prove async copy landed correctly
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        gmem_out[off + i] = smem[off + i];
    }
}

// Same but uses cp_async_wait_all()
__global__ void round_trip_waitall_kernel(const float* gmem_in, float* gmem_out) {
    extern __shared__ __align__(16) char smem_raw[];
    float* smem = reinterpret_cast<float*>(smem_raw);

    const int tid = threadIdx.x;
    const int off = tid * 4;
    uint32_t smem_addr = smem_ptr_u32(&smem[off]);

    cp_async_ca_16(smem_addr, &gmem_in[off]);
    cp_async_commit_group();
    cp_async_wait_all();
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        gmem_out[off + i] = smem[off + i];
    }
}

// Multi-stage: issue TWO committed groups, drain one at a time.
// Tests that wait_group<N> honors the N-groups-in-flight contract.
__global__ void round_trip_staged_kernel(const float* gmem_in_a,
                                         const float* gmem_in_b,
                                         float* gmem_out_a,
                                         float* gmem_out_b) {
    extern __shared__ __align__(16) char smem_raw[];
    float* smem_a = reinterpret_cast<float*>(smem_raw);
    float* smem_b = smem_a + blockDim.x * 4;

    const int tid = threadIdx.x;
    const int off = tid * 4;
    uint32_t smem_a_addr = smem_ptr_u32(&smem_a[off]);
    uint32_t smem_b_addr = smem_ptr_u32(&smem_b[off]);

    // Group 0: copy A
    cp_async_ca_16(smem_a_addr, &gmem_in_a[off]);
    cp_async_commit_group();
    // Group 1: copy B
    cp_async_ca_16(smem_b_addr, &gmem_in_b[off]);
    cp_async_commit_group();

    // Wait until at most 1 group remains => group 0 (A) is done
    cp_async_wait_group<1>();
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 4; i++) gmem_out_a[off + i] = smem_a[off + i];

    // Drain the last group
    cp_async_wait_group<0>();
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 4; i++) gmem_out_b[off + i] = smem_b[off + i];
}

static bool check_round_trip(const char* label,
                             const float* h_ref, const float* h_out, int n) {
    int mism = 0;
    for (int i = 0; i < n; i++) {
        if (h_ref[i] != h_out[i]) {
            if (mism < 3) {
                printf("    %s mismatch [%d]: ref=%.2f got=%.2f\n",
                       label, i, h_ref[i], h_out[i]);
            }
            mism++;
        }
    }
    return mism == 0;
}

static int run_theirs() {
    CUDA_CHECK(cudaFree(0));
    bool all_pass = true;

    const int THREADS = 64;         // 64 * 16 = 1024 bytes per buffer per block
    const int N = THREADS * 4;      // floats per buffer

    // Build reference pattern
    float h_in[N], h_ref[N], h_out[N];
    for (int i = 0; i < N; i++) {
        h_in[i] = (float)(i * 7 + 3);
        h_ref[i] = h_in[i];
    }

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    // --- wait_group<0> round-trip ---
    CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(float)));
    round_trip_wait0_kernel<<<1, THREADS, N * sizeof(float)>>>(d_in, d_out);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    if (check_round_trip("wait_group<0>", h_ref, h_out, N))
        printf("  wait_group<0>: OK (%d floats round-tripped)\n", N);
    else { printf("  wait_group<0>: FAIL\n"); all_pass = false; }

    // --- wait_all round-trip ---
    CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(float)));
    round_trip_waitall_kernel<<<1, THREADS, N * sizeof(float)>>>(d_in, d_out);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    if (check_round_trip("wait_all", h_ref, h_out, N))
        printf("  wait_all:     OK (%d floats round-tripped)\n", N);
    else { printf("  wait_all:     FAIL\n"); all_pass = false; }

    // --- two-stage pipeline: commit A, commit B, wait_group<1>, drain ---
    float *d_in_b = nullptr, *d_out_a = nullptr, *d_out_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in_b,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_b, N * sizeof(float)));
    float h_in_b[N], h_ref_b[N], h_out_a[N], h_out_b[N];
    for (int i = 0; i < N; i++) { h_in_b[i] = (float)(-i - 1); h_ref_b[i] = h_in_b[i]; }
    CUDA_CHECK(cudaMemcpy(d_in_b, h_in_b, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out_a, 0, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out_b, 0, N * sizeof(float)));

    round_trip_staged_kernel<<<1, THREADS, 2 * N * sizeof(float)>>>(
        d_in, d_in_b, d_out_a, d_out_b);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out_a, d_out_a, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_b, d_out_b, N * sizeof(float), cudaMemcpyDeviceToHost));
    bool ok_a = check_round_trip("staged-A", h_ref,   h_out_a, N);
    bool ok_b = check_round_trip("staged-B", h_ref_b, h_out_b, N);
    if (ok_a && ok_b) printf("  staged<1>/<0>: OK (2 groups drained in order)\n");
    else { printf("  staged<1>/<0>: FAIL\n"); all_pass = false; }

    // --- perf: wait_group<0> cycle ---
    GpuTimer t;
    const int ITERS = 200;
    t.begin();
    for (int i = 0; i < ITERS; i++)
        round_trip_wait0_kernel<<<1, THREADS, N * sizeof(float)>>>(d_in, d_out);
    t.end();
    printf("  perf:         %.2f us/launch (1024 B/block cp.async round-trip)\n",
           t.elapsed_ms() * 1000.0f / ITERS);

    cudaFree(d_in); cudaFree(d_out);
    cudaFree(d_in_b); cudaFree(d_out_a); cudaFree(d_out_b);

    if (all_pass) { PASS(); return 0; }
    else { FAIL("some subtests failed"); return 1; }
}


int main() {
  int rc_ours   = run_ours();
  int rc_theirs = run_theirs();
  return (rc_ours == 0 && rc_theirs == 0) ? 0 : 1;
}
