// ARCH: sm_90a
// 26_cp_async_ca_test.cu -- cp.async.ca copies GMEM to SMEM, then each
// thread reads its SMEM slot and writes to an output buffer for verification.
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
#include "../src/primitives/26_cp_async_ca.cuh"
#include "../src/primitives/28_cp_async_commit_wait.cuh"
#include "26_cp_async_ca.cuh"

// =============================================================================
// ours
// =============================================================================

// ARCH: sm_90a


__global__ void k_cpa(const uint32_t* gin, uint32_t* gout) {
  __shared__ __align__(16) uint32_t smem[128];
  cp_async_ca_4(smem_ptr_u32(&smem[threadIdx.x]),
                 gin + threadIdx.x);
  cp_async_commit_group();
  cp_async_wait_group<0>();
  __syncthreads();
  gout[threadIdx.x] = smem[threadIdx.x];
}

static int run_ours() {
  std::vector<uint32_t> hin(128);
  for (int i = 0; i < 128; ++i) hin[i] = 0xA0000000u + i;
  uint32_t *din = nullptr, *dout = nullptr;
  CUDA_CHECK(cudaMalloc(&din, 128 * 4));
  CUDA_CHECK(cudaMalloc(&dout, 128 * 4));
  CUDA_CHECK(cudaMemcpy(din, hin.data(), 128 * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dout, 0, 128 * 4));
  k_cpa<<<1, 128>>>(din, dout);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint32_t> hout(128);
  CUDA_CHECK(cudaMemcpy(hout.data(), dout, 128 * 4, cudaMemcpyDeviceToHost));
  cudaFree(din); cudaFree(dout);
  int fails = 0;
  for (int i = 0; i < 128; ++i) if (hout[i] != hin[i]) ++fails;
  printf("cp.async.ca 4B : fails = %d / 128\n", fails);
  if (fails) FAIL("cp.async.ca did not propagate data");
  PASS();
}

// =============================================================================
// theirs
// =============================================================================

// Runtime test: cp.async.ca -- GMEM->SMEM->GMEM roundtrip correctness + timing
// Test 3 variants (16B, 8B, 4B). 128 threads x 16 bytes = 2048 bytes.

static constexpr int THREADS = 128;
static constexpr int N16 = THREADS * 16 / 4;   // u32 elements = 512
static constexpr int N8  = THREADS *  8 / 4;   // u32 elements = 256
static constexpr int N4  = THREADS *  4 / 4;   // u32 elements = 128

// Each thread copies 16 bytes (4 x u32) via cp.async.ca.
__global__ void cp_async_ca_16_kernel(const uint32_t* __restrict__ gmem_in,
                                      uint32_t* __restrict__ gmem_out) {
    __shared__ uint32_t smem[N16];
    uint32_t thread_offset_u32 = threadIdx.x * 4;  // 4 u32 per thread
    uint32_t smem_addr = smem_ptr_u32(&smem[thread_offset_u32]);
    cp_async_ca_16(smem_addr, gmem_in + thread_offset_u32);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    // Copy SMEM back to GMEM.
    gmem_out[thread_offset_u32 + 0] = smem[thread_offset_u32 + 0];
    gmem_out[thread_offset_u32 + 1] = smem[thread_offset_u32 + 1];
    gmem_out[thread_offset_u32 + 2] = smem[thread_offset_u32 + 2];
    gmem_out[thread_offset_u32 + 3] = smem[thread_offset_u32 + 3];
}

// 8-byte variant: 2 u32 per thread.
__global__ void cp_async_ca_8_kernel(const uint32_t* __restrict__ gmem_in,
                                     uint32_t* __restrict__ gmem_out) {
    __shared__ uint32_t smem[N8];
    uint32_t thread_offset_u32 = threadIdx.x * 2;
    uint32_t smem_addr = smem_ptr_u32(&smem[thread_offset_u32]);
    cp_async_ca_8(smem_addr, gmem_in + thread_offset_u32);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    gmem_out[thread_offset_u32 + 0] = smem[thread_offset_u32 + 0];
    gmem_out[thread_offset_u32 + 1] = smem[thread_offset_u32 + 1];
}

// 4-byte variant: 1 u32 per thread.
__global__ void cp_async_ca_4_kernel(const uint32_t* __restrict__ gmem_in,
                                     uint32_t* __restrict__ gmem_out) {
    __shared__ uint32_t smem[N4];
    uint32_t thread_offset_u32 = threadIdx.x;
    uint32_t smem_addr = smem_ptr_u32(&smem[thread_offset_u32]);
    cp_async_ca_4(smem_addr, gmem_in + thread_offset_u32);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    gmem_out[thread_offset_u32] = smem[thread_offset_u32];
}

static bool run_variant(const char* name, void (*kernel)(const uint32_t*, uint32_t*),
                        int n_u32) {
    uint32_t *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  n_u32 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_out, n_u32 * sizeof(uint32_t)));

    uint32_t* h_in  = (uint32_t*)malloc(n_u32 * sizeof(uint32_t));
    uint32_t* h_out = (uint32_t*)malloc(n_u32 * sizeof(uint32_t));
    for (int i = 0; i < n_u32; i++) h_in[i] = (uint32_t)(i * 7 + 3);

    CUDA_CHECK(cudaMemcpy(d_in, h_in, n_u32 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, n_u32 * sizeof(uint32_t)));

    kernel<<<1, THREADS>>>(d_in, d_out);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, n_u32 * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    int mismatches = 0;
    for (int i = 0; i < n_u32; i++) if (h_in[i] != h_out[i]) mismatches++;

    if (mismatches == 0) printf("  %s: OK (%d u32 copied)\n", name, n_u32);
    else printf("  %s: FAIL (%d / %d mismatches)\n", name, mismatches, n_u32);

    free(h_in); free(h_out);
    cudaFree(d_in); cudaFree(d_out);
    return mismatches == 0;
}

static int run_theirs() {
    CUDA_CHECK(cudaFree(0));
    bool all_pass = true;
    all_pass &= run_variant("cp.async.ca.16", cp_async_ca_16_kernel, N16);
    all_pass &= run_variant("cp.async.ca.8",  cp_async_ca_8_kernel,  N8);
    all_pass &= run_variant("cp.async.ca.4",  cp_async_ca_4_kernel,  N4);

    // --- Timing: 16-byte variant ---
    uint32_t *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  N16 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_out, N16 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_in, 0x11, N16 * sizeof(uint32_t)));

    GpuTimer t;
    t.begin();
    for (int i = 0; i < 100; i++) cp_async_ca_16_kernel<<<1, THREADS>>>(d_in, d_out);
    t.end();
    printf("  perf (ca.16): %.2f us/launch (%d bytes)\n",
           t.elapsed_ms() * 1000.0f / 100, THREADS * 16);
    cudaFree(d_in); cudaFree(d_out);

    if (all_pass) { PASS(); return 0; }
    else { FAIL("some subtests failed"); return 1; }
}


// .L2::cache_hint variant test. Builds an L2 cache-policy via createpolicy.
// PTX policy is opaque; we just verify the round-trip data matches.
__global__ void k_cp_async_ca_l2hint(const int4* g_in, int4* g_out, int n) {
    extern __shared__ __align__(16) int4 smem[];
    int tid = threadIdx.x;
    uint64_t cache_policy;
    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;\n"
                 : "=l"(cache_policy));
    if (tid < n) {
        uint32_t smem_dst = smem_ptr_u32(&smem[tid]);
        cp_async_ca_16_l2hint(smem_dst, &g_in[tid], cache_policy);
    }
    cp_async_commit_group();
    cp_async_wait_all();
    __syncthreads();
    if (tid < n) g_out[tid] = smem[tid];
}

static int run_l2hint() {
    bool all_pass = true;
    constexpr int N = 16;
    int4* d_in;  int4* d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(int4)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(int4)));
    int4 h_in[N];
    for (int i = 0; i < N; i++) h_in[i] = make_int4(i, i+100, i+200, i+300);
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(int4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(int4)));

    k_cp_async_ca_l2hint<<<1, 32, N * sizeof(int4)>>>(d_in, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    int4 h_out[N];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(int4), cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < N; i++) {
        if (h_out[i].x != h_in[i].x || h_out[i].y != h_in[i].y ||
            h_out[i].z != h_in[i].z || h_out[i].w != h_in[i].w) bad++;
    }
    if (bad == 0) printf("  cp.async.ca.L2::cache_hint: OK (%d 16B copies round-tripped)\n", N);
    else { printf("  cp.async.ca.L2::cache_hint: FAIL (%d bad)\n", bad); all_pass = false; }

    cudaFree(d_in); cudaFree(d_out);
    if (all_pass) { PASS(); return 0; }
    else { FAIL("l2hint subtest failed"); return 1; }
}

int main() {
  int rc_ours   = run_ours();
  int rc_theirs = run_theirs();
  int rc_l2     = run_l2hint();
  return (rc_ours == 0 && rc_theirs == 0 && rc_l2 == 0) ? 0 : 1;
}
