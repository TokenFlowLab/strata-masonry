// ARCH: sm_90a
// 27_cp_async_cg_test.cu -- 16-byte L1-bypass cp.async copy with verification.
//
// Combined test: both ours' and theirs' coverage is exercised
// in a single binary (each side's main() became run_ours/run_theirs).

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "test_utils.cuh"
#include "../src/primitives/27_cp_async_cg.cuh"
#include "../src/primitives/28_cp_async_commit_wait.cuh"
#include "27_cp_async_cg.cuh"

// =============================================================================
// ours (originally guarded by PL_AGENTIC_SM100A || PL_AGENTIC_SM103A)
// =============================================================================

// ARCH: sm_90a


__global__ void k_cpg(const uint32_t* gin, uint32_t* gout) {
  __shared__ __align__(16) uint32_t smem[128];
  // Each thread-group of 4 consecutive threads does one 16-byte copy.
  if ((threadIdx.x & 3) == 0) {
    cp_async_cg_16(smem_ptr_u32(&smem[threadIdx.x]),
                    gin + threadIdx.x);
  }
  cp_async_commit_group();
  cp_async_wait_group<0>();
  __syncthreads();
  gout[threadIdx.x] = smem[threadIdx.x];
}

static int run_ours() {
  /* (orig args dropped) */
  std::vector<uint32_t> hin(128);
  for (int i = 0; i < 128; ++i) hin[i] = 0xB0000000u + i;
  uint32_t *din = nullptr, *dout = nullptr;
  CUDA_CHECK(cudaMalloc(&din, 128 * 4));
  CUDA_CHECK(cudaMalloc(&dout, 128 * 4));
  CUDA_CHECK(cudaMemcpy(din, hin.data(), 128 * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dout, 0, 128 * 4));
  k_cpg<<<1, 128>>>(din, dout);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint32_t> hout(128);
  CUDA_CHECK(cudaMemcpy(hout.data(), dout, 128 * 4, cudaMemcpyDeviceToHost));
  cudaFree(din); cudaFree(dout);
  int fails = 0;
  for (int i = 0; i < 128; ++i) if (hout[i] != hin[i]) ++fails;
  printf("cp.async.cg 16B : fails = %d / 128\n", fails);
  if (fails) FAIL("cp.async.cg did not propagate data");
  PASS();
}

// =============================================================================
// theirs (originally guarded by PL_AGENTIC_SM90A)
// =============================================================================

// Runtime test: cp.async.cg (16B streaming) -- GMEM->SMEM->GMEM roundtrip + timing

static constexpr int THREADS = 128;
static constexpr int N16 = THREADS * 16 / 4;   // u32 elements = 512

__global__ void cp_async_cg_16_kernel(const uint32_t* __restrict__ gmem_in,
                                      uint32_t* __restrict__ gmem_out) {
    __shared__ uint32_t smem[N16];
    uint32_t thread_offset_u32 = threadIdx.x * 4;   // 16 bytes per thread
    uint32_t smem_addr = smem_ptr_u32(&smem[thread_offset_u32]);
    cp_async_cg_16(smem_addr, gmem_in + thread_offset_u32);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    gmem_out[thread_offset_u32 + 0] = smem[thread_offset_u32 + 0];
    gmem_out[thread_offset_u32 + 1] = smem[thread_offset_u32 + 1];
    gmem_out[thread_offset_u32 + 2] = smem[thread_offset_u32 + 2];
    gmem_out[thread_offset_u32 + 3] = smem[thread_offset_u32 + 3];
}

static int run_theirs() {
  /* (orig args dropped) */
    CUDA_CHECK(cudaFree(0));
    bool all_pass = true;

    uint32_t *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  N16 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_out, N16 * sizeof(uint32_t)));

    uint32_t* h_in  = (uint32_t*)malloc(N16 * sizeof(uint32_t));
    uint32_t* h_out = (uint32_t*)malloc(N16 * sizeof(uint32_t));
    for (int i = 0; i < N16; i++) h_in[i] = (uint32_t)(i * 7 + 3);

    CUDA_CHECK(cudaMemcpy(d_in, h_in, N16 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, N16 * sizeof(uint32_t)));

    cp_async_cg_16_kernel<<<1, THREADS>>>(d_in, d_out);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, N16 * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    int mismatches = 0;
    for (int i = 0; i < N16; i++) if (h_in[i] != h_out[i]) mismatches++;

    if (mismatches == 0) {
        printf("  cp.async.cg.16: OK (%d u32 = %d bytes copied)\n", N16, N16 * 4);
    } else {
        printf("  cp.async.cg.16: FAIL (%d / %d mismatches)\n", mismatches, N16);
        all_pass = false;
    }

    // --- Timing ---
    GpuTimer t;
    t.begin();
    for (int i = 0; i < 100; i++) cp_async_cg_16_kernel<<<1, THREADS>>>(d_in, d_out);
    t.end();
    printf("  perf: %.2f us/launch (%d bytes)\n",
           t.elapsed_ms() * 1000.0f / 100, THREADS * 16);

    free(h_in); free(h_out);
    cudaFree(d_in); cudaFree(d_out);

    if (all_pass) { PASS(); return 0; }
    else { FAIL("some subtests failed"); return 1; }
}


// .L2::cache_hint variant for cp.async.cg.
__global__ void k_cp_async_cg_l2hint(const int4* g_in, int4* g_out, int n) {
    extern __shared__ __align__(16) int4 smem[];
    int tid = threadIdx.x;
    uint64_t cache_policy;
    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;\n"
                 : "=l"(cache_policy));
    if (tid < n) {
        uint32_t smem_dst = smem_ptr_u32(&smem[tid]);
        cp_async_cg_16_l2hint(smem_dst, &g_in[tid], cache_policy);
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

    k_cp_async_cg_l2hint<<<1, 32, N * sizeof(int4)>>>(d_in, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    int4 h_out[N];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(int4), cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < N; i++) {
        if (h_out[i].x != h_in[i].x || h_out[i].y != h_in[i].y ||
            h_out[i].z != h_in[i].z || h_out[i].w != h_in[i].w) bad++;
    }
    if (bad == 0) printf("  cp.async.cg.L2::cache_hint: OK (%d 16B copies round-tripped)\n", N);
    else { printf("  cp.async.cg.L2::cache_hint: FAIL (%d bad)\n", bad); all_pass = false; }

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
