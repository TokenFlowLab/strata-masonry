// ARCH: sm_90a
// 30_mbarrier_arrive_test.cu -- N threads each arrive once; try_wait
// completes when all N have arrived.
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
#include "../src/primitives/29_mbarrier_init.cuh"
#include "../src/primitives/30_mbarrier_arrive.cuh"
#include "../src/primitives/33_mbarrier_try_wait.cuh"
#include "29_mbarrier_init.cuh"
#include "30_mbarrier_arrive.cuh"
#include "33_mbarrier_try_wait.cuh"

// =============================================================================
// ours
// =============================================================================

// ARCH: sm_90a


__global__ void k_arrive(int* ok) {
  __shared__ __align__(16) uint64_t mbar;
  if (threadIdx.x == 0) mbarrier_init(smem_ptr_u32(&mbar), 32);
  __syncthreads();
  if (threadIdx.x < 32) mbarrier_arrive(smem_ptr_u32(&mbar));
  if (threadIdx.x == 0) {
    mbarrier_wait_parity(smem_ptr_u32(&mbar), 0);
    *ok = 1;
    mbarrier_inval(smem_ptr_u32(&mbar));
  }
}

static int run_ours() {
  int* d = nullptr; CUDA_CHECK(cudaMalloc(&d, 4));
  CUDA_CHECK(cudaMemset(d, 0, 4));
  k_arrive<<<1, 32>>>(d);
  CUDA_CHECK(cudaDeviceSynchronize());
  int h = 0; CUDA_CHECK(cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost));
  cudaFree(d);
  if (h != 1) FAIL("32-arrival mbarrier did not complete");
  printf("mbarrier.arrive (32 threads) : OK\n");
  PASS();
}

// =============================================================================
// theirs
// =============================================================================

// Runtime test: mbarrier.arrive producer/consumer sync
// Warp 0 writes a pattern to SMEM, arrives on full_mbar.
// Warp 1 waits on full_mbar (parity 0), reads SMEM, writes to output.
// Also exercises mbarrier_arrive_count with a single single-thread arrival.

// --- Kernel 1: full-warp arrive (arrival_count = 32). --------------------
// 64 threads, 2 warps. Warp 0 produces, Warp 1 consumes.
// Full barrier credited by all 32 threads of warp 0.
__global__ void producer_consumer_kernel(uint32_t* out, int n) {
    __shared__ __align__(8) uint64_t full_mbar[1];
    extern __shared__ __align__(16) uint32_t s_buf[];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    uint32_t mbar_addr = smem_ptr_u32(full_mbar);

    if (tid == 0) {
        mbarrier_init(mbar_addr, 32); // 32 arrivals (one warp)
    }
    __syncthreads();

    if (warp == 0) {
        // Producer: each thread writes 'lane' copies of a pattern
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            s_buf[lane * 4 + i] = (uint32_t)(lane * 100 + i + 1);
        }
        // Arrive -- the .release semantics publish our SMEM writes
        mbarrier_arrive_nostate(mbar_addr);
    } else if (warp == 1) {
        // Consumer: wait for phase 0 completion, then read SMEM.
        mbarrier_wait_parity(mbar_addr, 0);
        if (lane < n) {
            // Each consumer lane reads the 4 values produced by its counterpart lane
            for (int i = 0; i < 4; i++) {
                out[lane * 4 + i] = s_buf[lane * 4 + i];
            }
        }
    }

    __syncthreads();
    if (tid == 0) mbarrier_inval(mbar_addr);
}

// --- Kernel 2: single thread calls arrive_count(32). ---------------------
// arrival_count = 32 but a single thread credits 32 in one shot.
__global__ void single_thread_count32_kernel(uint32_t* out) {
    __shared__ __align__(8) uint64_t full_mbar[1];
    extern __shared__ __align__(16) uint32_t s_buf[];

    uint32_t mbar_addr = smem_ptr_u32(full_mbar);
    const int tid = threadIdx.x;

    if (tid == 0) mbarrier_init(mbar_addr, 32);
    __syncthreads();

    if (tid == 0) {
        // Producer: write pattern
        for (int i = 0; i < 32; i++) s_buf[i] = 0xABCD0000u + (uint32_t)i;
        // Credit 32 arrivals in one call
        uint64_t st = mbarrier_arrive_count(mbar_addr, 32);
        (void)st;
    }
    // Every thread in the block now waits for phase 0 completion
    mbarrier_wait_parity(mbar_addr, 0);

    if (tid < 32) out[tid] = s_buf[tid];

    __syncthreads();
    if (tid == 0) mbarrier_inval(mbar_addr);
}

static int run_theirs() {
    CUDA_CHECK(cudaFree(0));
    bool all_pass = true;

    // --- Kernel 1: producer/consumer with full-warp arrivals ---
    const int THREADS = 64;
    const int N = 32; // 32 lanes produce/consume
    uint32_t h_out[N * 4] = {};
    uint32_t* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(h_out)));
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(h_out)));
    // SMEM needs: N*4 uint32 = 512 bytes
    producer_consumer_kernel<<<1, THREADS, N * 4 * sizeof(uint32_t)>>>(d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));

    bool pc_pass = true;
    int mism = 0;
    for (int lane = 0; lane < N; lane++) {
        for (int i = 0; i < 4; i++) {
            uint32_t expect = (uint32_t)(lane * 100 + i + 1);
            if (h_out[lane * 4 + i] != expect) {
                if (mism < 5) {
                    printf("    prod/cons mismatch [%d]: got %u expected %u\n",
                           lane * 4 + i, h_out[lane * 4 + i], expect);
                }
                mism++;
                pc_pass = false;
            }
        }
    }
    if (pc_pass) printf("  prod/cons (arrive x32, wait): OK (128 values)\n");
    else { printf("  prod/cons: FAIL (%d mismatches)\n", mism); all_pass = false; }

    // --- Kernel 2: single-thread arrive_count(32) ---
    uint32_t h2_out[32] = {};
    uint32_t* d2_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d2_out, sizeof(h2_out)));
    CUDA_CHECK(cudaMemset(d2_out, 0, sizeof(h2_out)));
    single_thread_count32_kernel<<<1, THREADS, 32 * sizeof(uint32_t)>>>(d2_out);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h2_out, d2_out, sizeof(h2_out), cudaMemcpyDeviceToHost));

    bool ct_pass = true;
    mism = 0;
    for (int i = 0; i < 32; i++) {
        uint32_t expect = 0xABCD0000u + (uint32_t)i;
        if (h2_out[i] != expect) {
            if (mism < 5) {
                printf("    count32 mismatch [%d]: got 0x%08x expected 0x%08x\n",
                       i, h2_out[i], expect);
            }
            mism++;
            ct_pass = false;
        }
    }
    if (ct_pass) printf("  arrive_count(32) x1:          OK (32 values)\n");
    else { printf("  arrive_count(32): FAIL\n"); all_pass = false; }

    // --- perf ---
    GpuTimer t;
    const int ITERS = 200;
    t.begin();
    for (int i = 0; i < ITERS; i++)
        producer_consumer_kernel<<<1, THREADS, N * 4 * sizeof(uint32_t)>>>(d_out, N);
    t.end();
    printf("  perf:                         %.2f us/launch (warp-pair sync)\n",
           t.elapsed_ms() * 1000.0f / ITERS);

    cudaFree(d_out); cudaFree(d2_out);
    if (all_pass) { PASS(); return 0; }
    else { FAIL("some subtests failed"); return 1; }
}


int main() {
  int rc_ours   = run_ours();
  int rc_theirs = run_theirs();
  return (rc_ours == 0 && rc_theirs == 0) ? 0 : 1;
}
