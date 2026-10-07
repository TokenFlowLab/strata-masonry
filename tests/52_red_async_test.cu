// ARCH: sm_90a
// 52_red_async_test.cu -- cluster-scope red.async coupled to an mbarrier.
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
#include "../src/primitives/52_red_async.cuh"
#include "52_red_async.cuh"

// =============================================================================
// ours
// =============================================================================

// ARCH: sm_90a


__global__ void __cluster_dims__(2, 1, 1) k_r() {
  __shared__ __align__(16) uint32_t cnt;
  __shared__ __align__(16) uint64_t mbar;
  if (threadIdx.x == 0) {
    cnt = 0;
    mbarrier_init(smem_ptr_u32(&mbar), 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    red_async_cluster_add_u32(smem_ptr_u32(&cnt), 42, smem_ptr_u32(&mbar));
  }
  __syncthreads();
  if (threadIdx.x == 0)
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n"
                 :: "r"(smem_ptr_u32(&mbar)));
}

static int run_ours() {
  k_r<<<2, 32>>>();
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("red.async.cluster : compile + run OK\n");
  PASS();
}

// =============================================================================
// theirs
// =============================================================================

// Test: red_async -- compile + PTX verification

__global__ void red_async_kernel(uint32_t* out) {
    __shared__ uint32_t s_val;
    __shared__ uint64_t s_mbar;
    if (threadIdx.x == 0) {
        s_val = 0;
    }
    __syncthreads();

    uint32_t addr = smem_ptr_u32(&s_val);
    uint32_t mbar_addr = smem_ptr_u32(&s_mbar);

    // Async cluster-scope reduction add u32
    red_async_cluster_add_u32(addr, 1u, mbar_addr);

    __syncthreads();
    if (threadIdx.x == 0) {
        out[0] = s_val;
    }
}

static int run_theirs() {
  printf("52_red_async: compiled.\n"); PASS(); return 0; }


// =============================================================================
// red.async cluster max.u32 + add/min/max .s32
// =============================================================================
//
// Strategy: cluster=1, single thread issues one red.async.<op>.<type> against
// a CTA-shared counter, with mbarrier coupling. We mbarrier-wait until the
// reduction's tx bytes arrive, then read the counter to GMEM. This verifies
// the wrapper actually performs the reduction (not just compiles).

__global__ void __cluster_dims__(1, 1, 1)
k_red_med_max_u32(uint32_t* gout, uint32_t init, uint32_t v) {
  __shared__ __align__(16) uint32_t cnt;
  __shared__ __align__(16) uint64_t mbar;
  if (threadIdx.x == 0) {
    cnt = init;
    mbarrier_init(smem_ptr_u32(&mbar), 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    // Arrive + set expect_tx = 4 (one u32 reduction).
    asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 4;\n"
      :: "r"(smem_ptr_u32(&mbar)) : "memory");
    red_async_cluster_max_u32(smem_ptr_u32(&cnt), v, smem_ptr_u32(&mbar));
    // Wait for mbarrier to flip on phase 0.
    asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "WL_max_u32:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n"
      "@!p bra WL_max_u32;\n"
      "}\n"
      :: "r"(smem_ptr_u32(&mbar)) : "memory");
    gout[0] = cnt;
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n"
                 :: "r"(smem_ptr_u32(&mbar)));
  }
}

__global__ void __cluster_dims__(1, 1, 1)
k_red_med_add_s32(int32_t* gout, int32_t init, int32_t v) {
  __shared__ __align__(16) int32_t cnt;
  __shared__ __align__(16) uint64_t mbar;
  if (threadIdx.x == 0) {
    cnt = init;
    mbarrier_init(smem_ptr_u32(&mbar), 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 4;\n"
      :: "r"(smem_ptr_u32(&mbar)) : "memory");
    red_async_cluster_add_s32(smem_ptr_u32(&cnt), v, smem_ptr_u32(&mbar));
    asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "WL_add_s32:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n"
      "@!p bra WL_add_s32;\n"
      "}\n"
      :: "r"(smem_ptr_u32(&mbar)) : "memory");
    gout[0] = cnt;
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n"
                 :: "r"(smem_ptr_u32(&mbar)));
  }
}

__global__ void __cluster_dims__(1, 1, 1)
k_red_med_min_s32(int32_t* gout, int32_t init, int32_t v) {
  __shared__ __align__(16) int32_t cnt;
  __shared__ __align__(16) uint64_t mbar;
  if (threadIdx.x == 0) {
    cnt = init;
    mbarrier_init(smem_ptr_u32(&mbar), 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 4;\n"
      :: "r"(smem_ptr_u32(&mbar)) : "memory");
    red_async_cluster_min_s32(smem_ptr_u32(&cnt), v, smem_ptr_u32(&mbar));
    asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "WL_min_s32:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n"
      "@!p bra WL_min_s32;\n"
      "}\n"
      :: "r"(smem_ptr_u32(&mbar)) : "memory");
    gout[0] = cnt;
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n"
                 :: "r"(smem_ptr_u32(&mbar)));
  }
}

__global__ void __cluster_dims__(1, 1, 1)
k_red_med_max_s32(int32_t* gout, int32_t init, int32_t v) {
  __shared__ __align__(16) int32_t cnt;
  __shared__ __align__(16) uint64_t mbar;
  if (threadIdx.x == 0) {
    cnt = init;
    mbarrier_init(smem_ptr_u32(&mbar), 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 4;\n"
      :: "r"(smem_ptr_u32(&mbar)) : "memory");
    red_async_cluster_max_s32(smem_ptr_u32(&cnt), v, smem_ptr_u32(&mbar));
    asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "WL_max_s32:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n"
      "@!p bra WL_max_s32;\n"
      "}\n"
      :: "r"(smem_ptr_u32(&mbar)) : "memory");
    gout[0] = cnt;
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n"
                 :: "r"(smem_ptr_u32(&mbar)));
  }
}

static int run_med() {
  bool ok = true;

  // max.u32: init=10, v=42 -> expect 42
  {
    uint32_t* d = nullptr; CUDA_CHECK(cudaMalloc(&d, 4));
    CUDA_CHECK(cudaMemset(d, 0, 4));
    k_red_med_max_u32<<<1, 32>>>(d, 10u, 42u);
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t h = 0; CUDA_CHECK(cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost));
    cudaFree(d);
    if (h != 42u) { printf("  red.async max.u32: FAIL got %u expected 42\n", h); ok = false; }
    else printf("  red.async max.u32: OK (max(10,42)=%u)\n", h);
  }

  // max.u32: init=100, v=42 -> expect 100 (no change)
  {
    uint32_t* d = nullptr; CUDA_CHECK(cudaMalloc(&d, 4));
    CUDA_CHECK(cudaMemset(d, 0, 4));
    k_red_med_max_u32<<<1, 32>>>(d, 100u, 42u);
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t h = 0; CUDA_CHECK(cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost));
    cudaFree(d);
    if (h != 100u) { printf("  red.async max.u32 (no-change): FAIL got %u expected 100\n", h); ok = false; }
    else printf("  red.async max.u32 (no-change): OK (max(100,42)=%u)\n", h);
  }

  // add.s32: init=-5, v=10 -> expect 5
  {
    int32_t* d = nullptr; CUDA_CHECK(cudaMalloc(&d, 4));
    CUDA_CHECK(cudaMemset(d, 0, 4));
    k_red_med_add_s32<<<1, 32>>>(d, -5, 10);
    CUDA_CHECK(cudaDeviceSynchronize());
    int32_t h = 0; CUDA_CHECK(cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost));
    cudaFree(d);
    if (h != 5) { printf("  red.async add.s32: FAIL got %d expected 5\n", h); ok = false; }
    else printf("  red.async add.s32: OK (-5 + 10 = %d)\n", h);
  }

  // min.s32: init=10, v=-7 -> expect -7 (signed compare)
  {
    int32_t* d = nullptr; CUDA_CHECK(cudaMalloc(&d, 4));
    CUDA_CHECK(cudaMemset(d, 0, 4));
    k_red_med_min_s32<<<1, 32>>>(d, 10, -7);
    CUDA_CHECK(cudaDeviceSynchronize());
    int32_t h = 0; CUDA_CHECK(cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost));
    cudaFree(d);
    if (h != -7) { printf("  red.async min.s32: FAIL got %d expected -7\n", h); ok = false; }
    else printf("  red.async min.s32: OK (min(10,-7)=%d)\n", h);
  }

  // max.s32: init=-100, v=-50 -> expect -50 (signed compare)
  {
    int32_t* d = nullptr; CUDA_CHECK(cudaMalloc(&d, 4));
    CUDA_CHECK(cudaMemset(d, 0, 4));
    k_red_med_max_s32<<<1, 32>>>(d, -100, -50);
    CUDA_CHECK(cudaDeviceSynchronize());
    int32_t h = 0; CUDA_CHECK(cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost));
    cudaFree(d);
    if (h != -50) { printf("  red.async max.s32: FAIL got %d expected -50\n", h); ok = false; }
    else printf("  red.async max.s32: OK (max(-100,-50)=%d)\n", h);
  }

  if (ok) { PASS(); return 0; }
  else { FAIL("red.async max/s32 subtest failed"); return 1; }
}

int main() {
  int rc_ours   = run_ours();
  int rc_theirs = run_theirs();
  int rc_med    = run_med();
  return (rc_ours == 0 && rc_theirs == 0 && rc_med == 0) ? 0 : 1;
}
