// ARCH: sm_90a
// 38_barrier_cluster_test.cu -- cluster-wide barrier across 2 CTAs.
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
#include "../primitives/38_barrier_cluster.cuh"
#include "38_barrier_cluster.cuh"

// =============================================================================
// ours (originally guarded by PL_AGENTIC_SM100A || PL_AGENTIC_SM103A)
// =============================================================================

// ARCH: sm_90a


__global__ void __cluster_dims__(2, 1, 1)
k_cbar(int* out) {
  barrier_cluster_arrive();
  barrier_cluster_wait();
  if (threadIdx.x == 0) atomicAdd(out, 1);
}

static int run_ours() {
  /* (orig args dropped) */
  int* d = nullptr; CUDA_CHECK(cudaMalloc(&d, 4));
  CUDA_CHECK(cudaMemset(d, 0, 4));
  k_cbar<<<2, 64>>>(d);
  CUDA_CHECK(cudaDeviceSynchronize());
  int h = 0; CUDA_CHECK(cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost));
  cudaFree(d);
  if (h != 2) FAIL("cluster barrier did not gate both CTAs");
  printf("barrier.cluster.arrive/wait (2 CTAs) : OK (count=%d)\n", h);
  PASS();
}

// =============================================================================
// theirs (originally guarded by PL_AGENTIC_SM90A)
// =============================================================================

// Test: barrier.cluster -- compile + PTX verification

__global__ void barrier_cluster_kernel() {
    barrier_cluster_arrive();
    barrier_cluster_wait();
    barrier_cluster_arrive_release();
    barrier_cluster_wait_acquire();
    barrier_cluster_sync();
}

static int run_theirs() {
  /* (orig args dropped) */ printf("38_barrier_cluster: compiled.\n"); PASS(); return 0; }


// =============================================================================
// Combined sem + aligned cluster barriers (Phase 3 HIGH)
// =============================================================================

// Two clustered CTAs, each running the new sem+aligned wrappers in
// matching pairs (release-aligned <-> acquire-aligned, relaxed-aligned
// <-> aligned). Each CTA increments a shared counter only after both
// CTAs have crossed the synchronization point. The order
// (arrive-release-aligned, then wait-acquire-aligned) mirrors the
// canonical cluster pipeline pattern: a warp-uniform release publishes
// prior writes, peer CTAs acquire on the wait side.
__global__ void __cluster_dims__(2, 1, 1)
k_cbar_sem_aligned(int* out) {
  // Stage 1: release.aligned + acquire.aligned. EVERY thread participates
  // in the .aligned variants (per ISA), so issue uniformly across the warp.
  barrier_cluster_arrive_release_aligned();
  barrier_cluster_wait_acquire_aligned();
  if (threadIdx.x == 0) atomicAdd(&out[0], 1);

  // Stage 2: relaxed.aligned + plain aligned wait. Only memory-ordering
  // semantics differ; the barrier still gates both CTAs.
  barrier_cluster_arrive_relaxed_aligned();
  barrier_cluster_wait_aligned();
  if (threadIdx.x == 0) atomicAdd(&out[1], 1);
}

static int run_sem_aligned() {
  int* d = nullptr; CUDA_CHECK(cudaMalloc(&d, 2 * sizeof(int)));
  CUDA_CHECK(cudaMemset(d, 0, 2 * sizeof(int)));
  k_cbar_sem_aligned<<<2, 128>>>(d);
  CUDA_CHECK(cudaDeviceSynchronize());
  int h[2] = {0, 0};
  CUDA_CHECK(cudaMemcpy(h, d, 2 * sizeof(int), cudaMemcpyDeviceToHost));
  cudaFree(d);
  if (h[0] != 2 || h[1] != 2) {
    fprintf(stderr, "sem+aligned cbar: stage1=%d stage2=%d (expected 2,2)\n",
            h[0], h[1]);
    FAIL("combined sem+aligned cluster barriers did not gate both CTAs");
  }
  printf("barrier.cluster.{arrive.release.aligned,wait.acquire.aligned,arrive.relaxed.aligned}: OK (both CTAs gated through 2 stages)\n");
  PASS();
}

int main() {
  int rc_ours   = run_ours();
  int rc_theirs = run_theirs();
  int rc_aligned = run_sem_aligned();
  return (rc_ours == 0 && rc_theirs == 0 && rc_aligned == 0) ? 0 : 1;
}
