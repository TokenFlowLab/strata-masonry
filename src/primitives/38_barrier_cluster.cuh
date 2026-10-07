// 38_barrier_cluster.cuh -- barrier.cluster.arrive / .wait / .align
//
// ARCH: sm_90a
//
// Cluster-wide barrier. ALL threads in every CTA of the cluster participate;
// if one CTA exits without arriving, the others hang forever.
//
// Used to align CTA-pair lifecycles (e.g. post-init sync before peer CTAs
// reference shared mbarriers; pre-dealloc sync before tcgen05.dealloc in
// 2SM kernels). The aligned variants require every participating thread to
// use the .aligned form.
//
// PTX:    9.7.15.3 (barrier.cluster)
//

#pragma once

// -- arrive ------------------------------------------------------------------

__device__ __forceinline__
void barrier_cluster_arrive() {
  asm volatile("barrier.cluster.arrive;\n" ::: "memory");
}

__device__ __forceinline__
void barrier_cluster_arrive_release() {
  asm volatile("barrier.cluster.arrive.release;\n" ::: "memory");
}

__device__ __forceinline__
void barrier_cluster_arrive_aligned() {
  asm volatile("barrier.cluster.arrive.aligned;\n" ::: "memory");
}

// Combined sem + aligned forms. Required when a
// warp-uniform arrive is also publishing prior writes to peer CTAs --
// the canonical cluster pipeline pattern.
__device__ __forceinline__
void barrier_cluster_arrive_release_aligned() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
}

__device__ __forceinline__
void barrier_cluster_arrive_relaxed_aligned() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
}

// -- wait --------------------------------------------------------------------

__device__ __forceinline__
void barrier_cluster_wait() {
  asm volatile("barrier.cluster.wait;\n" ::: "memory");
}

__device__ __forceinline__
void barrier_cluster_wait_acquire() {
  asm volatile("barrier.cluster.wait.acquire;\n" ::: "memory");
}

__device__ __forceinline__
void barrier_cluster_wait_aligned() {
  asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory");
}

// Combined sem + aligned for wait side.
__device__ __forceinline__
void barrier_cluster_wait_acquire_aligned() {
  asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}

// -- combined arrive + wait (common post-init pattern) ----------------------

__device__ __forceinline__
void barrier_cluster_sync() {
  barrier_cluster_arrive();
  barrier_cluster_wait();
}
// Number of CTAs along the cluster's x dimension (%cluster_nctaid.x). Equivalent to
// cute::cluster_shape().x.
__device__ __forceinline__ uint32_t cluster_dim_x() {
  uint32_t n;
  asm volatile("mov.u32 %0, %%cluster_nctaid.x;\n" : "=r"(n));
  return n;
}


