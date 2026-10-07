// 25_tma_async_group.cuh -- cp.async.bulk.commit_group / .wait_group
//
// ARCH: sm_90a
//
// The bulk async-group namespace tracks TMA stores. It is SEPARATE from the
// plain cp.async.commit_group / wait_group namespace (see #28) -- one is
// for bulk/TMA, the other for non-bulk cp.async. The two do NOT interact.
//
// Two naming conventions are provided (different names, same instructions):
//   cp_async_bulk_commit_group() / cp_async_bulk_wait_group<N>()
//   tma_store_commit_group()     / tma_store_wait_group<N>()
//
// Usage:
//   tma_store_2d(...);                  // one or more
//   cp_async_bulk_commit_group();        // (or tma_store_commit_group)
//   cp_async_bulk_wait_group<0>();       // wait for all pending groups

#pragma once

// PTX:    9.7.10.28.6.1 (cp.async.bulk.commit_group), 9.7.10.28.6.2 (cp.async.bulk.wait_group)
//
#include <cuda.h>
#include <cstdint>

// -- cp_async_bulk_* convention ---------------------------------------------

__device__ __forceinline__
void cp_async_bulk_commit_group() {
  asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__
void cp_async_bulk_wait_group() {
  asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
}

// .read variant -- after a wait_group, ensures TMA-stored data is visible
// to subsequent reads in the issuing thread's generic proxy.
template <int N>
__device__ __forceinline__
void cp_async_bulk_wait_group_read() {
  asm volatile("cp.async.bulk.wait_group.read %0;\n" :: "n"(N) : "memory");
}

// -- tma_store_* convention --------------------------------------------------

__device__ __forceinline__
void tma_store_commit_group() {
  asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__
void tma_store_wait_group() {
  asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
}

// Explicit specialization for the common "wait for all" case.
template <>
__device__ __forceinline__
void tma_store_wait_group<0>() {
  asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
}
