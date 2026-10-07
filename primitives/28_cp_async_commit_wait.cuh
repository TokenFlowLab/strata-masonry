// 28_cp_async_commit_wait.cuh -- cp.async.commit_group / wait_group
//
// ARCH: sm_90a
//
// Async-group management for cp.async (non-bulk) copies (#26, #27).
// commit_group: closes the current async-group (batches all prior uncommitted
//   cp.async ops from this thread into a new group).
// wait_group<N>: blocks until at most N groups remain pending.
//
// SEPARATE namespace from cp.async.bulk.commit_group (TMA stores; #25).
// They do NOT interact.

#pragma once

// Source: knowledge/instructions/copy/cp_async.md
// PTX:    9.7.10.28.3.2 (cp.async.commit_group), 9.7.10.28.3.3 (cp.async.wait_group)
//
#include <cstdint>

// Close the current async-group.
__device__ __forceinline__
void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

// Wait until at most N async-groups remain pending.
template <int N>
__device__ __forceinline__
void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

// Wait for ALL outstanding cp.async groups (emits cp.async.wait_all).
__device__ __forceinline__
void cp_async_wait_all() {
  asm volatile("cp.async.wait_all;\n" ::: "memory");
}

// Same as cp_async_wait_all() but emits the equivalent cp.async.wait_group 0
// form. Kept as a separate function so callers can choose which PTX form
// they want to emit (the two compile to different SASS).
__device__ __forceinline__
void cp_async_wait_all_via_group0() {
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

// Signal mbarrier when this thread's pending cp.async ops complete.
__device__ __forceinline__
void cp_async_mbarrier_arrive(uint32_t mbar_smem) {
  asm volatile("cp.async.mbarrier.arrive.shared.b64 [%0];\n"
               :: "r"(mbar_smem) : "memory");
}

// Signal mbarrier (no-inc variant, for pre-counted barriers).
__device__ __forceinline__
void cp_async_mbarrier_arrive_noinc(uint32_t mbar_smem) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n"
               :: "r"(mbar_smem) : "memory");
}
