// 37_bar_sync.cuh -- bar.sync / barrier.cta.sync.aligned (named barriers)
//
// ARCH: sm_90a
//
// Named-barrier sync across a subset of threads in the CTA.
// Up to 16 distinct barriers per CTA (barrier_id in [0, 15]).
// thread_count must be a multiple of 32 (warp-aligned).
// Useful for syncing specific warp groups without syncing the whole CTA.
//
// Variants:
//   bar.sync N, thread_count          : block until thread_count threads arrive.
//   bar.sync 0                        : sync ALL threads in CTA (~ __syncthreads()).
//   barrier.cta.sync.aligned          : newer PTX spelling of bar.sync.
//   bar.arrive N, thread_count        : arrive without blocking (producer side).
//   barrier.cta.red.popc.aligned u32  : barrier + reduction (count predicate-true).

#pragma once

// Source: knowledge/instructions/barrier/bar_sync.md
// PTX:    9.7.15.1 (bar / barrier)
//
#include <cstdint>

// bar.sync N, thread_count.
template <int BARRIER_ID>
__device__ __forceinline__
void bar_sync(uint32_t thread_count) {
  static_assert(BARRIER_ID >= 0 && BARRIER_ID <= 15,
                "bar.sync: BARRIER_ID must be in [0, 15]");
  asm volatile("bar.sync %0, %1;\n"
               :: "n"(BARRIER_ID), "r"(thread_count) : "memory");
}

// bar.sync 0 -- equivalent to __syncthreads().
__device__ __forceinline__
void bar_sync_all() {
  asm volatile("bar.sync 0;\n" ::: "memory");
}

// Alternative spelling: barrier.cta.sync.aligned N, thread_count.
template <int BARRIER_ID>
__device__ __forceinline__
void barrier_cta_sync_aligned(uint32_t thread_count) {
  static_assert(BARRIER_ID >= 0 && BARRIER_ID <= 15,
                "barrier.cta.sync.aligned: BARRIER_ID must be in [0, 15]");
  asm volatile("barrier.cta.sync.aligned %0, %1;\n"
               :: "n"(BARRIER_ID), "r"(thread_count) : "memory");
}

// bar.arrive N, thread_count -- arrive without blocking.
template <int BARRIER_ID>
__device__ __forceinline__
void bar_arrive(uint32_t thread_count) {
  static_assert(BARRIER_ID >= 0 && BARRIER_ID <= 15,
                "bar.arrive: BARRIER_ID must be in [0, 15]");
  asm volatile("bar.arrive %0, %1;\n"
               :: "n"(BARRIER_ID), "r"(thread_count) : "memory");
}

// Runtime-id variants: bar.sync/arrive take the barrier id in a REGISTER (PTX allows
// a register operand), so callers with a computed id emit ONE BAR instruction instead
// of a switch + jump table over the template immediates.
__device__ __forceinline__
void bar_sync_dyn(uint32_t barrier_id, uint32_t thread_count) {
  asm volatile("bar.sync %0, %1;\n" :: "r"(barrier_id), "r"(thread_count) : "memory");
}
__device__ __forceinline__
void bar_arrive_dyn(uint32_t barrier_id, uint32_t thread_count) {
  asm volatile("bar.arrive %0, %1;\n" :: "r"(barrier_id), "r"(thread_count) : "memory");
}

// barrier.cta.red.popc.aligned: barrier + reduction.
// Returns the count of threads with pred = true.
template <int BARRIER_ID>
__device__ __forceinline__
uint32_t barrier_cta_red_popc(uint32_t thread_count, bool pred) {
  uint32_t count;
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.u32 p, %3, 0;\n\t"
    "barrier.cta.red.popc.aligned.u32 %0, %1, %2, p;\n\t"
    "}\n"
    : "=r"(count)
    : "n"(BARRIER_ID), "r"(thread_count), "r"((uint32_t)pred));
  return count;
}

// barrier.cta.red.and.aligned.pred: barrier + logical AND of predicates.
// Returns true iff every participating thread's pred is true.
template <int BARRIER_ID>
__device__ __forceinline__
bool barrier_cta_red_and(uint32_t thread_count, bool pred) {
  uint32_t result;
  asm volatile(
    "{\n\t"
    ".reg .pred p, q;\n\t"
    "setp.ne.u32 p, %3, 0;\n\t"
    "barrier.cta.red.and.aligned.pred q, %1, %2, p;\n\t"
    "selp.b32 %0, 1, 0, q;\n\t"
    "}\n"
    : "=r"(result)
    : "n"(BARRIER_ID), "r"(thread_count), "r"((uint32_t)pred));
  return result != 0;
}

// barrier.cta.red.or.aligned.pred: barrier + logical OR of predicates.
// Returns true iff at least one participating thread's pred is true.
template <int BARRIER_ID>
__device__ __forceinline__
bool barrier_cta_red_or(uint32_t thread_count, bool pred) {
  uint32_t result;
  asm volatile(
    "{\n\t"
    ".reg .pred p, q;\n\t"
    "setp.ne.u32 p, %3, 0;\n\t"
    "barrier.cta.red.or.aligned.pred q, %1, %2, p;\n\t"
    "selp.b32 %0, 1, 0, q;\n\t"
    "}\n"
    : "=r"(result)
    : "n"(BARRIER_ID), "r"(thread_count), "r"((uint32_t)pred));
  return result != 0;
}

// barrier.cta.arrive.aligned: newer PTX spelling of `bar.arrive` (#37
// already exposes the older name `bar_arrive`); kept for symmetry with
// the `barrier.cta.sync.aligned` form.
template <int BARRIER_ID>
__device__ __forceinline__
void barrier_cta_arrive_aligned(uint32_t thread_count) {
  static_assert(BARRIER_ID >= 0 && BARRIER_ID <= 15,
                "barrier.cta.arrive.aligned: BARRIER_ID must be in [0, 15]");
  asm volatile("barrier.cta.arrive.aligned %0, %1;\n"
               :: "n"(BARRIER_ID), "r"(thread_count) : "memory");
}
