// 48_redux_sync.cuh -- redux.sync.{add,min,max}.{u32,s32} / .{and,or,xor}.b32
//
// ARCH: sm_90a
//
// Hardware-accelerated warp-collective reduction across active lanes.
// Faster than shfl-based idioms because the hardware computes the result
// in one cycle.
//
// The mask argument selects which lanes participate (0xFFFFFFFF = all 32).
// Float ops (min/max with .abs/.NaN) are sm_100a+ only -- see primitive 49.

#pragma once

// PTX:    9.7.15.13 (redux.sync, integer)
//
#include <cstdint>

// -- add ---------------------------------------------------------------------

__device__ __forceinline__
int32_t redux_sync_add_s32(int32_t v, uint32_t mask = 0xFFFFFFFFu) {
  int32_t r;
  asm volatile("redux.sync.add.s32 %0, %1, %2;\n"
               : "=r"(r) : "r"(v), "r"(mask));
  return r;
}

__device__ __forceinline__
uint32_t redux_sync_add_u32(uint32_t v, uint32_t mask = 0xFFFFFFFFu) {
  uint32_t r;
  asm volatile("redux.sync.add.u32 %0, %1, %2;\n"
               : "=r"(r) : "r"(v), "r"(mask));
  return r;
}

// -- min / max ---------------------------------------------------------------

__device__ __forceinline__
int32_t redux_sync_min_s32(int32_t v, uint32_t mask = 0xFFFFFFFFu) {
  int32_t r;
  asm volatile("redux.sync.min.s32 %0, %1, %2;\n"
               : "=r"(r) : "r"(v), "r"(mask));
  return r;
}

__device__ __forceinline__
int32_t redux_sync_max_s32(int32_t v, uint32_t mask = 0xFFFFFFFFu) {
  int32_t r;
  asm volatile("redux.sync.max.s32 %0, %1, %2;\n"
               : "=r"(r) : "r"(v), "r"(mask));
  return r;
}

__device__ __forceinline__
uint32_t redux_sync_min_u32(uint32_t v, uint32_t mask = 0xFFFFFFFFu) {
  uint32_t r;
  asm volatile("redux.sync.min.u32 %0, %1, %2;\n"
               : "=r"(r) : "r"(v), "r"(mask));
  return r;
}

__device__ __forceinline__
uint32_t redux_sync_max_u32(uint32_t v, uint32_t mask = 0xFFFFFFFFu) {
  uint32_t r;
  asm volatile("redux.sync.max.u32 %0, %1, %2;\n"
               : "=r"(r) : "r"(v), "r"(mask));
  return r;
}

// -- bitwise ----------------------------------------------------------------

__device__ __forceinline__
uint32_t redux_sync_and_b32(uint32_t v, uint32_t mask = 0xFFFFFFFFu) {
  uint32_t r;
  asm volatile("redux.sync.and.b32 %0, %1, %2;\n"
               : "=r"(r) : "r"(v), "r"(mask));
  return r;
}

__device__ __forceinline__
uint32_t redux_sync_or_b32(uint32_t v, uint32_t mask = 0xFFFFFFFFu) {
  uint32_t r;
  asm volatile("redux.sync.or.b32 %0, %1, %2;\n"
               : "=r"(r) : "r"(v), "r"(mask));
  return r;
}

__device__ __forceinline__
uint32_t redux_sync_xor_b32(uint32_t v, uint32_t mask = 0xFFFFFFFFu) {
  uint32_t r;
  asm volatile("redux.sync.xor.b32 %0, %1, %2;\n"
               : "=r"(r) : "r"(v), "r"(mask));
  return r;
}
