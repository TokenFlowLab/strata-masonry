#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// 49_redux_sync_f32.cuh -- redux.sync.{min,max}{.abs}{.NaN}.f32 (Blackwell only)
//
// ARCH: sm_100a
//
// Floating-point warp reduction (min / max), with optional absolute-value
// and NaN-propagation modifiers. Introduced on Blackwell (SM100+). Used
// in FMHA online softmax to compute the running max across a warp.
// Source: knowledge/instructions/reduce/redux_sync.md
// PTX:    9.7.15.13 (redux.sync, FP32 + .NaN modifiers)
//
#include <cstdint>

__device__ __forceinline__ float redux_sync_min_f32(
    float v, uint32_t mask = 0xFFFFFFFFu) {
  float r;
  asm volatile("redux.sync.min.f32 %0, %1, %2;\n"
               : "=f"(r) : "f"(v), "r"(mask));
  return r;
}

__device__ __forceinline__ float redux_sync_max_f32(
    float v, uint32_t mask = 0xFFFFFFFFu) {
  float r;
  asm volatile("redux.sync.max.f32 %0, %1, %2;\n"
               : "=f"(r) : "f"(v), "r"(mask));
  return r;
}

__device__ __forceinline__ float redux_sync_min_abs_f32(
    float v, uint32_t mask = 0xFFFFFFFFu) {
  float r;
  asm volatile("redux.sync.min.abs.f32 %0, %1, %2;\n"
               : "=f"(r) : "f"(v), "r"(mask));
  return r;
}

__device__ __forceinline__ float redux_sync_max_abs_f32(
    float v, uint32_t mask = 0xFFFFFFFFu) {
  float r;
  asm volatile("redux.sync.max.abs.f32 %0, %1, %2;\n"
               : "=f"(r) : "f"(v), "r"(mask));
  return r;
}

__device__ __forceinline__ float redux_sync_max_nan_f32(
    float v, uint32_t mask = 0xFFFFFFFFu) {
  float r;
  asm volatile("redux.sync.max.NaN.f32 %0, %1, %2;\n"
               : "=f"(r) : "f"(v), "r"(mask));
  return r;
}

// .NaN cross-products (NaN-propagating reductions; FMHA online softmax wants
// min.NaN to keep poisoned-tile signalling). Completes the {min,max} x
// {plain,abs} x {NaN} matrix.

__device__ __forceinline__ float redux_sync_min_nan_f32(
    float v, uint32_t mask = 0xFFFFFFFFu) {
  float r;
  asm volatile("redux.sync.min.NaN.f32 %0, %1, %2;\n"
               : "=f"(r) : "f"(v), "r"(mask));
  return r;
}

__device__ __forceinline__ float redux_sync_min_abs_nan_f32(
    float v, uint32_t mask = 0xFFFFFFFFu) {
  float r;
  asm volatile("redux.sync.min.abs.NaN.f32 %0, %1, %2;\n"
               : "=f"(r) : "f"(v), "r"(mask));
  return r;
}

__device__ __forceinline__ float redux_sync_max_abs_nan_f32(
    float v, uint32_t mask = 0xFFFFFFFFu) {
  float r;
  asm volatile("redux.sync.max.abs.NaN.f32 %0, %1, %2;\n"
               : "=f"(r) : "f"(v), "r"(mask));
  return r;
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
