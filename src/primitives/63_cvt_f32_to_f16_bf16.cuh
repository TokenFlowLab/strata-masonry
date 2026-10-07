// 63_cvt_f32_to_f16_bf16.cuh -- cvt.rn.{f16,bf16}.f32  (pair-packing cvt)
//
// ARCH: sm_90a
//
// Convert a pair of FP32 values into one packed b32 containing two FP16
// (or BF16) values. Rounds-to-nearest-even. Used when staging epilogue data
// from FP32 accumulators before stmatrix to SMEM.
//
// Two name forms are provided for the packed conversions (different naming
// conventions, identical body):
//   cvt_pack_f32_to_f16x2(lo, hi)  /  cvt_f32x2_to_f16x2(a, b)
//   cvt_pack_f32_to_bf16x2(lo, hi) /  cvt_f32x2_to_bf16x2(a, b)

#pragma once

// PTX:    9.7.10.24 (cvt: f32 -> f16 / bf16)
//
#include <cstdint>

// -- packed pair conversions -------------------------------------------------

// Pack two FP32 -> one b32 = {f16_hi, f16_lo}.
__device__ __forceinline__
uint32_t cvt_pack_f32_to_f16x2(float lo, float hi) {
  uint32_t r;
  asm volatile("cvt.rn.f16x2.f32 %0, %2, %1;\n"
               : "=r"(r) : "f"(lo), "f"(hi));
  return r;
}

__device__ __forceinline__
uint32_t cvt_f32x2_to_f16x2(float a, float b) {
  uint32_t r;
  asm volatile("cvt.rn.f16x2.f32 %0, %2, %1;\n"
               : "=r"(r) : "f"(a), "f"(b));
  return r;
}

// Pack two FP32 -> one b32 = {bf16_hi, bf16_lo}.
__device__ __forceinline__
uint32_t cvt_pack_f32_to_bf16x2(float lo, float hi) {
  uint32_t r;
  asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;\n"
               : "=r"(r) : "f"(lo), "f"(hi));
  return r;
}

__device__ __forceinline__
uint32_t cvt_f32x2_to_bf16x2(float a, float b) {
  uint32_t r;
  asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;\n"
               : "=r"(r) : "f"(a), "f"(b));
  return r;
}

// Same, saturating: overflow clamps to the largest finite bf16 instead of +/-Inf, NaN -> 0.
// **Gotcha:** PTX packs {hi, lo}, so the FIRST argument lands in the high half.
__device__ __forceinline__
uint32_t cvt_pack_f32_to_bf16x2_satfinite(float lo, float hi) {
  uint32_t r;
  asm volatile("cvt.rn.satfinite.bf16x2.f32 %0, %2, %1;\n"
               : "=r"(r) : "f"(lo), "f"(hi));
  return r;
}

// -- packed pair conversions with .relu modifier ----------------------------
// `.relu` clamps negative results to +0 (fused activation epilogues).
// Note: `.ftz` is NOT legal on packed f16x2/bf16x2 cvt forms per ptxas
// (only on scalar forms, see below).

__device__ __forceinline__
uint32_t cvt_pack_f32_to_f16x2_relu(float lo, float hi) {
  uint32_t r;
  asm volatile("cvt.rn.relu.f16x2.f32 %0, %2, %1;\n"
               : "=r"(r) : "f"(lo), "f"(hi));
  return r;
}

__device__ __forceinline__
uint32_t cvt_pack_f32_to_bf16x2_relu(float lo, float hi) {
  uint32_t r;
  asm volatile("cvt.rn.relu.bf16x2.f32 %0, %2, %1;\n"
               : "=r"(r) : "f"(lo), "f"(hi));
  return r;
}

// -- scalar conversions (one FP32 -> one FP16/BF16, low halfword) -----------

__device__ __forceinline__
uint16_t cvt_f32_to_f16(float v) {
  uint16_t r;
  asm volatile("cvt.rn.f16.f32 %0, %1;\n"
               : "=h"(r) : "f"(v));
  return r;
}

__device__ __forceinline__
uint16_t cvt_f32_to_bf16(float v) {
  uint16_t r;
  asm volatile("cvt.rn.bf16.f32 %0, %1;\n"
               : "=h"(r) : "f"(v));
  return r;
}

// -- scalar conversions with .ftz / .relu modifiers -------------------------
// Scalar forms accept both modifiers (unlike packed forms above).

__device__ __forceinline__
uint16_t cvt_f32_to_f16_ftz(float v) {
  uint16_t r;
  asm volatile("cvt.rn.ftz.f16.f32 %0, %1;\n"
               : "=h"(r) : "f"(v));
  return r;
}

__device__ __forceinline__
uint16_t cvt_f32_to_f16_relu(float v) {
  uint16_t r;
  asm volatile("cvt.rn.relu.f16.f32 %0, %1;\n"
               : "=h"(r) : "f"(v));
  return r;
}

__device__ __forceinline__
uint16_t cvt_f32_to_bf16_ftz(float v) {
  uint16_t r;
  asm volatile("cvt.rn.ftz.bf16.f32 %0, %1;\n"
               : "=h"(r) : "f"(v));
  return r;
}

__device__ __forceinline__
uint16_t cvt_f32_to_bf16_relu(float v) {
  uint16_t r;
  asm volatile("cvt.rn.relu.bf16.f32 %0, %1;\n"
               : "=h"(r) : "f"(v));
  return r;
}
