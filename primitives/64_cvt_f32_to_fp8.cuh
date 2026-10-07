// 64_cvt_f32_to_fp8.cuh -- cvt.rn.satfinite.e{4m3,5m2}x2.f32
//                          (FP32 pair -> FP8 pair, plus scalar wrappers)
//
// ARCH: sm_90a
//
// FP8 cvt produces a packed 16-bit register ({e_hi:8, e_lo:8}). The PTX
// mnemonic outputs a .b16 destination, so the asm constraint is "=h".
//
// E4M3: 4-bit exponent, 3-bit mantissa (higher precision, smaller range, max 448)
// E5M2: 5-bit exponent, 2-bit mantissa (lower precision, larger range)
//
// Two name forms are provided for the packed conversions (different naming
// conventions, identical body):
//   cvt_pack_f32_to_e4m3x2(lo, hi)  /  cvt_f32x2_to_e4m3x2(a, b)
//   cvt_pack_f32_to_e5m2x2(lo, hi)  /  cvt_f32x2_to_e5m2x2(a, b)

#pragma once

// Source: knowledge/instructions/convert/cvt.md
// PTX:    9.7.10.24 (cvt: f32 -> e4m3 / e5m2)
//
#include <cstdint>

// -- packed pair conversions -------------------------------------------------

__device__ __forceinline__
uint16_t cvt_pack_f32_to_e4m3x2(float lo, float hi) {
  uint16_t r;
  asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;\n"
               : "=h"(r) : "f"(lo), "f"(hi));
  return r;
}

__device__ __forceinline__
uint16_t cvt_f32x2_to_e4m3x2(float a, float b) {
  uint16_t r;
  asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;\n"
               : "=h"(r) : "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__
uint16_t cvt_pack_f32_to_e5m2x2(float lo, float hi) {
  uint16_t r;
  asm volatile("cvt.rn.satfinite.e5m2x2.f32 %0, %2, %1;\n"
               : "=h"(r) : "f"(lo), "f"(hi));
  return r;
}

__device__ __forceinline__
uint16_t cvt_f32x2_to_e5m2x2(float a, float b) {
  uint16_t r;
  asm volatile("cvt.rn.satfinite.e5m2x2.f32 %0, %2, %1;\n"
               : "=h"(r) : "f"(a), "f"(b));
  return r;
}

// -- packed pair conversions with .relu modifier ----------------------------
// `.relu` clamps negative results to +0 (fused activation FP8 epilogues).

__device__ __forceinline__
uint16_t cvt_pack_f32_to_e4m3x2_relu(float lo, float hi) {
  uint16_t r;
  asm volatile("cvt.rn.relu.satfinite.e4m3x2.f32 %0, %2, %1;\n"
               : "=h"(r) : "f"(lo), "f"(hi));
  return r;
}

__device__ __forceinline__
uint16_t cvt_pack_f32_to_e5m2x2_relu(float lo, float hi) {
  uint16_t r;
  asm volatile("cvt.rn.relu.satfinite.e5m2x2.f32 %0, %2, %1;\n"
               : "=h"(r) : "f"(lo), "f"(hi));
  return r;
}

// -- scalar (one FP32 -> one FP8 in low byte; high byte mirrors low) --------

__device__ __forceinline__
uint16_t cvt_f32_to_e4m3(float v) {
  uint16_t r;
  asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %1;\n"
               : "=h"(r) : "f"(v));
  return r;
}

__device__ __forceinline__
uint16_t cvt_f32_to_e5m2(float v) {
  uint16_t r;
  asm volatile("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %1;\n"
               : "=h"(r) : "f"(v));
  return r;
}
