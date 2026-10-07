// 71_cvt_f32_to_int8.cuh -- cvt.rni.sat.s8.f32 (FP32 -> INT8, round nearest, saturate)
//
// ARCH: sm_80+
//
// Single-instruction quantize-saturate conversion used by epilogues that
// produce int8 outputs (e.g. 8-bit per-tensor quantization). The PTX
// destination is .s32 (the .s8 result is sign-extended into a 32-bit
// register); this wrapper truncates to int8_t for the caller.
//
// Source: knowledge/instructions/convert/cvt.md
// PTX:    9.7.10.24 (cvt.{rni}{.sat}.s8.f32)
//
#pragma once

#include <cstdint>

__device__ __forceinline__
int8_t cvt_f32_to_s8_sat(float val) {
  int32_t i32;
  asm("cvt.rni.sat.s8.f32 %0, %1;\n" : "=r"(i32) : "f"(val));
  return static_cast<int8_t>(i32);
}
