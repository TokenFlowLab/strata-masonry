// 72_tanh_approx.cuh -- tanh.approx.{f32,f16,f16x2,bf16,bf16x2}
//                        (fast hardware hyperbolic tangent)
//
// ARCH: sm_75
//
// Raw inline-PTX wrappers around the `tanh.approx` instruction family.
// One MUFU.TANH SASS issue per call. Saturates to +/-1.0 for inputs
// outside ~[-9, 9] (FP32; the f16 / bf16 forms saturate sooner because
// the dynamic range is smaller).
//
// Corner cases (per PTX ISA 9.7.3.22 / 9.7.4.9):
//   tanh(+/-Inf) = +/-1.0
//   tanh(NaN)    = NaN
//   tanh(+/-0)   = +/-0  (sign preserved)
//
// All variants here are RAW 1:1 wrappers around a single PTX instruction.
// Higher-level builders (sigmoid, silu, SwiGLU) live in
// composites/130_swiglu_act.cuh.
//
// PTX availability:
//   tanh.approx.f32              PTX 7.0+, sm_75+
//   tanh.approx.f16 / f16x2      PTX 7.0+, sm_75+
//   tanh.approx.bf16 / bf16x2    PTX 7.8+, sm_90+

#pragma once

// PTX:    9.7.3.22  (tanh.approx.f32)
//         9.7.4.9   (tanh.approx.{f16, f16x2, bf16, bf16x2})
//
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// -- FP32 -------------------------------------------------------------------

__device__ __forceinline__
float tanh_approx_f32(float z) {
  float d;
  asm volatile("tanh.approx.f32 %0, %1;\n" : "=f"(d) : "f"(z));
  return d;
}

// -- FP16 scalar + packed ---------------------------------------------------

__device__ __forceinline__
__half tanh_approx_f16(__half z) {
  __half d;
  asm volatile("tanh.approx.f16 %0, %1;\n"
               : "=h"(*reinterpret_cast<uint16_t*>(&d))
               : "h"(*reinterpret_cast<uint16_t*>(&z)));
  return d;
}

// Packed 2xFP16. Operates on the two half-words in parallel:
//   d[0] = tanh(z[0]); d[1] = tanh(z[1])
__device__ __forceinline__
__half2 tanh_approx_f16x2(__half2 z) {
  __half2 d;
  asm volatile("tanh.approx.f16x2 %0, %1;\n"
               : "=r"(*reinterpret_cast<uint32_t*>(&d))
               : "r"(*reinterpret_cast<uint32_t*>(&z)));
  return d;
}

// -- BF16 scalar + packed ---------------------------------------------------

__device__ __forceinline__
__nv_bfloat16 tanh_approx_bf16(__nv_bfloat16 z) {
  __nv_bfloat16 d;
  asm volatile("tanh.approx.bf16 %0, %1;\n"
               : "=h"(*reinterpret_cast<uint16_t*>(&d))
               : "h"(*reinterpret_cast<uint16_t*>(&z)));
  return d;
}

// Packed 2xBF16. Operates on the two half-words in parallel:
//   d[0] = tanh(z[0]); d[1] = tanh(z[1])
__device__ __forceinline__
__nv_bfloat162 tanh_approx_bf16x2(__nv_bfloat162 z) {
  __nv_bfloat162 d;
  asm volatile("tanh.approx.bf16x2 %0, %1;\n"
               : "=r"(*reinterpret_cast<uint32_t*>(&d))
               : "r"(*reinterpret_cast<uint32_t*>(&z)));
  return d;
}
