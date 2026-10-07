// 76_packed_f32x2.cuh -- add/sub/mul/fma.f32x2  (packed 2x FP32)
//
// ARCH: sm_100a
//
// Two packed FP32 values in one .b64 register, computed in parallel -- a 2x
// throughput path for plain FP32 element-wise math (full IEEE precision, NOT
// fp16x2/bf16x2). Used to vectorize softmax / O-rescale element loops where
// each thread owns a contiguous run of FP32 values.
//
// The .b64 operand packs {lo f32, hi f32}; the op is element-wise, so a
// float2 reinterpreted to b64 round-trips consistently (lo<->.x, hi<->.y).
// There is no f32x2 form of exp2/max -- only add/sub/mul/fma (PTX 9.7.3).

#pragma once

// Source: PTX_ISA.md 9.7.3 (PTX ISA 9.4 sec 9.7.3.3-9.7.3.6: add/sub/mul/fma)
// PTX:    add/sub/mul/fma.f32x2  [PTX 8.6+, sm_100+]
//
#include <cstdint>
#include <cstring>
#include <vector_types.h>   // float2

namespace {
__device__ __forceinline__ uint64_t f32x2_bits(float2 v) {
  uint64_t b; __builtin_memcpy(&b, &v, 8); return b;
}
__device__ __forceinline__ float2 f32x2_make(uint64_t b) {
  float2 v; __builtin_memcpy(&v, &b, 8); return v;
}
}  // namespace

// d = a * b   (element-wise, round-to-nearest-even)
__device__ __forceinline__ float2 fmul2(float2 a, float2 b) {
  uint64_t d;
  asm volatile("mul.f32x2 %0, %1, %2;\n" : "=l"(d) : "l"(f32x2_bits(a)), "l"(f32x2_bits(b)));
  return f32x2_make(d);
}

// d = a + b
__device__ __forceinline__ float2 fadd2(float2 a, float2 b) {
  uint64_t d;
  asm volatile("add.f32x2 %0, %1, %2;\n" : "=l"(d) : "l"(f32x2_bits(a)), "l"(f32x2_bits(b)));
  return f32x2_make(d);
}

// d = a - b
__device__ __forceinline__ float2 fsub2(float2 a, float2 b) {
  uint64_t d;
  asm volatile("sub.f32x2 %0, %1, %2;\n" : "=l"(d) : "l"(f32x2_bits(a)), "l"(f32x2_bits(b)));
  return f32x2_make(d);
}

// d = a * b + c
__device__ __forceinline__ float2 ffma2(float2 a, float2 b, float2 c) {
  uint64_t d;
  asm volatile("fma.rn.f32x2 %0, %1, %2, %3;\n"
               : "=l"(d) : "l"(f32x2_bits(a)), "l"(f32x2_bits(b)), "l"(f32x2_bits(c)));
  return f32x2_make(d);
}

// scalar broadcast to both lanes
__device__ __forceinline__ float2 f32x2_splat(float s) { return make_float2(s, s); }

// c = a + b, lane-wise.
__device__ __forceinline__ void add_f32x2(float2& c, float2 const& a, float2 const& b) {
  asm volatile("add.f32x2 %0, %1, %2;\n"
               : "=l"(reinterpret_cast<uint64_t&>(c))
               : "l"(reinterpret_cast<uint64_t const&>(a)),
                 "l"(reinterpret_cast<uint64_t const&>(b)));
}

// c = a * b, lane-wise.
__device__ __forceinline__ void mul_f32x2(float2& c, float2 const& a, float2 const& b) {
  asm volatile("mul.f32x2 %0, %1, %2;\n"
               : "=l"(reinterpret_cast<uint64_t&>(c))
               : "l"(reinterpret_cast<uint64_t const&>(a)),
                 "l"(reinterpret_cast<uint64_t const&>(b)));
}

// d = a * b + c, lane-wise, round-to-nearest.
__device__ __forceinline__ void fma_f32x2(float2& d, float2 const& a, float2 const& b,
                                          float2 const& c) {
  asm volatile("fma.rn.f32x2 %0, %1, %2, %3;\n"
               : "=l"(reinterpret_cast<uint64_t&>(d))
               : "l"(reinterpret_cast<uint64_t const&>(a)),
                 "l"(reinterpret_cast<uint64_t const&>(b)),
                 "l"(reinterpret_cast<uint64_t const&>(c)));
}

// Packed 2x ex2 emulation (built on add/sub/fma.f32x2) lives in 77_ex2_approx.cuh.
