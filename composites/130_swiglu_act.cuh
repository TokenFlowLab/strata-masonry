// 130_swiglu_act.cuh -- gated-activation builders: sigmoid, silu, SwiGLU.
//
// ARCH: sm_75 (FP32, FP16, FP16x2)  /  sm_90 (BF16, BF16x2)
//
// Built on primitive 72 (`tanh.approx`) via the half-angle identity:
//   sigmoid(z) = 1 / (1 + exp(-z))
//              = 0.5 + 0.5 * tanh(z / 2)
//   silu(z)    = z * sigmoid(z)
//              = 0.5 * z * (1 + tanh(z / 2))
//              = fma(0.5*z, tanh(z/2), 0.5*z)
//   SwiGLU(g,u)= silu(g) * u
//
// Properties vs the `__expf`-based form `z / (1 + __expf(-z))`:
//   - Same instruction count, but uses one SFU slot instead of two
//     (no `rcp.approx` in the critical path).
//   - Never produces +/-Inf for finite inputs (`tanh.approx` clamps to
//     +/-1.0 for |z| > ~9.0). Avoids the special-value slow path that
//     `__expf` enters on large accumulator magnitudes.
//
// This is the canonical fast form used by CUTLASS's `Silu` on SM 9.0+
// and by trtllm-gen's `GemmGatedAct` EPI on Blackwell.

#pragma once

// Source: knowledge/instructions/math/tanh.md (sec 6-8)
//
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "../primitives/72_tanh_approx.cuh"

// ============================================================================
// FP32
// ============================================================================

__device__ __forceinline__
float sigmoid_approx_f32(float z) {
  const float t = tanh_approx_f32(0.5f * z);
  return __fmaf_rn(0.5f, t, 0.5f);  // 0.5 + 0.5 * t
}

__device__ __forceinline__
float silu_approx_f32(float z) {
  const float half_z = 0.5f * z;
  const float t      = tanh_approx_f32(half_z);
  return __fmaf_rn(half_z, t, half_z);  // 0.5*z + 0.5*z * t
}

__device__ __forceinline__
float swiglu_act_f32(float gate, float up) {
  return silu_approx_f32(gate) * up;
}

// ============================================================================
// FP16 (scalar + packed). Same identity; uses tanh.approx.f16 / f16x2.
// ============================================================================

__device__ __forceinline__
__half sigmoid_approx_f16(__half z) {
  const __half half  = __float2half(0.5f);
  const __half t     = tanh_approx_f16(__hmul(half, z));
  return __hfma(half, t, half);
}

__device__ __forceinline__
__half silu_approx_f16(__half z) {
  const __half half  = __float2half(0.5f);
  const __half half_z = __hmul(half, z);
  const __half t      = tanh_approx_f16(half_z);
  return __hfma(half_z, t, half_z);
}

__device__ __forceinline__
__half swiglu_act_f16(__half gate, __half up) {
  return __hmul(silu_approx_f16(gate), up);
}

// Packed 2xFP16. Operates on the two half-words in parallel.
__device__ __forceinline__
__half2 silu_approx_f16x2(__half2 z) {
  const __half2 half  = __float2half2_rn(0.5f);
  const __half2 half_z = __hmul2(half, z);
  const __half2 t      = tanh_approx_f16x2(half_z);
  return __hfma2(half_z, t, half_z);
}

__device__ __forceinline__
__half2 swiglu_act_f16x2(__half2 gate, __half2 up) {
  return __hmul2(silu_approx_f16x2(gate), up);
}

// ============================================================================
// BF16 (scalar + packed). Same identity; uses tanh.approx.bf16 / bf16x2.
// ============================================================================

__device__ __forceinline__
__nv_bfloat16 sigmoid_approx_bf16(__nv_bfloat16 z) {
  const __nv_bfloat16 half = __float2bfloat16(0.5f);
  const __nv_bfloat16 t    = tanh_approx_bf16(__hmul(half, z));
  return __hfma(half, t, half);
}

__device__ __forceinline__
__nv_bfloat16 silu_approx_bf16(__nv_bfloat16 z) {
  const __nv_bfloat16 half   = __float2bfloat16(0.5f);
  const __nv_bfloat16 half_z = __hmul(half, z);
  const __nv_bfloat16 t      = tanh_approx_bf16(half_z);
  return __hfma(half_z, t, half_z);
}

__device__ __forceinline__
__nv_bfloat16 swiglu_act_bf16(__nv_bfloat16 gate, __nv_bfloat16 up) {
  return __hmul(silu_approx_bf16(gate), up);
}

// Packed 2xBF16. Operates on the two half-words in parallel. Useful when
// the EPI gathers (up, gate) pairs into a single __nv_bfloat162 for
// SIMD-style activation.
__device__ __forceinline__
__nv_bfloat162 silu_approx_bf16x2(__nv_bfloat162 z) {
  const __nv_bfloat162 half   = __float2bfloat162_rn(0.5f);
  const __nv_bfloat162 half_z = __hmul2(half, z);
  const __nv_bfloat162 t      = tanh_approx_bf16x2(half_z);
  return __hfma2(half_z, t, half_z);
}

__device__ __forceinline__
__nv_bfloat162 swiglu_act_bf16x2(__nv_bfloat162 gate, __nv_bfloat162 up) {
  return __hmul2(silu_approx_bf16x2(gate), up);
}
