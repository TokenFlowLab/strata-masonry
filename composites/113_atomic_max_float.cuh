// 113_atomic_max_float.cuh -- atomicMax on floats, via an order-preserving uint32 encoding.
//
// There is no atom.max.f32. The trick is to map float -> uint32 so that the unsigned integer
// order matches the float order, run the integer atomicMax (atom.shared.max.u32), and map back:
//
//   non-negative f (sign 0):  set the sign bit          -> lands above every encoded negative
//   negative f     (sign 1):  invert all 32 bits        -> reverses the magnitude order, which
//                                                          is what IEEE-754 negatives need
//
// Both cases are the same expression, so it is branchless:
//   mask = -int32_t(u >> 31) | 0x80000000   ->  0x80000000 if sign 0, 0xFFFFFFFF if sign 1
//
// The inverse rebuilds the mask from the ENCODED sign, hence the different expression.
//
// Gotcha: the reduction buffer must be seeded with the encoded identity (float_to_u32 of
// -FLT_MAX) ONCE and never cleared -- an accumulating atomicMax across iterations IS the
// running max. Clearing it per tile silently makes the max per-tile instead of running.
//
// NaN is not handled: its encoding sorts above +inf, so a NaN input poisons the reduction.
//
// PTX:    9.7.15.15 (atom.shared.max.u32)
//

#pragma once

#include <cstdint>

// float -> order-preserving uint32.
__device__ __forceinline__ uint32_t float_to_u32(float f) {
  uint32_t u = __float_as_uint(f);
  uint32_t mask = -int32_t(u >> 31) | 0x80000000u;
  return u ^ mask;
}

// Inverse of float_to_u32.
__device__ __forceinline__ float u32_to_float(uint32_t u) {
  uint32_t mask = ((u >> 31) - 1u) | 0x80000000u;
  return __uint_as_float(u ^ mask);
}

// atomicMax over floats in SMEM. p must already hold an ENCODED value (see the seeding gotcha).
__device__ __forceinline__ void atomic_max_float_smem(float * p, float v) {
  atomicMax((uint32_t*)p, float_to_u32(v));
}
