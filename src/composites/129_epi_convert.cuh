// 129_epi_convert.cuh -- FP32 accumulator -> output dtype (library)
//
// ARCH: sm_90a
//
// Type conversion for epilogue: pack FP32 accumulator values into
// stmatrix-compatible register layouts (u32 for 16-bit, u16 for 8-bit).
//
// Two parallel APIs:
//   epi_pack4<EpiOutDtype>           4-of-f32 -> two u32 (handles FP16/BF16/
//                                    E4M3/E5M2/FP32 pass-through).
//   cvt_f32x2_pack<OutDtype>         2-of-f32 -> u32 (FP16/BF16).
//   cvt_f32x2_pack_fp8<OutDtype>     2-of-f32 -> u16 (E4M3/E5M2).
//   cvt_f32x8_to_4u32<OutDtype>      8-of-f32 -> 4 u32 (for stmatrix.x4
//                                    FP16/BF16).
//   cvt_f32_to_i8_sat                f32 -> saturated int8.

#pragma once

// PTX:    9.7.10.24 (cvt: f32 -> f16 / bf16 / e4m3 / e5m2 / e2m1)
//
#include <cstdint>
#include "../primitives/63_cvt_f32_to_f16_bf16.cuh"
#include "../primitives/64_cvt_f32_to_fp8.cuh"
#include "../primitives/71_cvt_f32_to_int8.cuh"

// =============================================================================
// epi_pack4 dispatch (handles FP32 pass-through too)
// =============================================================================

enum class EpiOutDtype { FP16, BF16, E4M3, E5M2, FP32 };

// Pack 4 FP32 into output dtype. Returns two 32-bit regs (r0, r1).
// For FP16/BF16: r0 = pack(a0,a1), r1 = pack(a2,a3).
// For FP8: r0 packs all 4 bytes, r1 = 0.
// For FP32: r0/r1 = raw bits of a0/a1; a2/a3 ignored.
template <EpiOutDtype D>
__device__ __forceinline__
void epi_pack4(float a0, float a1, float a2, float a3,
               uint32_t& r0, uint32_t& r1) {
  if constexpr (D == EpiOutDtype::FP16) {
    r0 = cvt_pack_f32_to_f16x2(a0, a1);
    r1 = cvt_pack_f32_to_f16x2(a2, a3);
  } else if constexpr (D == EpiOutDtype::BF16) {
    r0 = cvt_pack_f32_to_bf16x2(a0, a1);
    r1 = cvt_pack_f32_to_bf16x2(a2, a3);
  } else if constexpr (D == EpiOutDtype::E4M3) {
    uint16_t p0 = cvt_pack_f32_to_e4m3x2(a0, a1);
    uint16_t p1 = cvt_pack_f32_to_e4m3x2(a2, a3);
    r0 = (uint32_t)p0 | ((uint32_t)p1 << 16);
    r1 = 0;
  } else if constexpr (D == EpiOutDtype::E5M2) {
    uint16_t p0 = cvt_pack_f32_to_e5m2x2(a0, a1);
    uint16_t p1 = cvt_pack_f32_to_e5m2x2(a2, a3);
    r0 = (uint32_t)p0 | ((uint32_t)p1 << 16);
    r1 = 0;
  } else {  // FP32 pass-through (raw bits)
    r0 = __float_as_uint(a0);
    r1 = __float_as_uint(a1);
    (void)a2;
    (void)a3;
  }
}

// =============================================================================
// Per-pair pack helpers
// =============================================================================

enum class OutDtype { F16, BF16, E4M3, E5M2 };

// 2 f32 -> packed 16-bit pair (FP16/BF16). Returns 0 for 8-bit types.
template <OutDtype DT>
__device__ __forceinline__
uint32_t cvt_f32x2_pack(float a, float b) {
  if constexpr (DT == OutDtype::F16) {
    return cvt_f32x2_to_f16x2(a, b);
  } else if constexpr (DT == OutDtype::BF16) {
    return cvt_f32x2_to_bf16x2(a, b);
  } else {
    return 0u;
  }
}

// 2 f32 -> packed 8-bit pair (E4M3/E5M2). Returns 0 for 16-bit types.
template <OutDtype DT>
__device__ __forceinline__
uint16_t cvt_f32x2_pack_fp8(float a, float b) {
  if constexpr (DT == OutDtype::E4M3) {
    return cvt_f32x2_to_e4m3x2(a, b);
  } else if constexpr (DT == OutDtype::E5M2) {
    return cvt_f32x2_to_e5m2x2(a, b);
  } else {
    return 0u;
  }
}

// 8 f32 -> 4 u32 (for stmatrix.x4 FP16/BF16).
template <OutDtype DT>
__device__ __forceinline__
void cvt_f32x8_to_4u32(const float (&in)[8], uint32_t (&out)[4]) {
  out[0] = cvt_f32x2_pack<DT>(in[0], in[1]);
  out[1] = cvt_f32x2_pack<DT>(in[2], in[3]);
  out[2] = cvt_f32x2_pack<DT>(in[4], in[5]);
  out[3] = cvt_f32x2_pack<DT>(in[6], in[7]);
}

// =============================================================================
// Saturating int8 quantization
// =============================================================================

__device__ __forceinline__
int8_t cvt_f32_to_i8_sat(float val, float scale = 1.0f) {
  return cvt_f32_to_s8_sat(val * scale);
}
