// 77_ex2_approx.cuh -- ex2.approx.f32 (HW MUFU) + packed 2x ex2 polynomial emulation.
//
// ARCH: sm_100a  (the f32x2 emulation needs add/sub/fma.f32x2; the scalar HW
//                 wrapper alone works on much older arch -- MUFU.EX2 is sm_50+.)
//
// Two ways to compute 2^x for online-softmax / exp paths:
//
//   ex2_approx_f32(x)   -- one MUFU.EX2 SASS issue (ex2.approx.f32). This is what
//                          __exp2f() lowers to, exposed explicitly. ~1 ULP-ish
//                          hardware approximation, NOT the ~1 ULP libm exp2f.
//
//   ex2_emu_f32x2(x,y)  -- packed 2x 2^x with NO MUFU: 2^x = 2^floor(x) *
//                          poly3(frac(x)). Runs entirely on the f32x2 ALU pipe,
//                          so it OFFLOADS the MUFU.EX2 unit. Used by FlashAttention-4
//                          (apply_exp2_convert, ex2_emu_freq) to balance the softmax
//                          warp between the MUFU and FMA pipes when EX2-bound.
//
// Emulation algorithm (port of flash-attention cute/utils.py; ops non-ftz to match the
// CURRENT ex2_emulation_2 -- FA4's shipped SASS has zero .FTZ in the emu; the old
// e2e_asm2 asm block used .ftz variants):
//   1. clamp x >= -127           (max.f32, magic 0fC2FE0000)
//   2. add the fp32 round const 2^23+2^22 (0f4B400000) with round-DOWN
//      (add.rm.f32x2) -- the integer floor lands in the low mantissa bits.
//   3. subtract it back (round-nearest) to recover floor(x) as a float;
//      x - floor = frac in [0, 1).
//   4. poly3(frac) ~= 2^frac via Horner FMA (coeffs baked in the PTX below).
//   5. shift floor into the exponent field (shl 23) and int-ADD to the poly
//      bits -> multiply poly by 2^floor.
// Corner: ex2_emu of a masked -inf input -> clamps to -127 -> 2^-127 ~= 0 (so
// masked softmax entries still vanish, like ex2_approx_f32(-inf) = 0).
//
// PTX:    ex2.approx.f32                 [PTX 1.4+, sm_50+]
//         add/sub/fma.f32x2 (.rm/.rn)    [PTX 8.6+, sm_100+]   (PTX ISA 9.7.3)

#pragma once

#include <cstdint>
#include <cstring>
#include <vector_types.h>   // float2

// -- HW ex2.approx.f32 ------------------------------------------------------

// d = 2^z, single MUFU.EX2. .ftz so ptxas emits ONE MUFU.EX2 (the non-ftz form is lowered
// with a sub-(-126) range extension: FSETP/FMUL-0.5/square -> extra FMULs). Matches __exp2f.
__device__ __forceinline__ float ex2_approx_f32(float z) {
  float d;
  asm volatile("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(d) : "f"(z));
  return d;
}

// -- Packed 2x ex2 EMULATION (no MUFU) --------------------------------------

// (ox, oy) = (2^x, 2^y), computed on the f32x2 ALU pipe. Adapted from flash-attention
// (flash_attn/cute/utils.py e2e_asm2, BSD-3-Clause); see THIRD_PARTY_NOTICES.md.
__device__ __forceinline__ float2 ex2_emu_f32x2(float x, float y) {
  uint32_t ox, oy;
  asm volatile(
    "{\n\t"
    ".reg .f32 f1,f2,f3,f4,f5,f6,f7;\n\t"
    ".reg .b64 l1,l2,l3,l4,l5,l6,l7,l8,l9,l10;\n\t"
    ".reg .s32 r1,r2,r3,r4,r5,r6,r7,r8;\n\t"
    "max.f32 f1, %2, 0fC2FE0000;\n\t"
    "max.f32 f2, %3, 0fC2FE0000;\n\t"
    "mov.b64 l1, {f1, f2};\n\t"
    "mov.f32 f3, 0f4B400000;\n\t"
    "mov.b64 l2, {f3, f3};\n\t"
    "add.rm.f32x2 l7, l1, l2;\n\t"
    "sub.rn.f32x2 l8, l7, l2;\n\t"
    "sub.rn.f32x2 l9, l1, l8;\n\t"
    "mov.f32 f7, 0f3D9DF09D;\n\t"
    "mov.b64 l6, {f7, f7};\n\t"
    "mov.f32 f6, 0f3E6906A4;\n\t"
    "mov.b64 l5, {f6, f6};\n\t"
    "mov.f32 f5, 0f3F31F519;\n\t"
    "mov.b64 l4, {f5, f5};\n\t"
    "mov.f32 f4, 0f3F800000;\n\t"
    "mov.b64 l3, {f4, f4};\n\t"
    "fma.rn.f32x2 l10, l9, l6, l5;\n\t"
    "fma.rn.f32x2 l10, l10, l9, l4;\n\t"
    "fma.rn.f32x2 l10, l10, l9, l3;\n\t"
    "mov.b64 {r1, r2}, l7;\n\t"
    "mov.b64 {r3, r4}, l10;\n\t"
    "shl.b32 r5, r1, 23;\n\t"
    "add.s32 r7, r5, r3;\n\t"
    "shl.b32 r6, r2, 23;\n\t"
    "add.s32 r8, r6, r4;\n\t"
    "mov.b32 %0, r7;\n\t"
    "mov.b32 %1, r8;\n\t"
    "}\n"
    : "=r"(ox), "=r"(oy) : "f"(x), "f"(y));
  float2 r; __builtin_memcpy(&r.x, &ox, 4); __builtin_memcpy(&r.y, &oy, 4);
  return r;
}
