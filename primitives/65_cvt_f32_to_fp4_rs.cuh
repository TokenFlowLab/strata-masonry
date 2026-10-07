#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// 65_cvt_f32_to_fp4_rs.cuh -- cvt.rs.satfinite.{e2m1x4,e4m3x4,e5m2x4,e3m2x4,e2m3x4}.f32
//
// ARCH: sm_100a, sm_103a
//
// Stochastic-rounding FP32 -> 4-element narrow-precision conversion.
// Each call takes 4 FP32 inputs and a 32-bit random-bits register and
// produces 4 packed outputs:
//   e2m1x4 -> 16 bits (FP4)
//   e4m3x4, e5m2x4, e3m2x4, e2m3x4 -> 32 bits (FP6 / FP8)
// .rs rounds toward or away from zero on the carry out of adding rbits
// to the discarded mantissa bits of the input.
//
// Constraints (PTX ISA 9.7.10):
//   `rbits` is one 32-bit reg, all 32 bits used. FP8/FP6: high 16 -> a,b,
//   low 16 -> e,f. e2m1x4: high byte of each 16-bit half -> a,b, low -> e,f.
//   Caller supplies the entropy (e.g. via curand) per thread, per call.
//
// Issuer: any thread; per-thread instruction.
// Source: knowledge/instructions/convert/cvt.md
// PTX:    9.7.10.24 (cvt.rs: f32 -> e2m1 FP4)
//
#include <cstdint>

// FP32 -> 4xFP4 E2M1 packed in 16 bits, high nibble first:
// a -> d[15:12], b -> d[11:8], e -> d[7:4], f -> d[3:0].
__device__ __forceinline__ uint16_t cvt_rs_satfinite_e2m1x4_f32(
    float a, float b, float e, float f, uint32_t rbits)
{
    uint16_t d;
    asm volatile(
        "cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;\n"
        : "=h"(d) : "f"(a), "f"(b), "f"(e), "f"(f), "r"(rbits));
    return d;
}

// FP32 -> 4xFP8 E4M3 packed in 32 bits.
__device__ __forceinline__ uint32_t cvt_rs_satfinite_e4m3x4_f32(
    float a, float b, float e, float f, uint32_t rbits)
{
    uint32_t d;
    asm volatile(
        "cvt.rs.satfinite.e4m3x4.f32 %0, {%1, %2, %3, %4}, %5;\n"
        : "=r"(d) : "f"(a), "f"(b), "f"(e), "f"(f), "r"(rbits));
    return d;
}

// FP32 -> 4xFP8 E5M2 packed in 32 bits.
__device__ __forceinline__ uint32_t cvt_rs_satfinite_e5m2x4_f32(
    float a, float b, float e, float f, uint32_t rbits)
{
    uint32_t d;
    asm volatile(
        "cvt.rs.satfinite.e5m2x4.f32 %0, {%1, %2, %3, %4}, %5;\n"
        : "=r"(d) : "f"(a), "f"(b), "f"(e), "f"(f), "r"(rbits));
    return d;
}

// FP32 -> 4xFP6 E3M2 packed in 32 bits.
__device__ __forceinline__ uint32_t cvt_rs_satfinite_e3m2x4_f32(
    float a, float b, float e, float f, uint32_t rbits)
{
    uint32_t d;
    asm volatile(
        "cvt.rs.satfinite.e3m2x4.f32 %0, {%1, %2, %3, %4}, %5;\n"
        : "=r"(d) : "f"(a), "f"(b), "f"(e), "f"(f), "r"(rbits));
    return d;
}

// FP32 -> 4xFP6 E2M3 packed in 32 bits.
__device__ __forceinline__ uint32_t cvt_rs_satfinite_e2m3x4_f32(
    float a, float b, float e, float f, uint32_t rbits)
{
    uint32_t d;
    asm volatile(
        "cvt.rs.satfinite.e2m3x4.f32 %0, {%1, %2, %3, %4}, %5;\n"
        : "=r"(d) : "f"(a), "f"(b), "f"(e), "f"(f), "r"(rbits));
    return d;
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A
