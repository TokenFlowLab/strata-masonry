#pragma once
#if defined(PL_AGENTIC_SM90A)
// 54_wgmma_f16_rs.cuh -- wgmma.mma_async.sync.aligned.m64nNk16.f32.f16.f16 (RS form)
//
// ARCH: sm_90a
//
// RS form variant of WGMMA: A from registers (typically loaded via
// ldmatrix.x4 -- see #39), B from SMEM via descriptor. Used when A is
// produced by a prior op already in registers, avoiding an SMEM round-trip.
//
// Constraints (PTX 9.7.17):
//   A operand: 4 packed b32 registers per thread (ldmatrix output layout)
//   wgmma.fence.sync.aligned must precede the first MMA in a group (#59)
//
// Issuer: warp-group (128 threads).
// PTX:    9.7.17.5 (wgmma.mma_async, f16 RS)
//
#include <cstdint>

// RS form: A from registers (4 u32 from ldmatrix), B from SMEM descriptor.
// Accumulator: N/2 f32 registers per thread.

// m64n8k16 f32.f16.f16 RS -- 4 accum regs
__device__ __forceinline__
void wgmma_f16_rs_m64n8k16(
    float (&d)[4],
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint64_t b_desc,
    bool scale_d = true) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %9, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, %8, p, 1, 1, 1;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "l"(b_desc), "r"((uint32_t)scale_d));
}

// m64n16k16 f32.f16.f16 RS -- 8 accum regs
__device__ __forceinline__
void wgmma_f16_rs_m64n16k16(
    float (&d)[8],
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint64_t b_desc,
    bool scale_d = true) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %13, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, %12, p, 1, 1, 1;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "l"(b_desc), "r"((uint32_t)scale_d));
}

// m64n8k32 f32.f16.f16 RS sparse 2:4 -- 4 accum regs
// A_compact in 4 registers (8 FP16 packed), B from SMEM.
// E_meta=0x44444444 selects {0,1} per 4-group -> K_active=16. D = 16.
template <int SP_SEL>
__device__ __forceinline__
void wgmma_f16_rs_sp_m64n8k32(
    float (&d)[4],
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint64_t b_desc,
    uint32_t e_meta,
    bool scale_d = true) {
    static_assert(SP_SEL >= 0 && SP_SEL <= 3, "SP_SEL must be 0..3");
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %10, 0;\n"
        "wgmma.mma_async.sp.sync.aligned.m64n8k32.f32.f16.f16 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, %8, %9, %11, p, 1, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "l"(b_desc), "r"(e_meta),
          "r"((uint32_t)scale_d), "n"(SP_SEL));
}

#endif  // PL_AGENTIC_SM90A
