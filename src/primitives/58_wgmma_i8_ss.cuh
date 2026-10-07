#pragma once
#if defined(PL_AGENTIC_SM90A)
// 58_wgmma_i8_ss.cuh -- wgmma.mma_async.sync.aligned.m64nNk32.s32.s8.s8 (SS form)
//
// ARCH: sm_90a
//
// INT8 variant: K=32 with S32 accumulator (signed integer GEMM).
// Used in quantized inference paths.
//
// Constraints (PTX 9.7.17):
//   K must be 32 (single block per MMA)
//   only scale_d argument (no scale_a, scale_b, no transpose)
//   wgmma.fence.sync.aligned must precede the first MMA in a group (#59)
//
// Issuer: warp-group (128 threads).
// PTX:    9.7.17.5 (wgmma.mma_async, s8 / u8)
//
#include <cstdint>

// INT8 has K=32. Accum: N/2 s32 registers.
// Only scale_d argument (no scale_a, scale_b, no transpose).

// m64n8k32 s32.s8.s8 SS -- 4 accum regs
__device__ __forceinline__
void wgmma_i8_ss_m64n8k32(int32_t (&d)[4],
                            uint64_t a_desc, uint64_t b_desc,
                            bool scale_d = true) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %6, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.s8 "
        "{%0, %1, %2, %3}, %4, %5, p;\n}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "l"(a_desc), "l"(b_desc), "r"((uint32_t)scale_d));
}

// m64n16k32 s32.s8.s8 SS -- 8 accum regs
__device__ __forceinline__
void wgmma_i8_ss_m64n16k32(int32_t (&d)[8],
                             uint64_t a_desc, uint64_t b_desc,
                             bool scale_d = true) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %10, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k32.s32.s8.s8 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, p;\n}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]),
          "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7])
        : "l"(a_desc), "l"(b_desc), "r"((uint32_t)scale_d));
}

// m64n64k32 s32.s8.s8 SS -- 32 accum regs
// Production tile size for INT8 GEMM.
__device__ __forceinline__
void wgmma_i8_ss_m64n64k32(int32_t (&d)[32],
                            uint64_t a_desc, uint64_t b_desc,
                            bool scale_d = true) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k32.s32.s8.s8 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        " %8, %9, %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31}, "
        "%32, %33, p;\n}\n"
        : "+r"(d[0]),  "+r"(d[1]),  "+r"(d[2]),  "+r"(d[3]),
          "+r"(d[4]),  "+r"(d[5]),  "+r"(d[6]),  "+r"(d[7]),
          "+r"(d[8]),  "+r"(d[9]),  "+r"(d[10]), "+r"(d[11]),
          "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]),
          "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]),
          "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]),
          "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]),
          "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31])
        : "l"(a_desc), "l"(b_desc), "r"((uint32_t)scale_d));
}

// ---------------------------------------------------------------------------
// Unsigned and mixed-sign variants. Covers all 4 dtype combinations of
// {s8,u8} x {s8,u8} for the m64n8k32 shape (one wrapper per combo). Used by
// quantized GEMM where activations and weights may have different sign
// conventions.
// ---------------------------------------------------------------------------

// m64n8k32 s32.u8.u8 SS -- both A and B unsigned 8-bit.
__device__ __forceinline__
void wgmma_u8_ss_m64n8k32(int32_t (&d)[4],
                           uint64_t a_desc, uint64_t b_desc,
                           bool scale_d = true) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %6, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k32.s32.u8.u8 "
        "{%0, %1, %2, %3}, %4, %5, p;\n}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "l"(a_desc), "l"(b_desc), "r"((uint32_t)scale_d));
}

// m64n8k32 s32.s8.u8 SS -- A signed, B unsigned.
__device__ __forceinline__
void wgmma_s8u8_ss_m64n8k32(int32_t (&d)[4],
                             uint64_t a_desc, uint64_t b_desc,
                             bool scale_d = true) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %6, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.u8 "
        "{%0, %1, %2, %3}, %4, %5, p;\n}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "l"(a_desc), "l"(b_desc), "r"((uint32_t)scale_d));
}

// m64n8k32 s32.u8.s8 SS -- A unsigned, B signed.
__device__ __forceinline__
void wgmma_u8s8_ss_m64n8k32(int32_t (&d)[4],
                             uint64_t a_desc, uint64_t b_desc,
                             bool scale_d = true) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %6, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k32.s32.u8.s8 "
        "{%0, %1, %2, %3}, %4, %5, p;\n}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "l"(a_desc), "l"(b_desc), "r"((uint32_t)scale_d));
}

// ---------------------------------------------------------------------------
// .satfinite modifier: clamps overflow to int32 max/min instead of wrapping.
// Important for INT8 quantization where intermediate sums can overflow int32
// when K is large. Without .satfinite, overflow wraps; with it, the result
// saturates -- safer for downstream activation / requantization.
// ---------------------------------------------------------------------------

// m64n8k32 s32.s8.s8 SS .satfinite -- 4 accum regs
__device__ __forceinline__
void wgmma_i8_ss_m64n8k32_satfinite(int32_t (&d)[4],
                                     uint64_t a_desc, uint64_t b_desc,
                                     bool scale_d = true) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %6, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.s8.satfinite "
        "{%0, %1, %2, %3}, %4, %5, p;\n}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "l"(a_desc), "l"(b_desc), "r"((uint32_t)scale_d));
}

// m64n8k32 s32.u8.u8 SS .satfinite -- 4 accum regs
__device__ __forceinline__
void wgmma_u8_ss_m64n8k32_satfinite(int32_t (&d)[4],
                                     uint64_t a_desc, uint64_t b_desc,
                                     bool scale_d = true) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %6, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k32.s32.u8.u8.satfinite "
        "{%0, %1, %2, %3}, %4, %5, p;\n}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "l"(a_desc), "l"(b_desc), "r"((uint32_t)scale_d));
}

// m64n8k64 s32.s8.s8 SS sparse 2:4 -- 4 accum regs
// INT8 sparse: K_logical=64, K_active=32, expected D = 32 with all-ones.
template <int SP_SEL>
__device__ __forceinline__
void wgmma_i8_ss_sp_m64n8k64(int32_t (&d)[4],
                              uint64_t a_desc, uint64_t b_desc,
                              uint32_t e_meta,
                              bool scale_d = true) {
    static_assert(SP_SEL >= 0 && SP_SEL <= 3, "SP_SEL must be 0..3");
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %7, 0;\n"
        "wgmma.mma_async.sp.sync.aligned.m64n8k64.s32.s8.s8 "
        "{%0, %1, %2, %3}, %4, %5, %6, %8, p;\n}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "l"(a_desc), "l"(b_desc), "r"(e_meta),
          "r"((uint32_t)scale_d), "n"(SP_SEL));
}

#endif  // PL_AGENTIC_SM90A
