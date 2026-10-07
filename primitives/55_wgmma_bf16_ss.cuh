#pragma once
#if defined(PL_AGENTIC_SM90A)
// 55_wgmma_bf16_ss.cuh -- wgmma.mma_async.sync.aligned.m64nNk16.f32.bf16.bf16 (SS form)
//
// ARCH: sm_90a
//
// BF16 variant: same shape and accumulator size as FP16 (K=16, N/2 f32
// regs), only the operand element type differs. Most modern training
// uses BF16 for its wider exponent range vs FP16.
//
// Constraints (PTX 9.7.17):
//   K must be 16 (single block per MMA)
//   wgmma.fence.sync.aligned must precede the first MMA in a group (#59)
//
// Issuer: warp-group (128 threads).
// PTX:    9.7.17.5 (wgmma.mma_async, bf16)
//
#include <cstdint>

// m64n8k16 f32.bf16.bf16 SS -- 4 accum regs
__device__ __forceinline__
void wgmma_bf16_ss_m64n8k16(float (&d)[4],
                             uint64_t a_desc, uint64_t b_desc,
                             bool scale_d = true) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %6, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3}, %4, %5, p, 1, 1, 0, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "l"(a_desc), "l"(b_desc), "r"((uint32_t)scale_d));
}

// m64n16k16 f32.bf16.bf16 SS -- 8 accum regs
__device__ __forceinline__
void wgmma_bf16_ss_m64n16k16(float (&d)[8],
                              uint64_t a_desc, uint64_t b_desc,
                              bool scale_d = true) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %10, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, p, 1, 1, 0, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "l"(a_desc), "l"(b_desc), "r"((uint32_t)scale_d));
}

// m64n8k32 f32.bf16.bf16 SS sparse 2:4 -- 4 accum regs
template <int SP_SEL>
__device__ __forceinline__
void wgmma_bf16_ss_sp_m64n8k32(float (&d)[4],
                                uint64_t a_desc, uint64_t b_desc,
                                uint32_t e_meta,
                                bool scale_d = true) {
    static_assert(SP_SEL >= 0 && SP_SEL <= 3, "SP_SEL must be 0..3");
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %7, 0;\n"
        "wgmma.mma_async.sp.sync.aligned.m64n8k32.f32.bf16.bf16 "
        "{%0, %1, %2, %3}, %4, %5, %6, %8, p, 1, 1, 0, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "l"(a_desc), "l"(b_desc), "r"(e_meta),
          "r"((uint32_t)scale_d), "n"(SP_SEL));
}

// m64n64k16 f32.bf16.bf16 SS -- 32 accum regs (production size)
__device__ __forceinline__
void wgmma_bf16_ss_m64n64k16(float (&d)[32],
                              uint64_t a_desc, uint64_t b_desc,
                              bool scale_d = true) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        " %8, %9, %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31}, "
        "%32, %33, p, 1, 1, 0, 1;\n"
        "}\n"
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),
          "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "l"(a_desc), "l"(b_desc), "r"((uint32_t)scale_d));
}

#endif  // PL_AGENTIC_SM90A
