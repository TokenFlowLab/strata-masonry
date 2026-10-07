#pragma once
#if defined(PL_AGENTIC_SM90A)
// 56_wgmma_fp8_ss.cuh -- wgmma.mma_async.sync.aligned.m64nNk32.f32.{e4m3,e5m2}.{e4m3,e5m2} (SS form)
//
// ARCH: sm_90a
//
// FP8 variant: K=32 (double the FP16 K=16). Same accumulator size as
// FP16 because output M*N is unchanged. Mixed types (E4M3 x E5M2) use
// the same shape with different .atype/.btype modifiers.
//
// Constraints (PTX 9.7.17):
//   K must be 32 (single block per MMA)
//   no transpose immediates -- A and B layouts are fixed by the descriptor
//   wgmma.fence.sync.aligned must precede the first MMA in a group (#59)
//
// Issuer: warp-group (128 threads).
// Source: knowledge/instructions/mma/wgmma.md
// PTX:    9.7.17.5 (wgmma.mma_async, e4m3 / e5m2)
//
#include <cstdint>

// FP8 has K=32. Accum: N/2 f32 registers.
// No transpose arguments for FP8 (layout is fixed).

// m64n8k32 f32.e4m3.e4m3 SS -- 4 accum regs
__device__ __forceinline__
void wgmma_e4m3_ss_m64n8k32(float (&d)[4],
                              uint64_t a_desc, uint64_t b_desc,
                              bool scale_d = true) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %6, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 "
        "{%0, %1, %2, %3}, %4, %5, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "l"(a_desc), "l"(b_desc), "r"((uint32_t)scale_d));
}

// m64n16k32 f32.e4m3.e4m3 SS -- 8 accum regs
__device__ __forceinline__
void wgmma_e4m3_ss_m64n16k32(float (&d)[8],
                               uint64_t a_desc, uint64_t b_desc,
                               bool scale_d = true) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %10, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "l"(a_desc), "l"(b_desc), "r"((uint32_t)scale_d));
}

// m64n16k32 f32.e5m2.e4m3 SS (mixed) -- 8 accum regs
__device__ __forceinline__
void wgmma_e5m2_e4m3_ss_m64n16k32(float (&d)[8],
                                    uint64_t a_desc, uint64_t b_desc,
                                    bool scale_d = true) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %10, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k32.f32.e5m2.e4m3 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "l"(a_desc), "l"(b_desc), "r"((uint32_t)scale_d));
}

// m64n8k32 f32.e5m2.e5m2 SS -- 4 accum regs (training backward path)
__device__ __forceinline__
void wgmma_e5m2_ss_m64n8k32(float (&d)[4],
                              uint64_t a_desc, uint64_t b_desc,
                              bool scale_d = true) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %6, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e5m2.e5m2 "
        "{%0, %1, %2, %3}, %4, %5, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "l"(a_desc), "l"(b_desc), "r"((uint32_t)scale_d));
}

// m64n16k32 f32.e5m2.e5m2 SS -- 8 accum regs
__device__ __forceinline__
void wgmma_e5m2_ss_m64n16k32(float (&d)[8],
                               uint64_t a_desc, uint64_t b_desc,
                               bool scale_d = true) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %10, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k32.f32.e5m2.e5m2 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "l"(a_desc), "l"(b_desc), "r"((uint32_t)scale_d));
}

// m64n64k32 f32.e4m3.e4m3 SS -- 32 accum regs (production size)
__device__ __forceinline__
void wgmma_e4m3_ss_m64n64k32(float (&d)[32],
                               uint64_t a_desc, uint64_t b_desc,
                               bool scale_d = true) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        " %8, %9, %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31}, "
        "%32, %33, p, 1, 1;\n"
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

// m64n8k64 f32.e4m3.e4m3 SS sparse 2:4 -- 4 accum regs
// FP8 sparse: K_logical=64, A stored compact 64x32 (2:4).
// E_meta=0x44444444 selects positions {0,1} of each 4-group -> 32 active.
template <int SP_SEL>
__device__ __forceinline__
void wgmma_e4m3_ss_sp_m64n8k64(float (&d)[4],
                                uint64_t a_desc, uint64_t b_desc,
                                uint32_t e_meta,
                                bool scale_d = true) {
    static_assert(SP_SEL >= 0 && SP_SEL <= 3, "SP_SEL must be 0..3");
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %7, 0;\n"
        "wgmma.mma_async.sp.sync.aligned.m64n8k64.f32.e4m3.e4m3 "
        "{%0, %1, %2, %3}, %4, %5, %6, %8, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "l"(a_desc), "l"(b_desc), "r"(e_meta),
          "r"((uint32_t)scale_d), "n"(SP_SEL));
}

// m64n64k32 f32.e5m2.e5m2 SS -- 32 accum regs
__device__ __forceinline__
void wgmma_e5m2_ss_m64n64k32(float (&d)[32],
                               uint64_t a_desc, uint64_t b_desc,
                               bool scale_d = true) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k32.f32.e5m2.e5m2 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        " %8, %9, %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31}, "
        "%32, %33, p, 1, 1;\n"
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
