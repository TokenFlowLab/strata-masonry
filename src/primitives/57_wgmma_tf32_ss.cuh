#pragma once
#if defined(PL_AGENTIC_SM90A)
// 57_wgmma_tf32_ss.cuh -- wgmma.mma_async.sync.aligned.m64nNk8.f32.tf32.tf32 (SS form)
//
// ARCH: sm_90a
//
// TF32 variant: K=8 (half the FP16 K=16). TF32 is a 19-bit format
// stored in 32-bit registers (E8M10). Used when FP32 input precision is
// needed but TensorCore throughput is desired.
//
// Constraints (PTX 9.7.17):
//   K must be 8 (single block per MMA)
//   no transpose immediates -- layouts fixed by the descriptor
//   wgmma.fence.sync.aligned must precede the first MMA in a group (#59)
//
// Issuer: warp-group (128 threads).
// PTX:    9.7.17.5 (wgmma.mma_async, tf32), 9.7.17.6 (sparse 1:2)
//
#include <cstdint>

// TF32 has K=8. Accum: N/2 f32 registers.
// No transpose arguments for TF32 (layout is fixed).

// m64n8k8 f32.tf32.tf32 SS -- 4 accum regs
__device__ __forceinline__
void wgmma_tf32_ss_m64n8k8(float (&d)[4],
                            uint64_t a_desc, uint64_t b_desc,
                            bool scale_d = true) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %6, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k8.f32.tf32.tf32 "
        "{%0, %1, %2, %3}, %4, %5, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "l"(a_desc), "l"(b_desc), "r"((uint32_t)scale_d));
}

// m64n16k8 f32.tf32.tf32 SS -- 8 accum regs
__device__ __forceinline__
void wgmma_tf32_ss_m64n16k8(float (&d)[8],
                             uint64_t a_desc, uint64_t b_desc,
                             bool scale_d = true) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %10, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k8.f32.tf32.tf32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "l"(a_desc), "l"(b_desc), "r"((uint32_t)scale_d));
}

// m64n8k16 f32.tf32.tf32 SS sparse 1:2 -- 4 accum regs
// TF32 sparse uses 1:2 (not 2:4): each chunk of 2 elements has 1 zero.
// Metadata: 4-bit indices; only 0b1110 (selects T0,T1) and 0b0100 (T2,T3)
// are legal per PTX 9.7.17.6.2. Use 0xeeeeeeee to select first-of-pair.
template <int SP_SEL>
__device__ __forceinline__
void wgmma_tf32_ss_sp_m64n8k16(float (&d)[4],
                                uint64_t a_desc, uint64_t b_desc,
                                uint32_t e_meta,
                                bool scale_d = true) {
    static_assert(SP_SEL >= 0 && SP_SEL <= 3, "SP_SEL must be 0..3");
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %7, 0;\n"
        "wgmma.mma_async.sp.sync.aligned.m64n8k16.f32.tf32.tf32 "
        "{%0, %1, %2, %3}, %4, %5, %6, %8, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "l"(a_desc), "l"(b_desc), "r"(e_meta),
          "r"((uint32_t)scale_d), "n"(SP_SEL));
}

// m64n64k8 f32.tf32.tf32 SS -- 32 accum regs (production size)
__device__ __forceinline__
void wgmma_tf32_ss_m64n64k8(float (&d)[32],
                             uint64_t a_desc, uint64_t b_desc,
                             bool scale_d = true) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k8.f32.tf32.tf32 "
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
