#pragma once
#if defined(PL_AGENTIC_SM90A)
// 53_wgmma_f16_ss.cuh -- wgmma.mma_async.sync.aligned.m64nNk16.f32.f16.f16 (SS form)
//
// ARCH: sm_90a
//
// Hopper warp-group MMA: 128 threads (warpgroup) collectively issue an
// asynchronous m64xNxk16 FP16 matrix multiply with both A and B sourced
// from SMEM via 64-bit descriptors. Accumulator lives in registers
// (N/2 f32 per thread). Async: returns immediately; results visible
// after wgmma.wait_group N (see #59).
//
// N variants supported here: m64n8, m64n16. Larger N follows the same
// pattern with more accumulator registers.
//
// Constraints (PTX 9.7.17):
//   all 128 threads in the warpgroup must execute in lockstep (.sync.aligned)
//   wgmma.fence.sync.aligned must precede the first MMA in a group (#59)
//   accumulator registers must be claimed via setmaxnreg.inc (#46)
//
// Issuer: warp-group (128 threads, 4 aligned warps).
// PTX:    9.7.17.5 (wgmma.mma_async, f16 SS), 9.7.17.6 (sparse)
//
#include <cstdint>

// ---------------------------------------------------------------------------
// wgmma.mma_async.sync.aligned.m64nNk16.f32.f16.f16
// SS form: both A and B from SMEM via 64-bit descriptors.
// Accumulator: N/2 f32 registers per thread (PTX ISA 9.4, Section 9.7.17).
//   m64n8:  4 f32   m64n16: 8 f32   m64n64: 32 f32   m64n128: 64 f32
// ---------------------------------------------------------------------------

// m64n8k16 f32.f16.f16 SS -- 4 accum regs
__device__ __forceinline__
void wgmma_f16_ss_m64n8k16(
    float (&d)[4],
    uint64_t a_desc, uint64_t b_desc,
    bool scale_d = true) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %6, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 "
        "{%0, %1, %2, %3}, "
        "%4, %5, p, 1, 1, 0, 1;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "l"(a_desc), "l"(b_desc), "r"((uint32_t)scale_d));
}

// m64n16k16 f32.f16.f16 SS -- 8 accum regs
__device__ __forceinline__
void wgmma_f16_ss_m64n16k16(
    float (&d)[8],
    uint64_t a_desc, uint64_t b_desc,
    bool scale_d = true) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %10, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, "
        "%8, %9, p, 1, 1, 0, 1;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "l"(a_desc), "l"(b_desc), "r"((uint32_t)scale_d));
}

// m64n64k16 f32.f16.f16 SS -- 32 accum regs
__device__ __forceinline__
void wgmma_f16_ss_m64n64k16(
    float (&d)[32],
    uint64_t a_desc, uint64_t b_desc,
    bool scale_d = true) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
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

// m64n128k16 f32.f16.f16 SS -- 64 accum regs (production tile size)
__device__ __forceinline__
void wgmma_f16_ss_m64n128k16(
    float (&d)[64],
    uint64_t a_desc, uint64_t b_desc,
    bool scale_d = true) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        " %8, %9, %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31, "
        " %32, %33, %34, %35, %36, %37, %38, %39, "
        " %40, %41, %42, %43, %44, %45, %46, %47, "
        " %48, %49, %50, %51, %52, %53, %54, %55, "
        " %56, %57, %58, %59, %60, %61, %62, %63}, "
        "%64, %65, p, 1, 1, 0, 1;\n"
        "}\n"
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),
          "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
          "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]),
          "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
          "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]),
          "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
          "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]),
          "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
          "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]),
          "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "l"(a_desc), "l"(b_desc), "r"((uint32_t)scale_d));
}

// m64n256k16 f32.f16.f16 SS -- 128 accum regs
// Note: this uses ~128 register slots per thread for the accumulator alone;
// only useful when register-budget is loosened via setmaxnreg.inc and
// register pressure is not the bottleneck. Most FMHA / DeepSeek kernels
// stop at n=128 for FP16; keep n=256 as a corner-case tile size.
__device__ __forceinline__
void wgmma_f16_ss_m64n256k16(
    float (&d)[128],
    uint64_t a_desc, uint64_t b_desc,
    bool scale_d = true) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %130, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        " %8, %9, %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31, "
        " %32, %33, %34, %35, %36, %37, %38, %39, "
        " %40, %41, %42, %43, %44, %45, %46, %47, "
        " %48, %49, %50, %51, %52, %53, %54, %55, "
        " %56, %57, %58, %59, %60, %61, %62, %63, "
        " %64, %65, %66, %67, %68, %69, %70, %71, "
        " %72, %73, %74, %75, %76, %77, %78, %79, "
        " %80, %81, %82, %83, %84, %85, %86, %87, "
        " %88, %89, %90, %91, %92, %93, %94, %95, "
        " %96, %97, %98, %99, %100, %101, %102, %103, "
        " %104, %105, %106, %107, %108, %109, %110, %111, "
        " %112, %113, %114, %115, %116, %117, %118, %119, "
        " %120, %121, %122, %123, %124, %125, %126, %127}, "
        "%128, %129, p, 1, 1, 0, 1;\n"
        "}\n"
        : "+f"(d[0]),   "+f"(d[1]),   "+f"(d[2]),   "+f"(d[3]),
          "+f"(d[4]),   "+f"(d[5]),   "+f"(d[6]),   "+f"(d[7]),
          "+f"(d[8]),   "+f"(d[9]),   "+f"(d[10]),  "+f"(d[11]),
          "+f"(d[12]),  "+f"(d[13]),  "+f"(d[14]),  "+f"(d[15]),
          "+f"(d[16]),  "+f"(d[17]),  "+f"(d[18]),  "+f"(d[19]),
          "+f"(d[20]),  "+f"(d[21]),  "+f"(d[22]),  "+f"(d[23]),
          "+f"(d[24]),  "+f"(d[25]),  "+f"(d[26]),  "+f"(d[27]),
          "+f"(d[28]),  "+f"(d[29]),  "+f"(d[30]),  "+f"(d[31]),
          "+f"(d[32]),  "+f"(d[33]),  "+f"(d[34]),  "+f"(d[35]),
          "+f"(d[36]),  "+f"(d[37]),  "+f"(d[38]),  "+f"(d[39]),
          "+f"(d[40]),  "+f"(d[41]),  "+f"(d[42]),  "+f"(d[43]),
          "+f"(d[44]),  "+f"(d[45]),  "+f"(d[46]),  "+f"(d[47]),
          "+f"(d[48]),  "+f"(d[49]),  "+f"(d[50]),  "+f"(d[51]),
          "+f"(d[52]),  "+f"(d[53]),  "+f"(d[54]),  "+f"(d[55]),
          "+f"(d[56]),  "+f"(d[57]),  "+f"(d[58]),  "+f"(d[59]),
          "+f"(d[60]),  "+f"(d[61]),  "+f"(d[62]),  "+f"(d[63]),
          "+f"(d[64]),  "+f"(d[65]),  "+f"(d[66]),  "+f"(d[67]),
          "+f"(d[68]),  "+f"(d[69]),  "+f"(d[70]),  "+f"(d[71]),
          "+f"(d[72]),  "+f"(d[73]),  "+f"(d[74]),  "+f"(d[75]),
          "+f"(d[76]),  "+f"(d[77]),  "+f"(d[78]),  "+f"(d[79]),
          "+f"(d[80]),  "+f"(d[81]),  "+f"(d[82]),  "+f"(d[83]),
          "+f"(d[84]),  "+f"(d[85]),  "+f"(d[86]),  "+f"(d[87]),
          "+f"(d[88]),  "+f"(d[89]),  "+f"(d[90]),  "+f"(d[91]),
          "+f"(d[92]),  "+f"(d[93]),  "+f"(d[94]),  "+f"(d[95]),
          "+f"(d[96]),  "+f"(d[97]),  "+f"(d[98]),  "+f"(d[99]),
          "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]),
          "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]),
          "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]),
          "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]),
          "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]),
          "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]),
          "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
        : "l"(a_desc), "l"(b_desc), "r"((uint32_t)scale_d));
}

// ---------------------------------------------------------------------------
// Sparse 2:4 variants (wgmma.mma_async.sp).
// Logical K = 32 (A stored compactly: 16x16 -> 8 f16 packed per thread).
// SP_SEL is an immediate (0..3).  E_meta is a per-thread u32; for the
// canonical "select first 2 of every 4" pattern, set E_meta = 0x44444444.
// PTX ISA 9.7.17.7 -- sparse wgmma.
// ---------------------------------------------------------------------------

// m64n8k32 f32.f16.f16 SS sparse -- 4 accum regs
template <int SP_SEL>
__device__ __forceinline__
void wgmma_f16_ss_sp_m64n8k32(
    float (&d)[4],
    uint64_t a_desc, uint64_t b_desc,
    uint32_t e_meta,
    bool scale_d = true) {
    static_assert(SP_SEL >= 0 && SP_SEL <= 3, "SP_SEL must be 0..3");
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %7, 0;\n"
        "wgmma.mma_async.sp.sync.aligned.m64n8k32.f32.f16.f16 "
        "{%0, %1, %2, %3}, "
        "%4, %5, %6, %8, p, 1, 1, 0, 1;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "l"(a_desc), "l"(b_desc), "r"(e_meta),
          "r"((uint32_t)scale_d), "n"(SP_SEL));
}

// m64n64k32 f32.f16.f16 SS sparse -- 32 accum regs (production size)
template <int SP_SEL>
__device__ __forceinline__
void wgmma_f16_ss_sp_m64n64k32(
    float (&d)[32],
    uint64_t a_desc, uint64_t b_desc,
    uint32_t e_meta,
    bool scale_d = true) {
    static_assert(SP_SEL >= 0 && SP_SEL <= 3, "SP_SEL must be 0..3");
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %35, 0;\n"
        "wgmma.mma_async.sp.sync.aligned.m64n64k32.f32.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        " %8, %9, %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31}, "
        "%32, %33, %34, %36, p, 1, 1, 0, 1;\n"
        "}\n"
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),
          "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "l"(a_desc), "l"(b_desc), "r"(e_meta),
          "r"((uint32_t)scale_d), "n"(SP_SEL));
}

#endif  // PL_AGENTIC_SM90A
