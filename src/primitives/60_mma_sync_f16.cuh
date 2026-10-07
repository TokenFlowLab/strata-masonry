#pragma once
#if defined(PL_AGENTIC_SM90A)
// 60_mma_sync_f16.cuh -- mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
//
// ARCH: sm_90a (also sm_75+)
//
// Pre-WGMMA per-warp tensor-core MMA: 32 threads collectively compute a
// fixed m16n8k16 FP16 -> FP32 multiply. Accumulator in registers.
// Synchronous: returns after the result is in registers (no commit/wait).
// Used for small specialized tiles where WGMMA's m64 minimum is wasteful.
//
// Constraints (PTX 9.7.16.5.14):
//   all 32 threads must execute in lockstep (.sync.aligned); predication undefined
//   fragment layouts are shape-dependent (ISA Tables 67-80)
//   pair with ldmatrix.x4 (A) + ldmatrix.x2.trans (B) for SMEM inputs
//
// Issuer: warp (32 threads).
// PTX:    9.7.16.5.14 (mma.sync, f16->f32 m16n8k16), 9.7.16.6 (mma.sp)
//
#include <cstdint>

// ---------------------------------------------------------------------------
// mma.sync: the pre-WGMMA warp-level tensor core MMA.
// 32 threads (one warp) collectively compute C = A*B + C.
// Shape: m16n8k16 for FP16 (each thread holds a register fragment).
// Still relevant for small MMA ops and ldmatrix-fed paths.
// ---------------------------------------------------------------------------

// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
// A: 4 u32 regs (8 FP16 each = 16x16 matrix distributed)
// B: 2 u32 regs (8 FP16 each = 16x8 matrix distributed)
// C/D: 4 f32 regs (16x8 accumulator distributed)
__device__ __forceinline__
void mma_sync_m16n8k16_f16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

// mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32
//
// Structured 2:4 sparsity variant of mma.sync. Logical K = 32 but A is
// stored compactly (16x16 = 8 FP16 per thread), and a sparsity metadata
// register E selects which 2 of every 4 K-positions are non-zero. The
// sparsity selector `F` is an immediate (0 or 1) chosen at compile time.
//
// Operand shapes (per thread):
//   A: 4 u32 regs (8 FP16) -- compact storage
//   B: 4 u32 regs (8 FP16 packed; 32x8 dense)
//   C/D: 4 f32 regs (16x8 accumulator)
//   E:   1 u32   (sparsity metadata: 8 groups x 4 bits, selecting 2 of 4)
//
template <int F>
__device__ __forceinline__
void mma_sp_sync_m16n8k32_f16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3,
    float c0, float c1, float c2, float c3,
    uint32_t e_meta) {
    static_assert(F == 0 || F == 1, "mma.sp F-selector must be 0 or 1");
    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, "
        "{%12, %13, %14, %15}, "
        "%16, %17;\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3),
          "r"(e_meta), "n"(F));
}

#endif  // PL_AGENTIC_SM90A
